#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""==============================================================================
الفصل 14: المشروع التخرجي الأول: وكيل هندسة برمجيات مستقل
==============================================================================

البرنامج الكامل للفصل 14 من كتاب «هندسة وكلاء الذكاء الاصطناعي».
المشروع التخرجي الأول: وكيل هندسة برمجيات مستقل — Capstone I — Autonomous Software Engineering Agent.

التشغيل:
    python ch14_swe_agent.py

المتطلبات: Python 3.12+ (الكتاب يستهدف 3.14+) و pydantic>=2.9
"""
# ملاحظة توافق: تُقيَّم الأنواع كسولاً في Python 3.14 (PEP 649)؛
# هذا السطر يمنح السلوك نفسه على Python 3.12/3.13.
from __future__ import annotations

import ast
import asyncio
import difflib
import hashlib
import os
import sys
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from pydantic import BaseModel, ConfigDict, Field


# =====================================================================
# 1. نماذج البيانات والرقعة الجراحية (SWE Models & Surgical Patch Schema)
# =====================================================================

class SWEWorkflowStage(StrEnum):
    LOCALIZE_AND_REPRODUCE = "localize_and_reproduce"
    SYNTHESIZE_PATCH = "synthesize_patch"
    VERIFY_IN_SANDBOX = "verify_in_sandbox"
    FINALIZE_REPORT = "finalize_report"
    ROLLBACK_AND_ABORT = "rollback_and_abort"


class SearchReplacePatch(BaseModel):
    """
    مخطط الرقعة الجراحية المهيكل: يفرض التفكير التشخيصي قبل كتابة الكود،
    ويستبدل مقطعاً فريداً ومحدداً بدلاً من إعادة كتابة الملف بالكامل.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    root_cause_diagnosis: str = Field(
        min_length=15,
        description="تحليل السبب الجذري للخلل بناءً على مخرجات تتبع الخطأ (Traceback)",
    )
    target_file: str = Field(
        description="اسم الملف المصدري المراد تعديله (يُمنع تعديل ملفات الاختبار)"
    )
    exact_search_block: str = Field(
        min_length=5,
        description="المقطع الأصلي المراد استبداله كما يظهر حرفياً في الملف المستهدف",
    )
    replacement_block: str = Field(
        min_length=5,
        description="المقطع البرمجي الجديد بعد الإصلاح",
    )


class SandboxTestResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool
    exit_code: int
    stdout: str
    stderr: str
    summary_output: str


class SWERunFinalReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    resolved: bool
    final_stage: SWEWorkflowStage
    attempts_used: int
    unified_diff: str
    audit_trail: list[str]


# =====================================================================
# 2. البوابات الأمنية: حارس AST وحارس نزاهة الاختبارات (Security Gates)
# =====================================================================

class PatchValidationError(ValueError):
    """يُرفع عندما تفشل الرقعة البرمجية في اجتياز فحوصات التفرد أو AST أو الأمان."""


class ASTAndSecurityGate:
    """
    يفحص الكود المعدّل في الذاكرة قبل كتابته على القرص:
    1. يتأكد من خلوه من الأخطاء النحوية (SyntaxError).
    2. يمنع استيراد المكتبات الخطرة أو دوال التنفيذ العشوائي.
    """

    _FORBIDDEN_IMPORTS: frozenset[str] = frozenset({
        "subprocess", "socket", "shutil", "ctypes", "pty", "urllib"
    })
    _FORBIDDEN_CALLS: frozenset[str] = frozenset({"eval", "exec", "compile", "__import__"})

    @classmethod
    def verify_python_source(cls, source_code: str, filename: str) -> None:
        try:
            tree = ast.parse(source_code, filename=filename)
        except SyntaxError as exc:
            raise PatchValidationError(
                f"رفض الرقعة بسبب خطأ نحوي (SyntaxError) في السطر {exc.lineno}: {exc.msg}"
            ) from exc

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_pkg = alias.name.split(".")[0]
                    if root_pkg in cls._FORBIDDEN_IMPORTS:
                        raise PatchValidationError(
                            f"انتهاك أمني في AST: يُمنع استيراد المكتبة '{root_pkg}'."
                        )
            elif isinstance(node, ast.ImportFrom) and node.module:
                root_pkg = node.module.split(".")[0]
                if root_pkg in cls._FORBIDDEN_IMPORTS:
                    raise PatchValidationError(
                        f"انتهاك أمني في AST: يُمنع الاستيراد من '{root_pkg}'."
                    )
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in cls._FORBIDDEN_CALLS:
                    raise PatchValidationError(
                        f"انتهاك أمني في AST: يُمنع استدعاء الدالة الخطرة '{node.func.id}()'."
                    )


class WorkspaceSnapshotManager:
    """
    يدير اللقطة المرجعية (Transactional Snapshot) لملفات المشروع،
    ويحمي ملفات الاختبارات من التعديل (Zero Reward-Hacking Guard).
    """

    def __init__(self, workspace_dir: Path) -> None:
        self._workspace = workspace_dir.resolve()
        self._initial_files: dict[str, str] = {}
        self._test_file_hashes: dict[str, str] = {}
        self._capture_initial_state()

    @staticmethod
    def _is_test_file(rel_path: str) -> bool:
        name = Path(rel_path).name
        return name.startswith("test_") or name.endswith("_test.py")

    @staticmethod
    def _sha256(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _capture_initial_state(self) -> None:
        for file_path in sorted(self._workspace.glob("*.py")):
            rel = file_path.name
            content = file_path.read_text(encoding="utf-8")
            self._initial_files[rel] = content
            if self._is_test_file(rel):
                self._test_file_hashes[rel] = self._sha256(content)

    def verify_test_integrity(self) -> None:
        """التأكد من عدم العبث بملفات الاختبار قبل أو بعد تشغيل صندوق الرمل."""
        for rel, expected_hash in self._test_file_hashes.items():
            path = self._workspace / rel
            if not path.exists():
                raise PatchValidationError(
                    f"انتهاك نزاهة الاختبارات: تم حذف ملف الاختبار '{rel}'!"
                )
            actual_hash = self._sha256(path.read_text(encoding="utf-8"))
            if actual_hash != expected_hash:
                raise PatchValidationError(
                    f"انتهاك نزاهة الاختبارات (Reward Hacking): تم تعديل ملف الاختبار '{rel}'!"
                )

    def apply_surgical_patch(self, patch: SearchReplacePatch) -> None:
        """تطبيق رقعة البحث والاستبدال بعد اجتياز جميع البوابات الأمنية والنحوية."""
        target_rel = Path(patch.target_file).name
        if self._is_test_file(target_rel):
            raise PatchValidationError(
                f"مرفوض أمنياً: لا يُسمح للوكيل بتعديل ملفات الاختبار ('{target_rel}'). "
                "يجب إصلاح الكود المصدري نفسه!"
            )

        target_path = (self._workspace / target_rel).resolve()
        if target_path.parent != self._workspace or not target_path.exists():
            raise PatchValidationError(
                f"الملف المستهدف '{target_rel}' غير موجود في مساحة العمل."
            )

        current_content = target_path.read_text(encoding="utf-8")
        occurrences = current_content.count(patch.exact_search_block)

        if occurrences == 0:
            raise PatchValidationError(
                f"فشل مطابقة exact_search_block: المقطع المطلوب غير موجود حرفياً في '{target_rel}'."
            )
        if occurrences > 1:
            raise PatchValidationError(
                f"غموض في المطابقة: المقطع يظهر {occurrences} مرات في '{target_rel}'. أضف أسطر سياق إضافية."
            )

        patched_content = current_content.replace(
            patch.exact_search_block, patch.replacement_block, 1
        )

        # فحص AST والأمان في الذاكرة قبل الكتابة على القرص!
        ASTAndSecurityGate.verify_python_source(patched_content, filename=target_rel)

        target_path.write_text(patched_content, encoding="utf-8")

    def rollback_all(self) -> None:
        """إرجاع مساحة العمل بالكامل إلى حالتها الأصلية النظيفة عند الفشل."""
        for rel, original_content in self._initial_files.items():
            (self._workspace / rel).write_text(original_content, encoding="utf-8")

    def compute_unified_diff(self) -> str:
        """توليد تقرير الفروقات المعياري (Unified Git Diff) لجميع الملفات المعدلة."""
        diffs: list[str] = []
        for rel, original_content in self._initial_files.items():
            current_content = (self._workspace / rel).read_text(encoding="utf-8")
            if current_content != original_content:
                diff_lines = difflib.unified_diff(
                    original_content.splitlines(keepends=True),
                    current_content.splitlines(keepends=True),
                    fromfile=f"a/{rel}",
                    tofile=f"b/{rel}",
                )
                diffs.append("".join(diff_lines))
        return "\n".join(diffs) if diffs else "No changes made."

    def read_source_context(self) -> dict[str, str]:
        return {
            p.name: p.read_text(encoding="utf-8")
            for p in sorted(self._workspace.glob("*.py"))
        }


# =====================================================================
# 3. مشغل الاختبارات المعزول (Hardened Subprocess Test Runner)
# =====================================================================

class IsolatedTestRunner:
    """
    يشغل حزمة اختبارات بايثون داخل عملية فرعية معزولة (python -I)
    مع تطهير متغيرات البيئة وتحديد سقف زمني صارم وحجم مخرجات مقيد.
    """

    def __init__(self, workspace_dir: Path, timeout_sec: float = 5.0) -> None:
        self._workspace = workspace_dir
        self._timeout_sec = timeout_sec

    async def run_unittest_suite(self) -> SandboxTestResult:
        clean_env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",  # وضع العزل الصارم في بايثون (Isolated Mode)
            "-m",
            "unittest",
            "discover",
            "-s",
            str(self._workspace),
            "-p",
            "test_*.py",
            "-v",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._workspace),
            env=clean_env,
        )

        try:
            async with asyncio.timeout(self._timeout_sec):
                stdout_bytes, stderr_bytes = await proc.communicate()
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return SandboxTestResult(
                passed=False,
                exit_code=-1,
                stdout="",
                stderr="TimeoutError: تجاوزت الاختبارات السقف الزمني (حلقة لا نهائية محتملة).",
                summary_output="TIMEOUT_EXCEEDED",
            )

        stdout_str = stdout_bytes.decode("utf-8", errors="replace")[:4000]
        stderr_str = stderr_bytes.decode("utf-8", errors="replace")[:4000]
        exit_code = proc.returncode or 0
        passed = exit_code == 0

        combined = f"{stdout_str}\n{stderr_str}".strip()
        return SandboxTestResult(
            passed=passed,
            exit_code=exit_code,
            stdout=stdout_str,
            stderr=stderr_str,
            summary_output=combined,
        )


# =====================================================================
# 4. عقل المهندس البرمجي وحلقة الـ SWE Harness (SWE Planner & Harness)
# =====================================================================

class SWEReasoningEngineProtocol(Protocol):
    async def propose_surgical_patch(
        self,
        issue_description: str,
        workspace_files: dict[str, str],
        latest_test_failure: str,
        attempt_number: int,
    ) -> SearchReplacePatch: ...


class SimulatedSWEReasoningEngine:
    """
    محاكاة واقعية لنموذج هندسة برمجية يتفاعل مع مخرجات الاختبارات الفعلية:
    - في المحاولة رقم 1: يصلح معادلة الخصم (طرح الخصم بدلاً من جمعه)،
      لكنه ينسى تقريب النتيجة النهائية عبر round(..., 2)، فيفشل اختبار الكسور!
    - في المحاولة رقم 2: يقرأ Traceback الفشل المتبقي، ويضيف round(..., 2) فتنجح كل الاختبارات!
    """

    async def propose_surgical_patch(
        self,
        issue_description: str,
        workspace_files: dict[str, str],
        latest_test_failure: str,
        attempt_number: int,
    ) -> SearchReplacePatch:
        await asyncio.sleep(0.02)
        if attempt_number == 1:
            return SearchReplacePatch(
                root_cause_diagnosis=(
                    "الاختبارات تظهر أن السعر يزيد بعد الخصم لأن السطر يستخدم "
                    "(1.0 + discount_rate) بدلاً من (1.0 - discount_rate)."
                ),
                target_file="billing_engine.py",
                exact_search_block=(
                    "    discounted = subtotal * (1.0 + discount_rate)\n"
                    "    total_with_tax = discounted * (1.0 + tax_rate)\n"
                    "    return total_with_tax"
                ),
                replacement_block=(
                    "    discounted = subtotal * (1.0 - discount_rate)\n"
                    "    total_with_tax = discounted * (1.0 + tax_rate)\n"
                    "    return total_with_tax"
                ),
            )

        # المحاولة الثانية بعد قراءة خطأ التقريب العشري في مخرجات unittest
        return SearchReplacePatch(
            root_cause_diagnosis=(
                "تم حل مشكلة اتجاه الخصم، ولكن اختبار test_fractional_tax_rounding فشل "
                "بسبب عدم تقريب الناتج النهائي لمنزلتين عشريتين (106.64325 != 106.64)."
            ),
            target_file="billing_engine.py",
            exact_search_block=(
                "    discounted = subtotal * (1.0 - discount_rate)\n"
                "    total_with_tax = discounted * (1.0 + tax_rate)\n"
                "    return total_with_tax"
            ),
            replacement_block=(
                "    discounted = subtotal * (1.0 - discount_rate)\n"
                "    total_with_tax = discounted * (1.0 + tax_rate)\n"
                "    return round(total_with_tax, 2)"
            ),
        )


class AutonomousSWEAgentHarness:
    """
    إطار التحكم المتكامل لوكيل الهندسة البرمجية المستقل (SWE Agent Harness).
    يربط مراحل التوطين، التعديل الجراحي، فحص AST، والتحقق في صندوق الرمل مع التراجع التلقائي.
    """

    def __init__(
        self,
        workspace_dir: Path,
        reasoning_engine: SWEReasoningEngineProtocol,
        max_patch_attempts: int = 3,
    ) -> None:
        self._workspace = workspace_dir
        self._engine = reasoning_engine
        self._max_attempts = max_patch_attempts
        self._snapshot = WorkspaceSnapshotManager(workspace_dir)
        self._test_runner = IsolatedTestRunner(workspace_dir)
        self._audit_log: list[str] = []

    def _log(self, msg: str) -> None:
        self._audit_log.append(msg)

    async def resolve_issue(self, issue_description: str) -> SWERunFinalReport:
        # المرحلة 1: التوطين وإعادة إنتاج العلة (LOCALIZE_AND_REPRODUCE)
        self._log(
            f"[{SWEWorkflowStage.LOCALIZE_AND_REPRODUCE}] تشغيل حزمة الاختبارات المبدئية لتأكيد العلة..."
        )
        self._snapshot.verify_test_integrity()
        initial_test = await self._test_runner.run_unittest_suite()

        if initial_test.passed:
            self._log("الاختبارات المبدئية ناجحة بالفعل؛ لا توجد علة لإصلاحها.")
            return SWERunFinalReport(
                resolved=True,
                final_stage=SWEWorkflowStage.FINALIZE_REPORT,
                attempts_used=0,
                unified_diff="No changes made.",
                audit_trail=self._audit_log,
            )

        latest_failure_trace = initial_test.summary_output
        self._log(
            "تمت إعادة إنتاج العلة بنجاح في صندوق الرمل (Initial Test Suite Failed as Expected)."
        )

        # المرحلة 2 و 3: حلقة توليد الرقعة الجراحية والتحقق (SYNTHESIZE_PATCH <-> VERIFY_IN_SANDBOX)
        for attempt in range(1, self._max_attempts + 1):
            self._log(
                f"[{SWEWorkflowStage.SYNTHESIZE_PATCH}] المحاولة #{attempt}/{self._max_attempts}: طلب رقعة جراحية من النموذج..."
            )
            patch = await self._engine.propose_surgical_patch(
                issue_description=issue_description,
                workspace_files=self._snapshot.read_source_context(),
                latest_test_failure=latest_failure_trace,
                attempt_number=attempt,
            )
            self._log(f"تشخيص الوكيل (#{attempt}): {patch.root_cause_diagnosis}")

            try:
                self._snapshot.apply_surgical_patch(patch)
                self._log(
                    f"تم التحقق من الرقعة عبر AST وتطبيقها على '{patch.target_file}'."
                )
            except PatchValidationError as exc:
                self._log(f"⚠️ رُفضت الرقعة قبل التنفيذ: {exc}")
                latest_failure_trace = f"PATCH_VALIDATION_ERROR: {exc}"
                continue

            # التحقق من نزاهة الاختبارات ثم تشغيلها في صندوق الرمل المعزول
            self._log(
                f"[{SWEWorkflowStage.VERIFY_IN_SANDBOX}] تشغيل الاختبارات في عملية معزولة (python -I)..."
            )
            self._snapshot.verify_test_integrity()
            test_res = await self._test_runner.run_unittest_suite()

            if test_res.passed:
                self._log(
                    f"✅ نجحت جميع الاختبارات في المحاولة #{attempt}! جاري توليد تقرير Unified Diff."
                )
                return SWERunFinalReport(
                    resolved=True,
                    final_stage=SWEWorkflowStage.FINALIZE_REPORT,
                    attempts_used=attempt,
                    unified_diff=self._snapshot.compute_unified_diff(),
                    audit_trail=self._audit_log,
                )

            self._log(
                f"❌ لا تزال هناك اختبارات فاشلة بعد المحاولة #{attempt}. جاري تغذية الوكيل بـ Traceback الجديد..."
            )
            latest_failure_trace = test_res.summary_output

        # المرحلة 5: التراجع التلقائي عند استنفاد المحاولات (ROLLBACK_AND_ABORT)
        self._log(
            f"[{SWEWorkflowStage.ROLLBACK_AND_ABORT}] استُنفدت المحاولات ({self._max_attempts}). جاري استعادة اللقطة الأصلية النظيفة للمستودع!"
        )
        self._snapshot.rollback_all()
        return SWERunFinalReport(
            resolved=False,
            final_stage=SWEWorkflowStage.ROLLBACK_AND_ABORT,
            attempts_used=self._max_attempts,
            unified_diff="Rolled back to clean state.",
            audit_trail=self._audit_log,
        )


# =====================================================================
# 5. إعداد المستودع البرمجي المعطوب وتشغيل المشروع التخرجي الأول
# =====================================================================

BUGGY_BILLING_ENGINE_CODE = """\
def compute_enterprise_invoice(
    subtotal: float,
    discount_rate: float,
    tax_rate: float,
) -> float:
    \"\"\"
    Calculates the final invoice total after applying tier discount and tax.
    Must return the total rounded to 2 decimal places.
    \"\"\"
    if subtotal < 0:
        raise ValueError("Subtotal cannot be negative")
    # BUG 1: Adding discount_rate instead of subtracting it!
    # BUG 2: Missing round(..., 2) on the returned amount!
    discounted = subtotal * (1.0 + discount_rate)
    total_with_tax = discounted * (1.0 + tax_rate)
    return total_with_tax
"""

TEST_BILLING_ENGINE_CODE = """\
import unittest
from billing_engine import compute_enterprise_invoice


class TestEnterpriseBillingEngine(unittest.TestCase):
    def test_standard_discount_and_tax(self) -> None:
        # 200.0 with 10% discount = 180.0; plus 5% tax = 189.0
        self.assertEqual(compute_enterprise_invoice(200.0, 0.10, 0.05), 189.0)

    def test_fractional_tax_rounding(self) -> None:
        # 115.50 with 15% discount = 98.175; plus 8.625% tax = 106.64259375 -> 106.64
        self.assertEqual(compute_enterprise_invoice(115.50, 0.15, 0.08625), 106.64)


if __name__ == "__main__":
    unittest.main()
"""


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="swe_capstone_repo_") as tmp_dir:
        repo_path = Path(tmp_dir)
        (repo_path / "billing_engine.py").write_text(
            BUGGY_BILLING_ENGINE_CODE, encoding="utf-8"
        )
        (repo_path / "test_billing_engine.py").write_text(
            TEST_BILLING_ENGINE_CODE, encoding="utf-8"
        )

        harness = AutonomousSWEAgentHarness(
            workspace_dir=repo_path,
            reasoning_engine=SimulatedSWEReasoningEngine(),
            max_patch_attempts=3,
        )

        issue = (
            "ISSUE-402: دالة compute_enterprise_invoice في ملف billing_engine.py "
            "تزيد قيمة الفاتورة عند تطبيق الخصم ولا تقرب الكسور العشرية لمنزلتين."
        )

        print("=" * 80)
        print("🛠️ تشغيل المشروع التخرجي الأول: Autonomous SWE Agent Harness")
        print("=" * 80)

        report = await harness.resolve_issue(issue)

        print("\n📋 1. سجل التدقيق الهندسي لدورة حياة الوكيل (SWE Audit Trail):")
        for step_idx, entry in enumerate(report.audit_trail, start=1):
            print(f"   {step_idx}. {entry}")

        print("\n" + "=" * 80)
        print(
            f"🏁 2. النتيجة النهائية: Resolved={report.resolved} | "
            f"Stage={report.final_stage} | Attempts={report.attempts_used}"
        )
        print("=" * 80)
        print("📄 3. رقعة Git النهائية المولّدة (Generated Unified Diff for PR):")
        print("-" * 80)
        print(report.unified_diff)


if __name__ == "__main__":
    asyncio.run(main())
