#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""==============================================================================
الفصل 8: بيئة التنفيذ المعزولة وأمن تشغيل الأكواد
==============================================================================

البرنامج الكامل للفصل 8 من كتاب «هندسة وكلاء الذكاء الاصطناعي».
بيئة التنفيذ المعزولة وأمن تشغيل الأكواد — Sandboxing, Secure Code Execution & Defense-in-Depth.

التشغيل:
    python ch08_sandbox_security.py

المتطلبات: Python 3.12+ (الكتاب يستهدف 3.14+) و pydantic>=2.9
"""
# ملاحظة توافق: تُقيَّم الأنواع كسولاً في Python 3.14 (PEP 649)؛
# هذا السطر يمنح السلوك نفسه على Python 3.12/3.13.
from __future__ import annotations

import ast
import asyncio
from dataclasses import dataclass, field
from enum import StrEnum, auto
from pathlib import Path
import re
import sys
import tempfile
import textwrap
import time
from typing import Protocol


# =====================================================================
# 1. إعدادات وسياسات الأمان لصندوق الرمل (Sandbox Security Policy)
# =====================================================================
@dataclass(slots=True, frozen=True, kw_only=True)
class SandboxPolicy:
    """القيود الأمنية الصارمة التي يفرضها الـ Harness على أي كود يكتبه الوكيل"""

    timeout_sec: float = 3.0
    max_memory_mb: int = 128
    max_processes: int = 16  # حماية ضد قنابل التشعب (Fork Bombs)
    max_output_chars: int = 4_000
    allow_network: bool = False
    allowed_imports: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "math",
                "statistics",
                "json",
                "datetime",
                "collections",
                "itertools",
                "functools",
                "re",
                "decimal",
                "fractions",
                "random",
                "csv",
            }
        )
    )


class SandboxStatus(StrEnum):
    SUCCESS = auto()             # تم التنفيذ بنجاح (Exit Code 0)
    BLOCKED_BY_AST = auto()      # تم الحظر في الطبقة 1 قبل التنفيذ لوجود انتهاك أمني
    SYNTAX_ERROR = auto()        # خطأ نحوي في الكود الذي كتبه الوكيل
    RUNTIME_ERROR = auto()       # خطأ أثناء التشغيل (مثل ZeroDivisionError أو MemoryError)
    TIMEOUT_KILLED = auto()      # تم قتل العملية لتجاوزها المهلة الزمنية


@dataclass(slots=True, frozen=True)
class SandboxExecutionResult:
    """النتيجة المهيكلة والآمنة التي تعود إلى الوكيل ليقرأها أو يصحح كوده"""

    status: SandboxStatus
    stdout: str
    stderr: str
    exit_code: int | None
    execution_ms: float
    security_violations: tuple[str, ...] = ()

    def to_agent_observation(self) -> str:
        """تحويل النتيجة إلى رسالة واضحة ومؤمنة يفهمها النموذج اللغوي"""
        match self.status:
            case SandboxStatus.BLOCKED_BY_AST:
                reasons = "; ".join(self.security_violations)
                return f"🚨 [SECURITY BLOCK - AST GATEKEEPER]: Execution refused. Violations: {reasons}"
            case SandboxStatus.SYNTAX_ERROR:
                return f"❌ [SYNTAX ERROR]: Fix your Python syntax:\n{self.stderr}"
            case SandboxStatus.TIMEOUT_KILLED:
                return (
                    f"⏱️ [TIMEOUT KILLED]: Execution exceeded time limit and was terminated. "
                    f"Check for infinite loops!\nPartial stdout: {self.stdout}"
                )
            case SandboxStatus.RUNTIME_ERROR:
                return (
                    f"⚠️ [RUNTIME ERROR (Exit {self.exit_code})]:\n"
                    f"STDOUT:\n{self.stdout}\nSTDERR:\n{self.stderr}"
                )
            case SandboxStatus.SUCCESS:
                return f"✅ [EXECUTION SUCCESS ({self.execution_ms}ms)]:\n{self.stdout or '(No stdout produced)'}"


# =====================================================================
# 2. الطبقة الدفاعية الأولى: البوابة الساكنة لشجرة الكود (Static AST Gatekeeper)
# =====================================================================
class _ASTSecurityVisitor(ast.NodeVisitor):
    """فاحص عميق لشجرة الكود يمنع الاستيرادات غير المصرح بها وحيل الهروب من صندوق الرمل"""

    # الدوال المدمجة الخطيرة التي يُمنع استدعاؤها مباشرة
    _FORBIDDEN_CALLS: frozenset[str] = frozenset(
        {"eval", "exec", "compile", "__import__", "input", "breakpoint", " memoryview"}
    )

    # السمات الداخلية (Dunder Attributes) التي تُستخدم في اختراق صناديق بايثون!
    _FORBIDDEN_ATTRIBUTES: frozenset[str] = frozenset(
        {
            "__subclasses__",
            "__bases__",
            "__base__",
            "__mro__",
            "__globals__",
            "__builtins__",
            "__code__",
            "__closure__",
            "f_globals",
            "f_locals",
            "f_back",
        }
    )

    def __init__(self, allowed_imports: frozenset[str]) -> None:
        self.allowed_imports = allowed_imports
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root_pkg = alias.name.split(".")[0]
            if root_pkg not in self.allowed_imports:
                self.violations.append(
                    f"Line {node.lineno}: Importing '{root_pkg}' is forbidden by SandboxPolicy."
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            root_pkg = node.module.split(".")[0]
            if root_pkg not in self.allowed_imports:
                self.violations.append(
                    f"Line {node.lineno}: Importing from '{root_pkg}' is forbidden by SandboxPolicy."
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in self._FORBIDDEN_CALLS:
            self.violations.append(
                f"Line {node.lineno}: Calling built-in '{node.func.id}()' is strictly prohibited."
            )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in self._FORBIDDEN_ATTRIBUTES or (
            node.attr.startswith("__") and node.attr.endswith("__") and node.attr != "__init__"
        ):
            self.violations.append(
                f"Line {node.lineno}: Accessing internal dunder attribute '{node.attr}' is prohibited."
            )
        self.generic_visit(node)


class ASTSecurityGatekeeper:
    """الطبقة 1: تفحص الكود في أقل من 1 ميلي ثانية قبل تشغيل أي عملية على النظام"""

    @staticmethod
    def inspect_code(
        source_code: str, policy: SandboxPolicy
    ) -> tuple[bool, list[str], str | None]:
        """يعيد: (هل_الكود_آمن، قائمة_الانتهاكات_الأمنية، رسالة_الخطأ_النحوي_إن_وجدت)"""
        try:
            tree = ast.parse(source_code)
        except SyntaxError as syn_err:
            err_msg = f"Line {syn_err.lineno}, Col {syn_err.offset}: {syn_err.msg}"
            return False, [], err_msg

        visitor = _ASTSecurityVisitor(policy.allowed_imports)
        visitor.visit(tree)
        return (len(visitor.violations) == 0), visitor.violations, None


# =====================================================================
# 3. الطبقة الدفاعية الثانية والثالثة: محركات العزل (Pluggable Sandbox Drivers)
# =====================================================================
class SandboxRuntimeDriver(Protocol):
    """عقد برمجي يسمح بتبديل محرك العزل (Local Subprocess / Docker / MicroVM) بشفافية"""

    async def run_isolated(
        self, code: str, policy: SandboxPolicy
    ) -> tuple[int | None, str, str, bool]:
        """يعيد: (exit_code, stdout, stderr, timed_out)"""
        ...


class HardenedSubprocessDriver:
    """محرك عزل محلي محصن يعمل على الفور:

    - يشغل بايثون بوضع العزل التام (-I: يتجاهل متغيرات PYTHONPATH ومجلدات المستخدم).
    - ينشئ مجلداً مؤقتاً معزولاً يُدمر فور انتهاء الكود.
    - يصفر متغيرات البيئة تماماً (Zero Host Secrets).
    - يفرض قيود الذاكرة والمعالج على أنظمة POSIX عبر وحدة resource.
    """

    @staticmethod
    def _build_rlimit_wrapper(user_script_path: str, policy: SandboxPolicy) -> str:
        """غلاف برمجي يطبق قيود الذاكرة والمعالج على مستوى نواة النظام قبل تشغيل كود الوكيل"""
        mem_bytes = policy.max_memory_mb * 1024 * 1024
        cpu_seconds = max(1, int(policy.timeout_sec) + 1)
        return textwrap.dedent(
            f"""\
            import sys
            try:
                import resource
                # تقييد الذاكرة العشوائية القصوى (Address Space) ووقت المعالج
                resource.setrlimit(resource.RLIMIT_AS, ({mem_bytes}, {mem_bytes}))
                resource.setrlimit(resource.RLIMIT_CPU, ({cpu_seconds}, {cpu_seconds}))
            except (ImportError, ValueError, OSError):
                pass  # في حال التشغيل على نظام لا يدعم RLIMIT_AS مثل Windows أو macOS بعض الإصدارات

            with open({user_script_path!r}, "rb") as f:
                code_obj = compile(f.read(), "agent_sandbox_script.py", "exec")
            exec(code_obj, {{"__name__": "__main__"}})
            """
        )

    async def run_isolated(
        self, code: str, policy: SandboxPolicy
    ) -> tuple[int | None, str, str, bool]:
        with tempfile.TemporaryDirectory(prefix="agent_sbx_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            user_file = tmp_path / "user_code.py"
            runner_file = tmp_path / "_runner.py"

            user_file.write_text(code, encoding="utf-8")
            runner_file.write_text(
                self._build_rlimit_wrapper(str(user_file), policy), encoding="utf-8"
            )

            # بيئة معقمة تماماً: لا مفاتيح API ولا متغيرات نظام المضيف!
            scrubbed_env = {
                "PATH": "/usr/bin:/bin",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONDONTWRITEBYTECODE": "1",
            }

            # الراية -I تفعل Isolated Mode في بايثون
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                str(runner_file),
                cwd=tmp_dir,
                env=scrubbed_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                async with asyncio.timeout(policy.timeout_sec):
                    stdout_b, stderr_b = await proc.communicate()
                return (
                    proc.returncode,
                    stdout_b.decode("utf-8", errors="replace"),
                    stderr_b.decode("utf-8", errors="replace"),
                    False,
                )
            except TimeoutError:
                # قتل فوري للعملية الخارجة عن السيطرة وتنظيفها
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass
                return None, "", "Process killed due to timeout.", True


class RootlessDockerDriver:
    """محرك الحاويات المعزولة للإنتاج (Production Docker Sandbox Driver).

    يوضح الأوامر القياسية العالمية لتشغيل كود الوكيل داخل حاوية معزولة الشبكة وعديمة الصلاحيات.
    """

    def __init__(self, image: str = "python:3.14-slim") -> None:
        self.image = image

    def build_docker_command(self, code: str, policy: SandboxPolicy) -> list[str]:
        """بناء أمر Docker المحصن بأعلى معايير الأمان العالمية (CIS Docker Benchmark)"""
        net_flag = "bridge" if policy.allow_network else "none"
        return [
            "docker",
            "run",
            "--rm",                                      # تدمير الحاوية فوراً بعد التنفيذ
            "-i",
            "--network", net_flag,                       # عزل الشبكة تماماً لمنع تسريب البيانات!
            "--read-only",                               # نظام ملفات الحاوية للقراءة فقط!
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=32m", # مساحة مؤقتة صغيرة في الذاكرة لا تسمح بتشغيل ملفات تنفيذية
            "--memory", f"{policy.max_memory_mb}m",      # سقف صارم للذاكرة العشوائية
            "--memory-swap", f"{policy.max_memory_mb}m", # منع استخدام Swap على القرص
            "--cpus", "0.5",                             # نصف نواة معالج كحد أقصى
            "--pids-limit", str(policy.max_processes),   # إحباط قنابل التشعب (Fork Bombs)
            "--cap-drop", "ALL",                         # سحب جميع صلاحيات نواة Linux (Capabilities)
            "--security-opt", "no-new-privileges",       # منع التصعيد الأمني داخل الحاوية
            "--user", "65534:65534",                     # التشغيل كمستخدم nobody عديم الصلاحيات
            self.image,
            "python3", "-I", "-c", code,
        ]

    async def run_isolated(
        self, code: str, policy: SandboxPolicy
    ) -> tuple[int | None, str, str, bool]:
        cmd = self.build_docker_command(code, policy)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            async with asyncio.timeout(policy.timeout_sec):
                out_b, err_b = await proc.communicate()
            return proc.returncode, out_b.decode("utf-8", "replace"), err_b.decode("utf-8", "replace"), False
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return None, "", "Container killed on timeout.", True


# =====================================================================
# 4. الطبقة الرابعة والمنسق الرئيسي: MultiLayerSandboxHarness
# =====================================================================
class MultiLayerSandboxHarness:
    """إطار التحكم الشامل لتشغيل أكواد الوكيل عبر الطبقات الدفاعية الأربع"""

    # أنماط الأسرار لطمسها تلقائياً إذا تسربت في المخرجات (Secret Redaction)
    _SECRET_REGEX = re.compile(
        r"(sk-[a-zA-Z0-9]{16,}|ghp_[a-zA-Z0-9]{20,}|AKIA[0-9A-Z]{16}|Bearer\s+[a-zA-Z0-9\-._~+/]+=*)"
    )

    def __init__(
        self,
        driver: SandboxRuntimeDriver,
        policy: SandboxPolicy | None = None,
    ) -> None:
        self.driver = driver
        self.policy = policy or SandboxPolicy()

    def _sanitize_output(self, text: str) -> str:
        """الطبقة 4: طمس أي مفاتيح سرية وتقليم الطول لحماية نافذة السياق"""
        redacted = self._SECRET_REGEX.sub("[🔒 REDACTED_SECRET]", text)
        if len(redacted) > self.policy.max_output_chars:
            return (
                redacted[: self.policy.max_output_chars]
                + "\n...[✂️ Output truncated by Sandbox Harness]..."
            )
        return redacted.strip()

    async def execute_code(self, raw_code: str) -> SandboxExecutionResult:
        """نقطة الدخول الموحدة لتشغيل الكود بأمان عبر الطبقات الأربع"""
        t0 = time.perf_counter()

        # الطبقة 1: الفحص الساكن لشجرة الكود (AST Gatekeeper) قبل لمس نظام التشغيل!
        is_safe, violations, syntax_err = ASTSecurityGatekeeper.inspect_code(
            raw_code, self.policy
        )

        if syntax_err is not None:
            return SandboxExecutionResult(
                status=SandboxStatus.SYNTAX_ERROR,
                stdout="",
                stderr=syntax_err,
                exit_code=None,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        if not is_safe:
            return SandboxExecutionResult(
                status=SandboxStatus.BLOCKED_BY_AST,
                stdout="",
                stderr="",
                exit_code=None,
                execution_ms=round((time.perf_counter() - t0) * 1000, 2),
                security_violations=tuple(violations),
            )

        # الطبقة 2 و3: التشغيل المعزول تحت حواجز الموارد والوقت
        exit_code, raw_out, raw_err, timed_out = await self.driver.run_isolated(
            raw_code, self.policy
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        # الطبقة 4: تطهير المخرجات وطمس الأسرار
        clean_out = self._sanitize_output(raw_out)
        clean_err = self._sanitize_output(raw_err)

        if timed_out:
            return SandboxExecutionResult(
                status=SandboxStatus.TIMEOUT_KILLED,
                stdout=clean_out,
                stderr=clean_err,
                exit_code=None,
                execution_ms=elapsed_ms,
            )

        if exit_code != 0:
            return SandboxExecutionResult(
                status=SandboxStatus.RUNTIME_ERROR,
                stdout=clean_out,
                stderr=clean_err,
                exit_code=exit_code,
                execution_ms=elapsed_ms,
            )

        return SandboxExecutionResult(
            status=SandboxStatus.SUCCESS,
            stdout=clean_out,
            stderr=clean_err,
            exit_code=0,
            execution_ms=elapsed_ms,
        )


# =====================================================================
# 5. مختبر اختراق دفاعي حي: اختبار 4 سيناريوهات واقعية ضد الـ Sandbox Harness!
# =====================================================================
async def main() -> None:
    policy = SandboxPolicy(timeout_sec=0.6, max_memory_mb=128)
    sandbox = MultiLayerSandboxHarness(driver=HardenedSubprocessDriver(), policy=policy)

    scenarios: list[tuple[str, str]] = [
        (
            "1️⃣ سيناريو شرعي: تحليل إحصائي ومالي",
            """
            import math, statistics, json
            revenues = [12050.5, 15400.0, 9800.25, 22100.75, 18300.0]
            report = {
                "mean": round(statistics.mean(revenues), 2),
                "stdev": round(statistics.stdev(revenues), 2),
                "growth_factor": round(math.sqrt(max(revenues) / min(revenues)), 3),
            }
            print(json.dumps(report, ensure_ascii=False))
            """,
        ),
        (
            "2️⃣ هجوم مباشر: محاولة سرقة متغيرات البيئة وتشغيل أوامر نظام",
            """
            import os, subprocess
            print(os.environ.get("OPENAI_API_KEY"))
            subprocess.run(["cat", "/etc/passwd"])
            """,
        ),
        (
            "3️⃣ هجوم متقدم: محاولة الهروب من بايثون عبر شجرة الكلاسات (__subclasses__)",
            """
            # حيلة شهيرة يستخدمها المخترقون لتجاوز حظر import!
            subclasses = ().__class__.__base__.__subclasses__()
            eval("print('Hacked!')")
            """,
        ),
        (
            "4️⃣ هجوم حجب خدمة (DoS): حلقة لانهائية تحاول تجميد الخادم",
            """
            counter = 0
            while True:
                counter += 1
            """,
        ),
    ]

    print("🛡️ بدء اختبار الاختراق الدفاعي لـ MultiLayerSandboxHarness:\n")
    for title, code_snippet in scenarios:
        clean_code = textwrap.dedent(code_snippet).strip()
        result = await sandbox.execute_code(clean_code)
        print(f"─── {title} ───")
        print(f"   {result.to_agent_observation()}\n")

    # عرض أمر Docker الإنتاجي الذي يولده RootlessDockerDriver للمقارنة
    docker_driver = RootlessDockerDriver()
    docker_cmd = " ".join(docker_driver.build_docker_command("print(42)", policy))
    print("🐳 أمر Docker الإنتاجي المحصن (Rootless / Air-Gapped):")
    print(f"   $ {docker_cmd}")


if __name__ == "__main__":
    asyncio.run(main())
