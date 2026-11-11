# -*- coding: utf-8 -*-
"""
import ast

مقتطف من الفصل 14 — «المشروع التخرجي الأول: وكيل هندسة برمجيات مستقل».
النوع: example. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch14_swe_agent.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

import ast
from pydantic import BaseModel, ConfigDict, Field


class CodeQualityAuditResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    passed_quality_gate: bool
    duplicate_function_names: list[str]
    suspicious_magic_numbers: list[float]
    rejection_reason: str


class ASTCodeQualityGuard:
    """
    بوابة تدقيق نحوية (AST Static Analyzer) تكشف الدوال المكررة
    وتمنع الوكيل البرمجي من الغش عبر تثبيت قيم الاختبارات (Hardcoded Test Literals).
    """

    @staticmethod
    def _find_duplicate_functions_in_body(body: list[ast.stmt]) -> list[str]:
        seen: set[str] = set()
        duplicates: list[str] = []
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if stmt.name in seen and stmt.name not in duplicates:
                    duplicates.append(stmt.name)
                seen.add(stmt.name)
                # فحص الدوال المتداخلة داخل جسم الدالة نفسها
                duplicates.extend(
                    ASTCodeQualityGuard._find_duplicate_functions_in_body(stmt.body)
                )
            elif isinstance(stmt, ast.ClassDef):
                # فحص تكرار الدوال داخل نطاق الفئة الواحدة
                duplicates.extend(
                    ASTCodeQualityGuard._find_duplicate_functions_in_body(stmt.body)
                )
        return duplicates

    def audit_patched_code(
        self,
        source_code: str,
        forbidden_test_literals: set[float],
    ) -> CodeQualityAuditResult:
        tree = ast.parse(source_code)

        # 1. فحص تكرار تعريف الدوال في نفس النطاق
        duplicate_funcs = self._find_duplicate_functions_in_body(tree.body)

        # 2. فحص تثبيت أرقام الاختبارات داخل جمل المقارنة الشرطية (ast.Compare)
        suspicious_numbers: list[float] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                for subnode in ast.walk(node):
                    if isinstance(subnode, ast.Constant) and isinstance(
                        subnode.value, (int, float)
                    ) and not isinstance(subnode.value, bool):
                        numeric_val = float(subnode.value)
                        if (
                            numeric_val in forbidden_test_literals
                            and numeric_val not in suspicious_numbers
                        ):
                            suspicious_numbers.append(numeric_val)

        reasons: list[str] = []
        if duplicate_funcs:
            reasons.append(
                f"تم رصد تعريف دوال مكررة الاسم في نفس النطاق: {duplicate_funcs}"
            )
        if suspicious_numbers:
            reasons.append(
                f"غش مكافأة مكتشف (Hardcoded Test Bypass): جمل مقارنة تستخدم أرقام الاختبار مباشرة: {suspicious_numbers}"
            )

        passed = (len(duplicate_funcs) == 0) and (len(suspicious_numbers) == 0)
        return CodeQualityAuditResult(
            passed_quality_gate=passed,
            duplicate_function_names=duplicate_funcs,
            suspicious_magic_numbers=suspicious_numbers,
            rejection_reason=" | ".join(reasons) if reasons else "OK",
        )
