# -*- coding: utf-8 -*-
"""
مقتطف الحل النموذجي لتحدي الفصل الثالث داخل StructuredGenerationHarness:

مقتطف من الفصل 3 — «العقل والموجهات ككود برمجي والمخرجات المهيكلة».
النوع: lab_solution. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch03_prompts_as_code.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

# مقتطف الحل النموذجي لتحدي الفصل الثالث داخل StructuredGenerationHarness:
@staticmethod
def _format_compact_feedback(
    error: ValidationError,
    failed_fields_seen: set[str],
    field_hints: dict[str, str] | None = None,
) -> str:
    error_lines: list[str] = []
    hints = field_hints or {}

    for err in error.errors(include_url=False):
        loc_tuple = err.get("loc", ("root",))
        field_name = str(loc_tuple[0]) if loc_tuple else "root"
        loc_path = " -> ".join(str(part) for part in loc_tuple)
        msg = err.get("msg", "Invalid value")
        input_val = err.get("input", "N/A")

        line = f"  - Field `{loc_path}` (got: {input_val!r}): {msg}"

        # التحقق مما إذا كان هذا الحقل قد فشل في محاولة سابقة!
        if field_name in failed_fields_seen and field_name in hints:
            line += f"\n    💡 [ESCALATION HINT for `{field_name}`]: {hints[field_name]}"

        # تسجيل الحقل في قائمة الحقول التي فشلت
        failed_fields_seen.add(field_name)
        error_lines.append(line)

    joined_errors = "\n".join(error_lines)
    return (
        "<validation_error_feedback>\n"
        "  Your previous JSON response failed strict validation:\n"
        f"{joined_errors}\n"
        "  Apply the hints above and return ONLY the corrected JSON object.\n"
        "</validation_error_feedback>"
    )
