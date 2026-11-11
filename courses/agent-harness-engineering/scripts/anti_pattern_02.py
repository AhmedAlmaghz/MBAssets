# -*- coding: utf-8 -*-
"""
❌ ANTI-PATTERN: كارثة أمنية محققة!

مقتطف من الفصل 4 — «هندسة الأدوات واستدعاء الدوال».
النوع: anti_pattern. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch04_tool_registry.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

# ❌ ANTI-PATTERN: كارثة أمنية محققة!
@app.tool()
async def delete_user_document(user_id: str, doc_id: str) -> str:
    # إذا خدع المهاجم النموذج اللغوي عبر Prompt Injection، سيمرر النموذج user_id الخاص بضحية أخرى!
    ...
