# -*- coding: utf-8 -*-
"""
❌ ANTI-PATTERN: غرفة الدردشة الفوضوية! كل وكيل يقرأ ضوضاء وهلوسات الوكلاء الآخرين!

مقتطف من الفصل 11 — «تنسيق الأنظمة متعددة الوكلاء».
النوع: anti_pattern. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch11_multi_agent_orchestration.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

# ❌ ANTI-PATTERN: غرفة الدردشة الفوضوية! كل وكيل يقرأ ضوضاء وهلوسات الوكلاء الآخرين!
for agent in [planner, researcher, coder, reviewer]:
    reply = await agent.run(shared_messages)  # تضخم تربيعي للتوكنز وتشتت كامل للانتباه!
    shared_messages.append(reply)
