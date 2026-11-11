# -*- coding: utf-8 -*-
"""
مقتطف الحل النموذجي لتحدي الفصل الأول داخل دالة run() في AgentHarness:

مقتطف من الفصل 1 — «ما هو وكيل الذكاء الاصطناعي؟ وما هو إطار التحكم؟».
النوع: lab_solution. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch01_agent_harness_foundations.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

# مقتطف الحل النموذجي لتحدي الفصل الأول داخل دالة run() في AgentHarness:
last_action: ToolAction | None = None
repeat_count: int = 0

# ... داخل حلقة while وبالتحديد داخل فرع case LLMResponse(thought=thought, action=ToolAction() as action):
if action == last_action:
    repeat_count += 1
else:
    last_action = action
    repeat_count = 1

if repeat_count == 2:
    # المرة الثانية: لا ننفذ الأداة، بل نحقن تحذيراً صارماً لتصحيح المسار
    history.append(
        Message(
            role=Role.USER,
            content=f"System Harness Warning: لقد قمت باستدعاء الأداة '{action.name}' بنفس المدخلات للتو. يُمنع تكرار نفس العملية. غيّر المدخلات أو اتخذ مساراً آخر.",
        )
    )
    continue  # تخطي تنفيذ الأداة والعودة لبداية الحلقة للتفكير مجدداً

elif repeat_count >= 3:
    # المرة الثالثة: قطع الدائرة فوراً لحماية الميزانية والوقت
    metrics.elapsed_seconds = time.monotonic() - start_time
    return HarnessResult(
        stop_reason=StopReason.REPETITIVE_LOOP_DETECTED,
        final_output=f"تم إيقاف الوكيل قسرياً لدخوله في حلقة مفرغة على الأداة: {action.name}",
        metrics=metrics,
        context=context,
        history=history,
    )
