# -*- coding: utf-8 -*-
"""
مقتطف الحل النموذجي لتحدي الفصل العاشر داخل دالة run() في ReWOODAGHarness:

مقتطف من الفصل 10 — «أنماط التفكير والتخطيط المتقدم».
النوع: lab_solution. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch10_rewoo_reflection.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

# مقتطف الحل النموذجي لتحدي الفصل العاشر داخل دالة run() في ReWOODAGHarness:
while sorter.is_active():
    ready_step_ids = list(sorter.get_ready())
    if not ready_step_ids:
        break

    wave_log.append(ready_step_ids)
    tasks: list[asyncio.Task[tuple[str, str]]] = []
    async with asyncio.TaskGroup() as tg:
        for sid in ready_step_ids:
            tasks.append(tg.create_task(self._execute_step(step_lookup[sid], evidence)))

    failed_step_info: tuple[str, str] | None = None
    for task in tasks:
        sid, result_text = task.result()
        if result_text.startswith("ERROR"):
            failed_step_info = (sid, result_text)
        else:
            evidence[sid] = result_text
            sorter.done(sid)

    # إذا فشلت خطوة في هذه الموجة، نقطع تنفيذ الفروع المعتمدة عليها ونستدعي المخطط التكيفي!
    if failed_step_info is not None and self.replanner_fn is not None:
        failed_id, err_msg = failed_step_info
        llm_calls += 1
        recovery_plan = await self.replanner_fn(
            user_goal, evidence, f"Step {failed_id} failed with: {err_msg}"
        )
        # تنفيذ خطة الإنقاذ البديلة ودمج أدلتها الناجحة
        for rec_step in recovery_plan.steps:
            r_id, r_out = await self._execute_step(rec_step, evidence)
            evidence[r_id] = r_out
        break  # الخروج من الخطة القديمة المتضررة والانتقال إلى الـ Solver بالأدلة المحدثة
