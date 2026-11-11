# -*- coding: utf-8 -*-
"""
مقتطف الحل النموذجي لتحدي الفصل السادس داخل حلقة while في StateGraphHarness.run():

مقتطف من الفصل 6 — «حلقة التنفيذ وآلات الحالة الموجهة».
النوع: lab_solution. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch06_execution_loop_state_graph.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

# مقتطف الحل النموذجي لتحدي الفصل السادس داخل حلقة while في StateGraphHarness.run():
steps += 1
visit_counts[current_node] += 1
node_t0 = time.perf_counter()

# 1. أخذ نسخة احتياطية عميقة ومعزولة (Transactional Snapshot) قبل دخول العقدة
state_snapshot: StateT = state.model_copy(deep=True)

try:
    # 2. تمرير النسخة الاحتياطية للعقدة؛ فإذا شوهتها العقدة ثم انهارت، تبقى state الأصلية سليمة!
    updated_state = await spec.handler(state_snapshot)
    state = updated_state  # اعتماد التعديلات (Commit) بعد نجاح العقدة بالكامل
    next_node = await self._resolve_next_node(current_node, state)
    note_msg = f"Visit #{visit_counts[current_node]}/{spec.max_visits} [COMMITTED]"

except Exception as exc:
    # 3. التراجع التلقائي (Auto-Rollback): تجاهل state_snapshot المشوهة والاحتفاظ بـ state السليمة
    if spec.quota_fallback_node:
        next_node = spec.quota_fallback_node
        note_msg = (
            f"🚨 [ROLLED BACK] Exception ({type(exc).__name__}: {exc}). "
            f"State restored & rerouted to fallback '{next_node}'."
        )
    else:
        raise  # إذا لم يكن للعقدة مسار طوارئ، نرفع الاستثناء لإيقاف الرسم البياني

node_ms = round((time.perf_counter() - node_t0) * 1000, 2)
audit_trail.append(
    TransitionAuditEntry(
        step_index=steps,
        from_node=current_node,
        to_node=next_node,
        duration_ms=node_ms,
        state_hash_after=self._compute_state_hash(state),
        note=note_msg,
    )
)
current_node = next_node
