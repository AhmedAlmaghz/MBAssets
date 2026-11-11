# -*- coding: utf-8 -*-
"""
❌ نمط ضار: تجميد عامل FastAPI/Uvicorn في الذاكرة لساعات بانتظار موافقة المدير!

مقتطف من الفصل 15 — «المشروع التخرجي الثاني: منظومة مالية متعددة الوكلاء».
النوع: anti_pattern. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch15_financial_swarm.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

# ❌ نمط ضار: تجميد عامل FastAPI/Uvicorn في الذاكرة لساعات بانتظار موافقة المدير!
@app.post("/run-agent")
async def run_agent_endpoint():
    proposal = await agent.propose_trade()
    approved = await wait_for_slack_webhook_or_human()  # كارثة! سيقطع البروكسي الاتصال بـ 504 Timeout
