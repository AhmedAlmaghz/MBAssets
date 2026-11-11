# -*- coding: utf-8 -*-
"""
❌ ANTI-PATTERN: كود سيئ جداً وغير صالح للإنتاج أبداً

مقتطف من الفصل 1 — «ما هو وكيل الذكاء الاصطناعي؟ وما هو إطار التحكم؟».
النوع: anti_pattern. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch01_agent_harness_foundations.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

# ❌ ANTI-PATTERN: كود سيئ جداً وغير صالح للإنتاج أبداً
def run_bad_agent(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    while True:  # كارثة 1: حلقة لانهائية بدون سقف خطوات أو مهلة زمنية!
        response = openai.chat.completions.create(  # كارثة 2: ارتباط مباشر بمزود واحد ومتزامن (Blocking)
            model="gpt-4o", messages=messages
        )
        msg = response.choices[0].message
        if not msg.tool_calls:
            return msg.content
        for tool in msg.tool_calls:
            # كارثة 3: استخدام eval أو تنفيذ الأداة بدون تحقق أو معالجة استثناءات!
            result = eval(f"{tool.function.name}(**{tool.function.arguments})")
            messages.append({"role": "tool", "content": str(result)})
