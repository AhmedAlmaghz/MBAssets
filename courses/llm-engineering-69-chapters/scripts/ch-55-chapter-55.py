#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — بناء وكلاء ذكيين (AI Agents)
Large Language Model Engineering — Agents

Book slug : llm-engineering-69-chapters
Chapter   : ch-55-chapter-55 (lesson 55)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-55-chapter-55.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — بناء وكلاء ذكيين (AI Agents)
Large Language Model Engineering — Agents

Book slug : llm-engineering-69-chapters
Chapter   : ch-55-chapter-55 (lesson 55)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-55-chapter-55.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — بناء وكلاء ذكيين (AI Agents)
Large Language Model Engineering — Agents

Book slug : llm-engineering-69-chapters
Chapter   : ch-55-chapter-55 (lesson 55)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_55.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 55: الفصل 55: بناء وكلاء ذكيين (AI Agents)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_tools() -> None:
    tools = {'add': lambda a, b: a + b, 'multiply': lambda a, b: a * b}
    plan = [{'name': 'add', 'arguments': {'a': 2, 'b': 3}}, {'name': 'multiply', 'arguments': {'a': 4, 'b': 5}}]
    results = []
    for call in plan:
        if call['name'] not in tools or set(call['arguments']) != {'a', 'b'}:
            raise ValueError('tool call rejected by allow-list/schema.')
        results.append(tools[call['name']](**call['arguments']))
    print('bounded tool results:', results)
    print('external network/filesystem access: disabled')

if __name__ == "__main__":
    print("Chapter 55 lab: Run an allow-listed, bounded tool-calling agent")
    run_tools()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-10-applications-agents/chapter-55.md
# ────────────────────────────────────────────────────────────────────────
# requirements.txt
# openai>=1.40.0
# langchain>=0.3.0
# langchain-openai>=0.2.0
# langgraph>=0.2.0
# crewai>=0.70.0
# duckduckgo-search>=6.0.0

import json
import re
from openai import OpenAI
from typing import Callable

client = OpenAI(api_key="sk-...")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-10-applications-agents/chapter-55.md
# ────────────────────────────────────────────────────────────────────────
class SimpleReActAgent:
    """
    تطبيق ReAct يدوي بالكامل، بدون أي إطار خارجي،
    لفهم الآلية الداخلية الحقيقية قبل استخدام LangChain/CrewAI.
    """

    def __init__(self, llm_client: OpenAI, tools: dict[str, Callable], max_iterations: int = 5):
        self.client = llm_client
        self.tools = tools  # {"tool_name": callable_function}
        self.max_iterations = max_iterations

    def _build_system_prompt(self) -> str:
        tool_descriptions = "\n".join(
            f"- {name}: {func.__doc__}" for name, func in self.tools.items()
        )
        return f"""أنت وكيل ذكي يحل المهام عبر التفكير المتسلسل واستخدام الأدوات.

الأدوات المتاحة لك:
{tool_descriptions}

يجب أن تتبع الصيغة التالية بدقة في كل خطوة:
Thought: [تفكيرك في الخطوة التالية المطلوبة]
Action: [اسم الأداة المطلوب استخدامها، أو "Finish" إذا انتهيت]
Action Input: [المدخل المطلوب للأداة بصيغة JSON]

بعد كل Action، ستحصل على Observation تحتوي النتيجة الفعلية.
عندما تصل لإجابة نهائية كافية، استخدم Action: Finish مع الإجابة في Action Input.
"""

    def _parse_llm_output(self, text: str) -> dict:
        """تحليل مخرجات LLM لاستخراج Thought/Action/Action Input"""
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", text, re.DOTALL)
        action_match = re.search(r"Action:\s*(.+?)(?=Action Input:|$)", text, re.DOTALL)
        input_match = re.search(r"Action Input:\s*(.+?)$", text, re.DOTALL)

        return {
            "thought": thought_match.group(1).strip() if thought_match else "",
            "action": action_match.group(1).strip() if action_match else "",
            "action_input": input_match.group(1).strip() if input_match else "",
        }

    def run(self, task: str) -> str:
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": f"المهمة: {task}"}
        ]

        for iteration in range(self.max_iterations):
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.2,
            )
            llm_output = response.choices[0].message.content
            print(f"\n--- التكرار {iteration + 1} ---\n{llm_output}")

            parsed = self._parse_llm_output(llm_output)

            if parsed["action"].strip().lower() == "finish":
                return parsed["action_input"]

            # تنفيذ الأداة الفعلية
            tool_name = parsed["action"].strip()
            if tool_name not in self.tools:
                observation = f"خطأ: الأداة '{tool_name}' غير موجودة."
            else:
                try:
                    tool_args = json.loads(parsed["action_input"])
                    observation = str(self.tools[tool_name](**tool_args))
                except Exception as e:
                    observation = f"خطأ في تنفيذ الأداة: {str(e)}"

            print(f"🔍 Observation: {observation}")

            # إضافة السياق التراكمي للمحادثة
            messages.append({"role": "assistant", "content": llm_output})
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        return "⚠️ تجاوز الوكيل الحد الأقصى للتكرارات دون الوصول لإجابة نهائية."


# --- تعريف الأدوات ---
def search_flight_price(destination: str, date: str) -> str:
    """يبحث عن سعر رحلة طيران لوجهة معيّنة وتاريخ محدد"""
    # في الواقع: استدعاء API حقيقي لموقع حجز طيران
    mock_prices = {"دبي": 1850, "القاهرة": 950, "اسطنبول": 1200}
    price = mock_prices.get(destination, 2500)
    return f"سعر الرحلة إلى {destination} بتاريخ {date}: {price} ريال"

def calculator(expression: str) -> str:
    """يُجري عملية حسابية رياضية بسيطة"""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception:
        return "تعبير رياضي غير صالح"


# --- التشغيل ---
agent = SimpleReActAgent(
    llm_client=client,
    tools={"search_flight_price": search_flight_price, "calculator": calculator}
)

result = agent.run("ما هو سعر رحلة الطيران إلى دبي بتاريخ 2024-12-15، وإذا أضفنا رسوم مطار 150 ريال كم يصبح المجموع؟")
print(f"\n✅ الإجابة النهائية: {result}")

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-10-applications-agents/chapter-55.md
# ────────────────────────────────────────────────────────────────────────
import json

def get_weather(city: str) -> str:
    """دالة حقيقية تُستدعى فعليًا"""
    mock_weather = {"الرياض": "32°C، مشمس", "جدة": "35°C، رطب", "دبي": "38°C، حار جدًا"}
    return mock_weather.get(city, "بيانات الطقس غير متوفرة لهذه المدينة")


# تعريف الأداة بصيغة JSON Schema التي يفهمها النموذج
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "يُرجع حالة الطقس الحالية لمدينة معيّنة",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "اسم المدينة بالعربية"}
                },
                "required": ["city"]
            }
        }
    }
]

available_functions = {"get_weather": get_weather}


def run_function_calling_agent(user_query: str) -> str:
    messages = [{"role": "user", "content": user_query}]

    # الخطوة 1: النموذج يقرر إن كان يحتاج استدعاء دالة أم لا
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools_schema,
        tool_choice="auto",  # يُترك للنموذج القرار (أو "required" لإجباره)
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if not tool_calls:
        return response_message.content  # لم يحتج استدعاء أي أداة

    # الخطوة 2: تنفيذ كل استدعاء دالة طلبه النموذج فعليًا
    messages.append(response_message)

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        function_to_call = available_functions[function_name]
        function_result = function_to_call(**function_args)

        # إضافة نتيجة التنفيذ الفعلي إلى سياق المحادثة
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": function_name,
            "content": function_result,
        })

    # الخطوة 3: النموذج يُولّد إجابة نهائية بناءً على نتائج الأدوات
    final_response = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages
    )
    return final_response.choices[0].message.content


result = run_function_calling_agent("كيف الطقس في جدة اليوم؟")
print(result)

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-10-applications-agents/chapter-55.md
# ────────────────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from typing import TypedDict, Annotated
import operator

# --- تعريف حالة الـ Graph ---
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # تتراكم الرسائل عبر كل خطوة

# --- تعريف الأدوات بصيغة LangChain ---
@tool
def search_product_price(product_name: str) -> str:
    """يبحث عن سعر منتج معيّن في قاعدة بيانات المتجر"""
    mock_db = {"لابتوب": 3500, "هاتف": 2200, "سماعات": 450}
    return f"سعر {product_name}: {mock_db.get(product_name, 'غير متوفر')} ريال"

@tool
def apply_discount(price: float, discount_percent: float) -> str:
    """يُطبّق نسبة خصم على سعر معيّن"""
    final_price = price * (1 - discount_percent / 100)
    return f"السعر بعد الخصم: {final_price:.2f} ريال"

tools = [search_product_price, apply_discount]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

# --- تعريف عقد (Nodes) الرسم البياني ---
def call_model(state: AgentState) -> dict:
    """العقدة المسؤولة عن استدعاء LLM"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def call_tools(state: AgentState) -> dict:
    """العقدة المسؤولة عن تنفيذ الأدوات التي طلبها النموذج"""
    last_message = state["messages"][-1]
    tool_messages = []

    tools_by_name = {t.name: t for t in tools}
    for tool_call in last_message.tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        tool_messages.append(
            ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
        )

    return {"messages": tool_messages}

def should_continue(state: AgentState) -> str:
    """منطق التوجيه: هل نحتاج استدعاء أداة أخرى أم انتهينا؟"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    return "end"

# --- بناء الرسم البياني ---
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", call_tools)
workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent", should_continue, {"continue": "tools", "end": END}
)
workflow.add_edge("tools", "agent")  # بعد تنفيذ الأداة، نعود للنموذج مجددًا

graph = workflow.compile()

# --- التشغيل ---
result = graph.invoke({
    "messages": [HumanMessage(content="ما سعر اللابتوب بعد خصم 20%؟")]
})

print(result["messages"][-1].content)

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-10-applications-agents/chapter-55.md
# ────────────────────────────────────────────────────────────────────────
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# --- تعريف الوكلاء المتخصصين ---
researcher = Agent(
    role="باحث محتوى",
    goal="جمع معلومات دقيقة وحديثة حول الموضوع المطلوب",
    backstory="أنت باحث محترف متخصص في جمع الحقائق والبيانات الموثوقة من مصادر متعددة.",
    llm=llm,
    verbose=True,
)

writer = Agent(
    role="كاتب محتوى",
    goal="كتابة مقال جذاب وواضح بناءً على نتائج البحث",
    backstory="أنت كاتب محترف يُحوّل المعلومات التقنية الجافة إلى محتوى سلس وممتع للقراءة.",
    llm=llm,
    verbose=True,
)

reviewer = Agent(
    role="مراجع جودة",
    goal="مراجعة المقال النهائي والتأكد من دقته ووضوحه وخلوه من الأخطاء",
    backstory="أنت محرر دقيق تراجع كل التفاصيل وتُحسّن جودة النص النهائي.",
    llm=llm,
    verbose=True,
)

# --- تعريف المهام المتسلسلة ---
research_task = Task(
    description="ابحث عن أحدث التطورات في مجال 'الحوسبة الكمومية' وأهم تطبيقاتها العملية الحالية.",
    expected_output="قائمة من 5 نقاط رئيسية مع شرح موجز لكل نقطة.",
    agent=researcher,
)

writing_task = Task(
    description="اكتب مقالًا من 300 كلمة حول الحوسبة الكمومية بناءً على نتائج البحث.",
    expected_output="مقال متكامل وسلس بعنوان جذاب.",
    agent=writer,
    context=[research_task],  # يعتمد على مخرجات مهمة البحث
)

review_task = Task(
    description="راجع المقال وتأكد من دقته العلمية ووضوحه، وأصلح أي أخطاء.",
    expected_output="النسخة النهائية المُحسَّنة من المقال.",
    agent=reviewer,
    context=[writing_task],
)

# --- تجميع الفريق وتشغيله ---
content_crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[research_task, writing_task, review_task],
    process=Process.sequential,  # تنفيذ تسلسلي: بحث → كتابة → مراجعة
    verbose=True,
)

final_result = content_crew.kickoff()
print(f"\n📄 المقال النهائي:\n{final_result}")

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-10-applications-agents/chapter-55.md
# ────────────────────────────────────────────────────────────────────────
import pandas as pd

# محاكاة قاعدة بيانات مبيعات
sales_data = pd.DataFrame({
    "product": ["لابتوب", "هاتف", "سماعات", "لابتوب", "هاتف"],
    "region": ["الرياض", "جدة", "الرياض", "جدة", "الرياض"],
    "revenue": [15000, 8800, 2250, 10500, 13200],
})

@tool
def query_sales_data(filter_column: str, filter_value: str) -> str:
    """يستعلم عن بيانات المبيعات مع فلتر معيّن (مثل product='لابتوب' أو region='الرياض')"""
    filtered = sales_data[sales_data[filter_column] == filter_value]
    if filtered.empty:
        return "لا توجد بيانات مطابقة"
    total_revenue = filtered["revenue"].sum()
    return f"إجمالي الإيرادات لـ {filter_value}: {total_revenue} ريال عبر {len(filtered)} عملية"

@tool
def compare_two_values(value1: float, value2: float) -> str:
    """يقارن بين قيمتين رقميتين ويُحدد الفرق والنسبة المئوية"""
    diff = value1 - value2
    pct = (diff / value2 * 100) if value2 != 0 else 0
    return f"الفرق: {diff:.2f}، نسبة التغيّر: {pct:.1f}%"


# --- بناء وكيل تحليل البيانات باستخدام LangGraph (نفس البنية من الجزء الثالث) ---
analytics_tools = [query_sales_data, compare_two_values]
analytics_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(analytics_tools)

def call_analytics_model(state: AgentState) -> dict:
    response = analytics_llm.invoke(state["messages"])
    return {"messages": [response]}

def call_analytics_tools(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    tools_by_name = {t.name: t for t in analytics_tools}
    tool_messages = [
        ToolMessage(
            content=str(tools_by_name[tc["name"]].invoke(tc["args"])),
            tool_call_id=tc["id"]
        )
        for tc in last_message.tool_calls
    ]
    return {"messages": tool_messages}

analytics_workflow = StateGraph(AgentState)
analytics_workflow.add_node("agent", call_analytics_model)
analytics_workflow.add_node("tools", call_analytics_tools)
analytics_workflow.set_entry_point("agent")
analytics_workflow.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
analytics_workflow.add_edge("tools", "agent")

analytics_agent = analytics_workflow.compile()

# --- الاستخدام ---
result = analytics_agent.invoke({
    "messages": [HumanMessage(
        content="قارن إيرادات اللابتوب في الرياض مقابل جدة، وأخبرني أيهما أعلى وبكم نسبة"
    )]
})
print(result["messages"][-1].content)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-55-chapter-55.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 55: الفصل 55: بناء وكلاء ذكيين (AI Agents)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_tools() -> None:
    tools = {'add': lambda a, b: a + b, 'multiply': lambda a, b: a * b}
    plan = [{'name': 'add', 'arguments': {'a': 2, 'b': 3}}, {'name': 'multiply', 'arguments': {'a': 4, 'b': 5}}]
    results = []
    for call in plan:
        if call['name'] not in tools or set(call['arguments']) != {'a', 'b'}:
            raise ValueError('tool call rejected by allow-list/schema.')
        results.append(tools[call['name']](**call['arguments']))
    print('bounded tool results:', results)
    print('external network/filesystem access: disabled')

if __name__ == "__main__":
    print("Chapter 55 lab: Run an allow-listed, bounded tool-calling agent")
    run_tools()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-55-chapter-55.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# requirements.txt
# openai>=1.40.0
# langchain>=0.3.0
# langchain-openai>=0.2.0
# langgraph>=0.2.0
# crewai>=0.70.0
# duckduckgo-search>=6.0.0

import json
import re
from openai import OpenAI
from typing import Callable

client = OpenAI(api_key="sk-...")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-55-chapter-55.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class SimpleReActAgent:
    """
    تطبيق ReAct يدوي بالكامل، بدون أي إطار خارجي،
    لفهم الآلية الداخلية الحقيقية قبل استخدام LangChain/CrewAI.
    """

    def __init__(self, llm_client: OpenAI, tools: dict[str, Callable], max_iterations: int = 5):
        self.client = llm_client
        self.tools = tools  # {"tool_name": callable_function}
        self.max_iterations = max_iterations

    def _build_system_prompt(self) -> str:
        tool_descriptions = "\n".join(
            f"- {name}: {func.__doc__}" for name, func in self.tools.items()
        )
        return f"""أنت وكيل ذكي يحل المهام عبر التفكير المتسلسل واستخدام الأدوات.

الأدوات المتاحة لك:
{tool_descriptions}

يجب أن تتبع الصيغة التالية بدقة في كل خطوة:
Thought: [تفكيرك في الخطوة التالية المطلوبة]
Action: [اسم الأداة المطلوب استخدامها، أو "Finish" إذا انتهيت]
Action Input: [المدخل المطلوب للأداة بصيغة JSON]

بعد كل Action، ستحصل على Observation تحتوي النتيجة الفعلية.
عندما تصل لإجابة نهائية كافية، استخدم Action: Finish مع الإجابة في Action Input.
"""

    def _parse_llm_output(self, text: str) -> dict:
        """تحليل مخرجات LLM لاستخراج Thought/Action/Action Input"""
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", text, re.DOTALL)
        action_match = re.search(r"Action:\s*(.+?)(?=Action Input:|$)", text, re.DOTALL)
        input_match = re.search(r"Action Input:\s*(.+?)$", text, re.DOTALL)

        return {
            "thought": thought_match.group(1).strip() if thought_match else "",
            "action": action_match.group(1).strip() if action_match else "",
            "action_input": input_match.group(1).strip() if input_match else "",
        }

    def run(self, task: str) -> str:
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": f"المهمة: {task}"}
        ]

        for iteration in range(self.max_iterations):
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.2,
            )
            llm_output = response.choices[0].message.content
            print(f"\n--- التكرار {iteration + 1} ---\n{llm_output}")

            parsed = self._parse_llm_output(llm_output)

            if parsed["action"].strip().lower() == "finish":
                return parsed["action_input"]

            # تنفيذ الأداة الفعلية
            tool_name = parsed["action"].strip()
            if tool_name not in self.tools:
                observation = f"خطأ: الأداة '{tool_name}' غير موجودة."
            else:
                try:
                    tool_args = json.loads(parsed["action_input"])
                    observation = str(self.tools[tool_name](**tool_args))
                except Exception as e:
                    observation = f"خطأ في تنفيذ الأداة: {str(e)}"

            print(f"🔍 Observation: {observation}")

            # إضافة السياق التراكمي للمحادثة
            messages.append({"role": "assistant", "content": llm_output})
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        return "⚠️ تجاوز الوكيل الحد الأقصى للتكرارات دون الوصول لإجابة نهائية."


# --- تعريف الأدوات ---
def search_flight_price(destination: str, date: str) -> str:
    """يبحث عن سعر رحلة طيران لوجهة معيّنة وتاريخ محدد"""
    # في الواقع: استدعاء API حقيقي لموقع حجز طيران
    mock_prices = {"دبي": 1850, "القاهرة": 950, "اسطنبول": 1200}
    price = mock_prices.get(destination, 2500)
    return f"سعر الرحلة إلى {destination} بتاريخ {date}: {price} ريال"

def calculator(expression: str) -> str:
    """يُجري عملية حسابية رياضية بسيطة"""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception:
        return "تعبير رياضي غير صالح"


# --- التشغيل ---
agent = SimpleReActAgent(
    llm_client=client,
    tools={"search_flight_price": search_flight_price, "calculator": calculator}
)

result = agent.run("ما هو سعر رحلة الطيران إلى دبي بتاريخ 2024-12-15، وإذا أضفنا رسوم مطار 150 ريال كم يصبح المجموع؟")
print(f"\n✅ الإجابة النهائية: {result}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-55-chapter-55.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import json

def get_weather(city: str) -> str:
    """دالة حقيقية تُستدعى فعليًا"""
    mock_weather = {"الرياض": "32°C، مشمس", "جدة": "35°C، رطب", "دبي": "38°C، حار جدًا"}
    return mock_weather.get(city, "بيانات الطقس غير متوفرة لهذه المدينة")


# تعريف الأداة بصيغة JSON Schema التي يفهمها النموذج
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "يُرجع حالة الطقس الحالية لمدينة معيّنة",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "اسم المدينة بالعربية"}
                },
                "required": ["city"]
            }
        }
    }
]

available_functions = {"get_weather": get_weather}


def run_function_calling_agent(user_query: str) -> str:
    messages = [{"role": "user", "content": user_query}]

    # الخطوة 1: النموذج يقرر إن كان يحتاج استدعاء دالة أم لا
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools_schema,
        tool_choice="auto",  # يُترك للنموذج القرار (أو "required" لإجباره)
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if not tool_calls:
        return response_message.content  # لم يحتج استدعاء أي أداة

    # الخطوة 2: تنفيذ كل استدعاء دالة طلبه النموذج فعليًا
    messages.append(response_message)

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        function_to_call = available_functions[function_name]
        function_result = function_to_call(**function_args)

        # إضافة نتيجة التنفيذ الفعلي إلى سياق المحادثة
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": function_name,
            "content": function_result,
        })

    # الخطوة 3: النموذج يُولّد إجابة نهائية بناءً على نتائج الأدوات
    final_response = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages
    )
    return final_response.choices[0].message.content


result = run_function_calling_agent("كيف الطقس في جدة اليوم؟")
print(result)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-55-chapter-55.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from typing import TypedDict, Annotated
import operator

# --- تعريف حالة الـ Graph ---
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # تتراكم الرسائل عبر كل خطوة

# --- تعريف الأدوات بصيغة LangChain ---
@tool
def search_product_price(product_name: str) -> str:
    """يبحث عن سعر منتج معيّن في قاعدة بيانات المتجر"""
    mock_db = {"لابتوب": 3500, "هاتف": 2200, "سماعات": 450}
    return f"سعر {product_name}: {mock_db.get(product_name, 'غير متوفر')} ريال"

@tool
def apply_discount(price: float, discount_percent: float) -> str:
    """يُطبّق نسبة خصم على سعر معيّن"""
    final_price = price * (1 - discount_percent / 100)
    return f"السعر بعد الخصم: {final_price:.2f} ريال"

tools = [search_product_price, apply_discount]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

# --- تعريف عقد (Nodes) الرسم البياني ---
def call_model(state: AgentState) -> dict:
    """العقدة المسؤولة عن استدعاء LLM"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def call_tools(state: AgentState) -> dict:
    """العقدة المسؤولة عن تنفيذ الأدوات التي طلبها النموذج"""
    last_message = state["messages"][-1]
    tool_messages = []

    tools_by_name = {t.name: t for t in tools}
    for tool_call in last_message.tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        tool_messages.append(
            ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
        )

    return {"messages": tool_messages}

def should_continue(state: AgentState) -> str:
    """منطق التوجيه: هل نحتاج استدعاء أداة أخرى أم انتهينا؟"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    return "end"

# --- بناء الرسم البياني ---
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", call_tools)
workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent", should_continue, {"continue": "tools", "end": END}
)
workflow.add_edge("tools", "agent")  # بعد تنفيذ الأداة، نعود للنموذج مجددًا

graph = workflow.compile()

# --- التشغيل ---
result = graph.invoke({
    "messages": [HumanMessage(content="ما سعر اللابتوب بعد خصم 20%؟")]
})

print(result["messages"][-1].content)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-55-chapter-55.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# --- تعريف الوكلاء المتخصصين ---
researcher = Agent(
    role="باحث محتوى",
    goal="جمع معلومات دقيقة وحديثة حول الموضوع المطلوب",
    backstory="أنت باحث محترف متخصص في جمع الحقائق والبيانات الموثوقة من مصادر متعددة.",
    llm=llm,
    verbose=True,
)

writer = Agent(
    role="كاتب محتوى",
    goal="كتابة مقال جذاب وواضح بناءً على نتائج البحث",
    backstory="أنت كاتب محترف يُحوّل المعلومات التقنية الجافة إلى محتوى سلس وممتع للقراءة.",
    llm=llm,
    verbose=True,
)

reviewer = Agent(
    role="مراجع جودة",
    goal="مراجعة المقال النهائي والتأكد من دقته ووضوحه وخلوه من الأخطاء",
    backstory="أنت محرر دقيق تراجع كل التفاصيل وتُحسّن جودة النص النهائي.",
    llm=llm,
    verbose=True,
)

# --- تعريف المهام المتسلسلة ---
research_task = Task(
    description="ابحث عن أحدث التطورات في مجال 'الحوسبة الكمومية' وأهم تطبيقاتها العملية الحالية.",
    expected_output="قائمة من 5 نقاط رئيسية مع شرح موجز لكل نقطة.",
    agent=researcher,
)

writing_task = Task(
    description="اكتب مقالًا من 300 كلمة حول الحوسبة الكمومية بناءً على نتائج البحث.",
    expected_output="مقال متكامل وسلس بعنوان جذاب.",
    agent=writer,
    context=[research_task],  # يعتمد على مخرجات مهمة البحث
)

review_task = Task(
    description="راجع المقال وتأكد من دقته العلمية ووضوحه، وأصلح أي أخطاء.",
    expected_output="النسخة النهائية المُحسَّنة من المقال.",
    agent=reviewer,
    context=[writing_task],
)

# --- تجميع الفريق وتشغيله ---
content_crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[research_task, writing_task, review_task],
    process=Process.sequential,  # تنفيذ تسلسلي: بحث → كتابة → مراجعة
    verbose=True,
)

final_result = content_crew.kickoff()
print(f"\n📄 المقال النهائي:\n{final_result}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-55-chapter-55.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import pandas as pd

# محاكاة قاعدة بيانات مبيعات
sales_data = pd.DataFrame({
    "product": ["لابتوب", "هاتف", "سماعات", "لابتوب", "هاتف"],
    "region": ["الرياض", "جدة", "الرياض", "جدة", "الرياض"],
    "revenue": [15000, 8800, 2250, 10500, 13200],
})

@tool
def query_sales_data(filter_column: str, filter_value: str) -> str:
    """يستعلم عن بيانات المبيعات مع فلتر معيّن (مثل product='لابتوب' أو region='الرياض')"""
    filtered = sales_data[sales_data[filter_column] == filter_value]
    if filtered.empty:
        return "لا توجد بيانات مطابقة"
    total_revenue = filtered["revenue"].sum()
    return f"إجمالي الإيرادات لـ {filter_value}: {total_revenue} ريال عبر {len(filtered)} عملية"

@tool
def compare_two_values(value1: float, value2: float) -> str:
    """يقارن بين قيمتين رقميتين ويُحدد الفرق والنسبة المئوية"""
    diff = value1 - value2
    pct = (diff / value2 * 100) if value2 != 0 else 0
    return f"الفرق: {diff:.2f}، نسبة التغيّر: {pct:.1f}%"


# --- بناء وكيل تحليل البيانات باستخدام LangGraph (نفس البنية من الجزء الثالث) ---
analytics_tools = [query_sales_data, compare_two_values]
analytics_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(analytics_tools)

def call_analytics_model(state: AgentState) -> dict:
    response = analytics_llm.invoke(state["messages"])
    return {"messages": [response]}

def call_analytics_tools(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    tools_by_name = {t.name: t for t in analytics_tools}
    tool_messages = [
        ToolMessage(
            content=str(tools_by_name[tc["name"]].invoke(tc["args"])),
            tool_call_id=tc["id"]
        )
        for tc in last_message.tool_calls
    ]
    return {"messages": tool_messages}

analytics_workflow = StateGraph(AgentState)
analytics_workflow.add_node("agent", call_analytics_model)
analytics_workflow.add_node("tools", call_analytics_tools)
analytics_workflow.set_entry_point("agent")
analytics_workflow.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
analytics_workflow.add_edge("tools", "agent")

analytics_agent = analytics_workflow.compile()

# --- الاستخدام ---
result = analytics_agent.invoke({
    "messages": [HumanMessage(
        content="قارن إيرادات اللابتوب في الرياض مقابل جدة، وأخبرني أيهما أعلى وبكم نسبة"
    )]
})
print(result["messages"][-1].content)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-55-chapter-55.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — بناء وكلاء ذكيين (AI Agents)
Large Language Model Engineering — Agents

Book slug : llm-engineering-69-chapters
Chapter   : ch-55-chapter-55 (lesson 55)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_55.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 55: الفصل 55: بناء وكلاء ذكيين (AI Agents)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_tools() -> None:
    tools = {'add': lambda a, b: a + b, 'multiply': lambda a, b: a * b}
    plan = [{'name': 'add', 'arguments': {'a': 2, 'b': 3}}, {'name': 'multiply', 'arguments': {'a': 4, 'b': 5}}]
    results = []
    for call in plan:
        if call['name'] not in tools or set(call['arguments']) != {'a', 'b'}:
            raise ValueError('tool call rejected by allow-list/schema.')
        results.append(tools[call['name']](**call['arguments']))
    print('bounded tool results:', results)
    print('external network/filesystem access: disabled')

if __name__ == "__main__":
    print("Chapter 55 lab: Run an allow-listed, bounded tool-calling agent")
    run_tools()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-10-applications-agents/chapter-55.md
# ────────────────────────────────────────────────────────────────────────
# requirements.txt
# openai>=1.40.0
# langchain>=0.3.0
# langchain-openai>=0.2.0
# langgraph>=0.2.0
# crewai>=0.70.0
# duckduckgo-search>=6.0.0

import json
import re
from openai import OpenAI
from typing import Callable

client = OpenAI(api_key="sk-...")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-10-applications-agents/chapter-55.md
# ────────────────────────────────────────────────────────────────────────
class SimpleReActAgent:
    """
    تطبيق ReAct يدوي بالكامل، بدون أي إطار خارجي،
    لفهم الآلية الداخلية الحقيقية قبل استخدام LangChain/CrewAI.
    """

    def __init__(self, llm_client: OpenAI, tools: dict[str, Callable], max_iterations: int = 5):
        self.client = llm_client
        self.tools = tools  # {"tool_name": callable_function}
        self.max_iterations = max_iterations

    def _build_system_prompt(self) -> str:
        tool_descriptions = "\n".join(
            f"- {name}: {func.__doc__}" for name, func in self.tools.items()
        )
        return f"""أنت وكيل ذكي يحل المهام عبر التفكير المتسلسل واستخدام الأدوات.

الأدوات المتاحة لك:
{tool_descriptions}

يجب أن تتبع الصيغة التالية بدقة في كل خطوة:
Thought: [تفكيرك في الخطوة التالية المطلوبة]
Action: [اسم الأداة المطلوب استخدامها، أو "Finish" إذا انتهيت]
Action Input: [المدخل المطلوب للأداة بصيغة JSON]

بعد كل Action، ستحصل على Observation تحتوي النتيجة الفعلية.
عندما تصل لإجابة نهائية كافية، استخدم Action: Finish مع الإجابة في Action Input.
"""

    def _parse_llm_output(self, text: str) -> dict:
        """تحليل مخرجات LLM لاستخراج Thought/Action/Action Input"""
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", text, re.DOTALL)
        action_match = re.search(r"Action:\s*(.+?)(?=Action Input:|$)", text, re.DOTALL)
        input_match = re.search(r"Action Input:\s*(.+?)$", text, re.DOTALL)

        return {
            "thought": thought_match.group(1).strip() if thought_match else "",
            "action": action_match.group(1).strip() if action_match else "",
            "action_input": input_match.group(1).strip() if input_match else "",
        }

    def run(self, task: str) -> str:
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": f"المهمة: {task}"}
        ]

        for iteration in range(self.max_iterations):
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.2,
            )
            llm_output = response.choices[0].message.content
            print(f"\n--- التكرار {iteration + 1} ---\n{llm_output}")

            parsed = self._parse_llm_output(llm_output)

            if parsed["action"].strip().lower() == "finish":
                return parsed["action_input"]

            # تنفيذ الأداة الفعلية
            tool_name = parsed["action"].strip()
            if tool_name not in self.tools:
                observation = f"خطأ: الأداة '{tool_name}' غير موجودة."
            else:
                try:
                    tool_args = json.loads(parsed["action_input"])
                    observation = str(self.tools[tool_name](**tool_args))
                except Exception as e:
                    observation = f"خطأ في تنفيذ الأداة: {str(e)}"

            print(f"🔍 Observation: {observation}")

            # إضافة السياق التراكمي للمحادثة
            messages.append({"role": "assistant", "content": llm_output})
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        return "⚠️ تجاوز الوكيل الحد الأقصى للتكرارات دون الوصول لإجابة نهائية."


# --- تعريف الأدوات ---
def search_flight_price(destination: str, date: str) -> str:
    """يبحث عن سعر رحلة طيران لوجهة معيّنة وتاريخ محدد"""
    # في الواقع: استدعاء API حقيقي لموقع حجز طيران
    mock_prices = {"دبي": 1850, "القاهرة": 950, "اسطنبول": 1200}
    price = mock_prices.get(destination, 2500)
    return f"سعر الرحلة إلى {destination} بتاريخ {date}: {price} ريال"

def calculator(expression: str) -> str:
    """يُجري عملية حسابية رياضية بسيطة"""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception:
        return "تعبير رياضي غير صالح"


# --- التشغيل ---
agent = SimpleReActAgent(
    llm_client=client,
    tools={"search_flight_price": search_flight_price, "calculator": calculator}
)

result = agent.run("ما هو سعر رحلة الطيران إلى دبي بتاريخ 2024-12-15، وإذا أضفنا رسوم مطار 150 ريال كم يصبح المجموع؟")
print(f"\n✅ الإجابة النهائية: {result}")

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-10-applications-agents/chapter-55.md
# ────────────────────────────────────────────────────────────────────────
import json

def get_weather(city: str) -> str:
    """دالة حقيقية تُستدعى فعليًا"""
    mock_weather = {"الرياض": "32°C، مشمس", "جدة": "35°C، رطب", "دبي": "38°C، حار جدًا"}
    return mock_weather.get(city, "بيانات الطقس غير متوفرة لهذه المدينة")


# تعريف الأداة بصيغة JSON Schema التي يفهمها النموذج
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "يُرجع حالة الطقس الحالية لمدينة معيّنة",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "اسم المدينة بالعربية"}
                },
                "required": ["city"]
            }
        }
    }
]

available_functions = {"get_weather": get_weather}


def run_function_calling_agent(user_query: str) -> str:
    messages = [{"role": "user", "content": user_query}]

    # الخطوة 1: النموذج يقرر إن كان يحتاج استدعاء دالة أم لا
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools_schema,
        tool_choice="auto",  # يُترك للنموذج القرار (أو "required" لإجباره)
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if not tool_calls:
        return response_message.content  # لم يحتج استدعاء أي أداة

    # الخطوة 2: تنفيذ كل استدعاء دالة طلبه النموذج فعليًا
    messages.append(response_message)

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        function_to_call = available_functions[function_name]
        function_result = function_to_call(**function_args)

        # إضافة نتيجة التنفيذ الفعلي إلى سياق المحادثة
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": function_name,
            "content": function_result,
        })

    # الخطوة 3: النموذج يُولّد إجابة نهائية بناءً على نتائج الأدوات
    final_response = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages
    )
    return final_response.choices[0].message.content


result = run_function_calling_agent("كيف الطقس في جدة اليوم؟")
print(result)

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-10-applications-agents/chapter-55.md
# ────────────────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from typing import TypedDict, Annotated
import operator

# --- تعريف حالة الـ Graph ---
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # تتراكم الرسائل عبر كل خطوة

# --- تعريف الأدوات بصيغة LangChain ---
@tool
def search_product_price(product_name: str) -> str:
    """يبحث عن سعر منتج معيّن في قاعدة بيانات المتجر"""
    mock_db = {"لابتوب": 3500, "هاتف": 2200, "سماعات": 450}
    return f"سعر {product_name}: {mock_db.get(product_name, 'غير متوفر')} ريال"

@tool
def apply_discount(price: float, discount_percent: float) -> str:
    """يُطبّق نسبة خصم على سعر معيّن"""
    final_price = price * (1 - discount_percent / 100)
    return f"السعر بعد الخصم: {final_price:.2f} ريال"

tools = [search_product_price, apply_discount]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

# --- تعريف عقد (Nodes) الرسم البياني ---
def call_model(state: AgentState) -> dict:
    """العقدة المسؤولة عن استدعاء LLM"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def call_tools(state: AgentState) -> dict:
    """العقدة المسؤولة عن تنفيذ الأدوات التي طلبها النموذج"""
    last_message = state["messages"][-1]
    tool_messages = []

    tools_by_name = {t.name: t for t in tools}
    for tool_call in last_message.tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        tool_messages.append(
            ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
        )

    return {"messages": tool_messages}

def should_continue(state: AgentState) -> str:
    """منطق التوجيه: هل نحتاج استدعاء أداة أخرى أم انتهينا؟"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    return "end"

# --- بناء الرسم البياني ---
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", call_tools)
workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent", should_continue, {"continue": "tools", "end": END}
)
workflow.add_edge("tools", "agent")  # بعد تنفيذ الأداة، نعود للنموذج مجددًا

graph = workflow.compile()

# --- التشغيل ---
result = graph.invoke({
    "messages": [HumanMessage(content="ما سعر اللابتوب بعد خصم 20%؟")]
})

print(result["messages"][-1].content)

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-10-applications-agents/chapter-55.md
# ────────────────────────────────────────────────────────────────────────
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# --- تعريف الوكلاء المتخصصين ---
researcher = Agent(
    role="باحث محتوى",
    goal="جمع معلومات دقيقة وحديثة حول الموضوع المطلوب",
    backstory="أنت باحث محترف متخصص في جمع الحقائق والبيانات الموثوقة من مصادر متعددة.",
    llm=llm,
    verbose=True,
)

writer = Agent(
    role="كاتب محتوى",
    goal="كتابة مقال جذاب وواضح بناءً على نتائج البحث",
    backstory="أنت كاتب محترف يُحوّل المعلومات التقنية الجافة إلى محتوى سلس وممتع للقراءة.",
    llm=llm,
    verbose=True,
)

reviewer = Agent(
    role="مراجع جودة",
    goal="مراجعة المقال النهائي والتأكد من دقته ووضوحه وخلوه من الأخطاء",
    backstory="أنت محرر دقيق تراجع كل التفاصيل وتُحسّن جودة النص النهائي.",
    llm=llm,
    verbose=True,
)

# --- تعريف المهام المتسلسلة ---
research_task = Task(
    description="ابحث عن أحدث التطورات في مجال 'الحوسبة الكمومية' وأهم تطبيقاتها العملية الحالية.",
    expected_output="قائمة من 5 نقاط رئيسية مع شرح موجز لكل نقطة.",
    agent=researcher,
)

writing_task = Task(
    description="اكتب مقالًا من 300 كلمة حول الحوسبة الكمومية بناءً على نتائج البحث.",
    expected_output="مقال متكامل وسلس بعنوان جذاب.",
    agent=writer,
    context=[research_task],  # يعتمد على مخرجات مهمة البحث
)

review_task = Task(
    description="راجع المقال وتأكد من دقته العلمية ووضوحه، وأصلح أي أخطاء.",
    expected_output="النسخة النهائية المُحسَّنة من المقال.",
    agent=reviewer,
    context=[writing_task],
)

# --- تجميع الفريق وتشغيله ---
content_crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[research_task, writing_task, review_task],
    process=Process.sequential,  # تنفيذ تسلسلي: بحث → كتابة → مراجعة
    verbose=True,
)

final_result = content_crew.kickoff()
print(f"\n📄 المقال النهائي:\n{final_result}")

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-10-applications-agents/chapter-55.md
# ────────────────────────────────────────────────────────────────────────
import pandas as pd

# محاكاة قاعدة بيانات مبيعات
sales_data = pd.DataFrame({
    "product": ["لابتوب", "هاتف", "سماعات", "لابتوب", "هاتف"],
    "region": ["الرياض", "جدة", "الرياض", "جدة", "الرياض"],
    "revenue": [15000, 8800, 2250, 10500, 13200],
})

@tool
def query_sales_data(filter_column: str, filter_value: str) -> str:
    """يستعلم عن بيانات المبيعات مع فلتر معيّن (مثل product='لابتوب' أو region='الرياض')"""
    filtered = sales_data[sales_data[filter_column] == filter_value]
    if filtered.empty:
        return "لا توجد بيانات مطابقة"
    total_revenue = filtered["revenue"].sum()
    return f"إجمالي الإيرادات لـ {filter_value}: {total_revenue} ريال عبر {len(filtered)} عملية"

@tool
def compare_two_values(value1: float, value2: float) -> str:
    """يقارن بين قيمتين رقميتين ويُحدد الفرق والنسبة المئوية"""
    diff = value1 - value2
    pct = (diff / value2 * 100) if value2 != 0 else 0
    return f"الفرق: {diff:.2f}، نسبة التغيّر: {pct:.1f}%"


# --- بناء وكيل تحليل البيانات باستخدام LangGraph (نفس البنية من الجزء الثالث) ---
analytics_tools = [query_sales_data, compare_two_values]
analytics_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(analytics_tools)

def call_analytics_model(state: AgentState) -> dict:
    response = analytics_llm.invoke(state["messages"])
    return {"messages": [response]}

def call_analytics_tools(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    tools_by_name = {t.name: t for t in analytics_tools}
    tool_messages = [
        ToolMessage(
            content=str(tools_by_name[tc["name"]].invoke(tc["args"])),
            tool_call_id=tc["id"]
        )
        for tc in last_message.tool_calls
    ]
    return {"messages": tool_messages}

analytics_workflow = StateGraph(AgentState)
analytics_workflow.add_node("agent", call_analytics_model)
analytics_workflow.add_node("tools", call_analytics_tools)
analytics_workflow.set_entry_point("agent")
analytics_workflow.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
analytics_workflow.add_edge("tools", "agent")

analytics_agent = analytics_workflow.compile()

# --- الاستخدام ---
result = analytics_agent.invoke({
    "messages": [HumanMessage(
        content="قارن إيرادات اللابتوب في الرياض مقابل جدة، وأخبرني أيهما أعلى وبكم نسبة"
    )]
})
print(result["messages"][-1].content)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-55-chapter-55.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 55: الفصل 55: بناء وكلاء ذكيين (AI Agents)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_tools() -> None:
    tools = {'add': lambda a, b: a + b, 'multiply': lambda a, b: a * b}
    plan = [{'name': 'add', 'arguments': {'a': 2, 'b': 3}}, {'name': 'multiply', 'arguments': {'a': 4, 'b': 5}}]
    results = []
    for call in plan:
        if call['name'] not in tools or set(call['arguments']) != {'a', 'b'}:
            raise ValueError('tool call rejected by allow-list/schema.')
        results.append(tools[call['name']](**call['arguments']))
    print('bounded tool results:', results)
    print('external network/filesystem access: disabled')

if __name__ == "__main__":
    print("Chapter 55 lab: Run an allow-listed, bounded tool-calling agent")
    run_tools()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-55-chapter-55.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# requirements.txt
# openai>=1.40.0
# langchain>=0.3.0
# langchain-openai>=0.2.0
# langgraph>=0.2.0
# crewai>=0.70.0
# duckduckgo-search>=6.0.0

import json
import re
from openai import OpenAI
from typing import Callable

client = OpenAI(api_key="sk-...")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-55-chapter-55.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class SimpleReActAgent:
    """
    تطبيق ReAct يدوي بالكامل، بدون أي إطار خارجي،
    لفهم الآلية الداخلية الحقيقية قبل استخدام LangChain/CrewAI.
    """

    def __init__(self, llm_client: OpenAI, tools: dict[str, Callable], max_iterations: int = 5):
        self.client = llm_client
        self.tools = tools  # {"tool_name": callable_function}
        self.max_iterations = max_iterations

    def _build_system_prompt(self) -> str:
        tool_descriptions = "\n".join(
            f"- {name}: {func.__doc__}" for name, func in self.tools.items()
        )
        return f"""أنت وكيل ذكي يحل المهام عبر التفكير المتسلسل واستخدام الأدوات.

الأدوات المتاحة لك:
{tool_descriptions}

يجب أن تتبع الصيغة التالية بدقة في كل خطوة:
Thought: [تفكيرك في الخطوة التالية المطلوبة]
Action: [اسم الأداة المطلوب استخدامها، أو "Finish" إذا انتهيت]
Action Input: [المدخل المطلوب للأداة بصيغة JSON]

بعد كل Action، ستحصل على Observation تحتوي النتيجة الفعلية.
عندما تصل لإجابة نهائية كافية، استخدم Action: Finish مع الإجابة في Action Input.
"""

    def _parse_llm_output(self, text: str) -> dict:
        """تحليل مخرجات LLM لاستخراج Thought/Action/Action Input"""
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", text, re.DOTALL)
        action_match = re.search(r"Action:\s*(.+?)(?=Action Input:|$)", text, re.DOTALL)
        input_match = re.search(r"Action Input:\s*(.+?)$", text, re.DOTALL)

        return {
            "thought": thought_match.group(1).strip() if thought_match else "",
            "action": action_match.group(1).strip() if action_match else "",
            "action_input": input_match.group(1).strip() if input_match else "",
        }

    def run(self, task: str) -> str:
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": f"المهمة: {task}"}
        ]

        for iteration in range(self.max_iterations):
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.2,
            )
            llm_output = response.choices[0].message.content
            print(f"\n--- التكرار {iteration + 1} ---\n{llm_output}")

            parsed = self._parse_llm_output(llm_output)

            if parsed["action"].strip().lower() == "finish":
                return parsed["action_input"]

            # تنفيذ الأداة الفعلية
            tool_name = parsed["action"].strip()
            if tool_name not in self.tools:
                observation = f"خطأ: الأداة '{tool_name}' غير موجودة."
            else:
                try:
                    tool_args = json.loads(parsed["action_input"])
                    observation = str(self.tools[tool_name](**tool_args))
                except Exception as e:
                    observation = f"خطأ في تنفيذ الأداة: {str(e)}"

            print(f"🔍 Observation: {observation}")

            # إضافة السياق التراكمي للمحادثة
            messages.append({"role": "assistant", "content": llm_output})
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        return "⚠️ تجاوز الوكيل الحد الأقصى للتكرارات دون الوصول لإجابة نهائية."


# --- تعريف الأدوات ---
def search_flight_price(destination: str, date: str) -> str:
    """يبحث عن سعر رحلة طيران لوجهة معيّنة وتاريخ محدد"""
    # في الواقع: استدعاء API حقيقي لموقع حجز طيران
    mock_prices = {"دبي": 1850, "القاهرة": 950, "اسطنبول": 1200}
    price = mock_prices.get(destination, 2500)
    return f"سعر الرحلة إلى {destination} بتاريخ {date}: {price} ريال"

def calculator(expression: str) -> str:
    """يُجري عملية حسابية رياضية بسيطة"""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception:
        return "تعبير رياضي غير صالح"


# --- التشغيل ---
agent = SimpleReActAgent(
    llm_client=client,
    tools={"search_flight_price": search_flight_price, "calculator": calculator}
)

result = agent.run("ما هو سعر رحلة الطيران إلى دبي بتاريخ 2024-12-15، وإذا أضفنا رسوم مطار 150 ريال كم يصبح المجموع؟")
print(f"\n✅ الإجابة النهائية: {result}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-55-chapter-55.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import json

def get_weather(city: str) -> str:
    """دالة حقيقية تُستدعى فعليًا"""
    mock_weather = {"الرياض": "32°C، مشمس", "جدة": "35°C، رطب", "دبي": "38°C، حار جدًا"}
    return mock_weather.get(city, "بيانات الطقس غير متوفرة لهذه المدينة")


# تعريف الأداة بصيغة JSON Schema التي يفهمها النموذج
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "يُرجع حالة الطقس الحالية لمدينة معيّنة",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "اسم المدينة بالعربية"}
                },
                "required": ["city"]
            }
        }
    }
]

available_functions = {"get_weather": get_weather}


def run_function_calling_agent(user_query: str) -> str:
    messages = [{"role": "user", "content": user_query}]

    # الخطوة 1: النموذج يقرر إن كان يحتاج استدعاء دالة أم لا
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools_schema,
        tool_choice="auto",  # يُترك للنموذج القرار (أو "required" لإجباره)
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if not tool_calls:
        return response_message.content  # لم يحتج استدعاء أي أداة

    # الخطوة 2: تنفيذ كل استدعاء دالة طلبه النموذج فعليًا
    messages.append(response_message)

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        function_to_call = available_functions[function_name]
        function_result = function_to_call(**function_args)

        # إضافة نتيجة التنفيذ الفعلي إلى سياق المحادثة
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": function_name,
            "content": function_result,
        })

    # الخطوة 3: النموذج يُولّد إجابة نهائية بناءً على نتائج الأدوات
    final_response = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages
    )
    return final_response.choices[0].message.content


result = run_function_calling_agent("كيف الطقس في جدة اليوم؟")
print(result)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-55-chapter-55.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from typing import TypedDict, Annotated
import operator

# --- تعريف حالة الـ Graph ---
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # تتراكم الرسائل عبر كل خطوة

# --- تعريف الأدوات بصيغة LangChain ---
@tool
def search_product_price(product_name: str) -> str:
    """يبحث عن سعر منتج معيّن في قاعدة بيانات المتجر"""
    mock_db = {"لابتوب": 3500, "هاتف": 2200, "سماعات": 450}
    return f"سعر {product_name}: {mock_db.get(product_name, 'غير متوفر')} ريال"

@tool
def apply_discount(price: float, discount_percent: float) -> str:
    """يُطبّق نسبة خصم على سعر معيّن"""
    final_price = price * (1 - discount_percent / 100)
    return f"السعر بعد الخصم: {final_price:.2f} ريال"

tools = [search_product_price, apply_discount]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

# --- تعريف عقد (Nodes) الرسم البياني ---
def call_model(state: AgentState) -> dict:
    """العقدة المسؤولة عن استدعاء LLM"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def call_tools(state: AgentState) -> dict:
    """العقدة المسؤولة عن تنفيذ الأدوات التي طلبها النموذج"""
    last_message = state["messages"][-1]
    tool_messages = []

    tools_by_name = {t.name: t for t in tools}
    for tool_call in last_message.tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        tool_messages.append(
            ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
        )

    return {"messages": tool_messages}

def should_continue(state: AgentState) -> str:
    """منطق التوجيه: هل نحتاج استدعاء أداة أخرى أم انتهينا؟"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    return "end"

# --- بناء الرسم البياني ---
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", call_tools)
workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent", should_continue, {"continue": "tools", "end": END}
)
workflow.add_edge("tools", "agent")  # بعد تنفيذ الأداة، نعود للنموذج مجددًا

graph = workflow.compile()

# --- التشغيل ---
result = graph.invoke({
    "messages": [HumanMessage(content="ما سعر اللابتوب بعد خصم 20%؟")]
})

print(result["messages"][-1].content)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-55-chapter-55.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# --- تعريف الوكلاء المتخصصين ---
researcher = Agent(
    role="باحث محتوى",
    goal="جمع معلومات دقيقة وحديثة حول الموضوع المطلوب",
    backstory="أنت باحث محترف متخصص في جمع الحقائق والبيانات الموثوقة من مصادر متعددة.",
    llm=llm,
    verbose=True,
)

writer = Agent(
    role="كاتب محتوى",
    goal="كتابة مقال جذاب وواضح بناءً على نتائج البحث",
    backstory="أنت كاتب محترف يُحوّل المعلومات التقنية الجافة إلى محتوى سلس وممتع للقراءة.",
    llm=llm,
    verbose=True,
)

reviewer = Agent(
    role="مراجع جودة",
    goal="مراجعة المقال النهائي والتأكد من دقته ووضوحه وخلوه من الأخطاء",
    backstory="أنت محرر دقيق تراجع كل التفاصيل وتُحسّن جودة النص النهائي.",
    llm=llm,
    verbose=True,
)

# --- تعريف المهام المتسلسلة ---
research_task = Task(
    description="ابحث عن أحدث التطورات في مجال 'الحوسبة الكمومية' وأهم تطبيقاتها العملية الحالية.",
    expected_output="قائمة من 5 نقاط رئيسية مع شرح موجز لكل نقطة.",
    agent=researcher,
)

writing_task = Task(
    description="اكتب مقالًا من 300 كلمة حول الحوسبة الكمومية بناءً على نتائج البحث.",
    expected_output="مقال متكامل وسلس بعنوان جذاب.",
    agent=writer,
    context=[research_task],  # يعتمد على مخرجات مهمة البحث
)

review_task = Task(
    description="راجع المقال وتأكد من دقته العلمية ووضوحه، وأصلح أي أخطاء.",
    expected_output="النسخة النهائية المُحسَّنة من المقال.",
    agent=reviewer,
    context=[writing_task],
)

# --- تجميع الفريق وتشغيله ---
content_crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[research_task, writing_task, review_task],
    process=Process.sequential,  # تنفيذ تسلسلي: بحث → كتابة → مراجعة
    verbose=True,
)

final_result = content_crew.kickoff()
print(f"\n📄 المقال النهائي:\n{final_result}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-55-chapter-55.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import pandas as pd

# محاكاة قاعدة بيانات مبيعات
sales_data = pd.DataFrame({
    "product": ["لابتوب", "هاتف", "سماعات", "لابتوب", "هاتف"],
    "region": ["الرياض", "جدة", "الرياض", "جدة", "الرياض"],
    "revenue": [15000, 8800, 2250, 10500, 13200],
})

@tool
def query_sales_data(filter_column: str, filter_value: str) -> str:
    """يستعلم عن بيانات المبيعات مع فلتر معيّن (مثل product='لابتوب' أو region='الرياض')"""
    filtered = sales_data[sales_data[filter_column] == filter_value]
    if filtered.empty:
        return "لا توجد بيانات مطابقة"
    total_revenue = filtered["revenue"].sum()
    return f"إجمالي الإيرادات لـ {filter_value}: {total_revenue} ريال عبر {len(filtered)} عملية"

@tool
def compare_two_values(value1: float, value2: float) -> str:
    """يقارن بين قيمتين رقميتين ويُحدد الفرق والنسبة المئوية"""
    diff = value1 - value2
    pct = (diff / value2 * 100) if value2 != 0 else 0
    return f"الفرق: {diff:.2f}، نسبة التغيّر: {pct:.1f}%"


# --- بناء وكيل تحليل البيانات باستخدام LangGraph (نفس البنية من الجزء الثالث) ---
analytics_tools = [query_sales_data, compare_two_values]
analytics_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(analytics_tools)

def call_analytics_model(state: AgentState) -> dict:
    response = analytics_llm.invoke(state["messages"])
    return {"messages": [response]}

def call_analytics_tools(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    tools_by_name = {t.name: t for t in analytics_tools}
    tool_messages = [
        ToolMessage(
            content=str(tools_by_name[tc["name"]].invoke(tc["args"])),
            tool_call_id=tc["id"]
        )
        for tc in last_message.tool_calls
    ]
    return {"messages": tool_messages}

analytics_workflow = StateGraph(AgentState)
analytics_workflow.add_node("agent", call_analytics_model)
analytics_workflow.add_node("tools", call_analytics_tools)
analytics_workflow.set_entry_point("agent")
analytics_workflow.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
analytics_workflow.add_edge("tools", "agent")

analytics_agent = analytics_workflow.compile()

# --- الاستخدام ---
result = analytics_agent.invoke({
    "messages": [HumanMessage(
        content="قارن إيرادات اللابتوب في الرياض مقابل جدة، وأخبرني أيهما أعلى وبكم نسبة"
    )]
})
print(result["messages"][-1].content)
