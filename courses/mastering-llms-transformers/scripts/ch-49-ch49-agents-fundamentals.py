#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — أسس الوكلاء الذكية: ReAct والأدوات
Mastering Transformers and Large Language Models — Agents Fundamentals

Book slug : mastering-llms-transformers
Chapter   : ch-49-ch49-agents-fundamentals (lesson 49)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-49-ch49-agents-fundamentals.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — أسس الوكلاء الذكية: ReAct والأدوات
Mastering Transformers and Large Language Models — Agents Fundamentals

Book slug : mastering-llms-transformers
Chapter   : ch-49-ch49-agents-fundamentals (lesson 49)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-49-ch49-agents-fundamentals.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — أسس الوكلاء الذكية: ReAct والأدوات
Mastering Transformers and Large Language Models — Agents Fundamentals

Book slug : mastering-llms-transformers
Chapter   : ch-49-ch49-agents-fundamentals (lesson 49)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-10-applications-agents/ch49-agents-fundamentals.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# === إعداد البيئة ===
# يفحص هذا القسم الحزم المطلوبة ويحاول تثبيت الناقص منها تلقائيًا.
# شغّل هذه الخلية أولًا، ثم بقية خلايا الدفتر.
import importlib.util
import subprocess
import sys

print("Python:", sys.version.split()[0])

# (اسم وحدة الاستيراد، اسم حزمة pip)
REQUIRED = [
    ("requests", "requests")
]
AUTO_INSTALL = {"PIL", "bs4", "datasets", "faiss", "fasttext", "librosa", "networkx", "opentelemetry", "peft", "prometheus_client", "pydantic", "rank_bm25", "requests", "sentence_transformers", "sentencepiece", "soundfile", "tokenizers", "trafilatura", "transformers", "wandb", "yaml"}


def has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


if not REQUIRED:
    print("لا توجد مكتبات خارجية مطلوبة؛ يكفي Python القياسي.")
else:
    missing = [(module, package) for module, package in REQUIRED if not has_module(module)]
    auto_specs = []
    manual_specs = []
    for module, package in missing:
        specs = package.split()
        if module in AUTO_INSTALL:
            auto_specs.extend(specs)
        else:
            manual_specs.extend(specs)
    auto_specs = list(dict.fromkeys(auto_specs))
    manual_specs = list(dict.fromkeys(manual_specs))

    if auto_specs:
        print("⏳ تثبيت تلقائي:", " ".join(auto_specs))
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *auto_specs])
            print("✅ تم التثبيت.")
        except Exception as exc:
            print("⚠️ تعذّر التثبيت التلقائي:", exc)
            print("شغّل يدويًا: %pip install", " ".join(auto_specs))
    if manual_specs:
        print("⚠️ حزم تحتاج تثبيتًا يدويًا:", " ".join(manual_specs))
        print("شغّل: %pip install", " ".join(manual_specs))
    if not auto_specs and not manual_specs:
        print("✅ كل الحزم المطلوبة متاحة.")

    print("\nالحالة النهائية:")
    for module, package in REQUIRED:
        print(("✅" if has_module(module) else "⚠️"), package)

# ────────────────────────────────────────────────────────────────────────
# [002] content/notebooks/part-10-applications-agents/ch49-agents-fundamentals.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
REACT_SYSTEM_PROMPT = """أنت وكيل ذكي يحل المشكلات عبر التفكير التدريجي واستخدام الأدوات المتاحة.

الأدوات المتاحة:
{tools_description}

اتّبع هذا التنسيق بدقة تامة:
السؤال: السؤال المطلوب حله
تفكير: فكّر خطوة بخطوة فيما تحتاج فعله
فعل: اسم_الأداة[المدخل المطلوب]
ملاحظة: (ستُعطى لك نتيجة الأداة تلقائيًا)
(كرّر تفكير/فعل/ملاحظة بقدر الحاجة)
تفكير: أصبحت لدي المعلومات الكافية الآن
الإجابة النهائية: الإجابة الكاملة

ابدأ الآن:
السؤال: {question}
"""

class Tool:
    def __init__(self, name: str, description: str, function: callable):
        self.name = name
        self.description = description
        self.function = function

    def execute(self, input_str: str) -> str:
        try:
            return str(self.function(input_str))
        except Exception as e:
            return f"خطأ في تنفيذ الأداة: {str(e)}"

class ReActAgent:
    def __init__(self, llm, tools: list, max_iterations=10):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations

    def _build_tools_description(self) -> str:
        return "\n".join([f"- {t.name}: {t.description}" for t in self.tools.values()])

    def run(self, question: str) -> dict:
        tools_desc = self._build_tools_description()
        prompt = REACT_SYSTEM_PROMPT.format(tools_description=tools_desc, question=question)
        full_trace = prompt

        for iteration in range(self.max_iterations):
            response = self.llm.generate(full_trace, stop_sequences=["ملاحظة:"])
            full_trace += response

            if "الإجابة النهائية:" in response:
                final_answer = response.split("الإجابة النهائية:")[-1].strip()
                return {"answer": final_answer, "iterations": iteration + 1, "trace": full_trace}

            if "فعل:" in response:
                action_line = response.split("فعل:")[-1].strip().split("\n")[0]
                tool_name, tool_input = self._parse_action(action_line)

                if tool_name in self.tools:
                    observation = self.tools[tool_name].execute(tool_input)
                else:
                    observation = f"خطأ: الأداة '{tool_name}' غير موجودة"

                full_trace += f"\nملاحظة: {observation}\n"
            else:
                # النموذج لم يُنتج فعلًا أو إجابة نهائية — توقف أو حاول مجددًا
                break

        return {"answer": "تجاوز الوكيل الحد الأقصى للتكرارات دون الوصول لإجابة نهائية",
                "iterations": self.max_iterations, "trace": full_trace}

    def _parse_action(self, action_line: str) -> tuple:
        """تحليل سطر مثل: 'بحث_ويب[ما هي عاصمة فرنسا]'"""
        import re
        match = re.match(r'(\w+)\[(.*)\]', action_line)
        if match:
            return match.group(1), match.group(2)
        return action_line, ""

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-10-applications-agents/ch49-agents-fundamentals.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import math
import requests

def calculator_tool(expression: str) -> str:
    """أداة حاسبة آمنة (تجنّب eval المباشر غير الآمن!)"""
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    try:
        # استخدام compile + eval بسياق محدود صارم كحل تعليمي مبسّط
        code = compile(expression, "<string>", "eval")
        for name in code.co_names:
            if name not in allowed_names:
                raise NameError(f"استخدام غير مسموح: {name}")
        return str(eval(code, {"__builtins__": {}}, allowed_names))
    except Exception as e:
        return f"خطأ في التعبير الرياضي: {str(e)}"

def web_search_tool(query: str, api_key: str = None) -> str:
    """أداة بحث ويب مبسّطة (تمثيل مفاهيمي لاستدعاء API بحث حقيقي)"""
    # في التطبيق الفعلي: استدعاء Tavily/SerpAPI/Bing Search API
    response = requests.get(
        "https://api.search-provider.com/search",
        params={"q": query, "api_key": api_key}
    )
    results = response.json().get("results", [])[:3]
    return "\n".join([f"- {r['title']}: {r['snippet']}" for r in results])

def python_code_executor_tool(code: str) -> str:
    """
    تنفيذ كود بايثون في بيئة معزولة (Sandboxed)
    ⚠️ في الإنتاج: يجب استخدام sandbox حقيقي (Docker/gVisor/E2B) وليس exec مباشر!
    """
    import io, contextlib
    output_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(output_buffer):
            exec(code, {"__builtins__": __builtins__})
        return output_buffer.getvalue() or "تم التنفيذ بنجاح دون مخرجات"
    except Exception as e:
        return f"خطأ في التنفيذ: {str(e)}"

# تجميع الأدوات للوكيل
agent_tools = [
    Tool("حاسبة", "يحسب تعبيرات رياضية. المدخل: تعبير رياضي بصيغة بايثون", calculator_tool),
    Tool("بحث_ويب", "يبحث في الإنترنت عن معلومات حديثة. المدخل: استعلام البحث",
         lambda q: web_search_tool(q, api_key="YOUR_KEY")),
    Tool("تنفيذ_كود", "ينفّذ كود بايثون ويُعيد المخرجات. المدخل: كود بايثون",
         python_code_executor_tool),
]

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-10-applications-agents/ch49-agents-fundamentals.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import json

# تعريف الأدوات بصيغة JSON Schema (المعيار المتبع في OpenAI/Anthropic APIs)
function_schemas = [
    {
        "name": "get_weather",
        "description": "الحصول على حالة الطقس الحالية لمدينة معيّنة",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "اسم المدينة"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["city"]
        }
    }
]

class FunctionCallingAgent:
    """
    محاكاة مفاهيمية لآلية Function Calling الحديثة
    النموذج يُنتج استدعاء دالة بصيغة JSON مُهيكَلة بدلاً من نص حر يحتاج تحليلًا (parsing)
    """
    def __init__(self, llm_client, available_functions: dict, function_schemas: list):
        self.llm_client = llm_client
        self.available_functions = available_functions
        self.function_schemas = function_schemas

    def run(self, user_message: str, max_turns=5) -> str:
        messages = [{"role": "user", "content": user_message}]

        for _ in range(max_turns):
            response = self.llm_client.chat_completion(
                messages=messages,
                functions=self.function_schemas,   # يُرسَل للـ API مباشرة
                function_call="auto"
            )

            if response.get("function_call"):
                func_name = response["function_call"]["name"]
                func_args = json.loads(response["function_call"]["arguments"])

                if func_name in self.available_functions:
                    result = self.available_functions[func_name](**func_args)
                else:
                    result = f"خطأ: دالة غير معروفة {func_name}"

                messages.append({"role": "assistant", "function_call": response["function_call"]})
                messages.append({"role": "function", "name": func_name, "content": str(result)})
            else:
                return response["content"]   # إجابة نهائية، لا حاجة لأداة أخرى

        return "تجاوز الحد الأقصى من جولات استدعاء الأدوات"

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-10-applications-agents/ch49-agents-fundamentals.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-49-ch49-agents-fundamentals.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# === إعداد البيئة ===
# يفحص هذا القسم الحزم المطلوبة ويحاول تثبيت الناقص منها تلقائيًا.
# شغّل هذه الخلية أولًا، ثم بقية خلايا الدفتر.
import importlib.util
import subprocess
import sys

print("Python:", sys.version.split()[0])

# (اسم وحدة الاستيراد، اسم حزمة pip)
REQUIRED = [
    ("requests", "requests")
]
AUTO_INSTALL = {"PIL", "bs4", "datasets", "faiss", "fasttext", "librosa", "networkx", "opentelemetry", "peft", "prometheus_client", "pydantic", "rank_bm25", "requests", "sentence_transformers", "sentencepiece", "soundfile", "tokenizers", "trafilatura", "transformers", "wandb", "yaml"}


def has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


if not REQUIRED:
    print("لا توجد مكتبات خارجية مطلوبة؛ يكفي Python القياسي.")
else:
    missing = [(module, package) for module, package in REQUIRED if not has_module(module)]
    auto_specs = []
    manual_specs = []
    for module, package in missing:
        specs = package.split()
        if module in AUTO_INSTALL:
            auto_specs.extend(specs)
        else:
            manual_specs.extend(specs)
    auto_specs = list(dict.fromkeys(auto_specs))
    manual_specs = list(dict.fromkeys(manual_specs))

    if auto_specs:
        print("⏳ تثبيت تلقائي:", " ".join(auto_specs))
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *auto_specs])
            print("✅ تم التثبيت.")
        except Exception as exc:
            print("⚠️ تعذّر التثبيت التلقائي:", exc)
            print("شغّل يدويًا: %pip install", " ".join(auto_specs))
    if manual_specs:
        print("⚠️ حزم تحتاج تثبيتًا يدويًا:", " ".join(manual_specs))
        print("شغّل: %pip install", " ".join(manual_specs))
    if not auto_specs and not manual_specs:
        print("✅ كل الحزم المطلوبة متاحة.")

    print("\nالحالة النهائية:")
    for module, package in REQUIRED:
        print(("✅" if has_module(module) else "⚠️"), package)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-49-ch49-agents-fundamentals.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
REACT_SYSTEM_PROMPT = """أنت وكيل ذكي يحل المشكلات عبر التفكير التدريجي واستخدام الأدوات المتاحة.

الأدوات المتاحة:
{tools_description}

اتّبع هذا التنسيق بدقة تامة:
السؤال: السؤال المطلوب حله
تفكير: فكّر خطوة بخطوة فيما تحتاج فعله
فعل: اسم_الأداة[المدخل المطلوب]
ملاحظة: (ستُعطى لك نتيجة الأداة تلقائيًا)
(كرّر تفكير/فعل/ملاحظة بقدر الحاجة)
تفكير: أصبحت لدي المعلومات الكافية الآن
الإجابة النهائية: الإجابة الكاملة

ابدأ الآن:
السؤال: {question}
"""

class Tool:
    def __init__(self, name: str, description: str, function: callable):
        self.name = name
        self.description = description
        self.function = function

    def execute(self, input_str: str) -> str:
        try:
            return str(self.function(input_str))
        except Exception as e:
            return f"خطأ في تنفيذ الأداة: {str(e)}"

class ReActAgent:
    def __init__(self, llm, tools: list, max_iterations=10):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations

    def _build_tools_description(self) -> str:
        return "\n".join([f"- {t.name}: {t.description}" for t in self.tools.values()])

    def run(self, question: str) -> dict:
        tools_desc = self._build_tools_description()
        prompt = REACT_SYSTEM_PROMPT.format(tools_description=tools_desc, question=question)
        full_trace = prompt

        for iteration in range(self.max_iterations):
            response = self.llm.generate(full_trace, stop_sequences=["ملاحظة:"])
            full_trace += response

            if "الإجابة النهائية:" in response:
                final_answer = response.split("الإجابة النهائية:")[-1].strip()
                return {"answer": final_answer, "iterations": iteration + 1, "trace": full_trace}

            if "فعل:" in response:
                action_line = response.split("فعل:")[-1].strip().split("\n")[0]
                tool_name, tool_input = self._parse_action(action_line)

                if tool_name in self.tools:
                    observation = self.tools[tool_name].execute(tool_input)
                else:
                    observation = f"خطأ: الأداة '{tool_name}' غير موجودة"

                full_trace += f"\nملاحظة: {observation}\n"
            else:
                # النموذج لم يُنتج فعلًا أو إجابة نهائية — توقف أو حاول مجددًا
                break

        return {"answer": "تجاوز الوكيل الحد الأقصى للتكرارات دون الوصول لإجابة نهائية",
                "iterations": self.max_iterations, "trace": full_trace}

    def _parse_action(self, action_line: str) -> tuple:
        """تحليل سطر مثل: 'بحث_ويب[ما هي عاصمة فرنسا]'"""
        import re
        match = re.match(r'(\w+)\[(.*)\]', action_line)
        if match:
            return match.group(1), match.group(2)
        return action_line, ""

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-49-ch49-agents-fundamentals.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import math
import requests

def calculator_tool(expression: str) -> str:
    """أداة حاسبة آمنة (تجنّب eval المباشر غير الآمن!)"""
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    try:
        # استخدام compile + eval بسياق محدود صارم كحل تعليمي مبسّط
        code = compile(expression, "<string>", "eval")
        for name in code.co_names:
            if name not in allowed_names:
                raise NameError(f"استخدام غير مسموح: {name}")
        return str(eval(code, {"__builtins__": {}}, allowed_names))
    except Exception as e:
        return f"خطأ في التعبير الرياضي: {str(e)}"

def web_search_tool(query: str, api_key: str = None) -> str:
    """أداة بحث ويب مبسّطة (تمثيل مفاهيمي لاستدعاء API بحث حقيقي)"""
    # في التطبيق الفعلي: استدعاء Tavily/SerpAPI/Bing Search API
    response = requests.get(
        "https://api.search-provider.com/search",
        params={"q": query, "api_key": api_key}
    )
    results = response.json().get("results", [])[:3]
    return "\n".join([f"- {r['title']}: {r['snippet']}" for r in results])

def python_code_executor_tool(code: str) -> str:
    """
    تنفيذ كود بايثون في بيئة معزولة (Sandboxed)
    ⚠️ في الإنتاج: يجب استخدام sandbox حقيقي (Docker/gVisor/E2B) وليس exec مباشر!
    """
    import io, contextlib
    output_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(output_buffer):
            exec(code, {"__builtins__": __builtins__})
        return output_buffer.getvalue() or "تم التنفيذ بنجاح دون مخرجات"
    except Exception as e:
        return f"خطأ في التنفيذ: {str(e)}"

# تجميع الأدوات للوكيل
agent_tools = [
    Tool("حاسبة", "يحسب تعبيرات رياضية. المدخل: تعبير رياضي بصيغة بايثون", calculator_tool),
    Tool("بحث_ويب", "يبحث في الإنترنت عن معلومات حديثة. المدخل: استعلام البحث",
         lambda q: web_search_tool(q, api_key="YOUR_KEY")),
    Tool("تنفيذ_كود", "ينفّذ كود بايثون ويُعيد المخرجات. المدخل: كود بايثون",
         python_code_executor_tool),
]

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-49-ch49-agents-fundamentals.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import json

# تعريف الأدوات بصيغة JSON Schema (المعيار المتبع في OpenAI/Anthropic APIs)
function_schemas = [
    {
        "name": "get_weather",
        "description": "الحصول على حالة الطقس الحالية لمدينة معيّنة",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "اسم المدينة"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["city"]
        }
    }
]

class FunctionCallingAgent:
    """
    محاكاة مفاهيمية لآلية Function Calling الحديثة
    النموذج يُنتج استدعاء دالة بصيغة JSON مُهيكَلة بدلاً من نص حر يحتاج تحليلًا (parsing)
    """
    def __init__(self, llm_client, available_functions: dict, function_schemas: list):
        self.llm_client = llm_client
        self.available_functions = available_functions
        self.function_schemas = function_schemas

    def run(self, user_message: str, max_turns=5) -> str:
        messages = [{"role": "user", "content": user_message}]

        for _ in range(max_turns):
            response = self.llm_client.chat_completion(
                messages=messages,
                functions=self.function_schemas,   # يُرسَل للـ API مباشرة
                function_call="auto"
            )

            if response.get("function_call"):
                func_name = response["function_call"]["name"]
                func_args = json.loads(response["function_call"]["arguments"])

                if func_name in self.available_functions:
                    result = self.available_functions[func_name](**func_args)
                else:
                    result = f"خطأ: دالة غير معروفة {func_name}"

                messages.append({"role": "assistant", "function_call": response["function_call"]})
                messages.append({"role": "function", "name": func_name, "content": str(result)})
            else:
                return response["content"]   # إجابة نهائية، لا حاجة لأداة أخرى

        return "تجاوز الحد الأقصى من جولات استدعاء الأدوات"

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-49-ch49-agents-fundamentals.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-49-ch49-agents-fundamentals.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — أسس الوكلاء الذكية: ReAct والأدوات
Mastering Transformers and Large Language Models — Agents Fundamentals

Book slug : mastering-llms-transformers
Chapter   : ch-49-ch49-agents-fundamentals (lesson 49)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-10-applications-agents/ch49-agents-fundamentals.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# === إعداد البيئة ===
# يفحص هذا القسم الحزم المطلوبة ويحاول تثبيت الناقص منها تلقائيًا.
# شغّل هذه الخلية أولًا، ثم بقية خلايا الدفتر.
import importlib.util
import subprocess
import sys

print("Python:", sys.version.split()[0])

# (اسم وحدة الاستيراد، اسم حزمة pip)
REQUIRED = [
    ("requests", "requests")
]
AUTO_INSTALL = {"PIL", "bs4", "datasets", "faiss", "fasttext", "librosa", "networkx", "opentelemetry", "peft", "prometheus_client", "pydantic", "rank_bm25", "requests", "sentence_transformers", "sentencepiece", "soundfile", "tokenizers", "trafilatura", "transformers", "wandb", "yaml"}


def has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


if not REQUIRED:
    print("لا توجد مكتبات خارجية مطلوبة؛ يكفي Python القياسي.")
else:
    missing = [(module, package) for module, package in REQUIRED if not has_module(module)]
    auto_specs = []
    manual_specs = []
    for module, package in missing:
        specs = package.split()
        if module in AUTO_INSTALL:
            auto_specs.extend(specs)
        else:
            manual_specs.extend(specs)
    auto_specs = list(dict.fromkeys(auto_specs))
    manual_specs = list(dict.fromkeys(manual_specs))

    if auto_specs:
        print("⏳ تثبيت تلقائي:", " ".join(auto_specs))
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *auto_specs])
            print("✅ تم التثبيت.")
        except Exception as exc:
            print("⚠️ تعذّر التثبيت التلقائي:", exc)
            print("شغّل يدويًا: %pip install", " ".join(auto_specs))
    if manual_specs:
        print("⚠️ حزم تحتاج تثبيتًا يدويًا:", " ".join(manual_specs))
        print("شغّل: %pip install", " ".join(manual_specs))
    if not auto_specs and not manual_specs:
        print("✅ كل الحزم المطلوبة متاحة.")

    print("\nالحالة النهائية:")
    for module, package in REQUIRED:
        print(("✅" if has_module(module) else "⚠️"), package)

# ────────────────────────────────────────────────────────────────────────
# [002] content/notebooks/part-10-applications-agents/ch49-agents-fundamentals.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
REACT_SYSTEM_PROMPT = """أنت وكيل ذكي يحل المشكلات عبر التفكير التدريجي واستخدام الأدوات المتاحة.

الأدوات المتاحة:
{tools_description}

اتّبع هذا التنسيق بدقة تامة:
السؤال: السؤال المطلوب حله
تفكير: فكّر خطوة بخطوة فيما تحتاج فعله
فعل: اسم_الأداة[المدخل المطلوب]
ملاحظة: (ستُعطى لك نتيجة الأداة تلقائيًا)
(كرّر تفكير/فعل/ملاحظة بقدر الحاجة)
تفكير: أصبحت لدي المعلومات الكافية الآن
الإجابة النهائية: الإجابة الكاملة

ابدأ الآن:
السؤال: {question}
"""

class Tool:
    def __init__(self, name: str, description: str, function: callable):
        self.name = name
        self.description = description
        self.function = function

    def execute(self, input_str: str) -> str:
        try:
            return str(self.function(input_str))
        except Exception as e:
            return f"خطأ في تنفيذ الأداة: {str(e)}"

class ReActAgent:
    def __init__(self, llm, tools: list, max_iterations=10):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations

    def _build_tools_description(self) -> str:
        return "\n".join([f"- {t.name}: {t.description}" for t in self.tools.values()])

    def run(self, question: str) -> dict:
        tools_desc = self._build_tools_description()
        prompt = REACT_SYSTEM_PROMPT.format(tools_description=tools_desc, question=question)
        full_trace = prompt

        for iteration in range(self.max_iterations):
            response = self.llm.generate(full_trace, stop_sequences=["ملاحظة:"])
            full_trace += response

            if "الإجابة النهائية:" in response:
                final_answer = response.split("الإجابة النهائية:")[-1].strip()
                return {"answer": final_answer, "iterations": iteration + 1, "trace": full_trace}

            if "فعل:" in response:
                action_line = response.split("فعل:")[-1].strip().split("\n")[0]
                tool_name, tool_input = self._parse_action(action_line)

                if tool_name in self.tools:
                    observation = self.tools[tool_name].execute(tool_input)
                else:
                    observation = f"خطأ: الأداة '{tool_name}' غير موجودة"

                full_trace += f"\nملاحظة: {observation}\n"
            else:
                # النموذج لم يُنتج فعلًا أو إجابة نهائية — توقف أو حاول مجددًا
                break

        return {"answer": "تجاوز الوكيل الحد الأقصى للتكرارات دون الوصول لإجابة نهائية",
                "iterations": self.max_iterations, "trace": full_trace}

    def _parse_action(self, action_line: str) -> tuple:
        """تحليل سطر مثل: 'بحث_ويب[ما هي عاصمة فرنسا]'"""
        import re
        match = re.match(r'(\w+)\[(.*)\]', action_line)
        if match:
            return match.group(1), match.group(2)
        return action_line, ""

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-10-applications-agents/ch49-agents-fundamentals.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import math
import requests

def calculator_tool(expression: str) -> str:
    """أداة حاسبة آمنة (تجنّب eval المباشر غير الآمن!)"""
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    try:
        # استخدام compile + eval بسياق محدود صارم كحل تعليمي مبسّط
        code = compile(expression, "<string>", "eval")
        for name in code.co_names:
            if name not in allowed_names:
                raise NameError(f"استخدام غير مسموح: {name}")
        return str(eval(code, {"__builtins__": {}}, allowed_names))
    except Exception as e:
        return f"خطأ في التعبير الرياضي: {str(e)}"

def web_search_tool(query: str, api_key: str = None) -> str:
    """أداة بحث ويب مبسّطة (تمثيل مفاهيمي لاستدعاء API بحث حقيقي)"""
    # في التطبيق الفعلي: استدعاء Tavily/SerpAPI/Bing Search API
    response = requests.get(
        "https://api.search-provider.com/search",
        params={"q": query, "api_key": api_key}
    )
    results = response.json().get("results", [])[:3]
    return "\n".join([f"- {r['title']}: {r['snippet']}" for r in results])

def python_code_executor_tool(code: str) -> str:
    """
    تنفيذ كود بايثون في بيئة معزولة (Sandboxed)
    ⚠️ في الإنتاج: يجب استخدام sandbox حقيقي (Docker/gVisor/E2B) وليس exec مباشر!
    """
    import io, contextlib
    output_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(output_buffer):
            exec(code, {"__builtins__": __builtins__})
        return output_buffer.getvalue() or "تم التنفيذ بنجاح دون مخرجات"
    except Exception as e:
        return f"خطأ في التنفيذ: {str(e)}"

# تجميع الأدوات للوكيل
agent_tools = [
    Tool("حاسبة", "يحسب تعبيرات رياضية. المدخل: تعبير رياضي بصيغة بايثون", calculator_tool),
    Tool("بحث_ويب", "يبحث في الإنترنت عن معلومات حديثة. المدخل: استعلام البحث",
         lambda q: web_search_tool(q, api_key="YOUR_KEY")),
    Tool("تنفيذ_كود", "ينفّذ كود بايثون ويُعيد المخرجات. المدخل: كود بايثون",
         python_code_executor_tool),
]

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-10-applications-agents/ch49-agents-fundamentals.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import json

# تعريف الأدوات بصيغة JSON Schema (المعيار المتبع في OpenAI/Anthropic APIs)
function_schemas = [
    {
        "name": "get_weather",
        "description": "الحصول على حالة الطقس الحالية لمدينة معيّنة",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "اسم المدينة"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["city"]
        }
    }
]

class FunctionCallingAgent:
    """
    محاكاة مفاهيمية لآلية Function Calling الحديثة
    النموذج يُنتج استدعاء دالة بصيغة JSON مُهيكَلة بدلاً من نص حر يحتاج تحليلًا (parsing)
    """
    def __init__(self, llm_client, available_functions: dict, function_schemas: list):
        self.llm_client = llm_client
        self.available_functions = available_functions
        self.function_schemas = function_schemas

    def run(self, user_message: str, max_turns=5) -> str:
        messages = [{"role": "user", "content": user_message}]

        for _ in range(max_turns):
            response = self.llm_client.chat_completion(
                messages=messages,
                functions=self.function_schemas,   # يُرسَل للـ API مباشرة
                function_call="auto"
            )

            if response.get("function_call"):
                func_name = response["function_call"]["name"]
                func_args = json.loads(response["function_call"]["arguments"])

                if func_name in self.available_functions:
                    result = self.available_functions[func_name](**func_args)
                else:
                    result = f"خطأ: دالة غير معروفة {func_name}"

                messages.append({"role": "assistant", "function_call": response["function_call"]})
                messages.append({"role": "function", "name": func_name, "content": str(result)})
            else:
                return response["content"]   # إجابة نهائية، لا حاجة لأداة أخرى

        return "تجاوز الحد الأقصى من جولات استدعاء الأدوات"

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-10-applications-agents/ch49-agents-fundamentals.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-49-ch49-agents-fundamentals.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# === إعداد البيئة ===
# يفحص هذا القسم الحزم المطلوبة ويحاول تثبيت الناقص منها تلقائيًا.
# شغّل هذه الخلية أولًا، ثم بقية خلايا الدفتر.
import importlib.util
import subprocess
import sys

print("Python:", sys.version.split()[0])

# (اسم وحدة الاستيراد، اسم حزمة pip)
REQUIRED = [
    ("requests", "requests")
]
AUTO_INSTALL = {"PIL", "bs4", "datasets", "faiss", "fasttext", "librosa", "networkx", "opentelemetry", "peft", "prometheus_client", "pydantic", "rank_bm25", "requests", "sentence_transformers", "sentencepiece", "soundfile", "tokenizers", "trafilatura", "transformers", "wandb", "yaml"}


def has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


if not REQUIRED:
    print("لا توجد مكتبات خارجية مطلوبة؛ يكفي Python القياسي.")
else:
    missing = [(module, package) for module, package in REQUIRED if not has_module(module)]
    auto_specs = []
    manual_specs = []
    for module, package in missing:
        specs = package.split()
        if module in AUTO_INSTALL:
            auto_specs.extend(specs)
        else:
            manual_specs.extend(specs)
    auto_specs = list(dict.fromkeys(auto_specs))
    manual_specs = list(dict.fromkeys(manual_specs))

    if auto_specs:
        print("⏳ تثبيت تلقائي:", " ".join(auto_specs))
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *auto_specs])
            print("✅ تم التثبيت.")
        except Exception as exc:
            print("⚠️ تعذّر التثبيت التلقائي:", exc)
            print("شغّل يدويًا: %pip install", " ".join(auto_specs))
    if manual_specs:
        print("⚠️ حزم تحتاج تثبيتًا يدويًا:", " ".join(manual_specs))
        print("شغّل: %pip install", " ".join(manual_specs))
    if not auto_specs and not manual_specs:
        print("✅ كل الحزم المطلوبة متاحة.")

    print("\nالحالة النهائية:")
    for module, package in REQUIRED:
        print(("✅" if has_module(module) else "⚠️"), package)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-49-ch49-agents-fundamentals.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
REACT_SYSTEM_PROMPT = """أنت وكيل ذكي يحل المشكلات عبر التفكير التدريجي واستخدام الأدوات المتاحة.

الأدوات المتاحة:
{tools_description}

اتّبع هذا التنسيق بدقة تامة:
السؤال: السؤال المطلوب حله
تفكير: فكّر خطوة بخطوة فيما تحتاج فعله
فعل: اسم_الأداة[المدخل المطلوب]
ملاحظة: (ستُعطى لك نتيجة الأداة تلقائيًا)
(كرّر تفكير/فعل/ملاحظة بقدر الحاجة)
تفكير: أصبحت لدي المعلومات الكافية الآن
الإجابة النهائية: الإجابة الكاملة

ابدأ الآن:
السؤال: {question}
"""

class Tool:
    def __init__(self, name: str, description: str, function: callable):
        self.name = name
        self.description = description
        self.function = function

    def execute(self, input_str: str) -> str:
        try:
            return str(self.function(input_str))
        except Exception as e:
            return f"خطأ في تنفيذ الأداة: {str(e)}"

class ReActAgent:
    def __init__(self, llm, tools: list, max_iterations=10):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations

    def _build_tools_description(self) -> str:
        return "\n".join([f"- {t.name}: {t.description}" for t in self.tools.values()])

    def run(self, question: str) -> dict:
        tools_desc = self._build_tools_description()
        prompt = REACT_SYSTEM_PROMPT.format(tools_description=tools_desc, question=question)
        full_trace = prompt

        for iteration in range(self.max_iterations):
            response = self.llm.generate(full_trace, stop_sequences=["ملاحظة:"])
            full_trace += response

            if "الإجابة النهائية:" in response:
                final_answer = response.split("الإجابة النهائية:")[-1].strip()
                return {"answer": final_answer, "iterations": iteration + 1, "trace": full_trace}

            if "فعل:" in response:
                action_line = response.split("فعل:")[-1].strip().split("\n")[0]
                tool_name, tool_input = self._parse_action(action_line)

                if tool_name in self.tools:
                    observation = self.tools[tool_name].execute(tool_input)
                else:
                    observation = f"خطأ: الأداة '{tool_name}' غير موجودة"

                full_trace += f"\nملاحظة: {observation}\n"
            else:
                # النموذج لم يُنتج فعلًا أو إجابة نهائية — توقف أو حاول مجددًا
                break

        return {"answer": "تجاوز الوكيل الحد الأقصى للتكرارات دون الوصول لإجابة نهائية",
                "iterations": self.max_iterations, "trace": full_trace}

    def _parse_action(self, action_line: str) -> tuple:
        """تحليل سطر مثل: 'بحث_ويب[ما هي عاصمة فرنسا]'"""
        import re
        match = re.match(r'(\w+)\[(.*)\]', action_line)
        if match:
            return match.group(1), match.group(2)
        return action_line, ""

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-49-ch49-agents-fundamentals.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import math
import requests

def calculator_tool(expression: str) -> str:
    """أداة حاسبة آمنة (تجنّب eval المباشر غير الآمن!)"""
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    try:
        # استخدام compile + eval بسياق محدود صارم كحل تعليمي مبسّط
        code = compile(expression, "<string>", "eval")
        for name in code.co_names:
            if name not in allowed_names:
                raise NameError(f"استخدام غير مسموح: {name}")
        return str(eval(code, {"__builtins__": {}}, allowed_names))
    except Exception as e:
        return f"خطأ في التعبير الرياضي: {str(e)}"

def web_search_tool(query: str, api_key: str = None) -> str:
    """أداة بحث ويب مبسّطة (تمثيل مفاهيمي لاستدعاء API بحث حقيقي)"""
    # في التطبيق الفعلي: استدعاء Tavily/SerpAPI/Bing Search API
    response = requests.get(
        "https://api.search-provider.com/search",
        params={"q": query, "api_key": api_key}
    )
    results = response.json().get("results", [])[:3]
    return "\n".join([f"- {r['title']}: {r['snippet']}" for r in results])

def python_code_executor_tool(code: str) -> str:
    """
    تنفيذ كود بايثون في بيئة معزولة (Sandboxed)
    ⚠️ في الإنتاج: يجب استخدام sandbox حقيقي (Docker/gVisor/E2B) وليس exec مباشر!
    """
    import io, contextlib
    output_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(output_buffer):
            exec(code, {"__builtins__": __builtins__})
        return output_buffer.getvalue() or "تم التنفيذ بنجاح دون مخرجات"
    except Exception as e:
        return f"خطأ في التنفيذ: {str(e)}"

# تجميع الأدوات للوكيل
agent_tools = [
    Tool("حاسبة", "يحسب تعبيرات رياضية. المدخل: تعبير رياضي بصيغة بايثون", calculator_tool),
    Tool("بحث_ويب", "يبحث في الإنترنت عن معلومات حديثة. المدخل: استعلام البحث",
         lambda q: web_search_tool(q, api_key="YOUR_KEY")),
    Tool("تنفيذ_كود", "ينفّذ كود بايثون ويُعيد المخرجات. المدخل: كود بايثون",
         python_code_executor_tool),
]

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-49-ch49-agents-fundamentals.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import json

# تعريف الأدوات بصيغة JSON Schema (المعيار المتبع في OpenAI/Anthropic APIs)
function_schemas = [
    {
        "name": "get_weather",
        "description": "الحصول على حالة الطقس الحالية لمدينة معيّنة",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "اسم المدينة"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["city"]
        }
    }
]

class FunctionCallingAgent:
    """
    محاكاة مفاهيمية لآلية Function Calling الحديثة
    النموذج يُنتج استدعاء دالة بصيغة JSON مُهيكَلة بدلاً من نص حر يحتاج تحليلًا (parsing)
    """
    def __init__(self, llm_client, available_functions: dict, function_schemas: list):
        self.llm_client = llm_client
        self.available_functions = available_functions
        self.function_schemas = function_schemas

    def run(self, user_message: str, max_turns=5) -> str:
        messages = [{"role": "user", "content": user_message}]

        for _ in range(max_turns):
            response = self.llm_client.chat_completion(
                messages=messages,
                functions=self.function_schemas,   # يُرسَل للـ API مباشرة
                function_call="auto"
            )

            if response.get("function_call"):
                func_name = response["function_call"]["name"]
                func_args = json.loads(response["function_call"]["arguments"])

                if func_name in self.available_functions:
                    result = self.available_functions[func_name](**func_args)
                else:
                    result = f"خطأ: دالة غير معروفة {func_name}"

                messages.append({"role": "assistant", "function_call": response["function_call"]})
                messages.append({"role": "function", "name": func_name, "content": str(result)})
            else:
                return response["content"]   # إجابة نهائية، لا حاجة لأداة أخرى

        return "تجاوز الحد الأقصى من جولات استدعاء الأدوات"

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-49-ch49-agents-fundamentals.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
