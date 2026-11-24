#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — معماريات الوكلاء والأنظمة متعددة الوكلاء
Mastering Transformers and Large Language Models — Multi Agent Systems

Book slug : mastering-llms-transformers
Chapter   : ch-50-ch50-multi-agent-systems (lesson 50)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-50-ch50-multi-agent-systems.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — معماريات الوكلاء والأنظمة متعددة الوكلاء
Mastering Transformers and Large Language Models — Multi Agent Systems

Book slug : mastering-llms-transformers
Chapter   : ch-50-ch50-multi-agent-systems (lesson 50)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-50-ch50-multi-agent-systems.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — معماريات الوكلاء والأنظمة متعددة الوكلاء
Mastering Transformers and Large Language Models — Multi Agent Systems

Book slug : mastering-llms-transformers
Chapter   : ch-50-ch50-multi-agent-systems (lesson 50)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-10-applications-agents/ch50-multi-agent-systems.ipynb#cell-1
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
# [002] content/notebooks/part-10-applications-agents/ch50-multi-agent-systems.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
class PlanAndExecuteAgent:
    """
    نمط أكثر كفاءة للمهام المعقدة:
    1. مخطِّط (Planner) يُنتج خطة كاملة من البداية
    2. مُنفِّذ (Executor) يُنفّذ كل خطوة من الخطة مستخدمًا ReAct/أدوات
    3. عند الفشل أو ظهور معلومة جديدة، يُعاد التخطيط (Re-planning)
    """
    def __init__(self, planner_llm, executor_agent: ReActAgent):
        self.planner_llm = planner_llm
        self.executor_agent = executor_agent

    def create_plan(self, goal: str) -> list:
        planning_prompt = f"""المهمة: {goal}

قسّم هذه المهمة إلى خطوات فرعية واضحة ومرتبة منطقيًا.
أعطِ النتيجة بصيغة قائمة مرقّمة بسيطة، كل خطوة قابلة للتنفيذ المستقل."""
        response = self.planner_llm.generate(planning_prompt)
        steps = [line.strip() for line in response.split("\n") if line.strip() and line[0].isdigit()]
        return steps

    def execute_plan(self, goal: str) -> dict:
        plan = self.create_plan(goal)
        results = []

        for i, step in enumerate(plan):
            context = f"الهدف الكلي: {goal}\nالخطوات المنجزة: {results}\nالخطوة الحالية: {step}"
            step_result = self.executor_agent.run(context)
            results.append({"step": step, "result": step_result["answer"]})

            # فحص: هل نحتاج إعادة تخطيط بناءً على نتيجة غير متوقعة؟
            if "فشل" in step_result["answer"] or "خطأ" in step_result["answer"]:
                replanning_prompt = f"""الخطة الأصلية: {plan}
الخطوة {i+1} فشلت: {step_result['answer']}
اقترح خطة مُعدَّلة للخطوات المتبقية."""
                revised_steps = self.planner_llm.generate(replanning_prompt)
                plan = plan[:i+1] + [s.strip() for s in revised_steps.split("\n") if s.strip()]

        return {"plan": plan, "execution_results": results}

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-10-applications-agents/ch50-multi-agent-systems.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class SpecializedAgent:
    """وكيل متخصص بدور واحد محدّد بوضوح"""
    def __init__(self, role: str, system_prompt: str, llm, tools: list = None):
        self.role = role
        self.system_prompt = system_prompt
        self.llm = llm
        self.tools = tools or []

    def process(self, task: str, context: dict = None) -> str:
        full_prompt = f"{self.system_prompt}\n\nالسياق: {context or ''}\nالمهمة: {task}"
        return self.llm.generate(full_prompt)

class MultiAgentOrchestrator:
    """
    منسّق يدير التعاون بين عدة وكلاء متخصصين
    نمط شائع: Manager-Worker (مدير يُوزّع المهام، عمّال متخصصون ينفّذون)
    """
    def __init__(self, manager_llm):
        self.manager_llm = manager_llm
        self.agents = {}

    def register_agent(self, agent: SpecializedAgent):
        self.agents[agent.role] = agent

    def delegate_task(self, goal: str) -> dict:
        # المدير يُقرّر أي وكيل متخصص يجب تكليفه بأي جزء من المهمة
        roles_description = "\n".join([f"- {role}: {a.system_prompt[:100]}"
                                         for role, a in self.agents.items()])
        delegation_prompt = f"""المهمة الكلية: {goal}
الوكلاء المتاحون:
{roles_description}

حدّد تسلسل الوكلاء المطلوبين ومهمة كل واحد، بصيغة JSON:
[{{"role": "...", "subtask": "..."}}]"""

        delegation_plan = json.loads(self.manager_llm.generate(delegation_prompt))

        results = {}
        accumulated_context = {}
        for step in delegation_plan:
            role, subtask = step["role"], step["subtask"]
            if role in self.agents:
                result = self.agents[role].process(subtask, context=accumulated_context)
                results[role] = result
                accumulated_context[role] = result

        synthesis_prompt = f"""المهمة الأصلية: {goal}
نتائج الوكلاء المتخصصين: {results}

ألّف إجابة نهائية متكاملة ومتماسكة."""
        final_answer = self.manager_llm.generate(synthesis_prompt)
        return {"final_answer": final_answer, "agent_contributions": results}

# مثال تطبيقي: فريق كتابة تقرير بحثي
orchestrator = MultiAgentOrchestrator(manager_llm=some_llm)
orchestrator.register_agent(SpecializedAgent(
    role="باحث", system_prompt="أنت باحث متخصص في جمع المعلومات الدقيقة والموثوقة.",
    llm=some_llm, tools=[web_search_tool]
))
orchestrator.register_agent(SpecializedAgent(
    role="محلل_بيانات", system_prompt="أنت محلل بيانات متخصص في استخراج الأنماط والإحصائيات.",
    llm=some_llm, tools=[python_code_executor_tool]
))
orchestrator.register_agent(SpecializedAgent(
    role="كاتب", system_prompt="أنت كاتب محترف يُنتج نصوصًا واضحة ومنظَّمة بأسلوب احترافي.",
    llm=some_llm
))
orchestrator.register_agent(SpecializedAgent(
    role="مراجع", system_prompt="أنت مراجع دقيق يتحقق من الأخطاء المنطقية والواقعية واللغوية.",
    llm=some_llm
))

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-10-applications-agents/ch50-multi-agent-systems.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def multi_agent_debate(question: str, agents: list, rounds=3) -> str:
    """
    نمط النقاش: عدة وكلاء (قد تكون نفس LLM بـ temperature مختلفة)
    يتناقشون إجاباتهم لتحسين الدقة النهائية عبر التصحيح المتبادل
    """
    responses = [agent.process(question) for agent in agents]

    for round_num in range(rounds):
        new_responses = []
        for i, agent in enumerate(agents):
            other_responses = [r for j, r in enumerate(responses) if j != i]
            debate_prompt = f"""السؤال: {question}
إجابتك السابقة: {responses[i]}
إجابات الوكلاء الآخرين: {other_responses}

راجع إجابتك في ضوء وجهات النظر الأخرى. إذا وجدت خطأً في تفكيرك، صحّحه.
أعطِ إجابتك المُحدَّثة:"""
            new_responses.append(agent.process(debate_prompt))
        responses = new_responses

    # الإجماع النهائي: التصويت بالأغلبية أو تجميع عبر وكيل حَكَم
    return responses

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-10-applications-agents/ch50-multi-agent-systems.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-50-ch50-multi-agent-systems.ipynb#cell-1
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-50-ch50-multi-agent-systems.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
class PlanAndExecuteAgent:
    """
    نمط أكثر كفاءة للمهام المعقدة:
    1. مخطِّط (Planner) يُنتج خطة كاملة من البداية
    2. مُنفِّذ (Executor) يُنفّذ كل خطوة من الخطة مستخدمًا ReAct/أدوات
    3. عند الفشل أو ظهور معلومة جديدة، يُعاد التخطيط (Re-planning)
    """
    def __init__(self, planner_llm, executor_agent: ReActAgent):
        self.planner_llm = planner_llm
        self.executor_agent = executor_agent

    def create_plan(self, goal: str) -> list:
        planning_prompt = f"""المهمة: {goal}

قسّم هذه المهمة إلى خطوات فرعية واضحة ومرتبة منطقيًا.
أعطِ النتيجة بصيغة قائمة مرقّمة بسيطة، كل خطوة قابلة للتنفيذ المستقل."""
        response = self.planner_llm.generate(planning_prompt)
        steps = [line.strip() for line in response.split("\n") if line.strip() and line[0].isdigit()]
        return steps

    def execute_plan(self, goal: str) -> dict:
        plan = self.create_plan(goal)
        results = []

        for i, step in enumerate(plan):
            context = f"الهدف الكلي: {goal}\nالخطوات المنجزة: {results}\nالخطوة الحالية: {step}"
            step_result = self.executor_agent.run(context)
            results.append({"step": step, "result": step_result["answer"]})

            # فحص: هل نحتاج إعادة تخطيط بناءً على نتيجة غير متوقعة؟
            if "فشل" in step_result["answer"] or "خطأ" in step_result["answer"]:
                replanning_prompt = f"""الخطة الأصلية: {plan}
الخطوة {i+1} فشلت: {step_result['answer']}
اقترح خطة مُعدَّلة للخطوات المتبقية."""
                revised_steps = self.planner_llm.generate(replanning_prompt)
                plan = plan[:i+1] + [s.strip() for s in revised_steps.split("\n") if s.strip()]

        return {"plan": plan, "execution_results": results}

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-50-ch50-multi-agent-systems.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class SpecializedAgent:
    """وكيل متخصص بدور واحد محدّد بوضوح"""
    def __init__(self, role: str, system_prompt: str, llm, tools: list = None):
        self.role = role
        self.system_prompt = system_prompt
        self.llm = llm
        self.tools = tools or []

    def process(self, task: str, context: dict = None) -> str:
        full_prompt = f"{self.system_prompt}\n\nالسياق: {context or ''}\nالمهمة: {task}"
        return self.llm.generate(full_prompt)

class MultiAgentOrchestrator:
    """
    منسّق يدير التعاون بين عدة وكلاء متخصصين
    نمط شائع: Manager-Worker (مدير يُوزّع المهام، عمّال متخصصون ينفّذون)
    """
    def __init__(self, manager_llm):
        self.manager_llm = manager_llm
        self.agents = {}

    def register_agent(self, agent: SpecializedAgent):
        self.agents[agent.role] = agent

    def delegate_task(self, goal: str) -> dict:
        # المدير يُقرّر أي وكيل متخصص يجب تكليفه بأي جزء من المهمة
        roles_description = "\n".join([f"- {role}: {a.system_prompt[:100]}"
                                         for role, a in self.agents.items()])
        delegation_prompt = f"""المهمة الكلية: {goal}
الوكلاء المتاحون:
{roles_description}

حدّد تسلسل الوكلاء المطلوبين ومهمة كل واحد، بصيغة JSON:
[{{"role": "...", "subtask": "..."}}]"""

        delegation_plan = json.loads(self.manager_llm.generate(delegation_prompt))

        results = {}
        accumulated_context = {}
        for step in delegation_plan:
            role, subtask = step["role"], step["subtask"]
            if role in self.agents:
                result = self.agents[role].process(subtask, context=accumulated_context)
                results[role] = result
                accumulated_context[role] = result

        synthesis_prompt = f"""المهمة الأصلية: {goal}
نتائج الوكلاء المتخصصين: {results}

ألّف إجابة نهائية متكاملة ومتماسكة."""
        final_answer = self.manager_llm.generate(synthesis_prompt)
        return {"final_answer": final_answer, "agent_contributions": results}

# مثال تطبيقي: فريق كتابة تقرير بحثي
orchestrator = MultiAgentOrchestrator(manager_llm=some_llm)
orchestrator.register_agent(SpecializedAgent(
    role="باحث", system_prompt="أنت باحث متخصص في جمع المعلومات الدقيقة والموثوقة.",
    llm=some_llm, tools=[web_search_tool]
))
orchestrator.register_agent(SpecializedAgent(
    role="محلل_بيانات", system_prompt="أنت محلل بيانات متخصص في استخراج الأنماط والإحصائيات.",
    llm=some_llm, tools=[python_code_executor_tool]
))
orchestrator.register_agent(SpecializedAgent(
    role="كاتب", system_prompt="أنت كاتب محترف يُنتج نصوصًا واضحة ومنظَّمة بأسلوب احترافي.",
    llm=some_llm
))
orchestrator.register_agent(SpecializedAgent(
    role="مراجع", system_prompt="أنت مراجع دقيق يتحقق من الأخطاء المنطقية والواقعية واللغوية.",
    llm=some_llm
))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-50-ch50-multi-agent-systems.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def multi_agent_debate(question: str, agents: list, rounds=3) -> str:
    """
    نمط النقاش: عدة وكلاء (قد تكون نفس LLM بـ temperature مختلفة)
    يتناقشون إجاباتهم لتحسين الدقة النهائية عبر التصحيح المتبادل
    """
    responses = [agent.process(question) for agent in agents]

    for round_num in range(rounds):
        new_responses = []
        for i, agent in enumerate(agents):
            other_responses = [r for j, r in enumerate(responses) if j != i]
            debate_prompt = f"""السؤال: {question}
إجابتك السابقة: {responses[i]}
إجابات الوكلاء الآخرين: {other_responses}

راجع إجابتك في ضوء وجهات النظر الأخرى. إذا وجدت خطأً في تفكيرك، صحّحه.
أعطِ إجابتك المُحدَّثة:"""
            new_responses.append(agent.process(debate_prompt))
        responses = new_responses

    # الإجماع النهائي: التصويت بالأغلبية أو تجميع عبر وكيل حَكَم
    return responses

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-50-ch50-multi-agent-systems.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-50-ch50-multi-agent-systems.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — معماريات الوكلاء والأنظمة متعددة الوكلاء
Mastering Transformers and Large Language Models — Multi Agent Systems

Book slug : mastering-llms-transformers
Chapter   : ch-50-ch50-multi-agent-systems (lesson 50)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-10-applications-agents/ch50-multi-agent-systems.ipynb#cell-1
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
# [002] content/notebooks/part-10-applications-agents/ch50-multi-agent-systems.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
class PlanAndExecuteAgent:
    """
    نمط أكثر كفاءة للمهام المعقدة:
    1. مخطِّط (Planner) يُنتج خطة كاملة من البداية
    2. مُنفِّذ (Executor) يُنفّذ كل خطوة من الخطة مستخدمًا ReAct/أدوات
    3. عند الفشل أو ظهور معلومة جديدة، يُعاد التخطيط (Re-planning)
    """
    def __init__(self, planner_llm, executor_agent: ReActAgent):
        self.planner_llm = planner_llm
        self.executor_agent = executor_agent

    def create_plan(self, goal: str) -> list:
        planning_prompt = f"""المهمة: {goal}

قسّم هذه المهمة إلى خطوات فرعية واضحة ومرتبة منطقيًا.
أعطِ النتيجة بصيغة قائمة مرقّمة بسيطة، كل خطوة قابلة للتنفيذ المستقل."""
        response = self.planner_llm.generate(planning_prompt)
        steps = [line.strip() for line in response.split("\n") if line.strip() and line[0].isdigit()]
        return steps

    def execute_plan(self, goal: str) -> dict:
        plan = self.create_plan(goal)
        results = []

        for i, step in enumerate(plan):
            context = f"الهدف الكلي: {goal}\nالخطوات المنجزة: {results}\nالخطوة الحالية: {step}"
            step_result = self.executor_agent.run(context)
            results.append({"step": step, "result": step_result["answer"]})

            # فحص: هل نحتاج إعادة تخطيط بناءً على نتيجة غير متوقعة؟
            if "فشل" in step_result["answer"] or "خطأ" in step_result["answer"]:
                replanning_prompt = f"""الخطة الأصلية: {plan}
الخطوة {i+1} فشلت: {step_result['answer']}
اقترح خطة مُعدَّلة للخطوات المتبقية."""
                revised_steps = self.planner_llm.generate(replanning_prompt)
                plan = plan[:i+1] + [s.strip() for s in revised_steps.split("\n") if s.strip()]

        return {"plan": plan, "execution_results": results}

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-10-applications-agents/ch50-multi-agent-systems.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class SpecializedAgent:
    """وكيل متخصص بدور واحد محدّد بوضوح"""
    def __init__(self, role: str, system_prompt: str, llm, tools: list = None):
        self.role = role
        self.system_prompt = system_prompt
        self.llm = llm
        self.tools = tools or []

    def process(self, task: str, context: dict = None) -> str:
        full_prompt = f"{self.system_prompt}\n\nالسياق: {context or ''}\nالمهمة: {task}"
        return self.llm.generate(full_prompt)

class MultiAgentOrchestrator:
    """
    منسّق يدير التعاون بين عدة وكلاء متخصصين
    نمط شائع: Manager-Worker (مدير يُوزّع المهام، عمّال متخصصون ينفّذون)
    """
    def __init__(self, manager_llm):
        self.manager_llm = manager_llm
        self.agents = {}

    def register_agent(self, agent: SpecializedAgent):
        self.agents[agent.role] = agent

    def delegate_task(self, goal: str) -> dict:
        # المدير يُقرّر أي وكيل متخصص يجب تكليفه بأي جزء من المهمة
        roles_description = "\n".join([f"- {role}: {a.system_prompt[:100]}"
                                         for role, a in self.agents.items()])
        delegation_prompt = f"""المهمة الكلية: {goal}
الوكلاء المتاحون:
{roles_description}

حدّد تسلسل الوكلاء المطلوبين ومهمة كل واحد، بصيغة JSON:
[{{"role": "...", "subtask": "..."}}]"""

        delegation_plan = json.loads(self.manager_llm.generate(delegation_prompt))

        results = {}
        accumulated_context = {}
        for step in delegation_plan:
            role, subtask = step["role"], step["subtask"]
            if role in self.agents:
                result = self.agents[role].process(subtask, context=accumulated_context)
                results[role] = result
                accumulated_context[role] = result

        synthesis_prompt = f"""المهمة الأصلية: {goal}
نتائج الوكلاء المتخصصين: {results}

ألّف إجابة نهائية متكاملة ومتماسكة."""
        final_answer = self.manager_llm.generate(synthesis_prompt)
        return {"final_answer": final_answer, "agent_contributions": results}

# مثال تطبيقي: فريق كتابة تقرير بحثي
orchestrator = MultiAgentOrchestrator(manager_llm=some_llm)
orchestrator.register_agent(SpecializedAgent(
    role="باحث", system_prompt="أنت باحث متخصص في جمع المعلومات الدقيقة والموثوقة.",
    llm=some_llm, tools=[web_search_tool]
))
orchestrator.register_agent(SpecializedAgent(
    role="محلل_بيانات", system_prompt="أنت محلل بيانات متخصص في استخراج الأنماط والإحصائيات.",
    llm=some_llm, tools=[python_code_executor_tool]
))
orchestrator.register_agent(SpecializedAgent(
    role="كاتب", system_prompt="أنت كاتب محترف يُنتج نصوصًا واضحة ومنظَّمة بأسلوب احترافي.",
    llm=some_llm
))
orchestrator.register_agent(SpecializedAgent(
    role="مراجع", system_prompt="أنت مراجع دقيق يتحقق من الأخطاء المنطقية والواقعية واللغوية.",
    llm=some_llm
))

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-10-applications-agents/ch50-multi-agent-systems.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def multi_agent_debate(question: str, agents: list, rounds=3) -> str:
    """
    نمط النقاش: عدة وكلاء (قد تكون نفس LLM بـ temperature مختلفة)
    يتناقشون إجاباتهم لتحسين الدقة النهائية عبر التصحيح المتبادل
    """
    responses = [agent.process(question) for agent in agents]

    for round_num in range(rounds):
        new_responses = []
        for i, agent in enumerate(agents):
            other_responses = [r for j, r in enumerate(responses) if j != i]
            debate_prompt = f"""السؤال: {question}
إجابتك السابقة: {responses[i]}
إجابات الوكلاء الآخرين: {other_responses}

راجع إجابتك في ضوء وجهات النظر الأخرى. إذا وجدت خطأً في تفكيرك، صحّحه.
أعطِ إجابتك المُحدَّثة:"""
            new_responses.append(agent.process(debate_prompt))
        responses = new_responses

    # الإجماع النهائي: التصويت بالأغلبية أو تجميع عبر وكيل حَكَم
    return responses

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-10-applications-agents/ch50-multi-agent-systems.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-50-ch50-multi-agent-systems.ipynb#cell-2
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-50-ch50-multi-agent-systems.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class PlanAndExecuteAgent:
    """
    نمط أكثر كفاءة للمهام المعقدة:
    1. مخطِّط (Planner) يُنتج خطة كاملة من البداية
    2. مُنفِّذ (Executor) يُنفّذ كل خطوة من الخطة مستخدمًا ReAct/أدوات
    3. عند الفشل أو ظهور معلومة جديدة، يُعاد التخطيط (Re-planning)
    """
    def __init__(self, planner_llm, executor_agent: ReActAgent):
        self.planner_llm = planner_llm
        self.executor_agent = executor_agent

    def create_plan(self, goal: str) -> list:
        planning_prompt = f"""المهمة: {goal}

قسّم هذه المهمة إلى خطوات فرعية واضحة ومرتبة منطقيًا.
أعطِ النتيجة بصيغة قائمة مرقّمة بسيطة، كل خطوة قابلة للتنفيذ المستقل."""
        response = self.planner_llm.generate(planning_prompt)
        steps = [line.strip() for line in response.split("\n") if line.strip() and line[0].isdigit()]
        return steps

    def execute_plan(self, goal: str) -> dict:
        plan = self.create_plan(goal)
        results = []

        for i, step in enumerate(plan):
            context = f"الهدف الكلي: {goal}\nالخطوات المنجزة: {results}\nالخطوة الحالية: {step}"
            step_result = self.executor_agent.run(context)
            results.append({"step": step, "result": step_result["answer"]})

            # فحص: هل نحتاج إعادة تخطيط بناءً على نتيجة غير متوقعة؟
            if "فشل" in step_result["answer"] or "خطأ" in step_result["answer"]:
                replanning_prompt = f"""الخطة الأصلية: {plan}
الخطوة {i+1} فشلت: {step_result['answer']}
اقترح خطة مُعدَّلة للخطوات المتبقية."""
                revised_steps = self.planner_llm.generate(replanning_prompt)
                plan = plan[:i+1] + [s.strip() for s in revised_steps.split("\n") if s.strip()]

        return {"plan": plan, "execution_results": results}

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-50-ch50-multi-agent-systems.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class SpecializedAgent:
    """وكيل متخصص بدور واحد محدّد بوضوح"""
    def __init__(self, role: str, system_prompt: str, llm, tools: list = None):
        self.role = role
        self.system_prompt = system_prompt
        self.llm = llm
        self.tools = tools or []

    def process(self, task: str, context: dict = None) -> str:
        full_prompt = f"{self.system_prompt}\n\nالسياق: {context or ''}\nالمهمة: {task}"
        return self.llm.generate(full_prompt)

class MultiAgentOrchestrator:
    """
    منسّق يدير التعاون بين عدة وكلاء متخصصين
    نمط شائع: Manager-Worker (مدير يُوزّع المهام، عمّال متخصصون ينفّذون)
    """
    def __init__(self, manager_llm):
        self.manager_llm = manager_llm
        self.agents = {}

    def register_agent(self, agent: SpecializedAgent):
        self.agents[agent.role] = agent

    def delegate_task(self, goal: str) -> dict:
        # المدير يُقرّر أي وكيل متخصص يجب تكليفه بأي جزء من المهمة
        roles_description = "\n".join([f"- {role}: {a.system_prompt[:100]}"
                                         for role, a in self.agents.items()])
        delegation_prompt = f"""المهمة الكلية: {goal}
الوكلاء المتاحون:
{roles_description}

حدّد تسلسل الوكلاء المطلوبين ومهمة كل واحد، بصيغة JSON:
[{{"role": "...", "subtask": "..."}}]"""

        delegation_plan = json.loads(self.manager_llm.generate(delegation_prompt))

        results = {}
        accumulated_context = {}
        for step in delegation_plan:
            role, subtask = step["role"], step["subtask"]
            if role in self.agents:
                result = self.agents[role].process(subtask, context=accumulated_context)
                results[role] = result
                accumulated_context[role] = result

        synthesis_prompt = f"""المهمة الأصلية: {goal}
نتائج الوكلاء المتخصصين: {results}

ألّف إجابة نهائية متكاملة ومتماسكة."""
        final_answer = self.manager_llm.generate(synthesis_prompt)
        return {"final_answer": final_answer, "agent_contributions": results}

# مثال تطبيقي: فريق كتابة تقرير بحثي
orchestrator = MultiAgentOrchestrator(manager_llm=some_llm)
orchestrator.register_agent(SpecializedAgent(
    role="باحث", system_prompt="أنت باحث متخصص في جمع المعلومات الدقيقة والموثوقة.",
    llm=some_llm, tools=[web_search_tool]
))
orchestrator.register_agent(SpecializedAgent(
    role="محلل_بيانات", system_prompt="أنت محلل بيانات متخصص في استخراج الأنماط والإحصائيات.",
    llm=some_llm, tools=[python_code_executor_tool]
))
orchestrator.register_agent(SpecializedAgent(
    role="كاتب", system_prompt="أنت كاتب محترف يُنتج نصوصًا واضحة ومنظَّمة بأسلوب احترافي.",
    llm=some_llm
))
orchestrator.register_agent(SpecializedAgent(
    role="مراجع", system_prompt="أنت مراجع دقيق يتحقق من الأخطاء المنطقية والواقعية واللغوية.",
    llm=some_llm
))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-50-ch50-multi-agent-systems.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def multi_agent_debate(question: str, agents: list, rounds=3) -> str:
    """
    نمط النقاش: عدة وكلاء (قد تكون نفس LLM بـ temperature مختلفة)
    يتناقشون إجاباتهم لتحسين الدقة النهائية عبر التصحيح المتبادل
    """
    responses = [agent.process(question) for agent in agents]

    for round_num in range(rounds):
        new_responses = []
        for i, agent in enumerate(agents):
            other_responses = [r for j, r in enumerate(responses) if j != i]
            debate_prompt = f"""السؤال: {question}
إجابتك السابقة: {responses[i]}
إجابات الوكلاء الآخرين: {other_responses}

راجع إجابتك في ضوء وجهات النظر الأخرى. إذا وجدت خطأً في تفكيرك، صحّحه.
أعطِ إجابتك المُحدَّثة:"""
            new_responses.append(agent.process(debate_prompt))
        responses = new_responses

    # الإجماع النهائي: التصويت بالأغلبية أو تجميع عبر وكيل حَكَم
    return responses

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-50-ch50-multi-agent-systems.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
