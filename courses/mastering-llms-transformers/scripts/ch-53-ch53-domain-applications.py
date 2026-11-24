#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — تطبيقات متخصصة حسب المجال
Mastering Transformers and Large Language Models — Domain Applications

Book slug : mastering-llms-transformers
Chapter   : ch-53-ch53-domain-applications (lesson 53)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-53-ch53-domain-applications.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — تطبيقات متخصصة حسب المجال
Mastering Transformers and Large Language Models — Domain Applications

Book slug : mastering-llms-transformers
Chapter   : ch-53-ch53-domain-applications (lesson 53)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-53-ch53-domain-applications.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — تطبيقات متخصصة حسب المجال
Mastering Transformers and Large Language Models — Domain Applications

Book slug : mastering-llms-transformers
Chapter   : ch-53-ch53-domain-applications (lesson 53)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-10-applications-agents/ch53-domain-applications.ipynb#cell-1
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
# [002] content/notebooks/part-10-applications-agents/ch53-domain-applications.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
class CodeAssistantAgent:
    """
    معمارية مرجعية لمساعد برمجي: فهم السياق (المشروع الكامل) + تنفيذ + تحقق
    """
    def __init__(self, llm, codebase_rag: SimpleRAGPipeline, code_executor):
        self.llm = llm
        self.codebase_rag = codebase_rag   # فهرسة الكود الموجود كمستندات RAG
        self.code_executor = code_executor

    def generate_code_with_context(self, instruction: str, current_file: str) -> dict:
        # 1. استرجاع أجزاء ذات صلة من بقية المشروع (دوال مشابهة، تعريفات أنواع)
        relevant_code = self.codebase_rag.retrieve(instruction, top_k=5)

        prompt = f"""أنت مساعد برمجي خبير. اتّبع نمط الكود الموجود في المشروع بدقة.

أجزاء ذات صلة من المشروع:
{[c['text'] for c in relevant_code]}

الملف الحالي:
{current_file}

المطلوب: {instruction}

اكتب الكود المطلوب فقط، متّسقًا مع أسلوب المشروع:"""
        generated_code = self.llm.generate(prompt)

        # 2. تحقق تلقائي: تشغيل الكود واختباره قبل تقديمه
        test_result = self.code_executor.run_with_tests(generated_code)
        if not test_result["passed"]:
            fix_prompt = f"""الكود التالي فشل في الاختبار:
{generated_code}
خطأ: {test_result['error']}
صحّح الكود:"""
            generated_code = self.llm.generate(fix_prompt)

        return {"code": generated_code, "verified": test_result["passed"]}

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-10-applications-agents/ch53-domain-applications.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class CustomerSupportAgent:
    """
    معمارية شاملة: RAG على قاعدة المعرفة + أدوات للتفاعل مع أنظمة حقيقية
    (تتبع الطلبات، إلخ) + تصعيد ذكي لموظف بشري عند الحاجة
    """
    def __init__(self, llm, knowledge_base_rag, order_system_tool, escalation_threshold=0.6):
        self.llm = llm
        self.kb_rag = knowledge_base_rag
        self.order_tool = order_system_tool
        self.escalation_threshold = escalation_threshold

    def handle_query(self, user_query: str, user_id: str) -> dict:
        # 1. تقييم أولي: هل هذا سؤال معرفي عام، أم يتطلب بيانات خاصة بالمستخدم؟
        intent_prompt = f"""صنّف نوع الاستفسار التالي:
الاستفسار: {user_query}
الأنواع الممكنة: سؤال_عام، استعلام_طلب، شكوى، طلب_استرجاع
أعطِ التصنيف فقط."""
        intent = self.llm.generate(intent_prompt).strip()

        if intent == "استعلام_طلب":
            order_info = self.order_tool.get_order_status(user_id)
            response = self.llm.generate(f"معلومات الطلب: {order_info}\nصُغ ردًا ودودًا: ")
        else:
            retrieved = self.kb_rag.retrieve_with_reranking(user_query, final_k=3)
            response = self.llm.generate(
                self.kb_rag.build_augmented_prompt(user_query, retrieved)
            )

        # 2. تقييم الثقة وقرار التصعيد
        confidence_prompt = f"""السؤال: {user_query}\nالإجابة المُقترَحة: {response}
ما مدى ثقتك بأن هذه الإجابة صحيحة وكافية؟ أعطِ رقمًا من 0 إلى 1 فقط."""
        try:
            confidence = float(self.llm.generate(confidence_prompt).strip())
        except ValueError:
            confidence = 0.5

        if confidence < self.escalation_threshold or intent == "شكوى":
            return {
                "response": response, "escalated": True,
                "escalation_reason": "ثقة منخفضة" if confidence < self.escalation_threshold else "شكوى تتطلب تدخلًا بشريًا"
            }
        return {"response": response, "escalated": False, "confidence": confidence}

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-10-applications-agents/ch53-domain-applications.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class GenericCopilotPattern:
    """
    النمط المعماري العام الذي تتشارك فيه كل "مساعدات Copilot" الناجحة:
    1. فهم السياق المحلي (الملف الحالي/المستند/الشاشة)
    2. فهم السياق العام (قاعدة المعرفة/المشروع الكامل عبر RAG)
    3. اقتراح فعل محدد وصغير (وليس تغييرًا جذريًا دفعة واحدة)
    4. تحقق تلقائي قبل العرض على المستخدم
    5. تغذية راجعة فورية (قبول/رفض المستخدم) تُحسِّن النظام لاحقًا
    """
    def __init__(self, llm, local_context_fn, global_rag, verifier_fn):
        self.llm = llm
        self.local_context_fn = local_context_fn      # يستخرج السياق المحلي الحالي
        self.global_rag = global_rag                    # معرفة عامة أوسع
        self.verifier_fn = verifier_fn                  # دالة تحقق مخصصة بالمجال
        self.feedback_log = []

    def suggest(self, trigger_context: dict) -> dict:
        local_ctx = self.local_context_fn(trigger_context)
        global_ctx = self.global_rag.retrieve(str(local_ctx), top_k=3)

        suggestion = self.llm.generate(f"""السياق المحلي: {local_ctx}
السياق العام ذو الصلة: {global_ctx}
اقترح الفعل التالي الأنسب والأكثر دقة:""")

        verified, verification_info = self.verifier_fn(suggestion)
        return {"suggestion": suggestion, "verified": verified, "details": verification_info}

    def log_feedback(self, suggestion_id: str, accepted: bool):
        self.feedback_log.append({"id": suggestion_id, "accepted": accepted})
        # في نظام ناضج: تُستخدم هذه البيانات لاحقًا لـ fine-tuning دوري (الفصل 23/30)

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-10-applications-agents/ch53-domain-applications.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-53-ch53-domain-applications.ipynb#cell-1
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-53-ch53-domain-applications.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
class CodeAssistantAgent:
    """
    معمارية مرجعية لمساعد برمجي: فهم السياق (المشروع الكامل) + تنفيذ + تحقق
    """
    def __init__(self, llm, codebase_rag: SimpleRAGPipeline, code_executor):
        self.llm = llm
        self.codebase_rag = codebase_rag   # فهرسة الكود الموجود كمستندات RAG
        self.code_executor = code_executor

    def generate_code_with_context(self, instruction: str, current_file: str) -> dict:
        # 1. استرجاع أجزاء ذات صلة من بقية المشروع (دوال مشابهة، تعريفات أنواع)
        relevant_code = self.codebase_rag.retrieve(instruction, top_k=5)

        prompt = f"""أنت مساعد برمجي خبير. اتّبع نمط الكود الموجود في المشروع بدقة.

أجزاء ذات صلة من المشروع:
{[c['text'] for c in relevant_code]}

الملف الحالي:
{current_file}

المطلوب: {instruction}

اكتب الكود المطلوب فقط، متّسقًا مع أسلوب المشروع:"""
        generated_code = self.llm.generate(prompt)

        # 2. تحقق تلقائي: تشغيل الكود واختباره قبل تقديمه
        test_result = self.code_executor.run_with_tests(generated_code)
        if not test_result["passed"]:
            fix_prompt = f"""الكود التالي فشل في الاختبار:
{generated_code}
خطأ: {test_result['error']}
صحّح الكود:"""
            generated_code = self.llm.generate(fix_prompt)

        return {"code": generated_code, "verified": test_result["passed"]}

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-53-ch53-domain-applications.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class CustomerSupportAgent:
    """
    معمارية شاملة: RAG على قاعدة المعرفة + أدوات للتفاعل مع أنظمة حقيقية
    (تتبع الطلبات، إلخ) + تصعيد ذكي لموظف بشري عند الحاجة
    """
    def __init__(self, llm, knowledge_base_rag, order_system_tool, escalation_threshold=0.6):
        self.llm = llm
        self.kb_rag = knowledge_base_rag
        self.order_tool = order_system_tool
        self.escalation_threshold = escalation_threshold

    def handle_query(self, user_query: str, user_id: str) -> dict:
        # 1. تقييم أولي: هل هذا سؤال معرفي عام، أم يتطلب بيانات خاصة بالمستخدم؟
        intent_prompt = f"""صنّف نوع الاستفسار التالي:
الاستفسار: {user_query}
الأنواع الممكنة: سؤال_عام، استعلام_طلب، شكوى، طلب_استرجاع
أعطِ التصنيف فقط."""
        intent = self.llm.generate(intent_prompt).strip()

        if intent == "استعلام_طلب":
            order_info = self.order_tool.get_order_status(user_id)
            response = self.llm.generate(f"معلومات الطلب: {order_info}\nصُغ ردًا ودودًا: ")
        else:
            retrieved = self.kb_rag.retrieve_with_reranking(user_query, final_k=3)
            response = self.llm.generate(
                self.kb_rag.build_augmented_prompt(user_query, retrieved)
            )

        # 2. تقييم الثقة وقرار التصعيد
        confidence_prompt = f"""السؤال: {user_query}\nالإجابة المُقترَحة: {response}
ما مدى ثقتك بأن هذه الإجابة صحيحة وكافية؟ أعطِ رقمًا من 0 إلى 1 فقط."""
        try:
            confidence = float(self.llm.generate(confidence_prompt).strip())
        except ValueError:
            confidence = 0.5

        if confidence < self.escalation_threshold or intent == "شكوى":
            return {
                "response": response, "escalated": True,
                "escalation_reason": "ثقة منخفضة" if confidence < self.escalation_threshold else "شكوى تتطلب تدخلًا بشريًا"
            }
        return {"response": response, "escalated": False, "confidence": confidence}

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-53-ch53-domain-applications.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class GenericCopilotPattern:
    """
    النمط المعماري العام الذي تتشارك فيه كل "مساعدات Copilot" الناجحة:
    1. فهم السياق المحلي (الملف الحالي/المستند/الشاشة)
    2. فهم السياق العام (قاعدة المعرفة/المشروع الكامل عبر RAG)
    3. اقتراح فعل محدد وصغير (وليس تغييرًا جذريًا دفعة واحدة)
    4. تحقق تلقائي قبل العرض على المستخدم
    5. تغذية راجعة فورية (قبول/رفض المستخدم) تُحسِّن النظام لاحقًا
    """
    def __init__(self, llm, local_context_fn, global_rag, verifier_fn):
        self.llm = llm
        self.local_context_fn = local_context_fn      # يستخرج السياق المحلي الحالي
        self.global_rag = global_rag                    # معرفة عامة أوسع
        self.verifier_fn = verifier_fn                  # دالة تحقق مخصصة بالمجال
        self.feedback_log = []

    def suggest(self, trigger_context: dict) -> dict:
        local_ctx = self.local_context_fn(trigger_context)
        global_ctx = self.global_rag.retrieve(str(local_ctx), top_k=3)

        suggestion = self.llm.generate(f"""السياق المحلي: {local_ctx}
السياق العام ذو الصلة: {global_ctx}
اقترح الفعل التالي الأنسب والأكثر دقة:""")

        verified, verification_info = self.verifier_fn(suggestion)
        return {"suggestion": suggestion, "verified": verified, "details": verification_info}

    def log_feedback(self, suggestion_id: str, accepted: bool):
        self.feedback_log.append({"id": suggestion_id, "accepted": accepted})
        # في نظام ناضج: تُستخدم هذه البيانات لاحقًا لـ fine-tuning دوري (الفصل 23/30)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-53-ch53-domain-applications.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-53-ch53-domain-applications.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — تطبيقات متخصصة حسب المجال
Mastering Transformers and Large Language Models — Domain Applications

Book slug : mastering-llms-transformers
Chapter   : ch-53-ch53-domain-applications (lesson 53)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-10-applications-agents/ch53-domain-applications.ipynb#cell-1
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
# [002] content/notebooks/part-10-applications-agents/ch53-domain-applications.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
class CodeAssistantAgent:
    """
    معمارية مرجعية لمساعد برمجي: فهم السياق (المشروع الكامل) + تنفيذ + تحقق
    """
    def __init__(self, llm, codebase_rag: SimpleRAGPipeline, code_executor):
        self.llm = llm
        self.codebase_rag = codebase_rag   # فهرسة الكود الموجود كمستندات RAG
        self.code_executor = code_executor

    def generate_code_with_context(self, instruction: str, current_file: str) -> dict:
        # 1. استرجاع أجزاء ذات صلة من بقية المشروع (دوال مشابهة، تعريفات أنواع)
        relevant_code = self.codebase_rag.retrieve(instruction, top_k=5)

        prompt = f"""أنت مساعد برمجي خبير. اتّبع نمط الكود الموجود في المشروع بدقة.

أجزاء ذات صلة من المشروع:
{[c['text'] for c in relevant_code]}

الملف الحالي:
{current_file}

المطلوب: {instruction}

اكتب الكود المطلوب فقط، متّسقًا مع أسلوب المشروع:"""
        generated_code = self.llm.generate(prompt)

        # 2. تحقق تلقائي: تشغيل الكود واختباره قبل تقديمه
        test_result = self.code_executor.run_with_tests(generated_code)
        if not test_result["passed"]:
            fix_prompt = f"""الكود التالي فشل في الاختبار:
{generated_code}
خطأ: {test_result['error']}
صحّح الكود:"""
            generated_code = self.llm.generate(fix_prompt)

        return {"code": generated_code, "verified": test_result["passed"]}

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-10-applications-agents/ch53-domain-applications.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class CustomerSupportAgent:
    """
    معمارية شاملة: RAG على قاعدة المعرفة + أدوات للتفاعل مع أنظمة حقيقية
    (تتبع الطلبات، إلخ) + تصعيد ذكي لموظف بشري عند الحاجة
    """
    def __init__(self, llm, knowledge_base_rag, order_system_tool, escalation_threshold=0.6):
        self.llm = llm
        self.kb_rag = knowledge_base_rag
        self.order_tool = order_system_tool
        self.escalation_threshold = escalation_threshold

    def handle_query(self, user_query: str, user_id: str) -> dict:
        # 1. تقييم أولي: هل هذا سؤال معرفي عام، أم يتطلب بيانات خاصة بالمستخدم؟
        intent_prompt = f"""صنّف نوع الاستفسار التالي:
الاستفسار: {user_query}
الأنواع الممكنة: سؤال_عام، استعلام_طلب، شكوى، طلب_استرجاع
أعطِ التصنيف فقط."""
        intent = self.llm.generate(intent_prompt).strip()

        if intent == "استعلام_طلب":
            order_info = self.order_tool.get_order_status(user_id)
            response = self.llm.generate(f"معلومات الطلب: {order_info}\nصُغ ردًا ودودًا: ")
        else:
            retrieved = self.kb_rag.retrieve_with_reranking(user_query, final_k=3)
            response = self.llm.generate(
                self.kb_rag.build_augmented_prompt(user_query, retrieved)
            )

        # 2. تقييم الثقة وقرار التصعيد
        confidence_prompt = f"""السؤال: {user_query}\nالإجابة المُقترَحة: {response}
ما مدى ثقتك بأن هذه الإجابة صحيحة وكافية؟ أعطِ رقمًا من 0 إلى 1 فقط."""
        try:
            confidence = float(self.llm.generate(confidence_prompt).strip())
        except ValueError:
            confidence = 0.5

        if confidence < self.escalation_threshold or intent == "شكوى":
            return {
                "response": response, "escalated": True,
                "escalation_reason": "ثقة منخفضة" if confidence < self.escalation_threshold else "شكوى تتطلب تدخلًا بشريًا"
            }
        return {"response": response, "escalated": False, "confidence": confidence}

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-10-applications-agents/ch53-domain-applications.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class GenericCopilotPattern:
    """
    النمط المعماري العام الذي تتشارك فيه كل "مساعدات Copilot" الناجحة:
    1. فهم السياق المحلي (الملف الحالي/المستند/الشاشة)
    2. فهم السياق العام (قاعدة المعرفة/المشروع الكامل عبر RAG)
    3. اقتراح فعل محدد وصغير (وليس تغييرًا جذريًا دفعة واحدة)
    4. تحقق تلقائي قبل العرض على المستخدم
    5. تغذية راجعة فورية (قبول/رفض المستخدم) تُحسِّن النظام لاحقًا
    """
    def __init__(self, llm, local_context_fn, global_rag, verifier_fn):
        self.llm = llm
        self.local_context_fn = local_context_fn      # يستخرج السياق المحلي الحالي
        self.global_rag = global_rag                    # معرفة عامة أوسع
        self.verifier_fn = verifier_fn                  # دالة تحقق مخصصة بالمجال
        self.feedback_log = []

    def suggest(self, trigger_context: dict) -> dict:
        local_ctx = self.local_context_fn(trigger_context)
        global_ctx = self.global_rag.retrieve(str(local_ctx), top_k=3)

        suggestion = self.llm.generate(f"""السياق المحلي: {local_ctx}
السياق العام ذو الصلة: {global_ctx}
اقترح الفعل التالي الأنسب والأكثر دقة:""")

        verified, verification_info = self.verifier_fn(suggestion)
        return {"suggestion": suggestion, "verified": verified, "details": verification_info}

    def log_feedback(self, suggestion_id: str, accepted: bool):
        self.feedback_log.append({"id": suggestion_id, "accepted": accepted})
        # في نظام ناضج: تُستخدم هذه البيانات لاحقًا لـ fine-tuning دوري (الفصل 23/30)

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-10-applications-agents/ch53-domain-applications.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-53-ch53-domain-applications.ipynb#cell-2
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-53-ch53-domain-applications.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class CodeAssistantAgent:
    """
    معمارية مرجعية لمساعد برمجي: فهم السياق (المشروع الكامل) + تنفيذ + تحقق
    """
    def __init__(self, llm, codebase_rag: SimpleRAGPipeline, code_executor):
        self.llm = llm
        self.codebase_rag = codebase_rag   # فهرسة الكود الموجود كمستندات RAG
        self.code_executor = code_executor

    def generate_code_with_context(self, instruction: str, current_file: str) -> dict:
        # 1. استرجاع أجزاء ذات صلة من بقية المشروع (دوال مشابهة، تعريفات أنواع)
        relevant_code = self.codebase_rag.retrieve(instruction, top_k=5)

        prompt = f"""أنت مساعد برمجي خبير. اتّبع نمط الكود الموجود في المشروع بدقة.

أجزاء ذات صلة من المشروع:
{[c['text'] for c in relevant_code]}

الملف الحالي:
{current_file}

المطلوب: {instruction}

اكتب الكود المطلوب فقط، متّسقًا مع أسلوب المشروع:"""
        generated_code = self.llm.generate(prompt)

        # 2. تحقق تلقائي: تشغيل الكود واختباره قبل تقديمه
        test_result = self.code_executor.run_with_tests(generated_code)
        if not test_result["passed"]:
            fix_prompt = f"""الكود التالي فشل في الاختبار:
{generated_code}
خطأ: {test_result['error']}
صحّح الكود:"""
            generated_code = self.llm.generate(fix_prompt)

        return {"code": generated_code, "verified": test_result["passed"]}

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-53-ch53-domain-applications.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class CustomerSupportAgent:
    """
    معمارية شاملة: RAG على قاعدة المعرفة + أدوات للتفاعل مع أنظمة حقيقية
    (تتبع الطلبات، إلخ) + تصعيد ذكي لموظف بشري عند الحاجة
    """
    def __init__(self, llm, knowledge_base_rag, order_system_tool, escalation_threshold=0.6):
        self.llm = llm
        self.kb_rag = knowledge_base_rag
        self.order_tool = order_system_tool
        self.escalation_threshold = escalation_threshold

    def handle_query(self, user_query: str, user_id: str) -> dict:
        # 1. تقييم أولي: هل هذا سؤال معرفي عام، أم يتطلب بيانات خاصة بالمستخدم؟
        intent_prompt = f"""صنّف نوع الاستفسار التالي:
الاستفسار: {user_query}
الأنواع الممكنة: سؤال_عام، استعلام_طلب، شكوى، طلب_استرجاع
أعطِ التصنيف فقط."""
        intent = self.llm.generate(intent_prompt).strip()

        if intent == "استعلام_طلب":
            order_info = self.order_tool.get_order_status(user_id)
            response = self.llm.generate(f"معلومات الطلب: {order_info}\nصُغ ردًا ودودًا: ")
        else:
            retrieved = self.kb_rag.retrieve_with_reranking(user_query, final_k=3)
            response = self.llm.generate(
                self.kb_rag.build_augmented_prompt(user_query, retrieved)
            )

        # 2. تقييم الثقة وقرار التصعيد
        confidence_prompt = f"""السؤال: {user_query}\nالإجابة المُقترَحة: {response}
ما مدى ثقتك بأن هذه الإجابة صحيحة وكافية؟ أعطِ رقمًا من 0 إلى 1 فقط."""
        try:
            confidence = float(self.llm.generate(confidence_prompt).strip())
        except ValueError:
            confidence = 0.5

        if confidence < self.escalation_threshold or intent == "شكوى":
            return {
                "response": response, "escalated": True,
                "escalation_reason": "ثقة منخفضة" if confidence < self.escalation_threshold else "شكوى تتطلب تدخلًا بشريًا"
            }
        return {"response": response, "escalated": False, "confidence": confidence}

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-53-ch53-domain-applications.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class GenericCopilotPattern:
    """
    النمط المعماري العام الذي تتشارك فيه كل "مساعدات Copilot" الناجحة:
    1. فهم السياق المحلي (الملف الحالي/المستند/الشاشة)
    2. فهم السياق العام (قاعدة المعرفة/المشروع الكامل عبر RAG)
    3. اقتراح فعل محدد وصغير (وليس تغييرًا جذريًا دفعة واحدة)
    4. تحقق تلقائي قبل العرض على المستخدم
    5. تغذية راجعة فورية (قبول/رفض المستخدم) تُحسِّن النظام لاحقًا
    """
    def __init__(self, llm, local_context_fn, global_rag, verifier_fn):
        self.llm = llm
        self.local_context_fn = local_context_fn      # يستخرج السياق المحلي الحالي
        self.global_rag = global_rag                    # معرفة عامة أوسع
        self.verifier_fn = verifier_fn                  # دالة تحقق مخصصة بالمجال
        self.feedback_log = []

    def suggest(self, trigger_context: dict) -> dict:
        local_ctx = self.local_context_fn(trigger_context)
        global_ctx = self.global_rag.retrieve(str(local_ctx), top_k=3)

        suggestion = self.llm.generate(f"""السياق المحلي: {local_ctx}
السياق العام ذو الصلة: {global_ctx}
اقترح الفعل التالي الأنسب والأكثر دقة:""")

        verified, verification_info = self.verifier_fn(suggestion)
        return {"suggestion": suggestion, "verified": verified, "details": verification_info}

    def log_feedback(self, suggestion_id: str, accepted: bool):
        self.feedback_log.append({"id": suggestion_id, "accepted": accepted})
        # في نظام ناضج: تُستخدم هذه البيانات لاحقًا لـ fine-tuning دوري (الفصل 23/30)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-53-ch53-domain-applications.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
