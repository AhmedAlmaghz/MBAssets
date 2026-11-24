#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الذاكرة طويلة المدى للوكلاء
Mastering Transformers and Large Language Models — Long Term Memory

Book slug : mastering-llms-transformers
Chapter   : ch-51-ch51-long-term-memory (lesson 51)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-51-ch51-long-term-memory.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الذاكرة طويلة المدى للوكلاء
Mastering Transformers and Large Language Models — Long Term Memory

Book slug : mastering-llms-transformers
Chapter   : ch-51-ch51-long-term-memory (lesson 51)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-51-ch51-long-term-memory.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الذاكرة طويلة المدى للوكلاء
Mastering Transformers and Large Language Models — Long Term Memory

Book slug : mastering-llms-transformers
Chapter   : ch-51-ch51-long-term-memory (lesson 51)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-10-applications-agents/ch51-long-term-memory.ipynb#cell-1
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
# [002] content/notebooks/part-10-applications-agents/ch51-long-term-memory.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import datetime

class EpisodicMemory:
    """
    ذاكرة عرضية: تخزين واسترجاع محادثات/أحداث سابقة
    تُبنى فوق نفس آليات RAG من الفصل 47، لكن المصدر هو تاريخ التفاعل
    """
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store   # يمكن استخدام FAISS أو قاعدة متجهات حقيقية (Pinecone/Weaviate)
        self.memory_log = []

    def store_interaction(self, user_id: str, user_message: str, assistant_response: str):
        timestamp = datetime.datetime.now()
        memory_entry = {
            "user_id": user_id,
            "content": f"المستخدم قال: {user_message}\nالمساعد ردّ: {assistant_response}",
            "timestamp": timestamp.isoformat()
        }
        embedding = self.embedder.encode([f"passage: {memory_entry['content']}"])[0]
        self.vector_store.add(embedding, metadata=memory_entry)
        self.memory_log.append(memory_entry)

    def retrieve_relevant_memories(self, user_id: str, current_query: str, top_k=5) -> list:
        query_embedding = self.embedder.encode([f"query: {current_query}"])[0]
        results = self.vector_store.search(query_embedding, top_k=top_k * 3)
        # تصفية حسب المستخدم + ترتيب حسب الصلة والحداثة معًا
        user_memories = [r for r in results if r["metadata"]["user_id"] == user_id]
        return self._rerank_by_relevance_and_recency(user_memories)[:top_k]

    def _rerank_by_relevance_and_recency(self, memories: list, recency_weight=0.3) -> list:
        """دمج التشابه الدلالي مع حداثة الذكرى — الذكريات الأحدث أكثر صلة غالبًا"""
        now = datetime.datetime.now()
        for mem in memories:
            age_days = (now - datetime.datetime.fromisoformat(mem["metadata"]["timestamp"])).days
            recency_score = 1 / (1 + age_days)   # يتناقص مع الزمن
            mem["combined_score"] = (1 - recency_weight) * mem["score"] + recency_weight * recency_score
        return sorted(memories, key=lambda x: x["combined_score"], reverse=True)

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-10-applications-agents/ch51-long-term-memory.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class SemanticMemoryExtractor:
    """
    نهج مستوحى من MemGPT وMem0:
    بعد كل محادثة، LLM يستخرج حقائق قابلة لإعادة الاستخدام طويل المدى
    ويُحدِّث "ملف شخصية" (profile) تراكمي للمستخدم
    """
    def __init__(self, llm):
        self.llm = llm
        self.user_profiles = {}   # {user_id: {facts: [...], preferences: [...]}}

    def extract_and_update_facts(self, user_id: str, conversation: str):
        existing_facts = self.user_profiles.get(user_id, {"facts": [], "preferences": []})

        extraction_prompt = f"""الملف الحالي للمستخدم:
الحقائق المعروفة: {existing_facts['facts']}
التفضيلات المعروفة: {existing_facts['preferences']}

المحادثة الجديدة:
{conversation}

استخرج أي حقائق جديدة أو تفضيلات جديدة عن المستخدم من هذه المحادثة فقط (معلومات دائمة ومفيدة مستقبلًا،
مثل: مهنته، اهتماماته، قيود يجب تذكّرها). إذا تناقضت معلومة جديدة مع حقيقة قديمة، أشر لذلك صراحة.

أعطِ النتيجة بصيغة JSON: {{"new_facts": [...], "new_preferences": [...], "contradictions": [...]}}"""

        result = json.loads(self.llm.generate(extraction_prompt))

        existing_facts["facts"].extend(result.get("new_facts", []))
        existing_facts["preferences"].extend(result.get("new_preferences", []))

        # معالجة التناقضات: إزالة الحقائق القديمة المتناقضة
        for contradiction in result.get("contradictions", []):
            existing_facts["facts"] = [f for f in existing_facts["facts"] if f != contradiction]

        self.user_profiles[user_id] = existing_facts
        return existing_facts

    def get_memory_context_for_prompt(self, user_id: str) -> str:
        """بناء ملخص موجز للذاكرة الدلالية لإدراجه في الـ system prompt"""
        profile = self.user_profiles.get(user_id, {"facts": [], "preferences": []})
        if not profile["facts"] and not profile["preferences"]:
            return ""
        return f"""معلومات معروفة عن هذا المستخدم:
الحقائق: {'; '.join(profile['facts'])}
التفضيلات: {'; '.join(profile['preferences'])}"""

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-10-applications-agents/ch51-long-term-memory.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class MemGPTStyleContextManager:
    """
    تطبيق مفاهيمي لفكرة MemGPT:
    - Main Context (نافذة السياق الحالية): محدودة، "سريعة"
    - External Context (ذاكرة خارجية): غير محدودة، يُسترجَع منها عند الحاجة
    - النموذج نفسه يُقرِّر متى "يُخرج" معلومات من السياق الرئيسي لحفظ مساحة،
      ومتى "يستدعي" معلومات من الذاكرة الخارجية
    """
    def __init__(self, llm, max_context_tokens=4000,
                 episodic_memory: EpisodicMemory = None,
                 semantic_memory: SemanticMemoryExtractor = None):
        self.llm = llm
        self.max_context_tokens = max_context_tokens
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory
        self.current_context = []   # الرسائل الحالية في النافذة

        # أدوات خاصة لإدارة الذاكرة، متاحة للنموذج نفسه كـ function calls
        self.memory_tools = [
            Tool("أرشفة_ذاكرة", "ينقل معلومة من السياق الحالي للذاكرة طويلة المدى",
                 self._archive_to_long_term),
            Tool("استرجاع_ذاكرة", "يبحث في الذاكرة طويلة المدى عن معلومة محدّدة",
                 self._search_long_term),
        ]

    def _estimate_tokens(self, messages: list) -> int:
        return sum(len(m["content"].split()) * 1.3 for m in messages)   # تقدير تقريبي

    def _archive_to_long_term(self, content: str) -> str:
        if self.episodic_memory:
            self.episodic_memory.store_interaction("system_archive", "", content)
        return "تمت الأرشفة بنجاح"

    def _search_long_term(self, query: str) -> str:
        if self.episodic_memory:
            results = self.episodic_memory.retrieve_relevant_memories("system_archive", query)
            return "\n".join([r["metadata"]["content"] for r in results])
        return "لا توجد ذاكرة طويلة المدى متاحة"

    def add_message(self, role: str, content: str):
        self.current_context.append({"role": role, "content": content})

        # إدارة الذاكرة التلقائية: عند اقتراب الامتلاء، أرشفة أقدم الرسائل
        if self._estimate_tokens(self.current_context) > self.max_context_tokens * 0.9:
            self._compress_and_archive_oldest()

    def _compress_and_archive_oldest(self, keep_recent=5):
        """ضغط وأرشفة أقدم جزء من المحادثة قبل حذفه من النافذة النشطة"""
        to_archive = self.current_context[:-keep_recent]
        self.current_context = self.current_context[-keep_recent:]

        summary_prompt = f"لخّص المحادثة التالية بإيجاز شديد مع حفظ أي معلومة مهمة: {to_archive}"
        summary = self.llm.generate(summary_prompt)
        self._archive_to_long_term(summary)

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-10-applications-agents/ch51-long-term-memory.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-51-ch51-long-term-memory.ipynb#cell-1
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-51-ch51-long-term-memory.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import datetime

class EpisodicMemory:
    """
    ذاكرة عرضية: تخزين واسترجاع محادثات/أحداث سابقة
    تُبنى فوق نفس آليات RAG من الفصل 47، لكن المصدر هو تاريخ التفاعل
    """
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store   # يمكن استخدام FAISS أو قاعدة متجهات حقيقية (Pinecone/Weaviate)
        self.memory_log = []

    def store_interaction(self, user_id: str, user_message: str, assistant_response: str):
        timestamp = datetime.datetime.now()
        memory_entry = {
            "user_id": user_id,
            "content": f"المستخدم قال: {user_message}\nالمساعد ردّ: {assistant_response}",
            "timestamp": timestamp.isoformat()
        }
        embedding = self.embedder.encode([f"passage: {memory_entry['content']}"])[0]
        self.vector_store.add(embedding, metadata=memory_entry)
        self.memory_log.append(memory_entry)

    def retrieve_relevant_memories(self, user_id: str, current_query: str, top_k=5) -> list:
        query_embedding = self.embedder.encode([f"query: {current_query}"])[0]
        results = self.vector_store.search(query_embedding, top_k=top_k * 3)
        # تصفية حسب المستخدم + ترتيب حسب الصلة والحداثة معًا
        user_memories = [r for r in results if r["metadata"]["user_id"] == user_id]
        return self._rerank_by_relevance_and_recency(user_memories)[:top_k]

    def _rerank_by_relevance_and_recency(self, memories: list, recency_weight=0.3) -> list:
        """دمج التشابه الدلالي مع حداثة الذكرى — الذكريات الأحدث أكثر صلة غالبًا"""
        now = datetime.datetime.now()
        for mem in memories:
            age_days = (now - datetime.datetime.fromisoformat(mem["metadata"]["timestamp"])).days
            recency_score = 1 / (1 + age_days)   # يتناقص مع الزمن
            mem["combined_score"] = (1 - recency_weight) * mem["score"] + recency_weight * recency_score
        return sorted(memories, key=lambda x: x["combined_score"], reverse=True)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-51-ch51-long-term-memory.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class SemanticMemoryExtractor:
    """
    نهج مستوحى من MemGPT وMem0:
    بعد كل محادثة، LLM يستخرج حقائق قابلة لإعادة الاستخدام طويل المدى
    ويُحدِّث "ملف شخصية" (profile) تراكمي للمستخدم
    """
    def __init__(self, llm):
        self.llm = llm
        self.user_profiles = {}   # {user_id: {facts: [...], preferences: [...]}}

    def extract_and_update_facts(self, user_id: str, conversation: str):
        existing_facts = self.user_profiles.get(user_id, {"facts": [], "preferences": []})

        extraction_prompt = f"""الملف الحالي للمستخدم:
الحقائق المعروفة: {existing_facts['facts']}
التفضيلات المعروفة: {existing_facts['preferences']}

المحادثة الجديدة:
{conversation}

استخرج أي حقائق جديدة أو تفضيلات جديدة عن المستخدم من هذه المحادثة فقط (معلومات دائمة ومفيدة مستقبلًا،
مثل: مهنته، اهتماماته، قيود يجب تذكّرها). إذا تناقضت معلومة جديدة مع حقيقة قديمة، أشر لذلك صراحة.

أعطِ النتيجة بصيغة JSON: {{"new_facts": [...], "new_preferences": [...], "contradictions": [...]}}"""

        result = json.loads(self.llm.generate(extraction_prompt))

        existing_facts["facts"].extend(result.get("new_facts", []))
        existing_facts["preferences"].extend(result.get("new_preferences", []))

        # معالجة التناقضات: إزالة الحقائق القديمة المتناقضة
        for contradiction in result.get("contradictions", []):
            existing_facts["facts"] = [f for f in existing_facts["facts"] if f != contradiction]

        self.user_profiles[user_id] = existing_facts
        return existing_facts

    def get_memory_context_for_prompt(self, user_id: str) -> str:
        """بناء ملخص موجز للذاكرة الدلالية لإدراجه في الـ system prompt"""
        profile = self.user_profiles.get(user_id, {"facts": [], "preferences": []})
        if not profile["facts"] and not profile["preferences"]:
            return ""
        return f"""معلومات معروفة عن هذا المستخدم:
الحقائق: {'; '.join(profile['facts'])}
التفضيلات: {'; '.join(profile['preferences'])}"""

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-51-ch51-long-term-memory.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class MemGPTStyleContextManager:
    """
    تطبيق مفاهيمي لفكرة MemGPT:
    - Main Context (نافذة السياق الحالية): محدودة، "سريعة"
    - External Context (ذاكرة خارجية): غير محدودة، يُسترجَع منها عند الحاجة
    - النموذج نفسه يُقرِّر متى "يُخرج" معلومات من السياق الرئيسي لحفظ مساحة،
      ومتى "يستدعي" معلومات من الذاكرة الخارجية
    """
    def __init__(self, llm, max_context_tokens=4000,
                 episodic_memory: EpisodicMemory = None,
                 semantic_memory: SemanticMemoryExtractor = None):
        self.llm = llm
        self.max_context_tokens = max_context_tokens
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory
        self.current_context = []   # الرسائل الحالية في النافذة

        # أدوات خاصة لإدارة الذاكرة، متاحة للنموذج نفسه كـ function calls
        self.memory_tools = [
            Tool("أرشفة_ذاكرة", "ينقل معلومة من السياق الحالي للذاكرة طويلة المدى",
                 self._archive_to_long_term),
            Tool("استرجاع_ذاكرة", "يبحث في الذاكرة طويلة المدى عن معلومة محدّدة",
                 self._search_long_term),
        ]

    def _estimate_tokens(self, messages: list) -> int:
        return sum(len(m["content"].split()) * 1.3 for m in messages)   # تقدير تقريبي

    def _archive_to_long_term(self, content: str) -> str:
        if self.episodic_memory:
            self.episodic_memory.store_interaction("system_archive", "", content)
        return "تمت الأرشفة بنجاح"

    def _search_long_term(self, query: str) -> str:
        if self.episodic_memory:
            results = self.episodic_memory.retrieve_relevant_memories("system_archive", query)
            return "\n".join([r["metadata"]["content"] for r in results])
        return "لا توجد ذاكرة طويلة المدى متاحة"

    def add_message(self, role: str, content: str):
        self.current_context.append({"role": role, "content": content})

        # إدارة الذاكرة التلقائية: عند اقتراب الامتلاء، أرشفة أقدم الرسائل
        if self._estimate_tokens(self.current_context) > self.max_context_tokens * 0.9:
            self._compress_and_archive_oldest()

    def _compress_and_archive_oldest(self, keep_recent=5):
        """ضغط وأرشفة أقدم جزء من المحادثة قبل حذفه من النافذة النشطة"""
        to_archive = self.current_context[:-keep_recent]
        self.current_context = self.current_context[-keep_recent:]

        summary_prompt = f"لخّص المحادثة التالية بإيجاز شديد مع حفظ أي معلومة مهمة: {to_archive}"
        summary = self.llm.generate(summary_prompt)
        self._archive_to_long_term(summary)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-51-ch51-long-term-memory.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-51-ch51-long-term-memory.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الذاكرة طويلة المدى للوكلاء
Mastering Transformers and Large Language Models — Long Term Memory

Book slug : mastering-llms-transformers
Chapter   : ch-51-ch51-long-term-memory (lesson 51)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-10-applications-agents/ch51-long-term-memory.ipynb#cell-1
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
# [002] content/notebooks/part-10-applications-agents/ch51-long-term-memory.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import datetime

class EpisodicMemory:
    """
    ذاكرة عرضية: تخزين واسترجاع محادثات/أحداث سابقة
    تُبنى فوق نفس آليات RAG من الفصل 47، لكن المصدر هو تاريخ التفاعل
    """
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store   # يمكن استخدام FAISS أو قاعدة متجهات حقيقية (Pinecone/Weaviate)
        self.memory_log = []

    def store_interaction(self, user_id: str, user_message: str, assistant_response: str):
        timestamp = datetime.datetime.now()
        memory_entry = {
            "user_id": user_id,
            "content": f"المستخدم قال: {user_message}\nالمساعد ردّ: {assistant_response}",
            "timestamp": timestamp.isoformat()
        }
        embedding = self.embedder.encode([f"passage: {memory_entry['content']}"])[0]
        self.vector_store.add(embedding, metadata=memory_entry)
        self.memory_log.append(memory_entry)

    def retrieve_relevant_memories(self, user_id: str, current_query: str, top_k=5) -> list:
        query_embedding = self.embedder.encode([f"query: {current_query}"])[0]
        results = self.vector_store.search(query_embedding, top_k=top_k * 3)
        # تصفية حسب المستخدم + ترتيب حسب الصلة والحداثة معًا
        user_memories = [r for r in results if r["metadata"]["user_id"] == user_id]
        return self._rerank_by_relevance_and_recency(user_memories)[:top_k]

    def _rerank_by_relevance_and_recency(self, memories: list, recency_weight=0.3) -> list:
        """دمج التشابه الدلالي مع حداثة الذكرى — الذكريات الأحدث أكثر صلة غالبًا"""
        now = datetime.datetime.now()
        for mem in memories:
            age_days = (now - datetime.datetime.fromisoformat(mem["metadata"]["timestamp"])).days
            recency_score = 1 / (1 + age_days)   # يتناقص مع الزمن
            mem["combined_score"] = (1 - recency_weight) * mem["score"] + recency_weight * recency_score
        return sorted(memories, key=lambda x: x["combined_score"], reverse=True)

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-10-applications-agents/ch51-long-term-memory.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class SemanticMemoryExtractor:
    """
    نهج مستوحى من MemGPT وMem0:
    بعد كل محادثة، LLM يستخرج حقائق قابلة لإعادة الاستخدام طويل المدى
    ويُحدِّث "ملف شخصية" (profile) تراكمي للمستخدم
    """
    def __init__(self, llm):
        self.llm = llm
        self.user_profiles = {}   # {user_id: {facts: [...], preferences: [...]}}

    def extract_and_update_facts(self, user_id: str, conversation: str):
        existing_facts = self.user_profiles.get(user_id, {"facts": [], "preferences": []})

        extraction_prompt = f"""الملف الحالي للمستخدم:
الحقائق المعروفة: {existing_facts['facts']}
التفضيلات المعروفة: {existing_facts['preferences']}

المحادثة الجديدة:
{conversation}

استخرج أي حقائق جديدة أو تفضيلات جديدة عن المستخدم من هذه المحادثة فقط (معلومات دائمة ومفيدة مستقبلًا،
مثل: مهنته، اهتماماته، قيود يجب تذكّرها). إذا تناقضت معلومة جديدة مع حقيقة قديمة، أشر لذلك صراحة.

أعطِ النتيجة بصيغة JSON: {{"new_facts": [...], "new_preferences": [...], "contradictions": [...]}}"""

        result = json.loads(self.llm.generate(extraction_prompt))

        existing_facts["facts"].extend(result.get("new_facts", []))
        existing_facts["preferences"].extend(result.get("new_preferences", []))

        # معالجة التناقضات: إزالة الحقائق القديمة المتناقضة
        for contradiction in result.get("contradictions", []):
            existing_facts["facts"] = [f for f in existing_facts["facts"] if f != contradiction]

        self.user_profiles[user_id] = existing_facts
        return existing_facts

    def get_memory_context_for_prompt(self, user_id: str) -> str:
        """بناء ملخص موجز للذاكرة الدلالية لإدراجه في الـ system prompt"""
        profile = self.user_profiles.get(user_id, {"facts": [], "preferences": []})
        if not profile["facts"] and not profile["preferences"]:
            return ""
        return f"""معلومات معروفة عن هذا المستخدم:
الحقائق: {'; '.join(profile['facts'])}
التفضيلات: {'; '.join(profile['preferences'])}"""

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-10-applications-agents/ch51-long-term-memory.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class MemGPTStyleContextManager:
    """
    تطبيق مفاهيمي لفكرة MemGPT:
    - Main Context (نافذة السياق الحالية): محدودة، "سريعة"
    - External Context (ذاكرة خارجية): غير محدودة، يُسترجَع منها عند الحاجة
    - النموذج نفسه يُقرِّر متى "يُخرج" معلومات من السياق الرئيسي لحفظ مساحة،
      ومتى "يستدعي" معلومات من الذاكرة الخارجية
    """
    def __init__(self, llm, max_context_tokens=4000,
                 episodic_memory: EpisodicMemory = None,
                 semantic_memory: SemanticMemoryExtractor = None):
        self.llm = llm
        self.max_context_tokens = max_context_tokens
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory
        self.current_context = []   # الرسائل الحالية في النافذة

        # أدوات خاصة لإدارة الذاكرة، متاحة للنموذج نفسه كـ function calls
        self.memory_tools = [
            Tool("أرشفة_ذاكرة", "ينقل معلومة من السياق الحالي للذاكرة طويلة المدى",
                 self._archive_to_long_term),
            Tool("استرجاع_ذاكرة", "يبحث في الذاكرة طويلة المدى عن معلومة محدّدة",
                 self._search_long_term),
        ]

    def _estimate_tokens(self, messages: list) -> int:
        return sum(len(m["content"].split()) * 1.3 for m in messages)   # تقدير تقريبي

    def _archive_to_long_term(self, content: str) -> str:
        if self.episodic_memory:
            self.episodic_memory.store_interaction("system_archive", "", content)
        return "تمت الأرشفة بنجاح"

    def _search_long_term(self, query: str) -> str:
        if self.episodic_memory:
            results = self.episodic_memory.retrieve_relevant_memories("system_archive", query)
            return "\n".join([r["metadata"]["content"] for r in results])
        return "لا توجد ذاكرة طويلة المدى متاحة"

    def add_message(self, role: str, content: str):
        self.current_context.append({"role": role, "content": content})

        # إدارة الذاكرة التلقائية: عند اقتراب الامتلاء، أرشفة أقدم الرسائل
        if self._estimate_tokens(self.current_context) > self.max_context_tokens * 0.9:
            self._compress_and_archive_oldest()

    def _compress_and_archive_oldest(self, keep_recent=5):
        """ضغط وأرشفة أقدم جزء من المحادثة قبل حذفه من النافذة النشطة"""
        to_archive = self.current_context[:-keep_recent]
        self.current_context = self.current_context[-keep_recent:]

        summary_prompt = f"لخّص المحادثة التالية بإيجاز شديد مع حفظ أي معلومة مهمة: {to_archive}"
        summary = self.llm.generate(summary_prompt)
        self._archive_to_long_term(summary)

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-10-applications-agents/ch51-long-term-memory.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-51-ch51-long-term-memory.ipynb#cell-2
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-51-ch51-long-term-memory.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import datetime

class EpisodicMemory:
    """
    ذاكرة عرضية: تخزين واسترجاع محادثات/أحداث سابقة
    تُبنى فوق نفس آليات RAG من الفصل 47، لكن المصدر هو تاريخ التفاعل
    """
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store   # يمكن استخدام FAISS أو قاعدة متجهات حقيقية (Pinecone/Weaviate)
        self.memory_log = []

    def store_interaction(self, user_id: str, user_message: str, assistant_response: str):
        timestamp = datetime.datetime.now()
        memory_entry = {
            "user_id": user_id,
            "content": f"المستخدم قال: {user_message}\nالمساعد ردّ: {assistant_response}",
            "timestamp": timestamp.isoformat()
        }
        embedding = self.embedder.encode([f"passage: {memory_entry['content']}"])[0]
        self.vector_store.add(embedding, metadata=memory_entry)
        self.memory_log.append(memory_entry)

    def retrieve_relevant_memories(self, user_id: str, current_query: str, top_k=5) -> list:
        query_embedding = self.embedder.encode([f"query: {current_query}"])[0]
        results = self.vector_store.search(query_embedding, top_k=top_k * 3)
        # تصفية حسب المستخدم + ترتيب حسب الصلة والحداثة معًا
        user_memories = [r for r in results if r["metadata"]["user_id"] == user_id]
        return self._rerank_by_relevance_and_recency(user_memories)[:top_k]

    def _rerank_by_relevance_and_recency(self, memories: list, recency_weight=0.3) -> list:
        """دمج التشابه الدلالي مع حداثة الذكرى — الذكريات الأحدث أكثر صلة غالبًا"""
        now = datetime.datetime.now()
        for mem in memories:
            age_days = (now - datetime.datetime.fromisoformat(mem["metadata"]["timestamp"])).days
            recency_score = 1 / (1 + age_days)   # يتناقص مع الزمن
            mem["combined_score"] = (1 - recency_weight) * mem["score"] + recency_weight * recency_score
        return sorted(memories, key=lambda x: x["combined_score"], reverse=True)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-51-ch51-long-term-memory.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class SemanticMemoryExtractor:
    """
    نهج مستوحى من MemGPT وMem0:
    بعد كل محادثة، LLM يستخرج حقائق قابلة لإعادة الاستخدام طويل المدى
    ويُحدِّث "ملف شخصية" (profile) تراكمي للمستخدم
    """
    def __init__(self, llm):
        self.llm = llm
        self.user_profiles = {}   # {user_id: {facts: [...], preferences: [...]}}

    def extract_and_update_facts(self, user_id: str, conversation: str):
        existing_facts = self.user_profiles.get(user_id, {"facts": [], "preferences": []})

        extraction_prompt = f"""الملف الحالي للمستخدم:
الحقائق المعروفة: {existing_facts['facts']}
التفضيلات المعروفة: {existing_facts['preferences']}

المحادثة الجديدة:
{conversation}

استخرج أي حقائق جديدة أو تفضيلات جديدة عن المستخدم من هذه المحادثة فقط (معلومات دائمة ومفيدة مستقبلًا،
مثل: مهنته، اهتماماته، قيود يجب تذكّرها). إذا تناقضت معلومة جديدة مع حقيقة قديمة، أشر لذلك صراحة.

أعطِ النتيجة بصيغة JSON: {{"new_facts": [...], "new_preferences": [...], "contradictions": [...]}}"""

        result = json.loads(self.llm.generate(extraction_prompt))

        existing_facts["facts"].extend(result.get("new_facts", []))
        existing_facts["preferences"].extend(result.get("new_preferences", []))

        # معالجة التناقضات: إزالة الحقائق القديمة المتناقضة
        for contradiction in result.get("contradictions", []):
            existing_facts["facts"] = [f for f in existing_facts["facts"] if f != contradiction]

        self.user_profiles[user_id] = existing_facts
        return existing_facts

    def get_memory_context_for_prompt(self, user_id: str) -> str:
        """بناء ملخص موجز للذاكرة الدلالية لإدراجه في الـ system prompt"""
        profile = self.user_profiles.get(user_id, {"facts": [], "preferences": []})
        if not profile["facts"] and not profile["preferences"]:
            return ""
        return f"""معلومات معروفة عن هذا المستخدم:
الحقائق: {'; '.join(profile['facts'])}
التفضيلات: {'; '.join(profile['preferences'])}"""

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-51-ch51-long-term-memory.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class MemGPTStyleContextManager:
    """
    تطبيق مفاهيمي لفكرة MemGPT:
    - Main Context (نافذة السياق الحالية): محدودة، "سريعة"
    - External Context (ذاكرة خارجية): غير محدودة، يُسترجَع منها عند الحاجة
    - النموذج نفسه يُقرِّر متى "يُخرج" معلومات من السياق الرئيسي لحفظ مساحة،
      ومتى "يستدعي" معلومات من الذاكرة الخارجية
    """
    def __init__(self, llm, max_context_tokens=4000,
                 episodic_memory: EpisodicMemory = None,
                 semantic_memory: SemanticMemoryExtractor = None):
        self.llm = llm
        self.max_context_tokens = max_context_tokens
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory
        self.current_context = []   # الرسائل الحالية في النافذة

        # أدوات خاصة لإدارة الذاكرة، متاحة للنموذج نفسه كـ function calls
        self.memory_tools = [
            Tool("أرشفة_ذاكرة", "ينقل معلومة من السياق الحالي للذاكرة طويلة المدى",
                 self._archive_to_long_term),
            Tool("استرجاع_ذاكرة", "يبحث في الذاكرة طويلة المدى عن معلومة محدّدة",
                 self._search_long_term),
        ]

    def _estimate_tokens(self, messages: list) -> int:
        return sum(len(m["content"].split()) * 1.3 for m in messages)   # تقدير تقريبي

    def _archive_to_long_term(self, content: str) -> str:
        if self.episodic_memory:
            self.episodic_memory.store_interaction("system_archive", "", content)
        return "تمت الأرشفة بنجاح"

    def _search_long_term(self, query: str) -> str:
        if self.episodic_memory:
            results = self.episodic_memory.retrieve_relevant_memories("system_archive", query)
            return "\n".join([r["metadata"]["content"] for r in results])
        return "لا توجد ذاكرة طويلة المدى متاحة"

    def add_message(self, role: str, content: str):
        self.current_context.append({"role": role, "content": content})

        # إدارة الذاكرة التلقائية: عند اقتراب الامتلاء، أرشفة أقدم الرسائل
        if self._estimate_tokens(self.current_context) > self.max_context_tokens * 0.9:
            self._compress_and_archive_oldest()

    def _compress_and_archive_oldest(self, keep_recent=5):
        """ضغط وأرشفة أقدم جزء من المحادثة قبل حذفه من النافذة النشطة"""
        to_archive = self.current_context[:-keep_recent]
        self.current_context = self.current_context[-keep_recent:]

        summary_prompt = f"لخّص المحادثة التالية بإيجاز شديد مع حفظ أي معلومة مهمة: {to_archive}"
        summary = self.llm.generate(summary_prompt)
        self._archive_to_long_term(summary)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-51-ch51-long-term-memory.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
