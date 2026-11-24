#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الذاكرة طويلة المدى للمحادثات (Long-term Memory)
Large Language Model Engineering — Long-term

Book slug : llm-engineering-69-chapters
Chapter   : ch-57-chapter-57 (lesson 57)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-57-chapter-57.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الذاكرة طويلة المدى للمحادثات (Long-term Memory)
Large Language Model Engineering — Long-term

Book slug : llm-engineering-69-chapters
Chapter   : ch-57-chapter-57 (lesson 57)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-57-chapter-57.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الذاكرة طويلة المدى للمحادثات (Long-term Memory)
Large Language Model Engineering — Long-term

Book slug : llm-engineering-69-chapters
Chapter   : ch-57-chapter-57 (lesson 57)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_57.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 57: الفصل 57: الذاكرة طويلة المدى للمحادثات (Long-term Memory)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_schema() -> None:
    schema = {'type': 'object', 'required': ['query', 'limit'], 'properties': {'query': str, 'limit': int}}
    arguments = {'query': 'attention', 'limit': 3}
    errors = []
    for key in schema['required']:
        if key not in arguments:
            errors.append(f'missing: {key}')
    for key, expected in schema['properties'].items():
        if key in arguments and (not isinstance(arguments[key], expected) or isinstance(arguments[key], bool)):
            errors.append(f'invalid type: {key}')
    print('arguments:', arguments)
    print('schema validation:', 'PASS' if not errors else errors)

if __name__ == "__main__":
    print("Chapter 57 lab: Validate structured tool arguments before execution")
    run_schema()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-10-applications-agents/chapter-57.md
# ────────────────────────────────────────────────────────────────────────
import os
import json
import time
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np

from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBED_DIM = embedder.get_sentence_embedding_dimension()


def call_llm(system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini",
             temperature: float = 0.3) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-10-applications-agents/chapter-57.md
# ────────────────────────────────────────────────────────────────────────
@dataclass
class MemoryItem:
    """ذكرى واحدة: نص + metadata ضرورية لحساب درجة الاسترجاع لاحقًا"""
    content: str
    timestamp: float
    importance: float          # 0-1، تُقدَّر عبر LLM وقت التسجيل
    embedding: np.ndarray
    memory_id: int
    access_count: int = 0      # عدد مرات الاسترجاع (مؤشر فائدة إضافي)
    last_accessed: float = field(default_factory=time.time)


class LongTermMemory:
    """
    ذاكرة دلالية طويلة المدى تُطبّق معادلة الاسترجاع المرجّح:
    score = α·recency + β·importance + γ·relevance
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.3, gamma: float = 0.4,
                 decay_lambda: float = 0.01, forgetting_threshold: float = 0.05):
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.decay_lambda = decay_lambda           # سرعة تلاشي الذاكرة (لكل ساعة)
        self.forgetting_threshold = forgetting_threshold
        self.memories: Dict[int, MemoryItem] = {}
        self.index = faiss.IndexFlatIP(EMBED_DIM)  # Inner Product على متجهات منظّمة = cosine sim
        self._next_id = 0

    def _estimate_importance(self, content: str) -> float:
        """نطلب من LLM تقدير أهمية الذكرى على مقياس 1-10، ثم نطبّعها لـ [0,1]"""
        prompt = f"""قيّم أهمية المعلومة التالية على مقياس من 1 إلى 10 لمساعد شخصي طويل المدى
(10 = معلومة حيوية دائمة مثل حساسية طبية أو تفضيل جوهري، 1 = تفصيل عابر لا قيمة له لاحقًا):

المعلومة: "{content}"

أعد رقمًا فقط بدون أي نص إضافي."""
        try:
            score = float(call_llm("أنت مقيّم دقيق.", prompt, temperature=0.0).strip())
            return max(1.0, min(10.0, score)) / 10.0
        except ValueError:
            return 0.5  # قيمة افتراضية آمنة عند فشل التحليل

    def add_memory(self, content: str, importance: Optional[float] = None) -> int:
        """تسجيل ذكرى جديدة: يحسب embedding وأهمية ويضيفها للفهرس"""
        embedding = embedder.encode([content], normalize_embeddings=True)[0]
        final_importance = importance if importance is not None else self._estimate_importance(content)

        memory_id = self._next_id
        self._next_id += 1

        item = MemoryItem(content=content, timestamp=time.time(), importance=final_importance,
                           embedding=embedding, memory_id=memory_id)
        self.memories[memory_id] = item
        self.index.add(embedding.reshape(1, -1))

        print(f"🧠 ذكرى جديدة [id={memory_id}, أهمية={final_importance:.2f}]: {content[:60]}...")
        return memory_id

    def _recency_score(self, item: MemoryItem, now: float) -> float:
        """تطبيق منحنى النسيان الأُسّي (Ebbinghaus-style decay)"""
        hours_elapsed = (now - item.timestamp) / 3600.0
        return math.exp(-self.decay_lambda * hours_elapsed)

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """استرجاع أهم k ذكريات وفق المعادلة المرجّحة الكاملة، وليس التشابه الدلالي فقط"""
        if not self.memories:
            return []

        query_embedding = embedder.encode([query], normalize_embeddings=True)[0]
        now = time.time()

        scored_memories = []
        for item in self.memories.values():
            relevance = float(np.dot(item.embedding, query_embedding))   # cosine (متجهات منظّمة)
            relevance_normalized = (relevance + 1) / 2
            recency = self._recency_score(item, now)
            score = (self.alpha * recency +
                     self.beta * item.importance +
                     self.gamma * relevance_normalized)
            scored_memories.append((score, item))

        scored_memories.sort(key=lambda x: -x[0])
        top_k = scored_memories[:k]

        results = []
        for score, item in top_k:
            item.access_count += 1
            item.last_accessed = now
            results.append({"content": item.content, "score": round(score, 3),
                             "importance": item.importance, "memory_id": item.memory_id})
        return results

    def forget_stale_memories(self):
        """أرشفة/حذف الذكريات التي انخفض مجموع (حداثة+أهمية) عن العتبة ولم تُستدعَ مؤخرًا"""
        now = time.time()
        to_remove = []
        for mid, item in self.memories.items():
            combined = 0.5 * self._recency_score(item, now) + 0.5 * item.importance
            if combined < self.forgetting_threshold and item.access_count == 0:
                to_remove.append(mid)

        for mid in to_remove:
            print(f"🗑️  نسيان الذكرى [id={mid}]: {self.memories[mid].content[:50]}...")
            del self.memories[mid]

        if to_remove:
            self._rebuild_index()  # إعادة بناء الفهرس بعد الحذف (FAISS لا يدعم حذفًا مباشرًا في IndexFlat)

    def _rebuild_index(self):
        self.index = faiss.IndexFlatIP(EMBED_DIM)
        if self.memories:
            embeddings = np.stack([m.embedding for m in self.memories.values()])
            self.index.add(embeddings)

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-10-applications-agents/chapter-57.md
# ────────────────────────────────────────────────────────────────────────
class SummaryMemory:
    """ذاكرة تلخيصية تُحدَّث كلما امتلأ الـ buffer قصير المدى، بحجم شبه ثابت"""

    def __init__(self, max_summary_words: int = 200):
        self.summary = ""
        self.max_summary_words = max_summary_words

    def update(self, new_messages: List[Dict[str, str]]):
        conversation_text = "\n".join(f"{m['role']}: {m['content']}" for m in new_messages)

        system_prompt = f"""أنت نظام ضغط ذاكرة. لديك ملخص سابق للمحادثة ورسائل جديدة.
ادمجهما في ملخص واحد جديد لا يتجاوز {self.max_summary_words} كلمة، يحافظ على:
- الحقائق الثابتة عن المستخدم (تفضيلاته، قراراته، معلوماته الشخصية)
- سياق المهام الجارية غير المكتملة
احذف التفاصيل العابرة وكرر فقط ما له قيمة مستقبلية."""

        user_prompt = f"""الملخص السابق:
{self.summary if self.summary else "(لا يوجد - هذه بداية المحادثة)"}

الرسائل الجديدة:
{conversation_text}

أعد الملخص المحدّث فقط."""

        self.summary = call_llm(system_prompt, user_prompt)
        return self.summary

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-10-applications-agents/chapter-57.md
# ────────────────────────────────────────────────────────────────────────
class ReflectionEngine:
    """
    كل N ذكرى جديدة، نسأل LLM: 'ما الأنماط العليا التي تستنتجها من هذه الذكريات؟'
    الاستنتاجات نفسها تُحفظ كذكريات جديدة بأهمية عالية (تماثل فكرة الورقة الأصلية)
    """

    def __init__(self, memory_store: LongTermMemory, trigger_every: int = 10):
        self.memory_store = memory_store
        self.trigger_every = trigger_every
        self._counter = 0

    def maybe_reflect(self):
        self._counter += 1
        if self._counter < self.trigger_every:
            return None
        self._counter = 0

        recent_memories = sorted(self.memory_store.memories.values(),
                                  key=lambda m: -m.timestamp)[:self.trigger_every]
        memories_text = "\n".join(f"- {m.content}" for m in recent_memories)

        system_prompt = """أنت محرك تأمل لمساعد ذكي طويل المدى. اقرأ مجموعة ذكريات متفرقة
واستنتج 1-2 نمط/استنتاج عام رفيع المستوى يفيد في فهم المستخدم مستقبلًا
(مثال: "يميل المستخدم لجدولة اجتماعات مهمة صباحًا ويتجنب بعد الظهر")."""

        insight = call_llm(system_prompt, f"الذكريات:\n{memories_text}\n\nالاستنتاجات:")

        # الاستنتاج نفسه يُحفظ كذكرى جديدة بأهمية عالية لأنه معرفة مُكثّفة ومُستخلصة
        self.memory_store.add_memory(f"[استنتاج]: {insight}", importance=0.85)
        return insight

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-10-applications-agents/chapter-57.md
# ────────────────────────────────────────────────────────────────────────
class PersistentAssistant:
    """مساعد يدمج: buffer قصير المدى + vector memory + summary + reflection"""

    def __init__(self, buffer_size: int = 6):
        self.buffer_size = buffer_size
        self.short_term_buffer: List[Dict[str, str]] = []
        self.long_term_memory = LongTermMemory(alpha=0.25, beta=0.35, gamma=0.40)
        self.summary_memory = SummaryMemory()
        self.reflection_engine = ReflectionEngine(self.long_term_memory, trigger_every=6)

    def _flush_buffer_to_summary(self):
        """عندما يمتلئ الـ buffer، أقدم الرسائل تُضغط في الملخص وتُفحص لاستخراج ذكريات دائمة"""
        oldest_messages = self.short_term_buffer[:2]  # أقدم رسالتين (دور المستخدم + الوكيل)
        self.short_term_buffer = self.short_term_buffer[2:]

        self.summary_memory.update(oldest_messages)

        # فحص: هل تحتوي هذه الرسائل القديمة على حقيقة تستحق التسجيل الدائم؟
        for msg in oldest_messages:
            if msg["role"] == "user":
                self._maybe_extract_fact(msg["content"])

    def _maybe_extract_fact(self, user_message: str):
        """يستخرج حقائق دائمة (تفضيلات، قيود، معلومات شخصية) من رسالة المستخدم إن وُجدت"""
        prompt = f"""هل تحتوي هذه الرسالة على حقيقة دائمة عن المستخدم تستحق التذكر لاحقًا
(تفضيل، قيد صحي، معلومة شخصية ثابتة)؟ إن وُجدت، أعدها كجملة واحدة موجزة.
إن لم توجد، أعد كلمة "لا" فقط بالضبط.

الرسالة: "{user_message}" """

        extracted = call_llm("أنت مستخرج حقائق دقيق ومحافظ.", prompt, temperature=0.0).strip()
        if extracted and extracted != "لا" and len(extracted) > 3:
            self.long_term_memory.add_memory(extracted)
            self.reflection_engine.maybe_reflect()

    def chat(self, user_message: str) -> str:
        # 1. استرجاع ذكريات ذات صلة بالرسالة الحالية
        relevant_memories = self.long_term_memory.retrieve(user_message, k=3)
        memories_context = "\n".join(f"- {m['content']}" for m in relevant_memories)

        # 2. بناء الـ prompt الكامل: ملخص تراكمي + ذكريات مسترجعة + buffer حالي
        buffer_text = "\n".join(f"{m['role']}: {m['content']}" for m in self.short_term_buffer)

        system_prompt = f"""أنت مساعد شخصي ذو ذاكرة طويلة المدى. استخدم المعلومات التالية
للرد بشكل متّسق ومخصص، دون ذكر صراحة أنك "تستعيد ذاكرة":

[ملخص المحادثات السابقة]:
{self.summary_memory.summary if self.summary_memory.summary else "لا يوجد تاريخ سابق"}

[ذكريات دائمة ذات صلة بالسؤال الحالي]:
{memories_context if memories_context else "لا توجد ذكريات ذات صلة"}

[المحادثة الحالية]:
{buffer_text}"""

        response = call_llm(system_prompt, user_message, model="gpt-4o-mini", temperature=0.7)

        # 3. تحديث الـ buffer
        self.short_term_buffer.append({"role": "user", "content": user_message})
        self.short_term_buffer.append({"role": "assistant", "content": response})

        if len(self.short_term_buffer) > self.buffer_size:
            self._flush_buffer_to_summary()

        return response


# ───────────── محاكاة تفاعل عبر "أيام" متعددة (جلسات منفصلة) ─────────────
if __name__ == "__main__":
    assistant = PersistentAssistant(buffer_size=4)

    print("=== اليوم 1 ===")
    print(assistant.chat("مرحبًا، أنا أحمد. أنا نباتي ولدي حساسية شديدة من المكسرات."))
    print(assistant.chat("أفضّل أن تكون اجتماعاتي المهمة في الصباح دائمًا، أنا لست نشيطًا مساءً."))
    print(assistant.chat("بالمناسبة، ما رأيك في الطقس اليوم؟"))  # رسالة تافهة - يجب ألا تُسجَّل كذكرى دائمة

    # محاكاة مرور وقت طويل (في تطبيق حقيقي: جلسة جديدة بعد أيام)
    print("\n=== اليوم 5 (جلسة جديدة) ===")
    print(assistant.chat("اقترح لي مطعمًا جيدًا لعشاء عمل مهم الليلة الساعة 8."))
    # النظام يجب أن يتذكر: نباتي + حساسية مكسرات (رغم أنه قد يتجاهل تفضيل "الصباح" لأن السياق مختلف)

    print("\n=== فحص الذاكرة طويلة المدى المُخزّنة ===")
    for mem in assistant.long_term_memory.memories.values():
        print(f"  [{mem.memory_id}] (أهمية={mem.importance:.2f}): {mem.content}")

    print("\n=== الملخص التراكمي الحالي ===")
    print(assistant.summary_memory.summary)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-57-chapter-57.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 57: الفصل 57: الذاكرة طويلة المدى للمحادثات (Long-term Memory)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_schema() -> None:
    schema = {'type': 'object', 'required': ['query', 'limit'], 'properties': {'query': str, 'limit': int}}
    arguments = {'query': 'attention', 'limit': 3}
    errors = []
    for key in schema['required']:
        if key not in arguments:
            errors.append(f'missing: {key}')
    for key, expected in schema['properties'].items():
        if key in arguments and (not isinstance(arguments[key], expected) or isinstance(arguments[key], bool)):
            errors.append(f'invalid type: {key}')
    print('arguments:', arguments)
    print('schema validation:', 'PASS' if not errors else errors)

if __name__ == "__main__":
    print("Chapter 57 lab: Validate structured tool arguments before execution")
    run_schema()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-57-chapter-57.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import os
import json
import time
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np

from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBED_DIM = embedder.get_sentence_embedding_dimension()


def call_llm(system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini",
             temperature: float = 0.3) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-57-chapter-57.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
@dataclass
class MemoryItem:
    """ذكرى واحدة: نص + metadata ضرورية لحساب درجة الاسترجاع لاحقًا"""
    content: str
    timestamp: float
    importance: float          # 0-1، تُقدَّر عبر LLM وقت التسجيل
    embedding: np.ndarray
    memory_id: int
    access_count: int = 0      # عدد مرات الاسترجاع (مؤشر فائدة إضافي)
    last_accessed: float = field(default_factory=time.time)


class LongTermMemory:
    """
    ذاكرة دلالية طويلة المدى تُطبّق معادلة الاسترجاع المرجّح:
    score = α·recency + β·importance + γ·relevance
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.3, gamma: float = 0.4,
                 decay_lambda: float = 0.01, forgetting_threshold: float = 0.05):
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.decay_lambda = decay_lambda           # سرعة تلاشي الذاكرة (لكل ساعة)
        self.forgetting_threshold = forgetting_threshold
        self.memories: Dict[int, MemoryItem] = {}
        self.index = faiss.IndexFlatIP(EMBED_DIM)  # Inner Product على متجهات منظّمة = cosine sim
        self._next_id = 0

    def _estimate_importance(self, content: str) -> float:
        """نطلب من LLM تقدير أهمية الذكرى على مقياس 1-10، ثم نطبّعها لـ [0,1]"""
        prompt = f"""قيّم أهمية المعلومة التالية على مقياس من 1 إلى 10 لمساعد شخصي طويل المدى
(10 = معلومة حيوية دائمة مثل حساسية طبية أو تفضيل جوهري، 1 = تفصيل عابر لا قيمة له لاحقًا):

المعلومة: "{content}"

أعد رقمًا فقط بدون أي نص إضافي."""
        try:
            score = float(call_llm("أنت مقيّم دقيق.", prompt, temperature=0.0).strip())
            return max(1.0, min(10.0, score)) / 10.0
        except ValueError:
            return 0.5  # قيمة افتراضية آمنة عند فشل التحليل

    def add_memory(self, content: str, importance: Optional[float] = None) -> int:
        """تسجيل ذكرى جديدة: يحسب embedding وأهمية ويضيفها للفهرس"""
        embedding = embedder.encode([content], normalize_embeddings=True)[0]
        final_importance = importance if importance is not None else self._estimate_importance(content)

        memory_id = self._next_id
        self._next_id += 1

        item = MemoryItem(content=content, timestamp=time.time(), importance=final_importance,
                           embedding=embedding, memory_id=memory_id)
        self.memories[memory_id] = item
        self.index.add(embedding.reshape(1, -1))

        print(f"🧠 ذكرى جديدة [id={memory_id}, أهمية={final_importance:.2f}]: {content[:60]}...")
        return memory_id

    def _recency_score(self, item: MemoryItem, now: float) -> float:
        """تطبيق منحنى النسيان الأُسّي (Ebbinghaus-style decay)"""
        hours_elapsed = (now - item.timestamp) / 3600.0
        return math.exp(-self.decay_lambda * hours_elapsed)

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """استرجاع أهم k ذكريات وفق المعادلة المرجّحة الكاملة، وليس التشابه الدلالي فقط"""
        if not self.memories:
            return []

        query_embedding = embedder.encode([query], normalize_embeddings=True)[0]
        now = time.time()

        scored_memories = []
        for item in self.memories.values():
            relevance = float(np.dot(item.embedding, query_embedding))   # cosine (متجهات منظّمة)
            relevance_normalized = (relevance + 1) / 2
            recency = self._recency_score(item, now)
            score = (self.alpha * recency +
                     self.beta * item.importance +
                     self.gamma * relevance_normalized)
            scored_memories.append((score, item))

        scored_memories.sort(key=lambda x: -x[0])
        top_k = scored_memories[:k]

        results = []
        for score, item in top_k:
            item.access_count += 1
            item.last_accessed = now
            results.append({"content": item.content, "score": round(score, 3),
                             "importance": item.importance, "memory_id": item.memory_id})
        return results

    def forget_stale_memories(self):
        """أرشفة/حذف الذكريات التي انخفض مجموع (حداثة+أهمية) عن العتبة ولم تُستدعَ مؤخرًا"""
        now = time.time()
        to_remove = []
        for mid, item in self.memories.items():
            combined = 0.5 * self._recency_score(item, now) + 0.5 * item.importance
            if combined < self.forgetting_threshold and item.access_count == 0:
                to_remove.append(mid)

        for mid in to_remove:
            print(f"🗑️  نسيان الذكرى [id={mid}]: {self.memories[mid].content[:50]}...")
            del self.memories[mid]

        if to_remove:
            self._rebuild_index()  # إعادة بناء الفهرس بعد الحذف (FAISS لا يدعم حذفًا مباشرًا في IndexFlat)

    def _rebuild_index(self):
        self.index = faiss.IndexFlatIP(EMBED_DIM)
        if self.memories:
            embeddings = np.stack([m.embedding for m in self.memories.values()])
            self.index.add(embeddings)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-57-chapter-57.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class SummaryMemory:
    """ذاكرة تلخيصية تُحدَّث كلما امتلأ الـ buffer قصير المدى، بحجم شبه ثابت"""

    def __init__(self, max_summary_words: int = 200):
        self.summary = ""
        self.max_summary_words = max_summary_words

    def update(self, new_messages: List[Dict[str, str]]):
        conversation_text = "\n".join(f"{m['role']}: {m['content']}" for m in new_messages)

        system_prompt = f"""أنت نظام ضغط ذاكرة. لديك ملخص سابق للمحادثة ورسائل جديدة.
ادمجهما في ملخص واحد جديد لا يتجاوز {self.max_summary_words} كلمة، يحافظ على:
- الحقائق الثابتة عن المستخدم (تفضيلاته، قراراته، معلوماته الشخصية)
- سياق المهام الجارية غير المكتملة
احذف التفاصيل العابرة وكرر فقط ما له قيمة مستقبلية."""

        user_prompt = f"""الملخص السابق:
{self.summary if self.summary else "(لا يوجد - هذه بداية المحادثة)"}

الرسائل الجديدة:
{conversation_text}

أعد الملخص المحدّث فقط."""

        self.summary = call_llm(system_prompt, user_prompt)
        return self.summary

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-57-chapter-57.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class ReflectionEngine:
    """
    كل N ذكرى جديدة، نسأل LLM: 'ما الأنماط العليا التي تستنتجها من هذه الذكريات؟'
    الاستنتاجات نفسها تُحفظ كذكريات جديدة بأهمية عالية (تماثل فكرة الورقة الأصلية)
    """

    def __init__(self, memory_store: LongTermMemory, trigger_every: int = 10):
        self.memory_store = memory_store
        self.trigger_every = trigger_every
        self._counter = 0

    def maybe_reflect(self):
        self._counter += 1
        if self._counter < self.trigger_every:
            return None
        self._counter = 0

        recent_memories = sorted(self.memory_store.memories.values(),
                                  key=lambda m: -m.timestamp)[:self.trigger_every]
        memories_text = "\n".join(f"- {m.content}" for m in recent_memories)

        system_prompt = """أنت محرك تأمل لمساعد ذكي طويل المدى. اقرأ مجموعة ذكريات متفرقة
واستنتج 1-2 نمط/استنتاج عام رفيع المستوى يفيد في فهم المستخدم مستقبلًا
(مثال: "يميل المستخدم لجدولة اجتماعات مهمة صباحًا ويتجنب بعد الظهر")."""

        insight = call_llm(system_prompt, f"الذكريات:\n{memories_text}\n\nالاستنتاجات:")

        # الاستنتاج نفسه يُحفظ كذكرى جديدة بأهمية عالية لأنه معرفة مُكثّفة ومُستخلصة
        self.memory_store.add_memory(f"[استنتاج]: {insight}", importance=0.85)
        return insight

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-57-chapter-57.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
class PersistentAssistant:
    """مساعد يدمج: buffer قصير المدى + vector memory + summary + reflection"""

    def __init__(self, buffer_size: int = 6):
        self.buffer_size = buffer_size
        self.short_term_buffer: List[Dict[str, str]] = []
        self.long_term_memory = LongTermMemory(alpha=0.25, beta=0.35, gamma=0.40)
        self.summary_memory = SummaryMemory()
        self.reflection_engine = ReflectionEngine(self.long_term_memory, trigger_every=6)

    def _flush_buffer_to_summary(self):
        """عندما يمتلئ الـ buffer، أقدم الرسائل تُضغط في الملخص وتُفحص لاستخراج ذكريات دائمة"""
        oldest_messages = self.short_term_buffer[:2]  # أقدم رسالتين (دور المستخدم + الوكيل)
        self.short_term_buffer = self.short_term_buffer[2:]

        self.summary_memory.update(oldest_messages)

        # فحص: هل تحتوي هذه الرسائل القديمة على حقيقة تستحق التسجيل الدائم؟
        for msg in oldest_messages:
            if msg["role"] == "user":
                self._maybe_extract_fact(msg["content"])

    def _maybe_extract_fact(self, user_message: str):
        """يستخرج حقائق دائمة (تفضيلات، قيود، معلومات شخصية) من رسالة المستخدم إن وُجدت"""
        prompt = f"""هل تحتوي هذه الرسالة على حقيقة دائمة عن المستخدم تستحق التذكر لاحقًا
(تفضيل، قيد صحي، معلومة شخصية ثابتة)؟ إن وُجدت، أعدها كجملة واحدة موجزة.
إن لم توجد، أعد كلمة "لا" فقط بالضبط.

الرسالة: "{user_message}" """

        extracted = call_llm("أنت مستخرج حقائق دقيق ومحافظ.", prompt, temperature=0.0).strip()
        if extracted and extracted != "لا" and len(extracted) > 3:
            self.long_term_memory.add_memory(extracted)
            self.reflection_engine.maybe_reflect()

    def chat(self, user_message: str) -> str:
        # 1. استرجاع ذكريات ذات صلة بالرسالة الحالية
        relevant_memories = self.long_term_memory.retrieve(user_message, k=3)
        memories_context = "\n".join(f"- {m['content']}" for m in relevant_memories)

        # 2. بناء الـ prompt الكامل: ملخص تراكمي + ذكريات مسترجعة + buffer حالي
        buffer_text = "\n".join(f"{m['role']}: {m['content']}" for m in self.short_term_buffer)

        system_prompt = f"""أنت مساعد شخصي ذو ذاكرة طويلة المدى. استخدم المعلومات التالية
للرد بشكل متّسق ومخصص، دون ذكر صراحة أنك "تستعيد ذاكرة":

[ملخص المحادثات السابقة]:
{self.summary_memory.summary if self.summary_memory.summary else "لا يوجد تاريخ سابق"}

[ذكريات دائمة ذات صلة بالسؤال الحالي]:
{memories_context if memories_context else "لا توجد ذكريات ذات صلة"}

[المحادثة الحالية]:
{buffer_text}"""

        response = call_llm(system_prompt, user_message, model="gpt-4o-mini", temperature=0.7)

        # 3. تحديث الـ buffer
        self.short_term_buffer.append({"role": "user", "content": user_message})
        self.short_term_buffer.append({"role": "assistant", "content": response})

        if len(self.short_term_buffer) > self.buffer_size:
            self._flush_buffer_to_summary()

        return response


# ───────────── محاكاة تفاعل عبر "أيام" متعددة (جلسات منفصلة) ─────────────
if __name__ == "__main__":
    assistant = PersistentAssistant(buffer_size=4)

    print("=== اليوم 1 ===")
    print(assistant.chat("مرحبًا، أنا أحمد. أنا نباتي ولدي حساسية شديدة من المكسرات."))
    print(assistant.chat("أفضّل أن تكون اجتماعاتي المهمة في الصباح دائمًا، أنا لست نشيطًا مساءً."))
    print(assistant.chat("بالمناسبة، ما رأيك في الطقس اليوم؟"))  # رسالة تافهة - يجب ألا تُسجَّل كذكرى دائمة

    # محاكاة مرور وقت طويل (في تطبيق حقيقي: جلسة جديدة بعد أيام)
    print("\n=== اليوم 5 (جلسة جديدة) ===")
    print(assistant.chat("اقترح لي مطعمًا جيدًا لعشاء عمل مهم الليلة الساعة 8."))
    # النظام يجب أن يتذكر: نباتي + حساسية مكسرات (رغم أنه قد يتجاهل تفضيل "الصباح" لأن السياق مختلف)

    print("\n=== فحص الذاكرة طويلة المدى المُخزّنة ===")
    for mem in assistant.long_term_memory.memories.values():
        print(f"  [{mem.memory_id}] (أهمية={mem.importance:.2f}): {mem.content}")

    print("\n=== الملخص التراكمي الحالي ===")
    print(assistant.summary_memory.summary)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-57-chapter-57.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الذاكرة طويلة المدى للمحادثات (Long-term Memory)
Large Language Model Engineering — Long-term

Book slug : llm-engineering-69-chapters
Chapter   : ch-57-chapter-57 (lesson 57)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_57.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 57: الفصل 57: الذاكرة طويلة المدى للمحادثات (Long-term Memory)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_schema() -> None:
    schema = {'type': 'object', 'required': ['query', 'limit'], 'properties': {'query': str, 'limit': int}}
    arguments = {'query': 'attention', 'limit': 3}
    errors = []
    for key in schema['required']:
        if key not in arguments:
            errors.append(f'missing: {key}')
    for key, expected in schema['properties'].items():
        if key in arguments and (not isinstance(arguments[key], expected) or isinstance(arguments[key], bool)):
            errors.append(f'invalid type: {key}')
    print('arguments:', arguments)
    print('schema validation:', 'PASS' if not errors else errors)

if __name__ == "__main__":
    print("Chapter 57 lab: Validate structured tool arguments before execution")
    run_schema()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-10-applications-agents/chapter-57.md
# ────────────────────────────────────────────────────────────────────────
import os
import json
import time
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np

from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBED_DIM = embedder.get_sentence_embedding_dimension()


def call_llm(system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini",
             temperature: float = 0.3) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-10-applications-agents/chapter-57.md
# ────────────────────────────────────────────────────────────────────────
@dataclass
class MemoryItem:
    """ذكرى واحدة: نص + metadata ضرورية لحساب درجة الاسترجاع لاحقًا"""
    content: str
    timestamp: float
    importance: float          # 0-1، تُقدَّر عبر LLM وقت التسجيل
    embedding: np.ndarray
    memory_id: int
    access_count: int = 0      # عدد مرات الاسترجاع (مؤشر فائدة إضافي)
    last_accessed: float = field(default_factory=time.time)


class LongTermMemory:
    """
    ذاكرة دلالية طويلة المدى تُطبّق معادلة الاسترجاع المرجّح:
    score = α·recency + β·importance + γ·relevance
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.3, gamma: float = 0.4,
                 decay_lambda: float = 0.01, forgetting_threshold: float = 0.05):
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.decay_lambda = decay_lambda           # سرعة تلاشي الذاكرة (لكل ساعة)
        self.forgetting_threshold = forgetting_threshold
        self.memories: Dict[int, MemoryItem] = {}
        self.index = faiss.IndexFlatIP(EMBED_DIM)  # Inner Product على متجهات منظّمة = cosine sim
        self._next_id = 0

    def _estimate_importance(self, content: str) -> float:
        """نطلب من LLM تقدير أهمية الذكرى على مقياس 1-10، ثم نطبّعها لـ [0,1]"""
        prompt = f"""قيّم أهمية المعلومة التالية على مقياس من 1 إلى 10 لمساعد شخصي طويل المدى
(10 = معلومة حيوية دائمة مثل حساسية طبية أو تفضيل جوهري، 1 = تفصيل عابر لا قيمة له لاحقًا):

المعلومة: "{content}"

أعد رقمًا فقط بدون أي نص إضافي."""
        try:
            score = float(call_llm("أنت مقيّم دقيق.", prompt, temperature=0.0).strip())
            return max(1.0, min(10.0, score)) / 10.0
        except ValueError:
            return 0.5  # قيمة افتراضية آمنة عند فشل التحليل

    def add_memory(self, content: str, importance: Optional[float] = None) -> int:
        """تسجيل ذكرى جديدة: يحسب embedding وأهمية ويضيفها للفهرس"""
        embedding = embedder.encode([content], normalize_embeddings=True)[0]
        final_importance = importance if importance is not None else self._estimate_importance(content)

        memory_id = self._next_id
        self._next_id += 1

        item = MemoryItem(content=content, timestamp=time.time(), importance=final_importance,
                           embedding=embedding, memory_id=memory_id)
        self.memories[memory_id] = item
        self.index.add(embedding.reshape(1, -1))

        print(f"🧠 ذكرى جديدة [id={memory_id}, أهمية={final_importance:.2f}]: {content[:60]}...")
        return memory_id

    def _recency_score(self, item: MemoryItem, now: float) -> float:
        """تطبيق منحنى النسيان الأُسّي (Ebbinghaus-style decay)"""
        hours_elapsed = (now - item.timestamp) / 3600.0
        return math.exp(-self.decay_lambda * hours_elapsed)

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """استرجاع أهم k ذكريات وفق المعادلة المرجّحة الكاملة، وليس التشابه الدلالي فقط"""
        if not self.memories:
            return []

        query_embedding = embedder.encode([query], normalize_embeddings=True)[0]
        now = time.time()

        scored_memories = []
        for item in self.memories.values():
            relevance = float(np.dot(item.embedding, query_embedding))   # cosine (متجهات منظّمة)
            relevance_normalized = (relevance + 1) / 2
            recency = self._recency_score(item, now)
            score = (self.alpha * recency +
                     self.beta * item.importance +
                     self.gamma * relevance_normalized)
            scored_memories.append((score, item))

        scored_memories.sort(key=lambda x: -x[0])
        top_k = scored_memories[:k]

        results = []
        for score, item in top_k:
            item.access_count += 1
            item.last_accessed = now
            results.append({"content": item.content, "score": round(score, 3),
                             "importance": item.importance, "memory_id": item.memory_id})
        return results

    def forget_stale_memories(self):
        """أرشفة/حذف الذكريات التي انخفض مجموع (حداثة+أهمية) عن العتبة ولم تُستدعَ مؤخرًا"""
        now = time.time()
        to_remove = []
        for mid, item in self.memories.items():
            combined = 0.5 * self._recency_score(item, now) + 0.5 * item.importance
            if combined < self.forgetting_threshold and item.access_count == 0:
                to_remove.append(mid)

        for mid in to_remove:
            print(f"🗑️  نسيان الذكرى [id={mid}]: {self.memories[mid].content[:50]}...")
            del self.memories[mid]

        if to_remove:
            self._rebuild_index()  # إعادة بناء الفهرس بعد الحذف (FAISS لا يدعم حذفًا مباشرًا في IndexFlat)

    def _rebuild_index(self):
        self.index = faiss.IndexFlatIP(EMBED_DIM)
        if self.memories:
            embeddings = np.stack([m.embedding for m in self.memories.values()])
            self.index.add(embeddings)

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-10-applications-agents/chapter-57.md
# ────────────────────────────────────────────────────────────────────────
class SummaryMemory:
    """ذاكرة تلخيصية تُحدَّث كلما امتلأ الـ buffer قصير المدى، بحجم شبه ثابت"""

    def __init__(self, max_summary_words: int = 200):
        self.summary = ""
        self.max_summary_words = max_summary_words

    def update(self, new_messages: List[Dict[str, str]]):
        conversation_text = "\n".join(f"{m['role']}: {m['content']}" for m in new_messages)

        system_prompt = f"""أنت نظام ضغط ذاكرة. لديك ملخص سابق للمحادثة ورسائل جديدة.
ادمجهما في ملخص واحد جديد لا يتجاوز {self.max_summary_words} كلمة، يحافظ على:
- الحقائق الثابتة عن المستخدم (تفضيلاته، قراراته، معلوماته الشخصية)
- سياق المهام الجارية غير المكتملة
احذف التفاصيل العابرة وكرر فقط ما له قيمة مستقبلية."""

        user_prompt = f"""الملخص السابق:
{self.summary if self.summary else "(لا يوجد - هذه بداية المحادثة)"}

الرسائل الجديدة:
{conversation_text}

أعد الملخص المحدّث فقط."""

        self.summary = call_llm(system_prompt, user_prompt)
        return self.summary

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-10-applications-agents/chapter-57.md
# ────────────────────────────────────────────────────────────────────────
class ReflectionEngine:
    """
    كل N ذكرى جديدة، نسأل LLM: 'ما الأنماط العليا التي تستنتجها من هذه الذكريات؟'
    الاستنتاجات نفسها تُحفظ كذكريات جديدة بأهمية عالية (تماثل فكرة الورقة الأصلية)
    """

    def __init__(self, memory_store: LongTermMemory, trigger_every: int = 10):
        self.memory_store = memory_store
        self.trigger_every = trigger_every
        self._counter = 0

    def maybe_reflect(self):
        self._counter += 1
        if self._counter < self.trigger_every:
            return None
        self._counter = 0

        recent_memories = sorted(self.memory_store.memories.values(),
                                  key=lambda m: -m.timestamp)[:self.trigger_every]
        memories_text = "\n".join(f"- {m.content}" for m in recent_memories)

        system_prompt = """أنت محرك تأمل لمساعد ذكي طويل المدى. اقرأ مجموعة ذكريات متفرقة
واستنتج 1-2 نمط/استنتاج عام رفيع المستوى يفيد في فهم المستخدم مستقبلًا
(مثال: "يميل المستخدم لجدولة اجتماعات مهمة صباحًا ويتجنب بعد الظهر")."""

        insight = call_llm(system_prompt, f"الذكريات:\n{memories_text}\n\nالاستنتاجات:")

        # الاستنتاج نفسه يُحفظ كذكرى جديدة بأهمية عالية لأنه معرفة مُكثّفة ومُستخلصة
        self.memory_store.add_memory(f"[استنتاج]: {insight}", importance=0.85)
        return insight

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-10-applications-agents/chapter-57.md
# ────────────────────────────────────────────────────────────────────────
class PersistentAssistant:
    """مساعد يدمج: buffer قصير المدى + vector memory + summary + reflection"""

    def __init__(self, buffer_size: int = 6):
        self.buffer_size = buffer_size
        self.short_term_buffer: List[Dict[str, str]] = []
        self.long_term_memory = LongTermMemory(alpha=0.25, beta=0.35, gamma=0.40)
        self.summary_memory = SummaryMemory()
        self.reflection_engine = ReflectionEngine(self.long_term_memory, trigger_every=6)

    def _flush_buffer_to_summary(self):
        """عندما يمتلئ الـ buffer، أقدم الرسائل تُضغط في الملخص وتُفحص لاستخراج ذكريات دائمة"""
        oldest_messages = self.short_term_buffer[:2]  # أقدم رسالتين (دور المستخدم + الوكيل)
        self.short_term_buffer = self.short_term_buffer[2:]

        self.summary_memory.update(oldest_messages)

        # فحص: هل تحتوي هذه الرسائل القديمة على حقيقة تستحق التسجيل الدائم؟
        for msg in oldest_messages:
            if msg["role"] == "user":
                self._maybe_extract_fact(msg["content"])

    def _maybe_extract_fact(self, user_message: str):
        """يستخرج حقائق دائمة (تفضيلات، قيود، معلومات شخصية) من رسالة المستخدم إن وُجدت"""
        prompt = f"""هل تحتوي هذه الرسالة على حقيقة دائمة عن المستخدم تستحق التذكر لاحقًا
(تفضيل، قيد صحي، معلومة شخصية ثابتة)؟ إن وُجدت، أعدها كجملة واحدة موجزة.
إن لم توجد، أعد كلمة "لا" فقط بالضبط.

الرسالة: "{user_message}" """

        extracted = call_llm("أنت مستخرج حقائق دقيق ومحافظ.", prompt, temperature=0.0).strip()
        if extracted and extracted != "لا" and len(extracted) > 3:
            self.long_term_memory.add_memory(extracted)
            self.reflection_engine.maybe_reflect()

    def chat(self, user_message: str) -> str:
        # 1. استرجاع ذكريات ذات صلة بالرسالة الحالية
        relevant_memories = self.long_term_memory.retrieve(user_message, k=3)
        memories_context = "\n".join(f"- {m['content']}" for m in relevant_memories)

        # 2. بناء الـ prompt الكامل: ملخص تراكمي + ذكريات مسترجعة + buffer حالي
        buffer_text = "\n".join(f"{m['role']}: {m['content']}" for m in self.short_term_buffer)

        system_prompt = f"""أنت مساعد شخصي ذو ذاكرة طويلة المدى. استخدم المعلومات التالية
للرد بشكل متّسق ومخصص، دون ذكر صراحة أنك "تستعيد ذاكرة":

[ملخص المحادثات السابقة]:
{self.summary_memory.summary if self.summary_memory.summary else "لا يوجد تاريخ سابق"}

[ذكريات دائمة ذات صلة بالسؤال الحالي]:
{memories_context if memories_context else "لا توجد ذكريات ذات صلة"}

[المحادثة الحالية]:
{buffer_text}"""

        response = call_llm(system_prompt, user_message, model="gpt-4o-mini", temperature=0.7)

        # 3. تحديث الـ buffer
        self.short_term_buffer.append({"role": "user", "content": user_message})
        self.short_term_buffer.append({"role": "assistant", "content": response})

        if len(self.short_term_buffer) > self.buffer_size:
            self._flush_buffer_to_summary()

        return response


# ───────────── محاكاة تفاعل عبر "أيام" متعددة (جلسات منفصلة) ─────────────
if __name__ == "__main__":
    assistant = PersistentAssistant(buffer_size=4)

    print("=== اليوم 1 ===")
    print(assistant.chat("مرحبًا، أنا أحمد. أنا نباتي ولدي حساسية شديدة من المكسرات."))
    print(assistant.chat("أفضّل أن تكون اجتماعاتي المهمة في الصباح دائمًا، أنا لست نشيطًا مساءً."))
    print(assistant.chat("بالمناسبة، ما رأيك في الطقس اليوم؟"))  # رسالة تافهة - يجب ألا تُسجَّل كذكرى دائمة

    # محاكاة مرور وقت طويل (في تطبيق حقيقي: جلسة جديدة بعد أيام)
    print("\n=== اليوم 5 (جلسة جديدة) ===")
    print(assistant.chat("اقترح لي مطعمًا جيدًا لعشاء عمل مهم الليلة الساعة 8."))
    # النظام يجب أن يتذكر: نباتي + حساسية مكسرات (رغم أنه قد يتجاهل تفضيل "الصباح" لأن السياق مختلف)

    print("\n=== فحص الذاكرة طويلة المدى المُخزّنة ===")
    for mem in assistant.long_term_memory.memories.values():
        print(f"  [{mem.memory_id}] (أهمية={mem.importance:.2f}): {mem.content}")

    print("\n=== الملخص التراكمي الحالي ===")
    print(assistant.summary_memory.summary)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-57-chapter-57.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 57: الفصل 57: الذاكرة طويلة المدى للمحادثات (Long-term Memory)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_schema() -> None:
    schema = {'type': 'object', 'required': ['query', 'limit'], 'properties': {'query': str, 'limit': int}}
    arguments = {'query': 'attention', 'limit': 3}
    errors = []
    for key in schema['required']:
        if key not in arguments:
            errors.append(f'missing: {key}')
    for key, expected in schema['properties'].items():
        if key in arguments and (not isinstance(arguments[key], expected) or isinstance(arguments[key], bool)):
            errors.append(f'invalid type: {key}')
    print('arguments:', arguments)
    print('schema validation:', 'PASS' if not errors else errors)

if __name__ == "__main__":
    print("Chapter 57 lab: Validate structured tool arguments before execution")
    run_schema()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-57-chapter-57.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import os
import json
import time
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np

from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBED_DIM = embedder.get_sentence_embedding_dimension()


def call_llm(system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini",
             temperature: float = 0.3) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-57-chapter-57.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
@dataclass
class MemoryItem:
    """ذكرى واحدة: نص + metadata ضرورية لحساب درجة الاسترجاع لاحقًا"""
    content: str
    timestamp: float
    importance: float          # 0-1، تُقدَّر عبر LLM وقت التسجيل
    embedding: np.ndarray
    memory_id: int
    access_count: int = 0      # عدد مرات الاسترجاع (مؤشر فائدة إضافي)
    last_accessed: float = field(default_factory=time.time)


class LongTermMemory:
    """
    ذاكرة دلالية طويلة المدى تُطبّق معادلة الاسترجاع المرجّح:
    score = α·recency + β·importance + γ·relevance
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.3, gamma: float = 0.4,
                 decay_lambda: float = 0.01, forgetting_threshold: float = 0.05):
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.decay_lambda = decay_lambda           # سرعة تلاشي الذاكرة (لكل ساعة)
        self.forgetting_threshold = forgetting_threshold
        self.memories: Dict[int, MemoryItem] = {}
        self.index = faiss.IndexFlatIP(EMBED_DIM)  # Inner Product على متجهات منظّمة = cosine sim
        self._next_id = 0

    def _estimate_importance(self, content: str) -> float:
        """نطلب من LLM تقدير أهمية الذكرى على مقياس 1-10، ثم نطبّعها لـ [0,1]"""
        prompt = f"""قيّم أهمية المعلومة التالية على مقياس من 1 إلى 10 لمساعد شخصي طويل المدى
(10 = معلومة حيوية دائمة مثل حساسية طبية أو تفضيل جوهري، 1 = تفصيل عابر لا قيمة له لاحقًا):

المعلومة: "{content}"

أعد رقمًا فقط بدون أي نص إضافي."""
        try:
            score = float(call_llm("أنت مقيّم دقيق.", prompt, temperature=0.0).strip())
            return max(1.0, min(10.0, score)) / 10.0
        except ValueError:
            return 0.5  # قيمة افتراضية آمنة عند فشل التحليل

    def add_memory(self, content: str, importance: Optional[float] = None) -> int:
        """تسجيل ذكرى جديدة: يحسب embedding وأهمية ويضيفها للفهرس"""
        embedding = embedder.encode([content], normalize_embeddings=True)[0]
        final_importance = importance if importance is not None else self._estimate_importance(content)

        memory_id = self._next_id
        self._next_id += 1

        item = MemoryItem(content=content, timestamp=time.time(), importance=final_importance,
                           embedding=embedding, memory_id=memory_id)
        self.memories[memory_id] = item
        self.index.add(embedding.reshape(1, -1))

        print(f"🧠 ذكرى جديدة [id={memory_id}, أهمية={final_importance:.2f}]: {content[:60]}...")
        return memory_id

    def _recency_score(self, item: MemoryItem, now: float) -> float:
        """تطبيق منحنى النسيان الأُسّي (Ebbinghaus-style decay)"""
        hours_elapsed = (now - item.timestamp) / 3600.0
        return math.exp(-self.decay_lambda * hours_elapsed)

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """استرجاع أهم k ذكريات وفق المعادلة المرجّحة الكاملة، وليس التشابه الدلالي فقط"""
        if not self.memories:
            return []

        query_embedding = embedder.encode([query], normalize_embeddings=True)[0]
        now = time.time()

        scored_memories = []
        for item in self.memories.values():
            relevance = float(np.dot(item.embedding, query_embedding))   # cosine (متجهات منظّمة)
            relevance_normalized = (relevance + 1) / 2
            recency = self._recency_score(item, now)
            score = (self.alpha * recency +
                     self.beta * item.importance +
                     self.gamma * relevance_normalized)
            scored_memories.append((score, item))

        scored_memories.sort(key=lambda x: -x[0])
        top_k = scored_memories[:k]

        results = []
        for score, item in top_k:
            item.access_count += 1
            item.last_accessed = now
            results.append({"content": item.content, "score": round(score, 3),
                             "importance": item.importance, "memory_id": item.memory_id})
        return results

    def forget_stale_memories(self):
        """أرشفة/حذف الذكريات التي انخفض مجموع (حداثة+أهمية) عن العتبة ولم تُستدعَ مؤخرًا"""
        now = time.time()
        to_remove = []
        for mid, item in self.memories.items():
            combined = 0.5 * self._recency_score(item, now) + 0.5 * item.importance
            if combined < self.forgetting_threshold and item.access_count == 0:
                to_remove.append(mid)

        for mid in to_remove:
            print(f"🗑️  نسيان الذكرى [id={mid}]: {self.memories[mid].content[:50]}...")
            del self.memories[mid]

        if to_remove:
            self._rebuild_index()  # إعادة بناء الفهرس بعد الحذف (FAISS لا يدعم حذفًا مباشرًا في IndexFlat)

    def _rebuild_index(self):
        self.index = faiss.IndexFlatIP(EMBED_DIM)
        if self.memories:
            embeddings = np.stack([m.embedding for m in self.memories.values()])
            self.index.add(embeddings)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-57-chapter-57.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class SummaryMemory:
    """ذاكرة تلخيصية تُحدَّث كلما امتلأ الـ buffer قصير المدى، بحجم شبه ثابت"""

    def __init__(self, max_summary_words: int = 200):
        self.summary = ""
        self.max_summary_words = max_summary_words

    def update(self, new_messages: List[Dict[str, str]]):
        conversation_text = "\n".join(f"{m['role']}: {m['content']}" for m in new_messages)

        system_prompt = f"""أنت نظام ضغط ذاكرة. لديك ملخص سابق للمحادثة ورسائل جديدة.
ادمجهما في ملخص واحد جديد لا يتجاوز {self.max_summary_words} كلمة، يحافظ على:
- الحقائق الثابتة عن المستخدم (تفضيلاته، قراراته، معلوماته الشخصية)
- سياق المهام الجارية غير المكتملة
احذف التفاصيل العابرة وكرر فقط ما له قيمة مستقبلية."""

        user_prompt = f"""الملخص السابق:
{self.summary if self.summary else "(لا يوجد - هذه بداية المحادثة)"}

الرسائل الجديدة:
{conversation_text}

أعد الملخص المحدّث فقط."""

        self.summary = call_llm(system_prompt, user_prompt)
        return self.summary

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-57-chapter-57.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
class ReflectionEngine:
    """
    كل N ذكرى جديدة، نسأل LLM: 'ما الأنماط العليا التي تستنتجها من هذه الذكريات؟'
    الاستنتاجات نفسها تُحفظ كذكريات جديدة بأهمية عالية (تماثل فكرة الورقة الأصلية)
    """

    def __init__(self, memory_store: LongTermMemory, trigger_every: int = 10):
        self.memory_store = memory_store
        self.trigger_every = trigger_every
        self._counter = 0

    def maybe_reflect(self):
        self._counter += 1
        if self._counter < self.trigger_every:
            return None
        self._counter = 0

        recent_memories = sorted(self.memory_store.memories.values(),
                                  key=lambda m: -m.timestamp)[:self.trigger_every]
        memories_text = "\n".join(f"- {m.content}" for m in recent_memories)

        system_prompt = """أنت محرك تأمل لمساعد ذكي طويل المدى. اقرأ مجموعة ذكريات متفرقة
واستنتج 1-2 نمط/استنتاج عام رفيع المستوى يفيد في فهم المستخدم مستقبلًا
(مثال: "يميل المستخدم لجدولة اجتماعات مهمة صباحًا ويتجنب بعد الظهر")."""

        insight = call_llm(system_prompt, f"الذكريات:\n{memories_text}\n\nالاستنتاجات:")

        # الاستنتاج نفسه يُحفظ كذكرى جديدة بأهمية عالية لأنه معرفة مُكثّفة ومُستخلصة
        self.memory_store.add_memory(f"[استنتاج]: {insight}", importance=0.85)
        return insight

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-57-chapter-57.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
class PersistentAssistant:
    """مساعد يدمج: buffer قصير المدى + vector memory + summary + reflection"""

    def __init__(self, buffer_size: int = 6):
        self.buffer_size = buffer_size
        self.short_term_buffer: List[Dict[str, str]] = []
        self.long_term_memory = LongTermMemory(alpha=0.25, beta=0.35, gamma=0.40)
        self.summary_memory = SummaryMemory()
        self.reflection_engine = ReflectionEngine(self.long_term_memory, trigger_every=6)

    def _flush_buffer_to_summary(self):
        """عندما يمتلئ الـ buffer، أقدم الرسائل تُضغط في الملخص وتُفحص لاستخراج ذكريات دائمة"""
        oldest_messages = self.short_term_buffer[:2]  # أقدم رسالتين (دور المستخدم + الوكيل)
        self.short_term_buffer = self.short_term_buffer[2:]

        self.summary_memory.update(oldest_messages)

        # فحص: هل تحتوي هذه الرسائل القديمة على حقيقة تستحق التسجيل الدائم؟
        for msg in oldest_messages:
            if msg["role"] == "user":
                self._maybe_extract_fact(msg["content"])

    def _maybe_extract_fact(self, user_message: str):
        """يستخرج حقائق دائمة (تفضيلات، قيود، معلومات شخصية) من رسالة المستخدم إن وُجدت"""
        prompt = f"""هل تحتوي هذه الرسالة على حقيقة دائمة عن المستخدم تستحق التذكر لاحقًا
(تفضيل، قيد صحي، معلومة شخصية ثابتة)؟ إن وُجدت، أعدها كجملة واحدة موجزة.
إن لم توجد، أعد كلمة "لا" فقط بالضبط.

الرسالة: "{user_message}" """

        extracted = call_llm("أنت مستخرج حقائق دقيق ومحافظ.", prompt, temperature=0.0).strip()
        if extracted and extracted != "لا" and len(extracted) > 3:
            self.long_term_memory.add_memory(extracted)
            self.reflection_engine.maybe_reflect()

    def chat(self, user_message: str) -> str:
        # 1. استرجاع ذكريات ذات صلة بالرسالة الحالية
        relevant_memories = self.long_term_memory.retrieve(user_message, k=3)
        memories_context = "\n".join(f"- {m['content']}" for m in relevant_memories)

        # 2. بناء الـ prompt الكامل: ملخص تراكمي + ذكريات مسترجعة + buffer حالي
        buffer_text = "\n".join(f"{m['role']}: {m['content']}" for m in self.short_term_buffer)

        system_prompt = f"""أنت مساعد شخصي ذو ذاكرة طويلة المدى. استخدم المعلومات التالية
للرد بشكل متّسق ومخصص، دون ذكر صراحة أنك "تستعيد ذاكرة":

[ملخص المحادثات السابقة]:
{self.summary_memory.summary if self.summary_memory.summary else "لا يوجد تاريخ سابق"}

[ذكريات دائمة ذات صلة بالسؤال الحالي]:
{memories_context if memories_context else "لا توجد ذكريات ذات صلة"}

[المحادثة الحالية]:
{buffer_text}"""

        response = call_llm(system_prompt, user_message, model="gpt-4o-mini", temperature=0.7)

        # 3. تحديث الـ buffer
        self.short_term_buffer.append({"role": "user", "content": user_message})
        self.short_term_buffer.append({"role": "assistant", "content": response})

        if len(self.short_term_buffer) > self.buffer_size:
            self._flush_buffer_to_summary()

        return response


# ───────────── محاكاة تفاعل عبر "أيام" متعددة (جلسات منفصلة) ─────────────
if __name__ == "__main__":
    assistant = PersistentAssistant(buffer_size=4)

    print("=== اليوم 1 ===")
    print(assistant.chat("مرحبًا، أنا أحمد. أنا نباتي ولدي حساسية شديدة من المكسرات."))
    print(assistant.chat("أفضّل أن تكون اجتماعاتي المهمة في الصباح دائمًا، أنا لست نشيطًا مساءً."))
    print(assistant.chat("بالمناسبة، ما رأيك في الطقس اليوم؟"))  # رسالة تافهة - يجب ألا تُسجَّل كذكرى دائمة

    # محاكاة مرور وقت طويل (في تطبيق حقيقي: جلسة جديدة بعد أيام)
    print("\n=== اليوم 5 (جلسة جديدة) ===")
    print(assistant.chat("اقترح لي مطعمًا جيدًا لعشاء عمل مهم الليلة الساعة 8."))
    # النظام يجب أن يتذكر: نباتي + حساسية مكسرات (رغم أنه قد يتجاهل تفضيل "الصباح" لأن السياق مختلف)

    print("\n=== فحص الذاكرة طويلة المدى المُخزّنة ===")
    for mem in assistant.long_term_memory.memories.values():
        print(f"  [{mem.memory_id}] (أهمية={mem.importance:.2f}): {mem.content}")

    print("\n=== الملخص التراكمي الحالي ===")
    print(assistant.summary_memory.summary)
