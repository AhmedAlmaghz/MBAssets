#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — بنية LLMOps — Versioning وCanary Deployment وSemantic Caching
Large Language Model Engineering — LLMOps

Book slug : llm-engineering-69-chapters
Chapter   : ch-61-chapter-61 (lesson 61)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-61-chapter-61.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — بنية LLMOps — Versioning وCanary Deployment وSemantic Caching
Large Language Model Engineering — LLMOps

Book slug : llm-engineering-69-chapters
Chapter   : ch-61-chapter-61 (lesson 61)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-61-chapter-61.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — بنية LLMOps — Versioning وCanary Deployment وSemantic Caching
Large Language Model Engineering — LLMOps

Book slug : llm-engineering-69-chapters
Chapter   : ch-61-chapter-61 (lesson 61)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_61.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 61: الفصل 61: بنية LLMOps — Versioning وCanary Deployment وSemantic Caching."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_llmops() -> None:
    versions = {'stable': {'requests': 1000, 'errors': 12, 'p95_ms': 220}, 'canary': {'requests': 100, 'errors': 1, 'p95_ms': 205}}
    rates = {name: data['errors'] / data['requests'] for name, data in versions.items()}
    allowed_gap = 0.02
    print('error rates:', {key: round(value, 3) for key, value in rates.items()})
    print('canary decision:', 'continue' if rates['canary'] <= rates['stable'] + allowed_gap else 'rollback')
    print('p95 latency:', {key: value['p95_ms'] for key, value in versions.items()})

if __name__ == "__main__":
    print("Chapter 61 lab: Compare canary and stable deployment metrics")
    run_llmops()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-11-deployment-llmops/chapter-61.md
# ────────────────────────────────────────────────────────────────────────
import hashlib
import json
import sqlite3
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional, Dict
from enum import Enum


class DeploymentStage(Enum):
    SHADOW = "shadow"
    CANARY_5 = "canary_5pct"
    CANARY_25 = "canary_25pct"
    CANARY_50 = "canary_50pct"
    FULL = "full_100pct"
    ROLLED_BACK = "rolled_back"


@dataclass
class ModelVersion:
    """وحدة النشر الكاملة — ليست فقط أوزان النموذج بل كل ما يؤثر على المخرجات"""
    deployment_id: str
    model_checkpoint: str          # مثلاً: "llama-3.1-8b-instruct-ft-v3"
    prompt_template_hash: str      # hash لمحتوى system prompt — يكشف أي تغيير ولو بسيط
    config: Dict                   # temperature, max_tokens, top_p, إلخ
    retrieval_index_version: Optional[str] = None  # إصدار فهرس RAG إن وُجد
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    stage: str = DeploymentStage.SHADOW.value
    approved_by: Optional[str] = None


class ModelRegistry:
    """سجل مركزي لكل نسخة تم نشرها، مع تاريخ كامل للمراجعة والتدقيق"""

    def __init__(self, db_path: str = "model_registry.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                deployment_id TEXT PRIMARY KEY,
                model_checkpoint TEXT, prompt_template_hash TEXT,
                config TEXT, retrieval_index_version TEXT,
                created_at TEXT, stage TEXT, approved_by TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS stage_transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                deployment_id TEXT, from_stage TEXT, to_stage TEXT,
                timestamp TEXT, metrics_snapshot TEXT
            )
        """)
        self.conn.commit()

    @staticmethod
    def compute_prompt_hash(prompt_template: str) -> str:
        """أي تغيير ولو حرف واحد في الـ prompt ينتج hash مختلف — يمنع الأخطاء الصامتة"""
        return hashlib.sha256(prompt_template.encode()).hexdigest()[:12]

    def register_version(self, model_checkpoint: str, prompt_template: str,
                          config: Dict, retrieval_index_version: Optional[str] = None) -> ModelVersion:
        prompt_hash = self.compute_prompt_hash(prompt_template)
        deployment_id = f"{model_checkpoint}_{prompt_hash}_{int(datetime.utcnow().timestamp())}"

        version = ModelVersion(
            deployment_id=deployment_id, model_checkpoint=model_checkpoint,
            prompt_template_hash=prompt_hash, config=config,
            retrieval_index_version=retrieval_index_version,
        )

        self.conn.execute(
            "INSERT INTO model_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (version.deployment_id, version.model_checkpoint, version.prompt_template_hash,
             json.dumps(version.config), version.retrieval_index_version,
             version.created_at, version.stage, version.approved_by)
        )
        self.conn.commit()
        print(f"📋 نسخة جديدة مُسجَّلة: {deployment_id}")
        return version

    def transition_stage(self, deployment_id: str, new_stage: DeploymentStage, metrics: Dict):
        """الانتقال بين مراحل النشر، مع تسجيل اللقطة الكاملة للمقاييس وقت القرار"""
        cursor = self.conn.execute(
            "SELECT stage FROM model_versions WHERE deployment_id = ?", (deployment_id,)
        )
        row = cursor.fetchone()
        old_stage = row[0] if row else "unknown"

        self.conn.execute(
            "UPDATE model_versions SET stage = ? WHERE deployment_id = ?",
            (new_stage.value, deployment_id)
        )
        self.conn.execute(
            "INSERT INTO stage_transitions (deployment_id, from_stage, to_stage, timestamp, metrics_snapshot) "
            "VALUES (?, ?, ?, ?, ?)",
            (deployment_id, old_stage, new_stage.value, datetime.utcnow().isoformat(), json.dumps(metrics))
        )
        self.conn.commit()
        print(f"🔄 {deployment_id}: {old_stage} → {new_stage.value} | المقاييس: {metrics}")

    def get_audit_trail(self, deployment_id: str) -> list:
        """سجل تدقيق كامل - ضروري عند التحقيق في مشكلة إنتاجية لاحقًا"""
        cursor = self.conn.execute(
            "SELECT from_stage, to_stage, timestamp, metrics_snapshot FROM stage_transitions "
            "WHERE deployment_id = ? ORDER BY timestamp", (deployment_id,)
        )
        return [{"from": r[0], "to": r[1], "time": r[2], "metrics": json.loads(r[3])} for r in cursor.fetchall()]

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-11-deployment-llmops/chapter-61.md
# ────────────────────────────────────────────────────────────────────────
import random
import hashlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List
import time


@dataclass
class RequestMetrics:
    """المقاييس المُسجَّلة لكل طلب — أساس قرار الترقية أو التراجع"""
    deployment_id: str
    latency_ms: float
    error_occurred: bool
    response_length: int
    user_feedback: int = None  # 1 = إيجابي، -1 = سلبي، None = لم يُقيَّم


class CanaryRouter:
    """
    يوزّع الحركة بين نسخة stable ونسخة canary وفق نسبة مُعدَّة، ويسجّل مقاييس كل مسار
    بشكل حتمي (نفس المستخدم → نفس النسخة دائمًا) لضمان تجربة متّسقة لكل مستخدم
    """

    def __init__(self, stable_handler: Callable, canary_handler: Callable, canary_percentage: float = 0.05):
        self.stable_handler = stable_handler
        self.canary_handler = canary_handler
        self.canary_percentage = canary_percentage
        self.metrics_log: List[RequestMetrics] = []

    def _should_route_to_canary(self, user_id: str) -> bool:
        """
        توزيع حتمي (deterministic) بناءً على hash الـ user_id، وليس عشوائيًا بالكامل في كل طلب —
        هذا يضمن أن نفس المستخدم يبقى في نفس المجموعة طوال فترة التجربة (consistency)
        """
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        bucket = (hash_value % 10000) / 10000.0
        return bucket < self.canary_percentage

    def route(self, user_id: str, query: str, deployment_id_stable: str, deployment_id_canary: str) -> Dict:
        use_canary = self._should_route_to_canary(user_id)
        handler = self.canary_handler if use_canary else self.stable_handler
        deployment_id = deployment_id_canary if use_canary else deployment_id_stable

        start = time.time()
        error_occurred = False
        try:
            response = handler(query)
        except Exception as e:
            response = f"حدث خطأ: {str(e)}"
            error_occurred = True

        latency_ms = (time.time() - start) * 1000

        metric = RequestMetrics(
            deployment_id=deployment_id, latency_ms=latency_ms,
            error_occurred=error_occurred, response_length=len(response),
        )
        self.metrics_log.append(metric)

        return {"response": response, "deployment_id": deployment_id, "is_canary": use_canary}

    def compute_comparison_report(self) -> Dict:
        """تقرير مقارنة إحصائي بين stable وcanary — أساس قرار الترقية للمرحلة التالية"""
        stable_metrics = [m for m in self.metrics_log if not self._is_canary_metric(m)]
        canary_metrics = [m for m in self.metrics_log if self._is_canary_metric(m)]

        def summarize(metrics: List[RequestMetrics]) -> Dict:
            if not metrics:
                return {"count": 0}
            return {
                "count": len(metrics),
                "avg_latency_ms": sum(m.latency_ms for m in metrics) / len(metrics),
                "error_rate": sum(m.error_occurred for m in metrics) / len(metrics),
                "avg_response_length": sum(m.response_length for m in metrics) / len(metrics),
            }

        return {"stable": summarize(stable_metrics), "canary": summarize(canary_metrics)}

    def _is_canary_metric(self, metric: RequestMetrics) -> bool:
        return "canary" in metric.deployment_id.lower()


def should_promote_canary(report: Dict, min_samples: int = 100,
                            max_latency_increase_pct: float = 20.0,
                            max_error_rate_increase: float = 0.02) -> tuple[bool, str]:
    """
    تطبيق معادلة القرار من القسم الرياضي: كل الشروط يجب أن تتحقق معًا
    """
    stable, canary = report["stable"], report["canary"]

    if canary.get("count", 0) < min_samples:
        return False, f"عينة غير كافية ({canary.get('count', 0)} < {min_samples})"

    if stable.get("count", 0) == 0:
        return False, "لا توجد بيانات stable للمقارنة"

    latency_increase_pct = ((canary["avg_latency_ms"] - stable["avg_latency_ms"]) / stable["avg_latency_ms"]) * 100
    if latency_increase_pct > max_latency_increase_pct:
        return False, f"زيادة زمن استجابة مفرطة: {latency_increase_pct:.1f}%"

    error_rate_increase = canary["error_rate"] - stable["error_rate"]
    if error_rate_increase > max_error_rate_increase:
        return False, f"زيادة معدل أخطاء مفرطة: {error_rate_increase*100:.1f} نقطة مئوية"

    return True, "كل المقاييس ضمن الحدود المقبولة ✅"

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-11-deployment-llmops/chapter-61.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import faiss
import time
import json
from dataclasses import dataclass
from typing import Optional, List
from sentence_transformers import SentenceTransformer


@dataclass
class CacheEntry:
    query: str
    response: str
    embedding: np.ndarray
    hit_count: int = 0
    created_at: float = 0.0


class SemanticCache:
    """
    كاش دلالي: يعيد إجابات لأسئلة مشابهة دلاليًا دون نداء LLM جديد
    عتبة التشابه عالية جدًا متعمَّدة لتجنّب إجابات خاطئة لأسئلة مختلفة المعنى
    """

    def __init__(self, similarity_threshold: float = 0.95, max_cache_size: int = 10000):
        self.embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.entries: List[CacheEntry] = []
        self.index = None
        self._dim = self.embedder.get_sentence_embedding_dimension()
        self._rebuild_index()

        # إحصائيات لمراقبة فعالية الكاش (ضرورية لإثبات قيمته أو اكتشاف مشاكله)
        self.total_queries = 0
        self.cache_hits = 0

    def _rebuild_index(self):
        self.index = faiss.IndexFlatIP(self._dim)
        if self.entries:
            embeddings = np.stack([e.embedding for e in self.entries]).astype("float32")
            self.index.add(embeddings)

    def lookup(self, query: str) -> Optional[Dict]:
        """البحث عن سؤال مشابه دلاليًا في الكاش قبل اللجوء لـ LLM"""
        self.total_queries += 1
        if not self.entries:
            return None

        query_embedding = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(query_embedding, k=1)

        top_score, top_idx = scores[0][0], indices[0][0]
        if top_idx != -1 and top_score >= self.threshold:
            matched_entry = self.entries[top_idx]
            matched_entry.hit_count += 1
            self.cache_hits += 1
            return {
                "response": matched_entry.response,
                "matched_query": matched_entry.query,
                "similarity": float(top_score),
                "from_cache": True,
            }
        return None

    def store(self, query: str, response: str):
        """تخزين زوج (سؤال، إجابة) جديد بعد نداء LLM حقيقي"""
        embedding = self.embedder.encode([query], normalize_embeddings=True)[0].astype("float32")
        entry = CacheEntry(query=query, response=response, embedding=embedding, created_at=time.time())
        self.entries.append(entry)

        if len(self.entries) > self.max_cache_size:
            # إزالة الأقل استخدامًا (LFU مبسّط) عند امتلاء الكاش
            self.entries.sort(key=lambda e: e.hit_count)
            self.entries.pop(0)

        self._rebuild_index()

    def get_hit_rate(self) -> float:
        return self.cache_hits / self.total_queries if self.total_queries > 0 else 0.0


class CachedLLMService:
    """تغليف كامل: يتحقق من الكاش أولًا، ويستدعي LLM فقط عند الحاجة الفعلية"""

    def __init__(self, llm_call_fn, cache: SemanticCache):
        self.llm_call_fn = llm_call_fn
        self.cache = cache
        self.total_cost_saved_calls = 0

    def answer(self, query: str) -> Dict:
        cached = self.cache.lookup(query)
        if cached:
            self.total_cost_saved_calls += 1
            print(f"⚡ Cache HIT (تشابه={cached['similarity']:.3f}) مع: '{cached['matched_query']}'")
            return {"answer": cached["response"], "source": "cache", "latency_ms": 2}  # الكاش شبه فوري

        start = time.time()
        response = self.llm_call_fn(query)
        latency_ms = (time.time() - start) * 1000

        self.cache.store(query, response)
        print(f"🔵 Cache MISS → نداء LLM جديد ({latency_ms:.0f}ms)")
        return {"answer": response, "source": "llm", "latency_ms": latency_ms}


# ───────────── تشغيل تجريبي متكامل ─────────────
if __name__ == "__main__":
    def mock_llm_call(query: str) -> str:
        """محاكاة نداء LLM بطيء ومكلف لأغراض العرض"""
        time.sleep(0.5)
        return f"إجابة مُولَّدة فعليًا لسؤال: '{query}'"

    cache = SemanticCache(similarity_threshold=0.90)
    service = CachedLLMService(mock_llm_call, cache)

    queries = [
        "كيف أُعيد ضبط كلمة المرور؟",
        "كيف أرجّع الباسورد تاعي؟",          # إعادة صياغة دلالية قريبة جدًا - يجب أن تضرب الكاش
        "ما هي سياسة استرجاع الأموال؟",      # سؤال مختلف تمامًا - cache miss
        "نسيت كلمة السر، كيف أستعيدها؟",     # قريبة جدًا من السؤال الأول أيضًا
    ]

    for q in queries:
        result = service.answer(q)
        print(f"  → {result['answer'][:50]}... (المصدر: {result['source']})\n")

    print(f"📊 معدل إصابة الكاش الإجمالي: {cache.get_hit_rate()*100:.1f}%")
    print(f"💰 عدد نداءات LLM الموفَّرة: {service.total_cost_saved_calls}")

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-11-deployment-llmops/chapter-61.md
# ────────────────────────────────────────────────────────────────────────
# deployment_workflow.py
"""محاكاة سير عمل كامل: من تسجيل نسخة جديدة حتى الترقية الكاملة أو التراجع"""

registry = ModelRegistry()

# 1. تسجيل النسخة الحالية المستقرة (stable) والنسخة الجديدة (candidate)
stable_version = registry.register_version(
    model_checkpoint="llama-3.1-8b-v2",
    prompt_template="أنت مساعد مفيد ودقيق...",
    config={"temperature": 0.3, "max_tokens": 300},
)

candidate_version = registry.register_version(
    model_checkpoint="llama-3.1-8b-v3-finetuned",
    prompt_template="أنت مساعد مفيد ودقيق ومختصر...",  # تغيير طفيف → hash مختلف تلقائيًا
    config={"temperature": 0.3, "max_tokens": 300},
)

# 2. بدء بمرحلة Canary 5%
router = CanaryRouter(
    stable_handler=lambda q: f"[stable] رد على: {q}",
    canary_handler=lambda q: f"[canary] رد على: {q}",
    canary_percentage=0.05,
)
registry.transition_stage(candidate_version.deployment_id, DeploymentStage.CANARY_5, {"initial": True})

# 3. محاكاة حركة مرور حقيقية (في الإنتاج: هذا يأتي من طلبات المستخدمين الفعلية)
import random
for i in range(500):
    user_id = f"user_{random.randint(1, 200)}"
    router.route(user_id, f"سؤال رقم {i}", stable_version.deployment_id, candidate_version.deployment_id)

# 4. تقييم القرار: هل نرقّي للمرحلة التالية؟
report = router.compute_comparison_report()
can_promote, reason = should_promote_canary(report, min_samples=20)

print(f"\n📊 تقرير المقارنة: {json.dumps(report, indent=2, ensure_ascii=False)}")
print(f"🎯 قرار الترقية: {'✅ تُرقَّى' if can_promote else '❌ لا تُرقَّى'} — السبب: {reason}")

if can_promote:
    registry.transition_stage(candidate_version.deployment_id, DeploymentStage.CANARY_25, report)
else:
    registry.transition_stage(candidate_version.deployment_id, DeploymentStage.ROLLED_BACK, report)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-61-chapter-61.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 61: الفصل 61: بنية LLMOps — Versioning وCanary Deployment وSemantic Caching."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_llmops() -> None:
    versions = {'stable': {'requests': 1000, 'errors': 12, 'p95_ms': 220}, 'canary': {'requests': 100, 'errors': 1, 'p95_ms': 205}}
    rates = {name: data['errors'] / data['requests'] for name, data in versions.items()}
    allowed_gap = 0.02
    print('error rates:', {key: round(value, 3) for key, value in rates.items()})
    print('canary decision:', 'continue' if rates['canary'] <= rates['stable'] + allowed_gap else 'rollback')
    print('p95 latency:', {key: value['p95_ms'] for key, value in versions.items()})

if __name__ == "__main__":
    print("Chapter 61 lab: Compare canary and stable deployment metrics")
    run_llmops()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-61-chapter-61.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import hashlib
import json
import sqlite3
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional, Dict
from enum import Enum


class DeploymentStage(Enum):
    SHADOW = "shadow"
    CANARY_5 = "canary_5pct"
    CANARY_25 = "canary_25pct"
    CANARY_50 = "canary_50pct"
    FULL = "full_100pct"
    ROLLED_BACK = "rolled_back"


@dataclass
class ModelVersion:
    """وحدة النشر الكاملة — ليست فقط أوزان النموذج بل كل ما يؤثر على المخرجات"""
    deployment_id: str
    model_checkpoint: str          # مثلاً: "llama-3.1-8b-instruct-ft-v3"
    prompt_template_hash: str      # hash لمحتوى system prompt — يكشف أي تغيير ولو بسيط
    config: Dict                   # temperature, max_tokens, top_p, إلخ
    retrieval_index_version: Optional[str] = None  # إصدار فهرس RAG إن وُجد
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    stage: str = DeploymentStage.SHADOW.value
    approved_by: Optional[str] = None


class ModelRegistry:
    """سجل مركزي لكل نسخة تم نشرها، مع تاريخ كامل للمراجعة والتدقيق"""

    def __init__(self, db_path: str = "model_registry.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                deployment_id TEXT PRIMARY KEY,
                model_checkpoint TEXT, prompt_template_hash TEXT,
                config TEXT, retrieval_index_version TEXT,
                created_at TEXT, stage TEXT, approved_by TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS stage_transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                deployment_id TEXT, from_stage TEXT, to_stage TEXT,
                timestamp TEXT, metrics_snapshot TEXT
            )
        """)
        self.conn.commit()

    @staticmethod
    def compute_prompt_hash(prompt_template: str) -> str:
        """أي تغيير ولو حرف واحد في الـ prompt ينتج hash مختلف — يمنع الأخطاء الصامتة"""
        return hashlib.sha256(prompt_template.encode()).hexdigest()[:12]

    def register_version(self, model_checkpoint: str, prompt_template: str,
                          config: Dict, retrieval_index_version: Optional[str] = None) -> ModelVersion:
        prompt_hash = self.compute_prompt_hash(prompt_template)
        deployment_id = f"{model_checkpoint}_{prompt_hash}_{int(datetime.utcnow().timestamp())}"

        version = ModelVersion(
            deployment_id=deployment_id, model_checkpoint=model_checkpoint,
            prompt_template_hash=prompt_hash, config=config,
            retrieval_index_version=retrieval_index_version,
        )

        self.conn.execute(
            "INSERT INTO model_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (version.deployment_id, version.model_checkpoint, version.prompt_template_hash,
             json.dumps(version.config), version.retrieval_index_version,
             version.created_at, version.stage, version.approved_by)
        )
        self.conn.commit()
        print(f"📋 نسخة جديدة مُسجَّلة: {deployment_id}")
        return version

    def transition_stage(self, deployment_id: str, new_stage: DeploymentStage, metrics: Dict):
        """الانتقال بين مراحل النشر، مع تسجيل اللقطة الكاملة للمقاييس وقت القرار"""
        cursor = self.conn.execute(
            "SELECT stage FROM model_versions WHERE deployment_id = ?", (deployment_id,)
        )
        row = cursor.fetchone()
        old_stage = row[0] if row else "unknown"

        self.conn.execute(
            "UPDATE model_versions SET stage = ? WHERE deployment_id = ?",
            (new_stage.value, deployment_id)
        )
        self.conn.execute(
            "INSERT INTO stage_transitions (deployment_id, from_stage, to_stage, timestamp, metrics_snapshot) "
            "VALUES (?, ?, ?, ?, ?)",
            (deployment_id, old_stage, new_stage.value, datetime.utcnow().isoformat(), json.dumps(metrics))
        )
        self.conn.commit()
        print(f"🔄 {deployment_id}: {old_stage} → {new_stage.value} | المقاييس: {metrics}")

    def get_audit_trail(self, deployment_id: str) -> list:
        """سجل تدقيق كامل - ضروري عند التحقيق في مشكلة إنتاجية لاحقًا"""
        cursor = self.conn.execute(
            "SELECT from_stage, to_stage, timestamp, metrics_snapshot FROM stage_transitions "
            "WHERE deployment_id = ? ORDER BY timestamp", (deployment_id,)
        )
        return [{"from": r[0], "to": r[1], "time": r[2], "metrics": json.loads(r[3])} for r in cursor.fetchall()]

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-61-chapter-61.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import random
import hashlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List
import time


@dataclass
class RequestMetrics:
    """المقاييس المُسجَّلة لكل طلب — أساس قرار الترقية أو التراجع"""
    deployment_id: str
    latency_ms: float
    error_occurred: bool
    response_length: int
    user_feedback: int = None  # 1 = إيجابي، -1 = سلبي، None = لم يُقيَّم


class CanaryRouter:
    """
    يوزّع الحركة بين نسخة stable ونسخة canary وفق نسبة مُعدَّة، ويسجّل مقاييس كل مسار
    بشكل حتمي (نفس المستخدم → نفس النسخة دائمًا) لضمان تجربة متّسقة لكل مستخدم
    """

    def __init__(self, stable_handler: Callable, canary_handler: Callable, canary_percentage: float = 0.05):
        self.stable_handler = stable_handler
        self.canary_handler = canary_handler
        self.canary_percentage = canary_percentage
        self.metrics_log: List[RequestMetrics] = []

    def _should_route_to_canary(self, user_id: str) -> bool:
        """
        توزيع حتمي (deterministic) بناءً على hash الـ user_id، وليس عشوائيًا بالكامل في كل طلب —
        هذا يضمن أن نفس المستخدم يبقى في نفس المجموعة طوال فترة التجربة (consistency)
        """
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        bucket = (hash_value % 10000) / 10000.0
        return bucket < self.canary_percentage

    def route(self, user_id: str, query: str, deployment_id_stable: str, deployment_id_canary: str) -> Dict:
        use_canary = self._should_route_to_canary(user_id)
        handler = self.canary_handler if use_canary else self.stable_handler
        deployment_id = deployment_id_canary if use_canary else deployment_id_stable

        start = time.time()
        error_occurred = False
        try:
            response = handler(query)
        except Exception as e:
            response = f"حدث خطأ: {str(e)}"
            error_occurred = True

        latency_ms = (time.time() - start) * 1000

        metric = RequestMetrics(
            deployment_id=deployment_id, latency_ms=latency_ms,
            error_occurred=error_occurred, response_length=len(response),
        )
        self.metrics_log.append(metric)

        return {"response": response, "deployment_id": deployment_id, "is_canary": use_canary}

    def compute_comparison_report(self) -> Dict:
        """تقرير مقارنة إحصائي بين stable وcanary — أساس قرار الترقية للمرحلة التالية"""
        stable_metrics = [m for m in self.metrics_log if not self._is_canary_metric(m)]
        canary_metrics = [m for m in self.metrics_log if self._is_canary_metric(m)]

        def summarize(metrics: List[RequestMetrics]) -> Dict:
            if not metrics:
                return {"count": 0}
            return {
                "count": len(metrics),
                "avg_latency_ms": sum(m.latency_ms for m in metrics) / len(metrics),
                "error_rate": sum(m.error_occurred for m in metrics) / len(metrics),
                "avg_response_length": sum(m.response_length for m in metrics) / len(metrics),
            }

        return {"stable": summarize(stable_metrics), "canary": summarize(canary_metrics)}

    def _is_canary_metric(self, metric: RequestMetrics) -> bool:
        return "canary" in metric.deployment_id.lower()


def should_promote_canary(report: Dict, min_samples: int = 100,
                            max_latency_increase_pct: float = 20.0,
                            max_error_rate_increase: float = 0.02) -> tuple[bool, str]:
    """
    تطبيق معادلة القرار من القسم الرياضي: كل الشروط يجب أن تتحقق معًا
    """
    stable, canary = report["stable"], report["canary"]

    if canary.get("count", 0) < min_samples:
        return False, f"عينة غير كافية ({canary.get('count', 0)} < {min_samples})"

    if stable.get("count", 0) == 0:
        return False, "لا توجد بيانات stable للمقارنة"

    latency_increase_pct = ((canary["avg_latency_ms"] - stable["avg_latency_ms"]) / stable["avg_latency_ms"]) * 100
    if latency_increase_pct > max_latency_increase_pct:
        return False, f"زيادة زمن استجابة مفرطة: {latency_increase_pct:.1f}%"

    error_rate_increase = canary["error_rate"] - stable["error_rate"]
    if error_rate_increase > max_error_rate_increase:
        return False, f"زيادة معدل أخطاء مفرطة: {error_rate_increase*100:.1f} نقطة مئوية"

    return True, "كل المقاييس ضمن الحدود المقبولة ✅"

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-61-chapter-61.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import faiss
import time
import json
from dataclasses import dataclass
from typing import Optional, List
from sentence_transformers import SentenceTransformer


@dataclass
class CacheEntry:
    query: str
    response: str
    embedding: np.ndarray
    hit_count: int = 0
    created_at: float = 0.0


class SemanticCache:
    """
    كاش دلالي: يعيد إجابات لأسئلة مشابهة دلاليًا دون نداء LLM جديد
    عتبة التشابه عالية جدًا متعمَّدة لتجنّب إجابات خاطئة لأسئلة مختلفة المعنى
    """

    def __init__(self, similarity_threshold: float = 0.95, max_cache_size: int = 10000):
        self.embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.entries: List[CacheEntry] = []
        self.index = None
        self._dim = self.embedder.get_sentence_embedding_dimension()
        self._rebuild_index()

        # إحصائيات لمراقبة فعالية الكاش (ضرورية لإثبات قيمته أو اكتشاف مشاكله)
        self.total_queries = 0
        self.cache_hits = 0

    def _rebuild_index(self):
        self.index = faiss.IndexFlatIP(self._dim)
        if self.entries:
            embeddings = np.stack([e.embedding for e in self.entries]).astype("float32")
            self.index.add(embeddings)

    def lookup(self, query: str) -> Optional[Dict]:
        """البحث عن سؤال مشابه دلاليًا في الكاش قبل اللجوء لـ LLM"""
        self.total_queries += 1
        if not self.entries:
            return None

        query_embedding = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(query_embedding, k=1)

        top_score, top_idx = scores[0][0], indices[0][0]
        if top_idx != -1 and top_score >= self.threshold:
            matched_entry = self.entries[top_idx]
            matched_entry.hit_count += 1
            self.cache_hits += 1
            return {
                "response": matched_entry.response,
                "matched_query": matched_entry.query,
                "similarity": float(top_score),
                "from_cache": True,
            }
        return None

    def store(self, query: str, response: str):
        """تخزين زوج (سؤال، إجابة) جديد بعد نداء LLM حقيقي"""
        embedding = self.embedder.encode([query], normalize_embeddings=True)[0].astype("float32")
        entry = CacheEntry(query=query, response=response, embedding=embedding, created_at=time.time())
        self.entries.append(entry)

        if len(self.entries) > self.max_cache_size:
            # إزالة الأقل استخدامًا (LFU مبسّط) عند امتلاء الكاش
            self.entries.sort(key=lambda e: e.hit_count)
            self.entries.pop(0)

        self._rebuild_index()

    def get_hit_rate(self) -> float:
        return self.cache_hits / self.total_queries if self.total_queries > 0 else 0.0


class CachedLLMService:
    """تغليف كامل: يتحقق من الكاش أولًا، ويستدعي LLM فقط عند الحاجة الفعلية"""

    def __init__(self, llm_call_fn, cache: SemanticCache):
        self.llm_call_fn = llm_call_fn
        self.cache = cache
        self.total_cost_saved_calls = 0

    def answer(self, query: str) -> Dict:
        cached = self.cache.lookup(query)
        if cached:
            self.total_cost_saved_calls += 1
            print(f"⚡ Cache HIT (تشابه={cached['similarity']:.3f}) مع: '{cached['matched_query']}'")
            return {"answer": cached["response"], "source": "cache", "latency_ms": 2}  # الكاش شبه فوري

        start = time.time()
        response = self.llm_call_fn(query)
        latency_ms = (time.time() - start) * 1000

        self.cache.store(query, response)
        print(f"🔵 Cache MISS → نداء LLM جديد ({latency_ms:.0f}ms)")
        return {"answer": response, "source": "llm", "latency_ms": latency_ms}


# ───────────── تشغيل تجريبي متكامل ─────────────
if __name__ == "__main__":
    def mock_llm_call(query: str) -> str:
        """محاكاة نداء LLM بطيء ومكلف لأغراض العرض"""
        time.sleep(0.5)
        return f"إجابة مُولَّدة فعليًا لسؤال: '{query}'"

    cache = SemanticCache(similarity_threshold=0.90)
    service = CachedLLMService(mock_llm_call, cache)

    queries = [
        "كيف أُعيد ضبط كلمة المرور؟",
        "كيف أرجّع الباسورد تاعي؟",          # إعادة صياغة دلالية قريبة جدًا - يجب أن تضرب الكاش
        "ما هي سياسة استرجاع الأموال؟",      # سؤال مختلف تمامًا - cache miss
        "نسيت كلمة السر، كيف أستعيدها؟",     # قريبة جدًا من السؤال الأول أيضًا
    ]

    for q in queries:
        result = service.answer(q)
        print(f"  → {result['answer'][:50]}... (المصدر: {result['source']})\n")

    print(f"📊 معدل إصابة الكاش الإجمالي: {cache.get_hit_rate()*100:.1f}%")
    print(f"💰 عدد نداءات LLM الموفَّرة: {service.total_cost_saved_calls}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-61-chapter-61.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# deployment_workflow.py
"""محاكاة سير عمل كامل: من تسجيل نسخة جديدة حتى الترقية الكاملة أو التراجع"""

registry = ModelRegistry()

# 1. تسجيل النسخة الحالية المستقرة (stable) والنسخة الجديدة (candidate)
stable_version = registry.register_version(
    model_checkpoint="llama-3.1-8b-v2",
    prompt_template="أنت مساعد مفيد ودقيق...",
    config={"temperature": 0.3, "max_tokens": 300},
)

candidate_version = registry.register_version(
    model_checkpoint="llama-3.1-8b-v3-finetuned",
    prompt_template="أنت مساعد مفيد ودقيق ومختصر...",  # تغيير طفيف → hash مختلف تلقائيًا
    config={"temperature": 0.3, "max_tokens": 300},
)

# 2. بدء بمرحلة Canary 5%
router = CanaryRouter(
    stable_handler=lambda q: f"[stable] رد على: {q}",
    canary_handler=lambda q: f"[canary] رد على: {q}",
    canary_percentage=0.05,
)
registry.transition_stage(candidate_version.deployment_id, DeploymentStage.CANARY_5, {"initial": True})

# 3. محاكاة حركة مرور حقيقية (في الإنتاج: هذا يأتي من طلبات المستخدمين الفعلية)
import random
for i in range(500):
    user_id = f"user_{random.randint(1, 200)}"
    router.route(user_id, f"سؤال رقم {i}", stable_version.deployment_id, candidate_version.deployment_id)

# 4. تقييم القرار: هل نرقّي للمرحلة التالية؟
report = router.compute_comparison_report()
can_promote, reason = should_promote_canary(report, min_samples=20)

print(f"\n📊 تقرير المقارنة: {json.dumps(report, indent=2, ensure_ascii=False)}")
print(f"🎯 قرار الترقية: {'✅ تُرقَّى' if can_promote else '❌ لا تُرقَّى'} — السبب: {reason}")

if can_promote:
    registry.transition_stage(candidate_version.deployment_id, DeploymentStage.CANARY_25, report)
else:
    registry.transition_stage(candidate_version.deployment_id, DeploymentStage.ROLLED_BACK, report)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-61-chapter-61.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — بنية LLMOps — Versioning وCanary Deployment وSemantic Caching
Large Language Model Engineering — LLMOps

Book slug : llm-engineering-69-chapters
Chapter   : ch-61-chapter-61 (lesson 61)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_61.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 61: الفصل 61: بنية LLMOps — Versioning وCanary Deployment وSemantic Caching."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_llmops() -> None:
    versions = {'stable': {'requests': 1000, 'errors': 12, 'p95_ms': 220}, 'canary': {'requests': 100, 'errors': 1, 'p95_ms': 205}}
    rates = {name: data['errors'] / data['requests'] for name, data in versions.items()}
    allowed_gap = 0.02
    print('error rates:', {key: round(value, 3) for key, value in rates.items()})
    print('canary decision:', 'continue' if rates['canary'] <= rates['stable'] + allowed_gap else 'rollback')
    print('p95 latency:', {key: value['p95_ms'] for key, value in versions.items()})

if __name__ == "__main__":
    print("Chapter 61 lab: Compare canary and stable deployment metrics")
    run_llmops()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-11-deployment-llmops/chapter-61.md
# ────────────────────────────────────────────────────────────────────────
import hashlib
import json
import sqlite3
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional, Dict
from enum import Enum


class DeploymentStage(Enum):
    SHADOW = "shadow"
    CANARY_5 = "canary_5pct"
    CANARY_25 = "canary_25pct"
    CANARY_50 = "canary_50pct"
    FULL = "full_100pct"
    ROLLED_BACK = "rolled_back"


@dataclass
class ModelVersion:
    """وحدة النشر الكاملة — ليست فقط أوزان النموذج بل كل ما يؤثر على المخرجات"""
    deployment_id: str
    model_checkpoint: str          # مثلاً: "llama-3.1-8b-instruct-ft-v3"
    prompt_template_hash: str      # hash لمحتوى system prompt — يكشف أي تغيير ولو بسيط
    config: Dict                   # temperature, max_tokens, top_p, إلخ
    retrieval_index_version: Optional[str] = None  # إصدار فهرس RAG إن وُجد
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    stage: str = DeploymentStage.SHADOW.value
    approved_by: Optional[str] = None


class ModelRegistry:
    """سجل مركزي لكل نسخة تم نشرها، مع تاريخ كامل للمراجعة والتدقيق"""

    def __init__(self, db_path: str = "model_registry.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                deployment_id TEXT PRIMARY KEY,
                model_checkpoint TEXT, prompt_template_hash TEXT,
                config TEXT, retrieval_index_version TEXT,
                created_at TEXT, stage TEXT, approved_by TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS stage_transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                deployment_id TEXT, from_stage TEXT, to_stage TEXT,
                timestamp TEXT, metrics_snapshot TEXT
            )
        """)
        self.conn.commit()

    @staticmethod
    def compute_prompt_hash(prompt_template: str) -> str:
        """أي تغيير ولو حرف واحد في الـ prompt ينتج hash مختلف — يمنع الأخطاء الصامتة"""
        return hashlib.sha256(prompt_template.encode()).hexdigest()[:12]

    def register_version(self, model_checkpoint: str, prompt_template: str,
                          config: Dict, retrieval_index_version: Optional[str] = None) -> ModelVersion:
        prompt_hash = self.compute_prompt_hash(prompt_template)
        deployment_id = f"{model_checkpoint}_{prompt_hash}_{int(datetime.utcnow().timestamp())}"

        version = ModelVersion(
            deployment_id=deployment_id, model_checkpoint=model_checkpoint,
            prompt_template_hash=prompt_hash, config=config,
            retrieval_index_version=retrieval_index_version,
        )

        self.conn.execute(
            "INSERT INTO model_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (version.deployment_id, version.model_checkpoint, version.prompt_template_hash,
             json.dumps(version.config), version.retrieval_index_version,
             version.created_at, version.stage, version.approved_by)
        )
        self.conn.commit()
        print(f"📋 نسخة جديدة مُسجَّلة: {deployment_id}")
        return version

    def transition_stage(self, deployment_id: str, new_stage: DeploymentStage, metrics: Dict):
        """الانتقال بين مراحل النشر، مع تسجيل اللقطة الكاملة للمقاييس وقت القرار"""
        cursor = self.conn.execute(
            "SELECT stage FROM model_versions WHERE deployment_id = ?", (deployment_id,)
        )
        row = cursor.fetchone()
        old_stage = row[0] if row else "unknown"

        self.conn.execute(
            "UPDATE model_versions SET stage = ? WHERE deployment_id = ?",
            (new_stage.value, deployment_id)
        )
        self.conn.execute(
            "INSERT INTO stage_transitions (deployment_id, from_stage, to_stage, timestamp, metrics_snapshot) "
            "VALUES (?, ?, ?, ?, ?)",
            (deployment_id, old_stage, new_stage.value, datetime.utcnow().isoformat(), json.dumps(metrics))
        )
        self.conn.commit()
        print(f"🔄 {deployment_id}: {old_stage} → {new_stage.value} | المقاييس: {metrics}")

    def get_audit_trail(self, deployment_id: str) -> list:
        """سجل تدقيق كامل - ضروري عند التحقيق في مشكلة إنتاجية لاحقًا"""
        cursor = self.conn.execute(
            "SELECT from_stage, to_stage, timestamp, metrics_snapshot FROM stage_transitions "
            "WHERE deployment_id = ? ORDER BY timestamp", (deployment_id,)
        )
        return [{"from": r[0], "to": r[1], "time": r[2], "metrics": json.loads(r[3])} for r in cursor.fetchall()]

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-11-deployment-llmops/chapter-61.md
# ────────────────────────────────────────────────────────────────────────
import random
import hashlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List
import time


@dataclass
class RequestMetrics:
    """المقاييس المُسجَّلة لكل طلب — أساس قرار الترقية أو التراجع"""
    deployment_id: str
    latency_ms: float
    error_occurred: bool
    response_length: int
    user_feedback: int = None  # 1 = إيجابي، -1 = سلبي، None = لم يُقيَّم


class CanaryRouter:
    """
    يوزّع الحركة بين نسخة stable ونسخة canary وفق نسبة مُعدَّة، ويسجّل مقاييس كل مسار
    بشكل حتمي (نفس المستخدم → نفس النسخة دائمًا) لضمان تجربة متّسقة لكل مستخدم
    """

    def __init__(self, stable_handler: Callable, canary_handler: Callable, canary_percentage: float = 0.05):
        self.stable_handler = stable_handler
        self.canary_handler = canary_handler
        self.canary_percentage = canary_percentage
        self.metrics_log: List[RequestMetrics] = []

    def _should_route_to_canary(self, user_id: str) -> bool:
        """
        توزيع حتمي (deterministic) بناءً على hash الـ user_id، وليس عشوائيًا بالكامل في كل طلب —
        هذا يضمن أن نفس المستخدم يبقى في نفس المجموعة طوال فترة التجربة (consistency)
        """
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        bucket = (hash_value % 10000) / 10000.0
        return bucket < self.canary_percentage

    def route(self, user_id: str, query: str, deployment_id_stable: str, deployment_id_canary: str) -> Dict:
        use_canary = self._should_route_to_canary(user_id)
        handler = self.canary_handler if use_canary else self.stable_handler
        deployment_id = deployment_id_canary if use_canary else deployment_id_stable

        start = time.time()
        error_occurred = False
        try:
            response = handler(query)
        except Exception as e:
            response = f"حدث خطأ: {str(e)}"
            error_occurred = True

        latency_ms = (time.time() - start) * 1000

        metric = RequestMetrics(
            deployment_id=deployment_id, latency_ms=latency_ms,
            error_occurred=error_occurred, response_length=len(response),
        )
        self.metrics_log.append(metric)

        return {"response": response, "deployment_id": deployment_id, "is_canary": use_canary}

    def compute_comparison_report(self) -> Dict:
        """تقرير مقارنة إحصائي بين stable وcanary — أساس قرار الترقية للمرحلة التالية"""
        stable_metrics = [m for m in self.metrics_log if not self._is_canary_metric(m)]
        canary_metrics = [m for m in self.metrics_log if self._is_canary_metric(m)]

        def summarize(metrics: List[RequestMetrics]) -> Dict:
            if not metrics:
                return {"count": 0}
            return {
                "count": len(metrics),
                "avg_latency_ms": sum(m.latency_ms for m in metrics) / len(metrics),
                "error_rate": sum(m.error_occurred for m in metrics) / len(metrics),
                "avg_response_length": sum(m.response_length for m in metrics) / len(metrics),
            }

        return {"stable": summarize(stable_metrics), "canary": summarize(canary_metrics)}

    def _is_canary_metric(self, metric: RequestMetrics) -> bool:
        return "canary" in metric.deployment_id.lower()


def should_promote_canary(report: Dict, min_samples: int = 100,
                            max_latency_increase_pct: float = 20.0,
                            max_error_rate_increase: float = 0.02) -> tuple[bool, str]:
    """
    تطبيق معادلة القرار من القسم الرياضي: كل الشروط يجب أن تتحقق معًا
    """
    stable, canary = report["stable"], report["canary"]

    if canary.get("count", 0) < min_samples:
        return False, f"عينة غير كافية ({canary.get('count', 0)} < {min_samples})"

    if stable.get("count", 0) == 0:
        return False, "لا توجد بيانات stable للمقارنة"

    latency_increase_pct = ((canary["avg_latency_ms"] - stable["avg_latency_ms"]) / stable["avg_latency_ms"]) * 100
    if latency_increase_pct > max_latency_increase_pct:
        return False, f"زيادة زمن استجابة مفرطة: {latency_increase_pct:.1f}%"

    error_rate_increase = canary["error_rate"] - stable["error_rate"]
    if error_rate_increase > max_error_rate_increase:
        return False, f"زيادة معدل أخطاء مفرطة: {error_rate_increase*100:.1f} نقطة مئوية"

    return True, "كل المقاييس ضمن الحدود المقبولة ✅"

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-11-deployment-llmops/chapter-61.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import faiss
import time
import json
from dataclasses import dataclass
from typing import Optional, List
from sentence_transformers import SentenceTransformer


@dataclass
class CacheEntry:
    query: str
    response: str
    embedding: np.ndarray
    hit_count: int = 0
    created_at: float = 0.0


class SemanticCache:
    """
    كاش دلالي: يعيد إجابات لأسئلة مشابهة دلاليًا دون نداء LLM جديد
    عتبة التشابه عالية جدًا متعمَّدة لتجنّب إجابات خاطئة لأسئلة مختلفة المعنى
    """

    def __init__(self, similarity_threshold: float = 0.95, max_cache_size: int = 10000):
        self.embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.entries: List[CacheEntry] = []
        self.index = None
        self._dim = self.embedder.get_sentence_embedding_dimension()
        self._rebuild_index()

        # إحصائيات لمراقبة فعالية الكاش (ضرورية لإثبات قيمته أو اكتشاف مشاكله)
        self.total_queries = 0
        self.cache_hits = 0

    def _rebuild_index(self):
        self.index = faiss.IndexFlatIP(self._dim)
        if self.entries:
            embeddings = np.stack([e.embedding for e in self.entries]).astype("float32")
            self.index.add(embeddings)

    def lookup(self, query: str) -> Optional[Dict]:
        """البحث عن سؤال مشابه دلاليًا في الكاش قبل اللجوء لـ LLM"""
        self.total_queries += 1
        if not self.entries:
            return None

        query_embedding = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(query_embedding, k=1)

        top_score, top_idx = scores[0][0], indices[0][0]
        if top_idx != -1 and top_score >= self.threshold:
            matched_entry = self.entries[top_idx]
            matched_entry.hit_count += 1
            self.cache_hits += 1
            return {
                "response": matched_entry.response,
                "matched_query": matched_entry.query,
                "similarity": float(top_score),
                "from_cache": True,
            }
        return None

    def store(self, query: str, response: str):
        """تخزين زوج (سؤال، إجابة) جديد بعد نداء LLM حقيقي"""
        embedding = self.embedder.encode([query], normalize_embeddings=True)[0].astype("float32")
        entry = CacheEntry(query=query, response=response, embedding=embedding, created_at=time.time())
        self.entries.append(entry)

        if len(self.entries) > self.max_cache_size:
            # إزالة الأقل استخدامًا (LFU مبسّط) عند امتلاء الكاش
            self.entries.sort(key=lambda e: e.hit_count)
            self.entries.pop(0)

        self._rebuild_index()

    def get_hit_rate(self) -> float:
        return self.cache_hits / self.total_queries if self.total_queries > 0 else 0.0


class CachedLLMService:
    """تغليف كامل: يتحقق من الكاش أولًا، ويستدعي LLM فقط عند الحاجة الفعلية"""

    def __init__(self, llm_call_fn, cache: SemanticCache):
        self.llm_call_fn = llm_call_fn
        self.cache = cache
        self.total_cost_saved_calls = 0

    def answer(self, query: str) -> Dict:
        cached = self.cache.lookup(query)
        if cached:
            self.total_cost_saved_calls += 1
            print(f"⚡ Cache HIT (تشابه={cached['similarity']:.3f}) مع: '{cached['matched_query']}'")
            return {"answer": cached["response"], "source": "cache", "latency_ms": 2}  # الكاش شبه فوري

        start = time.time()
        response = self.llm_call_fn(query)
        latency_ms = (time.time() - start) * 1000

        self.cache.store(query, response)
        print(f"🔵 Cache MISS → نداء LLM جديد ({latency_ms:.0f}ms)")
        return {"answer": response, "source": "llm", "latency_ms": latency_ms}


# ───────────── تشغيل تجريبي متكامل ─────────────
if __name__ == "__main__":
    def mock_llm_call(query: str) -> str:
        """محاكاة نداء LLM بطيء ومكلف لأغراض العرض"""
        time.sleep(0.5)
        return f"إجابة مُولَّدة فعليًا لسؤال: '{query}'"

    cache = SemanticCache(similarity_threshold=0.90)
    service = CachedLLMService(mock_llm_call, cache)

    queries = [
        "كيف أُعيد ضبط كلمة المرور؟",
        "كيف أرجّع الباسورد تاعي؟",          # إعادة صياغة دلالية قريبة جدًا - يجب أن تضرب الكاش
        "ما هي سياسة استرجاع الأموال؟",      # سؤال مختلف تمامًا - cache miss
        "نسيت كلمة السر، كيف أستعيدها؟",     # قريبة جدًا من السؤال الأول أيضًا
    ]

    for q in queries:
        result = service.answer(q)
        print(f"  → {result['answer'][:50]}... (المصدر: {result['source']})\n")

    print(f"📊 معدل إصابة الكاش الإجمالي: {cache.get_hit_rate()*100:.1f}%")
    print(f"💰 عدد نداءات LLM الموفَّرة: {service.total_cost_saved_calls}")

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-11-deployment-llmops/chapter-61.md
# ────────────────────────────────────────────────────────────────────────
# deployment_workflow.py
"""محاكاة سير عمل كامل: من تسجيل نسخة جديدة حتى الترقية الكاملة أو التراجع"""

registry = ModelRegistry()

# 1. تسجيل النسخة الحالية المستقرة (stable) والنسخة الجديدة (candidate)
stable_version = registry.register_version(
    model_checkpoint="llama-3.1-8b-v2",
    prompt_template="أنت مساعد مفيد ودقيق...",
    config={"temperature": 0.3, "max_tokens": 300},
)

candidate_version = registry.register_version(
    model_checkpoint="llama-3.1-8b-v3-finetuned",
    prompt_template="أنت مساعد مفيد ودقيق ومختصر...",  # تغيير طفيف → hash مختلف تلقائيًا
    config={"temperature": 0.3, "max_tokens": 300},
)

# 2. بدء بمرحلة Canary 5%
router = CanaryRouter(
    stable_handler=lambda q: f"[stable] رد على: {q}",
    canary_handler=lambda q: f"[canary] رد على: {q}",
    canary_percentage=0.05,
)
registry.transition_stage(candidate_version.deployment_id, DeploymentStage.CANARY_5, {"initial": True})

# 3. محاكاة حركة مرور حقيقية (في الإنتاج: هذا يأتي من طلبات المستخدمين الفعلية)
import random
for i in range(500):
    user_id = f"user_{random.randint(1, 200)}"
    router.route(user_id, f"سؤال رقم {i}", stable_version.deployment_id, candidate_version.deployment_id)

# 4. تقييم القرار: هل نرقّي للمرحلة التالية؟
report = router.compute_comparison_report()
can_promote, reason = should_promote_canary(report, min_samples=20)

print(f"\n📊 تقرير المقارنة: {json.dumps(report, indent=2, ensure_ascii=False)}")
print(f"🎯 قرار الترقية: {'✅ تُرقَّى' if can_promote else '❌ لا تُرقَّى'} — السبب: {reason}")

if can_promote:
    registry.transition_stage(candidate_version.deployment_id, DeploymentStage.CANARY_25, report)
else:
    registry.transition_stage(candidate_version.deployment_id, DeploymentStage.ROLLED_BACK, report)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-61-chapter-61.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 61: الفصل 61: بنية LLMOps — Versioning وCanary Deployment وSemantic Caching."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_llmops() -> None:
    versions = {'stable': {'requests': 1000, 'errors': 12, 'p95_ms': 220}, 'canary': {'requests': 100, 'errors': 1, 'p95_ms': 205}}
    rates = {name: data['errors'] / data['requests'] for name, data in versions.items()}
    allowed_gap = 0.02
    print('error rates:', {key: round(value, 3) for key, value in rates.items()})
    print('canary decision:', 'continue' if rates['canary'] <= rates['stable'] + allowed_gap else 'rollback')
    print('p95 latency:', {key: value['p95_ms'] for key, value in versions.items()})

if __name__ == "__main__":
    print("Chapter 61 lab: Compare canary and stable deployment metrics")
    run_llmops()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-61-chapter-61.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import hashlib
import json
import sqlite3
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional, Dict
from enum import Enum


class DeploymentStage(Enum):
    SHADOW = "shadow"
    CANARY_5 = "canary_5pct"
    CANARY_25 = "canary_25pct"
    CANARY_50 = "canary_50pct"
    FULL = "full_100pct"
    ROLLED_BACK = "rolled_back"


@dataclass
class ModelVersion:
    """وحدة النشر الكاملة — ليست فقط أوزان النموذج بل كل ما يؤثر على المخرجات"""
    deployment_id: str
    model_checkpoint: str          # مثلاً: "llama-3.1-8b-instruct-ft-v3"
    prompt_template_hash: str      # hash لمحتوى system prompt — يكشف أي تغيير ولو بسيط
    config: Dict                   # temperature, max_tokens, top_p, إلخ
    retrieval_index_version: Optional[str] = None  # إصدار فهرس RAG إن وُجد
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    stage: str = DeploymentStage.SHADOW.value
    approved_by: Optional[str] = None


class ModelRegistry:
    """سجل مركزي لكل نسخة تم نشرها، مع تاريخ كامل للمراجعة والتدقيق"""

    def __init__(self, db_path: str = "model_registry.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                deployment_id TEXT PRIMARY KEY,
                model_checkpoint TEXT, prompt_template_hash TEXT,
                config TEXT, retrieval_index_version TEXT,
                created_at TEXT, stage TEXT, approved_by TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS stage_transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                deployment_id TEXT, from_stage TEXT, to_stage TEXT,
                timestamp TEXT, metrics_snapshot TEXT
            )
        """)
        self.conn.commit()

    @staticmethod
    def compute_prompt_hash(prompt_template: str) -> str:
        """أي تغيير ولو حرف واحد في الـ prompt ينتج hash مختلف — يمنع الأخطاء الصامتة"""
        return hashlib.sha256(prompt_template.encode()).hexdigest()[:12]

    def register_version(self, model_checkpoint: str, prompt_template: str,
                          config: Dict, retrieval_index_version: Optional[str] = None) -> ModelVersion:
        prompt_hash = self.compute_prompt_hash(prompt_template)
        deployment_id = f"{model_checkpoint}_{prompt_hash}_{int(datetime.utcnow().timestamp())}"

        version = ModelVersion(
            deployment_id=deployment_id, model_checkpoint=model_checkpoint,
            prompt_template_hash=prompt_hash, config=config,
            retrieval_index_version=retrieval_index_version,
        )

        self.conn.execute(
            "INSERT INTO model_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (version.deployment_id, version.model_checkpoint, version.prompt_template_hash,
             json.dumps(version.config), version.retrieval_index_version,
             version.created_at, version.stage, version.approved_by)
        )
        self.conn.commit()
        print(f"📋 نسخة جديدة مُسجَّلة: {deployment_id}")
        return version

    def transition_stage(self, deployment_id: str, new_stage: DeploymentStage, metrics: Dict):
        """الانتقال بين مراحل النشر، مع تسجيل اللقطة الكاملة للمقاييس وقت القرار"""
        cursor = self.conn.execute(
            "SELECT stage FROM model_versions WHERE deployment_id = ?", (deployment_id,)
        )
        row = cursor.fetchone()
        old_stage = row[0] if row else "unknown"

        self.conn.execute(
            "UPDATE model_versions SET stage = ? WHERE deployment_id = ?",
            (new_stage.value, deployment_id)
        )
        self.conn.execute(
            "INSERT INTO stage_transitions (deployment_id, from_stage, to_stage, timestamp, metrics_snapshot) "
            "VALUES (?, ?, ?, ?, ?)",
            (deployment_id, old_stage, new_stage.value, datetime.utcnow().isoformat(), json.dumps(metrics))
        )
        self.conn.commit()
        print(f"🔄 {deployment_id}: {old_stage} → {new_stage.value} | المقاييس: {metrics}")

    def get_audit_trail(self, deployment_id: str) -> list:
        """سجل تدقيق كامل - ضروري عند التحقيق في مشكلة إنتاجية لاحقًا"""
        cursor = self.conn.execute(
            "SELECT from_stage, to_stage, timestamp, metrics_snapshot FROM stage_transitions "
            "WHERE deployment_id = ? ORDER BY timestamp", (deployment_id,)
        )
        return [{"from": r[0], "to": r[1], "time": r[2], "metrics": json.loads(r[3])} for r in cursor.fetchall()]

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-61-chapter-61.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import random
import hashlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List
import time


@dataclass
class RequestMetrics:
    """المقاييس المُسجَّلة لكل طلب — أساس قرار الترقية أو التراجع"""
    deployment_id: str
    latency_ms: float
    error_occurred: bool
    response_length: int
    user_feedback: int = None  # 1 = إيجابي، -1 = سلبي، None = لم يُقيَّم


class CanaryRouter:
    """
    يوزّع الحركة بين نسخة stable ونسخة canary وفق نسبة مُعدَّة، ويسجّل مقاييس كل مسار
    بشكل حتمي (نفس المستخدم → نفس النسخة دائمًا) لضمان تجربة متّسقة لكل مستخدم
    """

    def __init__(self, stable_handler: Callable, canary_handler: Callable, canary_percentage: float = 0.05):
        self.stable_handler = stable_handler
        self.canary_handler = canary_handler
        self.canary_percentage = canary_percentage
        self.metrics_log: List[RequestMetrics] = []

    def _should_route_to_canary(self, user_id: str) -> bool:
        """
        توزيع حتمي (deterministic) بناءً على hash الـ user_id، وليس عشوائيًا بالكامل في كل طلب —
        هذا يضمن أن نفس المستخدم يبقى في نفس المجموعة طوال فترة التجربة (consistency)
        """
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        bucket = (hash_value % 10000) / 10000.0
        return bucket < self.canary_percentage

    def route(self, user_id: str, query: str, deployment_id_stable: str, deployment_id_canary: str) -> Dict:
        use_canary = self._should_route_to_canary(user_id)
        handler = self.canary_handler if use_canary else self.stable_handler
        deployment_id = deployment_id_canary if use_canary else deployment_id_stable

        start = time.time()
        error_occurred = False
        try:
            response = handler(query)
        except Exception as e:
            response = f"حدث خطأ: {str(e)}"
            error_occurred = True

        latency_ms = (time.time() - start) * 1000

        metric = RequestMetrics(
            deployment_id=deployment_id, latency_ms=latency_ms,
            error_occurred=error_occurred, response_length=len(response),
        )
        self.metrics_log.append(metric)

        return {"response": response, "deployment_id": deployment_id, "is_canary": use_canary}

    def compute_comparison_report(self) -> Dict:
        """تقرير مقارنة إحصائي بين stable وcanary — أساس قرار الترقية للمرحلة التالية"""
        stable_metrics = [m for m in self.metrics_log if not self._is_canary_metric(m)]
        canary_metrics = [m for m in self.metrics_log if self._is_canary_metric(m)]

        def summarize(metrics: List[RequestMetrics]) -> Dict:
            if not metrics:
                return {"count": 0}
            return {
                "count": len(metrics),
                "avg_latency_ms": sum(m.latency_ms for m in metrics) / len(metrics),
                "error_rate": sum(m.error_occurred for m in metrics) / len(metrics),
                "avg_response_length": sum(m.response_length for m in metrics) / len(metrics),
            }

        return {"stable": summarize(stable_metrics), "canary": summarize(canary_metrics)}

    def _is_canary_metric(self, metric: RequestMetrics) -> bool:
        return "canary" in metric.deployment_id.lower()


def should_promote_canary(report: Dict, min_samples: int = 100,
                            max_latency_increase_pct: float = 20.0,
                            max_error_rate_increase: float = 0.02) -> tuple[bool, str]:
    """
    تطبيق معادلة القرار من القسم الرياضي: كل الشروط يجب أن تتحقق معًا
    """
    stable, canary = report["stable"], report["canary"]

    if canary.get("count", 0) < min_samples:
        return False, f"عينة غير كافية ({canary.get('count', 0)} < {min_samples})"

    if stable.get("count", 0) == 0:
        return False, "لا توجد بيانات stable للمقارنة"

    latency_increase_pct = ((canary["avg_latency_ms"] - stable["avg_latency_ms"]) / stable["avg_latency_ms"]) * 100
    if latency_increase_pct > max_latency_increase_pct:
        return False, f"زيادة زمن استجابة مفرطة: {latency_increase_pct:.1f}%"

    error_rate_increase = canary["error_rate"] - stable["error_rate"]
    if error_rate_increase > max_error_rate_increase:
        return False, f"زيادة معدل أخطاء مفرطة: {error_rate_increase*100:.1f} نقطة مئوية"

    return True, "كل المقاييس ضمن الحدود المقبولة ✅"

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-61-chapter-61.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import faiss
import time
import json
from dataclasses import dataclass
from typing import Optional, List
from sentence_transformers import SentenceTransformer


@dataclass
class CacheEntry:
    query: str
    response: str
    embedding: np.ndarray
    hit_count: int = 0
    created_at: float = 0.0


class SemanticCache:
    """
    كاش دلالي: يعيد إجابات لأسئلة مشابهة دلاليًا دون نداء LLM جديد
    عتبة التشابه عالية جدًا متعمَّدة لتجنّب إجابات خاطئة لأسئلة مختلفة المعنى
    """

    def __init__(self, similarity_threshold: float = 0.95, max_cache_size: int = 10000):
        self.embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.entries: List[CacheEntry] = []
        self.index = None
        self._dim = self.embedder.get_sentence_embedding_dimension()
        self._rebuild_index()

        # إحصائيات لمراقبة فعالية الكاش (ضرورية لإثبات قيمته أو اكتشاف مشاكله)
        self.total_queries = 0
        self.cache_hits = 0

    def _rebuild_index(self):
        self.index = faiss.IndexFlatIP(self._dim)
        if self.entries:
            embeddings = np.stack([e.embedding for e in self.entries]).astype("float32")
            self.index.add(embeddings)

    def lookup(self, query: str) -> Optional[Dict]:
        """البحث عن سؤال مشابه دلاليًا في الكاش قبل اللجوء لـ LLM"""
        self.total_queries += 1
        if not self.entries:
            return None

        query_embedding = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(query_embedding, k=1)

        top_score, top_idx = scores[0][0], indices[0][0]
        if top_idx != -1 and top_score >= self.threshold:
            matched_entry = self.entries[top_idx]
            matched_entry.hit_count += 1
            self.cache_hits += 1
            return {
                "response": matched_entry.response,
                "matched_query": matched_entry.query,
                "similarity": float(top_score),
                "from_cache": True,
            }
        return None

    def store(self, query: str, response: str):
        """تخزين زوج (سؤال، إجابة) جديد بعد نداء LLM حقيقي"""
        embedding = self.embedder.encode([query], normalize_embeddings=True)[0].astype("float32")
        entry = CacheEntry(query=query, response=response, embedding=embedding, created_at=time.time())
        self.entries.append(entry)

        if len(self.entries) > self.max_cache_size:
            # إزالة الأقل استخدامًا (LFU مبسّط) عند امتلاء الكاش
            self.entries.sort(key=lambda e: e.hit_count)
            self.entries.pop(0)

        self._rebuild_index()

    def get_hit_rate(self) -> float:
        return self.cache_hits / self.total_queries if self.total_queries > 0 else 0.0


class CachedLLMService:
    """تغليف كامل: يتحقق من الكاش أولًا، ويستدعي LLM فقط عند الحاجة الفعلية"""

    def __init__(self, llm_call_fn, cache: SemanticCache):
        self.llm_call_fn = llm_call_fn
        self.cache = cache
        self.total_cost_saved_calls = 0

    def answer(self, query: str) -> Dict:
        cached = self.cache.lookup(query)
        if cached:
            self.total_cost_saved_calls += 1
            print(f"⚡ Cache HIT (تشابه={cached['similarity']:.3f}) مع: '{cached['matched_query']}'")
            return {"answer": cached["response"], "source": "cache", "latency_ms": 2}  # الكاش شبه فوري

        start = time.time()
        response = self.llm_call_fn(query)
        latency_ms = (time.time() - start) * 1000

        self.cache.store(query, response)
        print(f"🔵 Cache MISS → نداء LLM جديد ({latency_ms:.0f}ms)")
        return {"answer": response, "source": "llm", "latency_ms": latency_ms}


# ───────────── تشغيل تجريبي متكامل ─────────────
if __name__ == "__main__":
    def mock_llm_call(query: str) -> str:
        """محاكاة نداء LLM بطيء ومكلف لأغراض العرض"""
        time.sleep(0.5)
        return f"إجابة مُولَّدة فعليًا لسؤال: '{query}'"

    cache = SemanticCache(similarity_threshold=0.90)
    service = CachedLLMService(mock_llm_call, cache)

    queries = [
        "كيف أُعيد ضبط كلمة المرور؟",
        "كيف أرجّع الباسورد تاعي؟",          # إعادة صياغة دلالية قريبة جدًا - يجب أن تضرب الكاش
        "ما هي سياسة استرجاع الأموال؟",      # سؤال مختلف تمامًا - cache miss
        "نسيت كلمة السر، كيف أستعيدها؟",     # قريبة جدًا من السؤال الأول أيضًا
    ]

    for q in queries:
        result = service.answer(q)
        print(f"  → {result['answer'][:50]}... (المصدر: {result['source']})\n")

    print(f"📊 معدل إصابة الكاش الإجمالي: {cache.get_hit_rate()*100:.1f}%")
    print(f"💰 عدد نداءات LLM الموفَّرة: {service.total_cost_saved_calls}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-61-chapter-61.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# deployment_workflow.py
"""محاكاة سير عمل كامل: من تسجيل نسخة جديدة حتى الترقية الكاملة أو التراجع"""

registry = ModelRegistry()

# 1. تسجيل النسخة الحالية المستقرة (stable) والنسخة الجديدة (candidate)
stable_version = registry.register_version(
    model_checkpoint="llama-3.1-8b-v2",
    prompt_template="أنت مساعد مفيد ودقيق...",
    config={"temperature": 0.3, "max_tokens": 300},
)

candidate_version = registry.register_version(
    model_checkpoint="llama-3.1-8b-v3-finetuned",
    prompt_template="أنت مساعد مفيد ودقيق ومختصر...",  # تغيير طفيف → hash مختلف تلقائيًا
    config={"temperature": 0.3, "max_tokens": 300},
)

# 2. بدء بمرحلة Canary 5%
router = CanaryRouter(
    stable_handler=lambda q: f"[stable] رد على: {q}",
    canary_handler=lambda q: f"[canary] رد على: {q}",
    canary_percentage=0.05,
)
registry.transition_stage(candidate_version.deployment_id, DeploymentStage.CANARY_5, {"initial": True})

# 3. محاكاة حركة مرور حقيقية (في الإنتاج: هذا يأتي من طلبات المستخدمين الفعلية)
import random
for i in range(500):
    user_id = f"user_{random.randint(1, 200)}"
    router.route(user_id, f"سؤال رقم {i}", stable_version.deployment_id, candidate_version.deployment_id)

# 4. تقييم القرار: هل نرقّي للمرحلة التالية؟
report = router.compute_comparison_report()
can_promote, reason = should_promote_canary(report, min_samples=20)

print(f"\n📊 تقرير المقارنة: {json.dumps(report, indent=2, ensure_ascii=False)}")
print(f"🎯 قرار الترقية: {'✅ تُرقَّى' if can_promote else '❌ لا تُرقَّى'} — السبب: {reason}")

if can_promote:
    registry.transition_stage(candidate_version.deployment_id, DeploymentStage.CANARY_25, report)
else:
    registry.transition_stage(candidate_version.deployment_id, DeploymentStage.ROLLED_BACK, report)
