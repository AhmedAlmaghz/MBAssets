#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — RAG الكامل (Retrieval-Augmented Generation)
Large Language Model Engineering — RAG

Book slug : llm-engineering-69-chapters
Chapter   : ch-53-chapter-53 (lesson 53)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-53-chapter-53.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — RAG الكامل (Retrieval-Augmented Generation)
Large Language Model Engineering — RAG

Book slug : llm-engineering-69-chapters
Chapter   : ch-53-chapter-53 (lesson 53)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-53-chapter-53.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — RAG الكامل (Retrieval-Augmented Generation)
Large Language Model Engineering — RAG

Book slug : llm-engineering-69-chapters
Chapter   : ch-53-chapter-53 (lesson 53)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_53.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 53: الفصل 53: RAG الكامل (Retrieval-Augmented Generation)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_rag() -> None:
    documents = {'d1': 'attention combines token values using learned weights', 'd2': 'a tokenizer maps text into token identifiers', 'd3': 'retrieval finds relevant source passages'}
    query = set('how does attention combine values'.split())
    ranked = sorted(((len(query & set(text.split())), doc_id, text) for doc_id, text in documents.items()), reverse=True)
    for score, doc_id, text in ranked[:2]:
        print(f'[{score}] [{doc_id}] {text}')
    print('answer policy: cite only retrieved IDs; abstain when evidence is absent.')

if __name__ == "__main__":
    print("Chapter 53 lab: Retrieve evidence and attach source citations")
    run_rag()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-10-applications-agents/chapter-53.md
# ────────────────────────────────────────────────────────────────────────
# requirements.txt
# torch>=2.4.0
# transformers>=4.44.0
# sentence-transformers>=3.0.0
# faiss-cpu>=1.8.0  # أو faiss-gpu للأداء الأعلى
# rank-bm25>=0.2.2
# chromadb>=0.5.0

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import re
from dataclasses import dataclass, field

device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-10-applications-agents/chapter-53.md
# ────────────────────────────────────────────────────────────────────────
@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_id: int
    metadata: dict = field(default_factory=dict)


def fixed_size_chunking(
    text: str, doc_id: str, chunk_size: int = 500, overlap: int = 50
) -> list[Chunk]:
    """
    تقطيع بحجم ثابت مع تداخل (Overlap) لتجنّب قطع المعنى عند الحدود.
    بسيط لكن فعّال كخط أساس (baseline).
    """
    words = text.split()
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(Chunk(
            text=" ".join(chunk_words),
            doc_id=doc_id,
            chunk_id=chunk_id,
        ))
        start += chunk_size - overlap  # التداخل يحافظ على السياق عند الحدود
        chunk_id += 1

    return chunks


def semantic_chunking(text: str, doc_id: str, similarity_threshold: float = 0.5) -> list[Chunk]:
    """
    تقطيع دلالي: يُقسّم النص عند نقاط التحوّل الموضوعي الفعلية،
    بدلًا من قطع تعسفي بعدد كلمات ثابت.
    """
    # تقسيم أولي إلى جمل
    sentences = re.split(r'(?<=[.!?؟])\s+', text.strip())
    sentences = [s for s in sentences if s.strip()]

    if len(sentences) <= 1:
        return [Chunk(text=text, doc_id=doc_id, chunk_id=0)]

    embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    embeddings = embedder.encode(sentences, convert_to_numpy=True)

    # حساب التشابه بين كل جملة والتي تليها
    chunks, current_chunk, chunk_id = [], [sentences[0]], 0
    for i in range(1, len(sentences)):
        sim = np.dot(embeddings[i-1], embeddings[i]) / (
            np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i]) + 1e-8
        )
        if sim < similarity_threshold:
            # نقطة تحوّل موضوعي → أغلق الـ Chunk الحالي وابدأ جديدًا
            chunks.append(Chunk(text=" ".join(current_chunk), doc_id=doc_id, chunk_id=chunk_id))
            chunk_id += 1
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    if current_chunk:
        chunks.append(Chunk(text=" ".join(current_chunk), doc_id=doc_id, chunk_id=chunk_id))

    return chunks

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-10-applications-agents/chapter-53.md
# ────────────────────────────────────────────────────────────────────────
class DenseRetriever:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        self.embedder = SentenceTransformer(model_name, device=device)
        self.index = None
        self.chunks: list[Chunk] = []

    def build_index(self, chunks: list[Chunk]):
        self.chunks = chunks
        texts = [c.text for c in chunks]

        embeddings = self.embedder.encode(
            texts, convert_to_numpy=True, show_progress_bar=True,
            normalize_embeddings=True  # تطبيع مسبق يجعل Inner Product = Cosine Similarity
        )

        dimension = embeddings.shape[1]
        # IndexFlatIP: بحث دقيق (Exact) عبر Inner Product — مناسب حتى ~مليون وثيقة
        # للحجم الأكبر، استخدم IndexIVFFlat أو IndexHNSWFlat (ANN تقريبي)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype(np.float32))

        print(f"✅ تم بناء فهرس يحتوي {self.index.ntotal} chunk")

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        query_embedding = self.embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.index.search(query_embedding, top_k)

        results = [
            (self.chunks[idx], float(score))
            for idx, score in zip(indices[0], scores[0]) if idx != -1
        ]
        return results

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-10-applications-agents/chapter-53.md
# ────────────────────────────────────────────────────────────────────────
class HybridRetriever:
    def __init__(self, dense_retriever: DenseRetriever):
        self.dense_retriever = dense_retriever
        self.bm25 = None
        self.chunks = dense_retriever.chunks

    def build_bm25_index(self):
        tokenized_corpus = [chunk.text.split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Min-Max normalization لجعل الدرجات قابلة للدمج العادل"""
        if scores.max() == scores.min():
            return np.zeros_like(scores)
        return (scores - scores.min()) / (scores.max() - scores.min())

    def search(
        self, query: str, top_k: int = 10, alpha: float = 0.6
    ) -> list[tuple[Chunk, float]]:
        """
        alpha: وزن Dense Retrieval مقابل BM25.
        alpha=1.0 → Dense فقط، alpha=0.0 → BM25 فقط
        """
        # 1. نتائج Dense على كامل القاعدة (للتطبيع الصحيح)
        dense_results = self.dense_retriever.search(query, top_k=len(self.chunks))
        dense_scores = {id(chunk): score for chunk, score in dense_results}

        # 2. نتائج BM25
        tokenized_query = query.split()
        bm25_raw_scores = self.bm25.get_scores(tokenized_query)

        # 3. تطبيع الدرجتين على نفس المقياس
        dense_score_array = np.array([dense_scores.get(id(c), 0) for c in self.chunks])
        dense_norm = self._normalize_scores(dense_score_array)
        bm25_norm = self._normalize_scores(bm25_raw_scores)

        # 4. الدمج الموزون
        hybrid_scores = alpha * dense_norm + (1 - alpha) * bm25_norm

        # 5. ترتيب واختيار Top-K
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        return [(self.chunks[i], float(hybrid_scores[i])) for i in top_indices]

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-10-applications-agents/chapter-53.md
# ────────────────────────────────────────────────────────────────────────
class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(
        self, query: str, candidates: list[tuple[Chunk, float]], top_n: int = 5
    ) -> list[tuple[Chunk, float]]:
        pairs = [[query, chunk.text] for chunk, _ in candidates]
        scores = self.model.predict(pairs)

        reranked = sorted(
            zip([c for c, _ in candidates], scores),
            key=lambda x: x[1], reverse=True
        )
        return reranked[:top_n]

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-10-applications-agents/chapter-53.md
# ────────────────────────────────────────────────────────────────────────
from openai import OpenAI  # أو أي مزوّد LLM آخر

class FullRAGPipeline:
    def __init__(self, llm_client, llm_model: str = "gpt-4o-mini"):
        self.dense_retriever = DenseRetriever()
        self.hybrid_retriever = None
        self.reranker = Reranker()
        self.llm_client = llm_client
        self.llm_model = llm_model

    def index_documents(self, documents: list[dict]):
        """documents: [{"id": "doc1", "text": "..."}, ...]"""
        all_chunks = []
        for doc in documents:
            chunks = fixed_size_chunking(doc["text"], doc_id=doc["id"], chunk_size=200, overlap=30)
            all_chunks.extend(chunks)

        self.dense_retriever.build_index(all_chunks)
        self.hybrid_retriever = HybridRetriever(self.dense_retriever)
        self.hybrid_retriever.build_bm25_index()

        print(f"📚 تمت فهرسة {len(documents)} وثيقة في {len(all_chunks)} chunk")

    def retrieve(self, query: str, initial_k: int = 20, final_n: int = 4) -> list[Chunk]:
        """مسار الاسترجاع الكامل: Hybrid Search → Re-ranking"""
        candidates = self.hybrid_retriever.search(query, top_k=initial_k, alpha=0.6)
        reranked = self.reranker.rerank(query, candidates, top_n=final_n)
        return [chunk for chunk, score in reranked]

    def build_prompt(self, query: str, context_chunks: list[Chunk]) -> str:
        context_text = "\n\n".join(
            f"[مصدر {i+1}]: {chunk.text}" for i, chunk in enumerate(context_chunks)
        )
        return f"""أنت مساعد يُجيب بدقة بناءً على السياق المُعطى فقط.
إذا لم تجد الإجابة في السياق، قل صراحة "لا أملك معلومات كافية للإجابة على هذا السؤال".
لا تختلق أي معلومة غير موجودة في السياق أدناه.

السياق:
{context_text}

السؤال: {query}

الإجابة (استشهد بأرقام المصادر إن أمكن):"""

    def answer(self, query: str) -> dict:
        retrieved_chunks = self.retrieve(query)
        prompt = self.build_prompt(query, retrieved_chunks)

        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # درجة حرارة منخفضة لتقليل الإبداع الزائد/الهلوسة
        )

        return {
            "query": query,
            "answer": response.choices[0].message.content,
            "sources": [
                {"doc_id": c.doc_id, "chunk_id": c.chunk_id, "text": c.text[:150] + "..."}
                for c in retrieved_chunks
            ]
        }


# --- الاستخدام الفعلي الكامل ---
if __name__ == "__main__":
    documents = [
        {
            "id": "policy_doc",
            "text": """سياسة الاسترجاع: يمكن للعملاء استرجاع أي منتج خلال 30 يومًا من تاريخ الشراء،
            بشرط الاحتفاظ بإيصال الشراء الأصلي وأن يكون المنتج في حالته الأصلية غير المستخدمة.
            المنتجات الإلكترونية تخضع لسياسة استرجاع خاصة مدتها 14 يومًا فقط بسبب طبيعتها.
            لا يُقبل استرجاع المنتجات الغذائية أو مستحضرات التجميل المفتوحة لأسباب صحية."""
        },
        {
            "id": "shipping_doc",
            "text": """سياسة الشحن: يستغرق الشحن المحلي 3-5 أيام عمل، والشحن الدولي 7-14 يومًا.
            الشحن مجاني للطلبات التي تتجاوز 200 ريال. رسوم الشحن القياسية هي 25 ريالًا
            للطلبات الأقل من ذلك. نوفر خيار الشحن السريع مقابل رسوم إضافية 50 ريالًا."""
        },
    ]

    client = OpenAI(api_key="sk-...")
    rag = FullRAGPipeline(llm_client=client)
    rag.index_documents(documents)

    result = rag.answer("كم يوم مهلة استرجاع المنتجات الإلكترونية؟")
    print(f"❓ السؤال: {result['query']}")
    print(f"💬 الإجابة: {result['answer']}")
    print(f"📎 المصادر: {[s['doc_id'] for s in result['sources']]}")

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-10-applications-agents/chapter-53.md
# ────────────────────────────────────────────────────────────────────────
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, answer_relevancy
from datasets import Dataset

def compare_rag_strategies(test_queries: list[str], ground_truths: list[str], rag_pipeline):
    """يقارن RAG بدون re-ranking مقابل RAG الكامل مع re-ranking"""

    results_naive, results_advanced = [], []

    for query in test_queries:
        # استراتيجية 1: Dense Retrieval فقط، بدون re-ranking (Naive RAG)
        naive_chunks = rag_pipeline.dense_retriever.search(query, top_k=4)
        naive_context = [c.text for c, _ in naive_chunks]
        naive_answer = rag_pipeline.answer(query)  # مبسّط للتوضيح

        # استراتيجية 2: Hybrid + Re-ranking (Advanced RAG)
        advanced_chunks = rag_pipeline.retrieve(query, initial_k=20, final_n=4)
        advanced_context = [c.text for c in advanced_chunks]
        advanced_answer = rag_pipeline.answer(query)

        results_naive.append({"contexts": naive_context, "answer": naive_answer["answer"]})
        results_advanced.append({"contexts": advanced_context, "answer": advanced_answer["answer"]})

    # في الواقع: نبني Dataset كامل لكل استراتيجية ونُقيّمها بـ Ragas كما في الفصل 50
    print("📊 النتيجة المتوقعة النمطية من دراسات الصناعة:")
    print("   Naive RAG     → Context Precision: ~0.65, Faithfulness: ~0.78")
    print("   Advanced RAG  → Context Precision: ~0.89, Faithfulness: ~0.93")
    print("   التحسن الرئيسي يأتي من Re-ranking الذي يُزيل الـ chunks المُسترجَعة")
    print("   بسبب تشابه سطحي زائف لكنها غير ذات صلة فعلية بالسؤال")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-53-chapter-53.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 53: الفصل 53: RAG الكامل (Retrieval-Augmented Generation)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_rag() -> None:
    documents = {'d1': 'attention combines token values using learned weights', 'd2': 'a tokenizer maps text into token identifiers', 'd3': 'retrieval finds relevant source passages'}
    query = set('how does attention combine values'.split())
    ranked = sorted(((len(query & set(text.split())), doc_id, text) for doc_id, text in documents.items()), reverse=True)
    for score, doc_id, text in ranked[:2]:
        print(f'[{score}] [{doc_id}] {text}')
    print('answer policy: cite only retrieved IDs; abstain when evidence is absent.')

if __name__ == "__main__":
    print("Chapter 53 lab: Retrieve evidence and attach source citations")
    run_rag()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-53-chapter-53.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# requirements.txt
# torch>=2.4.0
# transformers>=4.44.0
# sentence-transformers>=3.0.0
# faiss-cpu>=1.8.0  # أو faiss-gpu للأداء الأعلى
# rank-bm25>=0.2.2
# chromadb>=0.5.0

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import re
from dataclasses import dataclass, field

device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-53-chapter-53.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_id: int
    metadata: dict = field(default_factory=dict)


def fixed_size_chunking(
    text: str, doc_id: str, chunk_size: int = 500, overlap: int = 50
) -> list[Chunk]:
    """
    تقطيع بحجم ثابت مع تداخل (Overlap) لتجنّب قطع المعنى عند الحدود.
    بسيط لكن فعّال كخط أساس (baseline).
    """
    words = text.split()
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(Chunk(
            text=" ".join(chunk_words),
            doc_id=doc_id,
            chunk_id=chunk_id,
        ))
        start += chunk_size - overlap  # التداخل يحافظ على السياق عند الحدود
        chunk_id += 1

    return chunks


def semantic_chunking(text: str, doc_id: str, similarity_threshold: float = 0.5) -> list[Chunk]:
    """
    تقطيع دلالي: يُقسّم النص عند نقاط التحوّل الموضوعي الفعلية،
    بدلًا من قطع تعسفي بعدد كلمات ثابت.
    """
    # تقسيم أولي إلى جمل
    sentences = re.split(r'(?<=[.!?؟])\s+', text.strip())
    sentences = [s for s in sentences if s.strip()]

    if len(sentences) <= 1:
        return [Chunk(text=text, doc_id=doc_id, chunk_id=0)]

    embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    embeddings = embedder.encode(sentences, convert_to_numpy=True)

    # حساب التشابه بين كل جملة والتي تليها
    chunks, current_chunk, chunk_id = [], [sentences[0]], 0
    for i in range(1, len(sentences)):
        sim = np.dot(embeddings[i-1], embeddings[i]) / (
            np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i]) + 1e-8
        )
        if sim < similarity_threshold:
            # نقطة تحوّل موضوعي → أغلق الـ Chunk الحالي وابدأ جديدًا
            chunks.append(Chunk(text=" ".join(current_chunk), doc_id=doc_id, chunk_id=chunk_id))
            chunk_id += 1
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    if current_chunk:
        chunks.append(Chunk(text=" ".join(current_chunk), doc_id=doc_id, chunk_id=chunk_id))

    return chunks

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-53-chapter-53.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class DenseRetriever:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        self.embedder = SentenceTransformer(model_name, device=device)
        self.index = None
        self.chunks: list[Chunk] = []

    def build_index(self, chunks: list[Chunk]):
        self.chunks = chunks
        texts = [c.text for c in chunks]

        embeddings = self.embedder.encode(
            texts, convert_to_numpy=True, show_progress_bar=True,
            normalize_embeddings=True  # تطبيع مسبق يجعل Inner Product = Cosine Similarity
        )

        dimension = embeddings.shape[1]
        # IndexFlatIP: بحث دقيق (Exact) عبر Inner Product — مناسب حتى ~مليون وثيقة
        # للحجم الأكبر، استخدم IndexIVFFlat أو IndexHNSWFlat (ANN تقريبي)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype(np.float32))

        print(f"✅ تم بناء فهرس يحتوي {self.index.ntotal} chunk")

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        query_embedding = self.embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.index.search(query_embedding, top_k)

        results = [
            (self.chunks[idx], float(score))
            for idx, score in zip(indices[0], scores[0]) if idx != -1
        ]
        return results

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-53-chapter-53.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class HybridRetriever:
    def __init__(self, dense_retriever: DenseRetriever):
        self.dense_retriever = dense_retriever
        self.bm25 = None
        self.chunks = dense_retriever.chunks

    def build_bm25_index(self):
        tokenized_corpus = [chunk.text.split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Min-Max normalization لجعل الدرجات قابلة للدمج العادل"""
        if scores.max() == scores.min():
            return np.zeros_like(scores)
        return (scores - scores.min()) / (scores.max() - scores.min())

    def search(
        self, query: str, top_k: int = 10, alpha: float = 0.6
    ) -> list[tuple[Chunk, float]]:
        """
        alpha: وزن Dense Retrieval مقابل BM25.
        alpha=1.0 → Dense فقط، alpha=0.0 → BM25 فقط
        """
        # 1. نتائج Dense على كامل القاعدة (للتطبيع الصحيح)
        dense_results = self.dense_retriever.search(query, top_k=len(self.chunks))
        dense_scores = {id(chunk): score for chunk, score in dense_results}

        # 2. نتائج BM25
        tokenized_query = query.split()
        bm25_raw_scores = self.bm25.get_scores(tokenized_query)

        # 3. تطبيع الدرجتين على نفس المقياس
        dense_score_array = np.array([dense_scores.get(id(c), 0) for c in self.chunks])
        dense_norm = self._normalize_scores(dense_score_array)
        bm25_norm = self._normalize_scores(bm25_raw_scores)

        # 4. الدمج الموزون
        hybrid_scores = alpha * dense_norm + (1 - alpha) * bm25_norm

        # 5. ترتيب واختيار Top-K
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        return [(self.chunks[i], float(hybrid_scores[i])) for i in top_indices]

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-53-chapter-53.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(
        self, query: str, candidates: list[tuple[Chunk, float]], top_n: int = 5
    ) -> list[tuple[Chunk, float]]:
        pairs = [[query, chunk.text] for chunk, _ in candidates]
        scores = self.model.predict(pairs)

        reranked = sorted(
            zip([c for c, _ in candidates], scores),
            key=lambda x: x[1], reverse=True
        )
        return reranked[:top_n]

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-53-chapter-53.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from openai import OpenAI  # أو أي مزوّد LLM آخر

class FullRAGPipeline:
    def __init__(self, llm_client, llm_model: str = "gpt-4o-mini"):
        self.dense_retriever = DenseRetriever()
        self.hybrid_retriever = None
        self.reranker = Reranker()
        self.llm_client = llm_client
        self.llm_model = llm_model

    def index_documents(self, documents: list[dict]):
        """documents: [{"id": "doc1", "text": "..."}, ...]"""
        all_chunks = []
        for doc in documents:
            chunks = fixed_size_chunking(doc["text"], doc_id=doc["id"], chunk_size=200, overlap=30)
            all_chunks.extend(chunks)

        self.dense_retriever.build_index(all_chunks)
        self.hybrid_retriever = HybridRetriever(self.dense_retriever)
        self.hybrid_retriever.build_bm25_index()

        print(f"📚 تمت فهرسة {len(documents)} وثيقة في {len(all_chunks)} chunk")

    def retrieve(self, query: str, initial_k: int = 20, final_n: int = 4) -> list[Chunk]:
        """مسار الاسترجاع الكامل: Hybrid Search → Re-ranking"""
        candidates = self.hybrid_retriever.search(query, top_k=initial_k, alpha=0.6)
        reranked = self.reranker.rerank(query, candidates, top_n=final_n)
        return [chunk for chunk, score in reranked]

    def build_prompt(self, query: str, context_chunks: list[Chunk]) -> str:
        context_text = "\n\n".join(
            f"[مصدر {i+1}]: {chunk.text}" for i, chunk in enumerate(context_chunks)
        )
        return f"""أنت مساعد يُجيب بدقة بناءً على السياق المُعطى فقط.
إذا لم تجد الإجابة في السياق، قل صراحة "لا أملك معلومات كافية للإجابة على هذا السؤال".
لا تختلق أي معلومة غير موجودة في السياق أدناه.

السياق:
{context_text}

السؤال: {query}

الإجابة (استشهد بأرقام المصادر إن أمكن):"""

    def answer(self, query: str) -> dict:
        retrieved_chunks = self.retrieve(query)
        prompt = self.build_prompt(query, retrieved_chunks)

        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # درجة حرارة منخفضة لتقليل الإبداع الزائد/الهلوسة
        )

        return {
            "query": query,
            "answer": response.choices[0].message.content,
            "sources": [
                {"doc_id": c.doc_id, "chunk_id": c.chunk_id, "text": c.text[:150] + "..."}
                for c in retrieved_chunks
            ]
        }


# --- الاستخدام الفعلي الكامل ---
if __name__ == "__main__":
    documents = [
        {
            "id": "policy_doc",
            "text": """سياسة الاسترجاع: يمكن للعملاء استرجاع أي منتج خلال 30 يومًا من تاريخ الشراء،
            بشرط الاحتفاظ بإيصال الشراء الأصلي وأن يكون المنتج في حالته الأصلية غير المستخدمة.
            المنتجات الإلكترونية تخضع لسياسة استرجاع خاصة مدتها 14 يومًا فقط بسبب طبيعتها.
            لا يُقبل استرجاع المنتجات الغذائية أو مستحضرات التجميل المفتوحة لأسباب صحية."""
        },
        {
            "id": "shipping_doc",
            "text": """سياسة الشحن: يستغرق الشحن المحلي 3-5 أيام عمل، والشحن الدولي 7-14 يومًا.
            الشحن مجاني للطلبات التي تتجاوز 200 ريال. رسوم الشحن القياسية هي 25 ريالًا
            للطلبات الأقل من ذلك. نوفر خيار الشحن السريع مقابل رسوم إضافية 50 ريالًا."""
        },
    ]

    client = OpenAI(api_key="sk-...")
    rag = FullRAGPipeline(llm_client=client)
    rag.index_documents(documents)

    result = rag.answer("كم يوم مهلة استرجاع المنتجات الإلكترونية؟")
    print(f"❓ السؤال: {result['query']}")
    print(f"💬 الإجابة: {result['answer']}")
    print(f"📎 المصادر: {[s['doc_id'] for s in result['sources']]}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-53-chapter-53.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, answer_relevancy
from datasets import Dataset

def compare_rag_strategies(test_queries: list[str], ground_truths: list[str], rag_pipeline):
    """يقارن RAG بدون re-ranking مقابل RAG الكامل مع re-ranking"""

    results_naive, results_advanced = [], []

    for query in test_queries:
        # استراتيجية 1: Dense Retrieval فقط، بدون re-ranking (Naive RAG)
        naive_chunks = rag_pipeline.dense_retriever.search(query, top_k=4)
        naive_context = [c.text for c, _ in naive_chunks]
        naive_answer = rag_pipeline.answer(query)  # مبسّط للتوضيح

        # استراتيجية 2: Hybrid + Re-ranking (Advanced RAG)
        advanced_chunks = rag_pipeline.retrieve(query, initial_k=20, final_n=4)
        advanced_context = [c.text for c in advanced_chunks]
        advanced_answer = rag_pipeline.answer(query)

        results_naive.append({"contexts": naive_context, "answer": naive_answer["answer"]})
        results_advanced.append({"contexts": advanced_context, "answer": advanced_answer["answer"]})

    # في الواقع: نبني Dataset كامل لكل استراتيجية ونُقيّمها بـ Ragas كما في الفصل 50
    print("📊 النتيجة المتوقعة النمطية من دراسات الصناعة:")
    print("   Naive RAG     → Context Precision: ~0.65, Faithfulness: ~0.78")
    print("   Advanced RAG  → Context Precision: ~0.89, Faithfulness: ~0.93")
    print("   التحسن الرئيسي يأتي من Re-ranking الذي يُزيل الـ chunks المُسترجَعة")
    print("   بسبب تشابه سطحي زائف لكنها غير ذات صلة فعلية بالسؤال")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-53-chapter-53.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — RAG الكامل (Retrieval-Augmented Generation)
Large Language Model Engineering — RAG

Book slug : llm-engineering-69-chapters
Chapter   : ch-53-chapter-53 (lesson 53)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_53.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 53: الفصل 53: RAG الكامل (Retrieval-Augmented Generation)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_rag() -> None:
    documents = {'d1': 'attention combines token values using learned weights', 'd2': 'a tokenizer maps text into token identifiers', 'd3': 'retrieval finds relevant source passages'}
    query = set('how does attention combine values'.split())
    ranked = sorted(((len(query & set(text.split())), doc_id, text) for doc_id, text in documents.items()), reverse=True)
    for score, doc_id, text in ranked[:2]:
        print(f'[{score}] [{doc_id}] {text}')
    print('answer policy: cite only retrieved IDs; abstain when evidence is absent.')

if __name__ == "__main__":
    print("Chapter 53 lab: Retrieve evidence and attach source citations")
    run_rag()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-10-applications-agents/chapter-53.md
# ────────────────────────────────────────────────────────────────────────
# requirements.txt
# torch>=2.4.0
# transformers>=4.44.0
# sentence-transformers>=3.0.0
# faiss-cpu>=1.8.0  # أو faiss-gpu للأداء الأعلى
# rank-bm25>=0.2.2
# chromadb>=0.5.0

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import re
from dataclasses import dataclass, field

device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-10-applications-agents/chapter-53.md
# ────────────────────────────────────────────────────────────────────────
@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_id: int
    metadata: dict = field(default_factory=dict)


def fixed_size_chunking(
    text: str, doc_id: str, chunk_size: int = 500, overlap: int = 50
) -> list[Chunk]:
    """
    تقطيع بحجم ثابت مع تداخل (Overlap) لتجنّب قطع المعنى عند الحدود.
    بسيط لكن فعّال كخط أساس (baseline).
    """
    words = text.split()
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(Chunk(
            text=" ".join(chunk_words),
            doc_id=doc_id,
            chunk_id=chunk_id,
        ))
        start += chunk_size - overlap  # التداخل يحافظ على السياق عند الحدود
        chunk_id += 1

    return chunks


def semantic_chunking(text: str, doc_id: str, similarity_threshold: float = 0.5) -> list[Chunk]:
    """
    تقطيع دلالي: يُقسّم النص عند نقاط التحوّل الموضوعي الفعلية،
    بدلًا من قطع تعسفي بعدد كلمات ثابت.
    """
    # تقسيم أولي إلى جمل
    sentences = re.split(r'(?<=[.!?؟])\s+', text.strip())
    sentences = [s for s in sentences if s.strip()]

    if len(sentences) <= 1:
        return [Chunk(text=text, doc_id=doc_id, chunk_id=0)]

    embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    embeddings = embedder.encode(sentences, convert_to_numpy=True)

    # حساب التشابه بين كل جملة والتي تليها
    chunks, current_chunk, chunk_id = [], [sentences[0]], 0
    for i in range(1, len(sentences)):
        sim = np.dot(embeddings[i-1], embeddings[i]) / (
            np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i]) + 1e-8
        )
        if sim < similarity_threshold:
            # نقطة تحوّل موضوعي → أغلق الـ Chunk الحالي وابدأ جديدًا
            chunks.append(Chunk(text=" ".join(current_chunk), doc_id=doc_id, chunk_id=chunk_id))
            chunk_id += 1
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    if current_chunk:
        chunks.append(Chunk(text=" ".join(current_chunk), doc_id=doc_id, chunk_id=chunk_id))

    return chunks

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-10-applications-agents/chapter-53.md
# ────────────────────────────────────────────────────────────────────────
class DenseRetriever:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        self.embedder = SentenceTransformer(model_name, device=device)
        self.index = None
        self.chunks: list[Chunk] = []

    def build_index(self, chunks: list[Chunk]):
        self.chunks = chunks
        texts = [c.text for c in chunks]

        embeddings = self.embedder.encode(
            texts, convert_to_numpy=True, show_progress_bar=True,
            normalize_embeddings=True  # تطبيع مسبق يجعل Inner Product = Cosine Similarity
        )

        dimension = embeddings.shape[1]
        # IndexFlatIP: بحث دقيق (Exact) عبر Inner Product — مناسب حتى ~مليون وثيقة
        # للحجم الأكبر، استخدم IndexIVFFlat أو IndexHNSWFlat (ANN تقريبي)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype(np.float32))

        print(f"✅ تم بناء فهرس يحتوي {self.index.ntotal} chunk")

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        query_embedding = self.embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.index.search(query_embedding, top_k)

        results = [
            (self.chunks[idx], float(score))
            for idx, score in zip(indices[0], scores[0]) if idx != -1
        ]
        return results

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-10-applications-agents/chapter-53.md
# ────────────────────────────────────────────────────────────────────────
class HybridRetriever:
    def __init__(self, dense_retriever: DenseRetriever):
        self.dense_retriever = dense_retriever
        self.bm25 = None
        self.chunks = dense_retriever.chunks

    def build_bm25_index(self):
        tokenized_corpus = [chunk.text.split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Min-Max normalization لجعل الدرجات قابلة للدمج العادل"""
        if scores.max() == scores.min():
            return np.zeros_like(scores)
        return (scores - scores.min()) / (scores.max() - scores.min())

    def search(
        self, query: str, top_k: int = 10, alpha: float = 0.6
    ) -> list[tuple[Chunk, float]]:
        """
        alpha: وزن Dense Retrieval مقابل BM25.
        alpha=1.0 → Dense فقط، alpha=0.0 → BM25 فقط
        """
        # 1. نتائج Dense على كامل القاعدة (للتطبيع الصحيح)
        dense_results = self.dense_retriever.search(query, top_k=len(self.chunks))
        dense_scores = {id(chunk): score for chunk, score in dense_results}

        # 2. نتائج BM25
        tokenized_query = query.split()
        bm25_raw_scores = self.bm25.get_scores(tokenized_query)

        # 3. تطبيع الدرجتين على نفس المقياس
        dense_score_array = np.array([dense_scores.get(id(c), 0) for c in self.chunks])
        dense_norm = self._normalize_scores(dense_score_array)
        bm25_norm = self._normalize_scores(bm25_raw_scores)

        # 4. الدمج الموزون
        hybrid_scores = alpha * dense_norm + (1 - alpha) * bm25_norm

        # 5. ترتيب واختيار Top-K
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        return [(self.chunks[i], float(hybrid_scores[i])) for i in top_indices]

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-10-applications-agents/chapter-53.md
# ────────────────────────────────────────────────────────────────────────
class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(
        self, query: str, candidates: list[tuple[Chunk, float]], top_n: int = 5
    ) -> list[tuple[Chunk, float]]:
        pairs = [[query, chunk.text] for chunk, _ in candidates]
        scores = self.model.predict(pairs)

        reranked = sorted(
            zip([c for c, _ in candidates], scores),
            key=lambda x: x[1], reverse=True
        )
        return reranked[:top_n]

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-10-applications-agents/chapter-53.md
# ────────────────────────────────────────────────────────────────────────
from openai import OpenAI  # أو أي مزوّد LLM آخر

class FullRAGPipeline:
    def __init__(self, llm_client, llm_model: str = "gpt-4o-mini"):
        self.dense_retriever = DenseRetriever()
        self.hybrid_retriever = None
        self.reranker = Reranker()
        self.llm_client = llm_client
        self.llm_model = llm_model

    def index_documents(self, documents: list[dict]):
        """documents: [{"id": "doc1", "text": "..."}, ...]"""
        all_chunks = []
        for doc in documents:
            chunks = fixed_size_chunking(doc["text"], doc_id=doc["id"], chunk_size=200, overlap=30)
            all_chunks.extend(chunks)

        self.dense_retriever.build_index(all_chunks)
        self.hybrid_retriever = HybridRetriever(self.dense_retriever)
        self.hybrid_retriever.build_bm25_index()

        print(f"📚 تمت فهرسة {len(documents)} وثيقة في {len(all_chunks)} chunk")

    def retrieve(self, query: str, initial_k: int = 20, final_n: int = 4) -> list[Chunk]:
        """مسار الاسترجاع الكامل: Hybrid Search → Re-ranking"""
        candidates = self.hybrid_retriever.search(query, top_k=initial_k, alpha=0.6)
        reranked = self.reranker.rerank(query, candidates, top_n=final_n)
        return [chunk for chunk, score in reranked]

    def build_prompt(self, query: str, context_chunks: list[Chunk]) -> str:
        context_text = "\n\n".join(
            f"[مصدر {i+1}]: {chunk.text}" for i, chunk in enumerate(context_chunks)
        )
        return f"""أنت مساعد يُجيب بدقة بناءً على السياق المُعطى فقط.
إذا لم تجد الإجابة في السياق، قل صراحة "لا أملك معلومات كافية للإجابة على هذا السؤال".
لا تختلق أي معلومة غير موجودة في السياق أدناه.

السياق:
{context_text}

السؤال: {query}

الإجابة (استشهد بأرقام المصادر إن أمكن):"""

    def answer(self, query: str) -> dict:
        retrieved_chunks = self.retrieve(query)
        prompt = self.build_prompt(query, retrieved_chunks)

        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # درجة حرارة منخفضة لتقليل الإبداع الزائد/الهلوسة
        )

        return {
            "query": query,
            "answer": response.choices[0].message.content,
            "sources": [
                {"doc_id": c.doc_id, "chunk_id": c.chunk_id, "text": c.text[:150] + "..."}
                for c in retrieved_chunks
            ]
        }


# --- الاستخدام الفعلي الكامل ---
if __name__ == "__main__":
    documents = [
        {
            "id": "policy_doc",
            "text": """سياسة الاسترجاع: يمكن للعملاء استرجاع أي منتج خلال 30 يومًا من تاريخ الشراء،
            بشرط الاحتفاظ بإيصال الشراء الأصلي وأن يكون المنتج في حالته الأصلية غير المستخدمة.
            المنتجات الإلكترونية تخضع لسياسة استرجاع خاصة مدتها 14 يومًا فقط بسبب طبيعتها.
            لا يُقبل استرجاع المنتجات الغذائية أو مستحضرات التجميل المفتوحة لأسباب صحية."""
        },
        {
            "id": "shipping_doc",
            "text": """سياسة الشحن: يستغرق الشحن المحلي 3-5 أيام عمل، والشحن الدولي 7-14 يومًا.
            الشحن مجاني للطلبات التي تتجاوز 200 ريال. رسوم الشحن القياسية هي 25 ريالًا
            للطلبات الأقل من ذلك. نوفر خيار الشحن السريع مقابل رسوم إضافية 50 ريالًا."""
        },
    ]

    client = OpenAI(api_key="sk-...")
    rag = FullRAGPipeline(llm_client=client)
    rag.index_documents(documents)

    result = rag.answer("كم يوم مهلة استرجاع المنتجات الإلكترونية؟")
    print(f"❓ السؤال: {result['query']}")
    print(f"💬 الإجابة: {result['answer']}")
    print(f"📎 المصادر: {[s['doc_id'] for s in result['sources']]}")

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-10-applications-agents/chapter-53.md
# ────────────────────────────────────────────────────────────────────────
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, answer_relevancy
from datasets import Dataset

def compare_rag_strategies(test_queries: list[str], ground_truths: list[str], rag_pipeline):
    """يقارن RAG بدون re-ranking مقابل RAG الكامل مع re-ranking"""

    results_naive, results_advanced = [], []

    for query in test_queries:
        # استراتيجية 1: Dense Retrieval فقط، بدون re-ranking (Naive RAG)
        naive_chunks = rag_pipeline.dense_retriever.search(query, top_k=4)
        naive_context = [c.text for c, _ in naive_chunks]
        naive_answer = rag_pipeline.answer(query)  # مبسّط للتوضيح

        # استراتيجية 2: Hybrid + Re-ranking (Advanced RAG)
        advanced_chunks = rag_pipeline.retrieve(query, initial_k=20, final_n=4)
        advanced_context = [c.text for c in advanced_chunks]
        advanced_answer = rag_pipeline.answer(query)

        results_naive.append({"contexts": naive_context, "answer": naive_answer["answer"]})
        results_advanced.append({"contexts": advanced_context, "answer": advanced_answer["answer"]})

    # في الواقع: نبني Dataset كامل لكل استراتيجية ونُقيّمها بـ Ragas كما في الفصل 50
    print("📊 النتيجة المتوقعة النمطية من دراسات الصناعة:")
    print("   Naive RAG     → Context Precision: ~0.65, Faithfulness: ~0.78")
    print("   Advanced RAG  → Context Precision: ~0.89, Faithfulness: ~0.93")
    print("   التحسن الرئيسي يأتي من Re-ranking الذي يُزيل الـ chunks المُسترجَعة")
    print("   بسبب تشابه سطحي زائف لكنها غير ذات صلة فعلية بالسؤال")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-53-chapter-53.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 53: الفصل 53: RAG الكامل (Retrieval-Augmented Generation)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_rag() -> None:
    documents = {'d1': 'attention combines token values using learned weights', 'd2': 'a tokenizer maps text into token identifiers', 'd3': 'retrieval finds relevant source passages'}
    query = set('how does attention combine values'.split())
    ranked = sorted(((len(query & set(text.split())), doc_id, text) for doc_id, text in documents.items()), reverse=True)
    for score, doc_id, text in ranked[:2]:
        print(f'[{score}] [{doc_id}] {text}')
    print('answer policy: cite only retrieved IDs; abstain when evidence is absent.')

if __name__ == "__main__":
    print("Chapter 53 lab: Retrieve evidence and attach source citations")
    run_rag()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-53-chapter-53.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# requirements.txt
# torch>=2.4.0
# transformers>=4.44.0
# sentence-transformers>=3.0.0
# faiss-cpu>=1.8.0  # أو faiss-gpu للأداء الأعلى
# rank-bm25>=0.2.2
# chromadb>=0.5.0

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import re
from dataclasses import dataclass, field

device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-53-chapter-53.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_id: int
    metadata: dict = field(default_factory=dict)


def fixed_size_chunking(
    text: str, doc_id: str, chunk_size: int = 500, overlap: int = 50
) -> list[Chunk]:
    """
    تقطيع بحجم ثابت مع تداخل (Overlap) لتجنّب قطع المعنى عند الحدود.
    بسيط لكن فعّال كخط أساس (baseline).
    """
    words = text.split()
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(Chunk(
            text=" ".join(chunk_words),
            doc_id=doc_id,
            chunk_id=chunk_id,
        ))
        start += chunk_size - overlap  # التداخل يحافظ على السياق عند الحدود
        chunk_id += 1

    return chunks


def semantic_chunking(text: str, doc_id: str, similarity_threshold: float = 0.5) -> list[Chunk]:
    """
    تقطيع دلالي: يُقسّم النص عند نقاط التحوّل الموضوعي الفعلية،
    بدلًا من قطع تعسفي بعدد كلمات ثابت.
    """
    # تقسيم أولي إلى جمل
    sentences = re.split(r'(?<=[.!?؟])\s+', text.strip())
    sentences = [s for s in sentences if s.strip()]

    if len(sentences) <= 1:
        return [Chunk(text=text, doc_id=doc_id, chunk_id=0)]

    embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    embeddings = embedder.encode(sentences, convert_to_numpy=True)

    # حساب التشابه بين كل جملة والتي تليها
    chunks, current_chunk, chunk_id = [], [sentences[0]], 0
    for i in range(1, len(sentences)):
        sim = np.dot(embeddings[i-1], embeddings[i]) / (
            np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i]) + 1e-8
        )
        if sim < similarity_threshold:
            # نقطة تحوّل موضوعي → أغلق الـ Chunk الحالي وابدأ جديدًا
            chunks.append(Chunk(text=" ".join(current_chunk), doc_id=doc_id, chunk_id=chunk_id))
            chunk_id += 1
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    if current_chunk:
        chunks.append(Chunk(text=" ".join(current_chunk), doc_id=doc_id, chunk_id=chunk_id))

    return chunks

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-53-chapter-53.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class DenseRetriever:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        self.embedder = SentenceTransformer(model_name, device=device)
        self.index = None
        self.chunks: list[Chunk] = []

    def build_index(self, chunks: list[Chunk]):
        self.chunks = chunks
        texts = [c.text for c in chunks]

        embeddings = self.embedder.encode(
            texts, convert_to_numpy=True, show_progress_bar=True,
            normalize_embeddings=True  # تطبيع مسبق يجعل Inner Product = Cosine Similarity
        )

        dimension = embeddings.shape[1]
        # IndexFlatIP: بحث دقيق (Exact) عبر Inner Product — مناسب حتى ~مليون وثيقة
        # للحجم الأكبر، استخدم IndexIVFFlat أو IndexHNSWFlat (ANN تقريبي)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype(np.float32))

        print(f"✅ تم بناء فهرس يحتوي {self.index.ntotal} chunk")

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        query_embedding = self.embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.index.search(query_embedding, top_k)

        results = [
            (self.chunks[idx], float(score))
            for idx, score in zip(indices[0], scores[0]) if idx != -1
        ]
        return results

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-53-chapter-53.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
class HybridRetriever:
    def __init__(self, dense_retriever: DenseRetriever):
        self.dense_retriever = dense_retriever
        self.bm25 = None
        self.chunks = dense_retriever.chunks

    def build_bm25_index(self):
        tokenized_corpus = [chunk.text.split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Min-Max normalization لجعل الدرجات قابلة للدمج العادل"""
        if scores.max() == scores.min():
            return np.zeros_like(scores)
        return (scores - scores.min()) / (scores.max() - scores.min())

    def search(
        self, query: str, top_k: int = 10, alpha: float = 0.6
    ) -> list[tuple[Chunk, float]]:
        """
        alpha: وزن Dense Retrieval مقابل BM25.
        alpha=1.0 → Dense فقط، alpha=0.0 → BM25 فقط
        """
        # 1. نتائج Dense على كامل القاعدة (للتطبيع الصحيح)
        dense_results = self.dense_retriever.search(query, top_k=len(self.chunks))
        dense_scores = {id(chunk): score for chunk, score in dense_results}

        # 2. نتائج BM25
        tokenized_query = query.split()
        bm25_raw_scores = self.bm25.get_scores(tokenized_query)

        # 3. تطبيع الدرجتين على نفس المقياس
        dense_score_array = np.array([dense_scores.get(id(c), 0) for c in self.chunks])
        dense_norm = self._normalize_scores(dense_score_array)
        bm25_norm = self._normalize_scores(bm25_raw_scores)

        # 4. الدمج الموزون
        hybrid_scores = alpha * dense_norm + (1 - alpha) * bm25_norm

        # 5. ترتيب واختيار Top-K
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        return [(self.chunks[i], float(hybrid_scores[i])) for i in top_indices]

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-53-chapter-53.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(
        self, query: str, candidates: list[tuple[Chunk, float]], top_n: int = 5
    ) -> list[tuple[Chunk, float]]:
        pairs = [[query, chunk.text] for chunk, _ in candidates]
        scores = self.model.predict(pairs)

        reranked = sorted(
            zip([c for c, _ in candidates], scores),
            key=lambda x: x[1], reverse=True
        )
        return reranked[:top_n]

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-53-chapter-53.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from openai import OpenAI  # أو أي مزوّد LLM آخر

class FullRAGPipeline:
    def __init__(self, llm_client, llm_model: str = "gpt-4o-mini"):
        self.dense_retriever = DenseRetriever()
        self.hybrid_retriever = None
        self.reranker = Reranker()
        self.llm_client = llm_client
        self.llm_model = llm_model

    def index_documents(self, documents: list[dict]):
        """documents: [{"id": "doc1", "text": "..."}, ...]"""
        all_chunks = []
        for doc in documents:
            chunks = fixed_size_chunking(doc["text"], doc_id=doc["id"], chunk_size=200, overlap=30)
            all_chunks.extend(chunks)

        self.dense_retriever.build_index(all_chunks)
        self.hybrid_retriever = HybridRetriever(self.dense_retriever)
        self.hybrid_retriever.build_bm25_index()

        print(f"📚 تمت فهرسة {len(documents)} وثيقة في {len(all_chunks)} chunk")

    def retrieve(self, query: str, initial_k: int = 20, final_n: int = 4) -> list[Chunk]:
        """مسار الاسترجاع الكامل: Hybrid Search → Re-ranking"""
        candidates = self.hybrid_retriever.search(query, top_k=initial_k, alpha=0.6)
        reranked = self.reranker.rerank(query, candidates, top_n=final_n)
        return [chunk for chunk, score in reranked]

    def build_prompt(self, query: str, context_chunks: list[Chunk]) -> str:
        context_text = "\n\n".join(
            f"[مصدر {i+1}]: {chunk.text}" for i, chunk in enumerate(context_chunks)
        )
        return f"""أنت مساعد يُجيب بدقة بناءً على السياق المُعطى فقط.
إذا لم تجد الإجابة في السياق، قل صراحة "لا أملك معلومات كافية للإجابة على هذا السؤال".
لا تختلق أي معلومة غير موجودة في السياق أدناه.

السياق:
{context_text}

السؤال: {query}

الإجابة (استشهد بأرقام المصادر إن أمكن):"""

    def answer(self, query: str) -> dict:
        retrieved_chunks = self.retrieve(query)
        prompt = self.build_prompt(query, retrieved_chunks)

        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # درجة حرارة منخفضة لتقليل الإبداع الزائد/الهلوسة
        )

        return {
            "query": query,
            "answer": response.choices[0].message.content,
            "sources": [
                {"doc_id": c.doc_id, "chunk_id": c.chunk_id, "text": c.text[:150] + "..."}
                for c in retrieved_chunks
            ]
        }


# --- الاستخدام الفعلي الكامل ---
if __name__ == "__main__":
    documents = [
        {
            "id": "policy_doc",
            "text": """سياسة الاسترجاع: يمكن للعملاء استرجاع أي منتج خلال 30 يومًا من تاريخ الشراء،
            بشرط الاحتفاظ بإيصال الشراء الأصلي وأن يكون المنتج في حالته الأصلية غير المستخدمة.
            المنتجات الإلكترونية تخضع لسياسة استرجاع خاصة مدتها 14 يومًا فقط بسبب طبيعتها.
            لا يُقبل استرجاع المنتجات الغذائية أو مستحضرات التجميل المفتوحة لأسباب صحية."""
        },
        {
            "id": "shipping_doc",
            "text": """سياسة الشحن: يستغرق الشحن المحلي 3-5 أيام عمل، والشحن الدولي 7-14 يومًا.
            الشحن مجاني للطلبات التي تتجاوز 200 ريال. رسوم الشحن القياسية هي 25 ريالًا
            للطلبات الأقل من ذلك. نوفر خيار الشحن السريع مقابل رسوم إضافية 50 ريالًا."""
        },
    ]

    client = OpenAI(api_key="sk-...")
    rag = FullRAGPipeline(llm_client=client)
    rag.index_documents(documents)

    result = rag.answer("كم يوم مهلة استرجاع المنتجات الإلكترونية؟")
    print(f"❓ السؤال: {result['query']}")
    print(f"💬 الإجابة: {result['answer']}")
    print(f"📎 المصادر: {[s['doc_id'] for s in result['sources']]}")

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-53-chapter-53.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, answer_relevancy
from datasets import Dataset

def compare_rag_strategies(test_queries: list[str], ground_truths: list[str], rag_pipeline):
    """يقارن RAG بدون re-ranking مقابل RAG الكامل مع re-ranking"""

    results_naive, results_advanced = [], []

    for query in test_queries:
        # استراتيجية 1: Dense Retrieval فقط، بدون re-ranking (Naive RAG)
        naive_chunks = rag_pipeline.dense_retriever.search(query, top_k=4)
        naive_context = [c.text for c, _ in naive_chunks]
        naive_answer = rag_pipeline.answer(query)  # مبسّط للتوضيح

        # استراتيجية 2: Hybrid + Re-ranking (Advanced RAG)
        advanced_chunks = rag_pipeline.retrieve(query, initial_k=20, final_n=4)
        advanced_context = [c.text for c in advanced_chunks]
        advanced_answer = rag_pipeline.answer(query)

        results_naive.append({"contexts": naive_context, "answer": naive_answer["answer"]})
        results_advanced.append({"contexts": advanced_context, "answer": advanced_answer["answer"]})

    # في الواقع: نبني Dataset كامل لكل استراتيجية ونُقيّمها بـ Ragas كما في الفصل 50
    print("📊 النتيجة المتوقعة النمطية من دراسات الصناعة:")
    print("   Naive RAG     → Context Precision: ~0.65, Faithfulness: ~0.78")
    print("   Advanced RAG  → Context Precision: ~0.89, Faithfulness: ~0.93")
    print("   التحسن الرئيسي يأتي من Re-ranking الذي يُزيل الـ chunks المُسترجَعة")
    print("   بسبب تشابه سطحي زائف لكنها غير ذات صلة فعلية بالسؤال")
