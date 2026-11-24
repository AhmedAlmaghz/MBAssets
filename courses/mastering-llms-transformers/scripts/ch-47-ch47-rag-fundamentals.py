#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — أسس التوليد المعزّز بالاسترجاع: RAG
Mastering Transformers and Large Language Models — Rag Fundamentals

Book slug : mastering-llms-transformers
Chapter   : ch-47-ch47-rag-fundamentals (lesson 47)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-47-ch47-rag-fundamentals.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — أسس التوليد المعزّز بالاسترجاع: RAG
Mastering Transformers and Large Language Models — Rag Fundamentals

Book slug : mastering-llms-transformers
Chapter   : ch-47-ch47-rag-fundamentals (lesson 47)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-47-ch47-rag-fundamentals.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — أسس التوليد المعزّز بالاسترجاع: RAG
Mastering Transformers and Large Language Models — Rag Fundamentals

Book slug : mastering-llms-transformers
Chapter   : ch-47-ch47-rag-fundamentals (lesson 47)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-10-applications-agents/ch47-rag-fundamentals.ipynb#cell-1
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
    ("faiss", "faiss-cpu"),
    ("numpy", "numpy"),
    ("sentence_transformers", "sentence-transformers")
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
# [002] content/notebooks/part-10-applications-agents/ch47-rag-fundamentals.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class SimpleRAGPipeline:
    """
    تطبيق كامل لخط RAG أساسي — من تجزئة المستندات حتى توليد الإجابة
    """
    def __init__(self, embedding_model_name="intfloat/multilingual-e5-large",
                 generator_model=None, chunk_size=512, chunk_overlap=50):
        self.embedder = SentenceTransformer(embedding_model_name)
        self.generator = generator_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index = None
        self.chunks = []
        self.chunk_metadata = []

    def chunk_document(self, text: str, source_id: str) -> list:
        """
        تجزئة النص إلى قطع متراكبة (Overlapping Chunks)
        التراكب يمنع فقدان السياق عند قطع الجملة في منتصفها
        """
        words = text.split()
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            if len(chunk_words) < 20:   # تجاهل القطع الصغيرة جدًا في النهاية
                continue
            chunks.append({
                "text": " ".join(chunk_words),
                "source_id": source_id,
                "chunk_index": i // step
            })
        return chunks

    def index_documents(self, documents: dict):
        """documents: {source_id: full_text}"""
        all_chunks = []
        for source_id, text in documents.items():
            all_chunks.extend(self.chunk_document(text, source_id))

        self.chunks = [c["text"] for c in all_chunks]
        self.chunk_metadata = all_chunks

        # E5 يتطلب بادئة "passage: " للمستندات و "query: " للأسئلة (تفصيل مهم جدًا!)
        prefixed_chunks = [f"passage: {c}" for c in self.chunks]
        embeddings = self.embedder.encode(prefixed_chunks, normalize_embeddings=True,
                                            show_progress_bar=True)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)   # Inner Product = Cosine (بعد التطبيع)
        self.index.add(embeddings.astype(np.float32))

    def retrieve(self, query: str, top_k=5) -> list:
        query_embedding = self.embedder.encode([f"query: {query}"], normalize_embeddings=True)
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "text": self.chunks[idx],
                "score": float(score),
                "metadata": self.chunk_metadata[idx]
            })
        return results

    def build_augmented_prompt(self, query: str, retrieved_chunks: list) -> str:
        context = "\n\n".join([
            f"[مصدر {i+1}]: {chunk['text']}" for i, chunk in enumerate(retrieved_chunks)
        ])
        return f"""استخدم المعلومات التالية فقط للإجابة على السؤال. إذا لم تجد الإجابة في المعلومات المتوفرة، صرّح بذلك بوضوح ولا تختلق معلومات.

المعلومات المتاحة:
{context}

السؤال: {query}

الإجابة (مع الإشارة لرقم المصدر عند الاستشهاد):"""

    def query(self, question: str, top_k=5) -> dict:
        retrieved = self.retrieve(question, top_k)
        prompt = self.build_augmented_prompt(question, retrieved)
        answer = self.generator.generate(prompt)
        return {"answer": answer, "sources": retrieved, "prompt_used": prompt}

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-10-applications-agents/ch47-rag-fundamentals.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import re

def semantic_chunking(text: str, max_chunk_size=512, embedder=None, similarity_threshold=0.5):
    """
    تجزئة أذكى: تُقسّم النص لجمل أولاً، ثم تُجمّع الجمل المتتالية
    المتشابهة دلاليًا في نفس القطعة، وتفصل عند نقاط التحوّل الموضوعي
    """
    sentences = re.split(r'(?<=[.!؟])\s+', text)
    if embedder is None or len(sentences) < 2:
        # fallback بسيط: تجميع حسب الحجم فقط مع احترام حدود الجمل
        chunks, current_chunk = [], ""
        for sentence in sentences:
            if len(current_chunk.split()) + len(sentence.split()) > max_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    # تجزئة دلالية حقيقية باستخدام تشابه embeddings متتالية
    sentence_embeddings = embedder.encode(sentences, normalize_embeddings=True)
    chunks, current_chunk_sentences = [], [sentences[0]]

    for i in range(1, len(sentences)):
        similarity = np.dot(sentence_embeddings[i], sentence_embeddings[i-1])
        current_length = sum(len(s.split()) for s in current_chunk_sentences)

        if similarity < similarity_threshold or current_length > max_chunk_size:
            chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = [sentences[i]]
        else:
            current_chunk_sentences.append(sentences[i])

    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))
    return chunks

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-10-applications-agents/ch47-rag-fundamentals.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def evaluate_rag_components_separately(rag_pipeline, test_set: list):
    """
    إطار RAGAS المبسّط: تقييم كل مكوّن من مكونات RAG بشكل مستقل
    """
    results = {"retrieval_metrics": [], "generation_metrics": []}

    for example in test_set:
        retrieved = rag_pipeline.retrieve(example["question"], top_k=5)

        # 1. Context Precision: هل المستندات المُسترجَعة ذات صلة فعلًا؟
        relevant_retrieved = sum(
            1 for r in retrieved if r["metadata"]["source_id"] in example["relevant_source_ids"]
        )
        context_precision = relevant_retrieved / len(retrieved) if retrieved else 0

        # 2. Context Recall: هل استرجعنا كل المعلومات اللازمة للإجابة؟
        all_relevant_found = all(
            any(r["metadata"]["source_id"] == sid for r in retrieved)
            for sid in example["relevant_source_ids"]
        )
        context_recall = 1.0 if all_relevant_found else 0.0

        results["retrieval_metrics"].append({
            "precision": context_precision, "recall": context_recall
        })

        # 3. Faithfulness: هل الإجابة مبنية فعلًا على السياق المُسترجَع (وليست هلوسة)؟
        answer = rag_pipeline.query(example["question"])["answer"]
        faithfulness_prompt = f"""هل كل ادّعاء في الإجابة التالية مدعوم بالسياق المُعطى؟
السياق: {[r['text'] for r in retrieved]}
الإجابة: {answer}
أجب بـ "نعم" أو "لا" مع ذكر أي ادّعاء غير مدعوم."""
        # يتطلب LLM-as-a-Judge (من الفصل 43)
        results["generation_metrics"].append({"faithfulness_check": faithfulness_prompt})

    avg_precision = np.mean([m["precision"] for m in results["retrieval_metrics"]])
    avg_recall = np.mean([m["recall"] for m in results["retrieval_metrics"]])
    return {"avg_context_precision": avg_precision, "avg_context_recall": avg_recall}

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-10-applications-agents/ch47-rag-fundamentals.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-47-ch47-rag-fundamentals.ipynb#cell-1
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
    ("faiss", "faiss-cpu"),
    ("numpy", "numpy"),
    ("sentence_transformers", "sentence-transformers")
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-47-ch47-rag-fundamentals.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class SimpleRAGPipeline:
    """
    تطبيق كامل لخط RAG أساسي — من تجزئة المستندات حتى توليد الإجابة
    """
    def __init__(self, embedding_model_name="intfloat/multilingual-e5-large",
                 generator_model=None, chunk_size=512, chunk_overlap=50):
        self.embedder = SentenceTransformer(embedding_model_name)
        self.generator = generator_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index = None
        self.chunks = []
        self.chunk_metadata = []

    def chunk_document(self, text: str, source_id: str) -> list:
        """
        تجزئة النص إلى قطع متراكبة (Overlapping Chunks)
        التراكب يمنع فقدان السياق عند قطع الجملة في منتصفها
        """
        words = text.split()
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            if len(chunk_words) < 20:   # تجاهل القطع الصغيرة جدًا في النهاية
                continue
            chunks.append({
                "text": " ".join(chunk_words),
                "source_id": source_id,
                "chunk_index": i // step
            })
        return chunks

    def index_documents(self, documents: dict):
        """documents: {source_id: full_text}"""
        all_chunks = []
        for source_id, text in documents.items():
            all_chunks.extend(self.chunk_document(text, source_id))

        self.chunks = [c["text"] for c in all_chunks]
        self.chunk_metadata = all_chunks

        # E5 يتطلب بادئة "passage: " للمستندات و "query: " للأسئلة (تفصيل مهم جدًا!)
        prefixed_chunks = [f"passage: {c}" for c in self.chunks]
        embeddings = self.embedder.encode(prefixed_chunks, normalize_embeddings=True,
                                            show_progress_bar=True)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)   # Inner Product = Cosine (بعد التطبيع)
        self.index.add(embeddings.astype(np.float32))

    def retrieve(self, query: str, top_k=5) -> list:
        query_embedding = self.embedder.encode([f"query: {query}"], normalize_embeddings=True)
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "text": self.chunks[idx],
                "score": float(score),
                "metadata": self.chunk_metadata[idx]
            })
        return results

    def build_augmented_prompt(self, query: str, retrieved_chunks: list) -> str:
        context = "\n\n".join([
            f"[مصدر {i+1}]: {chunk['text']}" for i, chunk in enumerate(retrieved_chunks)
        ])
        return f"""استخدم المعلومات التالية فقط للإجابة على السؤال. إذا لم تجد الإجابة في المعلومات المتوفرة، صرّح بذلك بوضوح ولا تختلق معلومات.

المعلومات المتاحة:
{context}

السؤال: {query}

الإجابة (مع الإشارة لرقم المصدر عند الاستشهاد):"""

    def query(self, question: str, top_k=5) -> dict:
        retrieved = self.retrieve(question, top_k)
        prompt = self.build_augmented_prompt(question, retrieved)
        answer = self.generator.generate(prompt)
        return {"answer": answer, "sources": retrieved, "prompt_used": prompt}

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-47-ch47-rag-fundamentals.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import re

def semantic_chunking(text: str, max_chunk_size=512, embedder=None, similarity_threshold=0.5):
    """
    تجزئة أذكى: تُقسّم النص لجمل أولاً، ثم تُجمّع الجمل المتتالية
    المتشابهة دلاليًا في نفس القطعة، وتفصل عند نقاط التحوّل الموضوعي
    """
    sentences = re.split(r'(?<=[.!؟])\s+', text)
    if embedder is None or len(sentences) < 2:
        # fallback بسيط: تجميع حسب الحجم فقط مع احترام حدود الجمل
        chunks, current_chunk = [], ""
        for sentence in sentences:
            if len(current_chunk.split()) + len(sentence.split()) > max_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    # تجزئة دلالية حقيقية باستخدام تشابه embeddings متتالية
    sentence_embeddings = embedder.encode(sentences, normalize_embeddings=True)
    chunks, current_chunk_sentences = [], [sentences[0]]

    for i in range(1, len(sentences)):
        similarity = np.dot(sentence_embeddings[i], sentence_embeddings[i-1])
        current_length = sum(len(s.split()) for s in current_chunk_sentences)

        if similarity < similarity_threshold or current_length > max_chunk_size:
            chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = [sentences[i]]
        else:
            current_chunk_sentences.append(sentences[i])

    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))
    return chunks

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-47-ch47-rag-fundamentals.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def evaluate_rag_components_separately(rag_pipeline, test_set: list):
    """
    إطار RAGAS المبسّط: تقييم كل مكوّن من مكونات RAG بشكل مستقل
    """
    results = {"retrieval_metrics": [], "generation_metrics": []}

    for example in test_set:
        retrieved = rag_pipeline.retrieve(example["question"], top_k=5)

        # 1. Context Precision: هل المستندات المُسترجَعة ذات صلة فعلًا؟
        relevant_retrieved = sum(
            1 for r in retrieved if r["metadata"]["source_id"] in example["relevant_source_ids"]
        )
        context_precision = relevant_retrieved / len(retrieved) if retrieved else 0

        # 2. Context Recall: هل استرجعنا كل المعلومات اللازمة للإجابة؟
        all_relevant_found = all(
            any(r["metadata"]["source_id"] == sid for r in retrieved)
            for sid in example["relevant_source_ids"]
        )
        context_recall = 1.0 if all_relevant_found else 0.0

        results["retrieval_metrics"].append({
            "precision": context_precision, "recall": context_recall
        })

        # 3. Faithfulness: هل الإجابة مبنية فعلًا على السياق المُسترجَع (وليست هلوسة)؟
        answer = rag_pipeline.query(example["question"])["answer"]
        faithfulness_prompt = f"""هل كل ادّعاء في الإجابة التالية مدعوم بالسياق المُعطى؟
السياق: {[r['text'] for r in retrieved]}
الإجابة: {answer}
أجب بـ "نعم" أو "لا" مع ذكر أي ادّعاء غير مدعوم."""
        # يتطلب LLM-as-a-Judge (من الفصل 43)
        results["generation_metrics"].append({"faithfulness_check": faithfulness_prompt})

    avg_precision = np.mean([m["precision"] for m in results["retrieval_metrics"]])
    avg_recall = np.mean([m["recall"] for m in results["retrieval_metrics"]])
    return {"avg_context_precision": avg_precision, "avg_context_recall": avg_recall}

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-47-ch47-rag-fundamentals.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-47-ch47-rag-fundamentals.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — أسس التوليد المعزّز بالاسترجاع: RAG
Mastering Transformers and Large Language Models — Rag Fundamentals

Book slug : mastering-llms-transformers
Chapter   : ch-47-ch47-rag-fundamentals (lesson 47)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-10-applications-agents/ch47-rag-fundamentals.ipynb#cell-1
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
    ("faiss", "faiss-cpu"),
    ("numpy", "numpy"),
    ("sentence_transformers", "sentence-transformers")
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
# [002] content/notebooks/part-10-applications-agents/ch47-rag-fundamentals.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class SimpleRAGPipeline:
    """
    تطبيق كامل لخط RAG أساسي — من تجزئة المستندات حتى توليد الإجابة
    """
    def __init__(self, embedding_model_name="intfloat/multilingual-e5-large",
                 generator_model=None, chunk_size=512, chunk_overlap=50):
        self.embedder = SentenceTransformer(embedding_model_name)
        self.generator = generator_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index = None
        self.chunks = []
        self.chunk_metadata = []

    def chunk_document(self, text: str, source_id: str) -> list:
        """
        تجزئة النص إلى قطع متراكبة (Overlapping Chunks)
        التراكب يمنع فقدان السياق عند قطع الجملة في منتصفها
        """
        words = text.split()
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            if len(chunk_words) < 20:   # تجاهل القطع الصغيرة جدًا في النهاية
                continue
            chunks.append({
                "text": " ".join(chunk_words),
                "source_id": source_id,
                "chunk_index": i // step
            })
        return chunks

    def index_documents(self, documents: dict):
        """documents: {source_id: full_text}"""
        all_chunks = []
        for source_id, text in documents.items():
            all_chunks.extend(self.chunk_document(text, source_id))

        self.chunks = [c["text"] for c in all_chunks]
        self.chunk_metadata = all_chunks

        # E5 يتطلب بادئة "passage: " للمستندات و "query: " للأسئلة (تفصيل مهم جدًا!)
        prefixed_chunks = [f"passage: {c}" for c in self.chunks]
        embeddings = self.embedder.encode(prefixed_chunks, normalize_embeddings=True,
                                            show_progress_bar=True)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)   # Inner Product = Cosine (بعد التطبيع)
        self.index.add(embeddings.astype(np.float32))

    def retrieve(self, query: str, top_k=5) -> list:
        query_embedding = self.embedder.encode([f"query: {query}"], normalize_embeddings=True)
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "text": self.chunks[idx],
                "score": float(score),
                "metadata": self.chunk_metadata[idx]
            })
        return results

    def build_augmented_prompt(self, query: str, retrieved_chunks: list) -> str:
        context = "\n\n".join([
            f"[مصدر {i+1}]: {chunk['text']}" for i, chunk in enumerate(retrieved_chunks)
        ])
        return f"""استخدم المعلومات التالية فقط للإجابة على السؤال. إذا لم تجد الإجابة في المعلومات المتوفرة، صرّح بذلك بوضوح ولا تختلق معلومات.

المعلومات المتاحة:
{context}

السؤال: {query}

الإجابة (مع الإشارة لرقم المصدر عند الاستشهاد):"""

    def query(self, question: str, top_k=5) -> dict:
        retrieved = self.retrieve(question, top_k)
        prompt = self.build_augmented_prompt(question, retrieved)
        answer = self.generator.generate(prompt)
        return {"answer": answer, "sources": retrieved, "prompt_used": prompt}

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-10-applications-agents/ch47-rag-fundamentals.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import re

def semantic_chunking(text: str, max_chunk_size=512, embedder=None, similarity_threshold=0.5):
    """
    تجزئة أذكى: تُقسّم النص لجمل أولاً، ثم تُجمّع الجمل المتتالية
    المتشابهة دلاليًا في نفس القطعة، وتفصل عند نقاط التحوّل الموضوعي
    """
    sentences = re.split(r'(?<=[.!؟])\s+', text)
    if embedder is None or len(sentences) < 2:
        # fallback بسيط: تجميع حسب الحجم فقط مع احترام حدود الجمل
        chunks, current_chunk = [], ""
        for sentence in sentences:
            if len(current_chunk.split()) + len(sentence.split()) > max_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    # تجزئة دلالية حقيقية باستخدام تشابه embeddings متتالية
    sentence_embeddings = embedder.encode(sentences, normalize_embeddings=True)
    chunks, current_chunk_sentences = [], [sentences[0]]

    for i in range(1, len(sentences)):
        similarity = np.dot(sentence_embeddings[i], sentence_embeddings[i-1])
        current_length = sum(len(s.split()) for s in current_chunk_sentences)

        if similarity < similarity_threshold or current_length > max_chunk_size:
            chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = [sentences[i]]
        else:
            current_chunk_sentences.append(sentences[i])

    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))
    return chunks

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-10-applications-agents/ch47-rag-fundamentals.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def evaluate_rag_components_separately(rag_pipeline, test_set: list):
    """
    إطار RAGAS المبسّط: تقييم كل مكوّن من مكونات RAG بشكل مستقل
    """
    results = {"retrieval_metrics": [], "generation_metrics": []}

    for example in test_set:
        retrieved = rag_pipeline.retrieve(example["question"], top_k=5)

        # 1. Context Precision: هل المستندات المُسترجَعة ذات صلة فعلًا؟
        relevant_retrieved = sum(
            1 for r in retrieved if r["metadata"]["source_id"] in example["relevant_source_ids"]
        )
        context_precision = relevant_retrieved / len(retrieved) if retrieved else 0

        # 2. Context Recall: هل استرجعنا كل المعلومات اللازمة للإجابة؟
        all_relevant_found = all(
            any(r["metadata"]["source_id"] == sid for r in retrieved)
            for sid in example["relevant_source_ids"]
        )
        context_recall = 1.0 if all_relevant_found else 0.0

        results["retrieval_metrics"].append({
            "precision": context_precision, "recall": context_recall
        })

        # 3. Faithfulness: هل الإجابة مبنية فعلًا على السياق المُسترجَع (وليست هلوسة)؟
        answer = rag_pipeline.query(example["question"])["answer"]
        faithfulness_prompt = f"""هل كل ادّعاء في الإجابة التالية مدعوم بالسياق المُعطى؟
السياق: {[r['text'] for r in retrieved]}
الإجابة: {answer}
أجب بـ "نعم" أو "لا" مع ذكر أي ادّعاء غير مدعوم."""
        # يتطلب LLM-as-a-Judge (من الفصل 43)
        results["generation_metrics"].append({"faithfulness_check": faithfulness_prompt})

    avg_precision = np.mean([m["precision"] for m in results["retrieval_metrics"]])
    avg_recall = np.mean([m["recall"] for m in results["retrieval_metrics"]])
    return {"avg_context_precision": avg_precision, "avg_context_recall": avg_recall}

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-10-applications-agents/ch47-rag-fundamentals.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-47-ch47-rag-fundamentals.ipynb#cell-2
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
    ("faiss", "faiss-cpu"),
    ("numpy", "numpy"),
    ("sentence_transformers", "sentence-transformers")
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-47-ch47-rag-fundamentals.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class SimpleRAGPipeline:
    """
    تطبيق كامل لخط RAG أساسي — من تجزئة المستندات حتى توليد الإجابة
    """
    def __init__(self, embedding_model_name="intfloat/multilingual-e5-large",
                 generator_model=None, chunk_size=512, chunk_overlap=50):
        self.embedder = SentenceTransformer(embedding_model_name)
        self.generator = generator_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index = None
        self.chunks = []
        self.chunk_metadata = []

    def chunk_document(self, text: str, source_id: str) -> list:
        """
        تجزئة النص إلى قطع متراكبة (Overlapping Chunks)
        التراكب يمنع فقدان السياق عند قطع الجملة في منتصفها
        """
        words = text.split()
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            if len(chunk_words) < 20:   # تجاهل القطع الصغيرة جدًا في النهاية
                continue
            chunks.append({
                "text": " ".join(chunk_words),
                "source_id": source_id,
                "chunk_index": i // step
            })
        return chunks

    def index_documents(self, documents: dict):
        """documents: {source_id: full_text}"""
        all_chunks = []
        for source_id, text in documents.items():
            all_chunks.extend(self.chunk_document(text, source_id))

        self.chunks = [c["text"] for c in all_chunks]
        self.chunk_metadata = all_chunks

        # E5 يتطلب بادئة "passage: " للمستندات و "query: " للأسئلة (تفصيل مهم جدًا!)
        prefixed_chunks = [f"passage: {c}" for c in self.chunks]
        embeddings = self.embedder.encode(prefixed_chunks, normalize_embeddings=True,
                                            show_progress_bar=True)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)   # Inner Product = Cosine (بعد التطبيع)
        self.index.add(embeddings.astype(np.float32))

    def retrieve(self, query: str, top_k=5) -> list:
        query_embedding = self.embedder.encode([f"query: {query}"], normalize_embeddings=True)
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "text": self.chunks[idx],
                "score": float(score),
                "metadata": self.chunk_metadata[idx]
            })
        return results

    def build_augmented_prompt(self, query: str, retrieved_chunks: list) -> str:
        context = "\n\n".join([
            f"[مصدر {i+1}]: {chunk['text']}" for i, chunk in enumerate(retrieved_chunks)
        ])
        return f"""استخدم المعلومات التالية فقط للإجابة على السؤال. إذا لم تجد الإجابة في المعلومات المتوفرة، صرّح بذلك بوضوح ولا تختلق معلومات.

المعلومات المتاحة:
{context}

السؤال: {query}

الإجابة (مع الإشارة لرقم المصدر عند الاستشهاد):"""

    def query(self, question: str, top_k=5) -> dict:
        retrieved = self.retrieve(question, top_k)
        prompt = self.build_augmented_prompt(question, retrieved)
        answer = self.generator.generate(prompt)
        return {"answer": answer, "sources": retrieved, "prompt_used": prompt}

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-47-ch47-rag-fundamentals.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import re

def semantic_chunking(text: str, max_chunk_size=512, embedder=None, similarity_threshold=0.5):
    """
    تجزئة أذكى: تُقسّم النص لجمل أولاً، ثم تُجمّع الجمل المتتالية
    المتشابهة دلاليًا في نفس القطعة، وتفصل عند نقاط التحوّل الموضوعي
    """
    sentences = re.split(r'(?<=[.!؟])\s+', text)
    if embedder is None or len(sentences) < 2:
        # fallback بسيط: تجميع حسب الحجم فقط مع احترام حدود الجمل
        chunks, current_chunk = [], ""
        for sentence in sentences:
            if len(current_chunk.split()) + len(sentence.split()) > max_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    # تجزئة دلالية حقيقية باستخدام تشابه embeddings متتالية
    sentence_embeddings = embedder.encode(sentences, normalize_embeddings=True)
    chunks, current_chunk_sentences = [], [sentences[0]]

    for i in range(1, len(sentences)):
        similarity = np.dot(sentence_embeddings[i], sentence_embeddings[i-1])
        current_length = sum(len(s.split()) for s in current_chunk_sentences)

        if similarity < similarity_threshold or current_length > max_chunk_size:
            chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = [sentences[i]]
        else:
            current_chunk_sentences.append(sentences[i])

    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))
    return chunks

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-47-ch47-rag-fundamentals.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def evaluate_rag_components_separately(rag_pipeline, test_set: list):
    """
    إطار RAGAS المبسّط: تقييم كل مكوّن من مكونات RAG بشكل مستقل
    """
    results = {"retrieval_metrics": [], "generation_metrics": []}

    for example in test_set:
        retrieved = rag_pipeline.retrieve(example["question"], top_k=5)

        # 1. Context Precision: هل المستندات المُسترجَعة ذات صلة فعلًا؟
        relevant_retrieved = sum(
            1 for r in retrieved if r["metadata"]["source_id"] in example["relevant_source_ids"]
        )
        context_precision = relevant_retrieved / len(retrieved) if retrieved else 0

        # 2. Context Recall: هل استرجعنا كل المعلومات اللازمة للإجابة؟
        all_relevant_found = all(
            any(r["metadata"]["source_id"] == sid for r in retrieved)
            for sid in example["relevant_source_ids"]
        )
        context_recall = 1.0 if all_relevant_found else 0.0

        results["retrieval_metrics"].append({
            "precision": context_precision, "recall": context_recall
        })

        # 3. Faithfulness: هل الإجابة مبنية فعلًا على السياق المُسترجَع (وليست هلوسة)؟
        answer = rag_pipeline.query(example["question"])["answer"]
        faithfulness_prompt = f"""هل كل ادّعاء في الإجابة التالية مدعوم بالسياق المُعطى؟
السياق: {[r['text'] for r in retrieved]}
الإجابة: {answer}
أجب بـ "نعم" أو "لا" مع ذكر أي ادّعاء غير مدعوم."""
        # يتطلب LLM-as-a-Judge (من الفصل 43)
        results["generation_metrics"].append({"faithfulness_check": faithfulness_prompt})

    avg_precision = np.mean([m["precision"] for m in results["retrieval_metrics"]])
    avg_recall = np.mean([m["recall"] for m in results["retrieval_metrics"]])
    return {"avg_context_precision": avg_precision, "avg_context_recall": avg_recall}

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-47-ch47-rag-fundamentals.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
