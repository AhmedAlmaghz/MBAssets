#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — تقنيات RAG المتقدمة وGraphRAG
Mastering Transformers and Large Language Models — Advanced Rag

Book slug : mastering-llms-transformers
Chapter   : ch-48-ch48-advanced-rag (lesson 48)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-48-ch48-advanced-rag.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — تقنيات RAG المتقدمة وGraphRAG
Mastering Transformers and Large Language Models — Advanced Rag

Book slug : mastering-llms-transformers
Chapter   : ch-48-ch48-advanced-rag (lesson 48)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-48-ch48-advanced-rag.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — تقنيات RAG المتقدمة وGraphRAG
Mastering Transformers and Large Language Models — Advanced Rag

Book slug : mastering-llms-transformers
Chapter   : ch-48-ch48-advanced-rag (lesson 48)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-10-applications-agents/ch48-advanced-rag.ipynb#cell-1
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
    ("networkx", "networkx"),
    ("rank_bm25", "rank-bm25"),
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
# [002] content/notebooks/part-10-applications-agents/ch48-advanced-rag.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sentence_transformers import CrossEncoder

class RerankingRAGPipeline(SimpleRAGPipeline):
    def __init__(self, *args, reranker_model="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", **kwargs):
        super().__init__(*args, **kwargs)
        self.reranker = CrossEncoder(reranker_model)

    def retrieve_with_reranking(self, query: str, initial_k=25, final_k=5) -> list:
        # المرحلة 1: استرجاع واسع وسريع (Bi-encoder)
        candidates = self.retrieve(query, top_k=initial_k)

        # المرحلة 2: إعادة ترتيب دقيقة (Cross-encoder يرى السؤال والمستند معًا)
        pairs = [[query, c["text"]] for c in candidates]
        rerank_scores = self.reranker.predict(pairs)

        for candidate, score in zip(candidates, rerank_scores):
            candidate["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:final_k]

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-10-applications-agents/ch48-advanced-rag.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from rank_bm25 import BM25Okapi

class HybridSearchRAG(RerankingRAGPipeline):
    """
    دمج البحث الدلالي (Dense/Embedding) مع البحث اللفظي (Sparse/BM25)
    مهم جدًا للعربية حيث البحث الدلالي قد يفوّت تطابقات حرفية دقيقة
    (أسماء أعلام، مصطلحات تقنية، أرقام)
    """
    def index_documents(self, documents: dict):
        super().index_documents(documents)
        tokenized_chunks = [chunk.split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

    def hybrid_retrieve(self, query: str, top_k=10, alpha=0.5) -> list:
        """alpha: وزن الموازنة بين البحث الدلالي (1.0) واللفظي (0.0)"""
        # البحث الدلالي
        dense_results = self.retrieve(query, top_k=top_k * 2)
        dense_scores = {r["metadata"]["chunk_index"]: r["score"] for r in dense_results}

        # البحث اللفظي BM25
        bm25_scores_raw = self.bm25.get_scores(query.split())
        # تطبيع BM25 scores إلى [0, 1]
        max_bm25 = max(bm25_scores_raw) if max(bm25_scores_raw) > 0 else 1
        bm25_scores = {i: score / max_bm25 for i, score in enumerate(bm25_scores_raw)}

        # دمج النتائج بالموازنة الخطية (Reciprocal Rank Fusion أو Weighted Sum)
        all_indices = set(dense_scores.keys()) | set(bm25_scores.keys())
        combined_scores = {
            idx: alpha * dense_scores.get(idx, 0) + (1 - alpha) * bm25_scores.get(idx, 0)
            for idx in all_indices
        }

        top_indices = sorted(combined_scores, key=combined_scores.get, reverse=True)[:top_k]
        return [{"text": self.chunks[i], "combined_score": combined_scores[i]} for i in top_indices]

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-10-applications-agents/ch48-advanced-rag.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import networkx as nx

class GraphRAGPipeline:
    """
    تطبيق مبسّط لفكرة GraphRAG:
    1. استخراج الكيانات والعلاقات من المستندات (عبر LLM)
    2. بناء رسم بياني للمعرفة
    3. عند الاستعلام: استرجاع عبر المسارات في الرسم، وليس فقط التشابه النصي
    """
    def __init__(self, llm_extractor):
        self.llm_extractor = llm_extractor   # LLM يُستخدم لاستخراج الكيانات/العلاقات
        self.graph = nx.MultiDiGraph()

    def extract_entities_and_relations(self, text_chunk: str) -> list:
        extraction_prompt = f"""استخرج الكيانات (أشخاص، منظمات، أماكن، مفاهيم) والعلاقات بينها
من النص التالي بصيغة JSON: [{{"source": "", "relation": "", "target": ""}}]

النص: {text_chunk}
"""
        response = self.llm_extractor.generate(extraction_prompt)
        import json
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return []

    def build_graph_from_documents(self, chunks: list):
        for chunk in chunks:
            triples = self.extract_entities_and_relations(chunk)
            for triple in triples:
                self.graph.add_edge(
                    triple["source"], triple["target"],
                    relation=triple["relation"], source_text=chunk
                )

    def community_summarization(self):
        """
        خطوة GraphRAG المحورية: تقسيم الرسم لمجتمعات (communities) مترابطة
        وتلخيص كل مجتمع مسبقًا — يسمح بالإجابة على أسئلة "شاملة" (global queries)
        لا يمكن لـ chunk واحد الإجابة عنها
        """
        undirected = self.graph.to_undirected()
        communities = nx.community.louvain_communities(undirected)

        community_summaries = {}
        for i, community in enumerate(communities):
            subgraph = self.graph.subgraph(community)
            edges_text = "\n".join([
                f"{u} -[{d['relation']}]-> {v}" for u, v, d in subgraph.edges(data=True)
            ])
            summary_prompt = f"لخّص العلاقات التالية في فقرة متماسكة:\n{edges_text}"
            community_summaries[i] = self.llm_extractor.generate(summary_prompt)

        return community_summaries

    def graph_guided_retrieval(self, query_entities: list, max_hops=2) -> list:
        """استرجاع كل العقد المرتبطة بالكيانات المذكورة في السؤال ضمن مسافة معيّنة"""
        relevant_subgraph_nodes = set()
        for entity in query_entities:
            if entity in self.graph:
                neighbors = nx.single_source_shortest_path_length(
                    self.graph.to_undirected(), entity, cutoff=max_hops
                )
                relevant_subgraph_nodes.update(neighbors.keys())

        relevant_edges = [
            (u, v, d) for u, v, d in self.graph.edges(data=True)
            if u in relevant_subgraph_nodes and v in relevant_subgraph_nodes
        ]
        return relevant_edges

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-10-applications-agents/ch48-advanced-rag.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def agentic_rag_loop(llm, rag_pipeline, question: str, max_iterations=3):
    """
    حلقة استرجاع تكرارية ذكية: النموذج يقيّم هل المعلومات كافية،
    وإن لم تكن، يُعيد صياغة استعلام بحث جديد ويحاول مجددًا
    """
    accumulated_context = []
    current_query = question

    for iteration in range(max_iterations):
        retrieved = rag_pipeline.retrieve_with_reranking(current_query, final_k=3)
        accumulated_context.extend(retrieved)

        sufficiency_prompt = f"""السؤال الأصلي: {question}
المعلومات المتجمّعة حتى الآن: {[c['text'][:200] for c in accumulated_context]}

هل هذه المعلومات كافية للإجابة الكاملة على السؤال؟
إذا كانت كافية، أجب بـ: "كافٍ"
إذا لم تكن كافية، أجب بـ: "غير كافٍ: [استعلام بحث جديد أكثر تحديدًا]"
"""
        decision = llm.generate(sufficiency_prompt)

        if decision.strip().startswith("كافٍ"):
            break
        elif "غير كافٍ:" in decision:
            current_query = decision.split("غير كافٍ:")[-1].strip()
        else:
            break

    final_prompt = rag_pipeline.build_augmented_prompt(question, accumulated_context)
    return llm.generate(final_prompt)

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-10-applications-agents/ch48-advanced-rag.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-48-ch48-advanced-rag.ipynb#cell-1
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
    ("networkx", "networkx"),
    ("rank_bm25", "rank-bm25"),
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-48-ch48-advanced-rag.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sentence_transformers import CrossEncoder

class RerankingRAGPipeline(SimpleRAGPipeline):
    def __init__(self, *args, reranker_model="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", **kwargs):
        super().__init__(*args, **kwargs)
        self.reranker = CrossEncoder(reranker_model)

    def retrieve_with_reranking(self, query: str, initial_k=25, final_k=5) -> list:
        # المرحلة 1: استرجاع واسع وسريع (Bi-encoder)
        candidates = self.retrieve(query, top_k=initial_k)

        # المرحلة 2: إعادة ترتيب دقيقة (Cross-encoder يرى السؤال والمستند معًا)
        pairs = [[query, c["text"]] for c in candidates]
        rerank_scores = self.reranker.predict(pairs)

        for candidate, score in zip(candidates, rerank_scores):
            candidate["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:final_k]

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-48-ch48-advanced-rag.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from rank_bm25 import BM25Okapi

class HybridSearchRAG(RerankingRAGPipeline):
    """
    دمج البحث الدلالي (Dense/Embedding) مع البحث اللفظي (Sparse/BM25)
    مهم جدًا للعربية حيث البحث الدلالي قد يفوّت تطابقات حرفية دقيقة
    (أسماء أعلام، مصطلحات تقنية، أرقام)
    """
    def index_documents(self, documents: dict):
        super().index_documents(documents)
        tokenized_chunks = [chunk.split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

    def hybrid_retrieve(self, query: str, top_k=10, alpha=0.5) -> list:
        """alpha: وزن الموازنة بين البحث الدلالي (1.0) واللفظي (0.0)"""
        # البحث الدلالي
        dense_results = self.retrieve(query, top_k=top_k * 2)
        dense_scores = {r["metadata"]["chunk_index"]: r["score"] for r in dense_results}

        # البحث اللفظي BM25
        bm25_scores_raw = self.bm25.get_scores(query.split())
        # تطبيع BM25 scores إلى [0, 1]
        max_bm25 = max(bm25_scores_raw) if max(bm25_scores_raw) > 0 else 1
        bm25_scores = {i: score / max_bm25 for i, score in enumerate(bm25_scores_raw)}

        # دمج النتائج بالموازنة الخطية (Reciprocal Rank Fusion أو Weighted Sum)
        all_indices = set(dense_scores.keys()) | set(bm25_scores.keys())
        combined_scores = {
            idx: alpha * dense_scores.get(idx, 0) + (1 - alpha) * bm25_scores.get(idx, 0)
            for idx in all_indices
        }

        top_indices = sorted(combined_scores, key=combined_scores.get, reverse=True)[:top_k]
        return [{"text": self.chunks[i], "combined_score": combined_scores[i]} for i in top_indices]

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-48-ch48-advanced-rag.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import networkx as nx

class GraphRAGPipeline:
    """
    تطبيق مبسّط لفكرة GraphRAG:
    1. استخراج الكيانات والعلاقات من المستندات (عبر LLM)
    2. بناء رسم بياني للمعرفة
    3. عند الاستعلام: استرجاع عبر المسارات في الرسم، وليس فقط التشابه النصي
    """
    def __init__(self, llm_extractor):
        self.llm_extractor = llm_extractor   # LLM يُستخدم لاستخراج الكيانات/العلاقات
        self.graph = nx.MultiDiGraph()

    def extract_entities_and_relations(self, text_chunk: str) -> list:
        extraction_prompt = f"""استخرج الكيانات (أشخاص، منظمات، أماكن، مفاهيم) والعلاقات بينها
من النص التالي بصيغة JSON: [{{"source": "", "relation": "", "target": ""}}]

النص: {text_chunk}
"""
        response = self.llm_extractor.generate(extraction_prompt)
        import json
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return []

    def build_graph_from_documents(self, chunks: list):
        for chunk in chunks:
            triples = self.extract_entities_and_relations(chunk)
            for triple in triples:
                self.graph.add_edge(
                    triple["source"], triple["target"],
                    relation=triple["relation"], source_text=chunk
                )

    def community_summarization(self):
        """
        خطوة GraphRAG المحورية: تقسيم الرسم لمجتمعات (communities) مترابطة
        وتلخيص كل مجتمع مسبقًا — يسمح بالإجابة على أسئلة "شاملة" (global queries)
        لا يمكن لـ chunk واحد الإجابة عنها
        """
        undirected = self.graph.to_undirected()
        communities = nx.community.louvain_communities(undirected)

        community_summaries = {}
        for i, community in enumerate(communities):
            subgraph = self.graph.subgraph(community)
            edges_text = "\n".join([
                f"{u} -[{d['relation']}]-> {v}" for u, v, d in subgraph.edges(data=True)
            ])
            summary_prompt = f"لخّص العلاقات التالية في فقرة متماسكة:\n{edges_text}"
            community_summaries[i] = self.llm_extractor.generate(summary_prompt)

        return community_summaries

    def graph_guided_retrieval(self, query_entities: list, max_hops=2) -> list:
        """استرجاع كل العقد المرتبطة بالكيانات المذكورة في السؤال ضمن مسافة معيّنة"""
        relevant_subgraph_nodes = set()
        for entity in query_entities:
            if entity in self.graph:
                neighbors = nx.single_source_shortest_path_length(
                    self.graph.to_undirected(), entity, cutoff=max_hops
                )
                relevant_subgraph_nodes.update(neighbors.keys())

        relevant_edges = [
            (u, v, d) for u, v, d in self.graph.edges(data=True)
            if u in relevant_subgraph_nodes and v in relevant_subgraph_nodes
        ]
        return relevant_edges

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-48-ch48-advanced-rag.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def agentic_rag_loop(llm, rag_pipeline, question: str, max_iterations=3):
    """
    حلقة استرجاع تكرارية ذكية: النموذج يقيّم هل المعلومات كافية،
    وإن لم تكن، يُعيد صياغة استعلام بحث جديد ويحاول مجددًا
    """
    accumulated_context = []
    current_query = question

    for iteration in range(max_iterations):
        retrieved = rag_pipeline.retrieve_with_reranking(current_query, final_k=3)
        accumulated_context.extend(retrieved)

        sufficiency_prompt = f"""السؤال الأصلي: {question}
المعلومات المتجمّعة حتى الآن: {[c['text'][:200] for c in accumulated_context]}

هل هذه المعلومات كافية للإجابة الكاملة على السؤال؟
إذا كانت كافية، أجب بـ: "كافٍ"
إذا لم تكن كافية، أجب بـ: "غير كافٍ: [استعلام بحث جديد أكثر تحديدًا]"
"""
        decision = llm.generate(sufficiency_prompt)

        if decision.strip().startswith("كافٍ"):
            break
        elif "غير كافٍ:" in decision:
            current_query = decision.split("غير كافٍ:")[-1].strip()
        else:
            break

    final_prompt = rag_pipeline.build_augmented_prompt(question, accumulated_context)
    return llm.generate(final_prompt)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-48-ch48-advanced-rag.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-48-ch48-advanced-rag.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — تقنيات RAG المتقدمة وGraphRAG
Mastering Transformers and Large Language Models — Advanced Rag

Book slug : mastering-llms-transformers
Chapter   : ch-48-ch48-advanced-rag (lesson 48)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-10-applications-agents/ch48-advanced-rag.ipynb#cell-1
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
    ("networkx", "networkx"),
    ("rank_bm25", "rank-bm25"),
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
# [002] content/notebooks/part-10-applications-agents/ch48-advanced-rag.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sentence_transformers import CrossEncoder

class RerankingRAGPipeline(SimpleRAGPipeline):
    def __init__(self, *args, reranker_model="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", **kwargs):
        super().__init__(*args, **kwargs)
        self.reranker = CrossEncoder(reranker_model)

    def retrieve_with_reranking(self, query: str, initial_k=25, final_k=5) -> list:
        # المرحلة 1: استرجاع واسع وسريع (Bi-encoder)
        candidates = self.retrieve(query, top_k=initial_k)

        # المرحلة 2: إعادة ترتيب دقيقة (Cross-encoder يرى السؤال والمستند معًا)
        pairs = [[query, c["text"]] for c in candidates]
        rerank_scores = self.reranker.predict(pairs)

        for candidate, score in zip(candidates, rerank_scores):
            candidate["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:final_k]

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-10-applications-agents/ch48-advanced-rag.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from rank_bm25 import BM25Okapi

class HybridSearchRAG(RerankingRAGPipeline):
    """
    دمج البحث الدلالي (Dense/Embedding) مع البحث اللفظي (Sparse/BM25)
    مهم جدًا للعربية حيث البحث الدلالي قد يفوّت تطابقات حرفية دقيقة
    (أسماء أعلام، مصطلحات تقنية، أرقام)
    """
    def index_documents(self, documents: dict):
        super().index_documents(documents)
        tokenized_chunks = [chunk.split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

    def hybrid_retrieve(self, query: str, top_k=10, alpha=0.5) -> list:
        """alpha: وزن الموازنة بين البحث الدلالي (1.0) واللفظي (0.0)"""
        # البحث الدلالي
        dense_results = self.retrieve(query, top_k=top_k * 2)
        dense_scores = {r["metadata"]["chunk_index"]: r["score"] for r in dense_results}

        # البحث اللفظي BM25
        bm25_scores_raw = self.bm25.get_scores(query.split())
        # تطبيع BM25 scores إلى [0, 1]
        max_bm25 = max(bm25_scores_raw) if max(bm25_scores_raw) > 0 else 1
        bm25_scores = {i: score / max_bm25 for i, score in enumerate(bm25_scores_raw)}

        # دمج النتائج بالموازنة الخطية (Reciprocal Rank Fusion أو Weighted Sum)
        all_indices = set(dense_scores.keys()) | set(bm25_scores.keys())
        combined_scores = {
            idx: alpha * dense_scores.get(idx, 0) + (1 - alpha) * bm25_scores.get(idx, 0)
            for idx in all_indices
        }

        top_indices = sorted(combined_scores, key=combined_scores.get, reverse=True)[:top_k]
        return [{"text": self.chunks[i], "combined_score": combined_scores[i]} for i in top_indices]

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-10-applications-agents/ch48-advanced-rag.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import networkx as nx

class GraphRAGPipeline:
    """
    تطبيق مبسّط لفكرة GraphRAG:
    1. استخراج الكيانات والعلاقات من المستندات (عبر LLM)
    2. بناء رسم بياني للمعرفة
    3. عند الاستعلام: استرجاع عبر المسارات في الرسم، وليس فقط التشابه النصي
    """
    def __init__(self, llm_extractor):
        self.llm_extractor = llm_extractor   # LLM يُستخدم لاستخراج الكيانات/العلاقات
        self.graph = nx.MultiDiGraph()

    def extract_entities_and_relations(self, text_chunk: str) -> list:
        extraction_prompt = f"""استخرج الكيانات (أشخاص، منظمات، أماكن، مفاهيم) والعلاقات بينها
من النص التالي بصيغة JSON: [{{"source": "", "relation": "", "target": ""}}]

النص: {text_chunk}
"""
        response = self.llm_extractor.generate(extraction_prompt)
        import json
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return []

    def build_graph_from_documents(self, chunks: list):
        for chunk in chunks:
            triples = self.extract_entities_and_relations(chunk)
            for triple in triples:
                self.graph.add_edge(
                    triple["source"], triple["target"],
                    relation=triple["relation"], source_text=chunk
                )

    def community_summarization(self):
        """
        خطوة GraphRAG المحورية: تقسيم الرسم لمجتمعات (communities) مترابطة
        وتلخيص كل مجتمع مسبقًا — يسمح بالإجابة على أسئلة "شاملة" (global queries)
        لا يمكن لـ chunk واحد الإجابة عنها
        """
        undirected = self.graph.to_undirected()
        communities = nx.community.louvain_communities(undirected)

        community_summaries = {}
        for i, community in enumerate(communities):
            subgraph = self.graph.subgraph(community)
            edges_text = "\n".join([
                f"{u} -[{d['relation']}]-> {v}" for u, v, d in subgraph.edges(data=True)
            ])
            summary_prompt = f"لخّص العلاقات التالية في فقرة متماسكة:\n{edges_text}"
            community_summaries[i] = self.llm_extractor.generate(summary_prompt)

        return community_summaries

    def graph_guided_retrieval(self, query_entities: list, max_hops=2) -> list:
        """استرجاع كل العقد المرتبطة بالكيانات المذكورة في السؤال ضمن مسافة معيّنة"""
        relevant_subgraph_nodes = set()
        for entity in query_entities:
            if entity in self.graph:
                neighbors = nx.single_source_shortest_path_length(
                    self.graph.to_undirected(), entity, cutoff=max_hops
                )
                relevant_subgraph_nodes.update(neighbors.keys())

        relevant_edges = [
            (u, v, d) for u, v, d in self.graph.edges(data=True)
            if u in relevant_subgraph_nodes and v in relevant_subgraph_nodes
        ]
        return relevant_edges

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-10-applications-agents/ch48-advanced-rag.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def agentic_rag_loop(llm, rag_pipeline, question: str, max_iterations=3):
    """
    حلقة استرجاع تكرارية ذكية: النموذج يقيّم هل المعلومات كافية،
    وإن لم تكن، يُعيد صياغة استعلام بحث جديد ويحاول مجددًا
    """
    accumulated_context = []
    current_query = question

    for iteration in range(max_iterations):
        retrieved = rag_pipeline.retrieve_with_reranking(current_query, final_k=3)
        accumulated_context.extend(retrieved)

        sufficiency_prompt = f"""السؤال الأصلي: {question}
المعلومات المتجمّعة حتى الآن: {[c['text'][:200] for c in accumulated_context]}

هل هذه المعلومات كافية للإجابة الكاملة على السؤال؟
إذا كانت كافية، أجب بـ: "كافٍ"
إذا لم تكن كافية، أجب بـ: "غير كافٍ: [استعلام بحث جديد أكثر تحديدًا]"
"""
        decision = llm.generate(sufficiency_prompt)

        if decision.strip().startswith("كافٍ"):
            break
        elif "غير كافٍ:" in decision:
            current_query = decision.split("غير كافٍ:")[-1].strip()
        else:
            break

    final_prompt = rag_pipeline.build_augmented_prompt(question, accumulated_context)
    return llm.generate(final_prompt)

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-10-applications-agents/ch48-advanced-rag.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-48-ch48-advanced-rag.ipynb#cell-2
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
    ("networkx", "networkx"),
    ("rank_bm25", "rank-bm25"),
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-48-ch48-advanced-rag.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sentence_transformers import CrossEncoder

class RerankingRAGPipeline(SimpleRAGPipeline):
    def __init__(self, *args, reranker_model="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", **kwargs):
        super().__init__(*args, **kwargs)
        self.reranker = CrossEncoder(reranker_model)

    def retrieve_with_reranking(self, query: str, initial_k=25, final_k=5) -> list:
        # المرحلة 1: استرجاع واسع وسريع (Bi-encoder)
        candidates = self.retrieve(query, top_k=initial_k)

        # المرحلة 2: إعادة ترتيب دقيقة (Cross-encoder يرى السؤال والمستند معًا)
        pairs = [[query, c["text"]] for c in candidates]
        rerank_scores = self.reranker.predict(pairs)

        for candidate, score in zip(candidates, rerank_scores):
            candidate["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:final_k]

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-48-ch48-advanced-rag.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from rank_bm25 import BM25Okapi

class HybridSearchRAG(RerankingRAGPipeline):
    """
    دمج البحث الدلالي (Dense/Embedding) مع البحث اللفظي (Sparse/BM25)
    مهم جدًا للعربية حيث البحث الدلالي قد يفوّت تطابقات حرفية دقيقة
    (أسماء أعلام، مصطلحات تقنية، أرقام)
    """
    def index_documents(self, documents: dict):
        super().index_documents(documents)
        tokenized_chunks = [chunk.split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

    def hybrid_retrieve(self, query: str, top_k=10, alpha=0.5) -> list:
        """alpha: وزن الموازنة بين البحث الدلالي (1.0) واللفظي (0.0)"""
        # البحث الدلالي
        dense_results = self.retrieve(query, top_k=top_k * 2)
        dense_scores = {r["metadata"]["chunk_index"]: r["score"] for r in dense_results}

        # البحث اللفظي BM25
        bm25_scores_raw = self.bm25.get_scores(query.split())
        # تطبيع BM25 scores إلى [0, 1]
        max_bm25 = max(bm25_scores_raw) if max(bm25_scores_raw) > 0 else 1
        bm25_scores = {i: score / max_bm25 for i, score in enumerate(bm25_scores_raw)}

        # دمج النتائج بالموازنة الخطية (Reciprocal Rank Fusion أو Weighted Sum)
        all_indices = set(dense_scores.keys()) | set(bm25_scores.keys())
        combined_scores = {
            idx: alpha * dense_scores.get(idx, 0) + (1 - alpha) * bm25_scores.get(idx, 0)
            for idx in all_indices
        }

        top_indices = sorted(combined_scores, key=combined_scores.get, reverse=True)[:top_k]
        return [{"text": self.chunks[i], "combined_score": combined_scores[i]} for i in top_indices]

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-48-ch48-advanced-rag.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import networkx as nx

class GraphRAGPipeline:
    """
    تطبيق مبسّط لفكرة GraphRAG:
    1. استخراج الكيانات والعلاقات من المستندات (عبر LLM)
    2. بناء رسم بياني للمعرفة
    3. عند الاستعلام: استرجاع عبر المسارات في الرسم، وليس فقط التشابه النصي
    """
    def __init__(self, llm_extractor):
        self.llm_extractor = llm_extractor   # LLM يُستخدم لاستخراج الكيانات/العلاقات
        self.graph = nx.MultiDiGraph()

    def extract_entities_and_relations(self, text_chunk: str) -> list:
        extraction_prompt = f"""استخرج الكيانات (أشخاص، منظمات، أماكن، مفاهيم) والعلاقات بينها
من النص التالي بصيغة JSON: [{{"source": "", "relation": "", "target": ""}}]

النص: {text_chunk}
"""
        response = self.llm_extractor.generate(extraction_prompt)
        import json
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return []

    def build_graph_from_documents(self, chunks: list):
        for chunk in chunks:
            triples = self.extract_entities_and_relations(chunk)
            for triple in triples:
                self.graph.add_edge(
                    triple["source"], triple["target"],
                    relation=triple["relation"], source_text=chunk
                )

    def community_summarization(self):
        """
        خطوة GraphRAG المحورية: تقسيم الرسم لمجتمعات (communities) مترابطة
        وتلخيص كل مجتمع مسبقًا — يسمح بالإجابة على أسئلة "شاملة" (global queries)
        لا يمكن لـ chunk واحد الإجابة عنها
        """
        undirected = self.graph.to_undirected()
        communities = nx.community.louvain_communities(undirected)

        community_summaries = {}
        for i, community in enumerate(communities):
            subgraph = self.graph.subgraph(community)
            edges_text = "\n".join([
                f"{u} -[{d['relation']}]-> {v}" for u, v, d in subgraph.edges(data=True)
            ])
            summary_prompt = f"لخّص العلاقات التالية في فقرة متماسكة:\n{edges_text}"
            community_summaries[i] = self.llm_extractor.generate(summary_prompt)

        return community_summaries

    def graph_guided_retrieval(self, query_entities: list, max_hops=2) -> list:
        """استرجاع كل العقد المرتبطة بالكيانات المذكورة في السؤال ضمن مسافة معيّنة"""
        relevant_subgraph_nodes = set()
        for entity in query_entities:
            if entity in self.graph:
                neighbors = nx.single_source_shortest_path_length(
                    self.graph.to_undirected(), entity, cutoff=max_hops
                )
                relevant_subgraph_nodes.update(neighbors.keys())

        relevant_edges = [
            (u, v, d) for u, v, d in self.graph.edges(data=True)
            if u in relevant_subgraph_nodes and v in relevant_subgraph_nodes
        ]
        return relevant_edges

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-48-ch48-advanced-rag.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def agentic_rag_loop(llm, rag_pipeline, question: str, max_iterations=3):
    """
    حلقة استرجاع تكرارية ذكية: النموذج يقيّم هل المعلومات كافية،
    وإن لم تكن، يُعيد صياغة استعلام بحث جديد ويحاول مجددًا
    """
    accumulated_context = []
    current_query = question

    for iteration in range(max_iterations):
        retrieved = rag_pipeline.retrieve_with_reranking(current_query, final_k=3)
        accumulated_context.extend(retrieved)

        sufficiency_prompt = f"""السؤال الأصلي: {question}
المعلومات المتجمّعة حتى الآن: {[c['text'][:200] for c in accumulated_context]}

هل هذه المعلومات كافية للإجابة الكاملة على السؤال؟
إذا كانت كافية، أجب بـ: "كافٍ"
إذا لم تكن كافية، أجب بـ: "غير كافٍ: [استعلام بحث جديد أكثر تحديدًا]"
"""
        decision = llm.generate(sufficiency_prompt)

        if decision.strip().startswith("كافٍ"):
            break
        elif "غير كافٍ:" in decision:
            current_query = decision.split("غير كافٍ:")[-1].strip()
        else:
            break

    final_prompt = rag_pipeline.build_augmented_prompt(question, accumulated_context)
    return llm.generate(final_prompt)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-llms-transformers/notebooks/ch-48-ch48-advanced-rag.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
