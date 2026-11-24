#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — تقييم أنظمة RAG (RAG Evaluation)
Large Language Model Engineering — RAG

Book slug : llm-engineering-69-chapters
Chapter   : ch-48-chapter-48 (lesson 48)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-48-chapter-48.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — تقييم أنظمة RAG (RAG Evaluation)
Large Language Model Engineering — RAG

Book slug : llm-engineering-69-chapters
Chapter   : ch-48-chapter-48 (lesson 48)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-48-chapter-48.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — تقييم أنظمة RAG (RAG Evaluation)
Large Language Model Engineering — RAG

Book slug : llm-engineering-69-chapters
Chapter   : ch-48-chapter-48 (lesson 48)
Code files: 2 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_48.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 48: الفصل 48: تقييم أنظمة RAG (RAG Evaluation)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_rag_eval() -> None:
    relevant, retrieved, supported = ({'d1', 'd2'}, {'d1', 'd3'}, {'d1'})
    recall = len(relevant & retrieved) / len(relevant)
    precision = len(relevant & retrieved) / len(retrieved)
    grounding = len(supported) / max(len(retrieved), 1)
    print('context recall:', round(recall, 3))
    print('context precision:', round(precision, 3))
    print('supported retrieved evidence:', round(grounding, 3))

if __name__ == "__main__":
    print("Chapter 48 lab: Evaluate retrieval coverage and answer grounding")
    run_rag_eval()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-09-evaluation-testing/chapter-48.md
# ────────────────────────────────────────────────────────────────────────
"""
الفصل 48: تقييم أنظمة RAG (RAG Evaluation)
تنفيذ كامل لإطار RAGAS: Context Precision, Context Recall,
Faithfulness, Answer Relevancy — من الصفر، بدون أي مكتبة جاهزة للمقاييس.

المتطلبات:
    pip install torch transformers openai numpy

يتطلب التشغيل الفعلي: OPENAI_API_KEY في متغيرات البيئة.
"""

import os
import re
import json
import numpy as np
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
from dataclasses import dataclass
from typing import List, Dict, Optional


# ============================================================
# الجزء 1: محرك التضمين (Embeddings) من الصفر باستخدام PyTorch
# ============================================================

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
_embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)
_embed_model.eval()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_embed_model.to(DEVICE)


def mean_pooling(model_output, attention_mask: Tensor) -> Tensor:
    """تجميع متوسط الرموز مع استبعاد رموز الحشو عبر قناع الانتباه."""
    token_embeddings = model_output[0]  # [batch, seq_len, hidden]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


@torch.no_grad()
def embed_texts(texts: List[str], batch_size: int = 16) -> np.ndarray:
    """يحوّل قائمة نصوص إلى مصفوفة تضمينات (N, 384) مُطبّعة L2."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = _tokenizer(
            batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
        ).to(DEVICE)
        output = _embed_model(**encoded)
        pooled = mean_pooling(output, encoded["attention_mask"])
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        all_embeddings.append(normalized.cpu().numpy())
    return np.vstack(all_embeddings)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ============================================================
# الجزء 2: مخزن متجهات بسيط (Vector Store) للمسترجِع
# ============================================================

class SimpleVectorStore:
    """مخزن متجهات بسيط في الذاكرة، يكفي للتوضيح التعليمي."""

    def __init__(self):
        self.documents: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_documents(self, docs: List[str]) -> None:
        self.documents.extend(docs)
        self.embeddings = embed_texts(self.documents)

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        q_emb = embed_texts([query])[0]
        sims = self.embeddings @ q_emb  # مُطبّع مسبقًا → الضرب = جيب التمام
        top_k_idx = np.argsort(-sims)[:k]
        return [self.documents[i] for i in top_k_idx]


# ============================================================
# الجزء 3: عميل LLM (للتوليد وللحكم)
# ============================================================

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
JUDGE_MODEL = "gpt-4o-mini"
GENERATOR_MODEL = "gpt-4o-mini"


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.0,
             model: str = JUDGE_MODEL) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def generate_rag_answer(question: str, contexts: List[str]) -> str:
    """مولّد الإجابة: يعتمد فقط على السياق المُعطى له، لا أكثر."""
    context_block = "\n\n".join(f"[مصدر {i+1}]: {c}" for i, c in enumerate(contexts))
    system = (
        "أنت مساعد يجيب فقط بالاعتماد على المصادر المُعطاة أدناه. "
        "إن لم تجد الإجابة الكافية فيها، قل ذلك صراحة ولا تخترع معلومات."
    )
    user = f"المصادر:\n{context_block}\n\nالسؤال: {question}\n\nالإجابة:"
    return call_llm(system, user, temperature=0.0, model=GENERATOR_MODEL)


def rag_pipeline(question: str, store: SimpleVectorStore, k: int = 3) -> Dict:
    contexts = store.retrieve(question, k=k)
    answer = generate_rag_answer(question, contexts)
    return {"question": question, "contexts": contexts, "answer": answer}


# ============================================================
# الجزء 4: أدوات مشتركة — تفكيك الادعاءات والحكم عليها
# ============================================================

def _split_into_claims(text: str) -> List[str]:
    """
    يفكك نصًا إلى ادعاءات ذرية (Atomic Claims) عبر LLM.
    أدق من التقسيم بالنقطة لأن الجملة الواحدة قد تحمل أكثر من ادعاء مستقل.
    """
    system = (
        "فكك النص التالي إلى قائمة من الادعاءات الذرية (Atomic Claims) المستقلة، "
        "كل ادعاء يحمل معلومة واحدة فقط قابلة للتحقق بمفردها. "
        "أعد النتيجة كـ JSON array من النصوص فقط، دون أي شرح إضافي."
    )
    user = f"النص: {text}"
    raw = call_llm(system, user)
    try:
        claims = json.loads(raw)
        if isinstance(claims, list):
            return [str(c).strip() for c in claims if str(c).strip()]
    except json.JSONDecodeError:
        pass
    return [s.strip() for s in re.split(r"[.؟!]", text) if s.strip()]


def _judge_claim_attributable(claim: str, context_block: str) -> int:
    """هل يمكن استنتاج الادعاء من السياق المرجعي المُعطى فقط؟"""
    system = (
        "أنت مُقيّم دقيق. هل يمكن استنتاج الادعاء التالي من النص المرجعي المُعطى فقط؟ "
        "أجب 1 إن كان مدعومًا بالكامل بالنص، أو 0 إن لم يكن مذكورًا أو مدعومًا فيه. "
        "أعد فقط الرقم 0 أو 1 دون أي شرح."
    )
    user = f"النص المرجعي:\n{context_block}\n\nالادعاء: {claim}"
    verdict = call_llm(system, user)
    match = re.search(r"[01]", verdict)
    return int(match.group()) if match else 0


def _judge_context_relevance(question: str, ground_truth: str, context: str) -> int:
    """هل هذه القطعة المحددة ضرورية/مفيدة للإجابة على السؤال؟"""
    system = (
        "أنت مُقيّم دقيق. احكم هل القطعة النصية (context) ضرورية أو مفيدة "
        "للإجابة على السؤال، بالنظر إلى الإجابة المرجعية الصحيحة (ground truth). "
        "أجب 1 إن كانت ذات صلة ومفيدة، أو 0 إن كانت غير ذات صلة. "
        "أعد فقط الرقم 0 أو 1."
    )
    user = f"السؤال: {question}\nالإجابة المرجعية: {ground_truth}\nالقطعة: {context}"
    verdict = call_llm(system, user)
    match = re.search(r"[01]", verdict)
    return int(match.group()) if match else 0


# ============================================================
# الجزء 5: المقاييس الأربعة لـ RAGAS
# ============================================================

def context_precision_at_k(question: str, ground_truth: str,
                            ranked_contexts: List[str]) -> float:
    """
    Context Precision@K: دقة الاسترجاع المرجّحة بالرتبة.
    يعاقب القطع غير ذات الصلة بشدة أكبر كلما ظهرت في رتب مبكرة.
    """
    K = len(ranked_contexts)
    relevance = [_judge_context_relevance(question, ground_truth, c)
                 for c in ranked_contexts]
    total_relevant = sum(relevance)
    if total_relevant == 0:
        return 0.0

    weighted_sum = 0.0
    running_relevant = 0
    for k in range(1, K + 1):
        v_k = relevance[k - 1]
        running_relevant += v_k
        precision_at_k = running_relevant / k
        weighted_sum += precision_at_k * v_k

    return weighted_sum / total_relevant


def context_recall(ground_truth: str, retrieved_contexts: List[str]) -> float:
    """
    Context Recall: نسبة ادعاءات الإجابة المرجعية التي يمكن إيجاد
    سند لها داخل السياق المسترجَع مجتمعًا.
    """
    claims = _split_into_claims(ground_truth)
    if not claims:
        return 1.0
    context_block = "\n\n".join(retrieved_contexts)
    verdicts = [_judge_claim_attributable(c, context_block) for c in claims]
    return sum(verdicts) / len(claims)


def faithfulness(answer: str, retrieved_contexts: List[str]) -> float:
    """
    Faithfulness: نسبة ادعاءات الإجابة المولَّدة المدعومة بالسياق المسترجَع.
    يقيس الاتساق مع المصدر، وليس الصحة الواقعية المطلقة.
    """
    claims = _split_into_claims(answer)
    if not claims:
        return 1.0
    context_block = "\n\n".join(retrieved_contexts)
    verdicts = [_judge_claim_attributable(c, context_block) for c in claims]
    return sum(verdicts) / len(claims)


def _generate_synthetic_questions(answer: str, n: int = 3) -> List[str]:
    system = (
        f"اقترح {n} أسئلة مختلفة يمكن أن تكون هذه الإجابة ردًا مباشرًا ومحددًا عليها. "
        "أعد النتيجة كـ JSON array من النصوص فقط."
    )
    user = f"الإجابة: {answer}"
    raw = call_llm(system, user, temperature=0.3)
    try:
        questions = json.loads(raw)
        if isinstance(questions, list):
            return [str(q).strip() for q in questions][:n]
    except json.JSONDecodeError:
        pass
    return [answer]


def answer_relevancy(question: str, answer: str, n: int = 3) -> float:
    """
    Answer Relevancy: متوسط تشابه جيب التمام بين السؤال الأصلي وأسئلة
    اصطناعية مُستنبَطة عكسيًا من الإجابة.
    """
    synthetic_questions = _generate_synthetic_questions(answer, n=n)
    q_emb = embed_texts([question])[0]
    sq_embs = embed_texts(synthetic_questions)
    sims = [cosine_sim(q_emb, sq) for sq in sq_embs]
    return float(np.mean(sims))


# ============================================================
# الجزء 6: تجميع النتائج والتشخيص الآلي
# ============================================================

@dataclass
class RAGEvalResult:
    question: str
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float

    def ragas_score(self) -> float:
        """الدرجة الإجمالية = المتوسط التوافقي للمقاييس الأربعة (يعاقب أي قيمة منخفضة بشدة)."""
        scores = [self.context_precision, self.context_recall,
                  self.faithfulness, self.answer_relevancy]
        if min(scores) == 0:
            return 0.0
        return len(scores) / sum(1.0 / s for s in scores)

    def diagnosis(self) -> str:
        """تشخيص آلي مبسّط لتحديد مكوّن الفشل الأرجح."""
        retrieval_ok = self.context_precision >= 0.7 and self.context_recall >= 0.7
        generation_ok = self.faithfulness >= 0.7 and self.answer_relevancy >= 0.7

        if retrieval_ok and generation_ok:
            return "✅ سليم: الاسترجاع والتوليد كلاهما يعملان بشكل جيد."
        if not retrieval_ok and generation_ok:
            return "🔴 فشل في Retriever: السياق المسترجَع غير كافٍ أو غير دقيق."
        if retrieval_ok and not generation_ok:
            return "🔴 فشل في Generator: السياق جيد لكن النموذج لا يستخدمه بأمانة أو ينحرف عن السؤال."
        return "🔴 فشل مزدوج: مشاكل في كل من الاسترجاع والتوليد."

    def __repr__(self) -> str:
        return (
            f"RAGEvalResult(\n"
            f"  Context Precision = {self.context_precision:.2f}\n"
            f"  Context Recall    = {self.context_recall:.2f}\n"
            f"  Faithfulness      = {self.faithfulness:.2f}\n"
            f"  Answer Relevancy  = {self.answer_relevancy:.2f}\n"
            f"  RAGAS Score       = {self.ragas_score():.2f}\n"
            f")"
        )


def evaluate_rag_sample(question: str, ground_truth: str,
                         contexts: List[str], answer: str) -> RAGEvalResult:
    return RAGEvalResult(
        question=question,
        context_precision=context_precision_at_k(question, ground_truth, contexts),
        context_recall=context_recall(ground_truth, contexts),
        faithfulness=faithfulness(answer, contexts),
        answer_relevancy=answer_relevancy(question, answer),
    )


# ============================================================
# الجزء 7: التشغيل الكامل — قاعدة معرفة + تقييم حي + سيناريوهات تشخيصية
# ============================================================

if __name__ == "__main__":
    knowledge_base = [
        "نُشرت ورقة \"Attention Is All You Need\" عام 2017 من قبل فريق بحثي في Google، "
        "وقدّمت معمارية Transformer القائمة بالكامل على آلية الانتباه الذاتي دون الحاجة "
        "لشبكات تكرارية.",

        "نموذج BERT الذي طوّرته جوجل نُشر في أواخر عام 2018، ويعتمد على هدف تدريبي "
        "يُسمى Masked Language Modeling، حيث تُخفى نسبة من الكلمات ويُطلب من النموذج "
        "التنبؤ بها باستخدام السياق من الاتجاهين معًا.",

        "نموذج GPT-2 الذي طوّرته OpenAI نُشر عام 2019، ويعتمد على التنبؤ بالكلمة التالية "
        "بشكل أحادي الاتجاه (Autoregressive)، عكس BERT الذي يرى السياق كاملًا.",

        "تقنية RLHF (التعلم المعزز من تغذية راجعة بشرية) استُخدمت بشكل بارز لمحاذاة "
        "النماذج اللغوية مع تفضيلات البشر في نموذج InstructGPT عام 2022.",

        "معمارية Vision Transformer (ViT) طبّقت فكرة تقسيم الصورة إلى رقع صغيرة "
        "(Patches) ومعاملتها كرموز نصية تُغذّى لمحوّل قياسي.",

        "نموذج T5 طوّرته جوجل عام 2019 ووحّد كل مهام معالجة اللغة الطبيعية "
        "في صيغة واحدة: تحويل نص إلى نص (Text-to-Text).",
    ]

    store = SimpleVectorStore()
    store.add_documents(knowledge_base)

    # ---------- التشغيل الحي الكامل ----------
    question = "متى نُشر نموذج BERT وما هو هدفه التدريبي الأساسي؟"
    ground_truth = (
        "نُشر نموذج BERT في أواخر عام 2018 من قبل جوجل، "
        "ويعتمد على هدف Masked Language Modeling للتنبؤ بالكلمات المخفية "
        "باستخدام السياق الثنائي الاتجاه."
    )

    result = rag_pipeline(question, store, k=3)
    print("=== التشغيل الحي الكامل ===")
    print("السياقات المسترجَعة:")
    for c in result["contexts"]:
        print("  -", c[:80], "...")
    print("\nالإجابة المولَّدة:", result["answer"])

    live_scores = evaluate_rag_sample(
        question=question, ground_truth=ground_truth,
        contexts=result["contexts"], answer=result["answer"],
    )
    print("\n", live_scores)
    print(live_scores.diagnosis())

    # ---------- سيناريوهات تشخيصية موجَّهة ----------
    # نثبّت السياق والإجابة يدويًا لنعزل كل نوع فشل بمعزل عن عشوائية الاسترجاع الحي
    scenarios = {
        "استرجاع ضعيف (Retriever Failure)": {
            "contexts": [knowledge_base[2], knowledge_base[4]],  # عن GPT-2 وViT، لا علاقة بـ BERT
            "answer": "لا تحتوي المصادر المتاحة على معلومات كافية للإجابة بدقة على هذا السؤال.",
        },
        "هلوسة رغم استرجاع ممتاز (Generator Failure)": {
            "contexts": [knowledge_base[1]],  # القطعة الصحيحة تمامًا عن BERT
            "answer": "نُشر نموذج BERT عام 2020 من قبل OpenAI، ويعتمد فقط على "
                      "التنبؤ بالكلمة التالية بشكل أحادي الاتجاه.",
        },
    }

    print("\n\n=== سيناريوهات تشخيصية موجَّهة ===")
    for name, s in scenarios.items():
        r = evaluate_rag_sample(question, ground_truth, s["contexts"], s["answer"])
        print(f"\n--- {name} ---")
        print(r)
        print(r.diagnosis())

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-48-chapter-48.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 48: الفصل 48: تقييم أنظمة RAG (RAG Evaluation)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_rag_eval() -> None:
    relevant, retrieved, supported = ({'d1', 'd2'}, {'d1', 'd3'}, {'d1'})
    recall = len(relevant & retrieved) / len(relevant)
    precision = len(relevant & retrieved) / len(retrieved)
    grounding = len(supported) / max(len(retrieved), 1)
    print('context recall:', round(recall, 3))
    print('context precision:', round(precision, 3))
    print('supported retrieved evidence:', round(grounding, 3))

if __name__ == "__main__":
    print("Chapter 48 lab: Evaluate retrieval coverage and answer grounding")
    run_rag_eval()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-48-chapter-48.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
الفصل 48: تقييم أنظمة RAG (RAG Evaluation)
تنفيذ كامل لإطار RAGAS: Context Precision, Context Recall,
Faithfulness, Answer Relevancy — من الصفر، بدون أي مكتبة جاهزة للمقاييس.

المتطلبات:
    pip install torch transformers openai numpy

يتطلب التشغيل الفعلي: OPENAI_API_KEY في متغيرات البيئة.
"""

import os
import re
import json
import numpy as np
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
from dataclasses import dataclass
from typing import List, Dict, Optional


# ============================================================
# الجزء 1: محرك التضمين (Embeddings) من الصفر باستخدام PyTorch
# ============================================================

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
_embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)
_embed_model.eval()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_embed_model.to(DEVICE)


def mean_pooling(model_output, attention_mask: Tensor) -> Tensor:
    """تجميع متوسط الرموز مع استبعاد رموز الحشو عبر قناع الانتباه."""
    token_embeddings = model_output[0]  # [batch, seq_len, hidden]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


@torch.no_grad()
def embed_texts(texts: List[str], batch_size: int = 16) -> np.ndarray:
    """يحوّل قائمة نصوص إلى مصفوفة تضمينات (N, 384) مُطبّعة L2."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = _tokenizer(
            batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
        ).to(DEVICE)
        output = _embed_model(**encoded)
        pooled = mean_pooling(output, encoded["attention_mask"])
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        all_embeddings.append(normalized.cpu().numpy())
    return np.vstack(all_embeddings)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ============================================================
# الجزء 2: مخزن متجهات بسيط (Vector Store) للمسترجِع
# ============================================================

class SimpleVectorStore:
    """مخزن متجهات بسيط في الذاكرة، يكفي للتوضيح التعليمي."""

    def __init__(self):
        self.documents: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_documents(self, docs: List[str]) -> None:
        self.documents.extend(docs)
        self.embeddings = embed_texts(self.documents)

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        q_emb = embed_texts([query])[0]
        sims = self.embeddings @ q_emb  # مُطبّع مسبقًا → الضرب = جيب التمام
        top_k_idx = np.argsort(-sims)[:k]
        return [self.documents[i] for i in top_k_idx]


# ============================================================
# الجزء 3: عميل LLM (للتوليد وللحكم)
# ============================================================

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
JUDGE_MODEL = "gpt-4o-mini"
GENERATOR_MODEL = "gpt-4o-mini"


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.0,
             model: str = JUDGE_MODEL) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def generate_rag_answer(question: str, contexts: List[str]) -> str:
    """مولّد الإجابة: يعتمد فقط على السياق المُعطى له، لا أكثر."""
    context_block = "\n\n".join(f"[مصدر {i+1}]: {c}" for i, c in enumerate(contexts))
    system = (
        "أنت مساعد يجيب فقط بالاعتماد على المصادر المُعطاة أدناه. "
        "إن لم تجد الإجابة الكافية فيها، قل ذلك صراحة ولا تخترع معلومات."
    )
    user = f"المصادر:\n{context_block}\n\nالسؤال: {question}\n\nالإجابة:"
    return call_llm(system, user, temperature=0.0, model=GENERATOR_MODEL)


def rag_pipeline(question: str, store: SimpleVectorStore, k: int = 3) -> Dict:
    contexts = store.retrieve(question, k=k)
    answer = generate_rag_answer(question, contexts)
    return {"question": question, "contexts": contexts, "answer": answer}


# ============================================================
# الجزء 4: أدوات مشتركة — تفكيك الادعاءات والحكم عليها
# ============================================================

def _split_into_claims(text: str) -> List[str]:
    """
    يفكك نصًا إلى ادعاءات ذرية (Atomic Claims) عبر LLM.
    أدق من التقسيم بالنقطة لأن الجملة الواحدة قد تحمل أكثر من ادعاء مستقل.
    """
    system = (
        "فكك النص التالي إلى قائمة من الادعاءات الذرية (Atomic Claims) المستقلة، "
        "كل ادعاء يحمل معلومة واحدة فقط قابلة للتحقق بمفردها. "
        "أعد النتيجة كـ JSON array من النصوص فقط، دون أي شرح إضافي."
    )
    user = f"النص: {text}"
    raw = call_llm(system, user)
    try:
        claims = json.loads(raw)
        if isinstance(claims, list):
            return [str(c).strip() for c in claims if str(c).strip()]
    except json.JSONDecodeError:
        pass
    return [s.strip() for s in re.split(r"[.؟!]", text) if s.strip()]


def _judge_claim_attributable(claim: str, context_block: str) -> int:
    """هل يمكن استنتاج الادعاء من السياق المرجعي المُعطى فقط؟"""
    system = (
        "أنت مُقيّم دقيق. هل يمكن استنتاج الادعاء التالي من النص المرجعي المُعطى فقط؟ "
        "أجب 1 إن كان مدعومًا بالكامل بالنص، أو 0 إن لم يكن مذكورًا أو مدعومًا فيه. "
        "أعد فقط الرقم 0 أو 1 دون أي شرح."
    )
    user = f"النص المرجعي:\n{context_block}\n\nالادعاء: {claim}"
    verdict = call_llm(system, user)
    match = re.search(r"[01]", verdict)
    return int(match.group()) if match else 0


def _judge_context_relevance(question: str, ground_truth: str, context: str) -> int:
    """هل هذه القطعة المحددة ضرورية/مفيدة للإجابة على السؤال؟"""
    system = (
        "أنت مُقيّم دقيق. احكم هل القطعة النصية (context) ضرورية أو مفيدة "
        "للإجابة على السؤال، بالنظر إلى الإجابة المرجعية الصحيحة (ground truth). "
        "أجب 1 إن كانت ذات صلة ومفيدة، أو 0 إن كانت غير ذات صلة. "
        "أعد فقط الرقم 0 أو 1."
    )
    user = f"السؤال: {question}\nالإجابة المرجعية: {ground_truth}\nالقطعة: {context}"
    verdict = call_llm(system, user)
    match = re.search(r"[01]", verdict)
    return int(match.group()) if match else 0


# ============================================================
# الجزء 5: المقاييس الأربعة لـ RAGAS
# ============================================================

def context_precision_at_k(question: str, ground_truth: str,
                            ranked_contexts: List[str]) -> float:
    """
    Context Precision@K: دقة الاسترجاع المرجّحة بالرتبة.
    يعاقب القطع غير ذات الصلة بشدة أكبر كلما ظهرت في رتب مبكرة.
    """
    K = len(ranked_contexts)
    relevance = [_judge_context_relevance(question, ground_truth, c)
                 for c in ranked_contexts]
    total_relevant = sum(relevance)
    if total_relevant == 0:
        return 0.0

    weighted_sum = 0.0
    running_relevant = 0
    for k in range(1, K + 1):
        v_k = relevance[k - 1]
        running_relevant += v_k
        precision_at_k = running_relevant / k
        weighted_sum += precision_at_k * v_k

    return weighted_sum / total_relevant


def context_recall(ground_truth: str, retrieved_contexts: List[str]) -> float:
    """
    Context Recall: نسبة ادعاءات الإجابة المرجعية التي يمكن إيجاد
    سند لها داخل السياق المسترجَع مجتمعًا.
    """
    claims = _split_into_claims(ground_truth)
    if not claims:
        return 1.0
    context_block = "\n\n".join(retrieved_contexts)
    verdicts = [_judge_claim_attributable(c, context_block) for c in claims]
    return sum(verdicts) / len(claims)


def faithfulness(answer: str, retrieved_contexts: List[str]) -> float:
    """
    Faithfulness: نسبة ادعاءات الإجابة المولَّدة المدعومة بالسياق المسترجَع.
    يقيس الاتساق مع المصدر، وليس الصحة الواقعية المطلقة.
    """
    claims = _split_into_claims(answer)
    if not claims:
        return 1.0
    context_block = "\n\n".join(retrieved_contexts)
    verdicts = [_judge_claim_attributable(c, context_block) for c in claims]
    return sum(verdicts) / len(claims)


def _generate_synthetic_questions(answer: str, n: int = 3) -> List[str]:
    system = (
        f"اقترح {n} أسئلة مختلفة يمكن أن تكون هذه الإجابة ردًا مباشرًا ومحددًا عليها. "
        "أعد النتيجة كـ JSON array من النصوص فقط."
    )
    user = f"الإجابة: {answer}"
    raw = call_llm(system, user, temperature=0.3)
    try:
        questions = json.loads(raw)
        if isinstance(questions, list):
            return [str(q).strip() for q in questions][:n]
    except json.JSONDecodeError:
        pass
    return [answer]


def answer_relevancy(question: str, answer: str, n: int = 3) -> float:
    """
    Answer Relevancy: متوسط تشابه جيب التمام بين السؤال الأصلي وأسئلة
    اصطناعية مُستنبَطة عكسيًا من الإجابة.
    """
    synthetic_questions = _generate_synthetic_questions(answer, n=n)
    q_emb = embed_texts([question])[0]
    sq_embs = embed_texts(synthetic_questions)
    sims = [cosine_sim(q_emb, sq) for sq in sq_embs]
    return float(np.mean(sims))


# ============================================================
# الجزء 6: تجميع النتائج والتشخيص الآلي
# ============================================================

@dataclass
class RAGEvalResult:
    question: str
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float

    def ragas_score(self) -> float:
        """الدرجة الإجمالية = المتوسط التوافقي للمقاييس الأربعة (يعاقب أي قيمة منخفضة بشدة)."""
        scores = [self.context_precision, self.context_recall,
                  self.faithfulness, self.answer_relevancy]
        if min(scores) == 0:
            return 0.0
        return len(scores) / sum(1.0 / s for s in scores)

    def diagnosis(self) -> str:
        """تشخيص آلي مبسّط لتحديد مكوّن الفشل الأرجح."""
        retrieval_ok = self.context_precision >= 0.7 and self.context_recall >= 0.7
        generation_ok = self.faithfulness >= 0.7 and self.answer_relevancy >= 0.7

        if retrieval_ok and generation_ok:
            return "✅ سليم: الاسترجاع والتوليد كلاهما يعملان بشكل جيد."
        if not retrieval_ok and generation_ok:
            return "🔴 فشل في Retriever: السياق المسترجَع غير كافٍ أو غير دقيق."
        if retrieval_ok and not generation_ok:
            return "🔴 فشل في Generator: السياق جيد لكن النموذج لا يستخدمه بأمانة أو ينحرف عن السؤال."
        return "🔴 فشل مزدوج: مشاكل في كل من الاسترجاع والتوليد."

    def __repr__(self) -> str:
        return (
            f"RAGEvalResult(\n"
            f"  Context Precision = {self.context_precision:.2f}\n"
            f"  Context Recall    = {self.context_recall:.2f}\n"
            f"  Faithfulness      = {self.faithfulness:.2f}\n"
            f"  Answer Relevancy  = {self.answer_relevancy:.2f}\n"
            f"  RAGAS Score       = {self.ragas_score():.2f}\n"
            f")"
        )


def evaluate_rag_sample(question: str, ground_truth: str,
                         contexts: List[str], answer: str) -> RAGEvalResult:
    return RAGEvalResult(
        question=question,
        context_precision=context_precision_at_k(question, ground_truth, contexts),
        context_recall=context_recall(ground_truth, contexts),
        faithfulness=faithfulness(answer, contexts),
        answer_relevancy=answer_relevancy(question, answer),
    )


# ============================================================
# الجزء 7: التشغيل الكامل — قاعدة معرفة + تقييم حي + سيناريوهات تشخيصية
# ============================================================

if __name__ == "__main__":
    knowledge_base = [
        "نُشرت ورقة \"Attention Is All You Need\" عام 2017 من قبل فريق بحثي في Google، "
        "وقدّمت معمارية Transformer القائمة بالكامل على آلية الانتباه الذاتي دون الحاجة "
        "لشبكات تكرارية.",

        "نموذج BERT الذي طوّرته جوجل نُشر في أواخر عام 2018، ويعتمد على هدف تدريبي "
        "يُسمى Masked Language Modeling، حيث تُخفى نسبة من الكلمات ويُطلب من النموذج "
        "التنبؤ بها باستخدام السياق من الاتجاهين معًا.",

        "نموذج GPT-2 الذي طوّرته OpenAI نُشر عام 2019، ويعتمد على التنبؤ بالكلمة التالية "
        "بشكل أحادي الاتجاه (Autoregressive)، عكس BERT الذي يرى السياق كاملًا.",

        "تقنية RLHF (التعلم المعزز من تغذية راجعة بشرية) استُخدمت بشكل بارز لمحاذاة "
        "النماذج اللغوية مع تفضيلات البشر في نموذج InstructGPT عام 2022.",

        "معمارية Vision Transformer (ViT) طبّقت فكرة تقسيم الصورة إلى رقع صغيرة "
        "(Patches) ومعاملتها كرموز نصية تُغذّى لمحوّل قياسي.",

        "نموذج T5 طوّرته جوجل عام 2019 ووحّد كل مهام معالجة اللغة الطبيعية "
        "في صيغة واحدة: تحويل نص إلى نص (Text-to-Text).",
    ]

    store = SimpleVectorStore()
    store.add_documents(knowledge_base)

    # ---------- التشغيل الحي الكامل ----------
    question = "متى نُشر نموذج BERT وما هو هدفه التدريبي الأساسي؟"
    ground_truth = (
        "نُشر نموذج BERT في أواخر عام 2018 من قبل جوجل، "
        "ويعتمد على هدف Masked Language Modeling للتنبؤ بالكلمات المخفية "
        "باستخدام السياق الثنائي الاتجاه."
    )

    result = rag_pipeline(question, store, k=3)
    print("=== التشغيل الحي الكامل ===")
    print("السياقات المسترجَعة:")
    for c in result["contexts"]:
        print("  -", c[:80], "...")
    print("\nالإجابة المولَّدة:", result["answer"])

    live_scores = evaluate_rag_sample(
        question=question, ground_truth=ground_truth,
        contexts=result["contexts"], answer=result["answer"],
    )
    print("\n", live_scores)
    print(live_scores.diagnosis())

    # ---------- سيناريوهات تشخيصية موجَّهة ----------
    # نثبّت السياق والإجابة يدويًا لنعزل كل نوع فشل بمعزل عن عشوائية الاسترجاع الحي
    scenarios = {
        "استرجاع ضعيف (Retriever Failure)": {
            "contexts": [knowledge_base[2], knowledge_base[4]],  # عن GPT-2 وViT، لا علاقة بـ BERT
            "answer": "لا تحتوي المصادر المتاحة على معلومات كافية للإجابة بدقة على هذا السؤال.",
        },
        "هلوسة رغم استرجاع ممتاز (Generator Failure)": {
            "contexts": [knowledge_base[1]],  # القطعة الصحيحة تمامًا عن BERT
            "answer": "نُشر نموذج BERT عام 2020 من قبل OpenAI، ويعتمد فقط على "
                      "التنبؤ بالكلمة التالية بشكل أحادي الاتجاه.",
        },
    }

    print("\n\n=== سيناريوهات تشخيصية موجَّهة ===")
    for name, s in scenarios.items():
        r = evaluate_rag_sample(question, ground_truth, s["contexts"], s["answer"])
        print(f"\n--- {name} ---")
        print(r)
        print(r.diagnosis())

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-48-chapter-48.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — تقييم أنظمة RAG (RAG Evaluation)
Large Language Model Engineering — RAG

Book slug : llm-engineering-69-chapters
Chapter   : ch-48-chapter-48 (lesson 48)
Code files: 2 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_48.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 48: الفصل 48: تقييم أنظمة RAG (RAG Evaluation)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_rag_eval() -> None:
    relevant, retrieved, supported = ({'d1', 'd2'}, {'d1', 'd3'}, {'d1'})
    recall = len(relevant & retrieved) / len(relevant)
    precision = len(relevant & retrieved) / len(retrieved)
    grounding = len(supported) / max(len(retrieved), 1)
    print('context recall:', round(recall, 3))
    print('context precision:', round(precision, 3))
    print('supported retrieved evidence:', round(grounding, 3))

if __name__ == "__main__":
    print("Chapter 48 lab: Evaluate retrieval coverage and answer grounding")
    run_rag_eval()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-09-evaluation-testing/chapter-48.md
# ────────────────────────────────────────────────────────────────────────
"""
الفصل 48: تقييم أنظمة RAG (RAG Evaluation)
تنفيذ كامل لإطار RAGAS: Context Precision, Context Recall,
Faithfulness, Answer Relevancy — من الصفر، بدون أي مكتبة جاهزة للمقاييس.

المتطلبات:
    pip install torch transformers openai numpy

يتطلب التشغيل الفعلي: OPENAI_API_KEY في متغيرات البيئة.
"""

import os
import re
import json
import numpy as np
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
from dataclasses import dataclass
from typing import List, Dict, Optional


# ============================================================
# الجزء 1: محرك التضمين (Embeddings) من الصفر باستخدام PyTorch
# ============================================================

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
_embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)
_embed_model.eval()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_embed_model.to(DEVICE)


def mean_pooling(model_output, attention_mask: Tensor) -> Tensor:
    """تجميع متوسط الرموز مع استبعاد رموز الحشو عبر قناع الانتباه."""
    token_embeddings = model_output[0]  # [batch, seq_len, hidden]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


@torch.no_grad()
def embed_texts(texts: List[str], batch_size: int = 16) -> np.ndarray:
    """يحوّل قائمة نصوص إلى مصفوفة تضمينات (N, 384) مُطبّعة L2."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = _tokenizer(
            batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
        ).to(DEVICE)
        output = _embed_model(**encoded)
        pooled = mean_pooling(output, encoded["attention_mask"])
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        all_embeddings.append(normalized.cpu().numpy())
    return np.vstack(all_embeddings)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ============================================================
# الجزء 2: مخزن متجهات بسيط (Vector Store) للمسترجِع
# ============================================================

class SimpleVectorStore:
    """مخزن متجهات بسيط في الذاكرة، يكفي للتوضيح التعليمي."""

    def __init__(self):
        self.documents: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_documents(self, docs: List[str]) -> None:
        self.documents.extend(docs)
        self.embeddings = embed_texts(self.documents)

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        q_emb = embed_texts([query])[0]
        sims = self.embeddings @ q_emb  # مُطبّع مسبقًا → الضرب = جيب التمام
        top_k_idx = np.argsort(-sims)[:k]
        return [self.documents[i] for i in top_k_idx]


# ============================================================
# الجزء 3: عميل LLM (للتوليد وللحكم)
# ============================================================

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
JUDGE_MODEL = "gpt-4o-mini"
GENERATOR_MODEL = "gpt-4o-mini"


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.0,
             model: str = JUDGE_MODEL) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def generate_rag_answer(question: str, contexts: List[str]) -> str:
    """مولّد الإجابة: يعتمد فقط على السياق المُعطى له، لا أكثر."""
    context_block = "\n\n".join(f"[مصدر {i+1}]: {c}" for i, c in enumerate(contexts))
    system = (
        "أنت مساعد يجيب فقط بالاعتماد على المصادر المُعطاة أدناه. "
        "إن لم تجد الإجابة الكافية فيها، قل ذلك صراحة ولا تخترع معلومات."
    )
    user = f"المصادر:\n{context_block}\n\nالسؤال: {question}\n\nالإجابة:"
    return call_llm(system, user, temperature=0.0, model=GENERATOR_MODEL)


def rag_pipeline(question: str, store: SimpleVectorStore, k: int = 3) -> Dict:
    contexts = store.retrieve(question, k=k)
    answer = generate_rag_answer(question, contexts)
    return {"question": question, "contexts": contexts, "answer": answer}


# ============================================================
# الجزء 4: أدوات مشتركة — تفكيك الادعاءات والحكم عليها
# ============================================================

def _split_into_claims(text: str) -> List[str]:
    """
    يفكك نصًا إلى ادعاءات ذرية (Atomic Claims) عبر LLM.
    أدق من التقسيم بالنقطة لأن الجملة الواحدة قد تحمل أكثر من ادعاء مستقل.
    """
    system = (
        "فكك النص التالي إلى قائمة من الادعاءات الذرية (Atomic Claims) المستقلة، "
        "كل ادعاء يحمل معلومة واحدة فقط قابلة للتحقق بمفردها. "
        "أعد النتيجة كـ JSON array من النصوص فقط، دون أي شرح إضافي."
    )
    user = f"النص: {text}"
    raw = call_llm(system, user)
    try:
        claims = json.loads(raw)
        if isinstance(claims, list):
            return [str(c).strip() for c in claims if str(c).strip()]
    except json.JSONDecodeError:
        pass
    return [s.strip() for s in re.split(r"[.؟!]", text) if s.strip()]


def _judge_claim_attributable(claim: str, context_block: str) -> int:
    """هل يمكن استنتاج الادعاء من السياق المرجعي المُعطى فقط؟"""
    system = (
        "أنت مُقيّم دقيق. هل يمكن استنتاج الادعاء التالي من النص المرجعي المُعطى فقط؟ "
        "أجب 1 إن كان مدعومًا بالكامل بالنص، أو 0 إن لم يكن مذكورًا أو مدعومًا فيه. "
        "أعد فقط الرقم 0 أو 1 دون أي شرح."
    )
    user = f"النص المرجعي:\n{context_block}\n\nالادعاء: {claim}"
    verdict = call_llm(system, user)
    match = re.search(r"[01]", verdict)
    return int(match.group()) if match else 0


def _judge_context_relevance(question: str, ground_truth: str, context: str) -> int:
    """هل هذه القطعة المحددة ضرورية/مفيدة للإجابة على السؤال؟"""
    system = (
        "أنت مُقيّم دقيق. احكم هل القطعة النصية (context) ضرورية أو مفيدة "
        "للإجابة على السؤال، بالنظر إلى الإجابة المرجعية الصحيحة (ground truth). "
        "أجب 1 إن كانت ذات صلة ومفيدة، أو 0 إن كانت غير ذات صلة. "
        "أعد فقط الرقم 0 أو 1."
    )
    user = f"السؤال: {question}\nالإجابة المرجعية: {ground_truth}\nالقطعة: {context}"
    verdict = call_llm(system, user)
    match = re.search(r"[01]", verdict)
    return int(match.group()) if match else 0


# ============================================================
# الجزء 5: المقاييس الأربعة لـ RAGAS
# ============================================================

def context_precision_at_k(question: str, ground_truth: str,
                            ranked_contexts: List[str]) -> float:
    """
    Context Precision@K: دقة الاسترجاع المرجّحة بالرتبة.
    يعاقب القطع غير ذات الصلة بشدة أكبر كلما ظهرت في رتب مبكرة.
    """
    K = len(ranked_contexts)
    relevance = [_judge_context_relevance(question, ground_truth, c)
                 for c in ranked_contexts]
    total_relevant = sum(relevance)
    if total_relevant == 0:
        return 0.0

    weighted_sum = 0.0
    running_relevant = 0
    for k in range(1, K + 1):
        v_k = relevance[k - 1]
        running_relevant += v_k
        precision_at_k = running_relevant / k
        weighted_sum += precision_at_k * v_k

    return weighted_sum / total_relevant


def context_recall(ground_truth: str, retrieved_contexts: List[str]) -> float:
    """
    Context Recall: نسبة ادعاءات الإجابة المرجعية التي يمكن إيجاد
    سند لها داخل السياق المسترجَع مجتمعًا.
    """
    claims = _split_into_claims(ground_truth)
    if not claims:
        return 1.0
    context_block = "\n\n".join(retrieved_contexts)
    verdicts = [_judge_claim_attributable(c, context_block) for c in claims]
    return sum(verdicts) / len(claims)


def faithfulness(answer: str, retrieved_contexts: List[str]) -> float:
    """
    Faithfulness: نسبة ادعاءات الإجابة المولَّدة المدعومة بالسياق المسترجَع.
    يقيس الاتساق مع المصدر، وليس الصحة الواقعية المطلقة.
    """
    claims = _split_into_claims(answer)
    if not claims:
        return 1.0
    context_block = "\n\n".join(retrieved_contexts)
    verdicts = [_judge_claim_attributable(c, context_block) for c in claims]
    return sum(verdicts) / len(claims)


def _generate_synthetic_questions(answer: str, n: int = 3) -> List[str]:
    system = (
        f"اقترح {n} أسئلة مختلفة يمكن أن تكون هذه الإجابة ردًا مباشرًا ومحددًا عليها. "
        "أعد النتيجة كـ JSON array من النصوص فقط."
    )
    user = f"الإجابة: {answer}"
    raw = call_llm(system, user, temperature=0.3)
    try:
        questions = json.loads(raw)
        if isinstance(questions, list):
            return [str(q).strip() for q in questions][:n]
    except json.JSONDecodeError:
        pass
    return [answer]


def answer_relevancy(question: str, answer: str, n: int = 3) -> float:
    """
    Answer Relevancy: متوسط تشابه جيب التمام بين السؤال الأصلي وأسئلة
    اصطناعية مُستنبَطة عكسيًا من الإجابة.
    """
    synthetic_questions = _generate_synthetic_questions(answer, n=n)
    q_emb = embed_texts([question])[0]
    sq_embs = embed_texts(synthetic_questions)
    sims = [cosine_sim(q_emb, sq) for sq in sq_embs]
    return float(np.mean(sims))


# ============================================================
# الجزء 6: تجميع النتائج والتشخيص الآلي
# ============================================================

@dataclass
class RAGEvalResult:
    question: str
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float

    def ragas_score(self) -> float:
        """الدرجة الإجمالية = المتوسط التوافقي للمقاييس الأربعة (يعاقب أي قيمة منخفضة بشدة)."""
        scores = [self.context_precision, self.context_recall,
                  self.faithfulness, self.answer_relevancy]
        if min(scores) == 0:
            return 0.0
        return len(scores) / sum(1.0 / s for s in scores)

    def diagnosis(self) -> str:
        """تشخيص آلي مبسّط لتحديد مكوّن الفشل الأرجح."""
        retrieval_ok = self.context_precision >= 0.7 and self.context_recall >= 0.7
        generation_ok = self.faithfulness >= 0.7 and self.answer_relevancy >= 0.7

        if retrieval_ok and generation_ok:
            return "✅ سليم: الاسترجاع والتوليد كلاهما يعملان بشكل جيد."
        if not retrieval_ok and generation_ok:
            return "🔴 فشل في Retriever: السياق المسترجَع غير كافٍ أو غير دقيق."
        if retrieval_ok and not generation_ok:
            return "🔴 فشل في Generator: السياق جيد لكن النموذج لا يستخدمه بأمانة أو ينحرف عن السؤال."
        return "🔴 فشل مزدوج: مشاكل في كل من الاسترجاع والتوليد."

    def __repr__(self) -> str:
        return (
            f"RAGEvalResult(\n"
            f"  Context Precision = {self.context_precision:.2f}\n"
            f"  Context Recall    = {self.context_recall:.2f}\n"
            f"  Faithfulness      = {self.faithfulness:.2f}\n"
            f"  Answer Relevancy  = {self.answer_relevancy:.2f}\n"
            f"  RAGAS Score       = {self.ragas_score():.2f}\n"
            f")"
        )


def evaluate_rag_sample(question: str, ground_truth: str,
                         contexts: List[str], answer: str) -> RAGEvalResult:
    return RAGEvalResult(
        question=question,
        context_precision=context_precision_at_k(question, ground_truth, contexts),
        context_recall=context_recall(ground_truth, contexts),
        faithfulness=faithfulness(answer, contexts),
        answer_relevancy=answer_relevancy(question, answer),
    )


# ============================================================
# الجزء 7: التشغيل الكامل — قاعدة معرفة + تقييم حي + سيناريوهات تشخيصية
# ============================================================

if __name__ == "__main__":
    knowledge_base = [
        "نُشرت ورقة \"Attention Is All You Need\" عام 2017 من قبل فريق بحثي في Google، "
        "وقدّمت معمارية Transformer القائمة بالكامل على آلية الانتباه الذاتي دون الحاجة "
        "لشبكات تكرارية.",

        "نموذج BERT الذي طوّرته جوجل نُشر في أواخر عام 2018، ويعتمد على هدف تدريبي "
        "يُسمى Masked Language Modeling، حيث تُخفى نسبة من الكلمات ويُطلب من النموذج "
        "التنبؤ بها باستخدام السياق من الاتجاهين معًا.",

        "نموذج GPT-2 الذي طوّرته OpenAI نُشر عام 2019، ويعتمد على التنبؤ بالكلمة التالية "
        "بشكل أحادي الاتجاه (Autoregressive)، عكس BERT الذي يرى السياق كاملًا.",

        "تقنية RLHF (التعلم المعزز من تغذية راجعة بشرية) استُخدمت بشكل بارز لمحاذاة "
        "النماذج اللغوية مع تفضيلات البشر في نموذج InstructGPT عام 2022.",

        "معمارية Vision Transformer (ViT) طبّقت فكرة تقسيم الصورة إلى رقع صغيرة "
        "(Patches) ومعاملتها كرموز نصية تُغذّى لمحوّل قياسي.",

        "نموذج T5 طوّرته جوجل عام 2019 ووحّد كل مهام معالجة اللغة الطبيعية "
        "في صيغة واحدة: تحويل نص إلى نص (Text-to-Text).",
    ]

    store = SimpleVectorStore()
    store.add_documents(knowledge_base)

    # ---------- التشغيل الحي الكامل ----------
    question = "متى نُشر نموذج BERT وما هو هدفه التدريبي الأساسي؟"
    ground_truth = (
        "نُشر نموذج BERT في أواخر عام 2018 من قبل جوجل، "
        "ويعتمد على هدف Masked Language Modeling للتنبؤ بالكلمات المخفية "
        "باستخدام السياق الثنائي الاتجاه."
    )

    result = rag_pipeline(question, store, k=3)
    print("=== التشغيل الحي الكامل ===")
    print("السياقات المسترجَعة:")
    for c in result["contexts"]:
        print("  -", c[:80], "...")
    print("\nالإجابة المولَّدة:", result["answer"])

    live_scores = evaluate_rag_sample(
        question=question, ground_truth=ground_truth,
        contexts=result["contexts"], answer=result["answer"],
    )
    print("\n", live_scores)
    print(live_scores.diagnosis())

    # ---------- سيناريوهات تشخيصية موجَّهة ----------
    # نثبّت السياق والإجابة يدويًا لنعزل كل نوع فشل بمعزل عن عشوائية الاسترجاع الحي
    scenarios = {
        "استرجاع ضعيف (Retriever Failure)": {
            "contexts": [knowledge_base[2], knowledge_base[4]],  # عن GPT-2 وViT، لا علاقة بـ BERT
            "answer": "لا تحتوي المصادر المتاحة على معلومات كافية للإجابة بدقة على هذا السؤال.",
        },
        "هلوسة رغم استرجاع ممتاز (Generator Failure)": {
            "contexts": [knowledge_base[1]],  # القطعة الصحيحة تمامًا عن BERT
            "answer": "نُشر نموذج BERT عام 2020 من قبل OpenAI، ويعتمد فقط على "
                      "التنبؤ بالكلمة التالية بشكل أحادي الاتجاه.",
        },
    }

    print("\n\n=== سيناريوهات تشخيصية موجَّهة ===")
    for name, s in scenarios.items():
        r = evaluate_rag_sample(question, ground_truth, s["contexts"], s["answer"])
        print(f"\n--- {name} ---")
        print(r)
        print(r.diagnosis())

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-48-chapter-48.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 48: الفصل 48: تقييم أنظمة RAG (RAG Evaluation)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_rag_eval() -> None:
    relevant, retrieved, supported = ({'d1', 'd2'}, {'d1', 'd3'}, {'d1'})
    recall = len(relevant & retrieved) / len(relevant)
    precision = len(relevant & retrieved) / len(retrieved)
    grounding = len(supported) / max(len(retrieved), 1)
    print('context recall:', round(recall, 3))
    print('context precision:', round(precision, 3))
    print('supported retrieved evidence:', round(grounding, 3))

if __name__ == "__main__":
    print("Chapter 48 lab: Evaluate retrieval coverage and answer grounding")
    run_rag_eval()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-48-chapter-48.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
الفصل 48: تقييم أنظمة RAG (RAG Evaluation)
تنفيذ كامل لإطار RAGAS: Context Precision, Context Recall,
Faithfulness, Answer Relevancy — من الصفر، بدون أي مكتبة جاهزة للمقاييس.

المتطلبات:
    pip install torch transformers openai numpy

يتطلب التشغيل الفعلي: OPENAI_API_KEY في متغيرات البيئة.
"""

import os
import re
import json
import numpy as np
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
from dataclasses import dataclass
from typing import List, Dict, Optional


# ============================================================
# الجزء 1: محرك التضمين (Embeddings) من الصفر باستخدام PyTorch
# ============================================================

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
_embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)
_embed_model.eval()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_embed_model.to(DEVICE)


def mean_pooling(model_output, attention_mask: Tensor) -> Tensor:
    """تجميع متوسط الرموز مع استبعاد رموز الحشو عبر قناع الانتباه."""
    token_embeddings = model_output[0]  # [batch, seq_len, hidden]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


@torch.no_grad()
def embed_texts(texts: List[str], batch_size: int = 16) -> np.ndarray:
    """يحوّل قائمة نصوص إلى مصفوفة تضمينات (N, 384) مُطبّعة L2."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = _tokenizer(
            batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
        ).to(DEVICE)
        output = _embed_model(**encoded)
        pooled = mean_pooling(output, encoded["attention_mask"])
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        all_embeddings.append(normalized.cpu().numpy())
    return np.vstack(all_embeddings)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ============================================================
# الجزء 2: مخزن متجهات بسيط (Vector Store) للمسترجِع
# ============================================================

class SimpleVectorStore:
    """مخزن متجهات بسيط في الذاكرة، يكفي للتوضيح التعليمي."""

    def __init__(self):
        self.documents: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_documents(self, docs: List[str]) -> None:
        self.documents.extend(docs)
        self.embeddings = embed_texts(self.documents)

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        q_emb = embed_texts([query])[0]
        sims = self.embeddings @ q_emb  # مُطبّع مسبقًا → الضرب = جيب التمام
        top_k_idx = np.argsort(-sims)[:k]
        return [self.documents[i] for i in top_k_idx]


# ============================================================
# الجزء 3: عميل LLM (للتوليد وللحكم)
# ============================================================

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
JUDGE_MODEL = "gpt-4o-mini"
GENERATOR_MODEL = "gpt-4o-mini"


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.0,
             model: str = JUDGE_MODEL) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def generate_rag_answer(question: str, contexts: List[str]) -> str:
    """مولّد الإجابة: يعتمد فقط على السياق المُعطى له، لا أكثر."""
    context_block = "\n\n".join(f"[مصدر {i+1}]: {c}" for i, c in enumerate(contexts))
    system = (
        "أنت مساعد يجيب فقط بالاعتماد على المصادر المُعطاة أدناه. "
        "إن لم تجد الإجابة الكافية فيها، قل ذلك صراحة ولا تخترع معلومات."
    )
    user = f"المصادر:\n{context_block}\n\nالسؤال: {question}\n\nالإجابة:"
    return call_llm(system, user, temperature=0.0, model=GENERATOR_MODEL)


def rag_pipeline(question: str, store: SimpleVectorStore, k: int = 3) -> Dict:
    contexts = store.retrieve(question, k=k)
    answer = generate_rag_answer(question, contexts)
    return {"question": question, "contexts": contexts, "answer": answer}


# ============================================================
# الجزء 4: أدوات مشتركة — تفكيك الادعاءات والحكم عليها
# ============================================================

def _split_into_claims(text: str) -> List[str]:
    """
    يفكك نصًا إلى ادعاءات ذرية (Atomic Claims) عبر LLM.
    أدق من التقسيم بالنقطة لأن الجملة الواحدة قد تحمل أكثر من ادعاء مستقل.
    """
    system = (
        "فكك النص التالي إلى قائمة من الادعاءات الذرية (Atomic Claims) المستقلة، "
        "كل ادعاء يحمل معلومة واحدة فقط قابلة للتحقق بمفردها. "
        "أعد النتيجة كـ JSON array من النصوص فقط، دون أي شرح إضافي."
    )
    user = f"النص: {text}"
    raw = call_llm(system, user)
    try:
        claims = json.loads(raw)
        if isinstance(claims, list):
            return [str(c).strip() for c in claims if str(c).strip()]
    except json.JSONDecodeError:
        pass
    return [s.strip() for s in re.split(r"[.؟!]", text) if s.strip()]


def _judge_claim_attributable(claim: str, context_block: str) -> int:
    """هل يمكن استنتاج الادعاء من السياق المرجعي المُعطى فقط؟"""
    system = (
        "أنت مُقيّم دقيق. هل يمكن استنتاج الادعاء التالي من النص المرجعي المُعطى فقط؟ "
        "أجب 1 إن كان مدعومًا بالكامل بالنص، أو 0 إن لم يكن مذكورًا أو مدعومًا فيه. "
        "أعد فقط الرقم 0 أو 1 دون أي شرح."
    )
    user = f"النص المرجعي:\n{context_block}\n\nالادعاء: {claim}"
    verdict = call_llm(system, user)
    match = re.search(r"[01]", verdict)
    return int(match.group()) if match else 0


def _judge_context_relevance(question: str, ground_truth: str, context: str) -> int:
    """هل هذه القطعة المحددة ضرورية/مفيدة للإجابة على السؤال؟"""
    system = (
        "أنت مُقيّم دقيق. احكم هل القطعة النصية (context) ضرورية أو مفيدة "
        "للإجابة على السؤال، بالنظر إلى الإجابة المرجعية الصحيحة (ground truth). "
        "أجب 1 إن كانت ذات صلة ومفيدة، أو 0 إن كانت غير ذات صلة. "
        "أعد فقط الرقم 0 أو 1."
    )
    user = f"السؤال: {question}\nالإجابة المرجعية: {ground_truth}\nالقطعة: {context}"
    verdict = call_llm(system, user)
    match = re.search(r"[01]", verdict)
    return int(match.group()) if match else 0


# ============================================================
# الجزء 5: المقاييس الأربعة لـ RAGAS
# ============================================================

def context_precision_at_k(question: str, ground_truth: str,
                            ranked_contexts: List[str]) -> float:
    """
    Context Precision@K: دقة الاسترجاع المرجّحة بالرتبة.
    يعاقب القطع غير ذات الصلة بشدة أكبر كلما ظهرت في رتب مبكرة.
    """
    K = len(ranked_contexts)
    relevance = [_judge_context_relevance(question, ground_truth, c)
                 for c in ranked_contexts]
    total_relevant = sum(relevance)
    if total_relevant == 0:
        return 0.0

    weighted_sum = 0.0
    running_relevant = 0
    for k in range(1, K + 1):
        v_k = relevance[k - 1]
        running_relevant += v_k
        precision_at_k = running_relevant / k
        weighted_sum += precision_at_k * v_k

    return weighted_sum / total_relevant


def context_recall(ground_truth: str, retrieved_contexts: List[str]) -> float:
    """
    Context Recall: نسبة ادعاءات الإجابة المرجعية التي يمكن إيجاد
    سند لها داخل السياق المسترجَع مجتمعًا.
    """
    claims = _split_into_claims(ground_truth)
    if not claims:
        return 1.0
    context_block = "\n\n".join(retrieved_contexts)
    verdicts = [_judge_claim_attributable(c, context_block) for c in claims]
    return sum(verdicts) / len(claims)


def faithfulness(answer: str, retrieved_contexts: List[str]) -> float:
    """
    Faithfulness: نسبة ادعاءات الإجابة المولَّدة المدعومة بالسياق المسترجَع.
    يقيس الاتساق مع المصدر، وليس الصحة الواقعية المطلقة.
    """
    claims = _split_into_claims(answer)
    if not claims:
        return 1.0
    context_block = "\n\n".join(retrieved_contexts)
    verdicts = [_judge_claim_attributable(c, context_block) for c in claims]
    return sum(verdicts) / len(claims)


def _generate_synthetic_questions(answer: str, n: int = 3) -> List[str]:
    system = (
        f"اقترح {n} أسئلة مختلفة يمكن أن تكون هذه الإجابة ردًا مباشرًا ومحددًا عليها. "
        "أعد النتيجة كـ JSON array من النصوص فقط."
    )
    user = f"الإجابة: {answer}"
    raw = call_llm(system, user, temperature=0.3)
    try:
        questions = json.loads(raw)
        if isinstance(questions, list):
            return [str(q).strip() for q in questions][:n]
    except json.JSONDecodeError:
        pass
    return [answer]


def answer_relevancy(question: str, answer: str, n: int = 3) -> float:
    """
    Answer Relevancy: متوسط تشابه جيب التمام بين السؤال الأصلي وأسئلة
    اصطناعية مُستنبَطة عكسيًا من الإجابة.
    """
    synthetic_questions = _generate_synthetic_questions(answer, n=n)
    q_emb = embed_texts([question])[0]
    sq_embs = embed_texts(synthetic_questions)
    sims = [cosine_sim(q_emb, sq) for sq in sq_embs]
    return float(np.mean(sims))


# ============================================================
# الجزء 6: تجميع النتائج والتشخيص الآلي
# ============================================================

@dataclass
class RAGEvalResult:
    question: str
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float

    def ragas_score(self) -> float:
        """الدرجة الإجمالية = المتوسط التوافقي للمقاييس الأربعة (يعاقب أي قيمة منخفضة بشدة)."""
        scores = [self.context_precision, self.context_recall,
                  self.faithfulness, self.answer_relevancy]
        if min(scores) == 0:
            return 0.0
        return len(scores) / sum(1.0 / s for s in scores)

    def diagnosis(self) -> str:
        """تشخيص آلي مبسّط لتحديد مكوّن الفشل الأرجح."""
        retrieval_ok = self.context_precision >= 0.7 and self.context_recall >= 0.7
        generation_ok = self.faithfulness >= 0.7 and self.answer_relevancy >= 0.7

        if retrieval_ok and generation_ok:
            return "✅ سليم: الاسترجاع والتوليد كلاهما يعملان بشكل جيد."
        if not retrieval_ok and generation_ok:
            return "🔴 فشل في Retriever: السياق المسترجَع غير كافٍ أو غير دقيق."
        if retrieval_ok and not generation_ok:
            return "🔴 فشل في Generator: السياق جيد لكن النموذج لا يستخدمه بأمانة أو ينحرف عن السؤال."
        return "🔴 فشل مزدوج: مشاكل في كل من الاسترجاع والتوليد."

    def __repr__(self) -> str:
        return (
            f"RAGEvalResult(\n"
            f"  Context Precision = {self.context_precision:.2f}\n"
            f"  Context Recall    = {self.context_recall:.2f}\n"
            f"  Faithfulness      = {self.faithfulness:.2f}\n"
            f"  Answer Relevancy  = {self.answer_relevancy:.2f}\n"
            f"  RAGAS Score       = {self.ragas_score():.2f}\n"
            f")"
        )


def evaluate_rag_sample(question: str, ground_truth: str,
                         contexts: List[str], answer: str) -> RAGEvalResult:
    return RAGEvalResult(
        question=question,
        context_precision=context_precision_at_k(question, ground_truth, contexts),
        context_recall=context_recall(ground_truth, contexts),
        faithfulness=faithfulness(answer, contexts),
        answer_relevancy=answer_relevancy(question, answer),
    )


# ============================================================
# الجزء 7: التشغيل الكامل — قاعدة معرفة + تقييم حي + سيناريوهات تشخيصية
# ============================================================

if __name__ == "__main__":
    knowledge_base = [
        "نُشرت ورقة \"Attention Is All You Need\" عام 2017 من قبل فريق بحثي في Google، "
        "وقدّمت معمارية Transformer القائمة بالكامل على آلية الانتباه الذاتي دون الحاجة "
        "لشبكات تكرارية.",

        "نموذج BERT الذي طوّرته جوجل نُشر في أواخر عام 2018، ويعتمد على هدف تدريبي "
        "يُسمى Masked Language Modeling، حيث تُخفى نسبة من الكلمات ويُطلب من النموذج "
        "التنبؤ بها باستخدام السياق من الاتجاهين معًا.",

        "نموذج GPT-2 الذي طوّرته OpenAI نُشر عام 2019، ويعتمد على التنبؤ بالكلمة التالية "
        "بشكل أحادي الاتجاه (Autoregressive)، عكس BERT الذي يرى السياق كاملًا.",

        "تقنية RLHF (التعلم المعزز من تغذية راجعة بشرية) استُخدمت بشكل بارز لمحاذاة "
        "النماذج اللغوية مع تفضيلات البشر في نموذج InstructGPT عام 2022.",

        "معمارية Vision Transformer (ViT) طبّقت فكرة تقسيم الصورة إلى رقع صغيرة "
        "(Patches) ومعاملتها كرموز نصية تُغذّى لمحوّل قياسي.",

        "نموذج T5 طوّرته جوجل عام 2019 ووحّد كل مهام معالجة اللغة الطبيعية "
        "في صيغة واحدة: تحويل نص إلى نص (Text-to-Text).",
    ]

    store = SimpleVectorStore()
    store.add_documents(knowledge_base)

    # ---------- التشغيل الحي الكامل ----------
    question = "متى نُشر نموذج BERT وما هو هدفه التدريبي الأساسي؟"
    ground_truth = (
        "نُشر نموذج BERT في أواخر عام 2018 من قبل جوجل، "
        "ويعتمد على هدف Masked Language Modeling للتنبؤ بالكلمات المخفية "
        "باستخدام السياق الثنائي الاتجاه."
    )

    result = rag_pipeline(question, store, k=3)
    print("=== التشغيل الحي الكامل ===")
    print("السياقات المسترجَعة:")
    for c in result["contexts"]:
        print("  -", c[:80], "...")
    print("\nالإجابة المولَّدة:", result["answer"])

    live_scores = evaluate_rag_sample(
        question=question, ground_truth=ground_truth,
        contexts=result["contexts"], answer=result["answer"],
    )
    print("\n", live_scores)
    print(live_scores.diagnosis())

    # ---------- سيناريوهات تشخيصية موجَّهة ----------
    # نثبّت السياق والإجابة يدويًا لنعزل كل نوع فشل بمعزل عن عشوائية الاسترجاع الحي
    scenarios = {
        "استرجاع ضعيف (Retriever Failure)": {
            "contexts": [knowledge_base[2], knowledge_base[4]],  # عن GPT-2 وViT، لا علاقة بـ BERT
            "answer": "لا تحتوي المصادر المتاحة على معلومات كافية للإجابة بدقة على هذا السؤال.",
        },
        "هلوسة رغم استرجاع ممتاز (Generator Failure)": {
            "contexts": [knowledge_base[1]],  # القطعة الصحيحة تمامًا عن BERT
            "answer": "نُشر نموذج BERT عام 2020 من قبل OpenAI، ويعتمد فقط على "
                      "التنبؤ بالكلمة التالية بشكل أحادي الاتجاه.",
        },
    }

    print("\n\n=== سيناريوهات تشخيصية موجَّهة ===")
    for name, s in scenarios.items():
        r = evaluate_rag_sample(question, ground_truth, s["contexts"], s["answer"])
        print(f"\n--- {name} ---")
        print(r)
        print(r.diagnosis())
