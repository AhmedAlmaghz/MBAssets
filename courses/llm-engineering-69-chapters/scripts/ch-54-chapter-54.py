#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — التلخيص والإجابة على الأسئلة
Large Language Model Engineering — Chapter 54 — Applications Agents

Book slug : llm-engineering-69-chapters
Chapter   : ch-54-chapter-54 (lesson 54)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-54-chapter-54.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — التلخيص والإجابة على الأسئلة
Large Language Model Engineering — Chapter 54 — Applications Agents

Book slug : llm-engineering-69-chapters
Chapter   : ch-54-chapter-54 (lesson 54)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-54-chapter-54.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — التلخيص والإجابة على الأسئلة
Large Language Model Engineering — Chapter 54 — Applications Agents

Book slug : llm-engineering-69-chapters
Chapter   : ch-54-chapter-54 (lesson 54)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_54.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 54: الفصل 54: التلخيص والإجابة على الأسئلة."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_rouge() -> None:
    reference = 'attention combines token values using learned weights'.split()
    summary = 'attention combines values using weights'.split()
    previous = list(range(len(reference) + 1))
    for token in summary:
        current = [previous[0] + 1]
        for index, target in enumerate(reference, 1):
            current.append(previous[index - 1] + 1 if token == target else max(previous[index], current[-1]))
        previous = current
    lcs = previous[-1]
    precision, recall = (lcs / len(summary), lcs / len(reference))
    f1 = 2 * precision * recall / (precision + recall)
    print('LCS length:', lcs)
    print('ROUGE-L precision/recall/F1:', round(precision, 3), round(recall, 3), round(f1, 3))

if __name__ == "__main__":
    print("Chapter 54 lab: Score a summary with ROUGE-L")
    run_rouge()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-10-applications-agents/chapter-54.md
# ────────────────────────────────────────────────────────────────────────
# requirements.txt
# torch>=2.4.0
# transformers>=4.44.0
# rouge-score>=0.1.2
# bert-score>=0.3.13
# datasets>=2.20.0

import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering, pipeline,
    Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
)
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-10-applications-agents/chapter-54.md
# ────────────────────────────────────────────────────────────────────────
import networkx as nx
from sentence_transformers import SentenceTransformer
import re

class ExtractiveSummarizer:
    """
    خوارزمية TextRank: تبني رسمًا بيانيًا حيث كل جملة = عقدة،
    والوزن بين عقدتين = تشابههما الدلالي، ثم تُطبّق PageRank
    لإيجاد الجمل "الأكثر مركزية" (الأهم) في النص.
    """
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        self.embedder = SentenceTransformer(model_name)

    def summarize(self, text: str, num_sentences: int = 3) -> str:
        sentences = re.split(r'(?<=[.!?؟])\s+', text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if len(sentences) <= num_sentences:
            return text

        # 1. تضمين كل جملة
        embeddings = self.embedder.encode(sentences, convert_to_numpy=True)

        # 2. بناء مصفوفة التشابه (Similarity Matrix)
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8
                    )
                    similarity_matrix[i][j] = sim

        # 3. بناء رسم بياني وتطبيق PageRank
        graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(graph)

        # 4. اختيار أعلى الجمل مرتبة ثم إعادة ترتيبها بترتيبها الأصلي في النص
        ranked_indices = sorted(scores, key=scores.get, reverse=True)[:num_sentences]
        ranked_indices.sort()  # الحفاظ على التسلسل المنطقي الأصلي

        return " ".join([sentences[i] for i in ranked_indices])


# --- الاستخدام ---
extractive_summarizer = ExtractiveSummarizer()

long_text = """
يشهد قطاع الذكاء الاصطناعي نموًا متسارعًا غير مسبوق خلال السنوات الأخيرة.
الشركات الكبرى تستثمر مليارات الدولارات في تطوير نماذج لغوية ضخمة.
هذا التطور يطرح تحديات أخلاقية وتنظيمية جديدة لم تكن موجودة من قبل.
في المقابل، يرى الخبراء أن هذه التقنية ستُحدث تحولًا جذريًا في سوق العمل.
العديد من الوظائف التقليدية مهددة بالاستبدال، بينما تظهر وظائف جديدة كليًا.
الحكومات حول العالم تحاول وضع أطر تنظيمية لمواكبة هذا التسارع.
"""

summary = extractive_summarizer.summarize(long_text, num_sentences=2)
print(f"📝 الملخص الاستخراجي: {summary}")

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-10-applications-agents/chapter-54.md
# ────────────────────────────────────────────────────────────────────────
# --- الخطوة 1: تجهيز بيانات التلخيص ---
from datasets import Dataset

summarization_data = {
    "document": [
        "ارتفعت أسعار النفط العالمية بنسبة 5% اليوم بعد إعلان أوبك خفض الإنتاج. "
        "المحللون يتوقعون استمرار هذا الاتجاه الصعودي خلال الأسابيع القادمة بسبب "
        "التوترات الجيوسياسية المستمرة في منطقة الشرق الأوسط وتراجع المخزونات العالمية.",

        "أعلنت الشركة عن نتائج مالية قوية للربع الثالث، محققة أرباحًا تجاوزت توقعات "
        "المحللين بنسبة 15%. يُعزى هذا النمو إلى زيادة الطلب على منتجاتها الرقمية "
        "الجديدة وتوسعها في الأسواق الآسيوية الناشئة.",
    ],
    "summary": [
        "ارتفاع أسعار النفط 5% بعد قرار أوبك خفض الإنتاج وسط توترات جيوسياسية.",
        "الشركة تحقق أرباحًا تفوق التوقعات بـ15% مدفوعة بالمنتجات الرقمية والتوسع الآسيوي.",
    ]
}

dataset = Dataset.from_dict(summarization_data)
dataset = dataset.train_test_split(test_size=0.5, seed=42)

# --- الخطوة 2: استخدام نموذج mT5 (متعدد اللغات، يدعم العربية) ---
MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"  # مُدرَّب مسبقًا على التلخيص متعدد اللغات!

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 100

def preprocess_function(examples):
    inputs = tokenizer(
        examples["document"], max_length=MAX_INPUT_LENGTH, truncation=True, padding=False
    )
    labels = tokenizer(
        text_target=examples["summary"], max_length=MAX_TARGET_LENGTH, truncation=True, padding=False
    )
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

# --- الخطوة 3: دالة المقاييس باستخدام ROUGE ---
rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    for pred, label in zip(decoded_preds, decoded_labels):
        scores = rouge.score(label, pred)  # ترتيب: (reference, hypothesis)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": np.mean(rouge1_scores),
        "rouge2": np.mean(rouge2_scores),
        "rougeL": np.mean(rougeL_scores),
    }

# --- الخطوة 4: التدريب ---
training_args = Seq2SeqTrainingArguments(
    output_dir="./mt5_arabic_summarizer",
    learning_rate=3e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LENGTH,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    logging_steps=1,
    report_to="none",
    fp16=torch.cuda.is_available(),
)

trainer = Seq2SeqTrainer(
    model=model, args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# --- الخطوة 5: الاستدلال على نص جديد ---
def generate_summary(text: str, max_length: int = 100) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=MAX_INPUT_LENGTH).to(device)
    with torch.no_grad():
        summary_ids = model.generate(
            **inputs, max_length=max_length, num_beams=4,
            length_penalty=2.0,  # يُشجّع ملخصات أطول قليلًا (قيمة > 1)
            early_stopping=True
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-10-applications-agents/chapter-54.md
# ────────────────────────────────────────────────────────────────────────
# --- استخدام نموذج AraBERT مُدرَّب مسبقًا على SQuAD-style QA ---

QA_MODEL_NAME = "wissamantoun/araelectra-base-artydiqa"  # نموذج QA عربي جاهز

qa_pipeline = pipeline(
    "question-answering",
    model=QA_MODEL_NAME,
    tokenizer=QA_MODEL_NAME,
    device=0 if torch.cuda.is_available() else -1
)

context = """
تأسست المملكة العربية السعودية الحديثة في عام 1932 على يد الملك عبدالعزيز آل سعود،
بعد توحيد عدة مناطق ومدن تحت راية واحدة. تُعد الرياض عاصمة المملكة ومركزها السياسي
والإداري، بينما تُعتبر جدة البوابة الاقتصادية الرئيسية للبلاد على ساحل البحر الأحمر.
"""

question = "من أسس المملكة العربية السعودية الحديثة؟"

result = qa_pipeline(question=question, context=context)
print(f"❓ السؤال: {question}")
print(f"💬 الإجابة: {result['answer']}")
print(f"📊 درجة الثقة: {result['score']:.3f}")
print(f"📍 الموقع في النص: [{result['start']}:{result['end']}]")

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-10-applications-agents/chapter-54.md
# ────────────────────────────────────────────────────────────────────────
import torch.nn as nn
from transformers import AutoModel

class CustomQAModel(nn.Module):
    """
    بناء يدوي لرأس QA فوق BERT، لفهم الآلية الداخلية بعمق
    (ما يحدث فعليًا خلف `AutoModelForQuestionAnswering`)
    """
    def __init__(self, backbone_name: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.bert.config.hidden_size
        # طبقة خطية واحدة تُنتج قيمتين لكل Token: (start_logit, end_logit)
        self.qa_outputs = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden]

        logits = self.qa_outputs(sequence_output)  # [batch, seq_len, 2]
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # [batch, seq_len]
        end_logits = end_logits.squeeze(-1)

        loss = None
        if start_positions is not None and end_positions is not None:
            loss_fn = nn.CrossEntropyLoss()
            start_loss = loss_fn(start_logits, start_positions)
            end_loss = loss_fn(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2

        return {"loss": loss, "start_logits": start_logits, "end_logits": end_logits}

    def predict_answer(self, input_ids, attention_mask, tokenizer) -> str:
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)

        start_idx = torch.argmax(outputs["start_logits"], dim=-1).item()
        end_idx = torch.argmax(outputs["end_logits"], dim=-1).item()

        if end_idx < start_idx:  # حماية من تنبؤات غير منطقية
            end_idx = start_idx

        answer_tokens = input_ids[0][start_idx:end_idx+1]
        return tokenizer.decode(answer_tokens, skip_special_tokens=True)

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-10-applications-agents/chapter-54.md
# ────────────────────────────────────────────────────────────────────────
def comprehensive_summary_evaluation(
    generated_summaries: list[str], reference_summaries: list[str]
) -> dict:
    """يقارن ROUGE التقليدي مقابل BERTScore الدلالي الحديث"""

    # --- ROUGE ---
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    rouge1_f, rouge2_f, rougeL_f = [], [], []

    for gen, ref in zip(generated_summaries, reference_summaries):
        scores = rouge.score(ref, gen)
        rouge1_f.append(scores["rouge1"].fmeasure)
        rouge2_f.append(scores["rouge2"].fmeasure)
        rougeL_f.append(scores["rougeL"].fmeasure)

    # --- BERTScore ---
    P, R, F1 = bert_score(
        generated_summaries, reference_summaries,
        lang="ar", model_type="bert-base-multilingual-cased"
    )

    return {
        "ROUGE-1": round(np.mean(rouge1_f), 3),
        "ROUGE-2": round(np.mean(rouge2_f), 3),
        "ROUGE-L": round(np.mean(rougeL_f), 3),
        "BERTScore-F1": round(F1.mean().item(), 3),
    }


# --- مثال يكشف الفرق بين المقياسين ---
generated = ["شهدت أسعار النفط ارتفاعًا ملحوظًا بسبب قرارات أوبك."]
reference = ["ارتفعت أسعار النفط العالمية نتيجة خفض أوبك للإنتاج."]

report = comprehensive_summary_evaluation(generated, reference)
print("📊 تقرير التقييم المقارن:")
for metric, value in report.items():
    print(f"  {metric}: {value}")

# النتيجة النمطية: ROUGE منخفض نسبيًا (كلمات مختلفة)،
# لكن BERTScore مرتفع جدًا (المعنى متطابق تقريبًا)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-54-chapter-54.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 54: الفصل 54: التلخيص والإجابة على الأسئلة."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_rouge() -> None:
    reference = 'attention combines token values using learned weights'.split()
    summary = 'attention combines values using weights'.split()
    previous = list(range(len(reference) + 1))
    for token in summary:
        current = [previous[0] + 1]
        for index, target in enumerate(reference, 1):
            current.append(previous[index - 1] + 1 if token == target else max(previous[index], current[-1]))
        previous = current
    lcs = previous[-1]
    precision, recall = (lcs / len(summary), lcs / len(reference))
    f1 = 2 * precision * recall / (precision + recall)
    print('LCS length:', lcs)
    print('ROUGE-L precision/recall/F1:', round(precision, 3), round(recall, 3), round(f1, 3))

if __name__ == "__main__":
    print("Chapter 54 lab: Score a summary with ROUGE-L")
    run_rouge()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-54-chapter-54.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# requirements.txt
# torch>=2.4.0
# transformers>=4.44.0
# rouge-score>=0.1.2
# bert-score>=0.3.13
# datasets>=2.20.0

import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering, pipeline,
    Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
)
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-54-chapter-54.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import networkx as nx
from sentence_transformers import SentenceTransformer
import re

class ExtractiveSummarizer:
    """
    خوارزمية TextRank: تبني رسمًا بيانيًا حيث كل جملة = عقدة،
    والوزن بين عقدتين = تشابههما الدلالي، ثم تُطبّق PageRank
    لإيجاد الجمل "الأكثر مركزية" (الأهم) في النص.
    """
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        self.embedder = SentenceTransformer(model_name)

    def summarize(self, text: str, num_sentences: int = 3) -> str:
        sentences = re.split(r'(?<=[.!?؟])\s+', text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if len(sentences) <= num_sentences:
            return text

        # 1. تضمين كل جملة
        embeddings = self.embedder.encode(sentences, convert_to_numpy=True)

        # 2. بناء مصفوفة التشابه (Similarity Matrix)
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8
                    )
                    similarity_matrix[i][j] = sim

        # 3. بناء رسم بياني وتطبيق PageRank
        graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(graph)

        # 4. اختيار أعلى الجمل مرتبة ثم إعادة ترتيبها بترتيبها الأصلي في النص
        ranked_indices = sorted(scores, key=scores.get, reverse=True)[:num_sentences]
        ranked_indices.sort()  # الحفاظ على التسلسل المنطقي الأصلي

        return " ".join([sentences[i] for i in ranked_indices])


# --- الاستخدام ---
extractive_summarizer = ExtractiveSummarizer()

long_text = """
يشهد قطاع الذكاء الاصطناعي نموًا متسارعًا غير مسبوق خلال السنوات الأخيرة.
الشركات الكبرى تستثمر مليارات الدولارات في تطوير نماذج لغوية ضخمة.
هذا التطور يطرح تحديات أخلاقية وتنظيمية جديدة لم تكن موجودة من قبل.
في المقابل، يرى الخبراء أن هذه التقنية ستُحدث تحولًا جذريًا في سوق العمل.
العديد من الوظائف التقليدية مهددة بالاستبدال، بينما تظهر وظائف جديدة كليًا.
الحكومات حول العالم تحاول وضع أطر تنظيمية لمواكبة هذا التسارع.
"""

summary = extractive_summarizer.summarize(long_text, num_sentences=2)
print(f"📝 الملخص الاستخراجي: {summary}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-54-chapter-54.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# --- الخطوة 1: تجهيز بيانات التلخيص ---
from datasets import Dataset

summarization_data = {
    "document": [
        "ارتفعت أسعار النفط العالمية بنسبة 5% اليوم بعد إعلان أوبك خفض الإنتاج. "
        "المحللون يتوقعون استمرار هذا الاتجاه الصعودي خلال الأسابيع القادمة بسبب "
        "التوترات الجيوسياسية المستمرة في منطقة الشرق الأوسط وتراجع المخزونات العالمية.",

        "أعلنت الشركة عن نتائج مالية قوية للربع الثالث، محققة أرباحًا تجاوزت توقعات "
        "المحللين بنسبة 15%. يُعزى هذا النمو إلى زيادة الطلب على منتجاتها الرقمية "
        "الجديدة وتوسعها في الأسواق الآسيوية الناشئة.",
    ],
    "summary": [
        "ارتفاع أسعار النفط 5% بعد قرار أوبك خفض الإنتاج وسط توترات جيوسياسية.",
        "الشركة تحقق أرباحًا تفوق التوقعات بـ15% مدفوعة بالمنتجات الرقمية والتوسع الآسيوي.",
    ]
}

dataset = Dataset.from_dict(summarization_data)
dataset = dataset.train_test_split(test_size=0.5, seed=42)

# --- الخطوة 2: استخدام نموذج mT5 (متعدد اللغات، يدعم العربية) ---
MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"  # مُدرَّب مسبقًا على التلخيص متعدد اللغات!

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 100

def preprocess_function(examples):
    inputs = tokenizer(
        examples["document"], max_length=MAX_INPUT_LENGTH, truncation=True, padding=False
    )
    labels = tokenizer(
        text_target=examples["summary"], max_length=MAX_TARGET_LENGTH, truncation=True, padding=False
    )
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

# --- الخطوة 3: دالة المقاييس باستخدام ROUGE ---
rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    for pred, label in zip(decoded_preds, decoded_labels):
        scores = rouge.score(label, pred)  # ترتيب: (reference, hypothesis)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": np.mean(rouge1_scores),
        "rouge2": np.mean(rouge2_scores),
        "rougeL": np.mean(rougeL_scores),
    }

# --- الخطوة 4: التدريب ---
training_args = Seq2SeqTrainingArguments(
    output_dir="./mt5_arabic_summarizer",
    learning_rate=3e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LENGTH,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    logging_steps=1,
    report_to="none",
    fp16=torch.cuda.is_available(),
)

trainer = Seq2SeqTrainer(
    model=model, args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# --- الخطوة 5: الاستدلال على نص جديد ---
def generate_summary(text: str, max_length: int = 100) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=MAX_INPUT_LENGTH).to(device)
    with torch.no_grad():
        summary_ids = model.generate(
            **inputs, max_length=max_length, num_beams=4,
            length_penalty=2.0,  # يُشجّع ملخصات أطول قليلًا (قيمة > 1)
            early_stopping=True
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-54-chapter-54.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# --- استخدام نموذج AraBERT مُدرَّب مسبقًا على SQuAD-style QA ---

QA_MODEL_NAME = "wissamantoun/araelectra-base-artydiqa"  # نموذج QA عربي جاهز

qa_pipeline = pipeline(
    "question-answering",
    model=QA_MODEL_NAME,
    tokenizer=QA_MODEL_NAME,
    device=0 if torch.cuda.is_available() else -1
)

context = """
تأسست المملكة العربية السعودية الحديثة في عام 1932 على يد الملك عبدالعزيز آل سعود،
بعد توحيد عدة مناطق ومدن تحت راية واحدة. تُعد الرياض عاصمة المملكة ومركزها السياسي
والإداري، بينما تُعتبر جدة البوابة الاقتصادية الرئيسية للبلاد على ساحل البحر الأحمر.
"""

question = "من أسس المملكة العربية السعودية الحديثة؟"

result = qa_pipeline(question=question, context=context)
print(f"❓ السؤال: {question}")
print(f"💬 الإجابة: {result['answer']}")
print(f"📊 درجة الثقة: {result['score']:.3f}")
print(f"📍 الموقع في النص: [{result['start']}:{result['end']}]")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-54-chapter-54.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import torch.nn as nn
from transformers import AutoModel

class CustomQAModel(nn.Module):
    """
    بناء يدوي لرأس QA فوق BERT، لفهم الآلية الداخلية بعمق
    (ما يحدث فعليًا خلف `AutoModelForQuestionAnswering`)
    """
    def __init__(self, backbone_name: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.bert.config.hidden_size
        # طبقة خطية واحدة تُنتج قيمتين لكل Token: (start_logit, end_logit)
        self.qa_outputs = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden]

        logits = self.qa_outputs(sequence_output)  # [batch, seq_len, 2]
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # [batch, seq_len]
        end_logits = end_logits.squeeze(-1)

        loss = None
        if start_positions is not None and end_positions is not None:
            loss_fn = nn.CrossEntropyLoss()
            start_loss = loss_fn(start_logits, start_positions)
            end_loss = loss_fn(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2

        return {"loss": loss, "start_logits": start_logits, "end_logits": end_logits}

    def predict_answer(self, input_ids, attention_mask, tokenizer) -> str:
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)

        start_idx = torch.argmax(outputs["start_logits"], dim=-1).item()
        end_idx = torch.argmax(outputs["end_logits"], dim=-1).item()

        if end_idx < start_idx:  # حماية من تنبؤات غير منطقية
            end_idx = start_idx

        answer_tokens = input_ids[0][start_idx:end_idx+1]
        return tokenizer.decode(answer_tokens, skip_special_tokens=True)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-54-chapter-54.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def comprehensive_summary_evaluation(
    generated_summaries: list[str], reference_summaries: list[str]
) -> dict:
    """يقارن ROUGE التقليدي مقابل BERTScore الدلالي الحديث"""

    # --- ROUGE ---
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    rouge1_f, rouge2_f, rougeL_f = [], [], []

    for gen, ref in zip(generated_summaries, reference_summaries):
        scores = rouge.score(ref, gen)
        rouge1_f.append(scores["rouge1"].fmeasure)
        rouge2_f.append(scores["rouge2"].fmeasure)
        rougeL_f.append(scores["rougeL"].fmeasure)

    # --- BERTScore ---
    P, R, F1 = bert_score(
        generated_summaries, reference_summaries,
        lang="ar", model_type="bert-base-multilingual-cased"
    )

    return {
        "ROUGE-1": round(np.mean(rouge1_f), 3),
        "ROUGE-2": round(np.mean(rouge2_f), 3),
        "ROUGE-L": round(np.mean(rougeL_f), 3),
        "BERTScore-F1": round(F1.mean().item(), 3),
    }


# --- مثال يكشف الفرق بين المقياسين ---
generated = ["شهدت أسعار النفط ارتفاعًا ملحوظًا بسبب قرارات أوبك."]
reference = ["ارتفعت أسعار النفط العالمية نتيجة خفض أوبك للإنتاج."]

report = comprehensive_summary_evaluation(generated, reference)
print("📊 تقرير التقييم المقارن:")
for metric, value in report.items():
    print(f"  {metric}: {value}")

# النتيجة النمطية: ROUGE منخفض نسبيًا (كلمات مختلفة)،
# لكن BERTScore مرتفع جدًا (المعنى متطابق تقريبًا)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-54-chapter-54.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — التلخيص والإجابة على الأسئلة
Large Language Model Engineering — Chapter 54 — Applications Agents

Book slug : llm-engineering-69-chapters
Chapter   : ch-54-chapter-54 (lesson 54)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_54.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 54: الفصل 54: التلخيص والإجابة على الأسئلة."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_rouge() -> None:
    reference = 'attention combines token values using learned weights'.split()
    summary = 'attention combines values using weights'.split()
    previous = list(range(len(reference) + 1))
    for token in summary:
        current = [previous[0] + 1]
        for index, target in enumerate(reference, 1):
            current.append(previous[index - 1] + 1 if token == target else max(previous[index], current[-1]))
        previous = current
    lcs = previous[-1]
    precision, recall = (lcs / len(summary), lcs / len(reference))
    f1 = 2 * precision * recall / (precision + recall)
    print('LCS length:', lcs)
    print('ROUGE-L precision/recall/F1:', round(precision, 3), round(recall, 3), round(f1, 3))

if __name__ == "__main__":
    print("Chapter 54 lab: Score a summary with ROUGE-L")
    run_rouge()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-10-applications-agents/chapter-54.md
# ────────────────────────────────────────────────────────────────────────
# requirements.txt
# torch>=2.4.0
# transformers>=4.44.0
# rouge-score>=0.1.2
# bert-score>=0.3.13
# datasets>=2.20.0

import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering, pipeline,
    Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
)
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-10-applications-agents/chapter-54.md
# ────────────────────────────────────────────────────────────────────────
import networkx as nx
from sentence_transformers import SentenceTransformer
import re

class ExtractiveSummarizer:
    """
    خوارزمية TextRank: تبني رسمًا بيانيًا حيث كل جملة = عقدة،
    والوزن بين عقدتين = تشابههما الدلالي، ثم تُطبّق PageRank
    لإيجاد الجمل "الأكثر مركزية" (الأهم) في النص.
    """
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        self.embedder = SentenceTransformer(model_name)

    def summarize(self, text: str, num_sentences: int = 3) -> str:
        sentences = re.split(r'(?<=[.!?؟])\s+', text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if len(sentences) <= num_sentences:
            return text

        # 1. تضمين كل جملة
        embeddings = self.embedder.encode(sentences, convert_to_numpy=True)

        # 2. بناء مصفوفة التشابه (Similarity Matrix)
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8
                    )
                    similarity_matrix[i][j] = sim

        # 3. بناء رسم بياني وتطبيق PageRank
        graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(graph)

        # 4. اختيار أعلى الجمل مرتبة ثم إعادة ترتيبها بترتيبها الأصلي في النص
        ranked_indices = sorted(scores, key=scores.get, reverse=True)[:num_sentences]
        ranked_indices.sort()  # الحفاظ على التسلسل المنطقي الأصلي

        return " ".join([sentences[i] for i in ranked_indices])


# --- الاستخدام ---
extractive_summarizer = ExtractiveSummarizer()

long_text = """
يشهد قطاع الذكاء الاصطناعي نموًا متسارعًا غير مسبوق خلال السنوات الأخيرة.
الشركات الكبرى تستثمر مليارات الدولارات في تطوير نماذج لغوية ضخمة.
هذا التطور يطرح تحديات أخلاقية وتنظيمية جديدة لم تكن موجودة من قبل.
في المقابل، يرى الخبراء أن هذه التقنية ستُحدث تحولًا جذريًا في سوق العمل.
العديد من الوظائف التقليدية مهددة بالاستبدال، بينما تظهر وظائف جديدة كليًا.
الحكومات حول العالم تحاول وضع أطر تنظيمية لمواكبة هذا التسارع.
"""

summary = extractive_summarizer.summarize(long_text, num_sentences=2)
print(f"📝 الملخص الاستخراجي: {summary}")

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-10-applications-agents/chapter-54.md
# ────────────────────────────────────────────────────────────────────────
# --- الخطوة 1: تجهيز بيانات التلخيص ---
from datasets import Dataset

summarization_data = {
    "document": [
        "ارتفعت أسعار النفط العالمية بنسبة 5% اليوم بعد إعلان أوبك خفض الإنتاج. "
        "المحللون يتوقعون استمرار هذا الاتجاه الصعودي خلال الأسابيع القادمة بسبب "
        "التوترات الجيوسياسية المستمرة في منطقة الشرق الأوسط وتراجع المخزونات العالمية.",

        "أعلنت الشركة عن نتائج مالية قوية للربع الثالث، محققة أرباحًا تجاوزت توقعات "
        "المحللين بنسبة 15%. يُعزى هذا النمو إلى زيادة الطلب على منتجاتها الرقمية "
        "الجديدة وتوسعها في الأسواق الآسيوية الناشئة.",
    ],
    "summary": [
        "ارتفاع أسعار النفط 5% بعد قرار أوبك خفض الإنتاج وسط توترات جيوسياسية.",
        "الشركة تحقق أرباحًا تفوق التوقعات بـ15% مدفوعة بالمنتجات الرقمية والتوسع الآسيوي.",
    ]
}

dataset = Dataset.from_dict(summarization_data)
dataset = dataset.train_test_split(test_size=0.5, seed=42)

# --- الخطوة 2: استخدام نموذج mT5 (متعدد اللغات، يدعم العربية) ---
MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"  # مُدرَّب مسبقًا على التلخيص متعدد اللغات!

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 100

def preprocess_function(examples):
    inputs = tokenizer(
        examples["document"], max_length=MAX_INPUT_LENGTH, truncation=True, padding=False
    )
    labels = tokenizer(
        text_target=examples["summary"], max_length=MAX_TARGET_LENGTH, truncation=True, padding=False
    )
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

# --- الخطوة 3: دالة المقاييس باستخدام ROUGE ---
rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    for pred, label in zip(decoded_preds, decoded_labels):
        scores = rouge.score(label, pred)  # ترتيب: (reference, hypothesis)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": np.mean(rouge1_scores),
        "rouge2": np.mean(rouge2_scores),
        "rougeL": np.mean(rougeL_scores),
    }

# --- الخطوة 4: التدريب ---
training_args = Seq2SeqTrainingArguments(
    output_dir="./mt5_arabic_summarizer",
    learning_rate=3e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LENGTH,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    logging_steps=1,
    report_to="none",
    fp16=torch.cuda.is_available(),
)

trainer = Seq2SeqTrainer(
    model=model, args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# --- الخطوة 5: الاستدلال على نص جديد ---
def generate_summary(text: str, max_length: int = 100) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=MAX_INPUT_LENGTH).to(device)
    with torch.no_grad():
        summary_ids = model.generate(
            **inputs, max_length=max_length, num_beams=4,
            length_penalty=2.0,  # يُشجّع ملخصات أطول قليلًا (قيمة > 1)
            early_stopping=True
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-10-applications-agents/chapter-54.md
# ────────────────────────────────────────────────────────────────────────
# --- استخدام نموذج AraBERT مُدرَّب مسبقًا على SQuAD-style QA ---

QA_MODEL_NAME = "wissamantoun/araelectra-base-artydiqa"  # نموذج QA عربي جاهز

qa_pipeline = pipeline(
    "question-answering",
    model=QA_MODEL_NAME,
    tokenizer=QA_MODEL_NAME,
    device=0 if torch.cuda.is_available() else -1
)

context = """
تأسست المملكة العربية السعودية الحديثة في عام 1932 على يد الملك عبدالعزيز آل سعود،
بعد توحيد عدة مناطق ومدن تحت راية واحدة. تُعد الرياض عاصمة المملكة ومركزها السياسي
والإداري، بينما تُعتبر جدة البوابة الاقتصادية الرئيسية للبلاد على ساحل البحر الأحمر.
"""

question = "من أسس المملكة العربية السعودية الحديثة؟"

result = qa_pipeline(question=question, context=context)
print(f"❓ السؤال: {question}")
print(f"💬 الإجابة: {result['answer']}")
print(f"📊 درجة الثقة: {result['score']:.3f}")
print(f"📍 الموقع في النص: [{result['start']}:{result['end']}]")

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-10-applications-agents/chapter-54.md
# ────────────────────────────────────────────────────────────────────────
import torch.nn as nn
from transformers import AutoModel

class CustomQAModel(nn.Module):
    """
    بناء يدوي لرأس QA فوق BERT، لفهم الآلية الداخلية بعمق
    (ما يحدث فعليًا خلف `AutoModelForQuestionAnswering`)
    """
    def __init__(self, backbone_name: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.bert.config.hidden_size
        # طبقة خطية واحدة تُنتج قيمتين لكل Token: (start_logit, end_logit)
        self.qa_outputs = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden]

        logits = self.qa_outputs(sequence_output)  # [batch, seq_len, 2]
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # [batch, seq_len]
        end_logits = end_logits.squeeze(-1)

        loss = None
        if start_positions is not None and end_positions is not None:
            loss_fn = nn.CrossEntropyLoss()
            start_loss = loss_fn(start_logits, start_positions)
            end_loss = loss_fn(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2

        return {"loss": loss, "start_logits": start_logits, "end_logits": end_logits}

    def predict_answer(self, input_ids, attention_mask, tokenizer) -> str:
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)

        start_idx = torch.argmax(outputs["start_logits"], dim=-1).item()
        end_idx = torch.argmax(outputs["end_logits"], dim=-1).item()

        if end_idx < start_idx:  # حماية من تنبؤات غير منطقية
            end_idx = start_idx

        answer_tokens = input_ids[0][start_idx:end_idx+1]
        return tokenizer.decode(answer_tokens, skip_special_tokens=True)

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-10-applications-agents/chapter-54.md
# ────────────────────────────────────────────────────────────────────────
def comprehensive_summary_evaluation(
    generated_summaries: list[str], reference_summaries: list[str]
) -> dict:
    """يقارن ROUGE التقليدي مقابل BERTScore الدلالي الحديث"""

    # --- ROUGE ---
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    rouge1_f, rouge2_f, rougeL_f = [], [], []

    for gen, ref in zip(generated_summaries, reference_summaries):
        scores = rouge.score(ref, gen)
        rouge1_f.append(scores["rouge1"].fmeasure)
        rouge2_f.append(scores["rouge2"].fmeasure)
        rougeL_f.append(scores["rougeL"].fmeasure)

    # --- BERTScore ---
    P, R, F1 = bert_score(
        generated_summaries, reference_summaries,
        lang="ar", model_type="bert-base-multilingual-cased"
    )

    return {
        "ROUGE-1": round(np.mean(rouge1_f), 3),
        "ROUGE-2": round(np.mean(rouge2_f), 3),
        "ROUGE-L": round(np.mean(rougeL_f), 3),
        "BERTScore-F1": round(F1.mean().item(), 3),
    }


# --- مثال يكشف الفرق بين المقياسين ---
generated = ["شهدت أسعار النفط ارتفاعًا ملحوظًا بسبب قرارات أوبك."]
reference = ["ارتفعت أسعار النفط العالمية نتيجة خفض أوبك للإنتاج."]

report = comprehensive_summary_evaluation(generated, reference)
print("📊 تقرير التقييم المقارن:")
for metric, value in report.items():
    print(f"  {metric}: {value}")

# النتيجة النمطية: ROUGE منخفض نسبيًا (كلمات مختلفة)،
# لكن BERTScore مرتفع جدًا (المعنى متطابق تقريبًا)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-54-chapter-54.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 54: الفصل 54: التلخيص والإجابة على الأسئلة."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_rouge() -> None:
    reference = 'attention combines token values using learned weights'.split()
    summary = 'attention combines values using weights'.split()
    previous = list(range(len(reference) + 1))
    for token in summary:
        current = [previous[0] + 1]
        for index, target in enumerate(reference, 1):
            current.append(previous[index - 1] + 1 if token == target else max(previous[index], current[-1]))
        previous = current
    lcs = previous[-1]
    precision, recall = (lcs / len(summary), lcs / len(reference))
    f1 = 2 * precision * recall / (precision + recall)
    print('LCS length:', lcs)
    print('ROUGE-L precision/recall/F1:', round(precision, 3), round(recall, 3), round(f1, 3))

if __name__ == "__main__":
    print("Chapter 54 lab: Score a summary with ROUGE-L")
    run_rouge()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-54-chapter-54.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# requirements.txt
# torch>=2.4.0
# transformers>=4.44.0
# rouge-score>=0.1.2
# bert-score>=0.3.13
# datasets>=2.20.0

import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering, pipeline,
    Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
)
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-54-chapter-54.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import networkx as nx
from sentence_transformers import SentenceTransformer
import re

class ExtractiveSummarizer:
    """
    خوارزمية TextRank: تبني رسمًا بيانيًا حيث كل جملة = عقدة،
    والوزن بين عقدتين = تشابههما الدلالي، ثم تُطبّق PageRank
    لإيجاد الجمل "الأكثر مركزية" (الأهم) في النص.
    """
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        self.embedder = SentenceTransformer(model_name)

    def summarize(self, text: str, num_sentences: int = 3) -> str:
        sentences = re.split(r'(?<=[.!?؟])\s+', text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if len(sentences) <= num_sentences:
            return text

        # 1. تضمين كل جملة
        embeddings = self.embedder.encode(sentences, convert_to_numpy=True)

        # 2. بناء مصفوفة التشابه (Similarity Matrix)
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8
                    )
                    similarity_matrix[i][j] = sim

        # 3. بناء رسم بياني وتطبيق PageRank
        graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(graph)

        # 4. اختيار أعلى الجمل مرتبة ثم إعادة ترتيبها بترتيبها الأصلي في النص
        ranked_indices = sorted(scores, key=scores.get, reverse=True)[:num_sentences]
        ranked_indices.sort()  # الحفاظ على التسلسل المنطقي الأصلي

        return " ".join([sentences[i] for i in ranked_indices])


# --- الاستخدام ---
extractive_summarizer = ExtractiveSummarizer()

long_text = """
يشهد قطاع الذكاء الاصطناعي نموًا متسارعًا غير مسبوق خلال السنوات الأخيرة.
الشركات الكبرى تستثمر مليارات الدولارات في تطوير نماذج لغوية ضخمة.
هذا التطور يطرح تحديات أخلاقية وتنظيمية جديدة لم تكن موجودة من قبل.
في المقابل، يرى الخبراء أن هذه التقنية ستُحدث تحولًا جذريًا في سوق العمل.
العديد من الوظائف التقليدية مهددة بالاستبدال، بينما تظهر وظائف جديدة كليًا.
الحكومات حول العالم تحاول وضع أطر تنظيمية لمواكبة هذا التسارع.
"""

summary = extractive_summarizer.summarize(long_text, num_sentences=2)
print(f"📝 الملخص الاستخراجي: {summary}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-54-chapter-54.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# --- الخطوة 1: تجهيز بيانات التلخيص ---
from datasets import Dataset

summarization_data = {
    "document": [
        "ارتفعت أسعار النفط العالمية بنسبة 5% اليوم بعد إعلان أوبك خفض الإنتاج. "
        "المحللون يتوقعون استمرار هذا الاتجاه الصعودي خلال الأسابيع القادمة بسبب "
        "التوترات الجيوسياسية المستمرة في منطقة الشرق الأوسط وتراجع المخزونات العالمية.",

        "أعلنت الشركة عن نتائج مالية قوية للربع الثالث، محققة أرباحًا تجاوزت توقعات "
        "المحللين بنسبة 15%. يُعزى هذا النمو إلى زيادة الطلب على منتجاتها الرقمية "
        "الجديدة وتوسعها في الأسواق الآسيوية الناشئة.",
    ],
    "summary": [
        "ارتفاع أسعار النفط 5% بعد قرار أوبك خفض الإنتاج وسط توترات جيوسياسية.",
        "الشركة تحقق أرباحًا تفوق التوقعات بـ15% مدفوعة بالمنتجات الرقمية والتوسع الآسيوي.",
    ]
}

dataset = Dataset.from_dict(summarization_data)
dataset = dataset.train_test_split(test_size=0.5, seed=42)

# --- الخطوة 2: استخدام نموذج mT5 (متعدد اللغات، يدعم العربية) ---
MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"  # مُدرَّب مسبقًا على التلخيص متعدد اللغات!

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 100

def preprocess_function(examples):
    inputs = tokenizer(
        examples["document"], max_length=MAX_INPUT_LENGTH, truncation=True, padding=False
    )
    labels = tokenizer(
        text_target=examples["summary"], max_length=MAX_TARGET_LENGTH, truncation=True, padding=False
    )
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

# --- الخطوة 3: دالة المقاييس باستخدام ROUGE ---
rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    for pred, label in zip(decoded_preds, decoded_labels):
        scores = rouge.score(label, pred)  # ترتيب: (reference, hypothesis)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": np.mean(rouge1_scores),
        "rouge2": np.mean(rouge2_scores),
        "rougeL": np.mean(rougeL_scores),
    }

# --- الخطوة 4: التدريب ---
training_args = Seq2SeqTrainingArguments(
    output_dir="./mt5_arabic_summarizer",
    learning_rate=3e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LENGTH,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    logging_steps=1,
    report_to="none",
    fp16=torch.cuda.is_available(),
)

trainer = Seq2SeqTrainer(
    model=model, args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# --- الخطوة 5: الاستدلال على نص جديد ---
def generate_summary(text: str, max_length: int = 100) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=MAX_INPUT_LENGTH).to(device)
    with torch.no_grad():
        summary_ids = model.generate(
            **inputs, max_length=max_length, num_beams=4,
            length_penalty=2.0,  # يُشجّع ملخصات أطول قليلًا (قيمة > 1)
            early_stopping=True
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-54-chapter-54.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# --- استخدام نموذج AraBERT مُدرَّب مسبقًا على SQuAD-style QA ---

QA_MODEL_NAME = "wissamantoun/araelectra-base-artydiqa"  # نموذج QA عربي جاهز

qa_pipeline = pipeline(
    "question-answering",
    model=QA_MODEL_NAME,
    tokenizer=QA_MODEL_NAME,
    device=0 if torch.cuda.is_available() else -1
)

context = """
تأسست المملكة العربية السعودية الحديثة في عام 1932 على يد الملك عبدالعزيز آل سعود،
بعد توحيد عدة مناطق ومدن تحت راية واحدة. تُعد الرياض عاصمة المملكة ومركزها السياسي
والإداري، بينما تُعتبر جدة البوابة الاقتصادية الرئيسية للبلاد على ساحل البحر الأحمر.
"""

question = "من أسس المملكة العربية السعودية الحديثة؟"

result = qa_pipeline(question=question, context=context)
print(f"❓ السؤال: {question}")
print(f"💬 الإجابة: {result['answer']}")
print(f"📊 درجة الثقة: {result['score']:.3f}")
print(f"📍 الموقع في النص: [{result['start']}:{result['end']}]")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-54-chapter-54.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import torch.nn as nn
from transformers import AutoModel

class CustomQAModel(nn.Module):
    """
    بناء يدوي لرأس QA فوق BERT، لفهم الآلية الداخلية بعمق
    (ما يحدث فعليًا خلف `AutoModelForQuestionAnswering`)
    """
    def __init__(self, backbone_name: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.bert.config.hidden_size
        # طبقة خطية واحدة تُنتج قيمتين لكل Token: (start_logit, end_logit)
        self.qa_outputs = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden]

        logits = self.qa_outputs(sequence_output)  # [batch, seq_len, 2]
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # [batch, seq_len]
        end_logits = end_logits.squeeze(-1)

        loss = None
        if start_positions is not None and end_positions is not None:
            loss_fn = nn.CrossEntropyLoss()
            start_loss = loss_fn(start_logits, start_positions)
            end_loss = loss_fn(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2

        return {"loss": loss, "start_logits": start_logits, "end_logits": end_logits}

    def predict_answer(self, input_ids, attention_mask, tokenizer) -> str:
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)

        start_idx = torch.argmax(outputs["start_logits"], dim=-1).item()
        end_idx = torch.argmax(outputs["end_logits"], dim=-1).item()

        if end_idx < start_idx:  # حماية من تنبؤات غير منطقية
            end_idx = start_idx

        answer_tokens = input_ids[0][start_idx:end_idx+1]
        return tokenizer.decode(answer_tokens, skip_special_tokens=True)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-54-chapter-54.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
def comprehensive_summary_evaluation(
    generated_summaries: list[str], reference_summaries: list[str]
) -> dict:
    """يقارن ROUGE التقليدي مقابل BERTScore الدلالي الحديث"""

    # --- ROUGE ---
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    rouge1_f, rouge2_f, rougeL_f = [], [], []

    for gen, ref in zip(generated_summaries, reference_summaries):
        scores = rouge.score(ref, gen)
        rouge1_f.append(scores["rouge1"].fmeasure)
        rouge2_f.append(scores["rouge2"].fmeasure)
        rougeL_f.append(scores["rougeL"].fmeasure)

    # --- BERTScore ---
    P, R, F1 = bert_score(
        generated_summaries, reference_summaries,
        lang="ar", model_type="bert-base-multilingual-cased"
    )

    return {
        "ROUGE-1": round(np.mean(rouge1_f), 3),
        "ROUGE-2": round(np.mean(rouge2_f), 3),
        "ROUGE-L": round(np.mean(rougeL_f), 3),
        "BERTScore-F1": round(F1.mean().item(), 3),
    }


# --- مثال يكشف الفرق بين المقياسين ---
generated = ["شهدت أسعار النفط ارتفاعًا ملحوظًا بسبب قرارات أوبك."]
reference = ["ارتفعت أسعار النفط العالمية نتيجة خفض أوبك للإنتاج."]

report = comprehensive_summary_evaluation(generated, reference)
print("📊 تقرير التقييم المقارن:")
for metric, value in report.items():
    print(f"  {metric}: {value}")

# النتيجة النمطية: ROUGE منخفض نسبيًا (كلمات مختلفة)،
# لكن BERTScore مرتفع جدًا (المعنى متطابق تقريبًا)
