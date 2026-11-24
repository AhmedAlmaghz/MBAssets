#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الترجمة الآلية (Machine Translation)
Large Language Model Engineering — Machine

Book slug : llm-engineering-69-chapters
Chapter   : ch-52-chapter-52 (lesson 52)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-52-chapter-52.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الترجمة الآلية (Machine Translation)
Large Language Model Engineering — Machine

Book slug : llm-engineering-69-chapters
Chapter   : ch-52-chapter-52 (lesson 52)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-52-chapter-52.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الترجمة الآلية (Machine Translation)
Large Language Model Engineering — Machine

Book slug : llm-engineering-69-chapters
Chapter   : ch-52-chapter-52 (lesson 52)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_52.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 52: الفصل 52: الترجمة الآلية (Machine Translation)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_bleu() -> None:
    reference = 'the cat is on the mat'.split()
    candidate = 'the cat sat on mat'.split()
    matches = sum((Counter(zip(candidate, candidate[1:])) & Counter(zip(reference, reference[1:]))).values())
    total = max(len(candidate) - 1, 1)
    brevity = min(1.0, math.exp(1 - len(reference) / len(candidate)))
    score = brevity * matches / total
    print('clipped bigram precision:', round(matches / total, 3))
    print('brevity penalty:', round(brevity, 3))
    print('toy BLEU-2 (without smoothing):', round(score, 3))

if __name__ == "__main__":
    print("Chapter 52 lab: Calculate corpus BLEU with brevity penalty")
    run_bleu()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-10-applications-agents/chapter-52.md
# ────────────────────────────────────────────────────────────────────────
# requirements.txt
# torch>=2.4.0
# transformers>=4.44.0
# datasets>=2.20.0
# sacrebleu>=2.4.0
# unbabel-comet>=2.2.0
# sentencepiece>=0.2.0

import torch
from transformers import (
    MarianMTModel, MarianTokenizer,
    AutoModelForSeq2SeqLM, AutoTokenizer,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
import sacrebleu
from comet import download_model, load_from_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-10-applications-agents/chapter-52.md
# ────────────────────────────────────────────────────────────────────────
# --- استخدام MarianMT للترجمة من الإنجليزية إلى العربية ---

MODEL_NAME = "Helsinki-NLP/opus-mt-en-ar"

tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME).to(device)

def translate(texts: list[str], max_length: int = 128) -> list[str]:
    """ترجمة دفعة من الجمل من الإنجليزية إلى العربية"""
    inputs = tokenizer(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length).to(device)

    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,          # Beam Search لجودة أفضل (سنتعمق به في الفصل 38)
            early_stopping=True,
            no_repeat_ngram_size=3,  # تجنّب تكرار نفس العبارة
        )

    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]


# --- الاستخدام ---
source_sentences = [
    "The weather is beautiful today.",
    "I would like to order a cup of coffee, please.",
    "Artificial intelligence is transforming the world rapidly."
]

translations = translate(source_sentences)
for src, trans in zip(source_sentences, translations):
    print(f"EN: {src}\nAR: {trans}\n")

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-10-applications-agents/chapter-52.md
# ────────────────────────────────────────────────────────────────────────
# --- الخطوة 1: تجهيز بيانات موازية مخصصة (مثال: نطاق طبي) ---
# في الواقع: تُحمَّل من مصادر مثل OPUS, ParaCrawl, أو بيانات داخلية

medical_parallel_data = {
    "en": [
        "The patient should take two tablets daily after meals.",
        "Symptoms include fever, headache, and fatigue.",
        "This medication may cause drowsiness.",
    ],
    "ar": [
        "يجب أن يتناول المريض قرصين يوميًا بعد الوجبات.",
        "تشمل الأعراض الحمى والصداع والإرهاق.",
        "قد يسبب هذا الدواء النعاس.",
    ]
}

from datasets import Dataset
raw_dataset = Dataset.from_dict(medical_parallel_data)
raw_dataset = raw_dataset.train_test_split(test_size=0.33, seed=42)

# --- الخطوة 2: دالة الترميز للترجمة (Seq2Seq) ---
MODEL_NAME = "Helsinki-NLP/opus-mt-en-ar"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

MAX_LENGTH = 128

def preprocess_function(examples):
    inputs = examples["en"]
    targets = examples["ar"]

    model_inputs = tokenizer(
        inputs, max_length=MAX_LENGTH, truncation=True, padding=False
    )

    # لترميز الهدف، نستخدم text_target في الإصدارات الحديثة من transformers
    labels = tokenizer(
        text_target=targets, max_length=MAX_LENGTH, truncation=True, padding=False
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = raw_dataset.map(preprocess_function, batched=True)

# --- الخطوة 3: Data Collator خاص بـ Seq2Seq ---
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, model=model, padding=True
)

# --- الخطوة 4: دالة المقاييس (BLEU + chrF أثناء التدريب) ---
import numpy as np

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # استبدال -100 بـ pad_token_id قبل فك الترميز
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # sacrebleu يتوقع المراجع كقائمة من القوائم (مرجع واحد أو أكثر لكل جملة)
    bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
    chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels])

    return {
        "bleu": bleu.score,
        "chrf": chrf.score,
    }

# --- الخطوة 5: إعدادات التدريب ---
training_args = Seq2SeqTrainingArguments(
    output_dir="./marian_medical_ar",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    num_train_epochs=10,
    predict_with_generate=True,   # مهم جدًا: يُولّد نصوصًا فعلية للتقييم وليس logits فقط
    generation_max_length=MAX_LENGTH,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    logging_steps=1,
    report_to="none",
    fp16=torch.cuda.is_available(),
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-10-applications-agents/chapter-52.md
# ────────────────────────────────────────────────────────────────────────
# comprehensive_mt_evaluation.py
"""
نظام تقييم شامل للترجمة الآلية يقارن BLEU, chrF, وCOMET جنبًا إلى جنب،
ويُظهر كيف يمكن أن تختلف النتائج بين المقاييس لنفس الترجمة.
"""

import sacrebleu
from comet import download_model, load_from_checkpoint

class MTEvaluator:
    def __init__(self, comet_model_name: str = "Unbabel/wmt22-comet-da"):
        print("📥 تحميل نموذج COMET (قد يستغرق بعض الوقت أول مرة)...")
        comet_path = download_model(comet_model_name)
        self.comet_model = load_from_checkpoint(comet_path)

    def evaluate_bleu(self, hypotheses: list[str], references: list[list[str]]) -> float:
        """BLEU: يتوقع مراجع متعددة محتملة لكل جملة (قائمة من القوائم)"""
        result = sacrebleu.corpus_bleu(hypotheses, references)
        return result.score

    def evaluate_chrf(self, hypotheses: list[str], references: list[list[str]]) -> float:
        result = sacrebleu.corpus_chrf(hypotheses, references, word_order=2)  # chrF++
        return result.score

    def evaluate_comet(
        self, sources: list[str], hypotheses: list[str], references: list[str]
    ) -> dict:
        """COMET يحتاج Source أيضًا، وليس فقط Reference — لأنه يفهم الدلالة الكاملة"""
        data = [
            {"src": s, "mt": h, "ref": r}
            for s, h, r in zip(sources, hypotheses, references)
        ]
        model_output = self.comet_model.predict(data, batch_size=8, gpus=0)
        return {
            "system_score": model_output.system_score,
            "segment_scores": model_output.scores,  # درجة لكل جملة على حدة
        }

    def full_report(
        self, sources: list[str], hypotheses: list[str], references: list[str]
    ) -> dict:
        """تقرير شامل بالمقاييس الثلاثة معًا"""
        refs_for_sacrebleu = [references]  # sacrebleu يتوقع هذا الشكل

        bleu_score = self.evaluate_bleu(hypotheses, refs_for_sacrebleu)
        chrf_score = self.evaluate_chrf(hypotheses, refs_for_sacrebleu)
        comet_result = self.evaluate_comet(sources, hypotheses, references)

        return {
            "BLEU": round(bleu_score, 2),
            "chrF++": round(chrf_score, 2),
            "COMET": round(comet_result["system_score"] * 100, 2),  # تطبيع لنفس المقياس 0-100
            "comet_segment_scores": comet_result["segment_scores"],
        }


# --- مثال توضيحي: حالة تكشف الفرق بين المقاييس ---
if __name__ == "__main__":
    evaluator = MTEvaluator()

    sources = [
        "The cat sat on the mat.",
        "She quickly finished her homework.",
    ]

    # ترجمة "صحيحة معنويًا" لكن بصياغة مختلفة عن المرجع
    hypotheses = [
        "استقرت القطة فوق البساط.",      # مرادفات: استقرت≈جلست, فوق≈على, البساط≈السجادة
        "أنهت واجباتها بسرعة.",
    ]

    references = [
        "جلست القطة على السجادة.",
        "هي أنهت واجبها المنزلي بسرعة.",
    ]

    report = evaluator.full_report(sources, hypotheses, references)

    print("📊 تقرير التقييم الشامل:")
    print(f"  BLEU   : {report['BLEU']}   ← سيكون منخفضًا نسبيًا (تطابق لفظي ضعيف)")
    print(f"  chrF++ : {report['chrF++']} ← أعلى من BLEU (تشابه حرفي جزئي)")
    print(f"  COMET  : {report['COMET']}  ← الأعلى والأكثر واقعية (يفهم أن المعنى متطابق)")

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-10-applications-agents/chapter-52.md
# ────────────────────────────────────────────────────────────────────────
class ProductionTranslationSystem:
    """نظام ترجمة إنتاجي يدمج الترجمة مع بوابة جودة تلقائية"""

    def __init__(self, model_name: str, comet_qe_model: str = "Unbabel/wmt22-cometkiwi-da"):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name).to(device)

        # نستخدم نسخة "Quality Estimation" من COMET التي لا تحتاج مرجعًا بشريًا!
        # هذا حاسم للإنتاج: لا نملك ترجمة مرجعية لكل طلب حقيقي من المستخدم
        print("📥 تحميل نموذج COMET-QE (Reference-free)...")
        comet_path = download_model(comet_qe_model)
        self.qe_model = load_from_checkpoint(comet_path)

        self.quality_threshold = 0.70  # عتبة دنيا مقبولة (0-1)

    def translate_with_quality_check(self, source_text: str) -> dict:
        # 1. الترجمة
        inputs = self.tokenizer(source_text, return_tensors="pt",
                                  truncation=True, max_length=128).to(device)
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs, max_length=128, num_beams=4, early_stopping=True
            )
        translation = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        # 2. تقييم الجودة بدون مرجع (Quality Estimation - QE)
        qe_data = [{"src": source_text, "mt": translation}]
        qe_result = self.qe_model.predict(qe_data, batch_size=1, gpus=0)
        quality_score = qe_result.system_score

        # 3. بوابة الجودة
        passed_quality_gate = quality_score >= self.quality_threshold

        return {
            "source": source_text,
            "translation": translation,
            "quality_score": round(quality_score, 3),
            "passed_quality_gate": passed_quality_gate,
            "action": (
                "✅ نشر مباشر" if passed_quality_gate
                else "⚠️ إرسال لمراجعة بشرية (Human-in-the-loop)"
            )
        }


# --- الاستخدام الفعلي ---
system = ProductionTranslationSystem("Helsinki-NLP/opus-mt-en-ar")

test_cases = [
    "Please contact customer support for further assistance.",
    "The quantum entanglement phenomenon violates classical intuitions about locality.",  # جملة تقنية صعبة
]

for text in test_cases:
    result = system.translate_with_quality_check(text)
    print(f"📝 المصدر: {result['source']}")
    print(f"🔄 الترجمة: {result['translation']}")
    print(f"📊 درجة الجودة: {result['quality_score']}")
    print(f"🚦 القرار: {result['action']}\n")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-52-chapter-52.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 52: الفصل 52: الترجمة الآلية (Machine Translation)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_bleu() -> None:
    reference = 'the cat is on the mat'.split()
    candidate = 'the cat sat on mat'.split()
    matches = sum((Counter(zip(candidate, candidate[1:])) & Counter(zip(reference, reference[1:]))).values())
    total = max(len(candidate) - 1, 1)
    brevity = min(1.0, math.exp(1 - len(reference) / len(candidate)))
    score = brevity * matches / total
    print('clipped bigram precision:', round(matches / total, 3))
    print('brevity penalty:', round(brevity, 3))
    print('toy BLEU-2 (without smoothing):', round(score, 3))

if __name__ == "__main__":
    print("Chapter 52 lab: Calculate corpus BLEU with brevity penalty")
    run_bleu()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-52-chapter-52.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# requirements.txt
# torch>=2.4.0
# transformers>=4.44.0
# datasets>=2.20.0
# sacrebleu>=2.4.0
# unbabel-comet>=2.2.0
# sentencepiece>=0.2.0

import torch
from transformers import (
    MarianMTModel, MarianTokenizer,
    AutoModelForSeq2SeqLM, AutoTokenizer,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
import sacrebleu
from comet import download_model, load_from_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-52-chapter-52.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# --- استخدام MarianMT للترجمة من الإنجليزية إلى العربية ---

MODEL_NAME = "Helsinki-NLP/opus-mt-en-ar"

tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME).to(device)

def translate(texts: list[str], max_length: int = 128) -> list[str]:
    """ترجمة دفعة من الجمل من الإنجليزية إلى العربية"""
    inputs = tokenizer(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length).to(device)

    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,          # Beam Search لجودة أفضل (سنتعمق به في الفصل 38)
            early_stopping=True,
            no_repeat_ngram_size=3,  # تجنّب تكرار نفس العبارة
        )

    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]


# --- الاستخدام ---
source_sentences = [
    "The weather is beautiful today.",
    "I would like to order a cup of coffee, please.",
    "Artificial intelligence is transforming the world rapidly."
]

translations = translate(source_sentences)
for src, trans in zip(source_sentences, translations):
    print(f"EN: {src}\nAR: {trans}\n")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-52-chapter-52.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# --- الخطوة 1: تجهيز بيانات موازية مخصصة (مثال: نطاق طبي) ---
# في الواقع: تُحمَّل من مصادر مثل OPUS, ParaCrawl, أو بيانات داخلية

medical_parallel_data = {
    "en": [
        "The patient should take two tablets daily after meals.",
        "Symptoms include fever, headache, and fatigue.",
        "This medication may cause drowsiness.",
    ],
    "ar": [
        "يجب أن يتناول المريض قرصين يوميًا بعد الوجبات.",
        "تشمل الأعراض الحمى والصداع والإرهاق.",
        "قد يسبب هذا الدواء النعاس.",
    ]
}

from datasets import Dataset
raw_dataset = Dataset.from_dict(medical_parallel_data)
raw_dataset = raw_dataset.train_test_split(test_size=0.33, seed=42)

# --- الخطوة 2: دالة الترميز للترجمة (Seq2Seq) ---
MODEL_NAME = "Helsinki-NLP/opus-mt-en-ar"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

MAX_LENGTH = 128

def preprocess_function(examples):
    inputs = examples["en"]
    targets = examples["ar"]

    model_inputs = tokenizer(
        inputs, max_length=MAX_LENGTH, truncation=True, padding=False
    )

    # لترميز الهدف، نستخدم text_target في الإصدارات الحديثة من transformers
    labels = tokenizer(
        text_target=targets, max_length=MAX_LENGTH, truncation=True, padding=False
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = raw_dataset.map(preprocess_function, batched=True)

# --- الخطوة 3: Data Collator خاص بـ Seq2Seq ---
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, model=model, padding=True
)

# --- الخطوة 4: دالة المقاييس (BLEU + chrF أثناء التدريب) ---
import numpy as np

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # استبدال -100 بـ pad_token_id قبل فك الترميز
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # sacrebleu يتوقع المراجع كقائمة من القوائم (مرجع واحد أو أكثر لكل جملة)
    bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
    chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels])

    return {
        "bleu": bleu.score,
        "chrf": chrf.score,
    }

# --- الخطوة 5: إعدادات التدريب ---
training_args = Seq2SeqTrainingArguments(
    output_dir="./marian_medical_ar",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    num_train_epochs=10,
    predict_with_generate=True,   # مهم جدًا: يُولّد نصوصًا فعلية للتقييم وليس logits فقط
    generation_max_length=MAX_LENGTH,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    logging_steps=1,
    report_to="none",
    fp16=torch.cuda.is_available(),
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-52-chapter-52.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# comprehensive_mt_evaluation.py
"""
نظام تقييم شامل للترجمة الآلية يقارن BLEU, chrF, وCOMET جنبًا إلى جنب،
ويُظهر كيف يمكن أن تختلف النتائج بين المقاييس لنفس الترجمة.
"""

import sacrebleu
from comet import download_model, load_from_checkpoint

class MTEvaluator:
    def __init__(self, comet_model_name: str = "Unbabel/wmt22-comet-da"):
        print("📥 تحميل نموذج COMET (قد يستغرق بعض الوقت أول مرة)...")
        comet_path = download_model(comet_model_name)
        self.comet_model = load_from_checkpoint(comet_path)

    def evaluate_bleu(self, hypotheses: list[str], references: list[list[str]]) -> float:
        """BLEU: يتوقع مراجع متعددة محتملة لكل جملة (قائمة من القوائم)"""
        result = sacrebleu.corpus_bleu(hypotheses, references)
        return result.score

    def evaluate_chrf(self, hypotheses: list[str], references: list[list[str]]) -> float:
        result = sacrebleu.corpus_chrf(hypotheses, references, word_order=2)  # chrF++
        return result.score

    def evaluate_comet(
        self, sources: list[str], hypotheses: list[str], references: list[str]
    ) -> dict:
        """COMET يحتاج Source أيضًا، وليس فقط Reference — لأنه يفهم الدلالة الكاملة"""
        data = [
            {"src": s, "mt": h, "ref": r}
            for s, h, r in zip(sources, hypotheses, references)
        ]
        model_output = self.comet_model.predict(data, batch_size=8, gpus=0)
        return {
            "system_score": model_output.system_score,
            "segment_scores": model_output.scores,  # درجة لكل جملة على حدة
        }

    def full_report(
        self, sources: list[str], hypotheses: list[str], references: list[str]
    ) -> dict:
        """تقرير شامل بالمقاييس الثلاثة معًا"""
        refs_for_sacrebleu = [references]  # sacrebleu يتوقع هذا الشكل

        bleu_score = self.evaluate_bleu(hypotheses, refs_for_sacrebleu)
        chrf_score = self.evaluate_chrf(hypotheses, refs_for_sacrebleu)
        comet_result = self.evaluate_comet(sources, hypotheses, references)

        return {
            "BLEU": round(bleu_score, 2),
            "chrF++": round(chrf_score, 2),
            "COMET": round(comet_result["system_score"] * 100, 2),  # تطبيع لنفس المقياس 0-100
            "comet_segment_scores": comet_result["segment_scores"],
        }


# --- مثال توضيحي: حالة تكشف الفرق بين المقاييس ---
if __name__ == "__main__":
    evaluator = MTEvaluator()

    sources = [
        "The cat sat on the mat.",
        "She quickly finished her homework.",
    ]

    # ترجمة "صحيحة معنويًا" لكن بصياغة مختلفة عن المرجع
    hypotheses = [
        "استقرت القطة فوق البساط.",      # مرادفات: استقرت≈جلست, فوق≈على, البساط≈السجادة
        "أنهت واجباتها بسرعة.",
    ]

    references = [
        "جلست القطة على السجادة.",
        "هي أنهت واجبها المنزلي بسرعة.",
    ]

    report = evaluator.full_report(sources, hypotheses, references)

    print("📊 تقرير التقييم الشامل:")
    print(f"  BLEU   : {report['BLEU']}   ← سيكون منخفضًا نسبيًا (تطابق لفظي ضعيف)")
    print(f"  chrF++ : {report['chrF++']} ← أعلى من BLEU (تشابه حرفي جزئي)")
    print(f"  COMET  : {report['COMET']}  ← الأعلى والأكثر واقعية (يفهم أن المعنى متطابق)")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-52-chapter-52.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
class ProductionTranslationSystem:
    """نظام ترجمة إنتاجي يدمج الترجمة مع بوابة جودة تلقائية"""

    def __init__(self, model_name: str, comet_qe_model: str = "Unbabel/wmt22-cometkiwi-da"):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name).to(device)

        # نستخدم نسخة "Quality Estimation" من COMET التي لا تحتاج مرجعًا بشريًا!
        # هذا حاسم للإنتاج: لا نملك ترجمة مرجعية لكل طلب حقيقي من المستخدم
        print("📥 تحميل نموذج COMET-QE (Reference-free)...")
        comet_path = download_model(comet_qe_model)
        self.qe_model = load_from_checkpoint(comet_path)

        self.quality_threshold = 0.70  # عتبة دنيا مقبولة (0-1)

    def translate_with_quality_check(self, source_text: str) -> dict:
        # 1. الترجمة
        inputs = self.tokenizer(source_text, return_tensors="pt",
                                  truncation=True, max_length=128).to(device)
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs, max_length=128, num_beams=4, early_stopping=True
            )
        translation = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        # 2. تقييم الجودة بدون مرجع (Quality Estimation - QE)
        qe_data = [{"src": source_text, "mt": translation}]
        qe_result = self.qe_model.predict(qe_data, batch_size=1, gpus=0)
        quality_score = qe_result.system_score

        # 3. بوابة الجودة
        passed_quality_gate = quality_score >= self.quality_threshold

        return {
            "source": source_text,
            "translation": translation,
            "quality_score": round(quality_score, 3),
            "passed_quality_gate": passed_quality_gate,
            "action": (
                "✅ نشر مباشر" if passed_quality_gate
                else "⚠️ إرسال لمراجعة بشرية (Human-in-the-loop)"
            )
        }


# --- الاستخدام الفعلي ---
system = ProductionTranslationSystem("Helsinki-NLP/opus-mt-en-ar")

test_cases = [
    "Please contact customer support for further assistance.",
    "The quantum entanglement phenomenon violates classical intuitions about locality.",  # جملة تقنية صعبة
]

for text in test_cases:
    result = system.translate_with_quality_check(text)
    print(f"📝 المصدر: {result['source']}")
    print(f"🔄 الترجمة: {result['translation']}")
    print(f"📊 درجة الجودة: {result['quality_score']}")
    print(f"🚦 القرار: {result['action']}\n")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-52-chapter-52.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الترجمة الآلية (Machine Translation)
Large Language Model Engineering — Machine

Book slug : llm-engineering-69-chapters
Chapter   : ch-52-chapter-52 (lesson 52)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_52.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 52: الفصل 52: الترجمة الآلية (Machine Translation)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_bleu() -> None:
    reference = 'the cat is on the mat'.split()
    candidate = 'the cat sat on mat'.split()
    matches = sum((Counter(zip(candidate, candidate[1:])) & Counter(zip(reference, reference[1:]))).values())
    total = max(len(candidate) - 1, 1)
    brevity = min(1.0, math.exp(1 - len(reference) / len(candidate)))
    score = brevity * matches / total
    print('clipped bigram precision:', round(matches / total, 3))
    print('brevity penalty:', round(brevity, 3))
    print('toy BLEU-2 (without smoothing):', round(score, 3))

if __name__ == "__main__":
    print("Chapter 52 lab: Calculate corpus BLEU with brevity penalty")
    run_bleu()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-10-applications-agents/chapter-52.md
# ────────────────────────────────────────────────────────────────────────
# requirements.txt
# torch>=2.4.0
# transformers>=4.44.0
# datasets>=2.20.0
# sacrebleu>=2.4.0
# unbabel-comet>=2.2.0
# sentencepiece>=0.2.0

import torch
from transformers import (
    MarianMTModel, MarianTokenizer,
    AutoModelForSeq2SeqLM, AutoTokenizer,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
import sacrebleu
from comet import download_model, load_from_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-10-applications-agents/chapter-52.md
# ────────────────────────────────────────────────────────────────────────
# --- استخدام MarianMT للترجمة من الإنجليزية إلى العربية ---

MODEL_NAME = "Helsinki-NLP/opus-mt-en-ar"

tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME).to(device)

def translate(texts: list[str], max_length: int = 128) -> list[str]:
    """ترجمة دفعة من الجمل من الإنجليزية إلى العربية"""
    inputs = tokenizer(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length).to(device)

    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,          # Beam Search لجودة أفضل (سنتعمق به في الفصل 38)
            early_stopping=True,
            no_repeat_ngram_size=3,  # تجنّب تكرار نفس العبارة
        )

    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]


# --- الاستخدام ---
source_sentences = [
    "The weather is beautiful today.",
    "I would like to order a cup of coffee, please.",
    "Artificial intelligence is transforming the world rapidly."
]

translations = translate(source_sentences)
for src, trans in zip(source_sentences, translations):
    print(f"EN: {src}\nAR: {trans}\n")

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-10-applications-agents/chapter-52.md
# ────────────────────────────────────────────────────────────────────────
# --- الخطوة 1: تجهيز بيانات موازية مخصصة (مثال: نطاق طبي) ---
# في الواقع: تُحمَّل من مصادر مثل OPUS, ParaCrawl, أو بيانات داخلية

medical_parallel_data = {
    "en": [
        "The patient should take two tablets daily after meals.",
        "Symptoms include fever, headache, and fatigue.",
        "This medication may cause drowsiness.",
    ],
    "ar": [
        "يجب أن يتناول المريض قرصين يوميًا بعد الوجبات.",
        "تشمل الأعراض الحمى والصداع والإرهاق.",
        "قد يسبب هذا الدواء النعاس.",
    ]
}

from datasets import Dataset
raw_dataset = Dataset.from_dict(medical_parallel_data)
raw_dataset = raw_dataset.train_test_split(test_size=0.33, seed=42)

# --- الخطوة 2: دالة الترميز للترجمة (Seq2Seq) ---
MODEL_NAME = "Helsinki-NLP/opus-mt-en-ar"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

MAX_LENGTH = 128

def preprocess_function(examples):
    inputs = examples["en"]
    targets = examples["ar"]

    model_inputs = tokenizer(
        inputs, max_length=MAX_LENGTH, truncation=True, padding=False
    )

    # لترميز الهدف، نستخدم text_target في الإصدارات الحديثة من transformers
    labels = tokenizer(
        text_target=targets, max_length=MAX_LENGTH, truncation=True, padding=False
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = raw_dataset.map(preprocess_function, batched=True)

# --- الخطوة 3: Data Collator خاص بـ Seq2Seq ---
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, model=model, padding=True
)

# --- الخطوة 4: دالة المقاييس (BLEU + chrF أثناء التدريب) ---
import numpy as np

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # استبدال -100 بـ pad_token_id قبل فك الترميز
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # sacrebleu يتوقع المراجع كقائمة من القوائم (مرجع واحد أو أكثر لكل جملة)
    bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
    chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels])

    return {
        "bleu": bleu.score,
        "chrf": chrf.score,
    }

# --- الخطوة 5: إعدادات التدريب ---
training_args = Seq2SeqTrainingArguments(
    output_dir="./marian_medical_ar",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    num_train_epochs=10,
    predict_with_generate=True,   # مهم جدًا: يُولّد نصوصًا فعلية للتقييم وليس logits فقط
    generation_max_length=MAX_LENGTH,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    logging_steps=1,
    report_to="none",
    fp16=torch.cuda.is_available(),
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-10-applications-agents/chapter-52.md
# ────────────────────────────────────────────────────────────────────────
# comprehensive_mt_evaluation.py
"""
نظام تقييم شامل للترجمة الآلية يقارن BLEU, chrF, وCOMET جنبًا إلى جنب،
ويُظهر كيف يمكن أن تختلف النتائج بين المقاييس لنفس الترجمة.
"""

import sacrebleu
from comet import download_model, load_from_checkpoint

class MTEvaluator:
    def __init__(self, comet_model_name: str = "Unbabel/wmt22-comet-da"):
        print("📥 تحميل نموذج COMET (قد يستغرق بعض الوقت أول مرة)...")
        comet_path = download_model(comet_model_name)
        self.comet_model = load_from_checkpoint(comet_path)

    def evaluate_bleu(self, hypotheses: list[str], references: list[list[str]]) -> float:
        """BLEU: يتوقع مراجع متعددة محتملة لكل جملة (قائمة من القوائم)"""
        result = sacrebleu.corpus_bleu(hypotheses, references)
        return result.score

    def evaluate_chrf(self, hypotheses: list[str], references: list[list[str]]) -> float:
        result = sacrebleu.corpus_chrf(hypotheses, references, word_order=2)  # chrF++
        return result.score

    def evaluate_comet(
        self, sources: list[str], hypotheses: list[str], references: list[str]
    ) -> dict:
        """COMET يحتاج Source أيضًا، وليس فقط Reference — لأنه يفهم الدلالة الكاملة"""
        data = [
            {"src": s, "mt": h, "ref": r}
            for s, h, r in zip(sources, hypotheses, references)
        ]
        model_output = self.comet_model.predict(data, batch_size=8, gpus=0)
        return {
            "system_score": model_output.system_score,
            "segment_scores": model_output.scores,  # درجة لكل جملة على حدة
        }

    def full_report(
        self, sources: list[str], hypotheses: list[str], references: list[str]
    ) -> dict:
        """تقرير شامل بالمقاييس الثلاثة معًا"""
        refs_for_sacrebleu = [references]  # sacrebleu يتوقع هذا الشكل

        bleu_score = self.evaluate_bleu(hypotheses, refs_for_sacrebleu)
        chrf_score = self.evaluate_chrf(hypotheses, refs_for_sacrebleu)
        comet_result = self.evaluate_comet(sources, hypotheses, references)

        return {
            "BLEU": round(bleu_score, 2),
            "chrF++": round(chrf_score, 2),
            "COMET": round(comet_result["system_score"] * 100, 2),  # تطبيع لنفس المقياس 0-100
            "comet_segment_scores": comet_result["segment_scores"],
        }


# --- مثال توضيحي: حالة تكشف الفرق بين المقاييس ---
if __name__ == "__main__":
    evaluator = MTEvaluator()

    sources = [
        "The cat sat on the mat.",
        "She quickly finished her homework.",
    ]

    # ترجمة "صحيحة معنويًا" لكن بصياغة مختلفة عن المرجع
    hypotheses = [
        "استقرت القطة فوق البساط.",      # مرادفات: استقرت≈جلست, فوق≈على, البساط≈السجادة
        "أنهت واجباتها بسرعة.",
    ]

    references = [
        "جلست القطة على السجادة.",
        "هي أنهت واجبها المنزلي بسرعة.",
    ]

    report = evaluator.full_report(sources, hypotheses, references)

    print("📊 تقرير التقييم الشامل:")
    print(f"  BLEU   : {report['BLEU']}   ← سيكون منخفضًا نسبيًا (تطابق لفظي ضعيف)")
    print(f"  chrF++ : {report['chrF++']} ← أعلى من BLEU (تشابه حرفي جزئي)")
    print(f"  COMET  : {report['COMET']}  ← الأعلى والأكثر واقعية (يفهم أن المعنى متطابق)")

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-10-applications-agents/chapter-52.md
# ────────────────────────────────────────────────────────────────────────
class ProductionTranslationSystem:
    """نظام ترجمة إنتاجي يدمج الترجمة مع بوابة جودة تلقائية"""

    def __init__(self, model_name: str, comet_qe_model: str = "Unbabel/wmt22-cometkiwi-da"):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name).to(device)

        # نستخدم نسخة "Quality Estimation" من COMET التي لا تحتاج مرجعًا بشريًا!
        # هذا حاسم للإنتاج: لا نملك ترجمة مرجعية لكل طلب حقيقي من المستخدم
        print("📥 تحميل نموذج COMET-QE (Reference-free)...")
        comet_path = download_model(comet_qe_model)
        self.qe_model = load_from_checkpoint(comet_path)

        self.quality_threshold = 0.70  # عتبة دنيا مقبولة (0-1)

    def translate_with_quality_check(self, source_text: str) -> dict:
        # 1. الترجمة
        inputs = self.tokenizer(source_text, return_tensors="pt",
                                  truncation=True, max_length=128).to(device)
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs, max_length=128, num_beams=4, early_stopping=True
            )
        translation = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        # 2. تقييم الجودة بدون مرجع (Quality Estimation - QE)
        qe_data = [{"src": source_text, "mt": translation}]
        qe_result = self.qe_model.predict(qe_data, batch_size=1, gpus=0)
        quality_score = qe_result.system_score

        # 3. بوابة الجودة
        passed_quality_gate = quality_score >= self.quality_threshold

        return {
            "source": source_text,
            "translation": translation,
            "quality_score": round(quality_score, 3),
            "passed_quality_gate": passed_quality_gate,
            "action": (
                "✅ نشر مباشر" if passed_quality_gate
                else "⚠️ إرسال لمراجعة بشرية (Human-in-the-loop)"
            )
        }


# --- الاستخدام الفعلي ---
system = ProductionTranslationSystem("Helsinki-NLP/opus-mt-en-ar")

test_cases = [
    "Please contact customer support for further assistance.",
    "The quantum entanglement phenomenon violates classical intuitions about locality.",  # جملة تقنية صعبة
]

for text in test_cases:
    result = system.translate_with_quality_check(text)
    print(f"📝 المصدر: {result['source']}")
    print(f"🔄 الترجمة: {result['translation']}")
    print(f"📊 درجة الجودة: {result['quality_score']}")
    print(f"🚦 القرار: {result['action']}\n")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-52-chapter-52.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 52: الفصل 52: الترجمة الآلية (Machine Translation)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_bleu() -> None:
    reference = 'the cat is on the mat'.split()
    candidate = 'the cat sat on mat'.split()
    matches = sum((Counter(zip(candidate, candidate[1:])) & Counter(zip(reference, reference[1:]))).values())
    total = max(len(candidate) - 1, 1)
    brevity = min(1.0, math.exp(1 - len(reference) / len(candidate)))
    score = brevity * matches / total
    print('clipped bigram precision:', round(matches / total, 3))
    print('brevity penalty:', round(brevity, 3))
    print('toy BLEU-2 (without smoothing):', round(score, 3))

if __name__ == "__main__":
    print("Chapter 52 lab: Calculate corpus BLEU with brevity penalty")
    run_bleu()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-52-chapter-52.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# requirements.txt
# torch>=2.4.0
# transformers>=4.44.0
# datasets>=2.20.0
# sacrebleu>=2.4.0
# unbabel-comet>=2.2.0
# sentencepiece>=0.2.0

import torch
from transformers import (
    MarianMTModel, MarianTokenizer,
    AutoModelForSeq2SeqLM, AutoTokenizer,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
import sacrebleu
from comet import download_model, load_from_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-52-chapter-52.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# --- استخدام MarianMT للترجمة من الإنجليزية إلى العربية ---

MODEL_NAME = "Helsinki-NLP/opus-mt-en-ar"

tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME).to(device)

def translate(texts: list[str], max_length: int = 128) -> list[str]:
    """ترجمة دفعة من الجمل من الإنجليزية إلى العربية"""
    inputs = tokenizer(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length).to(device)

    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,          # Beam Search لجودة أفضل (سنتعمق به في الفصل 38)
            early_stopping=True,
            no_repeat_ngram_size=3,  # تجنّب تكرار نفس العبارة
        )

    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]


# --- الاستخدام ---
source_sentences = [
    "The weather is beautiful today.",
    "I would like to order a cup of coffee, please.",
    "Artificial intelligence is transforming the world rapidly."
]

translations = translate(source_sentences)
for src, trans in zip(source_sentences, translations):
    print(f"EN: {src}\nAR: {trans}\n")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-52-chapter-52.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# --- الخطوة 1: تجهيز بيانات موازية مخصصة (مثال: نطاق طبي) ---
# في الواقع: تُحمَّل من مصادر مثل OPUS, ParaCrawl, أو بيانات داخلية

medical_parallel_data = {
    "en": [
        "The patient should take two tablets daily after meals.",
        "Symptoms include fever, headache, and fatigue.",
        "This medication may cause drowsiness.",
    ],
    "ar": [
        "يجب أن يتناول المريض قرصين يوميًا بعد الوجبات.",
        "تشمل الأعراض الحمى والصداع والإرهاق.",
        "قد يسبب هذا الدواء النعاس.",
    ]
}

from datasets import Dataset
raw_dataset = Dataset.from_dict(medical_parallel_data)
raw_dataset = raw_dataset.train_test_split(test_size=0.33, seed=42)

# --- الخطوة 2: دالة الترميز للترجمة (Seq2Seq) ---
MODEL_NAME = "Helsinki-NLP/opus-mt-en-ar"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

MAX_LENGTH = 128

def preprocess_function(examples):
    inputs = examples["en"]
    targets = examples["ar"]

    model_inputs = tokenizer(
        inputs, max_length=MAX_LENGTH, truncation=True, padding=False
    )

    # لترميز الهدف، نستخدم text_target في الإصدارات الحديثة من transformers
    labels = tokenizer(
        text_target=targets, max_length=MAX_LENGTH, truncation=True, padding=False
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = raw_dataset.map(preprocess_function, batched=True)

# --- الخطوة 3: Data Collator خاص بـ Seq2Seq ---
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, model=model, padding=True
)

# --- الخطوة 4: دالة المقاييس (BLEU + chrF أثناء التدريب) ---
import numpy as np

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # استبدال -100 بـ pad_token_id قبل فك الترميز
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # sacrebleu يتوقع المراجع كقائمة من القوائم (مرجع واحد أو أكثر لكل جملة)
    bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
    chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels])

    return {
        "bleu": bleu.score,
        "chrf": chrf.score,
    }

# --- الخطوة 5: إعدادات التدريب ---
training_args = Seq2SeqTrainingArguments(
    output_dir="./marian_medical_ar",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    num_train_epochs=10,
    predict_with_generate=True,   # مهم جدًا: يُولّد نصوصًا فعلية للتقييم وليس logits فقط
    generation_max_length=MAX_LENGTH,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    logging_steps=1,
    report_to="none",
    fp16=torch.cuda.is_available(),
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-52-chapter-52.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# comprehensive_mt_evaluation.py
"""
نظام تقييم شامل للترجمة الآلية يقارن BLEU, chrF, وCOMET جنبًا إلى جنب،
ويُظهر كيف يمكن أن تختلف النتائج بين المقاييس لنفس الترجمة.
"""

import sacrebleu
from comet import download_model, load_from_checkpoint

class MTEvaluator:
    def __init__(self, comet_model_name: str = "Unbabel/wmt22-comet-da"):
        print("📥 تحميل نموذج COMET (قد يستغرق بعض الوقت أول مرة)...")
        comet_path = download_model(comet_model_name)
        self.comet_model = load_from_checkpoint(comet_path)

    def evaluate_bleu(self, hypotheses: list[str], references: list[list[str]]) -> float:
        """BLEU: يتوقع مراجع متعددة محتملة لكل جملة (قائمة من القوائم)"""
        result = sacrebleu.corpus_bleu(hypotheses, references)
        return result.score

    def evaluate_chrf(self, hypotheses: list[str], references: list[list[str]]) -> float:
        result = sacrebleu.corpus_chrf(hypotheses, references, word_order=2)  # chrF++
        return result.score

    def evaluate_comet(
        self, sources: list[str], hypotheses: list[str], references: list[str]
    ) -> dict:
        """COMET يحتاج Source أيضًا، وليس فقط Reference — لأنه يفهم الدلالة الكاملة"""
        data = [
            {"src": s, "mt": h, "ref": r}
            for s, h, r in zip(sources, hypotheses, references)
        ]
        model_output = self.comet_model.predict(data, batch_size=8, gpus=0)
        return {
            "system_score": model_output.system_score,
            "segment_scores": model_output.scores,  # درجة لكل جملة على حدة
        }

    def full_report(
        self, sources: list[str], hypotheses: list[str], references: list[str]
    ) -> dict:
        """تقرير شامل بالمقاييس الثلاثة معًا"""
        refs_for_sacrebleu = [references]  # sacrebleu يتوقع هذا الشكل

        bleu_score = self.evaluate_bleu(hypotheses, refs_for_sacrebleu)
        chrf_score = self.evaluate_chrf(hypotheses, refs_for_sacrebleu)
        comet_result = self.evaluate_comet(sources, hypotheses, references)

        return {
            "BLEU": round(bleu_score, 2),
            "chrF++": round(chrf_score, 2),
            "COMET": round(comet_result["system_score"] * 100, 2),  # تطبيع لنفس المقياس 0-100
            "comet_segment_scores": comet_result["segment_scores"],
        }


# --- مثال توضيحي: حالة تكشف الفرق بين المقاييس ---
if __name__ == "__main__":
    evaluator = MTEvaluator()

    sources = [
        "The cat sat on the mat.",
        "She quickly finished her homework.",
    ]

    # ترجمة "صحيحة معنويًا" لكن بصياغة مختلفة عن المرجع
    hypotheses = [
        "استقرت القطة فوق البساط.",      # مرادفات: استقرت≈جلست, فوق≈على, البساط≈السجادة
        "أنهت واجباتها بسرعة.",
    ]

    references = [
        "جلست القطة على السجادة.",
        "هي أنهت واجبها المنزلي بسرعة.",
    ]

    report = evaluator.full_report(sources, hypotheses, references)

    print("📊 تقرير التقييم الشامل:")
    print(f"  BLEU   : {report['BLEU']}   ← سيكون منخفضًا نسبيًا (تطابق لفظي ضعيف)")
    print(f"  chrF++ : {report['chrF++']} ← أعلى من BLEU (تشابه حرفي جزئي)")
    print(f"  COMET  : {report['COMET']}  ← الأعلى والأكثر واقعية (يفهم أن المعنى متطابق)")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-52-chapter-52.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
class ProductionTranslationSystem:
    """نظام ترجمة إنتاجي يدمج الترجمة مع بوابة جودة تلقائية"""

    def __init__(self, model_name: str, comet_qe_model: str = "Unbabel/wmt22-cometkiwi-da"):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name).to(device)

        # نستخدم نسخة "Quality Estimation" من COMET التي لا تحتاج مرجعًا بشريًا!
        # هذا حاسم للإنتاج: لا نملك ترجمة مرجعية لكل طلب حقيقي من المستخدم
        print("📥 تحميل نموذج COMET-QE (Reference-free)...")
        comet_path = download_model(comet_qe_model)
        self.qe_model = load_from_checkpoint(comet_path)

        self.quality_threshold = 0.70  # عتبة دنيا مقبولة (0-1)

    def translate_with_quality_check(self, source_text: str) -> dict:
        # 1. الترجمة
        inputs = self.tokenizer(source_text, return_tensors="pt",
                                  truncation=True, max_length=128).to(device)
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs, max_length=128, num_beams=4, early_stopping=True
            )
        translation = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        # 2. تقييم الجودة بدون مرجع (Quality Estimation - QE)
        qe_data = [{"src": source_text, "mt": translation}]
        qe_result = self.qe_model.predict(qe_data, batch_size=1, gpus=0)
        quality_score = qe_result.system_score

        # 3. بوابة الجودة
        passed_quality_gate = quality_score >= self.quality_threshold

        return {
            "source": source_text,
            "translation": translation,
            "quality_score": round(quality_score, 3),
            "passed_quality_gate": passed_quality_gate,
            "action": (
                "✅ نشر مباشر" if passed_quality_gate
                else "⚠️ إرسال لمراجعة بشرية (Human-in-the-loop)"
            )
        }


# --- الاستخدام الفعلي ---
system = ProductionTranslationSystem("Helsinki-NLP/opus-mt-en-ar")

test_cases = [
    "Please contact customer support for further assistance.",
    "The quantum entanglement phenomenon violates classical intuitions about locality.",  # جملة تقنية صعبة
]

for text in test_cases:
    result = system.translate_with_quality_check(text)
    print(f"📝 المصدر: {result['source']}")
    print(f"🔄 الترجمة: {result['translation']}")
    print(f"📊 درجة الجودة: {result['quality_score']}")
    print(f"🚦 القرار: {result['action']}\n")
