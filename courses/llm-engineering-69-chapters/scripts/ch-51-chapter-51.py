#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — تحليل المشاعر واستخراج المعلومات (Sentiment Analysis & NER)
Large Language Model Engineering — Sentiment

Book slug : llm-engineering-69-chapters
Chapter   : ch-51-chapter-51 (lesson 51)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-51-chapter-51.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — تحليل المشاعر واستخراج المعلومات (Sentiment Analysis & NER)
Large Language Model Engineering — Sentiment

Book slug : llm-engineering-69-chapters
Chapter   : ch-51-chapter-51 (lesson 51)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-51-chapter-51.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — تحليل المشاعر واستخراج المعلومات (Sentiment Analysis & NER)
Large Language Model Engineering — Sentiment

Book slug : llm-engineering-69-chapters
Chapter   : ch-51-chapter-51 (lesson 51)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_51.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 51: الفصل 51: تحليل المشاعر واستخراج المعلومات (Sentiment Analysis & NER)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_nlp() -> None:
    text = 'Aisha said the model helped her team in Paris.'
    positive = {'helped', 'excellent', 'useful'}
    negative = {'failed', 'poor', 'harmful'}
    words = re.findall("\\b[\\w']+\\b", text.lower())
    sentiment = sum((word in positive for word in words)) - sum((word in negative for word in words))
    entities = re.findall('\\b[A-Z][a-z]+\\b', text)
    print('lexicon sentiment score:', sentiment)
    print('candidate proper-name entities:', entities)
    print('review ambiguous entities before production use.')

if __name__ == "__main__":
    print("Chapter 51 lab: Combine a sentiment lexicon with entity extraction")
    run_nlp()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-10-applications-agents/chapter-51.md
# ────────────────────────────────────────────────────────────────────────
# requirements.txt
# torch>=2.4.0
# transformers>=4.44.0
# datasets>=2.20.0
# seqeval>=1.2.2
# accelerate>=0.33.0

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    DataCollatorForTokenClassification
)
from datasets import load_dataset, Dataset
import numpy as np
from seqeval.metrics import classification_report, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ الجهاز المستخدم: {device}")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-10-applications-agents/chapter-51.md
# ────────────────────────────────────────────────────────────────────────
MODEL_NAME = "aubmindlab/bert-base-arabertv2"

# --- الخطوة 1: تحميل الـ Tokenizer والبيانات ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# بيانات تجريبية (في الواقع تُحمَّل من ملف/قاعدة بيانات حقيقية)
raw_data = {
    "text": [
        "هذا المنتج رائع جدًا وأنصح الجميع بشرائه",
        "جودة سيئة للغاية ولا أنصح بهذا المنتج إطلاقًا",
        "المنتج عادي، لا بأس به لكن ليس استثنائيًا",
        "تجربة ممتازة والتوصيل كان سريعًا جدًا",
        "خدمة العملاء سيئة والمنتج وصل تالفًا",
    ],
    "label": [2, 0, 1, 2, 0]  # 0: سلبي, 1: محايد, 2: إيجابي
}

dataset = Dataset.from_dict(raw_data)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# --- الخطوة 2: دالة الترميز (Tokenization) ---
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,   # الـ padding سيُنفَّذ ديناميكيًا عبر Collator
        max_length=128
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# --- الخطوة 3: تحميل النموذج مع رأس تصنيف ---
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label={0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"},
    label2id={"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
).to(device)

# --- الخطوة 4: Data Collator للـ Padding الديناميكي ---
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# --- الخطوة 5: دالة حساب المقاييس ---
def compute_metrics_sentiment(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

# --- الخطوة 6: إعدادات التدريب ---
training_args = TrainingArguments(
    output_dir="./sentiment_arabert",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",   # عطّل WandB إن لم يكن مُعدًّا
    fp16=torch.cuda.is_available(),  # Mixed Precision إن توفّر GPU
)

# --- الخطوة 7: التدريب ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics_sentiment,
)

trainer.train()

# --- الخطوة 8: الاستدلال (Inference) على نص جديد ---
def predict_sentiment(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred_id = int(np.argmax(probs))
    return {
        "label": model.config.id2label[pred_id],
        "confidence": float(probs[pred_id]),
        "all_probs": {model.config.id2label[i]: float(p) for i, p in enumerate(probs)}
    }

result = predict_sentiment("التطبيق بطيء جدًا ويتعطل باستمرار")
print(result)
# المخرج المتوقع: {'label': 'NEGATIVE', 'confidence': 0.91, ...}

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-10-applications-agents/chapter-51.md
# ────────────────────────────────────────────────────────────────────────
# --- الخطوة 1: تجهيز بيانات NER بمخطط BIO ---
# كل جملة = قائمة كلمات، وكل جملة تقابلها قائمة بنفس الطول من الوسوم

ner_label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
label2id_ner = {label: idx for idx, label in enumerate(ner_label_list)}
id2label_ner = {idx: label for idx, label in enumerate(ner_label_list)}

raw_ner_data = {
    "tokens": [
        ["ذهب", "محمد", "أحمد", "إلى", "القاهرة"],
        ["تعمل", "سارة", "في", "شركة", "جوجل"],
        ["زار", "الرئيس", "باريس", "الأسبوع", "الماضي"],
    ],
    "ner_tags": [
        [0, 1, 2, 0, 5],    # O B-PER I-PER O B-LOC
        [0, 1, 0, 3, 4],    # O B-PER O B-ORG I-ORG
        [0, 0, 5, 0, 0],    # O O B-LOC O O
    ]
}

ner_dataset = Dataset.from_dict(raw_ner_data)
ner_dataset = ner_dataset.train_test_split(test_size=0.33, seed=42)

tokenizer_ner = AutoTokenizer.from_pretrained(MODEL_NAME)

# --- الخطوة 2: محاذاة الوسوم مع الـ Subword Tokens (الجزء الأكثر دقة في NER) ---
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer_ner(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,  # مهم جدًا: الإدخال كلمات مُقسَّمة مسبقًا
        padding=False,
        max_length=128
    )

    all_labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # يربط كل Subword بكلمته الأصلية
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                # رموز خاصة مثل [CLS], [SEP], [PAD]
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # أول Subword من الكلمة → يأخذ الوسم الحقيقي
                label_ids.append(label[word_idx])
            else:
                # باقي الـ Subwords لنفس الكلمة → نتجاهلها في حساب الخسارة
                label_ids.append(-100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

tokenized_ner_datasets = ner_dataset.map(tokenize_and_align_labels, batched=True)

# --- الخطوة 3: تحميل النموذج لمهمة Token Classification ---
ner_model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(ner_label_list),
    id2label=id2label_ner,
    label2id=label2id_ner
).to(device)

# --- الخطوة 4: Collator خاص بـ Token Classification ---
ner_data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer_ner)

# --- الخطوة 5: دالة المقاييس باستخدام seqeval (المعيار الصناعي لـ NER) ---
def compute_metrics_ner(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # إزالة التنبؤات الخاصة بـ -100 (الـ Subwords الفرعية والـ Padding)
    true_predictions = [
        [id2label_ner[p] for p, l in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label_ner[l] for p, l in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]

    f1 = f1_score(true_labels, true_predictions)
    report = classification_report(true_labels, true_predictions, output_dict=True)

    return {
        "f1": f1,
        "precision": report["micro avg"]["precision"],
        "recall": report["micro avg"]["recall"],
    }

# --- الخطوة 6: إعدادات وتشغيل التدريب ---
ner_training_args = TrainingArguments(
    output_dir="./ner_arabert",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=5,
    report_to="none",
    fp16=torch.cuda.is_available(),
)

ner_trainer = Trainer(
    model=ner_model,
    args=ner_training_args,
    train_dataset=tokenized_ner_datasets["train"],
    eval_dataset=tokenized_ner_datasets["test"],
    data_collator=ner_data_collator,
    compute_metrics=compute_metrics_ner,
)

ner_trainer.train()

# --- الخطوة 7: Inference كامل مع إعادة تجميع الكيانات متعددة الكلمات ---
def extract_entities(text: str) -> list[dict]:
    tokens = text.split()  # تقسيم بسيط بالمسافات (في الإنتاج: استخدم أداة تجزئة أدق)
    inputs = tokenizer_ner(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True
    ).to(device)

    ner_model.eval()
    with torch.no_grad():
        logits = ner_model(**inputs).logits
    predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()

    word_ids = inputs.word_ids(batch_index=0)
    entities = []
    current_entity = None

    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        # نأخذ فقط أول Subword من كل كلمة (لتجنب التكرار)
        if idx > 0 and word_ids[idx-1] == word_idx:
            continue

        tag = id2label_ner[predictions[idx]]
        word = tokens[word_idx]

        if tag.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {"text": word, "type": tag[2:], "start_word": word_idx}
        elif tag.startswith("I-") and current_entity and current_entity["type"] == tag[2:]:
            current_entity["text"] += " " + word
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    return entities

result = extract_entities("التقى أحمد علي بوزير الخارجية في دبي")
print(result)
# المخرج المتوقع: [{'text': 'أحمد علي', 'type': 'PER', ...}, {'text': 'دبي', 'type': 'LOC', ...}]

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-10-applications-agents/chapter-51.md
# ────────────────────────────────────────────────────────────────────────
class ReviewAnalyzer:
    """نظام متكامل يجمع تحليل المشاعر واستخراج الكيانات من مراجعات المنتجات"""

    def __init__(self, sentiment_model_path: str, ner_model_path: str):
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            sentiment_model_path
        ).to(device).eval()

        self.ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_path)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(
            ner_model_path
        ).to(device).eval()

    @torch.no_grad()
    def analyze(self, review_text: str) -> dict:
        # 1. تحليل المشاعر
        sent_inputs = self.sentiment_tokenizer(
            review_text, return_tensors="pt", truncation=True
        ).to(device)
        sent_logits = self.sentiment_model(**sent_inputs).logits
        sent_probs = torch.softmax(sent_logits, dim=-1)[0]
        sent_label = self.sentiment_model.config.id2label[int(sent_probs.argmax())]

        # 2. استخراج الكيانات
        tokens = review_text.split()
        ner_inputs = self.ner_tokenizer(
            tokens, is_split_into_words=True, return_tensors="pt", truncation=True
        ).to(device)
        ner_logits = self.ner_model(**ner_inputs).logits
        ner_preds = torch.argmax(ner_logits, dim=-1)[0].cpu().numpy()

        word_ids = ner_inputs.word_ids(batch_index=0)
        entities = self._aggregate_entities(tokens, word_ids, ner_preds)

        return {
            "text": review_text,
            "sentiment": {
                "label": sent_label,
                "confidence": round(float(sent_probs.max()), 3)
            },
            "entities": entities,
            "actionable_insight": self._generate_insight(sent_label, entities)
        }

    def _aggregate_entities(self, tokens, word_ids, predictions):
        id2label = self.ner_model.config.id2label
        entities, current = [], None
        for idx, wid in enumerate(word_ids):
            if wid is None or (idx > 0 and word_ids[idx-1] == wid):
                continue
            tag = id2label[predictions[idx]]
            if tag.startswith("B-"):
                if current: entities.append(current)
                current = {"text": tokens[wid], "type": tag[2:]}
            elif tag.startswith("I-") and current and current["type"] == tag[2:]:
                current["text"] += " " + tokens[wid]
            else:
                if current: entities.append(current); current = None
        if current: entities.append(current)
        return entities

    def _generate_insight(self, sentiment, entities):
        if sentiment == "NEGATIVE" and entities:
            products = [e["text"] for e in entities if e["type"] in ("ORG", "PRODUCT")]
            if products:
                return f"⚠️ تنبيه: شكوى سلبية متعلقة بـ {', '.join(products)} — يُنصح بالمتابعة الفورية"
        return "✅ لا حاجة لإجراء عاجل"


# --- التشغيل الفعلي ---
analyzer = ReviewAnalyzer("./sentiment_arabert", "./ner_arabert")
report = analyzer.analyze("خدمة العملاء في شركة جوجل كانت سيئة جدًا")
print(report)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-51-chapter-51.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 51: الفصل 51: تحليل المشاعر واستخراج المعلومات (Sentiment Analysis & NER)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_nlp() -> None:
    text = 'Aisha said the model helped her team in Paris.'
    positive = {'helped', 'excellent', 'useful'}
    negative = {'failed', 'poor', 'harmful'}
    words = re.findall("\\b[\\w']+\\b", text.lower())
    sentiment = sum((word in positive for word in words)) - sum((word in negative for word in words))
    entities = re.findall('\\b[A-Z][a-z]+\\b', text)
    print('lexicon sentiment score:', sentiment)
    print('candidate proper-name entities:', entities)
    print('review ambiguous entities before production use.')

if __name__ == "__main__":
    print("Chapter 51 lab: Combine a sentiment lexicon with entity extraction")
    run_nlp()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-51-chapter-51.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# requirements.txt
# torch>=2.4.0
# transformers>=4.44.0
# datasets>=2.20.0
# seqeval>=1.2.2
# accelerate>=0.33.0

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    DataCollatorForTokenClassification
)
from datasets import load_dataset, Dataset
import numpy as np
from seqeval.metrics import classification_report, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ الجهاز المستخدم: {device}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-51-chapter-51.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
MODEL_NAME = "aubmindlab/bert-base-arabertv2"

# --- الخطوة 1: تحميل الـ Tokenizer والبيانات ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# بيانات تجريبية (في الواقع تُحمَّل من ملف/قاعدة بيانات حقيقية)
raw_data = {
    "text": [
        "هذا المنتج رائع جدًا وأنصح الجميع بشرائه",
        "جودة سيئة للغاية ولا أنصح بهذا المنتج إطلاقًا",
        "المنتج عادي، لا بأس به لكن ليس استثنائيًا",
        "تجربة ممتازة والتوصيل كان سريعًا جدًا",
        "خدمة العملاء سيئة والمنتج وصل تالفًا",
    ],
    "label": [2, 0, 1, 2, 0]  # 0: سلبي, 1: محايد, 2: إيجابي
}

dataset = Dataset.from_dict(raw_data)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# --- الخطوة 2: دالة الترميز (Tokenization) ---
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,   # الـ padding سيُنفَّذ ديناميكيًا عبر Collator
        max_length=128
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# --- الخطوة 3: تحميل النموذج مع رأس تصنيف ---
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label={0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"},
    label2id={"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
).to(device)

# --- الخطوة 4: Data Collator للـ Padding الديناميكي ---
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# --- الخطوة 5: دالة حساب المقاييس ---
def compute_metrics_sentiment(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

# --- الخطوة 6: إعدادات التدريب ---
training_args = TrainingArguments(
    output_dir="./sentiment_arabert",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",   # عطّل WandB إن لم يكن مُعدًّا
    fp16=torch.cuda.is_available(),  # Mixed Precision إن توفّر GPU
)

# --- الخطوة 7: التدريب ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics_sentiment,
)

trainer.train()

# --- الخطوة 8: الاستدلال (Inference) على نص جديد ---
def predict_sentiment(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred_id = int(np.argmax(probs))
    return {
        "label": model.config.id2label[pred_id],
        "confidence": float(probs[pred_id]),
        "all_probs": {model.config.id2label[i]: float(p) for i, p in enumerate(probs)}
    }

result = predict_sentiment("التطبيق بطيء جدًا ويتعطل باستمرار")
print(result)
# المخرج المتوقع: {'label': 'NEGATIVE', 'confidence': 0.91, ...}

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-51-chapter-51.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# --- الخطوة 1: تجهيز بيانات NER بمخطط BIO ---
# كل جملة = قائمة كلمات، وكل جملة تقابلها قائمة بنفس الطول من الوسوم

ner_label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
label2id_ner = {label: idx for idx, label in enumerate(ner_label_list)}
id2label_ner = {idx: label for idx, label in enumerate(ner_label_list)}

raw_ner_data = {
    "tokens": [
        ["ذهب", "محمد", "أحمد", "إلى", "القاهرة"],
        ["تعمل", "سارة", "في", "شركة", "جوجل"],
        ["زار", "الرئيس", "باريس", "الأسبوع", "الماضي"],
    ],
    "ner_tags": [
        [0, 1, 2, 0, 5],    # O B-PER I-PER O B-LOC
        [0, 1, 0, 3, 4],    # O B-PER O B-ORG I-ORG
        [0, 0, 5, 0, 0],    # O O B-LOC O O
    ]
}

ner_dataset = Dataset.from_dict(raw_ner_data)
ner_dataset = ner_dataset.train_test_split(test_size=0.33, seed=42)

tokenizer_ner = AutoTokenizer.from_pretrained(MODEL_NAME)

# --- الخطوة 2: محاذاة الوسوم مع الـ Subword Tokens (الجزء الأكثر دقة في NER) ---
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer_ner(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,  # مهم جدًا: الإدخال كلمات مُقسَّمة مسبقًا
        padding=False,
        max_length=128
    )

    all_labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # يربط كل Subword بكلمته الأصلية
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                # رموز خاصة مثل [CLS], [SEP], [PAD]
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # أول Subword من الكلمة → يأخذ الوسم الحقيقي
                label_ids.append(label[word_idx])
            else:
                # باقي الـ Subwords لنفس الكلمة → نتجاهلها في حساب الخسارة
                label_ids.append(-100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

tokenized_ner_datasets = ner_dataset.map(tokenize_and_align_labels, batched=True)

# --- الخطوة 3: تحميل النموذج لمهمة Token Classification ---
ner_model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(ner_label_list),
    id2label=id2label_ner,
    label2id=label2id_ner
).to(device)

# --- الخطوة 4: Collator خاص بـ Token Classification ---
ner_data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer_ner)

# --- الخطوة 5: دالة المقاييس باستخدام seqeval (المعيار الصناعي لـ NER) ---
def compute_metrics_ner(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # إزالة التنبؤات الخاصة بـ -100 (الـ Subwords الفرعية والـ Padding)
    true_predictions = [
        [id2label_ner[p] for p, l in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label_ner[l] for p, l in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]

    f1 = f1_score(true_labels, true_predictions)
    report = classification_report(true_labels, true_predictions, output_dict=True)

    return {
        "f1": f1,
        "precision": report["micro avg"]["precision"],
        "recall": report["micro avg"]["recall"],
    }

# --- الخطوة 6: إعدادات وتشغيل التدريب ---
ner_training_args = TrainingArguments(
    output_dir="./ner_arabert",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=5,
    report_to="none",
    fp16=torch.cuda.is_available(),
)

ner_trainer = Trainer(
    model=ner_model,
    args=ner_training_args,
    train_dataset=tokenized_ner_datasets["train"],
    eval_dataset=tokenized_ner_datasets["test"],
    data_collator=ner_data_collator,
    compute_metrics=compute_metrics_ner,
)

ner_trainer.train()

# --- الخطوة 7: Inference كامل مع إعادة تجميع الكيانات متعددة الكلمات ---
def extract_entities(text: str) -> list[dict]:
    tokens = text.split()  # تقسيم بسيط بالمسافات (في الإنتاج: استخدم أداة تجزئة أدق)
    inputs = tokenizer_ner(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True
    ).to(device)

    ner_model.eval()
    with torch.no_grad():
        logits = ner_model(**inputs).logits
    predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()

    word_ids = inputs.word_ids(batch_index=0)
    entities = []
    current_entity = None

    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        # نأخذ فقط أول Subword من كل كلمة (لتجنب التكرار)
        if idx > 0 and word_ids[idx-1] == word_idx:
            continue

        tag = id2label_ner[predictions[idx]]
        word = tokens[word_idx]

        if tag.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {"text": word, "type": tag[2:], "start_word": word_idx}
        elif tag.startswith("I-") and current_entity and current_entity["type"] == tag[2:]:
            current_entity["text"] += " " + word
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    return entities

result = extract_entities("التقى أحمد علي بوزير الخارجية في دبي")
print(result)
# المخرج المتوقع: [{'text': 'أحمد علي', 'type': 'PER', ...}, {'text': 'دبي', 'type': 'LOC', ...}]

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-51-chapter-51.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class ReviewAnalyzer:
    """نظام متكامل يجمع تحليل المشاعر واستخراج الكيانات من مراجعات المنتجات"""

    def __init__(self, sentiment_model_path: str, ner_model_path: str):
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            sentiment_model_path
        ).to(device).eval()

        self.ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_path)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(
            ner_model_path
        ).to(device).eval()

    @torch.no_grad()
    def analyze(self, review_text: str) -> dict:
        # 1. تحليل المشاعر
        sent_inputs = self.sentiment_tokenizer(
            review_text, return_tensors="pt", truncation=True
        ).to(device)
        sent_logits = self.sentiment_model(**sent_inputs).logits
        sent_probs = torch.softmax(sent_logits, dim=-1)[0]
        sent_label = self.sentiment_model.config.id2label[int(sent_probs.argmax())]

        # 2. استخراج الكيانات
        tokens = review_text.split()
        ner_inputs = self.ner_tokenizer(
            tokens, is_split_into_words=True, return_tensors="pt", truncation=True
        ).to(device)
        ner_logits = self.ner_model(**ner_inputs).logits
        ner_preds = torch.argmax(ner_logits, dim=-1)[0].cpu().numpy()

        word_ids = ner_inputs.word_ids(batch_index=0)
        entities = self._aggregate_entities(tokens, word_ids, ner_preds)

        return {
            "text": review_text,
            "sentiment": {
                "label": sent_label,
                "confidence": round(float(sent_probs.max()), 3)
            },
            "entities": entities,
            "actionable_insight": self._generate_insight(sent_label, entities)
        }

    def _aggregate_entities(self, tokens, word_ids, predictions):
        id2label = self.ner_model.config.id2label
        entities, current = [], None
        for idx, wid in enumerate(word_ids):
            if wid is None or (idx > 0 and word_ids[idx-1] == wid):
                continue
            tag = id2label[predictions[idx]]
            if tag.startswith("B-"):
                if current: entities.append(current)
                current = {"text": tokens[wid], "type": tag[2:]}
            elif tag.startswith("I-") and current and current["type"] == tag[2:]:
                current["text"] += " " + tokens[wid]
            else:
                if current: entities.append(current); current = None
        if current: entities.append(current)
        return entities

    def _generate_insight(self, sentiment, entities):
        if sentiment == "NEGATIVE" and entities:
            products = [e["text"] for e in entities if e["type"] in ("ORG", "PRODUCT")]
            if products:
                return f"⚠️ تنبيه: شكوى سلبية متعلقة بـ {', '.join(products)} — يُنصح بالمتابعة الفورية"
        return "✅ لا حاجة لإجراء عاجل"


# --- التشغيل الفعلي ---
analyzer = ReviewAnalyzer("./sentiment_arabert", "./ner_arabert")
report = analyzer.analyze("خدمة العملاء في شركة جوجل كانت سيئة جدًا")
print(report)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-51-chapter-51.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — تحليل المشاعر واستخراج المعلومات (Sentiment Analysis & NER)
Large Language Model Engineering — Sentiment

Book slug : llm-engineering-69-chapters
Chapter   : ch-51-chapter-51 (lesson 51)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_51.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 51: الفصل 51: تحليل المشاعر واستخراج المعلومات (Sentiment Analysis & NER)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_nlp() -> None:
    text = 'Aisha said the model helped her team in Paris.'
    positive = {'helped', 'excellent', 'useful'}
    negative = {'failed', 'poor', 'harmful'}
    words = re.findall("\\b[\\w']+\\b", text.lower())
    sentiment = sum((word in positive for word in words)) - sum((word in negative for word in words))
    entities = re.findall('\\b[A-Z][a-z]+\\b', text)
    print('lexicon sentiment score:', sentiment)
    print('candidate proper-name entities:', entities)
    print('review ambiguous entities before production use.')

if __name__ == "__main__":
    print("Chapter 51 lab: Combine a sentiment lexicon with entity extraction")
    run_nlp()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-10-applications-agents/chapter-51.md
# ────────────────────────────────────────────────────────────────────────
# requirements.txt
# torch>=2.4.0
# transformers>=4.44.0
# datasets>=2.20.0
# seqeval>=1.2.2
# accelerate>=0.33.0

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    DataCollatorForTokenClassification
)
from datasets import load_dataset, Dataset
import numpy as np
from seqeval.metrics import classification_report, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ الجهاز المستخدم: {device}")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-10-applications-agents/chapter-51.md
# ────────────────────────────────────────────────────────────────────────
MODEL_NAME = "aubmindlab/bert-base-arabertv2"

# --- الخطوة 1: تحميل الـ Tokenizer والبيانات ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# بيانات تجريبية (في الواقع تُحمَّل من ملف/قاعدة بيانات حقيقية)
raw_data = {
    "text": [
        "هذا المنتج رائع جدًا وأنصح الجميع بشرائه",
        "جودة سيئة للغاية ولا أنصح بهذا المنتج إطلاقًا",
        "المنتج عادي، لا بأس به لكن ليس استثنائيًا",
        "تجربة ممتازة والتوصيل كان سريعًا جدًا",
        "خدمة العملاء سيئة والمنتج وصل تالفًا",
    ],
    "label": [2, 0, 1, 2, 0]  # 0: سلبي, 1: محايد, 2: إيجابي
}

dataset = Dataset.from_dict(raw_data)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# --- الخطوة 2: دالة الترميز (Tokenization) ---
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,   # الـ padding سيُنفَّذ ديناميكيًا عبر Collator
        max_length=128
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# --- الخطوة 3: تحميل النموذج مع رأس تصنيف ---
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label={0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"},
    label2id={"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
).to(device)

# --- الخطوة 4: Data Collator للـ Padding الديناميكي ---
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# --- الخطوة 5: دالة حساب المقاييس ---
def compute_metrics_sentiment(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

# --- الخطوة 6: إعدادات التدريب ---
training_args = TrainingArguments(
    output_dir="./sentiment_arabert",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",   # عطّل WandB إن لم يكن مُعدًّا
    fp16=torch.cuda.is_available(),  # Mixed Precision إن توفّر GPU
)

# --- الخطوة 7: التدريب ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics_sentiment,
)

trainer.train()

# --- الخطوة 8: الاستدلال (Inference) على نص جديد ---
def predict_sentiment(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred_id = int(np.argmax(probs))
    return {
        "label": model.config.id2label[pred_id],
        "confidence": float(probs[pred_id]),
        "all_probs": {model.config.id2label[i]: float(p) for i, p in enumerate(probs)}
    }

result = predict_sentiment("التطبيق بطيء جدًا ويتعطل باستمرار")
print(result)
# المخرج المتوقع: {'label': 'NEGATIVE', 'confidence': 0.91, ...}

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-10-applications-agents/chapter-51.md
# ────────────────────────────────────────────────────────────────────────
# --- الخطوة 1: تجهيز بيانات NER بمخطط BIO ---
# كل جملة = قائمة كلمات، وكل جملة تقابلها قائمة بنفس الطول من الوسوم

ner_label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
label2id_ner = {label: idx for idx, label in enumerate(ner_label_list)}
id2label_ner = {idx: label for idx, label in enumerate(ner_label_list)}

raw_ner_data = {
    "tokens": [
        ["ذهب", "محمد", "أحمد", "إلى", "القاهرة"],
        ["تعمل", "سارة", "في", "شركة", "جوجل"],
        ["زار", "الرئيس", "باريس", "الأسبوع", "الماضي"],
    ],
    "ner_tags": [
        [0, 1, 2, 0, 5],    # O B-PER I-PER O B-LOC
        [0, 1, 0, 3, 4],    # O B-PER O B-ORG I-ORG
        [0, 0, 5, 0, 0],    # O O B-LOC O O
    ]
}

ner_dataset = Dataset.from_dict(raw_ner_data)
ner_dataset = ner_dataset.train_test_split(test_size=0.33, seed=42)

tokenizer_ner = AutoTokenizer.from_pretrained(MODEL_NAME)

# --- الخطوة 2: محاذاة الوسوم مع الـ Subword Tokens (الجزء الأكثر دقة في NER) ---
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer_ner(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,  # مهم جدًا: الإدخال كلمات مُقسَّمة مسبقًا
        padding=False,
        max_length=128
    )

    all_labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # يربط كل Subword بكلمته الأصلية
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                # رموز خاصة مثل [CLS], [SEP], [PAD]
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # أول Subword من الكلمة → يأخذ الوسم الحقيقي
                label_ids.append(label[word_idx])
            else:
                # باقي الـ Subwords لنفس الكلمة → نتجاهلها في حساب الخسارة
                label_ids.append(-100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

tokenized_ner_datasets = ner_dataset.map(tokenize_and_align_labels, batched=True)

# --- الخطوة 3: تحميل النموذج لمهمة Token Classification ---
ner_model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(ner_label_list),
    id2label=id2label_ner,
    label2id=label2id_ner
).to(device)

# --- الخطوة 4: Collator خاص بـ Token Classification ---
ner_data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer_ner)

# --- الخطوة 5: دالة المقاييس باستخدام seqeval (المعيار الصناعي لـ NER) ---
def compute_metrics_ner(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # إزالة التنبؤات الخاصة بـ -100 (الـ Subwords الفرعية والـ Padding)
    true_predictions = [
        [id2label_ner[p] for p, l in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label_ner[l] for p, l in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]

    f1 = f1_score(true_labels, true_predictions)
    report = classification_report(true_labels, true_predictions, output_dict=True)

    return {
        "f1": f1,
        "precision": report["micro avg"]["precision"],
        "recall": report["micro avg"]["recall"],
    }

# --- الخطوة 6: إعدادات وتشغيل التدريب ---
ner_training_args = TrainingArguments(
    output_dir="./ner_arabert",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=5,
    report_to="none",
    fp16=torch.cuda.is_available(),
)

ner_trainer = Trainer(
    model=ner_model,
    args=ner_training_args,
    train_dataset=tokenized_ner_datasets["train"],
    eval_dataset=tokenized_ner_datasets["test"],
    data_collator=ner_data_collator,
    compute_metrics=compute_metrics_ner,
)

ner_trainer.train()

# --- الخطوة 7: Inference كامل مع إعادة تجميع الكيانات متعددة الكلمات ---
def extract_entities(text: str) -> list[dict]:
    tokens = text.split()  # تقسيم بسيط بالمسافات (في الإنتاج: استخدم أداة تجزئة أدق)
    inputs = tokenizer_ner(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True
    ).to(device)

    ner_model.eval()
    with torch.no_grad():
        logits = ner_model(**inputs).logits
    predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()

    word_ids = inputs.word_ids(batch_index=0)
    entities = []
    current_entity = None

    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        # نأخذ فقط أول Subword من كل كلمة (لتجنب التكرار)
        if idx > 0 and word_ids[idx-1] == word_idx:
            continue

        tag = id2label_ner[predictions[idx]]
        word = tokens[word_idx]

        if tag.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {"text": word, "type": tag[2:], "start_word": word_idx}
        elif tag.startswith("I-") and current_entity and current_entity["type"] == tag[2:]:
            current_entity["text"] += " " + word
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    return entities

result = extract_entities("التقى أحمد علي بوزير الخارجية في دبي")
print(result)
# المخرج المتوقع: [{'text': 'أحمد علي', 'type': 'PER', ...}, {'text': 'دبي', 'type': 'LOC', ...}]

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-10-applications-agents/chapter-51.md
# ────────────────────────────────────────────────────────────────────────
class ReviewAnalyzer:
    """نظام متكامل يجمع تحليل المشاعر واستخراج الكيانات من مراجعات المنتجات"""

    def __init__(self, sentiment_model_path: str, ner_model_path: str):
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            sentiment_model_path
        ).to(device).eval()

        self.ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_path)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(
            ner_model_path
        ).to(device).eval()

    @torch.no_grad()
    def analyze(self, review_text: str) -> dict:
        # 1. تحليل المشاعر
        sent_inputs = self.sentiment_tokenizer(
            review_text, return_tensors="pt", truncation=True
        ).to(device)
        sent_logits = self.sentiment_model(**sent_inputs).logits
        sent_probs = torch.softmax(sent_logits, dim=-1)[0]
        sent_label = self.sentiment_model.config.id2label[int(sent_probs.argmax())]

        # 2. استخراج الكيانات
        tokens = review_text.split()
        ner_inputs = self.ner_tokenizer(
            tokens, is_split_into_words=True, return_tensors="pt", truncation=True
        ).to(device)
        ner_logits = self.ner_model(**ner_inputs).logits
        ner_preds = torch.argmax(ner_logits, dim=-1)[0].cpu().numpy()

        word_ids = ner_inputs.word_ids(batch_index=0)
        entities = self._aggregate_entities(tokens, word_ids, ner_preds)

        return {
            "text": review_text,
            "sentiment": {
                "label": sent_label,
                "confidence": round(float(sent_probs.max()), 3)
            },
            "entities": entities,
            "actionable_insight": self._generate_insight(sent_label, entities)
        }

    def _aggregate_entities(self, tokens, word_ids, predictions):
        id2label = self.ner_model.config.id2label
        entities, current = [], None
        for idx, wid in enumerate(word_ids):
            if wid is None or (idx > 0 and word_ids[idx-1] == wid):
                continue
            tag = id2label[predictions[idx]]
            if tag.startswith("B-"):
                if current: entities.append(current)
                current = {"text": tokens[wid], "type": tag[2:]}
            elif tag.startswith("I-") and current and current["type"] == tag[2:]:
                current["text"] += " " + tokens[wid]
            else:
                if current: entities.append(current); current = None
        if current: entities.append(current)
        return entities

    def _generate_insight(self, sentiment, entities):
        if sentiment == "NEGATIVE" and entities:
            products = [e["text"] for e in entities if e["type"] in ("ORG", "PRODUCT")]
            if products:
                return f"⚠️ تنبيه: شكوى سلبية متعلقة بـ {', '.join(products)} — يُنصح بالمتابعة الفورية"
        return "✅ لا حاجة لإجراء عاجل"


# --- التشغيل الفعلي ---
analyzer = ReviewAnalyzer("./sentiment_arabert", "./ner_arabert")
report = analyzer.analyze("خدمة العملاء في شركة جوجل كانت سيئة جدًا")
print(report)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-51-chapter-51.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 51: الفصل 51: تحليل المشاعر واستخراج المعلومات (Sentiment Analysis & NER)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_nlp() -> None:
    text = 'Aisha said the model helped her team in Paris.'
    positive = {'helped', 'excellent', 'useful'}
    negative = {'failed', 'poor', 'harmful'}
    words = re.findall("\\b[\\w']+\\b", text.lower())
    sentiment = sum((word in positive for word in words)) - sum((word in negative for word in words))
    entities = re.findall('\\b[A-Z][a-z]+\\b', text)
    print('lexicon sentiment score:', sentiment)
    print('candidate proper-name entities:', entities)
    print('review ambiguous entities before production use.')

if __name__ == "__main__":
    print("Chapter 51 lab: Combine a sentiment lexicon with entity extraction")
    run_nlp()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-51-chapter-51.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# requirements.txt
# torch>=2.4.0
# transformers>=4.44.0
# datasets>=2.20.0
# seqeval>=1.2.2
# accelerate>=0.33.0

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    DataCollatorForTokenClassification
)
from datasets import load_dataset, Dataset
import numpy as np
from seqeval.metrics import classification_report, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ الجهاز المستخدم: {device}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-51-chapter-51.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
MODEL_NAME = "aubmindlab/bert-base-arabertv2"

# --- الخطوة 1: تحميل الـ Tokenizer والبيانات ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# بيانات تجريبية (في الواقع تُحمَّل من ملف/قاعدة بيانات حقيقية)
raw_data = {
    "text": [
        "هذا المنتج رائع جدًا وأنصح الجميع بشرائه",
        "جودة سيئة للغاية ولا أنصح بهذا المنتج إطلاقًا",
        "المنتج عادي، لا بأس به لكن ليس استثنائيًا",
        "تجربة ممتازة والتوصيل كان سريعًا جدًا",
        "خدمة العملاء سيئة والمنتج وصل تالفًا",
    ],
    "label": [2, 0, 1, 2, 0]  # 0: سلبي, 1: محايد, 2: إيجابي
}

dataset = Dataset.from_dict(raw_data)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# --- الخطوة 2: دالة الترميز (Tokenization) ---
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,   # الـ padding سيُنفَّذ ديناميكيًا عبر Collator
        max_length=128
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# --- الخطوة 3: تحميل النموذج مع رأس تصنيف ---
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label={0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"},
    label2id={"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
).to(device)

# --- الخطوة 4: Data Collator للـ Padding الديناميكي ---
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# --- الخطوة 5: دالة حساب المقاييس ---
def compute_metrics_sentiment(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

# --- الخطوة 6: إعدادات التدريب ---
training_args = TrainingArguments(
    output_dir="./sentiment_arabert",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",   # عطّل WandB إن لم يكن مُعدًّا
    fp16=torch.cuda.is_available(),  # Mixed Precision إن توفّر GPU
)

# --- الخطوة 7: التدريب ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics_sentiment,
)

trainer.train()

# --- الخطوة 8: الاستدلال (Inference) على نص جديد ---
def predict_sentiment(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred_id = int(np.argmax(probs))
    return {
        "label": model.config.id2label[pred_id],
        "confidence": float(probs[pred_id]),
        "all_probs": {model.config.id2label[i]: float(p) for i, p in enumerate(probs)}
    }

result = predict_sentiment("التطبيق بطيء جدًا ويتعطل باستمرار")
print(result)
# المخرج المتوقع: {'label': 'NEGATIVE', 'confidence': 0.91, ...}

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-51-chapter-51.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# --- الخطوة 1: تجهيز بيانات NER بمخطط BIO ---
# كل جملة = قائمة كلمات، وكل جملة تقابلها قائمة بنفس الطول من الوسوم

ner_label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
label2id_ner = {label: idx for idx, label in enumerate(ner_label_list)}
id2label_ner = {idx: label for idx, label in enumerate(ner_label_list)}

raw_ner_data = {
    "tokens": [
        ["ذهب", "محمد", "أحمد", "إلى", "القاهرة"],
        ["تعمل", "سارة", "في", "شركة", "جوجل"],
        ["زار", "الرئيس", "باريس", "الأسبوع", "الماضي"],
    ],
    "ner_tags": [
        [0, 1, 2, 0, 5],    # O B-PER I-PER O B-LOC
        [0, 1, 0, 3, 4],    # O B-PER O B-ORG I-ORG
        [0, 0, 5, 0, 0],    # O O B-LOC O O
    ]
}

ner_dataset = Dataset.from_dict(raw_ner_data)
ner_dataset = ner_dataset.train_test_split(test_size=0.33, seed=42)

tokenizer_ner = AutoTokenizer.from_pretrained(MODEL_NAME)

# --- الخطوة 2: محاذاة الوسوم مع الـ Subword Tokens (الجزء الأكثر دقة في NER) ---
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer_ner(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,  # مهم جدًا: الإدخال كلمات مُقسَّمة مسبقًا
        padding=False,
        max_length=128
    )

    all_labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # يربط كل Subword بكلمته الأصلية
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                # رموز خاصة مثل [CLS], [SEP], [PAD]
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # أول Subword من الكلمة → يأخذ الوسم الحقيقي
                label_ids.append(label[word_idx])
            else:
                # باقي الـ Subwords لنفس الكلمة → نتجاهلها في حساب الخسارة
                label_ids.append(-100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

tokenized_ner_datasets = ner_dataset.map(tokenize_and_align_labels, batched=True)

# --- الخطوة 3: تحميل النموذج لمهمة Token Classification ---
ner_model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(ner_label_list),
    id2label=id2label_ner,
    label2id=label2id_ner
).to(device)

# --- الخطوة 4: Collator خاص بـ Token Classification ---
ner_data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer_ner)

# --- الخطوة 5: دالة المقاييس باستخدام seqeval (المعيار الصناعي لـ NER) ---
def compute_metrics_ner(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # إزالة التنبؤات الخاصة بـ -100 (الـ Subwords الفرعية والـ Padding)
    true_predictions = [
        [id2label_ner[p] for p, l in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label_ner[l] for p, l in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]

    f1 = f1_score(true_labels, true_predictions)
    report = classification_report(true_labels, true_predictions, output_dict=True)

    return {
        "f1": f1,
        "precision": report["micro avg"]["precision"],
        "recall": report["micro avg"]["recall"],
    }

# --- الخطوة 6: إعدادات وتشغيل التدريب ---
ner_training_args = TrainingArguments(
    output_dir="./ner_arabert",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=5,
    report_to="none",
    fp16=torch.cuda.is_available(),
)

ner_trainer = Trainer(
    model=ner_model,
    args=ner_training_args,
    train_dataset=tokenized_ner_datasets["train"],
    eval_dataset=tokenized_ner_datasets["test"],
    data_collator=ner_data_collator,
    compute_metrics=compute_metrics_ner,
)

ner_trainer.train()

# --- الخطوة 7: Inference كامل مع إعادة تجميع الكيانات متعددة الكلمات ---
def extract_entities(text: str) -> list[dict]:
    tokens = text.split()  # تقسيم بسيط بالمسافات (في الإنتاج: استخدم أداة تجزئة أدق)
    inputs = tokenizer_ner(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True
    ).to(device)

    ner_model.eval()
    with torch.no_grad():
        logits = ner_model(**inputs).logits
    predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()

    word_ids = inputs.word_ids(batch_index=0)
    entities = []
    current_entity = None

    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        # نأخذ فقط أول Subword من كل كلمة (لتجنب التكرار)
        if idx > 0 and word_ids[idx-1] == word_idx:
            continue

        tag = id2label_ner[predictions[idx]]
        word = tokens[word_idx]

        if tag.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {"text": word, "type": tag[2:], "start_word": word_idx}
        elif tag.startswith("I-") and current_entity and current_entity["type"] == tag[2:]:
            current_entity["text"] += " " + word
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    return entities

result = extract_entities("التقى أحمد علي بوزير الخارجية في دبي")
print(result)
# المخرج المتوقع: [{'text': 'أحمد علي', 'type': 'PER', ...}, {'text': 'دبي', 'type': 'LOC', ...}]

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-51-chapter-51.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
class ReviewAnalyzer:
    """نظام متكامل يجمع تحليل المشاعر واستخراج الكيانات من مراجعات المنتجات"""

    def __init__(self, sentiment_model_path: str, ner_model_path: str):
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            sentiment_model_path
        ).to(device).eval()

        self.ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_path)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(
            ner_model_path
        ).to(device).eval()

    @torch.no_grad()
    def analyze(self, review_text: str) -> dict:
        # 1. تحليل المشاعر
        sent_inputs = self.sentiment_tokenizer(
            review_text, return_tensors="pt", truncation=True
        ).to(device)
        sent_logits = self.sentiment_model(**sent_inputs).logits
        sent_probs = torch.softmax(sent_logits, dim=-1)[0]
        sent_label = self.sentiment_model.config.id2label[int(sent_probs.argmax())]

        # 2. استخراج الكيانات
        tokens = review_text.split()
        ner_inputs = self.ner_tokenizer(
            tokens, is_split_into_words=True, return_tensors="pt", truncation=True
        ).to(device)
        ner_logits = self.ner_model(**ner_inputs).logits
        ner_preds = torch.argmax(ner_logits, dim=-1)[0].cpu().numpy()

        word_ids = ner_inputs.word_ids(batch_index=0)
        entities = self._aggregate_entities(tokens, word_ids, ner_preds)

        return {
            "text": review_text,
            "sentiment": {
                "label": sent_label,
                "confidence": round(float(sent_probs.max()), 3)
            },
            "entities": entities,
            "actionable_insight": self._generate_insight(sent_label, entities)
        }

    def _aggregate_entities(self, tokens, word_ids, predictions):
        id2label = self.ner_model.config.id2label
        entities, current = [], None
        for idx, wid in enumerate(word_ids):
            if wid is None or (idx > 0 and word_ids[idx-1] == wid):
                continue
            tag = id2label[predictions[idx]]
            if tag.startswith("B-"):
                if current: entities.append(current)
                current = {"text": tokens[wid], "type": tag[2:]}
            elif tag.startswith("I-") and current and current["type"] == tag[2:]:
                current["text"] += " " + tokens[wid]
            else:
                if current: entities.append(current); current = None
        if current: entities.append(current)
        return entities

    def _generate_insight(self, sentiment, entities):
        if sentiment == "NEGATIVE" and entities:
            products = [e["text"] for e in entities if e["type"] in ("ORG", "PRODUCT")]
            if products:
                return f"⚠️ تنبيه: شكوى سلبية متعلقة بـ {', '.join(products)} — يُنصح بالمتابعة الفورية"
        return "✅ لا حاجة لإجراء عاجل"


# --- التشغيل الفعلي ---
analyzer = ReviewAnalyzer("./sentiment_arabert", "./ner_arabert")
report = analyzer.analyze("خدمة العملاء في شركة جوجل كانت سيئة جدًا")
print(report)
