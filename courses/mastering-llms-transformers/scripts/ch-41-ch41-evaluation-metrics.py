#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — أسس القياس الكمّي
Mastering Transformers and Large Language Models — Evaluation Metrics

Book slug : mastering-llms-transformers
Chapter   : ch-41-ch41-evaluation-metrics (lesson 41)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-41-ch41-evaluation-metrics.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — أسس القياس الكمّي
Mastering Transformers and Large Language Models — Evaluation Metrics

Book slug : mastering-llms-transformers
Chapter   : ch-41-ch41-evaluation-metrics (lesson 41)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-41-ch41-evaluation-metrics.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — أسس القياس الكمّي
Mastering Transformers and Large Language Models — Evaluation Metrics

Book slug : mastering-llms-transformers
Chapter   : ch-41-ch41-evaluation-metrics (lesson 41)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-09-evaluation/ch41-evaluation-metrics.ipynb#cell-1
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
    ("torch", "torch"),
    ("transformers", "transformers")
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
# [002] content/notebooks/part-09-evaluation/ch41-evaluation-metrics.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

def compute_bits_per_byte(model, tokenizer, text: str) -> float:
    """
    مقياس عادل للمقارنة عبر tokenizers مختلفة:
    نحسب log-likelihood الكلي ونقسمه على عدد البايتات الخام للنص
    (وليس عدد الرموز، الذي يختلف حسب tokenizer)
    """
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        # outputs.loss هو متوسط NLL لكل رمز (log base e)
        total_nll_nats = outputs.loss.item() * (input_ids.shape[1] - 1)

    num_bytes = len(text.encode("utf-8"))
    bits_per_byte = (total_nll_nats / 0.6931) / num_bytes   # تحويل nats -> bits (log2)
    return bits_per_byte

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-09-evaluation/ch41-evaluation-metrics.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from collections import Counter
import math

def compute_bleu(candidate: list, reference: list, max_n=4) -> float:
    """تطبيق مبسّط وتعليمي لـ BLEU من الصفر"""
    def get_ngrams(tokens, n):
        return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])

    precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = get_ngrams(candidate, n)
        ref_ngrams = get_ngrams(reference, n)
        overlap = sum(min(count, ref_ngrams[ng]) for ng, count in cand_ngrams.items())
        total = max(sum(cand_ngrams.values()), 1)
        precisions.append(overlap / total if total > 0 else 0)

    if min(precisions) == 0:
        return 0.0

    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    bp = 1.0 if len(candidate) > len(reference) else math.exp(1 - len(reference) / max(len(candidate), 1))
    return bp * geo_mean

def compute_rouge_l(candidate: list, reference: list) -> dict:
    """ROUGE-L: يعتمد على أطول تسلسل فرعي مشترك (LCS) — أكثر ملاءمة للتلخيص من BLEU"""
    def lcs_length(a, b):
        dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1]

    lcs = lcs_length(candidate, reference)
    precision = lcs / len(candidate) if candidate else 0
    recall = lcs / len(reference) if reference else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1}

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-09-evaluation/ch41-evaluation-metrics.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

def compute_bertscore(candidate: str, reference: str, model_name="bert-base-multilingual-cased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    def get_embeddings(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return F.normalize(outputs.last_hidden_state[0], dim=-1)  # [seq_len, hidden]

    cand_embeds = get_embeddings(candidate)
    ref_embeds = get_embeddings(reference)

    similarity_matrix = cand_embeds @ ref_embeds.T   # [cand_len, ref_len]
    recall = similarity_matrix.max(dim=0).values.mean().item()
    precision = similarity_matrix.max(dim=1).values.mean().item()
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1}

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-09-evaluation/ch41-evaluation-metrics.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-41-ch41-evaluation-metrics.ipynb#cell-1
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
    ("torch", "torch"),
    ("transformers", "transformers")
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-41-ch41-evaluation-metrics.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

def compute_bits_per_byte(model, tokenizer, text: str) -> float:
    """
    مقياس عادل للمقارنة عبر tokenizers مختلفة:
    نحسب log-likelihood الكلي ونقسمه على عدد البايتات الخام للنص
    (وليس عدد الرموز، الذي يختلف حسب tokenizer)
    """
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        # outputs.loss هو متوسط NLL لكل رمز (log base e)
        total_nll_nats = outputs.loss.item() * (input_ids.shape[1] - 1)

    num_bytes = len(text.encode("utf-8"))
    bits_per_byte = (total_nll_nats / 0.6931) / num_bytes   # تحويل nats -> bits (log2)
    return bits_per_byte

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-41-ch41-evaluation-metrics.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from collections import Counter
import math

def compute_bleu(candidate: list, reference: list, max_n=4) -> float:
    """تطبيق مبسّط وتعليمي لـ BLEU من الصفر"""
    def get_ngrams(tokens, n):
        return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])

    precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = get_ngrams(candidate, n)
        ref_ngrams = get_ngrams(reference, n)
        overlap = sum(min(count, ref_ngrams[ng]) for ng, count in cand_ngrams.items())
        total = max(sum(cand_ngrams.values()), 1)
        precisions.append(overlap / total if total > 0 else 0)

    if min(precisions) == 0:
        return 0.0

    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    bp = 1.0 if len(candidate) > len(reference) else math.exp(1 - len(reference) / max(len(candidate), 1))
    return bp * geo_mean

def compute_rouge_l(candidate: list, reference: list) -> dict:
    """ROUGE-L: يعتمد على أطول تسلسل فرعي مشترك (LCS) — أكثر ملاءمة للتلخيص من BLEU"""
    def lcs_length(a, b):
        dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1]

    lcs = lcs_length(candidate, reference)
    precision = lcs / len(candidate) if candidate else 0
    recall = lcs / len(reference) if reference else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1}

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-41-ch41-evaluation-metrics.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

def compute_bertscore(candidate: str, reference: str, model_name="bert-base-multilingual-cased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    def get_embeddings(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return F.normalize(outputs.last_hidden_state[0], dim=-1)  # [seq_len, hidden]

    cand_embeds = get_embeddings(candidate)
    ref_embeds = get_embeddings(reference)

    similarity_matrix = cand_embeds @ ref_embeds.T   # [cand_len, ref_len]
    recall = similarity_matrix.max(dim=0).values.mean().item()
    precision = similarity_matrix.max(dim=1).values.mean().item()
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1}

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-41-ch41-evaluation-metrics.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-41-ch41-evaluation-metrics.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — أسس القياس الكمّي
Mastering Transformers and Large Language Models — Evaluation Metrics

Book slug : mastering-llms-transformers
Chapter   : ch-41-ch41-evaluation-metrics (lesson 41)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-09-evaluation/ch41-evaluation-metrics.ipynb#cell-1
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
    ("torch", "torch"),
    ("transformers", "transformers")
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
# [002] content/notebooks/part-09-evaluation/ch41-evaluation-metrics.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

def compute_bits_per_byte(model, tokenizer, text: str) -> float:
    """
    مقياس عادل للمقارنة عبر tokenizers مختلفة:
    نحسب log-likelihood الكلي ونقسمه على عدد البايتات الخام للنص
    (وليس عدد الرموز، الذي يختلف حسب tokenizer)
    """
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        # outputs.loss هو متوسط NLL لكل رمز (log base e)
        total_nll_nats = outputs.loss.item() * (input_ids.shape[1] - 1)

    num_bytes = len(text.encode("utf-8"))
    bits_per_byte = (total_nll_nats / 0.6931) / num_bytes   # تحويل nats -> bits (log2)
    return bits_per_byte

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-09-evaluation/ch41-evaluation-metrics.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from collections import Counter
import math

def compute_bleu(candidate: list, reference: list, max_n=4) -> float:
    """تطبيق مبسّط وتعليمي لـ BLEU من الصفر"""
    def get_ngrams(tokens, n):
        return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])

    precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = get_ngrams(candidate, n)
        ref_ngrams = get_ngrams(reference, n)
        overlap = sum(min(count, ref_ngrams[ng]) for ng, count in cand_ngrams.items())
        total = max(sum(cand_ngrams.values()), 1)
        precisions.append(overlap / total if total > 0 else 0)

    if min(precisions) == 0:
        return 0.0

    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    bp = 1.0 if len(candidate) > len(reference) else math.exp(1 - len(reference) / max(len(candidate), 1))
    return bp * geo_mean

def compute_rouge_l(candidate: list, reference: list) -> dict:
    """ROUGE-L: يعتمد على أطول تسلسل فرعي مشترك (LCS) — أكثر ملاءمة للتلخيص من BLEU"""
    def lcs_length(a, b):
        dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1]

    lcs = lcs_length(candidate, reference)
    precision = lcs / len(candidate) if candidate else 0
    recall = lcs / len(reference) if reference else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1}

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-09-evaluation/ch41-evaluation-metrics.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

def compute_bertscore(candidate: str, reference: str, model_name="bert-base-multilingual-cased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    def get_embeddings(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return F.normalize(outputs.last_hidden_state[0], dim=-1)  # [seq_len, hidden]

    cand_embeds = get_embeddings(candidate)
    ref_embeds = get_embeddings(reference)

    similarity_matrix = cand_embeds @ ref_embeds.T   # [cand_len, ref_len]
    recall = similarity_matrix.max(dim=0).values.mean().item()
    precision = similarity_matrix.max(dim=1).values.mean().item()
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1}

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-09-evaluation/ch41-evaluation-metrics.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-41-ch41-evaluation-metrics.ipynb#cell-2
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
    ("torch", "torch"),
    ("transformers", "transformers")
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-41-ch41-evaluation-metrics.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

def compute_bits_per_byte(model, tokenizer, text: str) -> float:
    """
    مقياس عادل للمقارنة عبر tokenizers مختلفة:
    نحسب log-likelihood الكلي ونقسمه على عدد البايتات الخام للنص
    (وليس عدد الرموز، الذي يختلف حسب tokenizer)
    """
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        # outputs.loss هو متوسط NLL لكل رمز (log base e)
        total_nll_nats = outputs.loss.item() * (input_ids.shape[1] - 1)

    num_bytes = len(text.encode("utf-8"))
    bits_per_byte = (total_nll_nats / 0.6931) / num_bytes   # تحويل nats -> bits (log2)
    return bits_per_byte

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-41-ch41-evaluation-metrics.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from collections import Counter
import math

def compute_bleu(candidate: list, reference: list, max_n=4) -> float:
    """تطبيق مبسّط وتعليمي لـ BLEU من الصفر"""
    def get_ngrams(tokens, n):
        return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])

    precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = get_ngrams(candidate, n)
        ref_ngrams = get_ngrams(reference, n)
        overlap = sum(min(count, ref_ngrams[ng]) for ng, count in cand_ngrams.items())
        total = max(sum(cand_ngrams.values()), 1)
        precisions.append(overlap / total if total > 0 else 0)

    if min(precisions) == 0:
        return 0.0

    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    bp = 1.0 if len(candidate) > len(reference) else math.exp(1 - len(reference) / max(len(candidate), 1))
    return bp * geo_mean

def compute_rouge_l(candidate: list, reference: list) -> dict:
    """ROUGE-L: يعتمد على أطول تسلسل فرعي مشترك (LCS) — أكثر ملاءمة للتلخيص من BLEU"""
    def lcs_length(a, b):
        dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1]

    lcs = lcs_length(candidate, reference)
    precision = lcs / len(candidate) if candidate else 0
    recall = lcs / len(reference) if reference else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1}

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-41-ch41-evaluation-metrics.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

def compute_bertscore(candidate: str, reference: str, model_name="bert-base-multilingual-cased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    def get_embeddings(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return F.normalize(outputs.last_hidden_state[0], dim=-1)  # [seq_len, hidden]

    cand_embeds = get_embeddings(candidate)
    ref_embeds = get_embeddings(reference)

    similarity_matrix = cand_embeds @ ref_embeds.T   # [cand_len, ref_len]
    recall = similarity_matrix.max(dim=0).values.mean().item()
    precision = similarity_matrix.max(dim=1).values.mean().item()
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1}

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-41-ch41-evaluation-metrics.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
