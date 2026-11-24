#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — قياس الأداء: مقاييس التصنيف والحيرة
Mastering Transformers and Large Language Models — Metrics

Book slug : mastering-llms-transformers
Chapter   : ch-05-ch05-metrics (lesson 5)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-05-ch05-metrics.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — قياس الأداء: مقاييس التصنيف والحيرة
Mastering Transformers and Large Language Models — Metrics

Book slug : mastering-llms-transformers
Chapter   : ch-05-ch05-metrics (lesson 5)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-05-ch05-metrics.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — قياس الأداء: مقاييس التصنيف والحيرة
Mastering Transformers and Large Language Models — Metrics

Book slug : mastering-llms-transformers
Chapter   : ch-05-ch05-metrics (lesson 5)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-01-foundations/ch05-metrics.ipynb#cell-1
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
    ("numpy", "numpy")
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
# [002] content/notebooks/part-01-foundations/ch05-metrics.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """حساب عناصر مصفوفة الالتباس الأربعة للتصنيف الثنائي."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}

def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Precision = TP / (TP + FP)
    "من بين كل ما تنبأ به النموذج كإيجابي، كم نسبة كانت صحيحة فعلًا؟"
    مهمة عندما تكون تكلفة False Positive عالية (مثل: تصنيف بريد مهم كـ spam).
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    denominator = cm["TP"] + cm["FP"]
    return cm["TP"] / denominator if denominator > 0 else 0.0

def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Recall (Sensitivity) = TP / (TP + FN)
    "من بين كل الحالات الإيجابية الحقيقية، كم نسبة اكتشفها النموذج فعلًا؟"
    مهمة عندما تكون تكلفة False Negative عالية (مثل: تشخيص طبي خطير، كشف احتيال).
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    denominator = cm["TP"] + cm["FN"]
    return cm["TP"] / denominator if denominator > 0 else 0.0

def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    F1-Score = المتوسط التوافقي (Harmonic Mean) بين Precision وRecall.
    نستخدم المتوسط التوافقي (وليس الحسابي) لأنه يُعاقب بشدة الفارق الكبير
    بينهما — نموذج بـ Precision=1.0 وRecall=0.01 سيحصل على F1 منخفض جدًا
    (0.0198) رغم أن المتوسط الحسابي سيكون 0.505 (مُضلِّل تمامًا).

        F1 = 2 × (Precision × Recall) / (Precision + Recall)
    """
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy = (TP+TN) / الإجمالي — مُضلِّلة في البيانات غير المتوازنة."""
    return np.mean(y_true == y_pred)

# === توضيح عملي لمشكلة البيانات غير المتوازنة ===
def demonstrate_imbalanced_data_problem():
    # محاكاة: كشف احتيال، 99% من المعاملات شرعية (0) و1% احتيال (1)
    np.random.seed(42)
    n_samples = 1000
    y_true = np.array([0] * 990 + [1] * 10)

    # النموذج "الغبي": يُصنِّف كل شيء كـ"شرعي" (0)
    y_pred_dummy = np.zeros(n_samples)

    # النموذج الذكي نسبيًا: يكتشف 7 من أصل 10 حالات احتيال، مع بعض الأخطاء
    y_pred_smart = y_true.copy()
    y_pred_smart[990:993] = 0   # يُفوِّت 3 حالات احتيال حقيقية (FN)
    y_pred_smart[50:55] = 1     # يُصنِّف خطأً 5 معاملات شرعية كاحتيال (FP)

    print("=== النموذج الغبي (يتنبأ دائمًا بـ 'شرعي') ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred_dummy):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred_dummy):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred_dummy):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred_dummy):.4f}")

    print("\n=== النموذج الذكي (يكتشف الاحتيال فعليًا) ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred_smart):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred_smart):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred_smart):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred_smart):.4f}")

demonstrate_imbalanced_data_problem()
# === النموذج الغبي (يتنبأ دائمًا بـ 'شرعي') ===
# Accuracy: 0.9900   <- تبدو ممتازة! لكنها مُضلِّلة تمامًا
# Precision: 0.0000
# Recall: 0.0000
# F1-Score: 0.0000   <- يكشف الحقيقة فورًا: النموذج عديم الفائدة
#
# === النموذج الذكي (يكتشف الاحتيال فعليًا) ===
# Accuracy: 0.9920   <- أعلى بقليل فقط من الغبي! Accuracy لا تُميّز جيدًا هنا
# Precision: 0.5833
# Recall: 0.7000
# F1-Score: 0.6364   <- يعكس الأداء الحقيقي الأفضل بوضوح

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-01-foundations/ch05-metrics.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def calculate_perplexity(model_logits: np.ndarray, true_token_ids: np.ndarray) -> float:
    """
    Perplexity = exp(Cross-Entropy Loss المتوسطة لكل رمز)

    التفسير البديهي: "متوسط عدد الخيارات المتساوية الاحتمال التي
    يتردد بينها النموذج فعليًا عند كل خطوة تنبؤ". نموذج مثالي
    يُنتج Perplexity=1 (يقين تام)، بينما نموذج عشوائي تمامًا على
    مفردات حجمها N يُنتج Perplexity≈N.
    """
    # softmax على logits لكل موضع
    def stable_softmax(x):
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    probs = stable_softmax(model_logits)   # (seq_len, vocab_size)

    # احتمال الرمز الصحيح الفعلي عند كل موضع
    seq_len = len(true_token_ids)
    true_token_probs = probs[np.arange(seq_len), true_token_ids]
    true_token_probs = np.clip(true_token_probs, 1e-10, 1.0)

    # Cross-Entropy المتوسطة (بالـ nats، أساس e)
    avg_cross_entropy = -np.mean(np.log(true_token_probs))
    perplexity = np.exp(avg_cross_entropy)

    return perplexity

# مثال: مقارنة نموذجين على نفس الجملة
vocab_size = 1000
seq_len = 5
true_tokens = np.array([42, 17, 983, 5, 200])

# نموذج واثق وصحيح غالبًا (logits عالية للرموز الصحيحة)
np.random.seed(1)
confident_logits = np.random.randn(seq_len, vocab_size) * 0.5
for i, token in enumerate(true_tokens):
    confident_logits[i, token] += 8.0  # يُعزِّز احتمال الرمز الصحيح بشدة

# نموذج عشوائي تقريبًا
random_logits = np.random.randn(seq_len, vocab_size) * 0.5

ppl_confident = calculate_perplexity(confident_logits, true_tokens)
ppl_random = calculate_perplexity(random_logits, true_tokens)

print(f"Perplexity (نموذج واثق وصحيح): {ppl_confident:.2f}")
print(f"Perplexity (نموذج شبه عشوائي): {ppl_random:.2f}")
# Perplexity (نموذج واثق وصحيح): 1.02   <- قريب جدًا من 1 (مثالي تقريبًا)
# Perplexity (نموذج شبه عشوائي): 847.33  <- قريب من vocab_size (عشوائي تقريبًا)

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-01-foundations/ch05-metrics.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def compare_medical_classifiers():
    scenarios = {
        "نظام A (دقيق لكن يُفوِّت حالات)": {"precision": 0.95, "recall": 0.40},
        "نظام B (يكتشف أغلب الحالات لكن بإنذارات خاطئة أكثر)": {"precision": 0.60, "recall": 0.90},
    }

    for name, metrics in scenarios.items():
        p, r = metrics["precision"], metrics["recall"]
        f1 = 2 * p * r / (p + r)
        print(f"{name}:")
        print(f"  Precision={p}, Recall={r}, F1={f1:.4f}")

        # في السياق الطبي، Recall غالبًا أهم بكثير: تفويت مريض فعلي (FN)
        # قد يعني تأخير علاج حرج، بينما إنذار خاطئ (FP) يعني فقط
        # فحوصات إضافية غير ضرورية — تكلفة غير متماثلة بوضوح
        print(f"  → تفسير سريري: من كل 100 مريض فعلي، يكتشف {r*100:.0f} فقط\n")

compare_medical_classifiers()

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-01-foundations/ch05-metrics.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-05-ch05-metrics.ipynb#cell-1
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
    ("numpy", "numpy")
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-05-ch05-metrics.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """حساب عناصر مصفوفة الالتباس الأربعة للتصنيف الثنائي."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}

def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Precision = TP / (TP + FP)
    "من بين كل ما تنبأ به النموذج كإيجابي، كم نسبة كانت صحيحة فعلًا؟"
    مهمة عندما تكون تكلفة False Positive عالية (مثل: تصنيف بريد مهم كـ spam).
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    denominator = cm["TP"] + cm["FP"]
    return cm["TP"] / denominator if denominator > 0 else 0.0

def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Recall (Sensitivity) = TP / (TP + FN)
    "من بين كل الحالات الإيجابية الحقيقية، كم نسبة اكتشفها النموذج فعلًا؟"
    مهمة عندما تكون تكلفة False Negative عالية (مثل: تشخيص طبي خطير، كشف احتيال).
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    denominator = cm["TP"] + cm["FN"]
    return cm["TP"] / denominator if denominator > 0 else 0.0

def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    F1-Score = المتوسط التوافقي (Harmonic Mean) بين Precision وRecall.
    نستخدم المتوسط التوافقي (وليس الحسابي) لأنه يُعاقب بشدة الفارق الكبير
    بينهما — نموذج بـ Precision=1.0 وRecall=0.01 سيحصل على F1 منخفض جدًا
    (0.0198) رغم أن المتوسط الحسابي سيكون 0.505 (مُضلِّل تمامًا).

        F1 = 2 × (Precision × Recall) / (Precision + Recall)
    """
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy = (TP+TN) / الإجمالي — مُضلِّلة في البيانات غير المتوازنة."""
    return np.mean(y_true == y_pred)

# === توضيح عملي لمشكلة البيانات غير المتوازنة ===
def demonstrate_imbalanced_data_problem():
    # محاكاة: كشف احتيال، 99% من المعاملات شرعية (0) و1% احتيال (1)
    np.random.seed(42)
    n_samples = 1000
    y_true = np.array([0] * 990 + [1] * 10)

    # النموذج "الغبي": يُصنِّف كل شيء كـ"شرعي" (0)
    y_pred_dummy = np.zeros(n_samples)

    # النموذج الذكي نسبيًا: يكتشف 7 من أصل 10 حالات احتيال، مع بعض الأخطاء
    y_pred_smart = y_true.copy()
    y_pred_smart[990:993] = 0   # يُفوِّت 3 حالات احتيال حقيقية (FN)
    y_pred_smart[50:55] = 1     # يُصنِّف خطأً 5 معاملات شرعية كاحتيال (FP)

    print("=== النموذج الغبي (يتنبأ دائمًا بـ 'شرعي') ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred_dummy):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred_dummy):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred_dummy):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred_dummy):.4f}")

    print("\n=== النموذج الذكي (يكتشف الاحتيال فعليًا) ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred_smart):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred_smart):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred_smart):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred_smart):.4f}")

demonstrate_imbalanced_data_problem()
# === النموذج الغبي (يتنبأ دائمًا بـ 'شرعي') ===
# Accuracy: 0.9900   <- تبدو ممتازة! لكنها مُضلِّلة تمامًا
# Precision: 0.0000
# Recall: 0.0000
# F1-Score: 0.0000   <- يكشف الحقيقة فورًا: النموذج عديم الفائدة
#
# === النموذج الذكي (يكتشف الاحتيال فعليًا) ===
# Accuracy: 0.9920   <- أعلى بقليل فقط من الغبي! Accuracy لا تُميّز جيدًا هنا
# Precision: 0.5833
# Recall: 0.7000
# F1-Score: 0.6364   <- يعكس الأداء الحقيقي الأفضل بوضوح

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-05-ch05-metrics.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def calculate_perplexity(model_logits: np.ndarray, true_token_ids: np.ndarray) -> float:
    """
    Perplexity = exp(Cross-Entropy Loss المتوسطة لكل رمز)

    التفسير البديهي: "متوسط عدد الخيارات المتساوية الاحتمال التي
    يتردد بينها النموذج فعليًا عند كل خطوة تنبؤ". نموذج مثالي
    يُنتج Perplexity=1 (يقين تام)، بينما نموذج عشوائي تمامًا على
    مفردات حجمها N يُنتج Perplexity≈N.
    """
    # softmax على logits لكل موضع
    def stable_softmax(x):
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    probs = stable_softmax(model_logits)   # (seq_len, vocab_size)

    # احتمال الرمز الصحيح الفعلي عند كل موضع
    seq_len = len(true_token_ids)
    true_token_probs = probs[np.arange(seq_len), true_token_ids]
    true_token_probs = np.clip(true_token_probs, 1e-10, 1.0)

    # Cross-Entropy المتوسطة (بالـ nats، أساس e)
    avg_cross_entropy = -np.mean(np.log(true_token_probs))
    perplexity = np.exp(avg_cross_entropy)

    return perplexity

# مثال: مقارنة نموذجين على نفس الجملة
vocab_size = 1000
seq_len = 5
true_tokens = np.array([42, 17, 983, 5, 200])

# نموذج واثق وصحيح غالبًا (logits عالية للرموز الصحيحة)
np.random.seed(1)
confident_logits = np.random.randn(seq_len, vocab_size) * 0.5
for i, token in enumerate(true_tokens):
    confident_logits[i, token] += 8.0  # يُعزِّز احتمال الرمز الصحيح بشدة

# نموذج عشوائي تقريبًا
random_logits = np.random.randn(seq_len, vocab_size) * 0.5

ppl_confident = calculate_perplexity(confident_logits, true_tokens)
ppl_random = calculate_perplexity(random_logits, true_tokens)

print(f"Perplexity (نموذج واثق وصحيح): {ppl_confident:.2f}")
print(f"Perplexity (نموذج شبه عشوائي): {ppl_random:.2f}")
# Perplexity (نموذج واثق وصحيح): 1.02   <- قريب جدًا من 1 (مثالي تقريبًا)
# Perplexity (نموذج شبه عشوائي): 847.33  <- قريب من vocab_size (عشوائي تقريبًا)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-05-ch05-metrics.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def compare_medical_classifiers():
    scenarios = {
        "نظام A (دقيق لكن يُفوِّت حالات)": {"precision": 0.95, "recall": 0.40},
        "نظام B (يكتشف أغلب الحالات لكن بإنذارات خاطئة أكثر)": {"precision": 0.60, "recall": 0.90},
    }

    for name, metrics in scenarios.items():
        p, r = metrics["precision"], metrics["recall"]
        f1 = 2 * p * r / (p + r)
        print(f"{name}:")
        print(f"  Precision={p}, Recall={r}, F1={f1:.4f}")

        # في السياق الطبي، Recall غالبًا أهم بكثير: تفويت مريض فعلي (FN)
        # قد يعني تأخير علاج حرج، بينما إنذار خاطئ (FP) يعني فقط
        # فحوصات إضافية غير ضرورية — تكلفة غير متماثلة بوضوح
        print(f"  → تفسير سريري: من كل 100 مريض فعلي، يكتشف {r*100:.0f} فقط\n")

compare_medical_classifiers()

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-05-ch05-metrics.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-05-ch05-metrics.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — قياس الأداء: مقاييس التصنيف والحيرة
Mastering Transformers and Large Language Models — Metrics

Book slug : mastering-llms-transformers
Chapter   : ch-05-ch05-metrics (lesson 5)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-01-foundations/ch05-metrics.ipynb#cell-1
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
    ("numpy", "numpy")
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
# [002] content/notebooks/part-01-foundations/ch05-metrics.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """حساب عناصر مصفوفة الالتباس الأربعة للتصنيف الثنائي."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}

def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Precision = TP / (TP + FP)
    "من بين كل ما تنبأ به النموذج كإيجابي، كم نسبة كانت صحيحة فعلًا؟"
    مهمة عندما تكون تكلفة False Positive عالية (مثل: تصنيف بريد مهم كـ spam).
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    denominator = cm["TP"] + cm["FP"]
    return cm["TP"] / denominator if denominator > 0 else 0.0

def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Recall (Sensitivity) = TP / (TP + FN)
    "من بين كل الحالات الإيجابية الحقيقية، كم نسبة اكتشفها النموذج فعلًا؟"
    مهمة عندما تكون تكلفة False Negative عالية (مثل: تشخيص طبي خطير، كشف احتيال).
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    denominator = cm["TP"] + cm["FN"]
    return cm["TP"] / denominator if denominator > 0 else 0.0

def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    F1-Score = المتوسط التوافقي (Harmonic Mean) بين Precision وRecall.
    نستخدم المتوسط التوافقي (وليس الحسابي) لأنه يُعاقب بشدة الفارق الكبير
    بينهما — نموذج بـ Precision=1.0 وRecall=0.01 سيحصل على F1 منخفض جدًا
    (0.0198) رغم أن المتوسط الحسابي سيكون 0.505 (مُضلِّل تمامًا).

        F1 = 2 × (Precision × Recall) / (Precision + Recall)
    """
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy = (TP+TN) / الإجمالي — مُضلِّلة في البيانات غير المتوازنة."""
    return np.mean(y_true == y_pred)

# === توضيح عملي لمشكلة البيانات غير المتوازنة ===
def demonstrate_imbalanced_data_problem():
    # محاكاة: كشف احتيال، 99% من المعاملات شرعية (0) و1% احتيال (1)
    np.random.seed(42)
    n_samples = 1000
    y_true = np.array([0] * 990 + [1] * 10)

    # النموذج "الغبي": يُصنِّف كل شيء كـ"شرعي" (0)
    y_pred_dummy = np.zeros(n_samples)

    # النموذج الذكي نسبيًا: يكتشف 7 من أصل 10 حالات احتيال، مع بعض الأخطاء
    y_pred_smart = y_true.copy()
    y_pred_smart[990:993] = 0   # يُفوِّت 3 حالات احتيال حقيقية (FN)
    y_pred_smart[50:55] = 1     # يُصنِّف خطأً 5 معاملات شرعية كاحتيال (FP)

    print("=== النموذج الغبي (يتنبأ دائمًا بـ 'شرعي') ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred_dummy):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred_dummy):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred_dummy):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred_dummy):.4f}")

    print("\n=== النموذج الذكي (يكتشف الاحتيال فعليًا) ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred_smart):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred_smart):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred_smart):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred_smart):.4f}")

demonstrate_imbalanced_data_problem()
# === النموذج الغبي (يتنبأ دائمًا بـ 'شرعي') ===
# Accuracy: 0.9900   <- تبدو ممتازة! لكنها مُضلِّلة تمامًا
# Precision: 0.0000
# Recall: 0.0000
# F1-Score: 0.0000   <- يكشف الحقيقة فورًا: النموذج عديم الفائدة
#
# === النموذج الذكي (يكتشف الاحتيال فعليًا) ===
# Accuracy: 0.9920   <- أعلى بقليل فقط من الغبي! Accuracy لا تُميّز جيدًا هنا
# Precision: 0.5833
# Recall: 0.7000
# F1-Score: 0.6364   <- يعكس الأداء الحقيقي الأفضل بوضوح

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-01-foundations/ch05-metrics.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def calculate_perplexity(model_logits: np.ndarray, true_token_ids: np.ndarray) -> float:
    """
    Perplexity = exp(Cross-Entropy Loss المتوسطة لكل رمز)

    التفسير البديهي: "متوسط عدد الخيارات المتساوية الاحتمال التي
    يتردد بينها النموذج فعليًا عند كل خطوة تنبؤ". نموذج مثالي
    يُنتج Perplexity=1 (يقين تام)، بينما نموذج عشوائي تمامًا على
    مفردات حجمها N يُنتج Perplexity≈N.
    """
    # softmax على logits لكل موضع
    def stable_softmax(x):
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    probs = stable_softmax(model_logits)   # (seq_len, vocab_size)

    # احتمال الرمز الصحيح الفعلي عند كل موضع
    seq_len = len(true_token_ids)
    true_token_probs = probs[np.arange(seq_len), true_token_ids]
    true_token_probs = np.clip(true_token_probs, 1e-10, 1.0)

    # Cross-Entropy المتوسطة (بالـ nats، أساس e)
    avg_cross_entropy = -np.mean(np.log(true_token_probs))
    perplexity = np.exp(avg_cross_entropy)

    return perplexity

# مثال: مقارنة نموذجين على نفس الجملة
vocab_size = 1000
seq_len = 5
true_tokens = np.array([42, 17, 983, 5, 200])

# نموذج واثق وصحيح غالبًا (logits عالية للرموز الصحيحة)
np.random.seed(1)
confident_logits = np.random.randn(seq_len, vocab_size) * 0.5
for i, token in enumerate(true_tokens):
    confident_logits[i, token] += 8.0  # يُعزِّز احتمال الرمز الصحيح بشدة

# نموذج عشوائي تقريبًا
random_logits = np.random.randn(seq_len, vocab_size) * 0.5

ppl_confident = calculate_perplexity(confident_logits, true_tokens)
ppl_random = calculate_perplexity(random_logits, true_tokens)

print(f"Perplexity (نموذج واثق وصحيح): {ppl_confident:.2f}")
print(f"Perplexity (نموذج شبه عشوائي): {ppl_random:.2f}")
# Perplexity (نموذج واثق وصحيح): 1.02   <- قريب جدًا من 1 (مثالي تقريبًا)
# Perplexity (نموذج شبه عشوائي): 847.33  <- قريب من vocab_size (عشوائي تقريبًا)

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-01-foundations/ch05-metrics.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def compare_medical_classifiers():
    scenarios = {
        "نظام A (دقيق لكن يُفوِّت حالات)": {"precision": 0.95, "recall": 0.40},
        "نظام B (يكتشف أغلب الحالات لكن بإنذارات خاطئة أكثر)": {"precision": 0.60, "recall": 0.90},
    }

    for name, metrics in scenarios.items():
        p, r = metrics["precision"], metrics["recall"]
        f1 = 2 * p * r / (p + r)
        print(f"{name}:")
        print(f"  Precision={p}, Recall={r}, F1={f1:.4f}")

        # في السياق الطبي، Recall غالبًا أهم بكثير: تفويت مريض فعلي (FN)
        # قد يعني تأخير علاج حرج، بينما إنذار خاطئ (FP) يعني فقط
        # فحوصات إضافية غير ضرورية — تكلفة غير متماثلة بوضوح
        print(f"  → تفسير سريري: من كل 100 مريض فعلي، يكتشف {r*100:.0f} فقط\n")

compare_medical_classifiers()

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-01-foundations/ch05-metrics.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-05-ch05-metrics.ipynb#cell-2
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
    ("numpy", "numpy")
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-05-ch05-metrics.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """حساب عناصر مصفوفة الالتباس الأربعة للتصنيف الثنائي."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}

def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Precision = TP / (TP + FP)
    "من بين كل ما تنبأ به النموذج كإيجابي، كم نسبة كانت صحيحة فعلًا؟"
    مهمة عندما تكون تكلفة False Positive عالية (مثل: تصنيف بريد مهم كـ spam).
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    denominator = cm["TP"] + cm["FP"]
    return cm["TP"] / denominator if denominator > 0 else 0.0

def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Recall (Sensitivity) = TP / (TP + FN)
    "من بين كل الحالات الإيجابية الحقيقية، كم نسبة اكتشفها النموذج فعلًا؟"
    مهمة عندما تكون تكلفة False Negative عالية (مثل: تشخيص طبي خطير، كشف احتيال).
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    denominator = cm["TP"] + cm["FN"]
    return cm["TP"] / denominator if denominator > 0 else 0.0

def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    F1-Score = المتوسط التوافقي (Harmonic Mean) بين Precision وRecall.
    نستخدم المتوسط التوافقي (وليس الحسابي) لأنه يُعاقب بشدة الفارق الكبير
    بينهما — نموذج بـ Precision=1.0 وRecall=0.01 سيحصل على F1 منخفض جدًا
    (0.0198) رغم أن المتوسط الحسابي سيكون 0.505 (مُضلِّل تمامًا).

        F1 = 2 × (Precision × Recall) / (Precision + Recall)
    """
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy = (TP+TN) / الإجمالي — مُضلِّلة في البيانات غير المتوازنة."""
    return np.mean(y_true == y_pred)

# === توضيح عملي لمشكلة البيانات غير المتوازنة ===
def demonstrate_imbalanced_data_problem():
    # محاكاة: كشف احتيال، 99% من المعاملات شرعية (0) و1% احتيال (1)
    np.random.seed(42)
    n_samples = 1000
    y_true = np.array([0] * 990 + [1] * 10)

    # النموذج "الغبي": يُصنِّف كل شيء كـ"شرعي" (0)
    y_pred_dummy = np.zeros(n_samples)

    # النموذج الذكي نسبيًا: يكتشف 7 من أصل 10 حالات احتيال، مع بعض الأخطاء
    y_pred_smart = y_true.copy()
    y_pred_smart[990:993] = 0   # يُفوِّت 3 حالات احتيال حقيقية (FN)
    y_pred_smart[50:55] = 1     # يُصنِّف خطأً 5 معاملات شرعية كاحتيال (FP)

    print("=== النموذج الغبي (يتنبأ دائمًا بـ 'شرعي') ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred_dummy):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred_dummy):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred_dummy):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred_dummy):.4f}")

    print("\n=== النموذج الذكي (يكتشف الاحتيال فعليًا) ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred_smart):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred_smart):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred_smart):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred_smart):.4f}")

demonstrate_imbalanced_data_problem()
# === النموذج الغبي (يتنبأ دائمًا بـ 'شرعي') ===
# Accuracy: 0.9900   <- تبدو ممتازة! لكنها مُضلِّلة تمامًا
# Precision: 0.0000
# Recall: 0.0000
# F1-Score: 0.0000   <- يكشف الحقيقة فورًا: النموذج عديم الفائدة
#
# === النموذج الذكي (يكتشف الاحتيال فعليًا) ===
# Accuracy: 0.9920   <- أعلى بقليل فقط من الغبي! Accuracy لا تُميّز جيدًا هنا
# Precision: 0.5833
# Recall: 0.7000
# F1-Score: 0.6364   <- يعكس الأداء الحقيقي الأفضل بوضوح

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-05-ch05-metrics.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def calculate_perplexity(model_logits: np.ndarray, true_token_ids: np.ndarray) -> float:
    """
    Perplexity = exp(Cross-Entropy Loss المتوسطة لكل رمز)

    التفسير البديهي: "متوسط عدد الخيارات المتساوية الاحتمال التي
    يتردد بينها النموذج فعليًا عند كل خطوة تنبؤ". نموذج مثالي
    يُنتج Perplexity=1 (يقين تام)، بينما نموذج عشوائي تمامًا على
    مفردات حجمها N يُنتج Perplexity≈N.
    """
    # softmax على logits لكل موضع
    def stable_softmax(x):
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    probs = stable_softmax(model_logits)   # (seq_len, vocab_size)

    # احتمال الرمز الصحيح الفعلي عند كل موضع
    seq_len = len(true_token_ids)
    true_token_probs = probs[np.arange(seq_len), true_token_ids]
    true_token_probs = np.clip(true_token_probs, 1e-10, 1.0)

    # Cross-Entropy المتوسطة (بالـ nats، أساس e)
    avg_cross_entropy = -np.mean(np.log(true_token_probs))
    perplexity = np.exp(avg_cross_entropy)

    return perplexity

# مثال: مقارنة نموذجين على نفس الجملة
vocab_size = 1000
seq_len = 5
true_tokens = np.array([42, 17, 983, 5, 200])

# نموذج واثق وصحيح غالبًا (logits عالية للرموز الصحيحة)
np.random.seed(1)
confident_logits = np.random.randn(seq_len, vocab_size) * 0.5
for i, token in enumerate(true_tokens):
    confident_logits[i, token] += 8.0  # يُعزِّز احتمال الرمز الصحيح بشدة

# نموذج عشوائي تقريبًا
random_logits = np.random.randn(seq_len, vocab_size) * 0.5

ppl_confident = calculate_perplexity(confident_logits, true_tokens)
ppl_random = calculate_perplexity(random_logits, true_tokens)

print(f"Perplexity (نموذج واثق وصحيح): {ppl_confident:.2f}")
print(f"Perplexity (نموذج شبه عشوائي): {ppl_random:.2f}")
# Perplexity (نموذج واثق وصحيح): 1.02   <- قريب جدًا من 1 (مثالي تقريبًا)
# Perplexity (نموذج شبه عشوائي): 847.33  <- قريب من vocab_size (عشوائي تقريبًا)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-05-ch05-metrics.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def compare_medical_classifiers():
    scenarios = {
        "نظام A (دقيق لكن يُفوِّت حالات)": {"precision": 0.95, "recall": 0.40},
        "نظام B (يكتشف أغلب الحالات لكن بإنذارات خاطئة أكثر)": {"precision": 0.60, "recall": 0.90},
    }

    for name, metrics in scenarios.items():
        p, r = metrics["precision"], metrics["recall"]
        f1 = 2 * p * r / (p + r)
        print(f"{name}:")
        print(f"  Precision={p}, Recall={r}, F1={f1:.4f}")

        # في السياق الطبي، Recall غالبًا أهم بكثير: تفويت مريض فعلي (FN)
        # قد يعني تأخير علاج حرج، بينما إنذار خاطئ (FP) يعني فقط
        # فحوصات إضافية غير ضرورية — تكلفة غير متماثلة بوضوح
        print(f"  → تفسير سريري: من كل 100 مريض فعلي، يكتشف {r*100:.0f} فقط\n")

compare_medical_classifiers()

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-05-ch05-metrics.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
