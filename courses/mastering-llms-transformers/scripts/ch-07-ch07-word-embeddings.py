#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — التمثيلات المتجهية للكلمات: Word2Vec وGloVe
Mastering Transformers and Large Language Models — Word Embeddings

Book slug : mastering-llms-transformers
Chapter   : ch-07-ch07-word-embeddings (lesson 7)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-07-ch07-word-embeddings.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — التمثيلات المتجهية للكلمات: Word2Vec وGloVe
Mastering Transformers and Large Language Models — Word Embeddings

Book slug : mastering-llms-transformers
Chapter   : ch-07-ch07-word-embeddings (lesson 7)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-07-ch07-word-embeddings.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — التمثيلات المتجهية للكلمات: Word2Vec وGloVe
Mastering Transformers and Large Language Models — Word Embeddings

Book slug : mastering-llms-transformers
Chapter   : ch-07-ch07-word-embeddings (lesson 7)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-02-language-representation/ch07-word-embeddings.ipynb#cell-1
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
# [002] content/notebooks/part-02-language-representation/ch07-word-embeddings.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from collections import Counter
import random

class Word2VecSkipGram:
    """
    تطبيق Skip-gram مع Negative Sampling (Mikolov et al., 2013)
    من الصفر بـ NumPy. هذا هو التطبيق الأصلي الذي أحدث ثورة في
    تمثيل الكلمات قبل ظهور المحوّلات.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 100,
                 learning_rate: float = 0.025):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = learning_rate

        # مصفوفتا تضمين منفصلتان: واحدة للكلمة "المركزية" وأخرى للسياق
        # (هذا تفصيل تنفيذي مهم: نحتفظ بتمثيلين لكل كلمة أثناء التدريب،
        # ونستخدم عادة مصفوفة الكلمة المركزية فقط كـ Embeddings النهائية)
        self.W_center = np.random.uniform(-0.5, 0.5, (vocab_size, embedding_dim)) / embedding_dim
        self.W_context = np.zeros((vocab_size, embedding_dim))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def train_pair(self, center_idx: int, context_idx: int,
                    negative_indices: list):
        """
        خطوة تدريب واحدة: زوج (كلمة مركزية، كلمة سياق حقيقية)
        + عدة عينات سلبية (كلمات عشوائية ليست في السياق الفعلي).

        الهدف: نُعظِّم P(context_idx|center_idx) = σ(v_center · v_context)
        ونُصغِّر P(negative|center_idx) لكل عينة سلبية — بدلًا من
        حساب softmax كامل عبر كل المفردات (مُكلِف جدًا حسابيًا
        لمفردات بحجم عشرات الآلاف أو أكثر).
        """
        v_center = self.W_center[center_idx]

        # === العينة الإيجابية (السياق الحقيقي) ===
        v_context_pos = self.W_context[context_idx]
        score_pos = self.sigmoid(np.dot(v_center, v_context_pos))
        # الخطأ: نريد score_pos قريبًا من 1 (تصنيف "إيجابي" صحيح)
        grad_pos = score_pos - 1

        grad_center_total = grad_pos * v_context_pos
        self.W_context[context_idx] -= self.lr * grad_pos * v_center

        # === العينات السلبية (كلمات عشوائية، نريد score قريبًا من 0) ===
        for neg_idx in negative_indices:
            v_context_neg = self.W_context[neg_idx]
            score_neg = self.sigmoid(np.dot(v_center, v_context_neg))
            grad_neg = score_neg - 0  # نريد قريبًا من 0

            grad_center_total += grad_neg * v_context_neg
            self.W_context[neg_idx] -= self.lr * grad_neg * v_center

        # تحديث متجه الكلمة المركزية بتجميع كل التدرجات
        self.W_center[center_idx] -= self.lr * grad_center_total

        # حساب الخسارة للمراقبة (Negative Log-Likelihood)
        loss = -np.log(score_pos + 1e-10)
        for neg_idx in negative_indices:
            score_neg = self.sigmoid(np.dot(v_center, self.W_context[neg_idx]))
            loss -= np.log(1 - score_neg + 1e-10)
        return loss

class NegativeSampler:
    """
    اختيار عينات سلبية بناءً على توزيع تكرار الكلمات، لكن مع
    "تنعيم" (smoothing) الأس 3/4 — حيلة مهمة جدًا من الورقة الأصلية
    لمنع الكلمات شائعة جدًا (مثل أدوات الربط) من الهيمنة الكاملة
    على العينات السلبية، مع إعطاء الكلمات النادرة فرصة أكبر نسبيًا
    من توزيعها الخام.
    """
    def __init__(self, word_frequencies: dict, power: float = 0.75):
        words = list(word_frequencies.keys())
        freqs = np.array([word_frequencies[w] for w in words], dtype=np.float64)

        # التنعيم: رفع كل تكرار للأس 0.75 ثم إعادة التطبيع
        smoothed_freqs = freqs ** power
        self.probabilities = smoothed_freqs / smoothed_freqs.sum()
        self.words = words

    def sample(self, n: int, exclude: set = None) -> list:
        samples = []
        while len(samples) < n:
            candidate = np.random.choice(len(self.words), p=self.probabilities)
            if exclude is None or candidate not in exclude:
                samples.append(candidate)
        return samples

def demonstrate_smoothing_effect():
    """توضيح أثر الأس 3/4 عمليًا: تقليل هيمنة الكلمات الشائعة جدًا."""
    word_freqs = {"ال": 10000, "في": 5000, "كتاب": 50, "يراعة": 2}

    freqs = np.array(list(word_freqs.values()), dtype=np.float64)
    raw_probs = freqs / freqs.sum()
    smoothed_probs = (freqs ** 0.75) / (freqs ** 0.75).sum()

    for word, raw, smooth in zip(word_freqs.keys(), raw_probs, smoothed_probs):
        print(f"{word}: احتمال خام={raw:.4f}, احتمال مُنعَّم={smooth:.4f}")

demonstrate_smoothing_effect()
# ال: احتمال خام=0.6645, احتمال مُنعَّم=0.4850    <- انخفضت هيمنته
# في: احتمال خام=0.3322, احتمال مُنعَّم=0.3630
# كتاب: احتمال خام=0.0033, احتمال مُنعَّم=0.0125   <- زادت فرصته نسبيًا
# يراعة: احتمال خام=0.0001, احتمال مُنعَّم=0.0019   <- زادت فرصته بشكل ملحوظ نسبيًا

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-02-language-representation/ch07-word-embeddings.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def explain_glove_concept():
    """
    GloVe (Global Vectors, Pennington et al., 2014): بدلًا من
    التدريب التنبؤي المحلي (نافذة سياق صغيرة كما في Word2Vec)،
    يبني GloVe أولًا "مصفوفة تعايش عامة" (co-occurrence matrix)
    X حيث X[i][j] = عدد مرات ظهور الكلمة j في سياق الكلمة i
    عبر **المجموعة النصية بأكملها**، ثم يُدرِّب Embeddings بحيث:

        v_i · v_j + b_i + b_j ≈ log(X[i][j])

    الفكرة الجوهرية: نسب تعايش الكلمات (وليس القيم الخام فقط)
    تحمل معلومات دلالية قوية جدًا. مثال كلاسيكي من الورقة:

        P(ice|solid) / P(steam|solid)  >> 1    (الجليد صلب فعلًا)
        P(ice|gas) / P(steam|gas)      << 1    (البخار غاز فعلًا)
        P(ice|water) / P(steam|water)  ≈ 1     (كلاهما متعلق بالماء)

    GloVe يستغل هذه النسب مباشرة في دالة الخسارة، بعكس Word2Vec
    الذي يتعلّمها بشكل ضمني فقط عبر التنبؤ المحلي المتكرر.
    """
    pass

def weighted_least_squares_glove_loss(co_occurrence_matrix, word_vectors,
                                        context_vectors, biases, context_biases,
                                        x_max=100, alpha=0.75):
    """
    دالة خسارة GloVe الكاملة: Weighted Least Squares.
    الوزن f(X_ij) يمنع الأزواج النادرة جدًا (ضوضاء إحصائية) والمتكررة
    جدًا (مهيمنة بشكل غير مفيد) من التأثير المفرط على التدريب.
    """
    total_loss = 0.0
    rows, cols = co_occurrence_matrix.nonzero()

    for i, j in zip(rows, cols):
        x_ij = co_occurrence_matrix[i, j]

        # دالة الوزن f(x): تتصاعد حتى x_max ثم تستقر عند 1
        weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0

        prediction = (np.dot(word_vectors[i], context_vectors[j]) +
                      biases[i] + context_biases[j])
        target = np.log(x_ij + 1e-10)

        total_loss += weight * (prediction - target) ** 2

    return total_loss

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-02-language-representation/ch07-word-embeddings.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def demonstrate_static_vs_contextual_limitation():
    """
    القيد الجوهري لـ Word2Vec/GloVe (Static Embeddings):
    لكل كلمة متجه واحد ثابت بغض النظر عن السياق، لذا لا يمكنها
    التمييز بين المعاني المتعددة لنفس الكلمة (Polysemy).
    """
    sentences_with_bank = [
        "ذهبت إلى البنك لسحب المال",          # بنك = مؤسسة مالية
        "جلسنا على ضفة النهر (بنك النهر)",     # بنك = ضفة (استخدام قديم/مجازي)
    ]

    print("مع Word2Vec/GloVe (Static):")
    print("  كلمة 'بنك' → نفس المتجه بالضبط في كلا الجملتين")
    print("  (النموذج لا يستطيع التمييز بين المعنيين إطلاقًا)")

    print("\nمع BERT/GPT (Contextual، الفصل 12 لاحقًا):")
    print("  كلمة 'بنك' في الجملة 1 → متجه قريب من [مال، حساب، قرض]")
    print("  كلمة 'بنك' في الجملة 2 → متجه قريب من [نهر، ماء، ضفة]")
    print("  (نفس الرمز، تمثيلان مختلفان تمامًا حسب السياق المحيط)")

demonstrate_static_vs_contextual_limitation()

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-02-language-representation/ch07-word-embeddings.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def manual_skipgram_step():
    vocab = {"القطة": 0, "تحب": 1, "الحليب": 2, "اللذيذ": 3, "السيارة": 4}
    vocab_size, embed_dim = len(vocab), 4

    np.random.seed(0)
    model = Word2VecSkipGram(vocab_size, embed_dim, learning_rate=0.1)

    # الجملة: "القطة تحب الحليب اللذيذ"
    # مع نافذة سياق = 1: (تحب → القطة), (تحب → الحليب)
    center_word = "تحب"
    context_word = "الحليب"
    negative_word = "السيارة"  # كلمة غير ذات صلة، عينة سلبية

    center_idx = vocab[center_word]
    context_idx = vocab[context_word]
    negative_idx = vocab[negative_word]

    v_center_before = model.W_center[center_idx].copy()

    loss = model.train_pair(center_idx, context_idx, [negative_idx])

    v_center_after = model.W_center[center_idx]
    change = np.linalg.norm(v_center_after - v_center_before)

    print(f"الخسارة: {loss:.4f}")
    print(f"مقدار التغيير في متجه '{center_word}': {change:.6f}")
    print(f"المتجه قبل: {v_center_before.round(4)}")
    print(f"المتجه بعد: {v_center_after.round(4)}")

manual_skipgram_step()

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-02-language-representation/ch07-word-embeddings.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-07-ch07-word-embeddings.ipynb#cell-1
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-07-ch07-word-embeddings.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from collections import Counter
import random

class Word2VecSkipGram:
    """
    تطبيق Skip-gram مع Negative Sampling (Mikolov et al., 2013)
    من الصفر بـ NumPy. هذا هو التطبيق الأصلي الذي أحدث ثورة في
    تمثيل الكلمات قبل ظهور المحوّلات.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 100,
                 learning_rate: float = 0.025):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = learning_rate

        # مصفوفتا تضمين منفصلتان: واحدة للكلمة "المركزية" وأخرى للسياق
        # (هذا تفصيل تنفيذي مهم: نحتفظ بتمثيلين لكل كلمة أثناء التدريب،
        # ونستخدم عادة مصفوفة الكلمة المركزية فقط كـ Embeddings النهائية)
        self.W_center = np.random.uniform(-0.5, 0.5, (vocab_size, embedding_dim)) / embedding_dim
        self.W_context = np.zeros((vocab_size, embedding_dim))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def train_pair(self, center_idx: int, context_idx: int,
                    negative_indices: list):
        """
        خطوة تدريب واحدة: زوج (كلمة مركزية، كلمة سياق حقيقية)
        + عدة عينات سلبية (كلمات عشوائية ليست في السياق الفعلي).

        الهدف: نُعظِّم P(context_idx|center_idx) = σ(v_center · v_context)
        ونُصغِّر P(negative|center_idx) لكل عينة سلبية — بدلًا من
        حساب softmax كامل عبر كل المفردات (مُكلِف جدًا حسابيًا
        لمفردات بحجم عشرات الآلاف أو أكثر).
        """
        v_center = self.W_center[center_idx]

        # === العينة الإيجابية (السياق الحقيقي) ===
        v_context_pos = self.W_context[context_idx]
        score_pos = self.sigmoid(np.dot(v_center, v_context_pos))
        # الخطأ: نريد score_pos قريبًا من 1 (تصنيف "إيجابي" صحيح)
        grad_pos = score_pos - 1

        grad_center_total = grad_pos * v_context_pos
        self.W_context[context_idx] -= self.lr * grad_pos * v_center

        # === العينات السلبية (كلمات عشوائية، نريد score قريبًا من 0) ===
        for neg_idx in negative_indices:
            v_context_neg = self.W_context[neg_idx]
            score_neg = self.sigmoid(np.dot(v_center, v_context_neg))
            grad_neg = score_neg - 0  # نريد قريبًا من 0

            grad_center_total += grad_neg * v_context_neg
            self.W_context[neg_idx] -= self.lr * grad_neg * v_center

        # تحديث متجه الكلمة المركزية بتجميع كل التدرجات
        self.W_center[center_idx] -= self.lr * grad_center_total

        # حساب الخسارة للمراقبة (Negative Log-Likelihood)
        loss = -np.log(score_pos + 1e-10)
        for neg_idx in negative_indices:
            score_neg = self.sigmoid(np.dot(v_center, self.W_context[neg_idx]))
            loss -= np.log(1 - score_neg + 1e-10)
        return loss

class NegativeSampler:
    """
    اختيار عينات سلبية بناءً على توزيع تكرار الكلمات، لكن مع
    "تنعيم" (smoothing) الأس 3/4 — حيلة مهمة جدًا من الورقة الأصلية
    لمنع الكلمات شائعة جدًا (مثل أدوات الربط) من الهيمنة الكاملة
    على العينات السلبية، مع إعطاء الكلمات النادرة فرصة أكبر نسبيًا
    من توزيعها الخام.
    """
    def __init__(self, word_frequencies: dict, power: float = 0.75):
        words = list(word_frequencies.keys())
        freqs = np.array([word_frequencies[w] for w in words], dtype=np.float64)

        # التنعيم: رفع كل تكرار للأس 0.75 ثم إعادة التطبيع
        smoothed_freqs = freqs ** power
        self.probabilities = smoothed_freqs / smoothed_freqs.sum()
        self.words = words

    def sample(self, n: int, exclude: set = None) -> list:
        samples = []
        while len(samples) < n:
            candidate = np.random.choice(len(self.words), p=self.probabilities)
            if exclude is None or candidate not in exclude:
                samples.append(candidate)
        return samples

def demonstrate_smoothing_effect():
    """توضيح أثر الأس 3/4 عمليًا: تقليل هيمنة الكلمات الشائعة جدًا."""
    word_freqs = {"ال": 10000, "في": 5000, "كتاب": 50, "يراعة": 2}

    freqs = np.array(list(word_freqs.values()), dtype=np.float64)
    raw_probs = freqs / freqs.sum()
    smoothed_probs = (freqs ** 0.75) / (freqs ** 0.75).sum()

    for word, raw, smooth in zip(word_freqs.keys(), raw_probs, smoothed_probs):
        print(f"{word}: احتمال خام={raw:.4f}, احتمال مُنعَّم={smooth:.4f}")

demonstrate_smoothing_effect()
# ال: احتمال خام=0.6645, احتمال مُنعَّم=0.4850    <- انخفضت هيمنته
# في: احتمال خام=0.3322, احتمال مُنعَّم=0.3630
# كتاب: احتمال خام=0.0033, احتمال مُنعَّم=0.0125   <- زادت فرصته نسبيًا
# يراعة: احتمال خام=0.0001, احتمال مُنعَّم=0.0019   <- زادت فرصته بشكل ملحوظ نسبيًا

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-07-ch07-word-embeddings.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def explain_glove_concept():
    """
    GloVe (Global Vectors, Pennington et al., 2014): بدلًا من
    التدريب التنبؤي المحلي (نافذة سياق صغيرة كما في Word2Vec)،
    يبني GloVe أولًا "مصفوفة تعايش عامة" (co-occurrence matrix)
    X حيث X[i][j] = عدد مرات ظهور الكلمة j في سياق الكلمة i
    عبر **المجموعة النصية بأكملها**، ثم يُدرِّب Embeddings بحيث:

        v_i · v_j + b_i + b_j ≈ log(X[i][j])

    الفكرة الجوهرية: نسب تعايش الكلمات (وليس القيم الخام فقط)
    تحمل معلومات دلالية قوية جدًا. مثال كلاسيكي من الورقة:

        P(ice|solid) / P(steam|solid)  >> 1    (الجليد صلب فعلًا)
        P(ice|gas) / P(steam|gas)      << 1    (البخار غاز فعلًا)
        P(ice|water) / P(steam|water)  ≈ 1     (كلاهما متعلق بالماء)

    GloVe يستغل هذه النسب مباشرة في دالة الخسارة، بعكس Word2Vec
    الذي يتعلّمها بشكل ضمني فقط عبر التنبؤ المحلي المتكرر.
    """
    pass

def weighted_least_squares_glove_loss(co_occurrence_matrix, word_vectors,
                                        context_vectors, biases, context_biases,
                                        x_max=100, alpha=0.75):
    """
    دالة خسارة GloVe الكاملة: Weighted Least Squares.
    الوزن f(X_ij) يمنع الأزواج النادرة جدًا (ضوضاء إحصائية) والمتكررة
    جدًا (مهيمنة بشكل غير مفيد) من التأثير المفرط على التدريب.
    """
    total_loss = 0.0
    rows, cols = co_occurrence_matrix.nonzero()

    for i, j in zip(rows, cols):
        x_ij = co_occurrence_matrix[i, j]

        # دالة الوزن f(x): تتصاعد حتى x_max ثم تستقر عند 1
        weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0

        prediction = (np.dot(word_vectors[i], context_vectors[j]) +
                      biases[i] + context_biases[j])
        target = np.log(x_ij + 1e-10)

        total_loss += weight * (prediction - target) ** 2

    return total_loss

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-07-ch07-word-embeddings.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def demonstrate_static_vs_contextual_limitation():
    """
    القيد الجوهري لـ Word2Vec/GloVe (Static Embeddings):
    لكل كلمة متجه واحد ثابت بغض النظر عن السياق، لذا لا يمكنها
    التمييز بين المعاني المتعددة لنفس الكلمة (Polysemy).
    """
    sentences_with_bank = [
        "ذهبت إلى البنك لسحب المال",          # بنك = مؤسسة مالية
        "جلسنا على ضفة النهر (بنك النهر)",     # بنك = ضفة (استخدام قديم/مجازي)
    ]

    print("مع Word2Vec/GloVe (Static):")
    print("  كلمة 'بنك' → نفس المتجه بالضبط في كلا الجملتين")
    print("  (النموذج لا يستطيع التمييز بين المعنيين إطلاقًا)")

    print("\nمع BERT/GPT (Contextual، الفصل 12 لاحقًا):")
    print("  كلمة 'بنك' في الجملة 1 → متجه قريب من [مال، حساب، قرض]")
    print("  كلمة 'بنك' في الجملة 2 → متجه قريب من [نهر، ماء، ضفة]")
    print("  (نفس الرمز، تمثيلان مختلفان تمامًا حسب السياق المحيط)")

demonstrate_static_vs_contextual_limitation()

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-07-ch07-word-embeddings.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def manual_skipgram_step():
    vocab = {"القطة": 0, "تحب": 1, "الحليب": 2, "اللذيذ": 3, "السيارة": 4}
    vocab_size, embed_dim = len(vocab), 4

    np.random.seed(0)
    model = Word2VecSkipGram(vocab_size, embed_dim, learning_rate=0.1)

    # الجملة: "القطة تحب الحليب اللذيذ"
    # مع نافذة سياق = 1: (تحب → القطة), (تحب → الحليب)
    center_word = "تحب"
    context_word = "الحليب"
    negative_word = "السيارة"  # كلمة غير ذات صلة، عينة سلبية

    center_idx = vocab[center_word]
    context_idx = vocab[context_word]
    negative_idx = vocab[negative_word]

    v_center_before = model.W_center[center_idx].copy()

    loss = model.train_pair(center_idx, context_idx, [negative_idx])

    v_center_after = model.W_center[center_idx]
    change = np.linalg.norm(v_center_after - v_center_before)

    print(f"الخسارة: {loss:.4f}")
    print(f"مقدار التغيير في متجه '{center_word}': {change:.6f}")
    print(f"المتجه قبل: {v_center_before.round(4)}")
    print(f"المتجه بعد: {v_center_after.round(4)}")

manual_skipgram_step()

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-07-ch07-word-embeddings.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-07-ch07-word-embeddings.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — التمثيلات المتجهية للكلمات: Word2Vec وGloVe
Mastering Transformers and Large Language Models — Word Embeddings

Book slug : mastering-llms-transformers
Chapter   : ch-07-ch07-word-embeddings (lesson 7)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-02-language-representation/ch07-word-embeddings.ipynb#cell-1
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
# [002] content/notebooks/part-02-language-representation/ch07-word-embeddings.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from collections import Counter
import random

class Word2VecSkipGram:
    """
    تطبيق Skip-gram مع Negative Sampling (Mikolov et al., 2013)
    من الصفر بـ NumPy. هذا هو التطبيق الأصلي الذي أحدث ثورة في
    تمثيل الكلمات قبل ظهور المحوّلات.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 100,
                 learning_rate: float = 0.025):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = learning_rate

        # مصفوفتا تضمين منفصلتان: واحدة للكلمة "المركزية" وأخرى للسياق
        # (هذا تفصيل تنفيذي مهم: نحتفظ بتمثيلين لكل كلمة أثناء التدريب،
        # ونستخدم عادة مصفوفة الكلمة المركزية فقط كـ Embeddings النهائية)
        self.W_center = np.random.uniform(-0.5, 0.5, (vocab_size, embedding_dim)) / embedding_dim
        self.W_context = np.zeros((vocab_size, embedding_dim))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def train_pair(self, center_idx: int, context_idx: int,
                    negative_indices: list):
        """
        خطوة تدريب واحدة: زوج (كلمة مركزية، كلمة سياق حقيقية)
        + عدة عينات سلبية (كلمات عشوائية ليست في السياق الفعلي).

        الهدف: نُعظِّم P(context_idx|center_idx) = σ(v_center · v_context)
        ونُصغِّر P(negative|center_idx) لكل عينة سلبية — بدلًا من
        حساب softmax كامل عبر كل المفردات (مُكلِف جدًا حسابيًا
        لمفردات بحجم عشرات الآلاف أو أكثر).
        """
        v_center = self.W_center[center_idx]

        # === العينة الإيجابية (السياق الحقيقي) ===
        v_context_pos = self.W_context[context_idx]
        score_pos = self.sigmoid(np.dot(v_center, v_context_pos))
        # الخطأ: نريد score_pos قريبًا من 1 (تصنيف "إيجابي" صحيح)
        grad_pos = score_pos - 1

        grad_center_total = grad_pos * v_context_pos
        self.W_context[context_idx] -= self.lr * grad_pos * v_center

        # === العينات السلبية (كلمات عشوائية، نريد score قريبًا من 0) ===
        for neg_idx in negative_indices:
            v_context_neg = self.W_context[neg_idx]
            score_neg = self.sigmoid(np.dot(v_center, v_context_neg))
            grad_neg = score_neg - 0  # نريد قريبًا من 0

            grad_center_total += grad_neg * v_context_neg
            self.W_context[neg_idx] -= self.lr * grad_neg * v_center

        # تحديث متجه الكلمة المركزية بتجميع كل التدرجات
        self.W_center[center_idx] -= self.lr * grad_center_total

        # حساب الخسارة للمراقبة (Negative Log-Likelihood)
        loss = -np.log(score_pos + 1e-10)
        for neg_idx in negative_indices:
            score_neg = self.sigmoid(np.dot(v_center, self.W_context[neg_idx]))
            loss -= np.log(1 - score_neg + 1e-10)
        return loss

class NegativeSampler:
    """
    اختيار عينات سلبية بناءً على توزيع تكرار الكلمات، لكن مع
    "تنعيم" (smoothing) الأس 3/4 — حيلة مهمة جدًا من الورقة الأصلية
    لمنع الكلمات شائعة جدًا (مثل أدوات الربط) من الهيمنة الكاملة
    على العينات السلبية، مع إعطاء الكلمات النادرة فرصة أكبر نسبيًا
    من توزيعها الخام.
    """
    def __init__(self, word_frequencies: dict, power: float = 0.75):
        words = list(word_frequencies.keys())
        freqs = np.array([word_frequencies[w] for w in words], dtype=np.float64)

        # التنعيم: رفع كل تكرار للأس 0.75 ثم إعادة التطبيع
        smoothed_freqs = freqs ** power
        self.probabilities = smoothed_freqs / smoothed_freqs.sum()
        self.words = words

    def sample(self, n: int, exclude: set = None) -> list:
        samples = []
        while len(samples) < n:
            candidate = np.random.choice(len(self.words), p=self.probabilities)
            if exclude is None or candidate not in exclude:
                samples.append(candidate)
        return samples

def demonstrate_smoothing_effect():
    """توضيح أثر الأس 3/4 عمليًا: تقليل هيمنة الكلمات الشائعة جدًا."""
    word_freqs = {"ال": 10000, "في": 5000, "كتاب": 50, "يراعة": 2}

    freqs = np.array(list(word_freqs.values()), dtype=np.float64)
    raw_probs = freqs / freqs.sum()
    smoothed_probs = (freqs ** 0.75) / (freqs ** 0.75).sum()

    for word, raw, smooth in zip(word_freqs.keys(), raw_probs, smoothed_probs):
        print(f"{word}: احتمال خام={raw:.4f}, احتمال مُنعَّم={smooth:.4f}")

demonstrate_smoothing_effect()
# ال: احتمال خام=0.6645, احتمال مُنعَّم=0.4850    <- انخفضت هيمنته
# في: احتمال خام=0.3322, احتمال مُنعَّم=0.3630
# كتاب: احتمال خام=0.0033, احتمال مُنعَّم=0.0125   <- زادت فرصته نسبيًا
# يراعة: احتمال خام=0.0001, احتمال مُنعَّم=0.0019   <- زادت فرصته بشكل ملحوظ نسبيًا

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-02-language-representation/ch07-word-embeddings.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def explain_glove_concept():
    """
    GloVe (Global Vectors, Pennington et al., 2014): بدلًا من
    التدريب التنبؤي المحلي (نافذة سياق صغيرة كما في Word2Vec)،
    يبني GloVe أولًا "مصفوفة تعايش عامة" (co-occurrence matrix)
    X حيث X[i][j] = عدد مرات ظهور الكلمة j في سياق الكلمة i
    عبر **المجموعة النصية بأكملها**، ثم يُدرِّب Embeddings بحيث:

        v_i · v_j + b_i + b_j ≈ log(X[i][j])

    الفكرة الجوهرية: نسب تعايش الكلمات (وليس القيم الخام فقط)
    تحمل معلومات دلالية قوية جدًا. مثال كلاسيكي من الورقة:

        P(ice|solid) / P(steam|solid)  >> 1    (الجليد صلب فعلًا)
        P(ice|gas) / P(steam|gas)      << 1    (البخار غاز فعلًا)
        P(ice|water) / P(steam|water)  ≈ 1     (كلاهما متعلق بالماء)

    GloVe يستغل هذه النسب مباشرة في دالة الخسارة، بعكس Word2Vec
    الذي يتعلّمها بشكل ضمني فقط عبر التنبؤ المحلي المتكرر.
    """
    pass

def weighted_least_squares_glove_loss(co_occurrence_matrix, word_vectors,
                                        context_vectors, biases, context_biases,
                                        x_max=100, alpha=0.75):
    """
    دالة خسارة GloVe الكاملة: Weighted Least Squares.
    الوزن f(X_ij) يمنع الأزواج النادرة جدًا (ضوضاء إحصائية) والمتكررة
    جدًا (مهيمنة بشكل غير مفيد) من التأثير المفرط على التدريب.
    """
    total_loss = 0.0
    rows, cols = co_occurrence_matrix.nonzero()

    for i, j in zip(rows, cols):
        x_ij = co_occurrence_matrix[i, j]

        # دالة الوزن f(x): تتصاعد حتى x_max ثم تستقر عند 1
        weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0

        prediction = (np.dot(word_vectors[i], context_vectors[j]) +
                      biases[i] + context_biases[j])
        target = np.log(x_ij + 1e-10)

        total_loss += weight * (prediction - target) ** 2

    return total_loss

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-02-language-representation/ch07-word-embeddings.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def demonstrate_static_vs_contextual_limitation():
    """
    القيد الجوهري لـ Word2Vec/GloVe (Static Embeddings):
    لكل كلمة متجه واحد ثابت بغض النظر عن السياق، لذا لا يمكنها
    التمييز بين المعاني المتعددة لنفس الكلمة (Polysemy).
    """
    sentences_with_bank = [
        "ذهبت إلى البنك لسحب المال",          # بنك = مؤسسة مالية
        "جلسنا على ضفة النهر (بنك النهر)",     # بنك = ضفة (استخدام قديم/مجازي)
    ]

    print("مع Word2Vec/GloVe (Static):")
    print("  كلمة 'بنك' → نفس المتجه بالضبط في كلا الجملتين")
    print("  (النموذج لا يستطيع التمييز بين المعنيين إطلاقًا)")

    print("\nمع BERT/GPT (Contextual، الفصل 12 لاحقًا):")
    print("  كلمة 'بنك' في الجملة 1 → متجه قريب من [مال، حساب، قرض]")
    print("  كلمة 'بنك' في الجملة 2 → متجه قريب من [نهر، ماء، ضفة]")
    print("  (نفس الرمز، تمثيلان مختلفان تمامًا حسب السياق المحيط)")

demonstrate_static_vs_contextual_limitation()

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-02-language-representation/ch07-word-embeddings.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def manual_skipgram_step():
    vocab = {"القطة": 0, "تحب": 1, "الحليب": 2, "اللذيذ": 3, "السيارة": 4}
    vocab_size, embed_dim = len(vocab), 4

    np.random.seed(0)
    model = Word2VecSkipGram(vocab_size, embed_dim, learning_rate=0.1)

    # الجملة: "القطة تحب الحليب اللذيذ"
    # مع نافذة سياق = 1: (تحب → القطة), (تحب → الحليب)
    center_word = "تحب"
    context_word = "الحليب"
    negative_word = "السيارة"  # كلمة غير ذات صلة، عينة سلبية

    center_idx = vocab[center_word]
    context_idx = vocab[context_word]
    negative_idx = vocab[negative_word]

    v_center_before = model.W_center[center_idx].copy()

    loss = model.train_pair(center_idx, context_idx, [negative_idx])

    v_center_after = model.W_center[center_idx]
    change = np.linalg.norm(v_center_after - v_center_before)

    print(f"الخسارة: {loss:.4f}")
    print(f"مقدار التغيير في متجه '{center_word}': {change:.6f}")
    print(f"المتجه قبل: {v_center_before.round(4)}")
    print(f"المتجه بعد: {v_center_after.round(4)}")

manual_skipgram_step()

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-02-language-representation/ch07-word-embeddings.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-07-ch07-word-embeddings.ipynb#cell-2
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-07-ch07-word-embeddings.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from collections import Counter
import random

class Word2VecSkipGram:
    """
    تطبيق Skip-gram مع Negative Sampling (Mikolov et al., 2013)
    من الصفر بـ NumPy. هذا هو التطبيق الأصلي الذي أحدث ثورة في
    تمثيل الكلمات قبل ظهور المحوّلات.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 100,
                 learning_rate: float = 0.025):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = learning_rate

        # مصفوفتا تضمين منفصلتان: واحدة للكلمة "المركزية" وأخرى للسياق
        # (هذا تفصيل تنفيذي مهم: نحتفظ بتمثيلين لكل كلمة أثناء التدريب،
        # ونستخدم عادة مصفوفة الكلمة المركزية فقط كـ Embeddings النهائية)
        self.W_center = np.random.uniform(-0.5, 0.5, (vocab_size, embedding_dim)) / embedding_dim
        self.W_context = np.zeros((vocab_size, embedding_dim))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def train_pair(self, center_idx: int, context_idx: int,
                    negative_indices: list):
        """
        خطوة تدريب واحدة: زوج (كلمة مركزية، كلمة سياق حقيقية)
        + عدة عينات سلبية (كلمات عشوائية ليست في السياق الفعلي).

        الهدف: نُعظِّم P(context_idx|center_idx) = σ(v_center · v_context)
        ونُصغِّر P(negative|center_idx) لكل عينة سلبية — بدلًا من
        حساب softmax كامل عبر كل المفردات (مُكلِف جدًا حسابيًا
        لمفردات بحجم عشرات الآلاف أو أكثر).
        """
        v_center = self.W_center[center_idx]

        # === العينة الإيجابية (السياق الحقيقي) ===
        v_context_pos = self.W_context[context_idx]
        score_pos = self.sigmoid(np.dot(v_center, v_context_pos))
        # الخطأ: نريد score_pos قريبًا من 1 (تصنيف "إيجابي" صحيح)
        grad_pos = score_pos - 1

        grad_center_total = grad_pos * v_context_pos
        self.W_context[context_idx] -= self.lr * grad_pos * v_center

        # === العينات السلبية (كلمات عشوائية، نريد score قريبًا من 0) ===
        for neg_idx in negative_indices:
            v_context_neg = self.W_context[neg_idx]
            score_neg = self.sigmoid(np.dot(v_center, v_context_neg))
            grad_neg = score_neg - 0  # نريد قريبًا من 0

            grad_center_total += grad_neg * v_context_neg
            self.W_context[neg_idx] -= self.lr * grad_neg * v_center

        # تحديث متجه الكلمة المركزية بتجميع كل التدرجات
        self.W_center[center_idx] -= self.lr * grad_center_total

        # حساب الخسارة للمراقبة (Negative Log-Likelihood)
        loss = -np.log(score_pos + 1e-10)
        for neg_idx in negative_indices:
            score_neg = self.sigmoid(np.dot(v_center, self.W_context[neg_idx]))
            loss -= np.log(1 - score_neg + 1e-10)
        return loss

class NegativeSampler:
    """
    اختيار عينات سلبية بناءً على توزيع تكرار الكلمات، لكن مع
    "تنعيم" (smoothing) الأس 3/4 — حيلة مهمة جدًا من الورقة الأصلية
    لمنع الكلمات شائعة جدًا (مثل أدوات الربط) من الهيمنة الكاملة
    على العينات السلبية، مع إعطاء الكلمات النادرة فرصة أكبر نسبيًا
    من توزيعها الخام.
    """
    def __init__(self, word_frequencies: dict, power: float = 0.75):
        words = list(word_frequencies.keys())
        freqs = np.array([word_frequencies[w] for w in words], dtype=np.float64)

        # التنعيم: رفع كل تكرار للأس 0.75 ثم إعادة التطبيع
        smoothed_freqs = freqs ** power
        self.probabilities = smoothed_freqs / smoothed_freqs.sum()
        self.words = words

    def sample(self, n: int, exclude: set = None) -> list:
        samples = []
        while len(samples) < n:
            candidate = np.random.choice(len(self.words), p=self.probabilities)
            if exclude is None or candidate not in exclude:
                samples.append(candidate)
        return samples

def demonstrate_smoothing_effect():
    """توضيح أثر الأس 3/4 عمليًا: تقليل هيمنة الكلمات الشائعة جدًا."""
    word_freqs = {"ال": 10000, "في": 5000, "كتاب": 50, "يراعة": 2}

    freqs = np.array(list(word_freqs.values()), dtype=np.float64)
    raw_probs = freqs / freqs.sum()
    smoothed_probs = (freqs ** 0.75) / (freqs ** 0.75).sum()

    for word, raw, smooth in zip(word_freqs.keys(), raw_probs, smoothed_probs):
        print(f"{word}: احتمال خام={raw:.4f}, احتمال مُنعَّم={smooth:.4f}")

demonstrate_smoothing_effect()
# ال: احتمال خام=0.6645, احتمال مُنعَّم=0.4850    <- انخفضت هيمنته
# في: احتمال خام=0.3322, احتمال مُنعَّم=0.3630
# كتاب: احتمال خام=0.0033, احتمال مُنعَّم=0.0125   <- زادت فرصته نسبيًا
# يراعة: احتمال خام=0.0001, احتمال مُنعَّم=0.0019   <- زادت فرصته بشكل ملحوظ نسبيًا

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-07-ch07-word-embeddings.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def explain_glove_concept():
    """
    GloVe (Global Vectors, Pennington et al., 2014): بدلًا من
    التدريب التنبؤي المحلي (نافذة سياق صغيرة كما في Word2Vec)،
    يبني GloVe أولًا "مصفوفة تعايش عامة" (co-occurrence matrix)
    X حيث X[i][j] = عدد مرات ظهور الكلمة j في سياق الكلمة i
    عبر **المجموعة النصية بأكملها**، ثم يُدرِّب Embeddings بحيث:

        v_i · v_j + b_i + b_j ≈ log(X[i][j])

    الفكرة الجوهرية: نسب تعايش الكلمات (وليس القيم الخام فقط)
    تحمل معلومات دلالية قوية جدًا. مثال كلاسيكي من الورقة:

        P(ice|solid) / P(steam|solid)  >> 1    (الجليد صلب فعلًا)
        P(ice|gas) / P(steam|gas)      << 1    (البخار غاز فعلًا)
        P(ice|water) / P(steam|water)  ≈ 1     (كلاهما متعلق بالماء)

    GloVe يستغل هذه النسب مباشرة في دالة الخسارة، بعكس Word2Vec
    الذي يتعلّمها بشكل ضمني فقط عبر التنبؤ المحلي المتكرر.
    """
    pass

def weighted_least_squares_glove_loss(co_occurrence_matrix, word_vectors,
                                        context_vectors, biases, context_biases,
                                        x_max=100, alpha=0.75):
    """
    دالة خسارة GloVe الكاملة: Weighted Least Squares.
    الوزن f(X_ij) يمنع الأزواج النادرة جدًا (ضوضاء إحصائية) والمتكررة
    جدًا (مهيمنة بشكل غير مفيد) من التأثير المفرط على التدريب.
    """
    total_loss = 0.0
    rows, cols = co_occurrence_matrix.nonzero()

    for i, j in zip(rows, cols):
        x_ij = co_occurrence_matrix[i, j]

        # دالة الوزن f(x): تتصاعد حتى x_max ثم تستقر عند 1
        weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0

        prediction = (np.dot(word_vectors[i], context_vectors[j]) +
                      biases[i] + context_biases[j])
        target = np.log(x_ij + 1e-10)

        total_loss += weight * (prediction - target) ** 2

    return total_loss

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-07-ch07-word-embeddings.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def demonstrate_static_vs_contextual_limitation():
    """
    القيد الجوهري لـ Word2Vec/GloVe (Static Embeddings):
    لكل كلمة متجه واحد ثابت بغض النظر عن السياق، لذا لا يمكنها
    التمييز بين المعاني المتعددة لنفس الكلمة (Polysemy).
    """
    sentences_with_bank = [
        "ذهبت إلى البنك لسحب المال",          # بنك = مؤسسة مالية
        "جلسنا على ضفة النهر (بنك النهر)",     # بنك = ضفة (استخدام قديم/مجازي)
    ]

    print("مع Word2Vec/GloVe (Static):")
    print("  كلمة 'بنك' → نفس المتجه بالضبط في كلا الجملتين")
    print("  (النموذج لا يستطيع التمييز بين المعنيين إطلاقًا)")

    print("\nمع BERT/GPT (Contextual، الفصل 12 لاحقًا):")
    print("  كلمة 'بنك' في الجملة 1 → متجه قريب من [مال، حساب، قرض]")
    print("  كلمة 'بنك' في الجملة 2 → متجه قريب من [نهر، ماء، ضفة]")
    print("  (نفس الرمز، تمثيلان مختلفان تمامًا حسب السياق المحيط)")

demonstrate_static_vs_contextual_limitation()

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-07-ch07-word-embeddings.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def manual_skipgram_step():
    vocab = {"القطة": 0, "تحب": 1, "الحليب": 2, "اللذيذ": 3, "السيارة": 4}
    vocab_size, embed_dim = len(vocab), 4

    np.random.seed(0)
    model = Word2VecSkipGram(vocab_size, embed_dim, learning_rate=0.1)

    # الجملة: "القطة تحب الحليب اللذيذ"
    # مع نافذة سياق = 1: (تحب → القطة), (تحب → الحليب)
    center_word = "تحب"
    context_word = "الحليب"
    negative_word = "السيارة"  # كلمة غير ذات صلة، عينة سلبية

    center_idx = vocab[center_word]
    context_idx = vocab[context_word]
    negative_idx = vocab[negative_word]

    v_center_before = model.W_center[center_idx].copy()

    loss = model.train_pair(center_idx, context_idx, [negative_idx])

    v_center_after = model.W_center[center_idx]
    change = np.linalg.norm(v_center_after - v_center_before)

    print(f"الخسارة: {loss:.4f}")
    print(f"مقدار التغيير في متجه '{center_word}': {change:.6f}")
    print(f"المتجه قبل: {v_center_before.round(4)}")
    print(f"المتجه بعد: {v_center_after.round(4)}")

manual_skipgram_step()

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-llms-transformers/notebooks/ch-07-ch07-word-embeddings.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
