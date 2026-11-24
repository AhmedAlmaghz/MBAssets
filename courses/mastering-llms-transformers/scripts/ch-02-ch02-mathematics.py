#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الأسس الرياضية للتعلم العميق
Mastering Transformers and Large Language Models — Mathematics

Book slug : mastering-llms-transformers
Chapter   : ch-02-ch02-mathematics (lesson 2)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-02-ch02-mathematics.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الأسس الرياضية للتعلم العميق
Mastering Transformers and Large Language Models — Mathematics

Book slug : mastering-llms-transformers
Chapter   : ch-02-ch02-mathematics (lesson 2)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-02-ch02-mathematics.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الأسس الرياضية للتعلم العميق
Mastering Transformers and Large Language Models — Mathematics

Book slug : mastering-llms-transformers
Chapter   : ch-02-ch02-mathematics (lesson 2)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-01-foundations/ch02-mathematics.ipynb#cell-1
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
# [002] content/notebooks/part-01-foundations/ch02-mathematics.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# === 1. المتجهات والمصفوفات: البنية التحتية لكل شيء ===

v = np.array([1.0, 2.0, 3.0])          # متجه (تمثيل كلمة واحدة، بُعد 3)
M = np.array([[1, 0, 1],
              [0, 1, 1],
              [1, 1, 0]])               # مصفوفة تحويل (3×3)

# الضرب المصفوفي: جوهر كل طبقة خطية (Linear Layer) في أي شبكة عصبية
transformed_v = M @ v
print(f"المتجه بعد التحويل: {transformed_v}")

# === 2. التشابه بين المتجهات: أساس كل آلية Attention ===

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    يقيس زاوية الاتجاه بين متجهين بغض النظر عن حجمهما (magnitude).
    هذا بالضبط ما يُستخدَم لقياس "التشابه الدلالي" بين الكلمات
    في فضاء Embeddings (الفصل 7)، وكأساس نظري لعملية Attention Score
    (التي تستخدم الضرب النقطي كتقريب سريع لهذا التشابه).
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b + 1e-8)

word_king = np.array([0.8, 0.6, 0.1])
word_queen = np.array([0.75, 0.65, 0.15])
word_apple = np.array([0.1, 0.1, 0.9])

print(f"تشابه (ملك، ملكة): {cosine_similarity(word_king, word_queen):.4f}")
print(f"تشابه (ملك، تفاحة): {cosine_similarity(word_king, word_apple):.4f}")
# تشابه (ملك، ملكة): 0.9959   <- قريبان دلاليًا
# تشابه (ملك، تفاحة): 0.3812  <- بعيدان دلاليًا

# === 3. القيم الذاتية والمتجهات الذاتية (Eigenvalues/Eigenvectors) ===
# مفهوم مهم لفهم PCA، وتحليل سلوك المصفوفات في الشبكات العميقة
# (مثل تفسير استقرار/عدم استقرار التدرجات عبر الطبقات)

A = np.array([[4, 1], [2, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"\nالقيم الذاتية: {eigenvalues}")
print(f"المتجهات الذاتية:\n{eigenvectors}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-01-foundations/ch02-mathematics.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def softmax_naive(logits: np.ndarray) -> np.ndarray:
    """
    التطبيق الساذج لـ softmax — يُستخدَم فقط للتوضيح التعليمي،
    لأنه غير مستقر رقميًا (قد يُسبب Overflow مع logits كبيرة).
    """
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)

def softmax_stable(logits: np.ndarray) -> np.ndarray:
    """
    softmax المستقر رقميًا (النسخة المُستخدَمة فعليًا في كل مكتبة
    تعلّم عميق). الحيلة: طرح أقصى قيمة قبل الأس (exp) لا تُغيّر الناتج
    رياضيًا (لأن softmax(x) = softmax(x - c) لأي ثابت c) لكنها تمنع
    overflow لأن أكبر قيمة أس تُصبح exp(0)=1 بدلًا من قيمة ضخمة محتملة.
    """
    shifted_logits = logits - np.max(logits)
    exp_logits = np.exp(shifted_logits)
    return exp_logits / np.sum(exp_logits)

# توضيح المشكلة عمليًا
large_logits = np.array([1000.0, 1001.0, 1002.0])

try:
    result_naive = softmax_naive(large_logits)
    print(f"Naive softmax: {result_naive}")
except Exception as e:
    print(f"❌ Naive softmax فشل: {e}")
# ينتج: [nan, nan, nan] بسبب exp(1000) = inf ثم inf/inf = nan

result_stable = softmax_stable(large_logits)
print(f"✅ Stable softmax: {result_stable}")
# ينتج: [0.0900, 0.2447, 0.6652]  — نتيجة صحيحة ومستقرة

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-01-foundations/ch02-mathematics.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def entropy(p: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    الإنتروبيا (Shannon Entropy): تقيس "عدم اليقين" المتأصل في توزيع
    احتمالي. كلما كانت الإنتروبيا أعلى، كان التوزيع أكثر "انتشارًا"
    (uniform)، وكلما كانت أقل، كان أكثر "تركّزًا" حول قيمة واحدة.

        H(P) = -Σ p(x)·log(p(x))
    """
    p = np.clip(p, epsilon, 1.0)  # تجنّب log(0)
    return -np.sum(p * np.log(p))

def cross_entropy(p_true: np.ndarray, q_predicted: np.ndarray,
                   epsilon: float = 1e-10) -> float:
    """
    Cross-Entropy: تقيس "متوسط عدد البتات" اللازمة لترميز أحداث
    مسحوبة من التوزيع الحقيقي P، باستخدام كود مُصمَّم بناءً على
    التوزيع المُتوقَّع Q. هذه هي دالة الخسارة الأساسية لكل نموذج لغوي:

        H(P, Q) = -Σ p_true(x)·log(q_predicted(x))

    في التصنيف/توليد اللغة، p_true عادة one-hot (0 أو 1 فقط)،
    فتُختزَل المعادلة إلى: -log(q_predicted[الفئة_الصحيحة])
    """
    q_predicted = np.clip(q_predicted, epsilon, 1.0)
    return -np.sum(p_true * np.log(q_predicted))

def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    KL Divergence: تقيس "المسافة" (غير المتماثلة!) بين توزيعين
    احتماليين — كم "بتة إضافية" نُهدرها باستخدام Q بدلًا من P الحقيقي.

        KL(P||Q) = Σ p(x)·log(p(x)/q(x)) = H(P,Q) - H(P)

    تُستخدَم في: قيد RLHF (الفصل 33)، Knowledge Distillation (الفصل 25)،
    وVariational Autoencoders.
    ملاحظة حرجة: KL(P||Q) ≠ KL(Q||P) — ليست مسافة بالمعنى الرياضي الدقيق!
    """
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    return np.sum(p * np.log(p / q))

# مثال تطبيقي متكامل
true_distribution = np.array([1.0, 0.0, 0.0, 0.0])  # الكلمة الصحيحة: الفئة 0 (one-hot)

# نموذج جيد: واثق وصحيح
good_model_pred = np.array([0.85, 0.05, 0.05, 0.05])
# نموذج ضعيف: غير واثق إطلاقًا (توزيع شبه منتظم)
poor_model_pred = np.array([0.30, 0.25, 0.25, 0.20])
# نموذج سيء جدًا: واثق لكن خاطئ
wrong_confident_pred = np.array([0.05, 0.85, 0.05, 0.05])

for name, pred in [("نموذج جيد", good_model_pred),
                    ("نموذج ضعيف", poor_model_pred),
                    ("نموذج واثق وخاطئ", wrong_confident_pred)]:
    ce = cross_entropy(true_distribution, pred)
    print(f"{name}: Cross-Entropy Loss = {ce:.4f}")

# نموذج جيد: Cross-Entropy Loss = 0.1625
# نموذج ضعيف: Cross-Entropy Loss = 1.2040
# نموذج واثق وخاطئ: Cross-Entropy Loss = 2.9957  <- أسوأ عقوبة بكثير!

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-01-foundations/ch02-mathematics.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def verify_relationship(p_true, q_pred):
    """
    التحقق العملي من: H(P,Q) = H(P) + KL(P||Q)
    بمعنى: Cross-Entropy = Entropy الذاتية + "الفاقد الإضافي" بسبب
    استخدام توزيع تقريبي Q بدلًا من التوزيع الحقيقي P.
    """
    h_p = entropy(p_true)
    h_pq = cross_entropy(p_true, q_pred)
    kl_pq = kl_divergence(p_true, q_pred)

    print(f"H(P) = {h_p:.4f}")
    print(f"KL(P||Q) = {kl_pq:.4f}")
    print(f"H(P) + KL(P||Q) = {h_p + kl_pq:.4f}")
    print(f"H(P,Q) = {h_pq:.4f}")
    print(f"متطابقان؟ {np.isclose(h_p + kl_pq, h_pq)}")

p = np.array([0.5, 0.3, 0.2])
q = np.array([0.4, 0.4, 0.2])
verify_relationship(p, q)
# H(P) = 1.0297
# KL(P||Q) = 0.0338
# H(P) + KL(P||Q) = 1.0635
# H(P,Q) = 1.0635
# متطابقان؟ True

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-01-foundations/ch02-mathematics.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
vocab = ["نحن", "نذهب", "إلى", "المدرسة", "اليوم"]
true_idx = 3
true_dist = np.zeros(5)
true_dist[true_idx] = 1.0

predictions = {
    "تنبؤ ممتاز": np.array([0.02, 0.02, 0.05, 0.88, 0.03]),
    "تنبؤ متوسط": np.array([0.10, 0.10, 0.20, 0.40, 0.20]),
    "تنبؤ سيء": np.array([0.30, 0.30, 0.20, 0.10, 0.10]),
}

for name, pred in predictions.items():
    loss = cross_entropy(true_dist, pred)
    perplexity = np.exp(loss)  # سنشرح Perplexity بالتفصيل في الفصل 5
    print(f"{name}: Loss={loss:.4f}, Perplexity={perplexity:.4f}, "
          f"P(المدرسة)={pred[true_idx]:.2f}")

# تنبؤ ممتاز: Loss=0.1278, Perplexity=1.1363, P(المدرسة)=0.88
# تنبؤ متوسط: Loss=0.9163, Perplexity=2.4999, P(المدرسة)=0.40
# تنبؤ سيء:  Loss=2.3026, Perplexity=10.0000, P(المدرسة)=0.10

# ────────────────────────────────────────────────────────────────────────
# [007] content/notebooks/part-01-foundations/ch02-mathematics.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-02-ch02-mathematics.ipynb#cell-1
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-02-ch02-mathematics.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# === 1. المتجهات والمصفوفات: البنية التحتية لكل شيء ===

v = np.array([1.0, 2.0, 3.0])          # متجه (تمثيل كلمة واحدة، بُعد 3)
M = np.array([[1, 0, 1],
              [0, 1, 1],
              [1, 1, 0]])               # مصفوفة تحويل (3×3)

# الضرب المصفوفي: جوهر كل طبقة خطية (Linear Layer) في أي شبكة عصبية
transformed_v = M @ v
print(f"المتجه بعد التحويل: {transformed_v}")

# === 2. التشابه بين المتجهات: أساس كل آلية Attention ===

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    يقيس زاوية الاتجاه بين متجهين بغض النظر عن حجمهما (magnitude).
    هذا بالضبط ما يُستخدَم لقياس "التشابه الدلالي" بين الكلمات
    في فضاء Embeddings (الفصل 7)، وكأساس نظري لعملية Attention Score
    (التي تستخدم الضرب النقطي كتقريب سريع لهذا التشابه).
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b + 1e-8)

word_king = np.array([0.8, 0.6, 0.1])
word_queen = np.array([0.75, 0.65, 0.15])
word_apple = np.array([0.1, 0.1, 0.9])

print(f"تشابه (ملك، ملكة): {cosine_similarity(word_king, word_queen):.4f}")
print(f"تشابه (ملك، تفاحة): {cosine_similarity(word_king, word_apple):.4f}")
# تشابه (ملك، ملكة): 0.9959   <- قريبان دلاليًا
# تشابه (ملك، تفاحة): 0.3812  <- بعيدان دلاليًا

# === 3. القيم الذاتية والمتجهات الذاتية (Eigenvalues/Eigenvectors) ===
# مفهوم مهم لفهم PCA، وتحليل سلوك المصفوفات في الشبكات العميقة
# (مثل تفسير استقرار/عدم استقرار التدرجات عبر الطبقات)

A = np.array([[4, 1], [2, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"\nالقيم الذاتية: {eigenvalues}")
print(f"المتجهات الذاتية:\n{eigenvectors}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-02-ch02-mathematics.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def softmax_naive(logits: np.ndarray) -> np.ndarray:
    """
    التطبيق الساذج لـ softmax — يُستخدَم فقط للتوضيح التعليمي،
    لأنه غير مستقر رقميًا (قد يُسبب Overflow مع logits كبيرة).
    """
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)

def softmax_stable(logits: np.ndarray) -> np.ndarray:
    """
    softmax المستقر رقميًا (النسخة المُستخدَمة فعليًا في كل مكتبة
    تعلّم عميق). الحيلة: طرح أقصى قيمة قبل الأس (exp) لا تُغيّر الناتج
    رياضيًا (لأن softmax(x) = softmax(x - c) لأي ثابت c) لكنها تمنع
    overflow لأن أكبر قيمة أس تُصبح exp(0)=1 بدلًا من قيمة ضخمة محتملة.
    """
    shifted_logits = logits - np.max(logits)
    exp_logits = np.exp(shifted_logits)
    return exp_logits / np.sum(exp_logits)

# توضيح المشكلة عمليًا
large_logits = np.array([1000.0, 1001.0, 1002.0])

try:
    result_naive = softmax_naive(large_logits)
    print(f"Naive softmax: {result_naive}")
except Exception as e:
    print(f"❌ Naive softmax فشل: {e}")
# ينتج: [nan, nan, nan] بسبب exp(1000) = inf ثم inf/inf = nan

result_stable = softmax_stable(large_logits)
print(f"✅ Stable softmax: {result_stable}")
# ينتج: [0.0900, 0.2447, 0.6652]  — نتيجة صحيحة ومستقرة

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-02-ch02-mathematics.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def entropy(p: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    الإنتروبيا (Shannon Entropy): تقيس "عدم اليقين" المتأصل في توزيع
    احتمالي. كلما كانت الإنتروبيا أعلى، كان التوزيع أكثر "انتشارًا"
    (uniform)، وكلما كانت أقل، كان أكثر "تركّزًا" حول قيمة واحدة.

        H(P) = -Σ p(x)·log(p(x))
    """
    p = np.clip(p, epsilon, 1.0)  # تجنّب log(0)
    return -np.sum(p * np.log(p))

def cross_entropy(p_true: np.ndarray, q_predicted: np.ndarray,
                   epsilon: float = 1e-10) -> float:
    """
    Cross-Entropy: تقيس "متوسط عدد البتات" اللازمة لترميز أحداث
    مسحوبة من التوزيع الحقيقي P، باستخدام كود مُصمَّم بناءً على
    التوزيع المُتوقَّع Q. هذه هي دالة الخسارة الأساسية لكل نموذج لغوي:

        H(P, Q) = -Σ p_true(x)·log(q_predicted(x))

    في التصنيف/توليد اللغة، p_true عادة one-hot (0 أو 1 فقط)،
    فتُختزَل المعادلة إلى: -log(q_predicted[الفئة_الصحيحة])
    """
    q_predicted = np.clip(q_predicted, epsilon, 1.0)
    return -np.sum(p_true * np.log(q_predicted))

def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    KL Divergence: تقيس "المسافة" (غير المتماثلة!) بين توزيعين
    احتماليين — كم "بتة إضافية" نُهدرها باستخدام Q بدلًا من P الحقيقي.

        KL(P||Q) = Σ p(x)·log(p(x)/q(x)) = H(P,Q) - H(P)

    تُستخدَم في: قيد RLHF (الفصل 33)، Knowledge Distillation (الفصل 25)،
    وVariational Autoencoders.
    ملاحظة حرجة: KL(P||Q) ≠ KL(Q||P) — ليست مسافة بالمعنى الرياضي الدقيق!
    """
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    return np.sum(p * np.log(p / q))

# مثال تطبيقي متكامل
true_distribution = np.array([1.0, 0.0, 0.0, 0.0])  # الكلمة الصحيحة: الفئة 0 (one-hot)

# نموذج جيد: واثق وصحيح
good_model_pred = np.array([0.85, 0.05, 0.05, 0.05])
# نموذج ضعيف: غير واثق إطلاقًا (توزيع شبه منتظم)
poor_model_pred = np.array([0.30, 0.25, 0.25, 0.20])
# نموذج سيء جدًا: واثق لكن خاطئ
wrong_confident_pred = np.array([0.05, 0.85, 0.05, 0.05])

for name, pred in [("نموذج جيد", good_model_pred),
                    ("نموذج ضعيف", poor_model_pred),
                    ("نموذج واثق وخاطئ", wrong_confident_pred)]:
    ce = cross_entropy(true_distribution, pred)
    print(f"{name}: Cross-Entropy Loss = {ce:.4f}")

# نموذج جيد: Cross-Entropy Loss = 0.1625
# نموذج ضعيف: Cross-Entropy Loss = 1.2040
# نموذج واثق وخاطئ: Cross-Entropy Loss = 2.9957  <- أسوأ عقوبة بكثير!

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-02-ch02-mathematics.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def verify_relationship(p_true, q_pred):
    """
    التحقق العملي من: H(P,Q) = H(P) + KL(P||Q)
    بمعنى: Cross-Entropy = Entropy الذاتية + "الفاقد الإضافي" بسبب
    استخدام توزيع تقريبي Q بدلًا من التوزيع الحقيقي P.
    """
    h_p = entropy(p_true)
    h_pq = cross_entropy(p_true, q_pred)
    kl_pq = kl_divergence(p_true, q_pred)

    print(f"H(P) = {h_p:.4f}")
    print(f"KL(P||Q) = {kl_pq:.4f}")
    print(f"H(P) + KL(P||Q) = {h_p + kl_pq:.4f}")
    print(f"H(P,Q) = {h_pq:.4f}")
    print(f"متطابقان؟ {np.isclose(h_p + kl_pq, h_pq)}")

p = np.array([0.5, 0.3, 0.2])
q = np.array([0.4, 0.4, 0.2])
verify_relationship(p, q)
# H(P) = 1.0297
# KL(P||Q) = 0.0338
# H(P) + KL(P||Q) = 1.0635
# H(P,Q) = 1.0635
# متطابقان؟ True

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-02-ch02-mathematics.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
vocab = ["نحن", "نذهب", "إلى", "المدرسة", "اليوم"]
true_idx = 3
true_dist = np.zeros(5)
true_dist[true_idx] = 1.0

predictions = {
    "تنبؤ ممتاز": np.array([0.02, 0.02, 0.05, 0.88, 0.03]),
    "تنبؤ متوسط": np.array([0.10, 0.10, 0.20, 0.40, 0.20]),
    "تنبؤ سيء": np.array([0.30, 0.30, 0.20, 0.10, 0.10]),
}

for name, pred in predictions.items():
    loss = cross_entropy(true_dist, pred)
    perplexity = np.exp(loss)  # سنشرح Perplexity بالتفصيل في الفصل 5
    print(f"{name}: Loss={loss:.4f}, Perplexity={perplexity:.4f}, "
          f"P(المدرسة)={pred[true_idx]:.2f}")

# تنبؤ ممتاز: Loss=0.1278, Perplexity=1.1363, P(المدرسة)=0.88
# تنبؤ متوسط: Loss=0.9163, Perplexity=2.4999, P(المدرسة)=0.40
# تنبؤ سيء:  Loss=2.3026, Perplexity=10.0000, P(المدرسة)=0.10

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-llms-transformers/notebooks/ch-02-ch02-mathematics.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-02-ch02-mathematics.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الأسس الرياضية للتعلم العميق
Mastering Transformers and Large Language Models — Mathematics

Book slug : mastering-llms-transformers
Chapter   : ch-02-ch02-mathematics (lesson 2)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-01-foundations/ch02-mathematics.ipynb#cell-1
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
# [002] content/notebooks/part-01-foundations/ch02-mathematics.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# === 1. المتجهات والمصفوفات: البنية التحتية لكل شيء ===

v = np.array([1.0, 2.0, 3.0])          # متجه (تمثيل كلمة واحدة، بُعد 3)
M = np.array([[1, 0, 1],
              [0, 1, 1],
              [1, 1, 0]])               # مصفوفة تحويل (3×3)

# الضرب المصفوفي: جوهر كل طبقة خطية (Linear Layer) في أي شبكة عصبية
transformed_v = M @ v
print(f"المتجه بعد التحويل: {transformed_v}")

# === 2. التشابه بين المتجهات: أساس كل آلية Attention ===

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    يقيس زاوية الاتجاه بين متجهين بغض النظر عن حجمهما (magnitude).
    هذا بالضبط ما يُستخدَم لقياس "التشابه الدلالي" بين الكلمات
    في فضاء Embeddings (الفصل 7)، وكأساس نظري لعملية Attention Score
    (التي تستخدم الضرب النقطي كتقريب سريع لهذا التشابه).
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b + 1e-8)

word_king = np.array([0.8, 0.6, 0.1])
word_queen = np.array([0.75, 0.65, 0.15])
word_apple = np.array([0.1, 0.1, 0.9])

print(f"تشابه (ملك، ملكة): {cosine_similarity(word_king, word_queen):.4f}")
print(f"تشابه (ملك، تفاحة): {cosine_similarity(word_king, word_apple):.4f}")
# تشابه (ملك، ملكة): 0.9959   <- قريبان دلاليًا
# تشابه (ملك، تفاحة): 0.3812  <- بعيدان دلاليًا

# === 3. القيم الذاتية والمتجهات الذاتية (Eigenvalues/Eigenvectors) ===
# مفهوم مهم لفهم PCA، وتحليل سلوك المصفوفات في الشبكات العميقة
# (مثل تفسير استقرار/عدم استقرار التدرجات عبر الطبقات)

A = np.array([[4, 1], [2, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"\nالقيم الذاتية: {eigenvalues}")
print(f"المتجهات الذاتية:\n{eigenvectors}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-01-foundations/ch02-mathematics.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def softmax_naive(logits: np.ndarray) -> np.ndarray:
    """
    التطبيق الساذج لـ softmax — يُستخدَم فقط للتوضيح التعليمي،
    لأنه غير مستقر رقميًا (قد يُسبب Overflow مع logits كبيرة).
    """
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)

def softmax_stable(logits: np.ndarray) -> np.ndarray:
    """
    softmax المستقر رقميًا (النسخة المُستخدَمة فعليًا في كل مكتبة
    تعلّم عميق). الحيلة: طرح أقصى قيمة قبل الأس (exp) لا تُغيّر الناتج
    رياضيًا (لأن softmax(x) = softmax(x - c) لأي ثابت c) لكنها تمنع
    overflow لأن أكبر قيمة أس تُصبح exp(0)=1 بدلًا من قيمة ضخمة محتملة.
    """
    shifted_logits = logits - np.max(logits)
    exp_logits = np.exp(shifted_logits)
    return exp_logits / np.sum(exp_logits)

# توضيح المشكلة عمليًا
large_logits = np.array([1000.0, 1001.0, 1002.0])

try:
    result_naive = softmax_naive(large_logits)
    print(f"Naive softmax: {result_naive}")
except Exception as e:
    print(f"❌ Naive softmax فشل: {e}")
# ينتج: [nan, nan, nan] بسبب exp(1000) = inf ثم inf/inf = nan

result_stable = softmax_stable(large_logits)
print(f"✅ Stable softmax: {result_stable}")
# ينتج: [0.0900, 0.2447, 0.6652]  — نتيجة صحيحة ومستقرة

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-01-foundations/ch02-mathematics.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def entropy(p: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    الإنتروبيا (Shannon Entropy): تقيس "عدم اليقين" المتأصل في توزيع
    احتمالي. كلما كانت الإنتروبيا أعلى، كان التوزيع أكثر "انتشارًا"
    (uniform)، وكلما كانت أقل، كان أكثر "تركّزًا" حول قيمة واحدة.

        H(P) = -Σ p(x)·log(p(x))
    """
    p = np.clip(p, epsilon, 1.0)  # تجنّب log(0)
    return -np.sum(p * np.log(p))

def cross_entropy(p_true: np.ndarray, q_predicted: np.ndarray,
                   epsilon: float = 1e-10) -> float:
    """
    Cross-Entropy: تقيس "متوسط عدد البتات" اللازمة لترميز أحداث
    مسحوبة من التوزيع الحقيقي P، باستخدام كود مُصمَّم بناءً على
    التوزيع المُتوقَّع Q. هذه هي دالة الخسارة الأساسية لكل نموذج لغوي:

        H(P, Q) = -Σ p_true(x)·log(q_predicted(x))

    في التصنيف/توليد اللغة، p_true عادة one-hot (0 أو 1 فقط)،
    فتُختزَل المعادلة إلى: -log(q_predicted[الفئة_الصحيحة])
    """
    q_predicted = np.clip(q_predicted, epsilon, 1.0)
    return -np.sum(p_true * np.log(q_predicted))

def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    KL Divergence: تقيس "المسافة" (غير المتماثلة!) بين توزيعين
    احتماليين — كم "بتة إضافية" نُهدرها باستخدام Q بدلًا من P الحقيقي.

        KL(P||Q) = Σ p(x)·log(p(x)/q(x)) = H(P,Q) - H(P)

    تُستخدَم في: قيد RLHF (الفصل 33)، Knowledge Distillation (الفصل 25)،
    وVariational Autoencoders.
    ملاحظة حرجة: KL(P||Q) ≠ KL(Q||P) — ليست مسافة بالمعنى الرياضي الدقيق!
    """
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    return np.sum(p * np.log(p / q))

# مثال تطبيقي متكامل
true_distribution = np.array([1.0, 0.0, 0.0, 0.0])  # الكلمة الصحيحة: الفئة 0 (one-hot)

# نموذج جيد: واثق وصحيح
good_model_pred = np.array([0.85, 0.05, 0.05, 0.05])
# نموذج ضعيف: غير واثق إطلاقًا (توزيع شبه منتظم)
poor_model_pred = np.array([0.30, 0.25, 0.25, 0.20])
# نموذج سيء جدًا: واثق لكن خاطئ
wrong_confident_pred = np.array([0.05, 0.85, 0.05, 0.05])

for name, pred in [("نموذج جيد", good_model_pred),
                    ("نموذج ضعيف", poor_model_pred),
                    ("نموذج واثق وخاطئ", wrong_confident_pred)]:
    ce = cross_entropy(true_distribution, pred)
    print(f"{name}: Cross-Entropy Loss = {ce:.4f}")

# نموذج جيد: Cross-Entropy Loss = 0.1625
# نموذج ضعيف: Cross-Entropy Loss = 1.2040
# نموذج واثق وخاطئ: Cross-Entropy Loss = 2.9957  <- أسوأ عقوبة بكثير!

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-01-foundations/ch02-mathematics.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def verify_relationship(p_true, q_pred):
    """
    التحقق العملي من: H(P,Q) = H(P) + KL(P||Q)
    بمعنى: Cross-Entropy = Entropy الذاتية + "الفاقد الإضافي" بسبب
    استخدام توزيع تقريبي Q بدلًا من التوزيع الحقيقي P.
    """
    h_p = entropy(p_true)
    h_pq = cross_entropy(p_true, q_pred)
    kl_pq = kl_divergence(p_true, q_pred)

    print(f"H(P) = {h_p:.4f}")
    print(f"KL(P||Q) = {kl_pq:.4f}")
    print(f"H(P) + KL(P||Q) = {h_p + kl_pq:.4f}")
    print(f"H(P,Q) = {h_pq:.4f}")
    print(f"متطابقان؟ {np.isclose(h_p + kl_pq, h_pq)}")

p = np.array([0.5, 0.3, 0.2])
q = np.array([0.4, 0.4, 0.2])
verify_relationship(p, q)
# H(P) = 1.0297
# KL(P||Q) = 0.0338
# H(P) + KL(P||Q) = 1.0635
# H(P,Q) = 1.0635
# متطابقان؟ True

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-01-foundations/ch02-mathematics.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
vocab = ["نحن", "نذهب", "إلى", "المدرسة", "اليوم"]
true_idx = 3
true_dist = np.zeros(5)
true_dist[true_idx] = 1.0

predictions = {
    "تنبؤ ممتاز": np.array([0.02, 0.02, 0.05, 0.88, 0.03]),
    "تنبؤ متوسط": np.array([0.10, 0.10, 0.20, 0.40, 0.20]),
    "تنبؤ سيء": np.array([0.30, 0.30, 0.20, 0.10, 0.10]),
}

for name, pred in predictions.items():
    loss = cross_entropy(true_dist, pred)
    perplexity = np.exp(loss)  # سنشرح Perplexity بالتفصيل في الفصل 5
    print(f"{name}: Loss={loss:.4f}, Perplexity={perplexity:.4f}, "
          f"P(المدرسة)={pred[true_idx]:.2f}")

# تنبؤ ممتاز: Loss=0.1278, Perplexity=1.1363, P(المدرسة)=0.88
# تنبؤ متوسط: Loss=0.9163, Perplexity=2.4999, P(المدرسة)=0.40
# تنبؤ سيء:  Loss=2.3026, Perplexity=10.0000, P(المدرسة)=0.10

# ────────────────────────────────────────────────────────────────────────
# [007] content/notebooks/part-01-foundations/ch02-mathematics.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-02-ch02-mathematics.ipynb#cell-2
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-02-ch02-mathematics.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# === 1. المتجهات والمصفوفات: البنية التحتية لكل شيء ===

v = np.array([1.0, 2.0, 3.0])          # متجه (تمثيل كلمة واحدة، بُعد 3)
M = np.array([[1, 0, 1],
              [0, 1, 1],
              [1, 1, 0]])               # مصفوفة تحويل (3×3)

# الضرب المصفوفي: جوهر كل طبقة خطية (Linear Layer) في أي شبكة عصبية
transformed_v = M @ v
print(f"المتجه بعد التحويل: {transformed_v}")

# === 2. التشابه بين المتجهات: أساس كل آلية Attention ===

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    يقيس زاوية الاتجاه بين متجهين بغض النظر عن حجمهما (magnitude).
    هذا بالضبط ما يُستخدَم لقياس "التشابه الدلالي" بين الكلمات
    في فضاء Embeddings (الفصل 7)، وكأساس نظري لعملية Attention Score
    (التي تستخدم الضرب النقطي كتقريب سريع لهذا التشابه).
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b + 1e-8)

word_king = np.array([0.8, 0.6, 0.1])
word_queen = np.array([0.75, 0.65, 0.15])
word_apple = np.array([0.1, 0.1, 0.9])

print(f"تشابه (ملك، ملكة): {cosine_similarity(word_king, word_queen):.4f}")
print(f"تشابه (ملك، تفاحة): {cosine_similarity(word_king, word_apple):.4f}")
# تشابه (ملك، ملكة): 0.9959   <- قريبان دلاليًا
# تشابه (ملك، تفاحة): 0.3812  <- بعيدان دلاليًا

# === 3. القيم الذاتية والمتجهات الذاتية (Eigenvalues/Eigenvectors) ===
# مفهوم مهم لفهم PCA، وتحليل سلوك المصفوفات في الشبكات العميقة
# (مثل تفسير استقرار/عدم استقرار التدرجات عبر الطبقات)

A = np.array([[4, 1], [2, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"\nالقيم الذاتية: {eigenvalues}")
print(f"المتجهات الذاتية:\n{eigenvectors}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-02-ch02-mathematics.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def softmax_naive(logits: np.ndarray) -> np.ndarray:
    """
    التطبيق الساذج لـ softmax — يُستخدَم فقط للتوضيح التعليمي،
    لأنه غير مستقر رقميًا (قد يُسبب Overflow مع logits كبيرة).
    """
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)

def softmax_stable(logits: np.ndarray) -> np.ndarray:
    """
    softmax المستقر رقميًا (النسخة المُستخدَمة فعليًا في كل مكتبة
    تعلّم عميق). الحيلة: طرح أقصى قيمة قبل الأس (exp) لا تُغيّر الناتج
    رياضيًا (لأن softmax(x) = softmax(x - c) لأي ثابت c) لكنها تمنع
    overflow لأن أكبر قيمة أس تُصبح exp(0)=1 بدلًا من قيمة ضخمة محتملة.
    """
    shifted_logits = logits - np.max(logits)
    exp_logits = np.exp(shifted_logits)
    return exp_logits / np.sum(exp_logits)

# توضيح المشكلة عمليًا
large_logits = np.array([1000.0, 1001.0, 1002.0])

try:
    result_naive = softmax_naive(large_logits)
    print(f"Naive softmax: {result_naive}")
except Exception as e:
    print(f"❌ Naive softmax فشل: {e}")
# ينتج: [nan, nan, nan] بسبب exp(1000) = inf ثم inf/inf = nan

result_stable = softmax_stable(large_logits)
print(f"✅ Stable softmax: {result_stable}")
# ينتج: [0.0900, 0.2447, 0.6652]  — نتيجة صحيحة ومستقرة

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-02-ch02-mathematics.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def entropy(p: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    الإنتروبيا (Shannon Entropy): تقيس "عدم اليقين" المتأصل في توزيع
    احتمالي. كلما كانت الإنتروبيا أعلى، كان التوزيع أكثر "انتشارًا"
    (uniform)، وكلما كانت أقل، كان أكثر "تركّزًا" حول قيمة واحدة.

        H(P) = -Σ p(x)·log(p(x))
    """
    p = np.clip(p, epsilon, 1.0)  # تجنّب log(0)
    return -np.sum(p * np.log(p))

def cross_entropy(p_true: np.ndarray, q_predicted: np.ndarray,
                   epsilon: float = 1e-10) -> float:
    """
    Cross-Entropy: تقيس "متوسط عدد البتات" اللازمة لترميز أحداث
    مسحوبة من التوزيع الحقيقي P، باستخدام كود مُصمَّم بناءً على
    التوزيع المُتوقَّع Q. هذه هي دالة الخسارة الأساسية لكل نموذج لغوي:

        H(P, Q) = -Σ p_true(x)·log(q_predicted(x))

    في التصنيف/توليد اللغة، p_true عادة one-hot (0 أو 1 فقط)،
    فتُختزَل المعادلة إلى: -log(q_predicted[الفئة_الصحيحة])
    """
    q_predicted = np.clip(q_predicted, epsilon, 1.0)
    return -np.sum(p_true * np.log(q_predicted))

def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    KL Divergence: تقيس "المسافة" (غير المتماثلة!) بين توزيعين
    احتماليين — كم "بتة إضافية" نُهدرها باستخدام Q بدلًا من P الحقيقي.

        KL(P||Q) = Σ p(x)·log(p(x)/q(x)) = H(P,Q) - H(P)

    تُستخدَم في: قيد RLHF (الفصل 33)، Knowledge Distillation (الفصل 25)،
    وVariational Autoencoders.
    ملاحظة حرجة: KL(P||Q) ≠ KL(Q||P) — ليست مسافة بالمعنى الرياضي الدقيق!
    """
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    return np.sum(p * np.log(p / q))

# مثال تطبيقي متكامل
true_distribution = np.array([1.0, 0.0, 0.0, 0.0])  # الكلمة الصحيحة: الفئة 0 (one-hot)

# نموذج جيد: واثق وصحيح
good_model_pred = np.array([0.85, 0.05, 0.05, 0.05])
# نموذج ضعيف: غير واثق إطلاقًا (توزيع شبه منتظم)
poor_model_pred = np.array([0.30, 0.25, 0.25, 0.20])
# نموذج سيء جدًا: واثق لكن خاطئ
wrong_confident_pred = np.array([0.05, 0.85, 0.05, 0.05])

for name, pred in [("نموذج جيد", good_model_pred),
                    ("نموذج ضعيف", poor_model_pred),
                    ("نموذج واثق وخاطئ", wrong_confident_pred)]:
    ce = cross_entropy(true_distribution, pred)
    print(f"{name}: Cross-Entropy Loss = {ce:.4f}")

# نموذج جيد: Cross-Entropy Loss = 0.1625
# نموذج ضعيف: Cross-Entropy Loss = 1.2040
# نموذج واثق وخاطئ: Cross-Entropy Loss = 2.9957  <- أسوأ عقوبة بكثير!

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-02-ch02-mathematics.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def verify_relationship(p_true, q_pred):
    """
    التحقق العملي من: H(P,Q) = H(P) + KL(P||Q)
    بمعنى: Cross-Entropy = Entropy الذاتية + "الفاقد الإضافي" بسبب
    استخدام توزيع تقريبي Q بدلًا من التوزيع الحقيقي P.
    """
    h_p = entropy(p_true)
    h_pq = cross_entropy(p_true, q_pred)
    kl_pq = kl_divergence(p_true, q_pred)

    print(f"H(P) = {h_p:.4f}")
    print(f"KL(P||Q) = {kl_pq:.4f}")
    print(f"H(P) + KL(P||Q) = {h_p + kl_pq:.4f}")
    print(f"H(P,Q) = {h_pq:.4f}")
    print(f"متطابقان؟ {np.isclose(h_p + kl_pq, h_pq)}")

p = np.array([0.5, 0.3, 0.2])
q = np.array([0.4, 0.4, 0.2])
verify_relationship(p, q)
# H(P) = 1.0297
# KL(P||Q) = 0.0338
# H(P) + KL(P||Q) = 1.0635
# H(P,Q) = 1.0635
# متطابقان؟ True

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-llms-transformers/notebooks/ch-02-ch02-mathematics.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
vocab = ["نحن", "نذهب", "إلى", "المدرسة", "اليوم"]
true_idx = 3
true_dist = np.zeros(5)
true_dist[true_idx] = 1.0

predictions = {
    "تنبؤ ممتاز": np.array([0.02, 0.02, 0.05, 0.88, 0.03]),
    "تنبؤ متوسط": np.array([0.10, 0.10, 0.20, 0.40, 0.20]),
    "تنبؤ سيء": np.array([0.30, 0.30, 0.20, 0.10, 0.10]),
}

for name, pred in predictions.items():
    loss = cross_entropy(true_dist, pred)
    perplexity = np.exp(loss)  # سنشرح Perplexity بالتفصيل في الفصل 5
    print(f"{name}: Loss={loss:.4f}, Perplexity={perplexity:.4f}, "
          f"P(المدرسة)={pred[true_idx]:.2f}")

# تنبؤ ممتاز: Loss=0.1278, Perplexity=1.1363, P(المدرسة)=0.88
# تنبؤ متوسط: Loss=0.9163, Perplexity=2.4999, P(المدرسة)=0.40
# تنبؤ سيء:  Loss=2.3026, Perplexity=10.0000, P(المدرسة)=0.10

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-llms-transformers/notebooks/ch-02-ch02-mathematics.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
