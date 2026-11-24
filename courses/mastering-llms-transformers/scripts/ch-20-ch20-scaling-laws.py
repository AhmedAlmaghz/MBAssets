#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — قوانين التوسّع: Chinchilla
Mastering Transformers and Large Language Models — Scaling Laws

Book slug : mastering-llms-transformers
Chapter   : ch-20-ch20-scaling-laws (lesson 20)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-20-ch20-scaling-laws.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — قوانين التوسّع: Chinchilla
Mastering Transformers and Large Language Models — Scaling Laws

Book slug : mastering-llms-transformers
Chapter   : ch-20-ch20-scaling-laws (lesson 20)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-20-ch20-scaling-laws.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — قوانين التوسّع: Chinchilla
Mastering Transformers and Large Language Models — Scaling Laws

Book slug : mastering-llms-transformers
Chapter   : ch-20-ch20-scaling-laws (lesson 20)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-05-building-llms/ch20-scaling-laws.ipynb#cell-1
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
    ("numpy", "numpy"),
    ("scipy", "scipy")
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
# [002] content/notebooks/part-05-building-llms/ch20-scaling-laws.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from scipy.optimize import minimize

class ChinchillaScalingLaw:
    """
    تطبيق عملي لمعادلة Chinchilla لتقدير الخسارة المتوقعة لأي تركيبة
    (N, D)، وإيجاد التخصيص الأمثل لأي ميزانية حوسبة معطاة.

    الثوابت أدناه مأخوذة تقريبيًا من القيم المُقدَّرة في الورقة الأصلية
    (Hoffmann et al. 2022، الجدول A.3 - تقديرات عبر Approach 3).
    """
    def __init__(self):
        # ثوابت مُقدَّرة تجريبيًا من الورقة الأصلية (بوحدات مناسبة)
        self.E = 1.69       # إنتروبيا اللغة الطبيعية التقريبية (حد أدنى للخسارة)
        self.A = 406.4      # ثابت حد النموذج
        self.alpha = 0.34   # أس حد النموذج
        self.B = 410.7      # ثابت حد البيانات
        self.beta = 0.28    # أس حد البيانات

    def predict_loss(self, N: float, D: float) -> float:
        """
        التنبؤ بالخسارة النهائية (Cross-Entropy بالـ nats) لنموذج بحجم
        N معامل مُدرَّب على D رمز، وفق معادلة Chinchilla الإمبريقية.
        """
        return self.E + (self.A / N ** self.alpha) + (self.B / D ** self.beta)

    def compute_flops(self, N: float, D: float) -> float:
        """حساب ميزانية الحوسبة التقريبية باستخدام القاعدة C ≈ 6ND"""
        return 6 * N * D

    def find_optimal_allocation(self, compute_budget: float) -> dict:
        """
        إيجاد (N_opt, D_opt) التي تُدنّي الخسارة L(N,D) تحت قيد
        ميزانية حوسبة ثابتة C = 6ND، عبر تحسين عددي مباشر
        (بديل عملي عن الحل التحليلي بمضاعفات لاغرانج).
        """
        def objective(log_N):
            # نعمل في الفضاء اللوغاريتمي لتحسين استقرار التحسين العددي
            N = np.exp(log_N[0])
            D = compute_budget / (6 * N)  # اشتقاق D من قيد الميزانية مباشرة
            if D <= 0:
                return 1e10
            return self.predict_loss(N, D)

        # تخمين أولي معقول (نبدأ من افتراض D=20N كنقطة انطلاق)
        initial_N_guess = (compute_budget / (6 * 20)) ** 0.5
        result = minimize(objective, x0=[np.log(initial_N_guess)],
                           method="Nelder-Mead")

        N_opt = np.exp(result.x[0])
        D_opt = compute_budget / (6 * N_opt)
        loss_opt = self.predict_loss(N_opt, D_opt)

        return {
            "N_opt": N_opt,
            "D_opt": D_opt,
            "D_to_N_ratio": D_opt / N_opt,
            "predicted_loss": loss_opt,
            "compute_budget": compute_budget,
        }

def demonstrate_chinchilla_tradeoff():
    """
    توضيح عملي: لنفس ميزانية الحوسبة، ماذا يحدث للخسارة إذا انحرفنا
    عن التخصيص الأمثل (نموذج كبير جدًا+بيانات قليلة، أو العكس)؟
    """
    scaling_law = ChinchillaScalingLaw()
    compute_budget = 1e21  # ميزانية حوسبة ثابتة (FLOPs) مشابهة لتجربة متوسطة

    # الحل الأمثل وفق التحسين
    optimal = scaling_law.find_optimal_allocation(compute_budget)
    print("=== التخصيص الأمثل (Compute-Optimal) ===")
    print(f"N_opt (عدد المعاملات): {optimal['N_opt']:.3e}")
    print(f"D_opt (عدد الرموز): {optimal['D_opt']:.3e}")
    print(f"النسبة D/N: {optimal['D_to_N_ratio']:.2f}  (قريبة من 20 كما تتوقع Chinchilla)")
    print(f"الخسارة المتوقعة: {optimal['predicted_loss']:.4f}\n")

    # سيناريوهات بديلة بنفس ميزانية الحوسبة بالضبط لكن بتوزيع مختلف
    print("=== مقارنة سيناريوهات بديلة بنفس ميزانية الحوسبة ===")
    scenarios = {
        "نموذج ضخم جدًا + بيانات قليلة (مثل الاتجاه القديم لـ GPT-3)": 5.0,
        "التخصيص الأمثل (Chinchilla-style)": 1.0,
        "نموذج صغير + بيانات هائلة جدًا": 0.2,
    }

    for name, multiplier in scenarios.items():
        N = optimal["N_opt"] * (multiplier ** 0.5)
        D = compute_budget / (6 * N)
        loss = scaling_law.predict_loss(N, D)
        print(f"\n{name}:")
        print(f"   N = {N:.3e}, D = {D:.3e}, D/N = {D/N:.1f}")
        print(f"   الخسارة المتوقعة = {loss:.4f} "
              f"({'أفضل' if loss < optimal['predicted_loss'] else 'أسوأ'} "
              f"من الأمثل بفارق {abs(loss - optimal['predicted_loss']):.4f})")

demonstrate_chinchilla_tradeoff()

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-05-building-llms/ch20-scaling-laws.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def compare_training_optimal_vs_inference_optimal():
    """
    توضيح الفارق بين استراتيجية Chinchilla الكلاسيكية (الأمثل للتدريب فقط)
    واستراتيجية "over-training" الحديثة (الأمثل مع اعتبار تكلفة الاستدلال)
    كما طبّقتها Meta فعليًا في سلسلة LLaMA.
    """
    scaling_law = ChinchillaScalingLaw()

    scenarios = [
        {"name": "Chinchilla الأمثل نظريًا لـ N=7B",
         "N": 7e9, "D": 7e9 * 20},
        {"name": "LLaMA-2 7B الفعلي (over-trained عمدًا)",
         "N": 7e9, "D": 2e12},
    ]

    print(f"{'السيناريو':<45}{'N':<10}{'D':<12}{'D/N':<8}{'Loss المتوقع':<15}")
    print("-" * 90)
    for s in scenarios:
        loss = scaling_law.predict_loss(s["N"], s["D"])
        compute = scaling_law.compute_flops(s["N"], s["D"])
        print(f"{s['name']:<45}{s['N']:.1e}  {s['D']:.1e}  "
              f"{s['D']/s['N']:<8.1f}{loss:<15.4f}")
        print(f"   → ميزانية الحوسبة المطلوبة: {compute:.2e} FLOPs")

compare_training_optimal_vs_inference_optimal()

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-05-building-llms/ch20-scaling-laws.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def solve_chinchilla_example_manually(compute_budget: float,
                                        d_to_n_ratio: float = 20.0) -> dict:
    """
    حل تحليلي مباشر (وليس تحسينًا عدديًا) لإيجاد N وD المثاليين
    بافتراض نسبة D/N معينة (عادة 20 وفق قاعدة Chinchilla التقريبية)
    وقيد ميزانية حوسبة C = 6ND.

    الاشتقاق: C = 6ND = 6N(ratio×N) = 6×ratio×N²
    إذن: N = sqrt(C / (6×ratio))
    """
    N_opt = (compute_budget / (6 * d_to_n_ratio)) ** 0.5
    D_opt = d_to_n_ratio * N_opt
    verification_C = 6 * N_opt * D_opt

    return {
        "N_opt": N_opt,
        "D_opt": D_opt,
        "verification_compute": verification_C,
        "matches_budget": abs(verification_C - compute_budget) / compute_budget < 0.01,
    }

result = solve_chinchilla_example_manually(compute_budget=1e20)
print(f"N الأمثل: {result['N_opt']:.3e} معامل ({result['N_opt']/1e6:.0f} مليون)")
print(f"D الأمثل: {result['D_opt']:.3e} رمز ({result['D_opt']/1e9:.1f} مليار)")
print(f"التحقق من ميزانية الحوسبة: {result['verification_compute']:.3e} "
      f"(مطابق: {result['matches_budget']})")

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-05-building-llms/ch20-scaling-laws.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-20-ch20-scaling-laws.ipynb#cell-1
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
    ("numpy", "numpy"),
    ("scipy", "scipy")
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-20-ch20-scaling-laws.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from scipy.optimize import minimize

class ChinchillaScalingLaw:
    """
    تطبيق عملي لمعادلة Chinchilla لتقدير الخسارة المتوقعة لأي تركيبة
    (N, D)، وإيجاد التخصيص الأمثل لأي ميزانية حوسبة معطاة.

    الثوابت أدناه مأخوذة تقريبيًا من القيم المُقدَّرة في الورقة الأصلية
    (Hoffmann et al. 2022، الجدول A.3 - تقديرات عبر Approach 3).
    """
    def __init__(self):
        # ثوابت مُقدَّرة تجريبيًا من الورقة الأصلية (بوحدات مناسبة)
        self.E = 1.69       # إنتروبيا اللغة الطبيعية التقريبية (حد أدنى للخسارة)
        self.A = 406.4      # ثابت حد النموذج
        self.alpha = 0.34   # أس حد النموذج
        self.B = 410.7      # ثابت حد البيانات
        self.beta = 0.28    # أس حد البيانات

    def predict_loss(self, N: float, D: float) -> float:
        """
        التنبؤ بالخسارة النهائية (Cross-Entropy بالـ nats) لنموذج بحجم
        N معامل مُدرَّب على D رمز، وفق معادلة Chinchilla الإمبريقية.
        """
        return self.E + (self.A / N ** self.alpha) + (self.B / D ** self.beta)

    def compute_flops(self, N: float, D: float) -> float:
        """حساب ميزانية الحوسبة التقريبية باستخدام القاعدة C ≈ 6ND"""
        return 6 * N * D

    def find_optimal_allocation(self, compute_budget: float) -> dict:
        """
        إيجاد (N_opt, D_opt) التي تُدنّي الخسارة L(N,D) تحت قيد
        ميزانية حوسبة ثابتة C = 6ND، عبر تحسين عددي مباشر
        (بديل عملي عن الحل التحليلي بمضاعفات لاغرانج).
        """
        def objective(log_N):
            # نعمل في الفضاء اللوغاريتمي لتحسين استقرار التحسين العددي
            N = np.exp(log_N[0])
            D = compute_budget / (6 * N)  # اشتقاق D من قيد الميزانية مباشرة
            if D <= 0:
                return 1e10
            return self.predict_loss(N, D)

        # تخمين أولي معقول (نبدأ من افتراض D=20N كنقطة انطلاق)
        initial_N_guess = (compute_budget / (6 * 20)) ** 0.5
        result = minimize(objective, x0=[np.log(initial_N_guess)],
                           method="Nelder-Mead")

        N_opt = np.exp(result.x[0])
        D_opt = compute_budget / (6 * N_opt)
        loss_opt = self.predict_loss(N_opt, D_opt)

        return {
            "N_opt": N_opt,
            "D_opt": D_opt,
            "D_to_N_ratio": D_opt / N_opt,
            "predicted_loss": loss_opt,
            "compute_budget": compute_budget,
        }

def demonstrate_chinchilla_tradeoff():
    """
    توضيح عملي: لنفس ميزانية الحوسبة، ماذا يحدث للخسارة إذا انحرفنا
    عن التخصيص الأمثل (نموذج كبير جدًا+بيانات قليلة، أو العكس)؟
    """
    scaling_law = ChinchillaScalingLaw()
    compute_budget = 1e21  # ميزانية حوسبة ثابتة (FLOPs) مشابهة لتجربة متوسطة

    # الحل الأمثل وفق التحسين
    optimal = scaling_law.find_optimal_allocation(compute_budget)
    print("=== التخصيص الأمثل (Compute-Optimal) ===")
    print(f"N_opt (عدد المعاملات): {optimal['N_opt']:.3e}")
    print(f"D_opt (عدد الرموز): {optimal['D_opt']:.3e}")
    print(f"النسبة D/N: {optimal['D_to_N_ratio']:.2f}  (قريبة من 20 كما تتوقع Chinchilla)")
    print(f"الخسارة المتوقعة: {optimal['predicted_loss']:.4f}\n")

    # سيناريوهات بديلة بنفس ميزانية الحوسبة بالضبط لكن بتوزيع مختلف
    print("=== مقارنة سيناريوهات بديلة بنفس ميزانية الحوسبة ===")
    scenarios = {
        "نموذج ضخم جدًا + بيانات قليلة (مثل الاتجاه القديم لـ GPT-3)": 5.0,
        "التخصيص الأمثل (Chinchilla-style)": 1.0,
        "نموذج صغير + بيانات هائلة جدًا": 0.2,
    }

    for name, multiplier in scenarios.items():
        N = optimal["N_opt"] * (multiplier ** 0.5)
        D = compute_budget / (6 * N)
        loss = scaling_law.predict_loss(N, D)
        print(f"\n{name}:")
        print(f"   N = {N:.3e}, D = {D:.3e}, D/N = {D/N:.1f}")
        print(f"   الخسارة المتوقعة = {loss:.4f} "
              f"({'أفضل' if loss < optimal['predicted_loss'] else 'أسوأ'} "
              f"من الأمثل بفارق {abs(loss - optimal['predicted_loss']):.4f})")

demonstrate_chinchilla_tradeoff()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-20-ch20-scaling-laws.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def compare_training_optimal_vs_inference_optimal():
    """
    توضيح الفارق بين استراتيجية Chinchilla الكلاسيكية (الأمثل للتدريب فقط)
    واستراتيجية "over-training" الحديثة (الأمثل مع اعتبار تكلفة الاستدلال)
    كما طبّقتها Meta فعليًا في سلسلة LLaMA.
    """
    scaling_law = ChinchillaScalingLaw()

    scenarios = [
        {"name": "Chinchilla الأمثل نظريًا لـ N=7B",
         "N": 7e9, "D": 7e9 * 20},
        {"name": "LLaMA-2 7B الفعلي (over-trained عمدًا)",
         "N": 7e9, "D": 2e12},
    ]

    print(f"{'السيناريو':<45}{'N':<10}{'D':<12}{'D/N':<8}{'Loss المتوقع':<15}")
    print("-" * 90)
    for s in scenarios:
        loss = scaling_law.predict_loss(s["N"], s["D"])
        compute = scaling_law.compute_flops(s["N"], s["D"])
        print(f"{s['name']:<45}{s['N']:.1e}  {s['D']:.1e}  "
              f"{s['D']/s['N']:<8.1f}{loss:<15.4f}")
        print(f"   → ميزانية الحوسبة المطلوبة: {compute:.2e} FLOPs")

compare_training_optimal_vs_inference_optimal()

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-20-ch20-scaling-laws.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def solve_chinchilla_example_manually(compute_budget: float,
                                        d_to_n_ratio: float = 20.0) -> dict:
    """
    حل تحليلي مباشر (وليس تحسينًا عدديًا) لإيجاد N وD المثاليين
    بافتراض نسبة D/N معينة (عادة 20 وفق قاعدة Chinchilla التقريبية)
    وقيد ميزانية حوسبة C = 6ND.

    الاشتقاق: C = 6ND = 6N(ratio×N) = 6×ratio×N²
    إذن: N = sqrt(C / (6×ratio))
    """
    N_opt = (compute_budget / (6 * d_to_n_ratio)) ** 0.5
    D_opt = d_to_n_ratio * N_opt
    verification_C = 6 * N_opt * D_opt

    return {
        "N_opt": N_opt,
        "D_opt": D_opt,
        "verification_compute": verification_C,
        "matches_budget": abs(verification_C - compute_budget) / compute_budget < 0.01,
    }

result = solve_chinchilla_example_manually(compute_budget=1e20)
print(f"N الأمثل: {result['N_opt']:.3e} معامل ({result['N_opt']/1e6:.0f} مليون)")
print(f"D الأمثل: {result['D_opt']:.3e} رمز ({result['D_opt']/1e9:.1f} مليار)")
print(f"التحقق من ميزانية الحوسبة: {result['verification_compute']:.3e} "
      f"(مطابق: {result['matches_budget']})")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-20-ch20-scaling-laws.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-20-ch20-scaling-laws.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — قوانين التوسّع: Chinchilla
Mastering Transformers and Large Language Models — Scaling Laws

Book slug : mastering-llms-transformers
Chapter   : ch-20-ch20-scaling-laws (lesson 20)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-05-building-llms/ch20-scaling-laws.ipynb#cell-1
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
    ("numpy", "numpy"),
    ("scipy", "scipy")
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
# [002] content/notebooks/part-05-building-llms/ch20-scaling-laws.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from scipy.optimize import minimize

class ChinchillaScalingLaw:
    """
    تطبيق عملي لمعادلة Chinchilla لتقدير الخسارة المتوقعة لأي تركيبة
    (N, D)، وإيجاد التخصيص الأمثل لأي ميزانية حوسبة معطاة.

    الثوابت أدناه مأخوذة تقريبيًا من القيم المُقدَّرة في الورقة الأصلية
    (Hoffmann et al. 2022، الجدول A.3 - تقديرات عبر Approach 3).
    """
    def __init__(self):
        # ثوابت مُقدَّرة تجريبيًا من الورقة الأصلية (بوحدات مناسبة)
        self.E = 1.69       # إنتروبيا اللغة الطبيعية التقريبية (حد أدنى للخسارة)
        self.A = 406.4      # ثابت حد النموذج
        self.alpha = 0.34   # أس حد النموذج
        self.B = 410.7      # ثابت حد البيانات
        self.beta = 0.28    # أس حد البيانات

    def predict_loss(self, N: float, D: float) -> float:
        """
        التنبؤ بالخسارة النهائية (Cross-Entropy بالـ nats) لنموذج بحجم
        N معامل مُدرَّب على D رمز، وفق معادلة Chinchilla الإمبريقية.
        """
        return self.E + (self.A / N ** self.alpha) + (self.B / D ** self.beta)

    def compute_flops(self, N: float, D: float) -> float:
        """حساب ميزانية الحوسبة التقريبية باستخدام القاعدة C ≈ 6ND"""
        return 6 * N * D

    def find_optimal_allocation(self, compute_budget: float) -> dict:
        """
        إيجاد (N_opt, D_opt) التي تُدنّي الخسارة L(N,D) تحت قيد
        ميزانية حوسبة ثابتة C = 6ND، عبر تحسين عددي مباشر
        (بديل عملي عن الحل التحليلي بمضاعفات لاغرانج).
        """
        def objective(log_N):
            # نعمل في الفضاء اللوغاريتمي لتحسين استقرار التحسين العددي
            N = np.exp(log_N[0])
            D = compute_budget / (6 * N)  # اشتقاق D من قيد الميزانية مباشرة
            if D <= 0:
                return 1e10
            return self.predict_loss(N, D)

        # تخمين أولي معقول (نبدأ من افتراض D=20N كنقطة انطلاق)
        initial_N_guess = (compute_budget / (6 * 20)) ** 0.5
        result = minimize(objective, x0=[np.log(initial_N_guess)],
                           method="Nelder-Mead")

        N_opt = np.exp(result.x[0])
        D_opt = compute_budget / (6 * N_opt)
        loss_opt = self.predict_loss(N_opt, D_opt)

        return {
            "N_opt": N_opt,
            "D_opt": D_opt,
            "D_to_N_ratio": D_opt / N_opt,
            "predicted_loss": loss_opt,
            "compute_budget": compute_budget,
        }

def demonstrate_chinchilla_tradeoff():
    """
    توضيح عملي: لنفس ميزانية الحوسبة، ماذا يحدث للخسارة إذا انحرفنا
    عن التخصيص الأمثل (نموذج كبير جدًا+بيانات قليلة، أو العكس)؟
    """
    scaling_law = ChinchillaScalingLaw()
    compute_budget = 1e21  # ميزانية حوسبة ثابتة (FLOPs) مشابهة لتجربة متوسطة

    # الحل الأمثل وفق التحسين
    optimal = scaling_law.find_optimal_allocation(compute_budget)
    print("=== التخصيص الأمثل (Compute-Optimal) ===")
    print(f"N_opt (عدد المعاملات): {optimal['N_opt']:.3e}")
    print(f"D_opt (عدد الرموز): {optimal['D_opt']:.3e}")
    print(f"النسبة D/N: {optimal['D_to_N_ratio']:.2f}  (قريبة من 20 كما تتوقع Chinchilla)")
    print(f"الخسارة المتوقعة: {optimal['predicted_loss']:.4f}\n")

    # سيناريوهات بديلة بنفس ميزانية الحوسبة بالضبط لكن بتوزيع مختلف
    print("=== مقارنة سيناريوهات بديلة بنفس ميزانية الحوسبة ===")
    scenarios = {
        "نموذج ضخم جدًا + بيانات قليلة (مثل الاتجاه القديم لـ GPT-3)": 5.0,
        "التخصيص الأمثل (Chinchilla-style)": 1.0,
        "نموذج صغير + بيانات هائلة جدًا": 0.2,
    }

    for name, multiplier in scenarios.items():
        N = optimal["N_opt"] * (multiplier ** 0.5)
        D = compute_budget / (6 * N)
        loss = scaling_law.predict_loss(N, D)
        print(f"\n{name}:")
        print(f"   N = {N:.3e}, D = {D:.3e}, D/N = {D/N:.1f}")
        print(f"   الخسارة المتوقعة = {loss:.4f} "
              f"({'أفضل' if loss < optimal['predicted_loss'] else 'أسوأ'} "
              f"من الأمثل بفارق {abs(loss - optimal['predicted_loss']):.4f})")

demonstrate_chinchilla_tradeoff()

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-05-building-llms/ch20-scaling-laws.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def compare_training_optimal_vs_inference_optimal():
    """
    توضيح الفارق بين استراتيجية Chinchilla الكلاسيكية (الأمثل للتدريب فقط)
    واستراتيجية "over-training" الحديثة (الأمثل مع اعتبار تكلفة الاستدلال)
    كما طبّقتها Meta فعليًا في سلسلة LLaMA.
    """
    scaling_law = ChinchillaScalingLaw()

    scenarios = [
        {"name": "Chinchilla الأمثل نظريًا لـ N=7B",
         "N": 7e9, "D": 7e9 * 20},
        {"name": "LLaMA-2 7B الفعلي (over-trained عمدًا)",
         "N": 7e9, "D": 2e12},
    ]

    print(f"{'السيناريو':<45}{'N':<10}{'D':<12}{'D/N':<8}{'Loss المتوقع':<15}")
    print("-" * 90)
    for s in scenarios:
        loss = scaling_law.predict_loss(s["N"], s["D"])
        compute = scaling_law.compute_flops(s["N"], s["D"])
        print(f"{s['name']:<45}{s['N']:.1e}  {s['D']:.1e}  "
              f"{s['D']/s['N']:<8.1f}{loss:<15.4f}")
        print(f"   → ميزانية الحوسبة المطلوبة: {compute:.2e} FLOPs")

compare_training_optimal_vs_inference_optimal()

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-05-building-llms/ch20-scaling-laws.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def solve_chinchilla_example_manually(compute_budget: float,
                                        d_to_n_ratio: float = 20.0) -> dict:
    """
    حل تحليلي مباشر (وليس تحسينًا عدديًا) لإيجاد N وD المثاليين
    بافتراض نسبة D/N معينة (عادة 20 وفق قاعدة Chinchilla التقريبية)
    وقيد ميزانية حوسبة C = 6ND.

    الاشتقاق: C = 6ND = 6N(ratio×N) = 6×ratio×N²
    إذن: N = sqrt(C / (6×ratio))
    """
    N_opt = (compute_budget / (6 * d_to_n_ratio)) ** 0.5
    D_opt = d_to_n_ratio * N_opt
    verification_C = 6 * N_opt * D_opt

    return {
        "N_opt": N_opt,
        "D_opt": D_opt,
        "verification_compute": verification_C,
        "matches_budget": abs(verification_C - compute_budget) / compute_budget < 0.01,
    }

result = solve_chinchilla_example_manually(compute_budget=1e20)
print(f"N الأمثل: {result['N_opt']:.3e} معامل ({result['N_opt']/1e6:.0f} مليون)")
print(f"D الأمثل: {result['D_opt']:.3e} رمز ({result['D_opt']/1e9:.1f} مليار)")
print(f"التحقق من ميزانية الحوسبة: {result['verification_compute']:.3e} "
      f"(مطابق: {result['matches_budget']})")

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-05-building-llms/ch20-scaling-laws.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-20-ch20-scaling-laws.ipynb#cell-2
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
    ("numpy", "numpy"),
    ("scipy", "scipy")
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-20-ch20-scaling-laws.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from scipy.optimize import minimize

class ChinchillaScalingLaw:
    """
    تطبيق عملي لمعادلة Chinchilla لتقدير الخسارة المتوقعة لأي تركيبة
    (N, D)، وإيجاد التخصيص الأمثل لأي ميزانية حوسبة معطاة.

    الثوابت أدناه مأخوذة تقريبيًا من القيم المُقدَّرة في الورقة الأصلية
    (Hoffmann et al. 2022، الجدول A.3 - تقديرات عبر Approach 3).
    """
    def __init__(self):
        # ثوابت مُقدَّرة تجريبيًا من الورقة الأصلية (بوحدات مناسبة)
        self.E = 1.69       # إنتروبيا اللغة الطبيعية التقريبية (حد أدنى للخسارة)
        self.A = 406.4      # ثابت حد النموذج
        self.alpha = 0.34   # أس حد النموذج
        self.B = 410.7      # ثابت حد البيانات
        self.beta = 0.28    # أس حد البيانات

    def predict_loss(self, N: float, D: float) -> float:
        """
        التنبؤ بالخسارة النهائية (Cross-Entropy بالـ nats) لنموذج بحجم
        N معامل مُدرَّب على D رمز، وفق معادلة Chinchilla الإمبريقية.
        """
        return self.E + (self.A / N ** self.alpha) + (self.B / D ** self.beta)

    def compute_flops(self, N: float, D: float) -> float:
        """حساب ميزانية الحوسبة التقريبية باستخدام القاعدة C ≈ 6ND"""
        return 6 * N * D

    def find_optimal_allocation(self, compute_budget: float) -> dict:
        """
        إيجاد (N_opt, D_opt) التي تُدنّي الخسارة L(N,D) تحت قيد
        ميزانية حوسبة ثابتة C = 6ND، عبر تحسين عددي مباشر
        (بديل عملي عن الحل التحليلي بمضاعفات لاغرانج).
        """
        def objective(log_N):
            # نعمل في الفضاء اللوغاريتمي لتحسين استقرار التحسين العددي
            N = np.exp(log_N[0])
            D = compute_budget / (6 * N)  # اشتقاق D من قيد الميزانية مباشرة
            if D <= 0:
                return 1e10
            return self.predict_loss(N, D)

        # تخمين أولي معقول (نبدأ من افتراض D=20N كنقطة انطلاق)
        initial_N_guess = (compute_budget / (6 * 20)) ** 0.5
        result = minimize(objective, x0=[np.log(initial_N_guess)],
                           method="Nelder-Mead")

        N_opt = np.exp(result.x[0])
        D_opt = compute_budget / (6 * N_opt)
        loss_opt = self.predict_loss(N_opt, D_opt)

        return {
            "N_opt": N_opt,
            "D_opt": D_opt,
            "D_to_N_ratio": D_opt / N_opt,
            "predicted_loss": loss_opt,
            "compute_budget": compute_budget,
        }

def demonstrate_chinchilla_tradeoff():
    """
    توضيح عملي: لنفس ميزانية الحوسبة، ماذا يحدث للخسارة إذا انحرفنا
    عن التخصيص الأمثل (نموذج كبير جدًا+بيانات قليلة، أو العكس)؟
    """
    scaling_law = ChinchillaScalingLaw()
    compute_budget = 1e21  # ميزانية حوسبة ثابتة (FLOPs) مشابهة لتجربة متوسطة

    # الحل الأمثل وفق التحسين
    optimal = scaling_law.find_optimal_allocation(compute_budget)
    print("=== التخصيص الأمثل (Compute-Optimal) ===")
    print(f"N_opt (عدد المعاملات): {optimal['N_opt']:.3e}")
    print(f"D_opt (عدد الرموز): {optimal['D_opt']:.3e}")
    print(f"النسبة D/N: {optimal['D_to_N_ratio']:.2f}  (قريبة من 20 كما تتوقع Chinchilla)")
    print(f"الخسارة المتوقعة: {optimal['predicted_loss']:.4f}\n")

    # سيناريوهات بديلة بنفس ميزانية الحوسبة بالضبط لكن بتوزيع مختلف
    print("=== مقارنة سيناريوهات بديلة بنفس ميزانية الحوسبة ===")
    scenarios = {
        "نموذج ضخم جدًا + بيانات قليلة (مثل الاتجاه القديم لـ GPT-3)": 5.0,
        "التخصيص الأمثل (Chinchilla-style)": 1.0,
        "نموذج صغير + بيانات هائلة جدًا": 0.2,
    }

    for name, multiplier in scenarios.items():
        N = optimal["N_opt"] * (multiplier ** 0.5)
        D = compute_budget / (6 * N)
        loss = scaling_law.predict_loss(N, D)
        print(f"\n{name}:")
        print(f"   N = {N:.3e}, D = {D:.3e}, D/N = {D/N:.1f}")
        print(f"   الخسارة المتوقعة = {loss:.4f} "
              f"({'أفضل' if loss < optimal['predicted_loss'] else 'أسوأ'} "
              f"من الأمثل بفارق {abs(loss - optimal['predicted_loss']):.4f})")

demonstrate_chinchilla_tradeoff()

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-20-ch20-scaling-laws.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def compare_training_optimal_vs_inference_optimal():
    """
    توضيح الفارق بين استراتيجية Chinchilla الكلاسيكية (الأمثل للتدريب فقط)
    واستراتيجية "over-training" الحديثة (الأمثل مع اعتبار تكلفة الاستدلال)
    كما طبّقتها Meta فعليًا في سلسلة LLaMA.
    """
    scaling_law = ChinchillaScalingLaw()

    scenarios = [
        {"name": "Chinchilla الأمثل نظريًا لـ N=7B",
         "N": 7e9, "D": 7e9 * 20},
        {"name": "LLaMA-2 7B الفعلي (over-trained عمدًا)",
         "N": 7e9, "D": 2e12},
    ]

    print(f"{'السيناريو':<45}{'N':<10}{'D':<12}{'D/N':<8}{'Loss المتوقع':<15}")
    print("-" * 90)
    for s in scenarios:
        loss = scaling_law.predict_loss(s["N"], s["D"])
        compute = scaling_law.compute_flops(s["N"], s["D"])
        print(f"{s['name']:<45}{s['N']:.1e}  {s['D']:.1e}  "
              f"{s['D']/s['N']:<8.1f}{loss:<15.4f}")
        print(f"   → ميزانية الحوسبة المطلوبة: {compute:.2e} FLOPs")

compare_training_optimal_vs_inference_optimal()

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-20-ch20-scaling-laws.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def solve_chinchilla_example_manually(compute_budget: float,
                                        d_to_n_ratio: float = 20.0) -> dict:
    """
    حل تحليلي مباشر (وليس تحسينًا عدديًا) لإيجاد N وD المثاليين
    بافتراض نسبة D/N معينة (عادة 20 وفق قاعدة Chinchilla التقريبية)
    وقيد ميزانية حوسبة C = 6ND.

    الاشتقاق: C = 6ND = 6N(ratio×N) = 6×ratio×N²
    إذن: N = sqrt(C / (6×ratio))
    """
    N_opt = (compute_budget / (6 * d_to_n_ratio)) ** 0.5
    D_opt = d_to_n_ratio * N_opt
    verification_C = 6 * N_opt * D_opt

    return {
        "N_opt": N_opt,
        "D_opt": D_opt,
        "verification_compute": verification_C,
        "matches_budget": abs(verification_C - compute_budget) / compute_budget < 0.01,
    }

result = solve_chinchilla_example_manually(compute_budget=1e20)
print(f"N الأمثل: {result['N_opt']:.3e} معامل ({result['N_opt']/1e6:.0f} مليون)")
print(f"D الأمثل: {result['D_opt']:.3e} رمز ({result['D_opt']/1e9:.1f} مليار)")
print(f"التحقق من ميزانية الحوسبة: {result['verification_compute']:.3e} "
      f"(مطابق: {result['matches_budget']})")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-20-ch20-scaling-laws.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
