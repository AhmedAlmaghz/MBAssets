#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — خط التقييم المستمر
Mastering Transformers and Large Language Models — Continuous Evaluation

Book slug : mastering-llms-transformers
Chapter   : ch-46-ch46-continuous-evaluation (lesson 46)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-46-ch46-continuous-evaluation.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — خط التقييم المستمر
Mastering Transformers and Large Language Models — Continuous Evaluation

Book slug : mastering-llms-transformers
Chapter   : ch-46-ch46-continuous-evaluation (lesson 46)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-46-ch46-continuous-evaluation.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — خط التقييم المستمر
Mastering Transformers and Large Language Models — Continuous Evaluation

Book slug : mastering-llms-transformers
Chapter   : ch-46-ch46-continuous-evaluation (lesson 46)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-09-evaluation/ch46-continuous-evaluation.ipynb#cell-1
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
# [002] content/notebooks/part-09-evaluation/ch46-continuous-evaluation.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import json
import datetime
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class EvaluationSuite:
    """تعريف خط تقييم كامل قابل لإعادة الاستخدام عبر إصدارات النموذج"""
    smoke_tests: list = field(default_factory=list)
    benchmarks: dict = field(default_factory=dict)       # {name: eval_function}
    golden_set: list = field(default_factory=list)
    regression_thresholds: dict = field(default_factory=dict)  # {metric: min_acceptable_delta}

class ContinuousEvaluationPipeline:
    def __init__(self, suite: EvaluationSuite, baseline_model, candidate_model, judge_client):
        self.suite = suite
        self.baseline_model = baseline_model
        self.candidate_model = candidate_model
        self.judge_client = judge_client
        self.history = []

    def run_smoke_tests(self) -> bool:
        """اختبارات سريعة جدًا (ثوانٍ) — يجب أن تمر 100% وإلا رفض فوري"""
        for test in self.suite.smoke_tests:
            response = self.candidate_model.generate(test["prompt"])
            if not test["assertion_fn"](response):
                print(f"❌ فشل اختبار الدخان: {test['name']}")
                return False
        print("✅ كل اختبارات الدخان نجحت")
        return True

    def run_full_benchmarks(self) -> dict:
        results = {}
        for name, eval_fn in self.suite.benchmarks.items():
            score = eval_fn(self.candidate_model)
            results[name] = score
            print(f"📊 {name}: {score:.4f}")
        return results

    def run_golden_set_judge_comparison(self) -> dict:
        """مقارنة النموذج الجديد بالحالي على مجموعة ذهبية ثابتة"""
        wins, losses, ties = 0, 0, 0
        for example in self.suite.golden_set:
            response_baseline = self.baseline_model.generate(example["prompt"])
            response_candidate = self.candidate_model.generate(example["prompt"])

            verdict = pairwise_judge_with_position_swap(
                self.judge_client, example["prompt"], response_baseline, response_candidate
            )
            if verdict == "ب":   # النموذج المرشح أفضل
                wins += 1
            elif verdict == "أ":
                losses += 1
            else:
                ties += 1

        total = wins + losses + ties
        return {
            "win_rate": wins / total, "loss_rate": losses / total, "tie_rate": ties / total,
            "net_improvement": (wins - losses) / total
        }

    def check_regressions(self, benchmark_results: dict, baseline_results: dict) -> list:
        """كشف أي تراجع يتجاوز العتبة المسموحة"""
        regressions = []
        for metric, score in benchmark_results.items():
            baseline_score = baseline_results.get(metric, 0)
            delta = score - baseline_score
            threshold = self.suite.regression_thresholds.get(metric, -0.02)  # افتراضي: -2%
            if delta < threshold:
                regressions.append({
                    "metric": metric, "baseline": baseline_score,
                    "candidate": score, "delta": delta
                })
        return regressions

    def generate_report(self) -> dict:
        """تشغيل الخط الكامل وإنتاج تقرير نهائي"""
        report = {"timestamp": datetime.datetime.now().isoformat()}

        if not self.run_smoke_tests():
            report["status"] = "REJECTED_SMOKE_TEST_FAILURE"
            return report

        report["benchmark_results"] = self.run_full_benchmarks()
        report["judge_comparison"] = self.run_golden_set_judge_comparison()

        # افتراض وجود نتائج مرجعية محفوظة من قبل
        baseline_results = self._load_baseline_results()
        regressions = self.check_regressions(report["benchmark_results"], baseline_results)

        if regressions:
            report["status"] = "REGRESSION_DETECTED"
            report["regressions"] = regressions
        elif report["judge_comparison"]["net_improvement"] < -0.05:
            report["status"] = "QUALITY_DEGRADATION_DETECTED"
        else:
            report["status"] = "APPROVED_FOR_CANARY_DEPLOYMENT"

        self.history.append(report)
        return report

    def _load_baseline_results(self):
        # في التطبيق الفعلي: تحميل من قاعدة بيانات/ملف تتبّع تجارب (MLflow, W&B)
        return {}

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-09-evaluation/ch46-continuous-evaluation.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class ProductionDriftMonitor:
    """
    مراقبة سلوك النموذج في الإنتاج الحقيقي (وليس فقط قبل النشر)
    لكشف تدهور تدريجي غير متوقع (Model/Data Drift)
    """
    def __init__(self, window_size=1000, alert_threshold=0.1):
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.recent_scores = []
        self.baseline_mean = None

    def log_interaction_score(self, implicit_feedback_score: float):
        """
        implicit_feedback_score: إشارة ضمنية لجودة التفاعل
        (مثل: هل أعاد المستخدم صياغة السؤال؟ هل ضغط "إعجاب"؟ طول المحادثة؟)
        """
        self.recent_scores.append(implicit_feedback_score)
        if len(self.recent_scores) > self.window_size:
            self.recent_scores.pop(0)

        if self.baseline_mean is None and len(self.recent_scores) == self.window_size:
            self.baseline_mean = sum(self.recent_scores) / len(self.recent_scores)

    def check_drift(self) -> dict:
        if self.baseline_mean is None or len(self.recent_scores) < self.window_size:
            return {"status": "insufficient_data"}

        current_mean = sum(self.recent_scores) / len(self.recent_scores)
        relative_change = (current_mean - self.baseline_mean) / abs(self.baseline_mean)

        if abs(relative_change) > self.alert_threshold:
            return {
                "status": "DRIFT_DETECTED",
                "baseline": self.baseline_mean,
                "current": current_mean,
                "relative_change": relative_change
            }
        return {"status": "STABLE"}

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-09-evaluation/ch46-continuous-evaluation.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-46-ch46-continuous-evaluation.ipynb#cell-1
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-46-ch46-continuous-evaluation.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import json
import datetime
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class EvaluationSuite:
    """تعريف خط تقييم كامل قابل لإعادة الاستخدام عبر إصدارات النموذج"""
    smoke_tests: list = field(default_factory=list)
    benchmarks: dict = field(default_factory=dict)       # {name: eval_function}
    golden_set: list = field(default_factory=list)
    regression_thresholds: dict = field(default_factory=dict)  # {metric: min_acceptable_delta}

class ContinuousEvaluationPipeline:
    def __init__(self, suite: EvaluationSuite, baseline_model, candidate_model, judge_client):
        self.suite = suite
        self.baseline_model = baseline_model
        self.candidate_model = candidate_model
        self.judge_client = judge_client
        self.history = []

    def run_smoke_tests(self) -> bool:
        """اختبارات سريعة جدًا (ثوانٍ) — يجب أن تمر 100% وإلا رفض فوري"""
        for test in self.suite.smoke_tests:
            response = self.candidate_model.generate(test["prompt"])
            if not test["assertion_fn"](response):
                print(f"❌ فشل اختبار الدخان: {test['name']}")
                return False
        print("✅ كل اختبارات الدخان نجحت")
        return True

    def run_full_benchmarks(self) -> dict:
        results = {}
        for name, eval_fn in self.suite.benchmarks.items():
            score = eval_fn(self.candidate_model)
            results[name] = score
            print(f"📊 {name}: {score:.4f}")
        return results

    def run_golden_set_judge_comparison(self) -> dict:
        """مقارنة النموذج الجديد بالحالي على مجموعة ذهبية ثابتة"""
        wins, losses, ties = 0, 0, 0
        for example in self.suite.golden_set:
            response_baseline = self.baseline_model.generate(example["prompt"])
            response_candidate = self.candidate_model.generate(example["prompt"])

            verdict = pairwise_judge_with_position_swap(
                self.judge_client, example["prompt"], response_baseline, response_candidate
            )
            if verdict == "ب":   # النموذج المرشح أفضل
                wins += 1
            elif verdict == "أ":
                losses += 1
            else:
                ties += 1

        total = wins + losses + ties
        return {
            "win_rate": wins / total, "loss_rate": losses / total, "tie_rate": ties / total,
            "net_improvement": (wins - losses) / total
        }

    def check_regressions(self, benchmark_results: dict, baseline_results: dict) -> list:
        """كشف أي تراجع يتجاوز العتبة المسموحة"""
        regressions = []
        for metric, score in benchmark_results.items():
            baseline_score = baseline_results.get(metric, 0)
            delta = score - baseline_score
            threshold = self.suite.regression_thresholds.get(metric, -0.02)  # افتراضي: -2%
            if delta < threshold:
                regressions.append({
                    "metric": metric, "baseline": baseline_score,
                    "candidate": score, "delta": delta
                })
        return regressions

    def generate_report(self) -> dict:
        """تشغيل الخط الكامل وإنتاج تقرير نهائي"""
        report = {"timestamp": datetime.datetime.now().isoformat()}

        if not self.run_smoke_tests():
            report["status"] = "REJECTED_SMOKE_TEST_FAILURE"
            return report

        report["benchmark_results"] = self.run_full_benchmarks()
        report["judge_comparison"] = self.run_golden_set_judge_comparison()

        # افتراض وجود نتائج مرجعية محفوظة من قبل
        baseline_results = self._load_baseline_results()
        regressions = self.check_regressions(report["benchmark_results"], baseline_results)

        if regressions:
            report["status"] = "REGRESSION_DETECTED"
            report["regressions"] = regressions
        elif report["judge_comparison"]["net_improvement"] < -0.05:
            report["status"] = "QUALITY_DEGRADATION_DETECTED"
        else:
            report["status"] = "APPROVED_FOR_CANARY_DEPLOYMENT"

        self.history.append(report)
        return report

    def _load_baseline_results(self):
        # في التطبيق الفعلي: تحميل من قاعدة بيانات/ملف تتبّع تجارب (MLflow, W&B)
        return {}

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-46-ch46-continuous-evaluation.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class ProductionDriftMonitor:
    """
    مراقبة سلوك النموذج في الإنتاج الحقيقي (وليس فقط قبل النشر)
    لكشف تدهور تدريجي غير متوقع (Model/Data Drift)
    """
    def __init__(self, window_size=1000, alert_threshold=0.1):
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.recent_scores = []
        self.baseline_mean = None

    def log_interaction_score(self, implicit_feedback_score: float):
        """
        implicit_feedback_score: إشارة ضمنية لجودة التفاعل
        (مثل: هل أعاد المستخدم صياغة السؤال؟ هل ضغط "إعجاب"؟ طول المحادثة؟)
        """
        self.recent_scores.append(implicit_feedback_score)
        if len(self.recent_scores) > self.window_size:
            self.recent_scores.pop(0)

        if self.baseline_mean is None and len(self.recent_scores) == self.window_size:
            self.baseline_mean = sum(self.recent_scores) / len(self.recent_scores)

    def check_drift(self) -> dict:
        if self.baseline_mean is None or len(self.recent_scores) < self.window_size:
            return {"status": "insufficient_data"}

        current_mean = sum(self.recent_scores) / len(self.recent_scores)
        relative_change = (current_mean - self.baseline_mean) / abs(self.baseline_mean)

        if abs(relative_change) > self.alert_threshold:
            return {
                "status": "DRIFT_DETECTED",
                "baseline": self.baseline_mean,
                "current": current_mean,
                "relative_change": relative_change
            }
        return {"status": "STABLE"}

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-46-ch46-continuous-evaluation.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-46-ch46-continuous-evaluation.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — خط التقييم المستمر
Mastering Transformers and Large Language Models — Continuous Evaluation

Book slug : mastering-llms-transformers
Chapter   : ch-46-ch46-continuous-evaluation (lesson 46)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-09-evaluation/ch46-continuous-evaluation.ipynb#cell-1
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
# [002] content/notebooks/part-09-evaluation/ch46-continuous-evaluation.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import json
import datetime
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class EvaluationSuite:
    """تعريف خط تقييم كامل قابل لإعادة الاستخدام عبر إصدارات النموذج"""
    smoke_tests: list = field(default_factory=list)
    benchmarks: dict = field(default_factory=dict)       # {name: eval_function}
    golden_set: list = field(default_factory=list)
    regression_thresholds: dict = field(default_factory=dict)  # {metric: min_acceptable_delta}

class ContinuousEvaluationPipeline:
    def __init__(self, suite: EvaluationSuite, baseline_model, candidate_model, judge_client):
        self.suite = suite
        self.baseline_model = baseline_model
        self.candidate_model = candidate_model
        self.judge_client = judge_client
        self.history = []

    def run_smoke_tests(self) -> bool:
        """اختبارات سريعة جدًا (ثوانٍ) — يجب أن تمر 100% وإلا رفض فوري"""
        for test in self.suite.smoke_tests:
            response = self.candidate_model.generate(test["prompt"])
            if not test["assertion_fn"](response):
                print(f"❌ فشل اختبار الدخان: {test['name']}")
                return False
        print("✅ كل اختبارات الدخان نجحت")
        return True

    def run_full_benchmarks(self) -> dict:
        results = {}
        for name, eval_fn in self.suite.benchmarks.items():
            score = eval_fn(self.candidate_model)
            results[name] = score
            print(f"📊 {name}: {score:.4f}")
        return results

    def run_golden_set_judge_comparison(self) -> dict:
        """مقارنة النموذج الجديد بالحالي على مجموعة ذهبية ثابتة"""
        wins, losses, ties = 0, 0, 0
        for example in self.suite.golden_set:
            response_baseline = self.baseline_model.generate(example["prompt"])
            response_candidate = self.candidate_model.generate(example["prompt"])

            verdict = pairwise_judge_with_position_swap(
                self.judge_client, example["prompt"], response_baseline, response_candidate
            )
            if verdict == "ب":   # النموذج المرشح أفضل
                wins += 1
            elif verdict == "أ":
                losses += 1
            else:
                ties += 1

        total = wins + losses + ties
        return {
            "win_rate": wins / total, "loss_rate": losses / total, "tie_rate": ties / total,
            "net_improvement": (wins - losses) / total
        }

    def check_regressions(self, benchmark_results: dict, baseline_results: dict) -> list:
        """كشف أي تراجع يتجاوز العتبة المسموحة"""
        regressions = []
        for metric, score in benchmark_results.items():
            baseline_score = baseline_results.get(metric, 0)
            delta = score - baseline_score
            threshold = self.suite.regression_thresholds.get(metric, -0.02)  # افتراضي: -2%
            if delta < threshold:
                regressions.append({
                    "metric": metric, "baseline": baseline_score,
                    "candidate": score, "delta": delta
                })
        return regressions

    def generate_report(self) -> dict:
        """تشغيل الخط الكامل وإنتاج تقرير نهائي"""
        report = {"timestamp": datetime.datetime.now().isoformat()}

        if not self.run_smoke_tests():
            report["status"] = "REJECTED_SMOKE_TEST_FAILURE"
            return report

        report["benchmark_results"] = self.run_full_benchmarks()
        report["judge_comparison"] = self.run_golden_set_judge_comparison()

        # افتراض وجود نتائج مرجعية محفوظة من قبل
        baseline_results = self._load_baseline_results()
        regressions = self.check_regressions(report["benchmark_results"], baseline_results)

        if regressions:
            report["status"] = "REGRESSION_DETECTED"
            report["regressions"] = regressions
        elif report["judge_comparison"]["net_improvement"] < -0.05:
            report["status"] = "QUALITY_DEGRADATION_DETECTED"
        else:
            report["status"] = "APPROVED_FOR_CANARY_DEPLOYMENT"

        self.history.append(report)
        return report

    def _load_baseline_results(self):
        # في التطبيق الفعلي: تحميل من قاعدة بيانات/ملف تتبّع تجارب (MLflow, W&B)
        return {}

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-09-evaluation/ch46-continuous-evaluation.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class ProductionDriftMonitor:
    """
    مراقبة سلوك النموذج في الإنتاج الحقيقي (وليس فقط قبل النشر)
    لكشف تدهور تدريجي غير متوقع (Model/Data Drift)
    """
    def __init__(self, window_size=1000, alert_threshold=0.1):
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.recent_scores = []
        self.baseline_mean = None

    def log_interaction_score(self, implicit_feedback_score: float):
        """
        implicit_feedback_score: إشارة ضمنية لجودة التفاعل
        (مثل: هل أعاد المستخدم صياغة السؤال؟ هل ضغط "إعجاب"؟ طول المحادثة؟)
        """
        self.recent_scores.append(implicit_feedback_score)
        if len(self.recent_scores) > self.window_size:
            self.recent_scores.pop(0)

        if self.baseline_mean is None and len(self.recent_scores) == self.window_size:
            self.baseline_mean = sum(self.recent_scores) / len(self.recent_scores)

    def check_drift(self) -> dict:
        if self.baseline_mean is None or len(self.recent_scores) < self.window_size:
            return {"status": "insufficient_data"}

        current_mean = sum(self.recent_scores) / len(self.recent_scores)
        relative_change = (current_mean - self.baseline_mean) / abs(self.baseline_mean)

        if abs(relative_change) > self.alert_threshold:
            return {
                "status": "DRIFT_DETECTED",
                "baseline": self.baseline_mean,
                "current": current_mean,
                "relative_change": relative_change
            }
        return {"status": "STABLE"}

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-09-evaluation/ch46-continuous-evaluation.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-46-ch46-continuous-evaluation.ipynb#cell-2
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-46-ch46-continuous-evaluation.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import json
import datetime
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class EvaluationSuite:
    """تعريف خط تقييم كامل قابل لإعادة الاستخدام عبر إصدارات النموذج"""
    smoke_tests: list = field(default_factory=list)
    benchmarks: dict = field(default_factory=dict)       # {name: eval_function}
    golden_set: list = field(default_factory=list)
    regression_thresholds: dict = field(default_factory=dict)  # {metric: min_acceptable_delta}

class ContinuousEvaluationPipeline:
    def __init__(self, suite: EvaluationSuite, baseline_model, candidate_model, judge_client):
        self.suite = suite
        self.baseline_model = baseline_model
        self.candidate_model = candidate_model
        self.judge_client = judge_client
        self.history = []

    def run_smoke_tests(self) -> bool:
        """اختبارات سريعة جدًا (ثوانٍ) — يجب أن تمر 100% وإلا رفض فوري"""
        for test in self.suite.smoke_tests:
            response = self.candidate_model.generate(test["prompt"])
            if not test["assertion_fn"](response):
                print(f"❌ فشل اختبار الدخان: {test['name']}")
                return False
        print("✅ كل اختبارات الدخان نجحت")
        return True

    def run_full_benchmarks(self) -> dict:
        results = {}
        for name, eval_fn in self.suite.benchmarks.items():
            score = eval_fn(self.candidate_model)
            results[name] = score
            print(f"📊 {name}: {score:.4f}")
        return results

    def run_golden_set_judge_comparison(self) -> dict:
        """مقارنة النموذج الجديد بالحالي على مجموعة ذهبية ثابتة"""
        wins, losses, ties = 0, 0, 0
        for example in self.suite.golden_set:
            response_baseline = self.baseline_model.generate(example["prompt"])
            response_candidate = self.candidate_model.generate(example["prompt"])

            verdict = pairwise_judge_with_position_swap(
                self.judge_client, example["prompt"], response_baseline, response_candidate
            )
            if verdict == "ب":   # النموذج المرشح أفضل
                wins += 1
            elif verdict == "أ":
                losses += 1
            else:
                ties += 1

        total = wins + losses + ties
        return {
            "win_rate": wins / total, "loss_rate": losses / total, "tie_rate": ties / total,
            "net_improvement": (wins - losses) / total
        }

    def check_regressions(self, benchmark_results: dict, baseline_results: dict) -> list:
        """كشف أي تراجع يتجاوز العتبة المسموحة"""
        regressions = []
        for metric, score in benchmark_results.items():
            baseline_score = baseline_results.get(metric, 0)
            delta = score - baseline_score
            threshold = self.suite.regression_thresholds.get(metric, -0.02)  # افتراضي: -2%
            if delta < threshold:
                regressions.append({
                    "metric": metric, "baseline": baseline_score,
                    "candidate": score, "delta": delta
                })
        return regressions

    def generate_report(self) -> dict:
        """تشغيل الخط الكامل وإنتاج تقرير نهائي"""
        report = {"timestamp": datetime.datetime.now().isoformat()}

        if not self.run_smoke_tests():
            report["status"] = "REJECTED_SMOKE_TEST_FAILURE"
            return report

        report["benchmark_results"] = self.run_full_benchmarks()
        report["judge_comparison"] = self.run_golden_set_judge_comparison()

        # افتراض وجود نتائج مرجعية محفوظة من قبل
        baseline_results = self._load_baseline_results()
        regressions = self.check_regressions(report["benchmark_results"], baseline_results)

        if regressions:
            report["status"] = "REGRESSION_DETECTED"
            report["regressions"] = regressions
        elif report["judge_comparison"]["net_improvement"] < -0.05:
            report["status"] = "QUALITY_DEGRADATION_DETECTED"
        else:
            report["status"] = "APPROVED_FOR_CANARY_DEPLOYMENT"

        self.history.append(report)
        return report

    def _load_baseline_results(self):
        # في التطبيق الفعلي: تحميل من قاعدة بيانات/ملف تتبّع تجارب (MLflow, W&B)
        return {}

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-46-ch46-continuous-evaluation.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class ProductionDriftMonitor:
    """
    مراقبة سلوك النموذج في الإنتاج الحقيقي (وليس فقط قبل النشر)
    لكشف تدهور تدريجي غير متوقع (Model/Data Drift)
    """
    def __init__(self, window_size=1000, alert_threshold=0.1):
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.recent_scores = []
        self.baseline_mean = None

    def log_interaction_score(self, implicit_feedback_score: float):
        """
        implicit_feedback_score: إشارة ضمنية لجودة التفاعل
        (مثل: هل أعاد المستخدم صياغة السؤال؟ هل ضغط "إعجاب"؟ طول المحادثة؟)
        """
        self.recent_scores.append(implicit_feedback_score)
        if len(self.recent_scores) > self.window_size:
            self.recent_scores.pop(0)

        if self.baseline_mean is None and len(self.recent_scores) == self.window_size:
            self.baseline_mean = sum(self.recent_scores) / len(self.recent_scores)

    def check_drift(self) -> dict:
        if self.baseline_mean is None or len(self.recent_scores) < self.window_size:
            return {"status": "insufficient_data"}

        current_mean = sum(self.recent_scores) / len(self.recent_scores)
        relative_change = (current_mean - self.baseline_mean) / abs(self.baseline_mean)

        if abs(relative_change) > self.alert_threshold:
            return {
                "status": "DRIFT_DETECTED",
                "baseline": self.baseline_mean,
                "current": current_mean,
                "relative_change": relative_change
            }
        return {"status": "STABLE"}

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-46-ch46-continuous-evaluation.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
