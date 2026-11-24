#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — كشف انحراف البيانات والمفهوم — الحارس الصامت لجودة النموذج في الإنتاج
Large Language Model Engineering — Chapter 62 — Deployment Llmops

Book slug : llm-engineering-69-chapters
Chapter   : ch-62-chapter-62 (lesson 62)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-62-chapter-62.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — كشف انحراف البيانات والمفهوم — الحارس الصامت لجودة النموذج في الإنتاج
Large Language Model Engineering — Chapter 62 — Deployment Llmops

Book slug : llm-engineering-69-chapters
Chapter   : ch-62-chapter-62 (lesson 62)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-62-chapter-62.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — كشف انحراف البيانات والمفهوم — الحارس الصامت لجودة النموذج في الإنتاج
Large Language Model Engineering — Chapter 62 — Deployment Llmops

Book slug : llm-engineering-69-chapters
Chapter   : ch-62-chapter-62 (lesson 62)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_62.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 62: الفصل 62: كشف انحراف البيانات والمفهوم — الحارس الصامت لجودة النموذج في الإنتاج."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_drift() -> None:
    baseline, current = ([0.5, 0.3, 0.2], [0.2, 0.3, 0.5])
    epsilon = 1e-06
    psi = sum(((actual - expected) * math.log((actual + epsilon) / (expected + epsilon)) for expected, actual in zip(baseline, current)))
    print('baseline distribution:', baseline)
    print('current distribution:', current)
    print('population stability index:', round(psi, 4))
    print('alert threshold must be calibrated against historical data.')

if __name__ == "__main__":
    print("Chapter 62 lab: Detect a change in token-frequency distributions")
    run_drift()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-11-deployment-llmops/chapter-62.md
# ────────────────────────────────────────────────────────────────────────
"""
الفصل 62: كشف انحراف البيانات والمفهوم
الحالة: 🟢 كود تعليمي مُحاكٍ (educational simulation) — يُحاكي خط أنابيب كشف
         انحراف واقعي، ومناسب للتشغيل الفعلي على بيانات حقيقية بعد استبدال
         دالة توليد التضمينات بنموذج embedding حقيقي.
يتطلب: numpy, scipy, scikit-learn, pandas
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA


class DriftSeverity(Enum):
    NONE = "لا يوجد انحراف"
    MODERATE = "انحراف متوسط - يستحق المراقبة"
    SEVERE = "انحراف كبير - يستوجب تدخلاً"


@dataclass
class DriftReport:
    """تقرير انحراف موحّد يُنتج من كل طريقة كشف."""
    method_name: str
    score: float
    severity: DriftSeverity
    details: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"[{self.method_name}] الدرجة={self.score:.4f} "
                f"الخطورة={self.severity.value}")


class DriftDetector:
    """
    كاشف انحراف متعدد الطرق لمراقبة أنظمة LLM الإنتاجية.

    يُطبِّق ثلاث استراتيجيات مكمّلة:
    1. PSI على الخصائص العددية المُشتقّة من النص.
    2. اختبار كولموغوروف-سميرنوف للمتغيرات المستمرة.
    3. كشف قائم على التصنيف (Domain Classifier) على التضمينات.
    """

    def __init__(
        self,
        psi_moderate_threshold: float = 0.1,
        psi_severe_threshold: float = 0.25,
        classifier_auc_moderate_threshold: float = 0.60,
        classifier_auc_severe_threshold: float = 0.75,
    ):
        self.psi_moderate = psi_moderate_threshold
        self.psi_severe = psi_severe_threshold
        self.clf_moderate = classifier_auc_moderate_threshold
        self.clf_severe = classifier_auc_severe_threshold

    # ------------------------------------------------------------------
    # 1) PSI على خاصية عددية (مثلاً: طول الاستعلام، عدد الرموز، إلخ)
    # ------------------------------------------------------------------
    def compute_psi(
        self,
        reference: np.ndarray,
        production: np.ndarray,
        n_bins: int = 10,
    ) -> DriftReport:
        """
        يحسب PSI لخاصية عددية واحدة بين توزيعين.
        reference: عيّنة من الفترة المرجعية (baseline).
        production: عيّنة من فترة الإنتاج الحالية.
        """
        # بناء الحاويات (bins) اعتمادًا على توزيع المرجع فقط
        bin_edges = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
        bin_edges[0], bin_edges[-1] = -np.inf, np.inf  # لتغطية القيم المتطرفة

        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        prod_counts, _ = np.histogram(production, bins=bin_edges)

        # تحويل إلى نسب مع حماية من القسمة على صفر
        epsilon = 1e-6
        ref_pct = ref_counts / len(reference) + epsilon
        prod_pct = prod_counts / len(production) + epsilon

        psi_value = float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))

        if psi_value < self.psi_moderate:
            severity = DriftSeverity.NONE
        elif psi_value < self.psi_severe:
            severity = DriftSeverity.MODERATE
        else:
            severity = DriftSeverity.SEVERE

        return DriftReport(
            method_name="PSI",
            score=psi_value,
            severity=severity,
            details={"n_bins": n_bins, "ref_size": len(reference), "prod_size": len(production)},
        )

    # ------------------------------------------------------------------
    # 2) اختبار كولموغوروف-سميرنوف لمتغيّر مستمر
    # ------------------------------------------------------------------
    def compute_ks_test(
        self,
        reference: np.ndarray,
        production: np.ndarray,
        alpha: float = 0.01,
    ) -> DriftReport:
        """يكتشف الفرق بين توزيعين مستمرين عبر اختبار KS الإحصائي."""
        ks_statistic, p_value = stats.ks_2samp(reference, production)

        if p_value >= alpha:
            severity = DriftSeverity.NONE
        elif p_value >= alpha / 10:
            severity = DriftSeverity.MODERATE
        else:
            severity = DriftSeverity.SEVERE

        return DriftReport(
            method_name="KS-Test",
            score=float(ks_statistic),
            severity=severity,
            details={"p_value": float(p_value), "alpha": alpha},
        )

    # ------------------------------------------------------------------
    # 3) الكشف القائم على التصنيف (Classifier-Based / Domain Classifier)
    #    مُستوحى من Rabanser et al. "Failing Loudly" (NeurIPS 2019)
    # ------------------------------------------------------------------
    def compute_embedding_drift(
        self,
        reference_embeddings: np.ndarray,
        production_embeddings: np.ndarray,
        use_pca: bool = True,
        pca_components: int = 20,
        random_state: int = 42,
    ) -> DriftReport:
        """
        يدرّب مُصنِّفًا للتمييز بين تضمينات الفترتين؛ AUC قريب من 0.5
        يعني عدم وجود انحراف قابل للتمييز، وAUC مرتفع يعني انحرافًا واضحًا.
        """
        if use_pca:
            n_components = min(pca_components, reference_embeddings.shape[1])
            pca = PCA(n_components=n_components, random_state=random_state)
            combined = np.vstack([reference_embeddings, production_embeddings])
            combined_reduced = pca.fit_transform(combined)
            n_ref = len(reference_embeddings)
            reference_embeddings = combined_reduced[:n_ref]
            production_embeddings = combined_reduced[n_ref:]

        X = np.vstack([reference_embeddings, production_embeddings])
        y = np.concatenate([
            np.zeros(len(reference_embeddings)),
            np.ones(len(production_embeddings)),
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=random_state
        )

        clf = LogisticRegression(max_iter=1000, random_state=random_state)
        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)

        if auc < self.clf_moderate:
            severity = DriftSeverity.NONE
        elif auc < self.clf_severe:
            severity = DriftSeverity.MODERATE
        else:
            severity = DriftSeverity.SEVERE

        # أكثر الأبعاد تمييزًا بين الفترتين (للتفسير)
        top_dims = np.argsort(np.abs(clf.coef_[0]))[::-1][:5]

        return DriftReport(
            method_name="Classifier-Based (Domain Classifier)",
            score=float(auc),
            severity=severity,
            details={"top_discriminative_dims": top_dims.tolist(), "used_pca": use_pca},
        )

    # ------------------------------------------------------------------
    # تقرير شامل يُجمِّع كل الطرق معًا
    # ------------------------------------------------------------------
    def full_report(
        self,
        reference_embeddings: np.ndarray,
        production_embeddings: np.ndarray,
        reference_query_lengths: np.ndarray,
        production_query_lengths: np.ndarray,
    ) -> list[DriftReport]:
        reports = [
            self.compute_psi(reference_query_lengths, production_query_lengths),
            self.compute_ks_test(reference_query_lengths, production_query_lengths),
            self.compute_embedding_drift(reference_embeddings, production_embeddings),
        ]
        return reports


def simulate_text_embedding(texts: list[str], base_dim: int = 50, seed: int = 0) -> np.ndarray:
    """
    🟢 دالة محاكاة فقط (EDUCATIONAL SIMULATION): تُولِّد تضمينات اصطناعية
    متّسقة مع محتوى النص (hashing بسيط) بدل استدعاء نموذج embedding حقيقي،
    لأغراض هذا المثال التعليمي فقط. في الإنتاج الفعلي، استبدل هذه الدالة
    بنموذج تضمين حقيقي (مثل sentence-transformers أو OpenAI embeddings API).
    """
    rng = np.random.default_rng(seed)
    embeddings = []
    for text in texts:
        text_hash = hash(text) % (2**32)
        local_rng = np.random.default_rng(text_hash)
        embeddings.append(local_rng.normal(0, 1, base_dim))
    return np.array(embeddings)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # سيناريو محاكاة: إطلاق خط منتجات جديد يُسبب انحرافًا تدريجيًا
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)

    # الفترة المرجعية: استفسارات عن منتجات "إلكترونيات" نمطية
    reference_lengths = rng.normal(loc=45, scale=12, size=2000).clip(5, None)
    reference_texts = [f"سؤال عن شاحن أو سماعة رقم {i}" for i in range(2000)]
    reference_embeddings = simulate_text_embedding(reference_texts, seed=1)

    # فترة الإنتاج: بعد إطلاق خط منتجات "أجهزة منزلية ذكية" جديد تمامًا
    production_lengths = rng.normal(loc=68, scale=20, size=800).clip(5, None)
    production_texts = [f"سؤال عن ربط جهاز منزلي ذكي بالتطبيق رقم {i}" for i in range(800)]
    production_embeddings = simulate_text_embedding(production_texts, seed=2)

    detector = DriftDetector()
    reports = detector.full_report(
        reference_embeddings=reference_embeddings,
        production_embeddings=production_embeddings,
        reference_query_lengths=reference_lengths,
        production_query_lengths=production_lengths,
    )

    print("=== تقرير كشف الانحراف الشامل ===\n")
    for report in reports:
        print(report)
        if report.details:
            print(f"   تفاصيل إضافية: {report.details}")
        print()

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-11-deployment-llmops/chapter-62.md
# ────────────────────────────────────────────────────────────────────────
"""
مثال E2E: مراقبة انحراف أسبوعية على مدى 20 أسبوعًا
الحالة: 🟢 كود تعليمي يُحاكي سيناريو إنتاجيًا متكاملاً
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class WeeklyMonitoringResult:
    week: int
    psi_score: float
    embedding_auc: float
    escalation_rate: float  # مقياس وكيل (proxy) لـ Concept Drift
    overall_alert: bool


def simulate_weekly_monitoring(n_weeks: int = 20, drift_start_week: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    detector = DriftDetector()

    # الأسبوع صفر يُعرَّف كمرجع (baseline) ثابت طوال التجربة
    reference_lengths = rng.normal(45, 12, 1500).clip(5, None)
    reference_texts = [f"استفسار إلكترونيات نمطي {i}" for i in range(1500)]
    reference_embeddings = simulate_text_embedding(reference_texts, seed=100)

    results = []
    for week in range(1, n_weeks + 1):
        # محاكاة انحراف تدريجي يبدأ من drift_start_week
        drift_intensity = max(0, (week - drift_start_week) / (n_weeks - drift_start_week))

        prod_lengths = rng.normal(
            45 + drift_intensity * 25,   # الطول يزداد تدريجيًا
            12 + drift_intensity * 8,
            400,
        ).clip(5, None)

        n_new_topic = int(400 * drift_intensity)
        n_old_topic = 400 - n_new_topic
        prod_texts = (
            [f"استفسار إلكترونيات نمطي {i}" for i in range(n_old_topic)]
            + [f"استفسار منتج ذكي جديد {i}" for i in range(n_new_topic)]
        )
        prod_embeddings = simulate_text_embedding(prod_texts, seed=200 + week)

        psi_report = detector.compute_psi(reference_lengths, prod_lengths)
        emb_report = detector.compute_embedding_drift(reference_embeddings, prod_embeddings)

        # مقياس وكيل لـ Concept Drift: معدل التصعيد يرتفع مع الانحراف
        escalation_rate = 0.05 + drift_intensity * 0.25 + rng.normal(0, 0.01)
        escalation_rate = max(0, min(1, escalation_rate))

        overall_alert = (
            psi_report.severity != DriftSeverity.NONE
            or emb_report.severity != DriftSeverity.NONE
            or escalation_rate > 0.20
        )

        results.append(WeeklyMonitoringResult(
            week=week,
            psi_score=psi_report.score,
            embedding_auc=emb_report.score,
            escalation_rate=round(escalation_rate, 3),
            overall_alert=overall_alert,
        ))

    return pd.DataFrame([r.__dict__ for r in results])


if __name__ == "__main__":
    df = simulate_weekly_monitoring()
    print(df.to_string(index=False))

    first_alert_week = df[df["overall_alert"]]["week"].min()
    print(f"\n🚨 أول أسبوع يُطلق فيه إنذار انحراف: الأسبوع {first_alert_week}")
    print("   (بينما الشكوى البشرية الفعلية في قصتنا الافتتاحية لم تصل إلا بعد أسابيع إضافية كثيرة)")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-62-chapter-62.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 62: الفصل 62: كشف انحراف البيانات والمفهوم — الحارس الصامت لجودة النموذج في الإنتاج."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_drift() -> None:
    baseline, current = ([0.5, 0.3, 0.2], [0.2, 0.3, 0.5])
    epsilon = 1e-06
    psi = sum(((actual - expected) * math.log((actual + epsilon) / (expected + epsilon)) for expected, actual in zip(baseline, current)))
    print('baseline distribution:', baseline)
    print('current distribution:', current)
    print('population stability index:', round(psi, 4))
    print('alert threshold must be calibrated against historical data.')

if __name__ == "__main__":
    print("Chapter 62 lab: Detect a change in token-frequency distributions")
    run_drift()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-62-chapter-62.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
الفصل 62: كشف انحراف البيانات والمفهوم
الحالة: 🟢 كود تعليمي مُحاكٍ (educational simulation) — يُحاكي خط أنابيب كشف
         انحراف واقعي، ومناسب للتشغيل الفعلي على بيانات حقيقية بعد استبدال
         دالة توليد التضمينات بنموذج embedding حقيقي.
يتطلب: numpy, scipy, scikit-learn, pandas
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA


class DriftSeverity(Enum):
    NONE = "لا يوجد انحراف"
    MODERATE = "انحراف متوسط - يستحق المراقبة"
    SEVERE = "انحراف كبير - يستوجب تدخلاً"


@dataclass
class DriftReport:
    """تقرير انحراف موحّد يُنتج من كل طريقة كشف."""
    method_name: str
    score: float
    severity: DriftSeverity
    details: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"[{self.method_name}] الدرجة={self.score:.4f} "
                f"الخطورة={self.severity.value}")


class DriftDetector:
    """
    كاشف انحراف متعدد الطرق لمراقبة أنظمة LLM الإنتاجية.

    يُطبِّق ثلاث استراتيجيات مكمّلة:
    1. PSI على الخصائص العددية المُشتقّة من النص.
    2. اختبار كولموغوروف-سميرنوف للمتغيرات المستمرة.
    3. كشف قائم على التصنيف (Domain Classifier) على التضمينات.
    """

    def __init__(
        self,
        psi_moderate_threshold: float = 0.1,
        psi_severe_threshold: float = 0.25,
        classifier_auc_moderate_threshold: float = 0.60,
        classifier_auc_severe_threshold: float = 0.75,
    ):
        self.psi_moderate = psi_moderate_threshold
        self.psi_severe = psi_severe_threshold
        self.clf_moderate = classifier_auc_moderate_threshold
        self.clf_severe = classifier_auc_severe_threshold

    # ------------------------------------------------------------------
    # 1) PSI على خاصية عددية (مثلاً: طول الاستعلام، عدد الرموز، إلخ)
    # ------------------------------------------------------------------
    def compute_psi(
        self,
        reference: np.ndarray,
        production: np.ndarray,
        n_bins: int = 10,
    ) -> DriftReport:
        """
        يحسب PSI لخاصية عددية واحدة بين توزيعين.
        reference: عيّنة من الفترة المرجعية (baseline).
        production: عيّنة من فترة الإنتاج الحالية.
        """
        # بناء الحاويات (bins) اعتمادًا على توزيع المرجع فقط
        bin_edges = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
        bin_edges[0], bin_edges[-1] = -np.inf, np.inf  # لتغطية القيم المتطرفة

        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        prod_counts, _ = np.histogram(production, bins=bin_edges)

        # تحويل إلى نسب مع حماية من القسمة على صفر
        epsilon = 1e-6
        ref_pct = ref_counts / len(reference) + epsilon
        prod_pct = prod_counts / len(production) + epsilon

        psi_value = float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))

        if psi_value < self.psi_moderate:
            severity = DriftSeverity.NONE
        elif psi_value < self.psi_severe:
            severity = DriftSeverity.MODERATE
        else:
            severity = DriftSeverity.SEVERE

        return DriftReport(
            method_name="PSI",
            score=psi_value,
            severity=severity,
            details={"n_bins": n_bins, "ref_size": len(reference), "prod_size": len(production)},
        )

    # ------------------------------------------------------------------
    # 2) اختبار كولموغوروف-سميرنوف لمتغيّر مستمر
    # ------------------------------------------------------------------
    def compute_ks_test(
        self,
        reference: np.ndarray,
        production: np.ndarray,
        alpha: float = 0.01,
    ) -> DriftReport:
        """يكتشف الفرق بين توزيعين مستمرين عبر اختبار KS الإحصائي."""
        ks_statistic, p_value = stats.ks_2samp(reference, production)

        if p_value >= alpha:
            severity = DriftSeverity.NONE
        elif p_value >= alpha / 10:
            severity = DriftSeverity.MODERATE
        else:
            severity = DriftSeverity.SEVERE

        return DriftReport(
            method_name="KS-Test",
            score=float(ks_statistic),
            severity=severity,
            details={"p_value": float(p_value), "alpha": alpha},
        )

    # ------------------------------------------------------------------
    # 3) الكشف القائم على التصنيف (Classifier-Based / Domain Classifier)
    #    مُستوحى من Rabanser et al. "Failing Loudly" (NeurIPS 2019)
    # ------------------------------------------------------------------
    def compute_embedding_drift(
        self,
        reference_embeddings: np.ndarray,
        production_embeddings: np.ndarray,
        use_pca: bool = True,
        pca_components: int = 20,
        random_state: int = 42,
    ) -> DriftReport:
        """
        يدرّب مُصنِّفًا للتمييز بين تضمينات الفترتين؛ AUC قريب من 0.5
        يعني عدم وجود انحراف قابل للتمييز، وAUC مرتفع يعني انحرافًا واضحًا.
        """
        if use_pca:
            n_components = min(pca_components, reference_embeddings.shape[1])
            pca = PCA(n_components=n_components, random_state=random_state)
            combined = np.vstack([reference_embeddings, production_embeddings])
            combined_reduced = pca.fit_transform(combined)
            n_ref = len(reference_embeddings)
            reference_embeddings = combined_reduced[:n_ref]
            production_embeddings = combined_reduced[n_ref:]

        X = np.vstack([reference_embeddings, production_embeddings])
        y = np.concatenate([
            np.zeros(len(reference_embeddings)),
            np.ones(len(production_embeddings)),
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=random_state
        )

        clf = LogisticRegression(max_iter=1000, random_state=random_state)
        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)

        if auc < self.clf_moderate:
            severity = DriftSeverity.NONE
        elif auc < self.clf_severe:
            severity = DriftSeverity.MODERATE
        else:
            severity = DriftSeverity.SEVERE

        # أكثر الأبعاد تمييزًا بين الفترتين (للتفسير)
        top_dims = np.argsort(np.abs(clf.coef_[0]))[::-1][:5]

        return DriftReport(
            method_name="Classifier-Based (Domain Classifier)",
            score=float(auc),
            severity=severity,
            details={"top_discriminative_dims": top_dims.tolist(), "used_pca": use_pca},
        )

    # ------------------------------------------------------------------
    # تقرير شامل يُجمِّع كل الطرق معًا
    # ------------------------------------------------------------------
    def full_report(
        self,
        reference_embeddings: np.ndarray,
        production_embeddings: np.ndarray,
        reference_query_lengths: np.ndarray,
        production_query_lengths: np.ndarray,
    ) -> list[DriftReport]:
        reports = [
            self.compute_psi(reference_query_lengths, production_query_lengths),
            self.compute_ks_test(reference_query_lengths, production_query_lengths),
            self.compute_embedding_drift(reference_embeddings, production_embeddings),
        ]
        return reports


def simulate_text_embedding(texts: list[str], base_dim: int = 50, seed: int = 0) -> np.ndarray:
    """
    🟢 دالة محاكاة فقط (EDUCATIONAL SIMULATION): تُولِّد تضمينات اصطناعية
    متّسقة مع محتوى النص (hashing بسيط) بدل استدعاء نموذج embedding حقيقي،
    لأغراض هذا المثال التعليمي فقط. في الإنتاج الفعلي، استبدل هذه الدالة
    بنموذج تضمين حقيقي (مثل sentence-transformers أو OpenAI embeddings API).
    """
    rng = np.random.default_rng(seed)
    embeddings = []
    for text in texts:
        text_hash = hash(text) % (2**32)
        local_rng = np.random.default_rng(text_hash)
        embeddings.append(local_rng.normal(0, 1, base_dim))
    return np.array(embeddings)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # سيناريو محاكاة: إطلاق خط منتجات جديد يُسبب انحرافًا تدريجيًا
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)

    # الفترة المرجعية: استفسارات عن منتجات "إلكترونيات" نمطية
    reference_lengths = rng.normal(loc=45, scale=12, size=2000).clip(5, None)
    reference_texts = [f"سؤال عن شاحن أو سماعة رقم {i}" for i in range(2000)]
    reference_embeddings = simulate_text_embedding(reference_texts, seed=1)

    # فترة الإنتاج: بعد إطلاق خط منتجات "أجهزة منزلية ذكية" جديد تمامًا
    production_lengths = rng.normal(loc=68, scale=20, size=800).clip(5, None)
    production_texts = [f"سؤال عن ربط جهاز منزلي ذكي بالتطبيق رقم {i}" for i in range(800)]
    production_embeddings = simulate_text_embedding(production_texts, seed=2)

    detector = DriftDetector()
    reports = detector.full_report(
        reference_embeddings=reference_embeddings,
        production_embeddings=production_embeddings,
        reference_query_lengths=reference_lengths,
        production_query_lengths=production_lengths,
    )

    print("=== تقرير كشف الانحراف الشامل ===\n")
    for report in reports:
        print(report)
        if report.details:
            print(f"   تفاصيل إضافية: {report.details}")
        print()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-62-chapter-62.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
مثال E2E: مراقبة انحراف أسبوعية على مدى 20 أسبوعًا
الحالة: 🟢 كود تعليمي يُحاكي سيناريو إنتاجيًا متكاملاً
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class WeeklyMonitoringResult:
    week: int
    psi_score: float
    embedding_auc: float
    escalation_rate: float  # مقياس وكيل (proxy) لـ Concept Drift
    overall_alert: bool


def simulate_weekly_monitoring(n_weeks: int = 20, drift_start_week: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    detector = DriftDetector()

    # الأسبوع صفر يُعرَّف كمرجع (baseline) ثابت طوال التجربة
    reference_lengths = rng.normal(45, 12, 1500).clip(5, None)
    reference_texts = [f"استفسار إلكترونيات نمطي {i}" for i in range(1500)]
    reference_embeddings = simulate_text_embedding(reference_texts, seed=100)

    results = []
    for week in range(1, n_weeks + 1):
        # محاكاة انحراف تدريجي يبدأ من drift_start_week
        drift_intensity = max(0, (week - drift_start_week) / (n_weeks - drift_start_week))

        prod_lengths = rng.normal(
            45 + drift_intensity * 25,   # الطول يزداد تدريجيًا
            12 + drift_intensity * 8,
            400,
        ).clip(5, None)

        n_new_topic = int(400 * drift_intensity)
        n_old_topic = 400 - n_new_topic
        prod_texts = (
            [f"استفسار إلكترونيات نمطي {i}" for i in range(n_old_topic)]
            + [f"استفسار منتج ذكي جديد {i}" for i in range(n_new_topic)]
        )
        prod_embeddings = simulate_text_embedding(prod_texts, seed=200 + week)

        psi_report = detector.compute_psi(reference_lengths, prod_lengths)
        emb_report = detector.compute_embedding_drift(reference_embeddings, prod_embeddings)

        # مقياس وكيل لـ Concept Drift: معدل التصعيد يرتفع مع الانحراف
        escalation_rate = 0.05 + drift_intensity * 0.25 + rng.normal(0, 0.01)
        escalation_rate = max(0, min(1, escalation_rate))

        overall_alert = (
            psi_report.severity != DriftSeverity.NONE
            or emb_report.severity != DriftSeverity.NONE
            or escalation_rate > 0.20
        )

        results.append(WeeklyMonitoringResult(
            week=week,
            psi_score=psi_report.score,
            embedding_auc=emb_report.score,
            escalation_rate=round(escalation_rate, 3),
            overall_alert=overall_alert,
        ))

    return pd.DataFrame([r.__dict__ for r in results])


if __name__ == "__main__":
    df = simulate_weekly_monitoring()
    print(df.to_string(index=False))

    first_alert_week = df[df["overall_alert"]]["week"].min()
    print(f"\n🚨 أول أسبوع يُطلق فيه إنذار انحراف: الأسبوع {first_alert_week}")
    print("   (بينما الشكوى البشرية الفعلية في قصتنا الافتتاحية لم تصل إلا بعد أسابيع إضافية كثيرة)")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-62-chapter-62.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — كشف انحراف البيانات والمفهوم — الحارس الصامت لجودة النموذج في الإنتاج
Large Language Model Engineering — Chapter 62 — Deployment Llmops

Book slug : llm-engineering-69-chapters
Chapter   : ch-62-chapter-62 (lesson 62)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_62.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 62: الفصل 62: كشف انحراف البيانات والمفهوم — الحارس الصامت لجودة النموذج في الإنتاج."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_drift() -> None:
    baseline, current = ([0.5, 0.3, 0.2], [0.2, 0.3, 0.5])
    epsilon = 1e-06
    psi = sum(((actual - expected) * math.log((actual + epsilon) / (expected + epsilon)) for expected, actual in zip(baseline, current)))
    print('baseline distribution:', baseline)
    print('current distribution:', current)
    print('population stability index:', round(psi, 4))
    print('alert threshold must be calibrated against historical data.')

if __name__ == "__main__":
    print("Chapter 62 lab: Detect a change in token-frequency distributions")
    run_drift()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-11-deployment-llmops/chapter-62.md
# ────────────────────────────────────────────────────────────────────────
"""
الفصل 62: كشف انحراف البيانات والمفهوم
الحالة: 🟢 كود تعليمي مُحاكٍ (educational simulation) — يُحاكي خط أنابيب كشف
         انحراف واقعي، ومناسب للتشغيل الفعلي على بيانات حقيقية بعد استبدال
         دالة توليد التضمينات بنموذج embedding حقيقي.
يتطلب: numpy, scipy, scikit-learn, pandas
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA


class DriftSeverity(Enum):
    NONE = "لا يوجد انحراف"
    MODERATE = "انحراف متوسط - يستحق المراقبة"
    SEVERE = "انحراف كبير - يستوجب تدخلاً"


@dataclass
class DriftReport:
    """تقرير انحراف موحّد يُنتج من كل طريقة كشف."""
    method_name: str
    score: float
    severity: DriftSeverity
    details: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"[{self.method_name}] الدرجة={self.score:.4f} "
                f"الخطورة={self.severity.value}")


class DriftDetector:
    """
    كاشف انحراف متعدد الطرق لمراقبة أنظمة LLM الإنتاجية.

    يُطبِّق ثلاث استراتيجيات مكمّلة:
    1. PSI على الخصائص العددية المُشتقّة من النص.
    2. اختبار كولموغوروف-سميرنوف للمتغيرات المستمرة.
    3. كشف قائم على التصنيف (Domain Classifier) على التضمينات.
    """

    def __init__(
        self,
        psi_moderate_threshold: float = 0.1,
        psi_severe_threshold: float = 0.25,
        classifier_auc_moderate_threshold: float = 0.60,
        classifier_auc_severe_threshold: float = 0.75,
    ):
        self.psi_moderate = psi_moderate_threshold
        self.psi_severe = psi_severe_threshold
        self.clf_moderate = classifier_auc_moderate_threshold
        self.clf_severe = classifier_auc_severe_threshold

    # ------------------------------------------------------------------
    # 1) PSI على خاصية عددية (مثلاً: طول الاستعلام، عدد الرموز، إلخ)
    # ------------------------------------------------------------------
    def compute_psi(
        self,
        reference: np.ndarray,
        production: np.ndarray,
        n_bins: int = 10,
    ) -> DriftReport:
        """
        يحسب PSI لخاصية عددية واحدة بين توزيعين.
        reference: عيّنة من الفترة المرجعية (baseline).
        production: عيّنة من فترة الإنتاج الحالية.
        """
        # بناء الحاويات (bins) اعتمادًا على توزيع المرجع فقط
        bin_edges = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
        bin_edges[0], bin_edges[-1] = -np.inf, np.inf  # لتغطية القيم المتطرفة

        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        prod_counts, _ = np.histogram(production, bins=bin_edges)

        # تحويل إلى نسب مع حماية من القسمة على صفر
        epsilon = 1e-6
        ref_pct = ref_counts / len(reference) + epsilon
        prod_pct = prod_counts / len(production) + epsilon

        psi_value = float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))

        if psi_value < self.psi_moderate:
            severity = DriftSeverity.NONE
        elif psi_value < self.psi_severe:
            severity = DriftSeverity.MODERATE
        else:
            severity = DriftSeverity.SEVERE

        return DriftReport(
            method_name="PSI",
            score=psi_value,
            severity=severity,
            details={"n_bins": n_bins, "ref_size": len(reference), "prod_size": len(production)},
        )

    # ------------------------------------------------------------------
    # 2) اختبار كولموغوروف-سميرنوف لمتغيّر مستمر
    # ------------------------------------------------------------------
    def compute_ks_test(
        self,
        reference: np.ndarray,
        production: np.ndarray,
        alpha: float = 0.01,
    ) -> DriftReport:
        """يكتشف الفرق بين توزيعين مستمرين عبر اختبار KS الإحصائي."""
        ks_statistic, p_value = stats.ks_2samp(reference, production)

        if p_value >= alpha:
            severity = DriftSeverity.NONE
        elif p_value >= alpha / 10:
            severity = DriftSeverity.MODERATE
        else:
            severity = DriftSeverity.SEVERE

        return DriftReport(
            method_name="KS-Test",
            score=float(ks_statistic),
            severity=severity,
            details={"p_value": float(p_value), "alpha": alpha},
        )

    # ------------------------------------------------------------------
    # 3) الكشف القائم على التصنيف (Classifier-Based / Domain Classifier)
    #    مُستوحى من Rabanser et al. "Failing Loudly" (NeurIPS 2019)
    # ------------------------------------------------------------------
    def compute_embedding_drift(
        self,
        reference_embeddings: np.ndarray,
        production_embeddings: np.ndarray,
        use_pca: bool = True,
        pca_components: int = 20,
        random_state: int = 42,
    ) -> DriftReport:
        """
        يدرّب مُصنِّفًا للتمييز بين تضمينات الفترتين؛ AUC قريب من 0.5
        يعني عدم وجود انحراف قابل للتمييز، وAUC مرتفع يعني انحرافًا واضحًا.
        """
        if use_pca:
            n_components = min(pca_components, reference_embeddings.shape[1])
            pca = PCA(n_components=n_components, random_state=random_state)
            combined = np.vstack([reference_embeddings, production_embeddings])
            combined_reduced = pca.fit_transform(combined)
            n_ref = len(reference_embeddings)
            reference_embeddings = combined_reduced[:n_ref]
            production_embeddings = combined_reduced[n_ref:]

        X = np.vstack([reference_embeddings, production_embeddings])
        y = np.concatenate([
            np.zeros(len(reference_embeddings)),
            np.ones(len(production_embeddings)),
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=random_state
        )

        clf = LogisticRegression(max_iter=1000, random_state=random_state)
        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)

        if auc < self.clf_moderate:
            severity = DriftSeverity.NONE
        elif auc < self.clf_severe:
            severity = DriftSeverity.MODERATE
        else:
            severity = DriftSeverity.SEVERE

        # أكثر الأبعاد تمييزًا بين الفترتين (للتفسير)
        top_dims = np.argsort(np.abs(clf.coef_[0]))[::-1][:5]

        return DriftReport(
            method_name="Classifier-Based (Domain Classifier)",
            score=float(auc),
            severity=severity,
            details={"top_discriminative_dims": top_dims.tolist(), "used_pca": use_pca},
        )

    # ------------------------------------------------------------------
    # تقرير شامل يُجمِّع كل الطرق معًا
    # ------------------------------------------------------------------
    def full_report(
        self,
        reference_embeddings: np.ndarray,
        production_embeddings: np.ndarray,
        reference_query_lengths: np.ndarray,
        production_query_lengths: np.ndarray,
    ) -> list[DriftReport]:
        reports = [
            self.compute_psi(reference_query_lengths, production_query_lengths),
            self.compute_ks_test(reference_query_lengths, production_query_lengths),
            self.compute_embedding_drift(reference_embeddings, production_embeddings),
        ]
        return reports


def simulate_text_embedding(texts: list[str], base_dim: int = 50, seed: int = 0) -> np.ndarray:
    """
    🟢 دالة محاكاة فقط (EDUCATIONAL SIMULATION): تُولِّد تضمينات اصطناعية
    متّسقة مع محتوى النص (hashing بسيط) بدل استدعاء نموذج embedding حقيقي،
    لأغراض هذا المثال التعليمي فقط. في الإنتاج الفعلي، استبدل هذه الدالة
    بنموذج تضمين حقيقي (مثل sentence-transformers أو OpenAI embeddings API).
    """
    rng = np.random.default_rng(seed)
    embeddings = []
    for text in texts:
        text_hash = hash(text) % (2**32)
        local_rng = np.random.default_rng(text_hash)
        embeddings.append(local_rng.normal(0, 1, base_dim))
    return np.array(embeddings)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # سيناريو محاكاة: إطلاق خط منتجات جديد يُسبب انحرافًا تدريجيًا
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)

    # الفترة المرجعية: استفسارات عن منتجات "إلكترونيات" نمطية
    reference_lengths = rng.normal(loc=45, scale=12, size=2000).clip(5, None)
    reference_texts = [f"سؤال عن شاحن أو سماعة رقم {i}" for i in range(2000)]
    reference_embeddings = simulate_text_embedding(reference_texts, seed=1)

    # فترة الإنتاج: بعد إطلاق خط منتجات "أجهزة منزلية ذكية" جديد تمامًا
    production_lengths = rng.normal(loc=68, scale=20, size=800).clip(5, None)
    production_texts = [f"سؤال عن ربط جهاز منزلي ذكي بالتطبيق رقم {i}" for i in range(800)]
    production_embeddings = simulate_text_embedding(production_texts, seed=2)

    detector = DriftDetector()
    reports = detector.full_report(
        reference_embeddings=reference_embeddings,
        production_embeddings=production_embeddings,
        reference_query_lengths=reference_lengths,
        production_query_lengths=production_lengths,
    )

    print("=== تقرير كشف الانحراف الشامل ===\n")
    for report in reports:
        print(report)
        if report.details:
            print(f"   تفاصيل إضافية: {report.details}")
        print()

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-11-deployment-llmops/chapter-62.md
# ────────────────────────────────────────────────────────────────────────
"""
مثال E2E: مراقبة انحراف أسبوعية على مدى 20 أسبوعًا
الحالة: 🟢 كود تعليمي يُحاكي سيناريو إنتاجيًا متكاملاً
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class WeeklyMonitoringResult:
    week: int
    psi_score: float
    embedding_auc: float
    escalation_rate: float  # مقياس وكيل (proxy) لـ Concept Drift
    overall_alert: bool


def simulate_weekly_monitoring(n_weeks: int = 20, drift_start_week: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    detector = DriftDetector()

    # الأسبوع صفر يُعرَّف كمرجع (baseline) ثابت طوال التجربة
    reference_lengths = rng.normal(45, 12, 1500).clip(5, None)
    reference_texts = [f"استفسار إلكترونيات نمطي {i}" for i in range(1500)]
    reference_embeddings = simulate_text_embedding(reference_texts, seed=100)

    results = []
    for week in range(1, n_weeks + 1):
        # محاكاة انحراف تدريجي يبدأ من drift_start_week
        drift_intensity = max(0, (week - drift_start_week) / (n_weeks - drift_start_week))

        prod_lengths = rng.normal(
            45 + drift_intensity * 25,   # الطول يزداد تدريجيًا
            12 + drift_intensity * 8,
            400,
        ).clip(5, None)

        n_new_topic = int(400 * drift_intensity)
        n_old_topic = 400 - n_new_topic
        prod_texts = (
            [f"استفسار إلكترونيات نمطي {i}" for i in range(n_old_topic)]
            + [f"استفسار منتج ذكي جديد {i}" for i in range(n_new_topic)]
        )
        prod_embeddings = simulate_text_embedding(prod_texts, seed=200 + week)

        psi_report = detector.compute_psi(reference_lengths, prod_lengths)
        emb_report = detector.compute_embedding_drift(reference_embeddings, prod_embeddings)

        # مقياس وكيل لـ Concept Drift: معدل التصعيد يرتفع مع الانحراف
        escalation_rate = 0.05 + drift_intensity * 0.25 + rng.normal(0, 0.01)
        escalation_rate = max(0, min(1, escalation_rate))

        overall_alert = (
            psi_report.severity != DriftSeverity.NONE
            or emb_report.severity != DriftSeverity.NONE
            or escalation_rate > 0.20
        )

        results.append(WeeklyMonitoringResult(
            week=week,
            psi_score=psi_report.score,
            embedding_auc=emb_report.score,
            escalation_rate=round(escalation_rate, 3),
            overall_alert=overall_alert,
        ))

    return pd.DataFrame([r.__dict__ for r in results])


if __name__ == "__main__":
    df = simulate_weekly_monitoring()
    print(df.to_string(index=False))

    first_alert_week = df[df["overall_alert"]]["week"].min()
    print(f"\n🚨 أول أسبوع يُطلق فيه إنذار انحراف: الأسبوع {first_alert_week}")
    print("   (بينما الشكوى البشرية الفعلية في قصتنا الافتتاحية لم تصل إلا بعد أسابيع إضافية كثيرة)")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-62-chapter-62.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 62: الفصل 62: كشف انحراف البيانات والمفهوم — الحارس الصامت لجودة النموذج في الإنتاج."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_drift() -> None:
    baseline, current = ([0.5, 0.3, 0.2], [0.2, 0.3, 0.5])
    epsilon = 1e-06
    psi = sum(((actual - expected) * math.log((actual + epsilon) / (expected + epsilon)) for expected, actual in zip(baseline, current)))
    print('baseline distribution:', baseline)
    print('current distribution:', current)
    print('population stability index:', round(psi, 4))
    print('alert threshold must be calibrated against historical data.')

if __name__ == "__main__":
    print("Chapter 62 lab: Detect a change in token-frequency distributions")
    run_drift()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-62-chapter-62.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
الفصل 62: كشف انحراف البيانات والمفهوم
الحالة: 🟢 كود تعليمي مُحاكٍ (educational simulation) — يُحاكي خط أنابيب كشف
         انحراف واقعي، ومناسب للتشغيل الفعلي على بيانات حقيقية بعد استبدال
         دالة توليد التضمينات بنموذج embedding حقيقي.
يتطلب: numpy, scipy, scikit-learn, pandas
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA


class DriftSeverity(Enum):
    NONE = "لا يوجد انحراف"
    MODERATE = "انحراف متوسط - يستحق المراقبة"
    SEVERE = "انحراف كبير - يستوجب تدخلاً"


@dataclass
class DriftReport:
    """تقرير انحراف موحّد يُنتج من كل طريقة كشف."""
    method_name: str
    score: float
    severity: DriftSeverity
    details: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"[{self.method_name}] الدرجة={self.score:.4f} "
                f"الخطورة={self.severity.value}")


class DriftDetector:
    """
    كاشف انحراف متعدد الطرق لمراقبة أنظمة LLM الإنتاجية.

    يُطبِّق ثلاث استراتيجيات مكمّلة:
    1. PSI على الخصائص العددية المُشتقّة من النص.
    2. اختبار كولموغوروف-سميرنوف للمتغيرات المستمرة.
    3. كشف قائم على التصنيف (Domain Classifier) على التضمينات.
    """

    def __init__(
        self,
        psi_moderate_threshold: float = 0.1,
        psi_severe_threshold: float = 0.25,
        classifier_auc_moderate_threshold: float = 0.60,
        classifier_auc_severe_threshold: float = 0.75,
    ):
        self.psi_moderate = psi_moderate_threshold
        self.psi_severe = psi_severe_threshold
        self.clf_moderate = classifier_auc_moderate_threshold
        self.clf_severe = classifier_auc_severe_threshold

    # ------------------------------------------------------------------
    # 1) PSI على خاصية عددية (مثلاً: طول الاستعلام، عدد الرموز، إلخ)
    # ------------------------------------------------------------------
    def compute_psi(
        self,
        reference: np.ndarray,
        production: np.ndarray,
        n_bins: int = 10,
    ) -> DriftReport:
        """
        يحسب PSI لخاصية عددية واحدة بين توزيعين.
        reference: عيّنة من الفترة المرجعية (baseline).
        production: عيّنة من فترة الإنتاج الحالية.
        """
        # بناء الحاويات (bins) اعتمادًا على توزيع المرجع فقط
        bin_edges = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
        bin_edges[0], bin_edges[-1] = -np.inf, np.inf  # لتغطية القيم المتطرفة

        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        prod_counts, _ = np.histogram(production, bins=bin_edges)

        # تحويل إلى نسب مع حماية من القسمة على صفر
        epsilon = 1e-6
        ref_pct = ref_counts / len(reference) + epsilon
        prod_pct = prod_counts / len(production) + epsilon

        psi_value = float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))

        if psi_value < self.psi_moderate:
            severity = DriftSeverity.NONE
        elif psi_value < self.psi_severe:
            severity = DriftSeverity.MODERATE
        else:
            severity = DriftSeverity.SEVERE

        return DriftReport(
            method_name="PSI",
            score=psi_value,
            severity=severity,
            details={"n_bins": n_bins, "ref_size": len(reference), "prod_size": len(production)},
        )

    # ------------------------------------------------------------------
    # 2) اختبار كولموغوروف-سميرنوف لمتغيّر مستمر
    # ------------------------------------------------------------------
    def compute_ks_test(
        self,
        reference: np.ndarray,
        production: np.ndarray,
        alpha: float = 0.01,
    ) -> DriftReport:
        """يكتشف الفرق بين توزيعين مستمرين عبر اختبار KS الإحصائي."""
        ks_statistic, p_value = stats.ks_2samp(reference, production)

        if p_value >= alpha:
            severity = DriftSeverity.NONE
        elif p_value >= alpha / 10:
            severity = DriftSeverity.MODERATE
        else:
            severity = DriftSeverity.SEVERE

        return DriftReport(
            method_name="KS-Test",
            score=float(ks_statistic),
            severity=severity,
            details={"p_value": float(p_value), "alpha": alpha},
        )

    # ------------------------------------------------------------------
    # 3) الكشف القائم على التصنيف (Classifier-Based / Domain Classifier)
    #    مُستوحى من Rabanser et al. "Failing Loudly" (NeurIPS 2019)
    # ------------------------------------------------------------------
    def compute_embedding_drift(
        self,
        reference_embeddings: np.ndarray,
        production_embeddings: np.ndarray,
        use_pca: bool = True,
        pca_components: int = 20,
        random_state: int = 42,
    ) -> DriftReport:
        """
        يدرّب مُصنِّفًا للتمييز بين تضمينات الفترتين؛ AUC قريب من 0.5
        يعني عدم وجود انحراف قابل للتمييز، وAUC مرتفع يعني انحرافًا واضحًا.
        """
        if use_pca:
            n_components = min(pca_components, reference_embeddings.shape[1])
            pca = PCA(n_components=n_components, random_state=random_state)
            combined = np.vstack([reference_embeddings, production_embeddings])
            combined_reduced = pca.fit_transform(combined)
            n_ref = len(reference_embeddings)
            reference_embeddings = combined_reduced[:n_ref]
            production_embeddings = combined_reduced[n_ref:]

        X = np.vstack([reference_embeddings, production_embeddings])
        y = np.concatenate([
            np.zeros(len(reference_embeddings)),
            np.ones(len(production_embeddings)),
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=random_state
        )

        clf = LogisticRegression(max_iter=1000, random_state=random_state)
        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)

        if auc < self.clf_moderate:
            severity = DriftSeverity.NONE
        elif auc < self.clf_severe:
            severity = DriftSeverity.MODERATE
        else:
            severity = DriftSeverity.SEVERE

        # أكثر الأبعاد تمييزًا بين الفترتين (للتفسير)
        top_dims = np.argsort(np.abs(clf.coef_[0]))[::-1][:5]

        return DriftReport(
            method_name="Classifier-Based (Domain Classifier)",
            score=float(auc),
            severity=severity,
            details={"top_discriminative_dims": top_dims.tolist(), "used_pca": use_pca},
        )

    # ------------------------------------------------------------------
    # تقرير شامل يُجمِّع كل الطرق معًا
    # ------------------------------------------------------------------
    def full_report(
        self,
        reference_embeddings: np.ndarray,
        production_embeddings: np.ndarray,
        reference_query_lengths: np.ndarray,
        production_query_lengths: np.ndarray,
    ) -> list[DriftReport]:
        reports = [
            self.compute_psi(reference_query_lengths, production_query_lengths),
            self.compute_ks_test(reference_query_lengths, production_query_lengths),
            self.compute_embedding_drift(reference_embeddings, production_embeddings),
        ]
        return reports


def simulate_text_embedding(texts: list[str], base_dim: int = 50, seed: int = 0) -> np.ndarray:
    """
    🟢 دالة محاكاة فقط (EDUCATIONAL SIMULATION): تُولِّد تضمينات اصطناعية
    متّسقة مع محتوى النص (hashing بسيط) بدل استدعاء نموذج embedding حقيقي،
    لأغراض هذا المثال التعليمي فقط. في الإنتاج الفعلي، استبدل هذه الدالة
    بنموذج تضمين حقيقي (مثل sentence-transformers أو OpenAI embeddings API).
    """
    rng = np.random.default_rng(seed)
    embeddings = []
    for text in texts:
        text_hash = hash(text) % (2**32)
        local_rng = np.random.default_rng(text_hash)
        embeddings.append(local_rng.normal(0, 1, base_dim))
    return np.array(embeddings)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # سيناريو محاكاة: إطلاق خط منتجات جديد يُسبب انحرافًا تدريجيًا
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)

    # الفترة المرجعية: استفسارات عن منتجات "إلكترونيات" نمطية
    reference_lengths = rng.normal(loc=45, scale=12, size=2000).clip(5, None)
    reference_texts = [f"سؤال عن شاحن أو سماعة رقم {i}" for i in range(2000)]
    reference_embeddings = simulate_text_embedding(reference_texts, seed=1)

    # فترة الإنتاج: بعد إطلاق خط منتجات "أجهزة منزلية ذكية" جديد تمامًا
    production_lengths = rng.normal(loc=68, scale=20, size=800).clip(5, None)
    production_texts = [f"سؤال عن ربط جهاز منزلي ذكي بالتطبيق رقم {i}" for i in range(800)]
    production_embeddings = simulate_text_embedding(production_texts, seed=2)

    detector = DriftDetector()
    reports = detector.full_report(
        reference_embeddings=reference_embeddings,
        production_embeddings=production_embeddings,
        reference_query_lengths=reference_lengths,
        production_query_lengths=production_lengths,
    )

    print("=== تقرير كشف الانحراف الشامل ===\n")
    for report in reports:
        print(report)
        if report.details:
            print(f"   تفاصيل إضافية: {report.details}")
        print()

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-62-chapter-62.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
"""
مثال E2E: مراقبة انحراف أسبوعية على مدى 20 أسبوعًا
الحالة: 🟢 كود تعليمي يُحاكي سيناريو إنتاجيًا متكاملاً
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class WeeklyMonitoringResult:
    week: int
    psi_score: float
    embedding_auc: float
    escalation_rate: float  # مقياس وكيل (proxy) لـ Concept Drift
    overall_alert: bool


def simulate_weekly_monitoring(n_weeks: int = 20, drift_start_week: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    detector = DriftDetector()

    # الأسبوع صفر يُعرَّف كمرجع (baseline) ثابت طوال التجربة
    reference_lengths = rng.normal(45, 12, 1500).clip(5, None)
    reference_texts = [f"استفسار إلكترونيات نمطي {i}" for i in range(1500)]
    reference_embeddings = simulate_text_embedding(reference_texts, seed=100)

    results = []
    for week in range(1, n_weeks + 1):
        # محاكاة انحراف تدريجي يبدأ من drift_start_week
        drift_intensity = max(0, (week - drift_start_week) / (n_weeks - drift_start_week))

        prod_lengths = rng.normal(
            45 + drift_intensity * 25,   # الطول يزداد تدريجيًا
            12 + drift_intensity * 8,
            400,
        ).clip(5, None)

        n_new_topic = int(400 * drift_intensity)
        n_old_topic = 400 - n_new_topic
        prod_texts = (
            [f"استفسار إلكترونيات نمطي {i}" for i in range(n_old_topic)]
            + [f"استفسار منتج ذكي جديد {i}" for i in range(n_new_topic)]
        )
        prod_embeddings = simulate_text_embedding(prod_texts, seed=200 + week)

        psi_report = detector.compute_psi(reference_lengths, prod_lengths)
        emb_report = detector.compute_embedding_drift(reference_embeddings, prod_embeddings)

        # مقياس وكيل لـ Concept Drift: معدل التصعيد يرتفع مع الانحراف
        escalation_rate = 0.05 + drift_intensity * 0.25 + rng.normal(0, 0.01)
        escalation_rate = max(0, min(1, escalation_rate))

        overall_alert = (
            psi_report.severity != DriftSeverity.NONE
            or emb_report.severity != DriftSeverity.NONE
            or escalation_rate > 0.20
        )

        results.append(WeeklyMonitoringResult(
            week=week,
            psi_score=psi_report.score,
            embedding_auc=emb_report.score,
            escalation_rate=round(escalation_rate, 3),
            overall_alert=overall_alert,
        ))

    return pd.DataFrame([r.__dict__ for r in results])


if __name__ == "__main__":
    df = simulate_weekly_monitoring()
    print(df.to_string(index=False))

    first_alert_week = df[df["overall_alert"]]["week"].min()
    print(f"\n🚨 أول أسبوع يُطلق فيه إنذار انحراف: الأسبوع {first_alert_week}")
    print("   (بينما الشكوى البشرية الفعلية في قصتنا الافتتاحية لم تصل إلا بعد أسابيع إضافية كثيرة)")
