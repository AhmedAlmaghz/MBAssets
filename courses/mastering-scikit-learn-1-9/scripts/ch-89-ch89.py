#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — مخطط خط الأنابيب كاملاً
Mastering scikit-learn 1.9 — Chapter 89 — Pipelines

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-89-ch89 (lesson 89)
Code files: 12 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-89-ch89.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — مخطط خط الأنابيب كاملاً
Mastering scikit-learn 1.9 — Chapter 89 — Pipelines

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-89-ch89 (lesson 89)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part10/ch89.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def make_customers(n_rows=3500, seed=0):
    rng = np.random.default_rng(seed)
    customers = pd.DataFrame({
        "monthly_spend": rng.lognormal(5.2, 0.55, n_rows).round(2),
        "tenure_months": rng.integers(1, 61, n_rows),
        "support_tickets": rng.poisson(2.2, n_rows),
        "avg_session_min": rng.gamma(4.0, 5.0, n_rows).round(1),
        "discount_ratio": np.clip(rng.beta(2.0, 3.0, n_rows), 0, 1).round(3),
        "last_seen_days": rng.integers(0, 210, n_rows),
        "nps_score": rng.integers(0, 11, n_rows),
    })
    customers["channel"] = rng.choice(
        ["mobile", "web", "store"], n_rows, p=[0.52, 0.33, 0.15])
    customers["region"] = rng.choice(
        ["north", "south", "central"], n_rows, p=[0.42, 0.33, 0.25])
    customers["signup_at"] = pd.to_datetime("2021-01-01") + pd.to_timedelta(
        rng.integers(0, 1500, n_rows), unit="D")

    # القيم المفقودة: ليست خطأً في الإدخال بل واقع الاستخراج من ثلاثة مصادر
    customers.loc[rng.choice(n_rows, int(0.12 * n_rows), replace=False),
                  "monthly_spend"] = np.nan
    customers.loc[rng.choice(n_rows, int(0.07 * n_rows), replace=False),
                  "support_tickets"] = np.nan
    customers.loc[rng.choice(n_rows, int(0.05 * n_rows), replace=False),
                  "channel"] = None

    def centered(col):
        median = customers[col].median()
        return (customers[col].fillna(median) - median) / customers[col].std()

    # الهدف: خطّ لوجستي مبني على الخصائص نفسها مع ضجيج، فنعرف جودته نظرياً
    logit = (-2.70
             - 0.85 * (np.log1p(customers["monthly_spend"].fillna(200)) - 5.4) / 0.55
             + 0.30 * centered("tenure_months")
             + 0.55 * centered("support_tickets")
             - 0.40 * centered("avg_session_min")
             + 0.60 * (customers["last_seen_days"] - 105) / 60
             - 0.45 * centered("nps_score")
             + 0.35 * (customers["channel"] == "web").astype(float)
             - 0.30 * (customers["channel"] == "store").astype(float)
             + rng.normal(0, 0.55, n_rows))
    return customers, rng.binomial(1, 1 / (1 + np.exp(-logit)))


customers, churn = make_customers()
X_train, X_test, y_train, y_test = train_test_split(
    customers, churn, test_size=0.2, random_state=0, stratify=churn)

print(f"الصفوف: تدريب {len(X_train)} | اختبار {len(X_test)} | أعمدة {X_train.shape[1]}")
print("أنواع الأعمدة:", {c: str(t) for c, t in X_train.dtypes.items()})
print("قيم مفقودة:", X_train.isna().sum()[lambda s: s > 0].to_dict())
print(f"نسبة الملاءمين: {churn.mean() * 100:.1f} بالمئة من {len(churn)} صفاً")
print(customers.head(3).to_string())

# ────────────────────────────────────────────────────────────────────────
# [002] content/part10/ch89.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


def account_age(col):
    """يحوّل تاريخ التسجيل إلى ميزتين عدديتين: عمر الحساب ومؤشّر حداثته."""
    days = (pd.Timestamp("2025-01-01") - pd.DataFrame(col).iloc[:, 0]).dt.days
    return np.column_stack([days / 365.0, (days < 540).astype(float)])


def date_feature_names(estimator, input_features):
    return ["account_age_years", "joined_in_18_months"]


def build_pipeline():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("encode", OneHotEncoder(handle_unknown="ignore",
                                                     sparse_output=False))])
    datelike = FunctionTransformer(account_age, feature_names_out=date_feature_names)
    features = ColumnTransformer([
        ("num", numeric, make_column_selector(dtype_include=np.number)),
        ("cat", categorical,
         make_column_selector(dtype_include=["object", "category", "string"])),
        ("date", datelike, make_column_selector(dtype_include="datetime64[ns]")),
    ])
    return Pipeline([("features", features),
                     ("model", LogisticRegression(max_iter=3000))])


baseline = build_pipeline().fit(X_train, y_train)
feature_names = baseline.named_steps["features"].get_feature_names_out()
scaler_in = (baseline.named_steps["features"].named_transformers_["num"]
             .named_steps["scale"])

print(f"أعمدةFeatures: {X_train.shape[1]} | خصائص بعد المعالجة: {len(feature_names)}")
print("أسماء الخصائص:", list(feature_names))
print(f"صفوف رآها المطبِّع داخل الأنبوب: {scaler_in.n_samples_seen_:.0f}")
print("تكرارات المحسّن:", baseline.named_steps["model"].n_iter_)
print(f"دقة التدريب: {baseline.score(X_train, y_train):.4f} | "
      f"دقة الاختبار: {baseline.score(X_test, y_test):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part10/ch89.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier

search_space = [
    {"model": [LogisticRegression(max_iter=3000)],
     "model__C": [0.5, 5.0],
     "features__num__impute__strategy": ["median", "mean"]},
    {"model": [HistGradientBoostingClassifier(random_state=0, max_iter=50)],
     "model__max_depth": [3, 5],
     "model__learning_rate": [0.05, 0.2],
     "features__num__impute__strategy": ["median", "mean"]},
]

search = GridSearchCV(
    build_pipeline(),
    search_space,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
    scoring=["roc_auc", "average_precision", "neg_log_loss"],
    refit="roc_auc",
    n_jobs=-1,
    return_train_score=True,
    error_score="raise",
)
search.fit(X_train, y_train)

n_candidates = len(search.cv_results_["params"])
print(f"عدد التركيبات: {n_candidates} | طيّات لكل تركيبة: 3 "
      f"| عمليات تدريب فعلية: {n_candidates * 3 + 1}")
print(f"أفضل درجة تحقق: {search.best_score_:.4f}")
print("أفضل بارامترات:", {k: (v.__class__.__name__ if hasattr(v, "fit") else v)
                          for k, v in search.best_params_.items()})
print("مقياس إعادة التدريب:", search.refit)

# ────────────────────────────────────────────────────────────────────────
# [004] content/part10/ch89.md
# ────────────────────────────────────────────────────────────────────────
cv = search.cv_results_
audit = pd.DataFrame({
    "model": [p["model"].__class__.__name__ for p in cv["params"]],
    "tuning": [p.get("model__C", p.get("model__learning_rate")) for p in cv["params"]],
    "depth": [p.get("model__max_depth", np.nan) for p in cv["params"]],
    "impute": [p["features__num__impute__strategy"] for p in cv["params"]],
    "train_auc": cv["mean_train_roc_auc"],
    "test_auc": cv["mean_test_roc_auc"],
    "std_auc": cv["std_test_roc_auc"],
    "test_ap": cv["mean_test_average_precision"],
    "test_nll": cv["mean_test_neg_log_loss"],
    "fit_s": cv["mean_fit_time"],
}).sort_values("test_auc", ascending=False).reset_index(drop=True)
audit.insert(0, "rank", audit.index + 1)
print(audit.round(4).to_string(index=False))
print("\nفارق التدريب-التحقق للفائز: "
      f"{audit.loc[0, 'train_auc'] - audit.loc[0, 'test_auc']:+.4f}")
gaps = audit.train_auc - audit.test_auc
print("تركيبات يتجاوز فارقها ٠٫٠٥:", int((gaps > 0.05).sum()), "من", len(audit))
print("أكبر فارق في الجدول:", round(float(gaps.max()), 4),
      "لصالح:", audit.loc[int(gaps.idxmax()), "model"])

# ────────────────────────────────────────────────────────────────────────
# [005] content/part10/ch89.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import (classification_report, confusion_matrix,
                             average_precision_score, brier_score_loss)

COST_FALSE_POSITIVE = 1.0
COST_FALSE_NEGATIVE = 8.0
final_model = search.best_estimator_
y_pred = final_model.predict(X_test)
y_proba = final_model.predict_proba(X_test)[:, 1]
tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

print(classification_report(y_test, y_pred, target_names=["stayed", "churned"],
                            digits=3, zero_division=0))
print(f"سلبي صحيح {tn} | موجب خاطئ {fp} | سالب خاطئ {fn} | موجب صحيح {tp}")
print(f"الاسترجاع عند العتبة ٠٫٥: {tp / (tp + fn):.4f}")
print(f"متوسط الدقة: {average_precision_score(y_test, y_proba):.4f}")
print(f"درجة براير: {brier_score_loss(y_test, y_proba):.4f}")

cost = COST_FALSE_POSITIVE * fp + COST_FALSE_NEGATIVE * fn
print(f"التكلفة عند ٠٫٥: {cost:.0f} وحدة على {len(y_test)} صفاً "
      f"({cost / len(y_test):.3f} للصف)")
print(f"تكلفة عدم الفعل: "
      f"{COST_FALSE_NEGATIVE * int((y_test == 1).sum()):.0f} وحدة")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part10/ch89.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import make_scorer
from sklearn.model_selection import TunedThresholdClassifierCV


def campaign_cost(y_true, y_pred):
    """تكلفة الحملة: عرض هدر لواحد، وعميل مفقود بثمان."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    false_positives = int(((y_pred == 1) & (y_true == 0)).sum())
    false_negatives = int(((y_pred == 0) & (y_true == 1)).sum())
    return (COST_FALSE_POSITIVE * false_positives
            + COST_FALSE_NEGATIVE * false_negatives)


tuned = TunedThresholdClassifierCV(
    estimator=final_model,
    scoring=make_scorer(campaign_cost, greater_is_better=False),
    cv=3, n_jobs=None, random_state=0, store_cv_results=True)
tuned.fit(X_train, y_train)

thresholds = tuned.cv_results_["thresholds"]
costs = -tuned.cv_results_["scores"]
y_tuned = tuned.predict(X_test)
tn2, fp2, fn2, tp2 = confusion_matrix(y_test, y_tuned, labels=[0, 1]).ravel()
cost_tuned = COST_FALSE_POSITIVE * fp2 + COST_FALSE_NEGATIVE * fn2

print(f"أفضل عتبة: {tuned.best_threshold_:.4f} | "
      f"أفضل درجة (سالب التكلفة): {tuned.best_score_:.2f}")
print(f"عيّنات العتبة المفحوصة: {len(thresholds)}")
print(f"التكلفة عند ٠٫٥: {cost:.0f} | عند العتبة المضبوطة: {cost_tuned:.0f} "
      f"| التوفير: {(1 - cost_tuned / cost) * 100:.1f} بالمئة")
print(f"الاسترجاع قفز من {tp / (tp + fn):.3f} إلى {tp2 / (tp2 + fn2):.3f}")
print(f"عروض الهدر ارتفعت من {fp} إلى {fp2} — وهذا هو الثمن المدفوع")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part10/ch89.md
# ────────────────────────────────────────────────────────────────────────
import io
import cloudpickle
from sklearn.frozen import FrozenEstimator

payload = io.BytesIO()
cloudpickle.dump(final_model, payload)
size_kb = len(payload.getvalue()) / 1024
payload.seek(0)
restored = cloudpickle.load(payload)

frozen = FrozenEstimator(final_model)
frozen.fit(X_test, y_test)          # عملية صامتة لا تفعل شيئاً بالتصميم
print(f"حجم الملف: {size_kb:.1f} كيلوبايت")
print("أسماء الخطوات بعد الاستعادة:", [n for n, _ in restored.steps])
print("تطابق التنبؤات بعد الاستعادة:",
      bool((restored.predict(X_test) == final_model.predict(X_test)).all()))
print("تطابق الاحتمالات:",
      np.allclose(restored.predict_proba(X_test), final_model.predict_proba(X_test)))
print("نوع الغلاف:", type(frozen).__name__,
      "| fit ما زالت واجهة مكشوفة:", callable(getattr(frozen, "fit", None)))
print("تنبؤ النموذج المجمَّد مطابق:",
      bool((frozen.predict(X_test) == final_model.predict(X_test)).all()))
print("العتبة المضبوطة تسافر في مستند التسليم:", round(tuned.best_threshold_, 4))

# ────────────────────────────────────────────────────────────────────────
# [008] content/part10/ch89.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([("features", ColumnTransformer([("num", StandardScaler(), "num_cols")])),
                 ("model", LogisticRegression(max_iter=3000))])
pipe.set_params(model__C=2.0)

# ────────────────────────────────────────────────────────────────────────
# [009] content/part10/ch89.md
# ────────────────────────────────────────────────────────────────────────
customers_small, churn_small = make_customers(n_rows=2500, seed=7)
Xs_train, Xs_test, ys_train, ys_test = train_test_split(
    customers_small, churn_small, test_size=0.2, random_state=0, stratify=churn_small)

two_branch = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]),
     make_column_selector(dtype_include=np.number)),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("encode", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False))]),
     make_column_selector(dtype_include=["object", "category", "string"])),
])
pipe_two = Pipeline([("features", two_branch),
                     ("model", LogisticRegression(max_iter=3000))])
pipe_two.fit(Xs_train, ys_train)

names_two = pipe_two.named_steps["features"].get_feature_names_out()
scaler_two = (pipe_two.named_steps["features"].named_transformers_["num"]
              .named_steps["scale"])
print(f"خصائص بعد المعالجة: {len(names_two)}")
print(f"صفوف رآها المطبِّع: {scaler_two.n_samples_seen_:.0f} من {len(Xs_train)} صف تدريب")
print(f"درجة الاختبار: {pipe_two.score(Xs_test, ys_test):.4f}")
print("هل ظهر التاريخ في المخرجات:", any("signup" in n for n in names_two))

# ────────────────────────────────────────────────────────────────────────
# [010] content/part10/ch89.md
# ────────────────────────────────────────────────────────────────────────
with_flag = customers.copy()
with_flag["is_new_account"] = (with_flag["tenure_months"] <= 3).astype(bool)
Xf_train, Xf_test, yf_train, yf_test = train_test_split(
    with_flag, churn, test_size=0.2, random_state=0, stratify=churn)

boolean_branch = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]),
     make_column_selector(dtype_include=np.number)),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("encode", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False))]),
     make_column_selector(dtype_include=["object", "category", "string"])),
    ("date", FunctionTransformer(account_age, feature_names_out=date_feature_names),
     make_column_selector(dtype_include="datetime64[ns]")),
    ("flag", "passthrough", make_column_selector(dtype_include=bool)),
])
pipe_flag = Pipeline([("features", boolean_branch),
                      ("model", LogisticRegression(max_iter=3000))])
pipe_flag.fit(Xf_train, yf_train)

print(list(pipe_flag.named_steps["features"].get_feature_names_out()))
print(f"درجة الاختبار: {pipe_flag.score(Xf_test, yf_test):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [011] content/part10/ch89.md
# ────────────────────────────────────────────────────────────────────────
pipe_demo = Pipeline([("features", ColumnTransformer([("num", StandardScaler(), "num_cols")])),
                      ("model", LogisticRegression(max_iter=3000))])
print("عنوان النموذج داخل الأنبوب:",
      pipe_demo.get_params()["model"].__class__.__name__)
try:
    pipe_demo.fit(X_train[["monthly_spend"]], y_train)
except ValueError as exc:
    print("ValueError:", str(exc).splitlines()[0][:90])
pipe_demo.set_params(model__C=2.0)
print("بعد التصحيح، قيمة C:", pipe_demo.get_params()["model__C"])

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-89-ch89.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def make_customers(n_rows=3500, seed=0):
    rng = np.random.default_rng(seed)
    customers = pd.DataFrame({
        "monthly_spend": rng.lognormal(5.2, 0.55, n_rows).round(2),
        "tenure_months": rng.integers(1, 61, n_rows),
        "support_tickets": rng.poisson(2.2, n_rows),
        "avg_session_min": rng.gamma(4.0, 5.0, n_rows).round(1),
        "discount_ratio": np.clip(rng.beta(2.0, 3.0, n_rows), 0, 1).round(3),
        "last_seen_days": rng.integers(0, 210, n_rows),
        "nps_score": rng.integers(0, 11, n_rows),
    })
    customers["channel"] = rng.choice(
        ["mobile", "web", "store"], n_rows, p=[0.52, 0.33, 0.15])
    customers["region"] = rng.choice(
        ["north", "south", "central"], n_rows, p=[0.42, 0.33, 0.25])
    customers["signup_at"] = pd.to_datetime("2021-01-01") + pd.to_timedelta(
        rng.integers(0, 1500, n_rows), unit="D")

    # القيم المفقودة: ليست خطأً في الإدخال بل واقع الاستخراج من ثلاثة مصادر
    customers.loc[rng.choice(n_rows, int(0.12 * n_rows), replace=False),
                  "monthly_spend"] = np.nan
    customers.loc[rng.choice(n_rows, int(0.07 * n_rows), replace=False),
                  "support_tickets"] = np.nan
    customers.loc[rng.choice(n_rows, int(0.05 * n_rows), replace=False),
                  "channel"] = None

    def centered(col):
        median = customers[col].median()
        return (customers[col].fillna(median) - median) / customers[col].std()

    # الهدف: خطّ لوجستي مبني على الخصائص نفسها مع ضجيج، فنعرف جودته نظرياً
    logit = (-2.70
             - 0.85 * (np.log1p(customers["monthly_spend"].fillna(200)) - 5.4) / 0.55
             + 0.30 * centered("tenure_months")
             + 0.55 * centered("support_tickets")
             - 0.40 * centered("avg_session_min")
             + 0.60 * (customers["last_seen_days"] - 105) / 60
             - 0.45 * centered("nps_score")
             + 0.35 * (customers["channel"] == "web").astype(float)
             - 0.30 * (customers["channel"] == "store").astype(float)
             + rng.normal(0, 0.55, n_rows))
    return customers, rng.binomial(1, 1 / (1 + np.exp(-logit)))


customers, churn = make_customers()
X_train, X_test, y_train, y_test = train_test_split(
    customers, churn, test_size=0.2, random_state=0, stratify=churn)

print(f"الصفوف: تدريب {len(X_train)} | اختبار {len(X_test)} | أعمدة {X_train.shape[1]}")
print("أنواع الأعمدة:", {c: str(t) for c, t in X_train.dtypes.items()})
print("قيم مفقودة:", X_train.isna().sum()[lambda s: s > 0].to_dict())
print(f"نسبة الملاءمين: {churn.mean() * 100:.1f} بالمئة من {len(churn)} صفاً")
print(customers.head(3).to_string())

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-89-ch89.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


def account_age(col):
    """يحوّل تاريخ التسجيل إلى ميزتين عدديتين: عمر الحساب ومؤشّر حداثته."""
    days = (pd.Timestamp("2025-01-01") - pd.DataFrame(col).iloc[:, 0]).dt.days
    return np.column_stack([days / 365.0, (days < 540).astype(float)])


def date_feature_names(estimator, input_features):
    return ["account_age_years", "joined_in_18_months"]


def build_pipeline():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("encode", OneHotEncoder(handle_unknown="ignore",
                                                     sparse_output=False))])
    datelike = FunctionTransformer(account_age, feature_names_out=date_feature_names)
    features = ColumnTransformer([
        ("num", numeric, make_column_selector(dtype_include=np.number)),
        ("cat", categorical,
         make_column_selector(dtype_include=["object", "category", "string"])),
        ("date", datelike, make_column_selector(dtype_include="datetime64[ns]")),
    ])
    return Pipeline([("features", features),
                     ("model", LogisticRegression(max_iter=3000))])


baseline = build_pipeline().fit(X_train, y_train)
feature_names = baseline.named_steps["features"].get_feature_names_out()
scaler_in = (baseline.named_steps["features"].named_transformers_["num"]
             .named_steps["scale"])

print(f"أعمدةFeatures: {X_train.shape[1]} | خصائص بعد المعالجة: {len(feature_names)}")
print("أسماء الخصائص:", list(feature_names))
print(f"صفوف رآها المطبِّع داخل الأنبوب: {scaler_in.n_samples_seen_:.0f}")
print("تكرارات المحسّن:", baseline.named_steps["model"].n_iter_)
print(f"دقة التدريب: {baseline.score(X_train, y_train):.4f} | "
      f"دقة الاختبار: {baseline.score(X_test, y_test):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-89-ch89.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier

search_space = [
    {"model": [LogisticRegression(max_iter=3000)],
     "model__C": [0.5, 5.0],
     "features__num__impute__strategy": ["median", "mean"]},
    {"model": [HistGradientBoostingClassifier(random_state=0, max_iter=50)],
     "model__max_depth": [3, 5],
     "model__learning_rate": [0.05, 0.2],
     "features__num__impute__strategy": ["median", "mean"]},
]

search = GridSearchCV(
    build_pipeline(),
    search_space,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
    scoring=["roc_auc", "average_precision", "neg_log_loss"],
    refit="roc_auc",
    n_jobs=-1,
    return_train_score=True,
    error_score="raise",
)
search.fit(X_train, y_train)

n_candidates = len(search.cv_results_["params"])
print(f"عدد التركيبات: {n_candidates} | طيّات لكل تركيبة: 3 "
      f"| عمليات تدريب فعلية: {n_candidates * 3 + 1}")
print(f"أفضل درجة تحقق: {search.best_score_:.4f}")
print("أفضل بارامترات:", {k: (v.__class__.__name__ if hasattr(v, "fit") else v)
                          for k, v in search.best_params_.items()})
print("مقياس إعادة التدريب:", search.refit)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-89-ch89.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
cv = search.cv_results_
audit = pd.DataFrame({
    "model": [p["model"].__class__.__name__ for p in cv["params"]],
    "tuning": [p.get("model__C", p.get("model__learning_rate")) for p in cv["params"]],
    "depth": [p.get("model__max_depth", np.nan) for p in cv["params"]],
    "impute": [p["features__num__impute__strategy"] for p in cv["params"]],
    "train_auc": cv["mean_train_roc_auc"],
    "test_auc": cv["mean_test_roc_auc"],
    "std_auc": cv["std_test_roc_auc"],
    "test_ap": cv["mean_test_average_precision"],
    "test_nll": cv["mean_test_neg_log_loss"],
    "fit_s": cv["mean_fit_time"],
}).sort_values("test_auc", ascending=False).reset_index(drop=True)
audit.insert(0, "rank", audit.index + 1)
print(audit.round(4).to_string(index=False))
print("\nفارق التدريب-التحقق للفائز: "
      f"{audit.loc[0, 'train_auc'] - audit.loc[0, 'test_auc']:+.4f}")
gaps = audit.train_auc - audit.test_auc
print("تركيبات يتجاوز فارقها ٠٫٠٥:", int((gaps > 0.05).sum()), "من", len(audit))
print("أكبر فارق في الجدول:", round(float(gaps.max()), 4),
      "لصالح:", audit.loc[int(gaps.idxmax()), "model"])

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-89-ch89.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import (classification_report, confusion_matrix,
                             average_precision_score, brier_score_loss)

COST_FALSE_POSITIVE = 1.0
COST_FALSE_NEGATIVE = 8.0
final_model = search.best_estimator_
y_pred = final_model.predict(X_test)
y_proba = final_model.predict_proba(X_test)[:, 1]
tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

print(classification_report(y_test, y_pred, target_names=["stayed", "churned"],
                            digits=3, zero_division=0))
print(f"سلبي صحيح {tn} | موجب خاطئ {fp} | سالب خاطئ {fn} | موجب صحيح {tp}")
print(f"الاسترجاع عند العتبة ٠٫٥: {tp / (tp + fn):.4f}")
print(f"متوسط الدقة: {average_precision_score(y_test, y_proba):.4f}")
print(f"درجة براير: {brier_score_loss(y_test, y_proba):.4f}")

cost = COST_FALSE_POSITIVE * fp + COST_FALSE_NEGATIVE * fn
print(f"التكلفة عند ٠٫٥: {cost:.0f} وحدة على {len(y_test)} صفاً "
      f"({cost / len(y_test):.3f} للصف)")
print(f"تكلفة عدم الفعل: "
      f"{COST_FALSE_NEGATIVE * int((y_test == 1).sum()):.0f} وحدة")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-89-ch89.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import make_scorer
from sklearn.model_selection import TunedThresholdClassifierCV


def campaign_cost(y_true, y_pred):
    """تكلفة الحملة: عرض هدر لواحد، وعميل مفقود بثمان."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    false_positives = int(((y_pred == 1) & (y_true == 0)).sum())
    false_negatives = int(((y_pred == 0) & (y_true == 1)).sum())
    return (COST_FALSE_POSITIVE * false_positives
            + COST_FALSE_NEGATIVE * false_negatives)


tuned = TunedThresholdClassifierCV(
    estimator=final_model,
    scoring=make_scorer(campaign_cost, greater_is_better=False),
    cv=3, n_jobs=None, random_state=0, store_cv_results=True)
tuned.fit(X_train, y_train)

thresholds = tuned.cv_results_["thresholds"]
costs = -tuned.cv_results_["scores"]
y_tuned = tuned.predict(X_test)
tn2, fp2, fn2, tp2 = confusion_matrix(y_test, y_tuned, labels=[0, 1]).ravel()
cost_tuned = COST_FALSE_POSITIVE * fp2 + COST_FALSE_NEGATIVE * fn2

print(f"أفضل عتبة: {tuned.best_threshold_:.4f} | "
      f"أفضل درجة (سالب التكلفة): {tuned.best_score_:.2f}")
print(f"عيّنات العتبة المفحوصة: {len(thresholds)}")
print(f"التكلفة عند ٠٫٥: {cost:.0f} | عند العتبة المضبوطة: {cost_tuned:.0f} "
      f"| التوفير: {(1 - cost_tuned / cost) * 100:.1f} بالمئة")
print(f"الاسترجاع قفز من {tp / (tp + fn):.3f} إلى {tp2 / (tp2 + fn2):.3f}")
print(f"عروض الهدر ارتفعت من {fp} إلى {fp2} — وهذا هو الثمن المدفوع")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-89-ch89.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import io
import cloudpickle
from sklearn.frozen import FrozenEstimator

payload = io.BytesIO()
cloudpickle.dump(final_model, payload)
size_kb = len(payload.getvalue()) / 1024
payload.seek(0)
restored = cloudpickle.load(payload)

frozen = FrozenEstimator(final_model)
frozen.fit(X_test, y_test)          # عملية صامتة لا تفعل شيئاً بالتصميم
print(f"حجم الملف: {size_kb:.1f} كيلوبايت")
print("أسماء الخطوات بعد الاستعادة:", [n for n, _ in restored.steps])
print("تطابق التنبؤات بعد الاستعادة:",
      bool((restored.predict(X_test) == final_model.predict(X_test)).all()))
print("تطابق الاحتمالات:",
      np.allclose(restored.predict_proba(X_test), final_model.predict_proba(X_test)))
print("نوع الغلاف:", type(frozen).__name__,
      "| fit ما زالت واجهة مكشوفة:", callable(getattr(frozen, "fit", None)))
print("تنبؤ النموذج المجمَّد مطابق:",
      bool((frozen.predict(X_test) == final_model.predict(X_test)).all()))
print("العتبة المضبوطة تسافر في مستند التسليم:", round(tuned.best_threshold_, 4))

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-89-ch89.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([("features", ColumnTransformer([("num", StandardScaler(), "num_cols")])),
                 ("model", LogisticRegression(max_iter=3000))])
pipe.set_params(model__C=2.0)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-89-ch89.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
customers_small, churn_small = make_customers(n_rows=2500, seed=7)
Xs_train, Xs_test, ys_train, ys_test = train_test_split(
    customers_small, churn_small, test_size=0.2, random_state=0, stratify=churn_small)

two_branch = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]),
     make_column_selector(dtype_include=np.number)),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("encode", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False))]),
     make_column_selector(dtype_include=["object", "category", "string"])),
])
pipe_two = Pipeline([("features", two_branch),
                     ("model", LogisticRegression(max_iter=3000))])
pipe_two.fit(Xs_train, ys_train)

names_two = pipe_two.named_steps["features"].get_feature_names_out()
scaler_two = (pipe_two.named_steps["features"].named_transformers_["num"]
              .named_steps["scale"])
print(f"خصائص بعد المعالجة: {len(names_two)}")
print(f"صفوف رآها المطبِّع: {scaler_two.n_samples_seen_:.0f} من {len(Xs_train)} صف تدريب")
print(f"درجة الاختبار: {pipe_two.score(Xs_test, ys_test):.4f}")
print("هل ظهر التاريخ في المخرجات:", any("signup" in n for n in names_two))

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-89-ch89.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
with_flag = customers.copy()
with_flag["is_new_account"] = (with_flag["tenure_months"] <= 3).astype(bool)
Xf_train, Xf_test, yf_train, yf_test = train_test_split(
    with_flag, churn, test_size=0.2, random_state=0, stratify=churn)

boolean_branch = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]),
     make_column_selector(dtype_include=np.number)),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("encode", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False))]),
     make_column_selector(dtype_include=["object", "category", "string"])),
    ("date", FunctionTransformer(account_age, feature_names_out=date_feature_names),
     make_column_selector(dtype_include="datetime64[ns]")),
    ("flag", "passthrough", make_column_selector(dtype_include=bool)),
])
pipe_flag = Pipeline([("features", boolean_branch),
                      ("model", LogisticRegression(max_iter=3000))])
pipe_flag.fit(Xf_train, yf_train)

print(list(pipe_flag.named_steps["features"].get_feature_names_out()))
print(f"درجة الاختبار: {pipe_flag.score(Xf_test, yf_test):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-89-ch89.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
pipe_demo = Pipeline([("features", ColumnTransformer([("num", StandardScaler(), "num_cols")])),
                      ("model", LogisticRegression(max_iter=3000))])
print("عنوان النموذج داخل الأنبوب:",
      pipe_demo.get_params()["model"].__class__.__name__)
try:
    pipe_demo.fit(X_train[["monthly_spend"]], y_train)
except ValueError as exc:
    print("ValueError:", str(exc).splitlines()[0][:90])
pipe_demo.set_params(model__C=2.0)
print("بعد التصحيح، قيمة C:", pipe_demo.get_params()["model__C"])
