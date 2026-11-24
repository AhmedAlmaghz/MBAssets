#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — خط الأنابيب: عقد أمانة مع البيانات
Mastering scikit-learn 1.9 — Chapter 84 — Pipelines

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-84-ch84 (lesson 84)
Code files: 15 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-84-ch84.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — خط الأنابيب: عقد أمانة مع البيانات
Mastering scikit-learn 1.9 — Chapter 84 — Pipelines

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-84-ch84 (lesson 84)
Code files: 14 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part10/ch84.md
# ────────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

selector = SelectKBest(f_classif, k=10)
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

# ────────────────────────────────────────────────────────────────────────
# [002] content/part10/ch84.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=400, n_features=120, n_informative=10,
                           n_redundant=20, class_sep=0.7, flip_y=0.06,
                           random_state=42)
Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.10, random_state=42,
                                  stratify=y)

scaler_full = StandardScaler().fit(X)                  # الصفوف الـ400 كلها
selector_full = SelectKBest(f_classif, k=10).fit(X, y) # والهدف y كاملاً
keep = selector_full.get_support(indices=True)
leaky = LogisticRegression(max_iter=5000, C=0.3).fit(
    scaler_full.transform(Xa).to_numpy()[:, keep], ya)
print(f"المتسرّبة تعلّمت من {X.shape[0]} صفاً | درجتها: "
      f"{leaky.score(scaler_full.transform(Xb).to_numpy()[:, keep], yb):.4f}")

pipe = Pipeline([("sc", StandardScaler()),
                 ("sel", SelectKBest(f_classif, k=10)),
                 ("clf", LogisticRegression(max_iter=5000, C=0.3))]).fit(Xa, ya)
honest_idx = np.where(pipe.named_steps["sel"].get_support())[0]
print(f"الأنبوب تدرّب على {Xa.shape[0]} صفاً | درجته: {pipe.score(Xb, yb):.4f}")
print("اختار الأنبوب  :", [f"x{i}" for i in honest_idx])
print("اختيار المتسرّب:", [f"x{i}" for i in np.sort(keep)])
print("اختلفا في     :",
      [f"x{i}" for i in sorted(set(honest_idx) ^ set(keep))])

# ────────────────────────────────────────────────────────────────────────
# [003] content/part10/ch84.md
# ────────────────────────────────────────────────────────────────────────
def make_honest():
    return Pipeline([("sc", StandardScaler()),
                     ("sel", SelectKBest(f_classif, k=10)),
                     ("clf", LogisticRegression(max_iter=5000, C=0.3))])

honest_scores, leaky_scores = [], []
for seed in range(30):
    A, B, c, d = train_test_split(X, y, test_size=0.10, random_state=seed,
                                  stratify=y)
    honest_scores.append(make_honest().fit(A, c).score(B, d))
    sc_all = StandardScaler().fit(X)                     # تسرّب
    m = SelectKBest(f_classif, k=10).fit(X, y).get_support()  # تسرّب
    mdl = LogisticRegression(max_iter=5000, C=0.3).fit(
        sc_all.transform(A).to_numpy()[:, m], c)
    leaky_scores.append(mdl.score(sc_all.transform(B).to_numpy()[:, m], d))

h, l = np.array(honest_scores), np.array(leaky_scores)
gain = l - h
print(f"صادق  : المتوسط {h.mean():.4f} | الانحراف {h.std():.4f} "
      f"| المدى [{h.min():.4f}, {h.max():.4f}]")
print(f"متسرّب: المتوسط {l.mean():.4f} | الانحراف {l.std():.4f} "
      f"| المدى [{l.min():.4f}, {l.max():.4f}]")
print(f"متوسط الفارق: {gain.mean():+.4f} | فوق الصفر {int((gain > 0).sum())}"
      f" | تحت الصفر {int((gain < 0).sum())} | متساوٍ {int((gain == 0).sum())} من 30")
print(f"اتّساع المدى: من {(h.max() - h.min()) * 100:.1f} نقطة "
      f"إلى {(l.max() - l.min()) * 100:.1f} نقطة")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part10/ch84.md
# ────────────────────────────────────────────────────────────────────────
sc = pipe.named_steps["sc"]
print(f"صفوف رآها المُطبِّع: {sc.n_samples_seen_:.0f} من أصل {X.shape[0]}")
print(f"متوسّطه للعمود الأول: {sc.mean_[0]:.4f}")
print(f"متوسّطه على كامل البيانات: {scaler_full.mean_[0]:.4f}")
print("أعمدة أبقاها المنتقي:",
      [f"x{i}" for i in np.where(pipe.named_steps['sel'].get_support())[0]])

# ────────────────────────────────────────────────────────────────────────
# [005] content/part10/ch84.md
# ────────────────────────────────────────────────────────────────────────
X_b_t = pipe[:-1].transform(Xb)
manual = pipe.named_steps["clf"].predict(X_b_t)
print("تطابق المسارين:", bool((manual == pipe.predict(Xb)).all()))
print("نوع ناتج التحويل:", type(X_b_t).__name__, "| الشكل:", X_b_t.shape)
print("أسماء الخصائص بعد التحويل:", list(X_b_t.columns)[:4])

# ────────────────────────────────────────────────────────────────────────
# [006] content/part10/ch84.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.pipeline import make_pipeline

auto = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
print("أسماء make_pipeline:", [name for name, _ in auto.steps])

try:
    auto.set_params(clf__C=1.0)          # الاسم الذي ظننّاه منطقياً
except ValueError as exc:
    print("ValueError:", str(exc).splitlines()[0][:86])

auto.set_params(logisticregression__C=1.0)   # الاسم الحقيقي
print("بعد التصحيح:", auto.get_params()["logisticregression__C"])

# ────────────────────────────────────────────────────────────────────────
# [007] content/part10/ch84.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    Pipeline([("sc", StandardScaler()),
              ("clf", LogisticRegression(max_iter=5000))]),
    {"sc__with_mean": [True, False],
     "clf__C": [0.05, 0.3, 2.0]},
    cv=3, n_jobs=-1, return_train_score=True)
grid.fit(Xa, ya)

print("أفضل تركيبة:", grid.best_params_)
print(f"درجة التحقق: {grid.best_score_:.4f}")
print(f"درجة التدريب للفائز: "
      f"{grid.cv_results_['mean_train_score'][grid.best_index_]:.4f}")
print("مفاتيح الشبكة:", [k for k in grid.cv_results_ if k.startswith("param_")])

# ────────────────────────────────────────────────────────────────────────
# [008] content/part10/ch84.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

docs = [
    "free money click now win prize today",
    "limited offer buy cheap discount now",
    "urgent call now about your invoice payment",
    "the deployment failed on the production server",
    "meeting tomorrow at nine about project status",
    "team meeting notes attached for review",
    "please review the report before the deadline",
    "lunch at the cafe near the office tomorrow",
]
labels = np.array([1, 1, 1, 0, 0, 0, 0, 0])

text_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
    ("clf", CalibratedClassifierCV(LinearSVC(C=1.0), method="sigmoid", cv=2)),
])
text_pipeline.fit(docs, labels)

print("حجم مفردات TF-IDF:", len(text_pipeline.named_steps["tfidf"].vocabulary_))
proba = text_pipeline.predict_proba(["claim your free prize now"])
print("الاحتمال:", np.round(proba[0], 3),
      "| الفئة:", text_pipeline.classes_[proba.argmax()])
print("نوع الخطوة الأخيرة:", type(text_pipeline.named_steps["clf"]).__name__)

# ────────────────────────────────────────────────────────────────────────
# [009] content/part10/ch84.md
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
from sklearn.impute import SimpleImputer

frame = pd.DataFrame(Xa[:, :5].copy(),
                     columns=["age", "income", "tenure", "balance", "score"])
frame.loc[frame.sample(frac=0.3, random_state=42).index, "score"] = np.nan

frame_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                       ("clf", LogisticRegression(max_iter=5000))])
frame_pipe.fit(frame, ya)

print("feature_names_in_:", list(frame_pipe.feature_names_in_))
print("n_features_in_:", frame_pipe.n_features_in_)
print("الوسيط المستعاد للعمود المفقود:",
      round(frame_pipe.named_steps["imp"].statistics_[4], 4))
print("أسماء المخرجات:", list(frame_pipe[:-1].get_feature_names_out()))

# ────────────────────────────────────────────────────────────────────────
# [010] content/part10/ch84.md
# ────────────────────────────────────────────────────────────────────────
import io
import joblib

buffer = io.BytesIO()
joblib.dump(pipe, buffer)
buffer.seek(0)
restored = joblib.load(buffer)

print(f"حجم الملف: {len(buffer.getvalue()) / 1024:.1f} كيلوبايت")
print("تطابق التنبؤ بعد الاستعادة:",
      bool((restored.predict(Xb) == pipe.predict(Xb)).all()))
print("نفس أسماء الخطوات:", [n for n, _ in restored.steps])

# ────────────────────────────────────────────────────────────────────────
# [011] content/part10/ch84.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.set_params(clf__C=10.0)

# ────────────────────────────────────────────────────────────────────────
# [012] content/part10/ch84.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

Xf, yf = make_classification(n_samples=500, n_features=30, n_informative=6,
                             random_state=42)
Xa2, Xb2, ya2, yb2 = train_test_split(Xf, yf, test_size=0.2, random_state=42,
                                       stratify=yf)

miss = np.random.default_rng(0).choice(Xa2.shape[0], size=60, replace=False)
Xa2[miss, 3] = np.nan
Xa2[miss, 17] = np.nan

pipe3 = Pipeline([("imputer", SimpleImputer(strategy="median")),
                  ("scaler", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=5000))])
pipe3.fit(Xa2, ya2)

print(f"صفوف رآها المُطبِّع: {pipe3.named_steps['scaler'].n_samples_seen_:.0f}")
print(f"أعمدة بعد الاسترجاع: {pipe3.named_steps['imputer'].statistics_.size}")
print(f"مجموع وسائط الاسترجاع: {pipe3.named_steps['imputer'].statistics_.sum():.3f}")
print(f"درجة الاختبار: {pipe3.score(Xb2, yb2):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [013] content/part10/ch84.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV

pipe4 = Pipeline([("imputer", SimpleImputer(strategy="median")),
                  ("scaler", StandardScaler()),
                  ("sel", SelectKBest(f_classif, k=5)),
                  ("clf", LogisticRegression(max_iter=5000))])

grid2 = GridSearchCV(pipe4,
                     {"imputer__strategy": ["median", "mean"],
                      "clf__C": [0.5, 5.0]},
                     cv=4, return_train_score=True)
grid2.fit(Xa2, ya2)

i = grid2.best_index_
tr = grid2.cv_results_["mean_train_score"][i]
te = grid2.cv_results_["mean_test_score"][i]
print("أفضل تركيبة:", grid2.best_params_)
print(f"تدريب: {tr:.4f} | تحقق: {te:.4f} | الفجوة: {tr - te:+.4f}")

# ────────────────────────────────────────────────────────────────────────
# [014] content/part10/ch84.md
# ────────────────────────────────────────────────────────────────────────
auto_pipe = make_pipeline(StandardScaler(), LogisticRegression())
print("الأسماء الحقيقية:", [n for n, _ in auto_pipe.steps])
try:
    auto_pipe.set_params(clf__C=10.0)
except ValueError as exc:
    print("ValueError:", str(exc).splitlines()[0][:86])
auto_pipe.set_params(logisticregression__C=10.0)
print("بعد التصحيح:", auto_pipe.get_params()["logisticregression__C"])

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-84-ch84.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

selector = SelectKBest(f_classif, k=10)
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-84-ch84.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=400, n_features=120, n_informative=10,
                           n_redundant=20, class_sep=0.7, flip_y=0.06,
                           random_state=42)
Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.10, random_state=42,
                                  stratify=y)

scaler_full = StandardScaler().fit(X)                  # الصفوف الـ400 كلها
selector_full = SelectKBest(f_classif, k=10).fit(X, y) # والهدف y كاملاً
keep = selector_full.get_support(indices=True)
leaky = LogisticRegression(max_iter=5000, C=0.3).fit(
    scaler_full.transform(Xa).to_numpy()[:, keep], ya)
print(f"المتسرّبة تعلّمت من {X.shape[0]} صفاً | درجتها: "
      f"{leaky.score(scaler_full.transform(Xb).to_numpy()[:, keep], yb):.4f}")

pipe = Pipeline([("sc", StandardScaler()),
                 ("sel", SelectKBest(f_classif, k=10)),
                 ("clf", LogisticRegression(max_iter=5000, C=0.3))]).fit(Xa, ya)
honest_idx = np.where(pipe.named_steps["sel"].get_support())[0]
print(f"الأنبوب تدرّب على {Xa.shape[0]} صفاً | درجته: {pipe.score(Xb, yb):.4f}")
print("اختار الأنبوب  :", [f"x{i}" for i in honest_idx])
print("اختيار المتسرّب:", [f"x{i}" for i in np.sort(keep)])
print("اختلفا في     :",
      [f"x{i}" for i in sorted(set(honest_idx) ^ set(keep))])

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-84-ch84.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def make_honest():
    return Pipeline([("sc", StandardScaler()),
                     ("sel", SelectKBest(f_classif, k=10)),
                     ("clf", LogisticRegression(max_iter=5000, C=0.3))])

honest_scores, leaky_scores = [], []
for seed in range(30):
    A, B, c, d = train_test_split(X, y, test_size=0.10, random_state=seed,
                                  stratify=y)
    honest_scores.append(make_honest().fit(A, c).score(B, d))
    sc_all = StandardScaler().fit(X)                     # تسرّب
    m = SelectKBest(f_classif, k=10).fit(X, y).get_support()  # تسرّب
    mdl = LogisticRegression(max_iter=5000, C=0.3).fit(
        sc_all.transform(A).to_numpy()[:, m], c)
    leaky_scores.append(mdl.score(sc_all.transform(B).to_numpy()[:, m], d))

h, l = np.array(honest_scores), np.array(leaky_scores)
gain = l - h
print(f"صادق  : المتوسط {h.mean():.4f} | الانحراف {h.std():.4f} "
      f"| المدى [{h.min():.4f}, {h.max():.4f}]")
print(f"متسرّب: المتوسط {l.mean():.4f} | الانحراف {l.std():.4f} "
      f"| المدى [{l.min():.4f}, {l.max():.4f}]")
print(f"متوسط الفارق: {gain.mean():+.4f} | فوق الصفر {int((gain > 0).sum())}"
      f" | تحت الصفر {int((gain < 0).sum())} | متساوٍ {int((gain == 0).sum())} من 30")
print(f"اتّساع المدى: من {(h.max() - h.min()) * 100:.1f} نقطة "
      f"إلى {(l.max() - l.min()) * 100:.1f} نقطة")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-84-ch84.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
sc = pipe.named_steps["sc"]
print(f"صفوف رآها المُطبِّع: {sc.n_samples_seen_:.0f} من أصل {X.shape[0]}")
print(f"متوسّطه للعمود الأول: {sc.mean_[0]:.4f}")
print(f"متوسّطه على كامل البيانات: {scaler_full.mean_[0]:.4f}")
print("أعمدة أبقاها المنتقي:",
      [f"x{i}" for i in np.where(pipe.named_steps['sel'].get_support())[0]])

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-84-ch84.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
X_b_t = pipe[:-1].transform(Xb)
manual = pipe.named_steps["clf"].predict(X_b_t)
print("تطابق المسارين:", bool((manual == pipe.predict(Xb)).all()))
print("نوع ناتج التحويل:", type(X_b_t).__name__, "| الشكل:", X_b_t.shape)
print("أسماء الخصائص بعد التحويل:", list(X_b_t.columns)[:4])

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-84-ch84.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.pipeline import make_pipeline

auto = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
print("أسماء make_pipeline:", [name for name, _ in auto.steps])

try:
    auto.set_params(clf__C=1.0)          # الاسم الذي ظننّاه منطقياً
except ValueError as exc:
    print("ValueError:", str(exc).splitlines()[0][:86])

auto.set_params(logisticregression__C=1.0)   # الاسم الحقيقي
print("بعد التصحيح:", auto.get_params()["logisticregression__C"])

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-84-ch84.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    Pipeline([("sc", StandardScaler()),
              ("clf", LogisticRegression(max_iter=5000))]),
    {"sc__with_mean": [True, False],
     "clf__C": [0.05, 0.3, 2.0]},
    cv=3, n_jobs=-1, return_train_score=True)
grid.fit(Xa, ya)

print("أفضل تركيبة:", grid.best_params_)
print(f"درجة التحقق: {grid.best_score_:.4f}")
print(f"درجة التدريب للفائز: "
      f"{grid.cv_results_['mean_train_score'][grid.best_index_]:.4f}")
print("مفاتيح الشبكة:", [k for k in grid.cv_results_ if k.startswith("param_")])

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-84-ch84.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

docs = [
    "free money click now win prize today",
    "limited offer buy cheap discount now",
    "urgent call now about your invoice payment",
    "the deployment failed on the production server",
    "meeting tomorrow at nine about project status",
    "team meeting notes attached for review",
    "please review the report before the deadline",
    "lunch at the cafe near the office tomorrow",
]
labels = np.array([1, 1, 1, 0, 0, 0, 0, 0])

text_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
    ("clf", CalibratedClassifierCV(LinearSVC(C=1.0), method="sigmoid", cv=2)),
])
text_pipeline.fit(docs, labels)

print("حجم مفردات TF-IDF:", len(text_pipeline.named_steps["tfidf"].vocabulary_))
proba = text_pipeline.predict_proba(["claim your free prize now"])
print("الاحتمال:", np.round(proba[0], 3),
      "| الفئة:", text_pipeline.classes_[proba.argmax()])
print("نوع الخطوة الأخيرة:", type(text_pipeline.named_steps["clf"]).__name__)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-84-ch84.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
from sklearn.impute import SimpleImputer

frame = pd.DataFrame(Xa[:, :5].copy(),
                     columns=["age", "income", "tenure", "balance", "score"])
frame.loc[frame.sample(frac=0.3, random_state=42).index, "score"] = np.nan

frame_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                       ("clf", LogisticRegression(max_iter=5000))])
frame_pipe.fit(frame, ya)

print("feature_names_in_:", list(frame_pipe.feature_names_in_))
print("n_features_in_:", frame_pipe.n_features_in_)
print("الوسيط المستعاد للعمود المفقود:",
      round(frame_pipe.named_steps["imp"].statistics_[4], 4))
print("أسماء المخرجات:", list(frame_pipe[:-1].get_feature_names_out()))

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-84-ch84.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import io
import joblib

buffer = io.BytesIO()
joblib.dump(pipe, buffer)
buffer.seek(0)
restored = joblib.load(buffer)

print(f"حجم الملف: {len(buffer.getvalue()) / 1024:.1f} كيلوبايت")
print("تطابق التنبؤ بعد الاستعادة:",
      bool((restored.predict(Xb) == pipe.predict(Xb)).all()))
print("نفس أسماء الخطوات:", [n for n, _ in restored.steps])

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-84-ch84.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.set_params(clf__C=10.0)

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-84-ch84.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

Xf, yf = make_classification(n_samples=500, n_features=30, n_informative=6,
                             random_state=42)
Xa2, Xb2, ya2, yb2 = train_test_split(Xf, yf, test_size=0.2, random_state=42,
                                       stratify=yf)

miss = np.random.default_rng(0).choice(Xa2.shape[0], size=60, replace=False)
Xa2[miss, 3] = np.nan
Xa2[miss, 17] = np.nan

pipe3 = Pipeline([("imputer", SimpleImputer(strategy="median")),
                  ("scaler", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=5000))])
pipe3.fit(Xa2, ya2)

print(f"صفوف رآها المُطبِّع: {pipe3.named_steps['scaler'].n_samples_seen_:.0f}")
print(f"أعمدة بعد الاسترجاع: {pipe3.named_steps['imputer'].statistics_.size}")
print(f"مجموع وسائط الاسترجاع: {pipe3.named_steps['imputer'].statistics_.sum():.3f}")
print(f"درجة الاختبار: {pipe3.score(Xb2, yb2):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-84-ch84.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV

pipe4 = Pipeline([("imputer", SimpleImputer(strategy="median")),
                  ("scaler", StandardScaler()),
                  ("sel", SelectKBest(f_classif, k=5)),
                  ("clf", LogisticRegression(max_iter=5000))])

grid2 = GridSearchCV(pipe4,
                     {"imputer__strategy": ["median", "mean"],
                      "clf__C": [0.5, 5.0]},
                     cv=4, return_train_score=True)
grid2.fit(Xa2, ya2)

i = grid2.best_index_
tr = grid2.cv_results_["mean_train_score"][i]
te = grid2.cv_results_["mean_test_score"][i]
print("أفضل تركيبة:", grid2.best_params_)
print(f"تدريب: {tr:.4f} | تحقق: {te:.4f} | الفجوة: {tr - te:+.4f}")

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-84-ch84.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
auto_pipe = make_pipeline(StandardScaler(), LogisticRegression())
print("الأسماء الحقيقية:", [n for n, _ in auto_pipe.steps])
try:
    auto_pipe.set_params(clf__C=10.0)
except ValueError as exc:
    print("ValueError:", str(exc).splitlines()[0][:86])
auto_pipe.set_params(logisticregression__C=10.0)
print("بعد التصحيح:", auto_pipe.get_params()["logisticregression__C"])
