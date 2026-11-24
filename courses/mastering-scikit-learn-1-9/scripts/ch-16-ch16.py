#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — ترميز المتغيرات الفئوية
Mastering scikit-learn 1.9 — Chapter 16 — Data

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-16-ch16 (lesson 16)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-16-ch16.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — ترميز المتغيرات الفئوية
Mastering scikit-learn 1.9 — Chapter 16 — Data

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-16-ch16 (lesson 16)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part02/ch16.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.preprocessing import OneHotEncoder

cats = np.array([["Riyadh", "pro"], ["Cairo", "free"], ["Riyadh", "free"],
                 ["Doha", "pro"], ["Cairo", "pro"], ["Amman", "free"]], dtype=object)

ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(cats)
print("categories_ (مرتّبة أبجدياً):", [c.tolist() for c in ohe.categories_])
print("get_feature_names_out      :", list(ohe.get_feature_names_out(["city", "plan"])))
print("الشكل الناتج               :", ohe.transform(cats).shape)

print("\nخيارات حذف عمود (تفادي التكرار الخطي):")
for d in [None, "first", "if_binary"]:
    kw = {} if d is None else {"drop": d}
    print(f"  drop={str(d):14s} -> {OneHotEncoder(sparse_output=False, **kw).fit_transform(cats).shape[1]} عمود")

newcats = np.array([["Riyadh", "pro"], ["Muscat", "enterprise"]], dtype=object)
print("\nf값 جديد غير مرئي في التدريب:", ohe.transform(newcats).to_numpy().astype(int).tolist())
try:
    OneHotEncoder(sparse_output=False).fit(cats).transform(newcats)
except Exception as exc:
    print("مع handle_unknown='error'  ->", type(exc).__name__, ":", str(exc).splitlines()[0][:90])

# ────────────────────────────────────────────────────────────────────────
# [002] content/part02/ch16.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(cats)
print("categories_            :", [c.tolist() for c in oe.categories_])
print("الترميز                :", oe.transform(cats).to_numpy().ravel().astype(int).tolist())
print("قيمة جديدة            :", oe.transform(newcats).to_numpy().ravel().astype(int).tolist())

with_na = np.array([["b"], ["a"], [np.nan], ["z"]], dtype=object)
oen = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                     encoded_missing_value=-2).fit(with_na)
print("\nمع NaN والفئات        :", [c.tolist() for c in oen.categories_])
print("الترميز                :", oen.transform(with_na).to_numpy().ravel().astype(int).tolist())
print("الأعمدة المنقولة       :")
print(ohe.get_feature_names_out(["city", "plan"]))

# ────────────────────────────────────────────────────────────────────────
# [003] content/part02/ch16.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import TargetEncoder
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(11)
n, k = 1500, 300
latent = rng.normal(0, 1, k)
grp = rng.integers(0, k, n)
prob = 1 / (1 + np.exp(-latent[grp]))
y = (rng.random(n) < prob).astype(int)
grp_name = np.array([f"g{g:03d}" for g in grp], dtype=object)
Xnum = rng.normal(0, 1, (n, 2))
data = np.empty((n, 3), dtype=object)
data[:, 0] = grp_name
data[:, 1:] = Xnum
print(f"صفوف={n}  فئات={k}  متوسّط حجم الفئة={n / k:.1f}")

tr, te = train_test_split(np.arange(n), test_size=0.3, random_state=0, stratify=y)
Xtr, Xte, ytr, yte = data[tr], data[te], y[tr], y[te]

te_fit = TargetEncoder(target_type="binary", cv=5).fit(Xtr, ytr)
print("\n encodings_ لأول خمس فئات:", np.round(te_fit.encodings_[0][:5], 4).tolist())
print(" خاصية واحدة بدل ٣٠٠ عمود:", te_fit.transform(Xtr).shape[1] - 2, "عموداً للعمود الفئوي")
print(" fit يحتاج y بالضرورة    : TargetEncoder().fit_transform(X) بلا y لا يعمل")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part02/ch16.md
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# الطريقتان: حساب متوسّط التسمية على الكل، أو على التدريب فقط
leaky_map = pd.Series(y).groupby(pd.Series(grp_name)).mean()
Ztr_l = np.column_stack([Xnum[tr], pd.Series(grp_name[tr]).map(leaky_map).to_numpy()])
Zte_l = np.column_stack([Xnum[te], pd.Series(grp_name[te]).map(leaky_map).to_numpy()])
sc = StandardScaler().fit(Ztr_l)
ml = LogisticRegression(max_iter=3000).fit(sc.transform(Ztr_l), ytr)
print("متوسّط على الكل   : train=%.4f  test=%.4f" % (ml.score(sc.transform(Ztr_l), ytr),
                                                    ml.score(sc.transform(Zte_l), yte)))

train_map = pd.Series(ytr).groupby(pd.Series(grp_name[tr])).mean()
global_mean = ytr.mean()
Ztr_c = np.column_stack([Xnum[tr], pd.Series(grp_name[tr]).map(train_map).to_numpy()])
Zte_c = np.column_stack([Xnum[te], pd.Series(grp_name[te]).map(train_map).fillna(global_mean).to_numpy()])
sc2 = StandardScaler().fit(Ztr_c)
mc = LogisticRegression(max_iter=3000).fit(sc2.transform(Ztr_c), ytr)
print("متوسّط على التدريب: train=%.4f  test=%.4f" % (mc.score(sc2.transform(Ztr_c), ytr),
                                                    mc.score(sc2.transform(Zte_c), yte)))
gap = ml.score(sc.transform(Zte_l), yte) - mc.score(sc2.transform(Zte_c), yte)
print(f"\nالمكافأة الظاهرة من التسريب: {gap:+.4f} على بيانات الاختبار")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part02/ch16.md
# ────────────────────────────────────────────────────────────────────────
rare = np.array([[v] for v in ["a"] * 60 + ["b"] * 20 + ["c"] * 12 + ["d"] * 8], dtype=object)
print("التوزيع قبل الطيّ:", pd.Series(rare.ravel()).value_counts().to_dict())

for kw in [dict(), dict(min_frequency=15), dict(max_categories=2)]:
    e = OneHotEncoder(sparse_output=False, handle_unknown="infrequent_if_exist", **kw).fit(rare)
    print(f"  onehot {str(kw)[:26]:28s} -> {e.transform(rare).shape[1]} عمود  {list(e.get_feature_names_out())}")

er = OrdinalEncoder(min_frequency=15).fit(rare)
codes = sorted(set(er.transform(rare).to_numpy().ravel().tolist()))
print("  ordinal min_frequency=15  -> أكواد:", codes, "categories_:", er.categories_[0].tolist())

# ────────────────────────────────────────────────────────────────────────
# [006] content/part02/ch16.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder

labels = np.array(["ham", "spam", "ham", "spam", "ham"])
le = LabelEncoder().fit(labels)
print("classes_          :", le.classes_.tolist())
print("fit_transform     :", le.fit_transform(labels).tolist())
print("inverse_transform :", le.inverse_transform([0, 1, 0]).tolist())

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-16-ch16.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.preprocessing import OneHotEncoder

cats = np.array([["Riyadh", "pro"], ["Cairo", "free"], ["Riyadh", "free"],
                 ["Doha", "pro"], ["Cairo", "pro"], ["Amman", "free"]], dtype=object)

ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(cats)
print("categories_ (مرتّبة أبجدياً):", [c.tolist() for c in ohe.categories_])
print("get_feature_names_out      :", list(ohe.get_feature_names_out(["city", "plan"])))
print("الشكل الناتج               :", ohe.transform(cats).shape)

print("\nخيارات حذف عمود (تفادي التكرار الخطي):")
for d in [None, "first", "if_binary"]:
    kw = {} if d is None else {"drop": d}
    print(f"  drop={str(d):14s} -> {OneHotEncoder(sparse_output=False, **kw).fit_transform(cats).shape[1]} عمود")

newcats = np.array([["Riyadh", "pro"], ["Muscat", "enterprise"]], dtype=object)
print("\nf값 جديد غير مرئي في التدريب:", ohe.transform(newcats).to_numpy().astype(int).tolist())
try:
    OneHotEncoder(sparse_output=False).fit(cats).transform(newcats)
except Exception as exc:
    print("مع handle_unknown='error'  ->", type(exc).__name__, ":", str(exc).splitlines()[0][:90])

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-16-ch16.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(cats)
print("categories_            :", [c.tolist() for c in oe.categories_])
print("الترميز                :", oe.transform(cats).to_numpy().ravel().astype(int).tolist())
print("قيمة جديدة            :", oe.transform(newcats).to_numpy().ravel().astype(int).tolist())

with_na = np.array([["b"], ["a"], [np.nan], ["z"]], dtype=object)
oen = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                     encoded_missing_value=-2).fit(with_na)
print("\nمع NaN والفئات        :", [c.tolist() for c in oen.categories_])
print("الترميز                :", oen.transform(with_na).to_numpy().ravel().astype(int).tolist())
print("الأعمدة المنقولة       :")
print(ohe.get_feature_names_out(["city", "plan"]))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-16-ch16.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import TargetEncoder
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(11)
n, k = 1500, 300
latent = rng.normal(0, 1, k)
grp = rng.integers(0, k, n)
prob = 1 / (1 + np.exp(-latent[grp]))
y = (rng.random(n) < prob).astype(int)
grp_name = np.array([f"g{g:03d}" for g in grp], dtype=object)
Xnum = rng.normal(0, 1, (n, 2))
data = np.empty((n, 3), dtype=object)
data[:, 0] = grp_name
data[:, 1:] = Xnum
print(f"صفوف={n}  فئات={k}  متوسّط حجم الفئة={n / k:.1f}")

tr, te = train_test_split(np.arange(n), test_size=0.3, random_state=0, stratify=y)
Xtr, Xte, ytr, yte = data[tr], data[te], y[tr], y[te]

te_fit = TargetEncoder(target_type="binary", cv=5).fit(Xtr, ytr)
print("\n encodings_ لأول خمس فئات:", np.round(te_fit.encodings_[0][:5], 4).tolist())
print(" خاصية واحدة بدل ٣٠٠ عمود:", te_fit.transform(Xtr).shape[1] - 2, "عموداً للعمود الفئوي")
print(" fit يحتاج y بالضرورة    : TargetEncoder().fit_transform(X) بلا y لا يعمل")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-16-ch16.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# الطريقتان: حساب متوسّط التسمية على الكل، أو على التدريب فقط
leaky_map = pd.Series(y).groupby(pd.Series(grp_name)).mean()
Ztr_l = np.column_stack([Xnum[tr], pd.Series(grp_name[tr]).map(leaky_map).to_numpy()])
Zte_l = np.column_stack([Xnum[te], pd.Series(grp_name[te]).map(leaky_map).to_numpy()])
sc = StandardScaler().fit(Ztr_l)
ml = LogisticRegression(max_iter=3000).fit(sc.transform(Ztr_l), ytr)
print("متوسّط على الكل   : train=%.4f  test=%.4f" % (ml.score(sc.transform(Ztr_l), ytr),
                                                    ml.score(sc.transform(Zte_l), yte)))

train_map = pd.Series(ytr).groupby(pd.Series(grp_name[tr])).mean()
global_mean = ytr.mean()
Ztr_c = np.column_stack([Xnum[tr], pd.Series(grp_name[tr]).map(train_map).to_numpy()])
Zte_c = np.column_stack([Xnum[te], pd.Series(grp_name[te]).map(train_map).fillna(global_mean).to_numpy()])
sc2 = StandardScaler().fit(Ztr_c)
mc = LogisticRegression(max_iter=3000).fit(sc2.transform(Ztr_c), ytr)
print("متوسّط على التدريب: train=%.4f  test=%.4f" % (mc.score(sc2.transform(Ztr_c), ytr),
                                                    mc.score(sc2.transform(Zte_c), yte)))
gap = ml.score(sc.transform(Zte_l), yte) - mc.score(sc2.transform(Zte_c), yte)
print(f"\nالمكافأة الظاهرة من التسريب: {gap:+.4f} على بيانات الاختبار")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-16-ch16.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
rare = np.array([[v] for v in ["a"] * 60 + ["b"] * 20 + ["c"] * 12 + ["d"] * 8], dtype=object)
print("التوزيع قبل الطيّ:", pd.Series(rare.ravel()).value_counts().to_dict())

for kw in [dict(), dict(min_frequency=15), dict(max_categories=2)]:
    e = OneHotEncoder(sparse_output=False, handle_unknown="infrequent_if_exist", **kw).fit(rare)
    print(f"  onehot {str(kw)[:26]:28s} -> {e.transform(rare).shape[1]} عمود  {list(e.get_feature_names_out())}")

er = OrdinalEncoder(min_frequency=15).fit(rare)
codes = sorted(set(er.transform(rare).to_numpy().ravel().tolist()))
print("  ordinal min_frequency=15  -> أكواد:", codes, "categories_:", er.categories_[0].tolist())

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-16-ch16.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder

labels = np.array(["ham", "spam", "ham", "spam", "ham"])
le = LabelEncoder().fit(labels)
print("classes_          :", le.classes_.tolist())
print("fit_transform     :", le.fit_transform(labels).tolist())
print("inverse_transform :", le.inverse_transform([0, 1, 0]).tolist())
