#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — تقليل الأبعاد الخطي: PCA
Mastering scikit-learn 1.9 — PCA

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-68-ch68 (lesson 68)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-68-ch68.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — تقليل الأبعاد الخطي: PCA
Mastering scikit-learn 1.9 — PCA

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-68-ch68 (lesson 68)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part08/ch68.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import numpy as np

X, y = load_iris(return_X_y=True)
Xc = X - X.mean(axis=0)                     # التوسيط خطوة إلزامية
C = (Xc.T @ Xc) / (len(X) - 1)              # مصفوفة التغاير
eigvals, eigvecs = np.linalg.eigh(C)
order = np.argsort(eigvals)[::-1]

print(f"القيم الذاتية يدوياً (تنازلياً): {np.round(eigvals[order], 3)}")
print(f"أول متجه ذاتي يدوياً:            {np.round(eigvecs[:, order[0]], 3)}")

pca = PCA(n_components=2).fit(X)
print(f"قيم explained_variance_:          {np.round(pca.explained_variance_, 3)}")
print(f"مكوّنات components_:               {np.round(pca.components_, 3)}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part08/ch68.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris, load_digits
from sklearn.decomposition import PCA
import numpy as np

# (أ) Iris: النسب والتراكمي وخطأ إعادة البناء
X, y = load_iris(return_X_y=True)
pca_full = PCA().fit(X)
print(f"نسب التباين المفسَّر: {np.round(pca_full.explained_variance_ratio_, 4)}")
print(f"التراكمي:             {np.round(np.cumsum(pca_full.explained_variance_ratio_), 4)}")
for k in [1, 2, 3]:
    p = PCA(n_components=k).fit(X)
    X_back = p.inverse_transform(p.transform(X))
    print(f"k={k} | خطأ إعادة البناء (MSE): {float(np.mean((X - X_back) ** 2)):.4f}")

# (ب) Digits: القرار التلقائي بعتبة 95%
X_dig, _ = load_digits(return_X_y=True)
pca095 = PCA(n_components=0.95, svd_solver="full").fit(X_dig)
print(f"Digits: {X_dig.shape[1]} بُعداً -> {pca095.n_components_} مكوّناً "
      f"(تباين {pca095.explained_variance_ratio_.sum():.4f} | ضغط {1 - pca095.n_components_ / X_dig.shape[1]:.1%})")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part08/ch68.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

X, y = load_iris(return_X_y=True)
X_dirty = X.copy()
X_dirty[:, 0] *= 1000                       # عمود منطقي بمقياس منفلت

pca_raw = PCA(n_components=2).fit(X_dirty)
pca_scaled = PCA(n_components=2).fit(StandardScaler().fit_transform(X_dirty))
print(f"بدون تطبيع: نصيب المكون الأول: {pca_raw.explained_variance_ratio_[0]:.4f}")
print(f"مع تطبيع:   نصيب المكون الأول: {pca_scaled.explained_variance_ratio_[0]:.4f}")
print(f"أوزان المكون الأول بدون تطبيع: {np.round(pca_raw.components_[0], 4)}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part08/ch68.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

X, y = load_iris(return_X_y=True)
feat = load_iris().feature_names
Xs = StandardScaler().fit_transform(X)
pca = PCA(n_components=2).fit(Xs)

loadings = pd.DataFrame(pca.components_.T, index=feat, columns=["PC1", "PC2"]).round(3)
print(f"نسب التباين: PC1 = {pca.explained_variance_ratio_[0]:.3f}، PC2 = {pca.explained_variance_ratio_[1]:.3f}")
loadings

# ────────────────────────────────────────────────────────────────────────
# [005] content/part08/ch68.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

X, y = load_iris(return_X_y=True)

for k in [3, 7]:
    a = cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)), X, y, cv=10).mean()
    b = cross_val_score(make_pipeline(StandardScaler(), PCA(n_components=2), KNeighborsClassifier(n_neighbors=k)), X, y, cv=10).mean()
    print(f"iris نظيفة و k={k}: KNN مباشر {a:.4f} | بعد PCA(2) {b:.4f}")

rng = np.random.default_rng(42)
X_noisy = np.hstack([X, rng.normal(size=(len(X), 5))])   # خمسة أعمدة ضجيج
for k in [7]:
    a = cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)), X_noisy, y, cv=10).mean()
    b = cross_val_score(make_pipeline(StandardScaler(), PCA(n_components=2), KNeighborsClassifier(n_neighbors=k)), X_noisy, y, cv=10).mean()
    print(f"iris + 5 أعمدة ضجيج: KNN مباشر {a:.4f} | بعد PCA(2) {b:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part08/ch68.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA, IncrementalPCA, TruncatedSVD
from scipy import sparse
import numpy as np

X_dig, _ = load_digits(return_X_y=True)

# 1) TruncatedSVD: من متفرقة إلى مضغوطة بلا توسيط
X_sp = sparse.csr_matrix(X_dig)
svd = TruncatedSVD(n_components=20, random_state=42).fit(X_sp)
print(f"TruncatedSVD: {X_sp.shape} -> 1797×20 | تباين مفسَّر: {svd.explained_variance_ratio_.sum():.4f}")

# 2) IncrementalPCA: تدريب بدفاتر يوازي PCA الكامل
ipca = IncrementalPCA(n_components=20, batch_size=450)
for chunk in np.array_split(X_dig, 4):
    ipca.partial_fit(chunk)
pca = PCA(n_components=20, svd_solver="full").fit(X_dig)
rel = np.abs(ipca.explained_variance_ - pca.explained_variance_) / pca.explained_variance_
print(f"IncrementalPCA: متوسط فرق نسبي في القيم الذاتية {rel.mean():.2%} | تباين تراكمي {ipca.explained_variance_ratio_.sum():.4f}")

# 3) المحلّل العشوائي: نفس النتيجة بزمن أقل على البيانات الكبيرة
pca_r = PCA(n_components=20, svd_solver="randomized", random_state=42).fit(X_dig)
print(f"randomized: تباين مفسَّر {pca_r.explained_variance_ratio_.sum():.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] content/part08/ch68.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

data = fetch_california_housing()
X = data.data[:4000]
Z = StandardScaler().fit_transform(X)

pca = PCA(n_components=0.95, svd_solver="full").fit(Z)
print(f"الأبعاد الأصلية: {Z.shape[1]}")
print(f"مكونات تحقق 95%: {pca.n_components_} | نسبة الضغط: {1 - pca.n_components_ / Z.shape[1]:.1%}")

Z_back = pca.inverse_transform(pca.transform(Z))
rmse = float(np.sqrt(np.mean((Z - Z_back) ** 2)))
print(f"خطأ إعادة البناء (RMSE بوحدة الانحراف المعياري): {rmse:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] content/part08/ch68.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

X, y = make_moons(n_samples=600, noise=0.12, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

lin = make_pipeline(PCA(n_components=2), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
ker = make_pipeline(KernelPCA(n_components=2, kernel="rbf", gamma=15),
                    LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
print(f"LogisticRegression بعد PCA الخطي:   {lin.score(X_te, y_te):.4f}")
print(f"LogisticRegression بعد KernelPCA:   {ker.score(X_te, y_te):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-68-ch68.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import numpy as np

X, y = load_iris(return_X_y=True)
Xc = X - X.mean(axis=0)                     # التوسيط خطوة إلزامية
C = (Xc.T @ Xc) / (len(X) - 1)              # مصفوفة التغاير
eigvals, eigvecs = np.linalg.eigh(C)
order = np.argsort(eigvals)[::-1]

print(f"القيم الذاتية يدوياً (تنازلياً): {np.round(eigvals[order], 3)}")
print(f"أول متجه ذاتي يدوياً:            {np.round(eigvecs[:, order[0]], 3)}")

pca = PCA(n_components=2).fit(X)
print(f"قيم explained_variance_:          {np.round(pca.explained_variance_, 3)}")
print(f"مكوّنات components_:               {np.round(pca.components_, 3)}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-68-ch68.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris, load_digits
from sklearn.decomposition import PCA
import numpy as np

# (أ) Iris: النسب والتراكمي وخطأ إعادة البناء
X, y = load_iris(return_X_y=True)
pca_full = PCA().fit(X)
print(f"نسب التباين المفسَّر: {np.round(pca_full.explained_variance_ratio_, 4)}")
print(f"التراكمي:             {np.round(np.cumsum(pca_full.explained_variance_ratio_), 4)}")
for k in [1, 2, 3]:
    p = PCA(n_components=k).fit(X)
    X_back = p.inverse_transform(p.transform(X))
    print(f"k={k} | خطأ إعادة البناء (MSE): {float(np.mean((X - X_back) ** 2)):.4f}")

# (ب) Digits: القرار التلقائي بعتبة 95%
X_dig, _ = load_digits(return_X_y=True)
pca095 = PCA(n_components=0.95, svd_solver="full").fit(X_dig)
print(f"Digits: {X_dig.shape[1]} بُعداً -> {pca095.n_components_} مكوّناً "
      f"(تباين {pca095.explained_variance_ratio_.sum():.4f} | ضغط {1 - pca095.n_components_ / X_dig.shape[1]:.1%})")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-68-ch68.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

X, y = load_iris(return_X_y=True)
X_dirty = X.copy()
X_dirty[:, 0] *= 1000                       # عمود منطقي بمقياس منفلت

pca_raw = PCA(n_components=2).fit(X_dirty)
pca_scaled = PCA(n_components=2).fit(StandardScaler().fit_transform(X_dirty))
print(f"بدون تطبيع: نصيب المكون الأول: {pca_raw.explained_variance_ratio_[0]:.4f}")
print(f"مع تطبيع:   نصيب المكون الأول: {pca_scaled.explained_variance_ratio_[0]:.4f}")
print(f"أوزان المكون الأول بدون تطبيع: {np.round(pca_raw.components_[0], 4)}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-68-ch68.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

X, y = load_iris(return_X_y=True)
feat = load_iris().feature_names
Xs = StandardScaler().fit_transform(X)
pca = PCA(n_components=2).fit(Xs)

loadings = pd.DataFrame(pca.components_.T, index=feat, columns=["PC1", "PC2"]).round(3)
print(f"نسب التباين: PC1 = {pca.explained_variance_ratio_[0]:.3f}، PC2 = {pca.explained_variance_ratio_[1]:.3f}")
loadings

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-68-ch68.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

X, y = load_iris(return_X_y=True)

for k in [3, 7]:
    a = cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)), X, y, cv=10).mean()
    b = cross_val_score(make_pipeline(StandardScaler(), PCA(n_components=2), KNeighborsClassifier(n_neighbors=k)), X, y, cv=10).mean()
    print(f"iris نظيفة و k={k}: KNN مباشر {a:.4f} | بعد PCA(2) {b:.4f}")

rng = np.random.default_rng(42)
X_noisy = np.hstack([X, rng.normal(size=(len(X), 5))])   # خمسة أعمدة ضجيج
for k in [7]:
    a = cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)), X_noisy, y, cv=10).mean()
    b = cross_val_score(make_pipeline(StandardScaler(), PCA(n_components=2), KNeighborsClassifier(n_neighbors=k)), X_noisy, y, cv=10).mean()
    print(f"iris + 5 أعمدة ضجيج: KNN مباشر {a:.4f} | بعد PCA(2) {b:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-68-ch68.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA, IncrementalPCA, TruncatedSVD
from scipy import sparse
import numpy as np

X_dig, _ = load_digits(return_X_y=True)

# 1) TruncatedSVD: من متفرقة إلى مضغوطة بلا توسيط
X_sp = sparse.csr_matrix(X_dig)
svd = TruncatedSVD(n_components=20, random_state=42).fit(X_sp)
print(f"TruncatedSVD: {X_sp.shape} -> 1797×20 | تباين مفسَّر: {svd.explained_variance_ratio_.sum():.4f}")

# 2) IncrementalPCA: تدريب بدفاتر يوازي PCA الكامل
ipca = IncrementalPCA(n_components=20, batch_size=450)
for chunk in np.array_split(X_dig, 4):
    ipca.partial_fit(chunk)
pca = PCA(n_components=20, svd_solver="full").fit(X_dig)
rel = np.abs(ipca.explained_variance_ - pca.explained_variance_) / pca.explained_variance_
print(f"IncrementalPCA: متوسط فرق نسبي في القيم الذاتية {rel.mean():.2%} | تباين تراكمي {ipca.explained_variance_ratio_.sum():.4f}")

# 3) المحلّل العشوائي: نفس النتيجة بزمن أقل على البيانات الكبيرة
pca_r = PCA(n_components=20, svd_solver="randomized", random_state=42).fit(X_dig)
print(f"randomized: تباين مفسَّر {pca_r.explained_variance_ratio_.sum():.4f}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-68-ch68.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

data = fetch_california_housing()
X = data.data[:4000]
Z = StandardScaler().fit_transform(X)

pca = PCA(n_components=0.95, svd_solver="full").fit(Z)
print(f"الأبعاد الأصلية: {Z.shape[1]}")
print(f"مكونات تحقق 95%: {pca.n_components_} | نسبة الضغط: {1 - pca.n_components_ / Z.shape[1]:.1%}")

Z_back = pca.inverse_transform(pca.transform(Z))
rmse = float(np.sqrt(np.mean((Z - Z_back) ** 2)))
print(f"خطأ إعادة البناء (RMSE بوحدة الانحراف المعياري): {rmse:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-68-ch68.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

X, y = make_moons(n_samples=600, noise=0.12, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

lin = make_pipeline(PCA(n_components=2), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
ker = make_pipeline(KernelPCA(n_components=2, kernel="rbf", gamma=15),
                    LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
print(f"LogisticRegression بعد PCA الخطي:   {lin.score(X_te, y_te):.4f}")
print(f"LogisticRegression بعد KernelPCA:   {ker.score(X_te, y_te):.4f}")
