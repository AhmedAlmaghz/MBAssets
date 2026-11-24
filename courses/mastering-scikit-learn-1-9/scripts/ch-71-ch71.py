#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — النصوص: التحليل الدلالي الخفي والمواضيع
Mastering scikit-learn 1.9 — Chapter 71 — Unsupervised

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-71-ch71 (lesson 71)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-71-ch71.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — النصوص: التحليل الدلالي الخفي والمواضيع
Mastering scikit-learn 1.9 — Chapter 71 — Unsupervised

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-71-ch71 (lesson 71)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part08/ch71.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "model learns patterns from training data",
    "training data needs careful validation",
    "market prices change with financial risk",
    "financial models estimate market risk",
]
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(documents)
terms = vectorizer.get_feature_names_out()

print("شكل المصفوفة:", X_tfidf.shape)
print("عدد العناصر غير الصفرية:", X_tfidf.nnz)
for i, row in enumerate(X_tfidf):
    top = np.argsort(row.toarray().ravel())[-3:][::-1]
    print(i, [terms[j] for j in top])

# ────────────────────────────────────────────────────────────────────────
# [002] content/part08/ch71.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "neural network learns patterns from image data",
    "image model detects objects in a picture",
    "deep learning trains a neural network",
    "bank market prices and financial risk",
    "financial models estimate market prices",
    "bank lending depends on credit risk",
    "patient health records support medical care",
    "medical treatment improves patient health",
    "health care monitors patient recovery",
]
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(documents)
svd = TruncatedSVD(n_components=3, random_state=42)
Z = svd.fit_transform(X)
terms = vectorizer.get_feature_names_out()

print("شكل الوثائق في الفضاء الكامن:", Z.shape)
print("التباين المفسر لكل مكوّن:",
      np.round(svd.explained_variance_ratio_, 3))
for topic, weights in enumerate(svd.components_):
    top = np.argsort(weights)[-5:][::-1]
    print(f"المحور {topic + 1}:", [terms[i] for i in top])

# ────────────────────────────────────────────────────────────────────────
# [003] content/part08/ch71.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "neural network learns image patterns from training data",
    "deep learning model recognizes images and visual objects",
    "machine learning trains a model to classify image data",
    "computer vision detects objects in digital images",
    "image recognition uses neural networks and learned features",
    "bank market prices reveal financial risk",
    "stock market prices change with investment risk",
    "financial models estimate credit risk and loan defaults",
    "bank lending depends on credit scores and market rates",
    "investment portfolio tracks financial markets and returns",
    "patient health records support medical care",
    "medical treatment helps patient recovery",
    "health care monitors clinical outcomes and diagnosis",
    "doctors evaluate patient symptoms and medical history",
    "clinical care improves patient health and recovery",
]
vectorizer = CountVectorizer(stop_words="english")
X_counts = vectorizer.fit_transform(documents)
lda = LatentDirichletAllocation(n_components=3, max_iter=40,
                                learning_method="batch", random_state=42)
document_topics = np.asarray(lda.fit_transform(X_counts))
terms = vectorizer.get_feature_names_out()

for topic_id, weights in enumerate(lda.components_):
    top = np.argsort(weights)[-5:][::-1]
    print(f"الموضوع {topic_id + 1}:",
          [(terms[i], round(float(weights[i]), 2)) for i in top])
print("توزيع أول وثيقة:", np.round(document_topics[0], 3))
print("مجموع الاحتمالات:", round(float(document_topics[0].sum()), 3))

# ────────────────────────────────────────────────────────────────────────
# [004] content/part08/ch71.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline

train_docs = [
    "neural network learns image patterns",
    "deep learning model classifies images",
    "bank market prices and financial risk",
    "financial market changes affect bank lending",
    "patient health records support medical care",
    "medical treatment helps patient recovery",
]
new_docs = ["patient care uses medical records",
            "bank evaluates financial credit risk"]
model = make_pipeline(
    CountVectorizer(stop_words="english"),
    LatentDirichletAllocation(n_components=3, max_iter=15,
                              random_state=21),
)
model.fit(train_docs)
topic_mix = np.asarray(model.transform(new_docs))
print("شكل مزيج المواضيع:", topic_mix.shape)
print("لكل وثيقة، الموضوع الأرجح:", topic_mix.argmax(axis=1))
print("مجموع الأوزان لكل وثيقة:", np.round(topic_mix.sum(axis=1), 3))

# ────────────────────────────────────────────────────────────────────────
# [005] content/part08/ch71.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer

docs = [
    "model data training",
    "model data validation",
    "model data market",
    "model data patient",
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)
terms = vectorizer.get_feature_names_out()
for term in ("model", "market"):
    j = list(terms).index(term)
    print(term, "weights=", X[:, j].toarray().ravel().round(3))

# ────────────────────────────────────────────────────────────────────────
# [006] content/part08/ch71.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

docs = [
    "neural network learns image patterns",
    "deep learning model classifies images",
    "bank market prices and financial risk",
    "financial market changes affect bank lending",
    "patient health records support medical care",
    "medical treatment helps patient recovery",
]
vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(docs)
lda = LatentDirichletAllocation(n_components=3, max_iter=15,
                                random_state=42).fit(X)
mixture = np.asarray(lda.transform(X))
print("الموضوع الأرجح لكل وثيقة:", mixture.argmax(axis=1))
print("أقصى ثقة لكل وثيقة:", mixture.max(axis=1).round(2))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-71-ch71.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "model learns patterns from training data",
    "training data needs careful validation",
    "market prices change with financial risk",
    "financial models estimate market risk",
]
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(documents)
terms = vectorizer.get_feature_names_out()

print("شكل المصفوفة:", X_tfidf.shape)
print("عدد العناصر غير الصفرية:", X_tfidf.nnz)
for i, row in enumerate(X_tfidf):
    top = np.argsort(row.toarray().ravel())[-3:][::-1]
    print(i, [terms[j] for j in top])

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-71-ch71.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "neural network learns patterns from image data",
    "image model detects objects in a picture",
    "deep learning trains a neural network",
    "bank market prices and financial risk",
    "financial models estimate market prices",
    "bank lending depends on credit risk",
    "patient health records support medical care",
    "medical treatment improves patient health",
    "health care monitors patient recovery",
]
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(documents)
svd = TruncatedSVD(n_components=3, random_state=42)
Z = svd.fit_transform(X)
terms = vectorizer.get_feature_names_out()

print("شكل الوثائق في الفضاء الكامن:", Z.shape)
print("التباين المفسر لكل مكوّن:",
      np.round(svd.explained_variance_ratio_, 3))
for topic, weights in enumerate(svd.components_):
    top = np.argsort(weights)[-5:][::-1]
    print(f"المحور {topic + 1}:", [terms[i] for i in top])

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-71-ch71.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "neural network learns image patterns from training data",
    "deep learning model recognizes images and visual objects",
    "machine learning trains a model to classify image data",
    "computer vision detects objects in digital images",
    "image recognition uses neural networks and learned features",
    "bank market prices reveal financial risk",
    "stock market prices change with investment risk",
    "financial models estimate credit risk and loan defaults",
    "bank lending depends on credit scores and market rates",
    "investment portfolio tracks financial markets and returns",
    "patient health records support medical care",
    "medical treatment helps patient recovery",
    "health care monitors clinical outcomes and diagnosis",
    "doctors evaluate patient symptoms and medical history",
    "clinical care improves patient health and recovery",
]
vectorizer = CountVectorizer(stop_words="english")
X_counts = vectorizer.fit_transform(documents)
lda = LatentDirichletAllocation(n_components=3, max_iter=40,
                                learning_method="batch", random_state=42)
document_topics = np.asarray(lda.fit_transform(X_counts))
terms = vectorizer.get_feature_names_out()

for topic_id, weights in enumerate(lda.components_):
    top = np.argsort(weights)[-5:][::-1]
    print(f"الموضوع {topic_id + 1}:",
          [(terms[i], round(float(weights[i]), 2)) for i in top])
print("توزيع أول وثيقة:", np.round(document_topics[0], 3))
print("مجموع الاحتمالات:", round(float(document_topics[0].sum()), 3))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-71-ch71.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline

train_docs = [
    "neural network learns image patterns",
    "deep learning model classifies images",
    "bank market prices and financial risk",
    "financial market changes affect bank lending",
    "patient health records support medical care",
    "medical treatment helps patient recovery",
]
new_docs = ["patient care uses medical records",
            "bank evaluates financial credit risk"]
model = make_pipeline(
    CountVectorizer(stop_words="english"),
    LatentDirichletAllocation(n_components=3, max_iter=15,
                              random_state=21),
)
model.fit(train_docs)
topic_mix = np.asarray(model.transform(new_docs))
print("شكل مزيج المواضيع:", topic_mix.shape)
print("لكل وثيقة، الموضوع الأرجح:", topic_mix.argmax(axis=1))
print("مجموع الأوزان لكل وثيقة:", np.round(topic_mix.sum(axis=1), 3))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-71-ch71.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer

docs = [
    "model data training",
    "model data validation",
    "model data market",
    "model data patient",
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)
terms = vectorizer.get_feature_names_out()
for term in ("model", "market"):
    j = list(terms).index(term)
    print(term, "weights=", X[:, j].toarray().ravel().round(3))

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-71-ch71.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

docs = [
    "neural network learns image patterns",
    "deep learning model classifies images",
    "bank market prices and financial risk",
    "financial market changes affect bank lending",
    "patient health records support medical care",
    "medical treatment helps patient recovery",
]
vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(docs)
lda = LatentDirichletAllocation(n_components=3, max_iter=15,
                                random_state=42).fit(X)
mixture = np.asarray(lda.transform(X))
print("الموضوع الأرجح لكل وثيقة:", mixture.argmax(axis=1))
print("أقصى ثقة لكل وثيقة:", mixture.max(axis=1).round(2))
