#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — مشروع ١: تصنيف البريد المزعج وتحليل المشاعر
Mastering scikit-learn 1.9 — Chapter 99 — Projects

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-99-ch99 (lesson 99)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-99-ch99.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — مشروع ١: تصنيف البريد المزعج وتحليل المشاعر
Mastering scikit-learn 1.9 — Chapter 99 — Projects

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-99-ch99 (lesson 99)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part12/ch99.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

spam_templates = [
    "claim your free prize today",
    "limited offer click to win cash",
    "urgent winner confirm your reward",
    "cheap deal buy now special discount",
]
ham_templates = [
    "meeting moved to tomorrow morning",
    "please review the project report",
    "your appointment is confirmed for Monday",
    "the deployment completed successfully",
]
docs, y = [], []
for label, templates in ((1, spam_templates), (0, ham_templates)):
    for i in range(40):
        docs.append(templates[i % len(templates)] + f" reference {i % 7}")
        y.append(label)
order = np.random.default_rng(42).permutation(len(y))
docs = np.array(docs, dtype=object)[order].tolist()
y = np.asarray(y)[order]
print("عدد الرسائل:", len(docs), "| التوزيع:", np.bincount(y).tolist())
print("مثال مجهول الهوية:", docs[0])

# ────────────────────────────────────────────────────────────────────────
# [002] content/part12/ch99.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

spam = ["claim free prize click now", "limited cash reward today",
        "buy cheap deal now", "urgent winner claim reward"] * 30
ham = ["meeting tomorrow at nine", "please review the report",
       "deployment completed successfully", "appointment confirmed"] * 30
X = np.array(spam + ham, dtype=object)
y = np.r_[np.ones(len(spam), dtype=int), np.zeros(len(ham), dtype=int)]
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
model.fit(Xtr, ytr)
pred = model.predict(Xte)
print(classification_report(yte, pred, target_names=["مشروعة", "مزعجة"],
                            zero_division=0, digits=3))

# ────────────────────────────────────────────────────────────────────────
# [003] content/part12/ch99.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

spam = ["free prize click now", "claim cash reward", "cheap offer buy today"] * 40
ham = ["project meeting tomorrow", "review attached report", "schedule confirmed"] * 40
X = np.array(spam + ham, dtype=object)
y = np.r_[np.ones(len(spam), dtype=int), np.zeros(len(ham), dtype=int)]
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=7)
for name, clf in (("MultinomialNB", MultinomialNB()),
                  ("LinearSVC", LinearSVC(C=1.0))):
    pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
                     ("clf", clf)]).fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    score = (pipe.predict_proba(Xte)[:, 1] if hasattr(clf, "predict_proba")
             else pipe.decision_function(Xte))
    print(name, "precision=", round(precision_score(yte, pred), 3),
          "recall=", round(recall_score(yte, pred), 3),
          "average_precision=", round(average_precision_score(yte, score), 3))

# ────────────────────────────────────────────────────────────────────────
# [004] content/part12/ch99.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix

positive = ["excellent service and helpful staff",
            "the product works very well",
            "fast delivery and great quality"] * 25
negative = ["poor service and rude staff",
            "the product stopped working",
            "late delivery and bad quality"] * 25
X = np.array(positive + negative, dtype=object)
y = np.r_[np.ones(len(positive), dtype=int), np.zeros(len(negative), dtype=int)]
a, b, c, d = train_test_split(X, y, stratify=y, random_state=42)
model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)),
                      LogisticRegression(max_iter=1000)).fit(a, c)
print("مصفوفة الخلط [سلبية، إيجابية]:\n", confusion_matrix(d, model.predict(b)))
print("تنبؤ جملة جديدة:", model.predict(["excellent quality and fast service"])[0])

# ────────────────────────────────────────────────────────────────────────
# [005] content/part12/ch99.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer

texts = ["project status meeting tomorrow", "project report is ready",
         "meeting about deployment status"]
for ngram in ((1, 1), (1, 2)):
    v = TfidfVectorizer(ngram_range=ngram)
    matrix = v.fit_transform(texts)
    print(ngram, matrix.shape, v.get_feature_names_out().tolist())

# ────────────────────────────────────────────────────────────────────────
# [006] content/part12/ch99.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import classification_report

y_true = [0, 0, 0, 1, 1, 1]
y_pred = [0, 1, 0, 1, 0, 1]
print(classification_report(y_true, y_pred,
                            target_names=["مشروعة", "مزعجة"],
                            zero_division=0, digits=3))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-99-ch99.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np

spam_templates = [
    "claim your free prize today",
    "limited offer click to win cash",
    "urgent winner confirm your reward",
    "cheap deal buy now special discount",
]
ham_templates = [
    "meeting moved to tomorrow morning",
    "please review the project report",
    "your appointment is confirmed for Monday",
    "the deployment completed successfully",
]
docs, y = [], []
for label, templates in ((1, spam_templates), (0, ham_templates)):
    for i in range(40):
        docs.append(templates[i % len(templates)] + f" reference {i % 7}")
        y.append(label)
order = np.random.default_rng(42).permutation(len(y))
docs = np.array(docs, dtype=object)[order].tolist()
y = np.asarray(y)[order]
print("عدد الرسائل:", len(docs), "| التوزيع:", np.bincount(y).tolist())
print("مثال مجهول الهوية:", docs[0])

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-99-ch99.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

spam = ["claim free prize click now", "limited cash reward today",
        "buy cheap deal now", "urgent winner claim reward"] * 30
ham = ["meeting tomorrow at nine", "please review the report",
       "deployment completed successfully", "appointment confirmed"] * 30
X = np.array(spam + ham, dtype=object)
y = np.r_[np.ones(len(spam), dtype=int), np.zeros(len(ham), dtype=int)]
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
model.fit(Xtr, ytr)
pred = model.predict(Xte)
print(classification_report(yte, pred, target_names=["مشروعة", "مزعجة"],
                            zero_division=0, digits=3))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-99-ch99.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

spam = ["free prize click now", "claim cash reward", "cheap offer buy today"] * 40
ham = ["project meeting tomorrow", "review attached report", "schedule confirmed"] * 40
X = np.array(spam + ham, dtype=object)
y = np.r_[np.ones(len(spam), dtype=int), np.zeros(len(ham), dtype=int)]
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=7)
for name, clf in (("MultinomialNB", MultinomialNB()),
                  ("LinearSVC", LinearSVC(C=1.0))):
    pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
                     ("clf", clf)]).fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    score = (pipe.predict_proba(Xte)[:, 1] if hasattr(clf, "predict_proba")
             else pipe.decision_function(Xte))
    print(name, "precision=", round(precision_score(yte, pred), 3),
          "recall=", round(recall_score(yte, pred), 3),
          "average_precision=", round(average_precision_score(yte, score), 3))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-99-ch99.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix

positive = ["excellent service and helpful staff",
            "the product works very well",
            "fast delivery and great quality"] * 25
negative = ["poor service and rude staff",
            "the product stopped working",
            "late delivery and bad quality"] * 25
X = np.array(positive + negative, dtype=object)
y = np.r_[np.ones(len(positive), dtype=int), np.zeros(len(negative), dtype=int)]
a, b, c, d = train_test_split(X, y, stratify=y, random_state=42)
model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)),
                      LogisticRegression(max_iter=1000)).fit(a, c)
print("مصفوفة الخلط [سلبية، إيجابية]:\n", confusion_matrix(d, model.predict(b)))
print("تنبؤ جملة جديدة:", model.predict(["excellent quality and fast service"])[0])

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-99-ch99.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer

texts = ["project status meeting tomorrow", "project report is ready",
         "meeting about deployment status"]
for ngram in ((1, 1), (1, 2)):
    v = TfidfVectorizer(ngram_range=ngram)
    matrix = v.fit_transform(texts)
    print(ngram, matrix.shape, v.get_feature_names_out().tolist())

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-99-ch99.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import classification_report

y_true = [0, 0, 0, 1, 1, 1]
y_pred = [0, 1, 0, 1, 0, 1]
print(classification_report(y_true, y_pred,
                            target_names=["مشروعة", "مزعجة"],
                            zero_division=0, digits=3))
