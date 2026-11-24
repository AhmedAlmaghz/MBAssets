#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — النصوص: من مستند إلى متجه
Mastering scikit-learn 1.9 — Chapter 18 — Data

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-18-ch18 (lesson 18)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-18-ch18.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — النصوص: من مستند إلى متجه
Mastering scikit-learn 1.9 — Chapter 18 — Data

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-18-ch18 (lesson 18)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part02/ch18.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "الذكاء الاصطناعي يحلل البيانات الضخمة",
    "تعلم الآلة فرع من مجالات الذكاء الاصطناعي",
    "البيانات الضخمة تمثل وقود نماذج تعلم الآلة"
]

vectorizer = CountVectorizer()
X_counts = vectorizer.fit_transform(documents)

vocab = vectorizer.vocabulary_
feature_names = vectorizer.get_feature_names_out()

print(f"عدد المستندات المدخلة: {X_counts.shape[0]}")
print(f"إجمالي الكلمات الفريدة في القاموس: {X_counts.shape[1]}")
print("أول 5 كلمات في القاموس المفهرس:")
for word in feature_names[:5]:
    print(f"  - '{word}' -> الفهرس {vocab[word]}")

# ────────────────────────────────────────────────────────────────────────
# [002] content/part02/ch18.md
# ────────────────────────────────────────────────────────────────────────
# استخراج الكلمات المفردة والعبارات الثنائية مع اشتراط الظهور في مستندين على الأقل
cv_ngrams = CountVectorizer(
    ngram_range=(1, 2),
    min_df=2
)

X_filtered = cv_ngrams.fit_transform(documents)
filtered_features = list(cv_ngrams.get_feature_names_out())

print("الميزات المعتمدة بعد تطبيق التصفية وحزم n-grams (1, 2):")
for feat in filtered_features:
    print(f"  - {feat}")

# ────────────────────────────────────────────────────────────────────────
# [003] content/part02/ch18.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(documents)

# استخراج أوزان المستند الأول
dense_doc0 = X_tfidf.toarray()[0]
features = tfidf.get_feature_names_out()

# ترتيب الكلمات بحسب الوزن الدلالي داخل المستند
weighted_words = sorted(zip(features, dense_doc0), key=lambda x: -x[1])

print("أعلى الكلمات وزناً في المستند الأول:")
for word, score in weighted_words[:4]:
    print(f"  - الكلمة: '{word:<15}' -> وزن TF-IDF: {score:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part02/ch18.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import HashingVectorizer

# تحديد حجم ثابت للفضاء المتجهي: 4096 ميزة
hasher = HashingVectorizer(
    n_features=2**12,
    alternate_sign=False
)

X_hashed = hasher.transform(documents)

print(f"شكل المصفوفة الناتجة من التجزئة: {X_hashed.shape}")
print("هل يحتفظ المحوّل بأي حالة في الذاكرة (requires_fit)؟", hasattr(hasher, "vocabulary_"))
print(f"عدد العناصر غير الصفرية في المستند الأول: {X_hashed[0].nnz}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/part02/ch18.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# بيانات تدريبية نموذجية: نصوص رياضية واقتصادية
corpus_train = [
    "فاز الفريق في المباراة النهائية وتوج بلقب البطولة",
    "سجل المهاجم هدف الفوز في الدقيقة الأخيرة من اللقاء",
    "حقق البطل الميدالية الذهبية في بطولة ألعاب القوى",
    "صعدت مؤشرات أسهم البورصة وانخفضت معدلات التضخم",
    "أعلن البنك المركزي عن رفع أسعار الفائدة المصرفية",
    "تراجعت أرباح الشركة الفصلية نتيجة الركود الاقتصادي العالمي"
]
labels_train = ["رياضة", "رياضة", "رياضة", "اقتصاد", "اقتصاد", "اقتصاد"]

# بناء خط الأنابيب النصي الكامل
text_pipeline = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),
    MultinomialNB(alpha=0.2)
)

# تدريب خط الأنابيب في سطر واحد
text_pipeline.fit(corpus_train, labels_train)

# اختبار خط الأنابيب على نصوص جديدة غير مسبوقة
new_articles = [
    "ارتفاع كبير في تداولات أسهم قطاع البنوك",
    "مباراة حاسمة بين المنتخبين للتأهل إلى النهائي"
]

predictions = text_pipeline.predict(new_articles)
probabilities = text_pipeline.predict_proba(new_articles)

print("نتائج تصنيف النصوص الجديدة:")
for doc, pred, prob in zip(new_articles, predictions, probabilities):
    conf = max(prob)
    print(f"  النص: '{doc}'")
    print(f"  التصنيف المتوقع: [{pred}] بثقة: {conf:.1%}\n")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part02/ch18.md
# ────────────────────────────────────────────────────────────────────────
import re
from sklearn.preprocessing import FunctionTransformer

def clean_arabic(texts):
    cleaned = []
    for t in texts:
        t = re.sub(r"[إأآ]", "ا", t)
        t = re.sub(r"ى\b", "ي", t)
        t = re.sub(r"ة\b", "ه", t)
        cleaned.append(t)
    return cleaned

arabic_cleaner = FunctionTransformer(clean_arabic)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-18-ch18.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "الذكاء الاصطناعي يحلل البيانات الضخمة",
    "تعلم الآلة فرع من مجالات الذكاء الاصطناعي",
    "البيانات الضخمة تمثل وقود نماذج تعلم الآلة"
]

vectorizer = CountVectorizer()
X_counts = vectorizer.fit_transform(documents)

vocab = vectorizer.vocabulary_
feature_names = vectorizer.get_feature_names_out()

print(f"عدد المستندات المدخلة: {X_counts.shape[0]}")
print(f"إجمالي الكلمات الفريدة في القاموس: {X_counts.shape[1]}")
print("أول 5 كلمات في القاموس المفهرس:")
for word in feature_names[:5]:
    print(f"  - '{word}' -> الفهرس {vocab[word]}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-18-ch18.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# استخراج الكلمات المفردة والعبارات الثنائية مع اشتراط الظهور في مستندين على الأقل
cv_ngrams = CountVectorizer(
    ngram_range=(1, 2),
    min_df=2
)

X_filtered = cv_ngrams.fit_transform(documents)
filtered_features = list(cv_ngrams.get_feature_names_out())

print("الميزات المعتمدة بعد تطبيق التصفية وحزم n-grams (1, 2):")
for feat in filtered_features:
    print(f"  - {feat}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-18-ch18.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(documents)

# استخراج أوزان المستند الأول
dense_doc0 = X_tfidf.toarray()[0]
features = tfidf.get_feature_names_out()

# ترتيب الكلمات بحسب الوزن الدلالي داخل المستند
weighted_words = sorted(zip(features, dense_doc0), key=lambda x: -x[1])

print("أعلى الكلمات وزناً في المستند الأول:")
for word, score in weighted_words[:4]:
    print(f"  - الكلمة: '{word:<15}' -> وزن TF-IDF: {score:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-18-ch18.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import HashingVectorizer

# تحديد حجم ثابت للفضاء المتجهي: 4096 ميزة
hasher = HashingVectorizer(
    n_features=2**12,
    alternate_sign=False
)

X_hashed = hasher.transform(documents)

print(f"شكل المصفوفة الناتجة من التجزئة: {X_hashed.shape}")
print("هل يحتفظ المحوّل بأي حالة في الذاكرة (requires_fit)؟", hasattr(hasher, "vocabulary_"))
print(f"عدد العناصر غير الصفرية في المستند الأول: {X_hashed[0].nnz}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-18-ch18.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# بيانات تدريبية نموذجية: نصوص رياضية واقتصادية
corpus_train = [
    "فاز الفريق في المباراة النهائية وتوج بلقب البطولة",
    "سجل المهاجم هدف الفوز في الدقيقة الأخيرة من اللقاء",
    "حقق البطل الميدالية الذهبية في بطولة ألعاب القوى",
    "صعدت مؤشرات أسهم البورصة وانخفضت معدلات التضخم",
    "أعلن البنك المركزي عن رفع أسعار الفائدة المصرفية",
    "تراجعت أرباح الشركة الفصلية نتيجة الركود الاقتصادي العالمي"
]
labels_train = ["رياضة", "رياضة", "رياضة", "اقتصاد", "اقتصاد", "اقتصاد"]

# بناء خط الأنابيب النصي الكامل
text_pipeline = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),
    MultinomialNB(alpha=0.2)
)

# تدريب خط الأنابيب في سطر واحد
text_pipeline.fit(corpus_train, labels_train)

# اختبار خط الأنابيب على نصوص جديدة غير مسبوقة
new_articles = [
    "ارتفاع كبير في تداولات أسهم قطاع البنوك",
    "مباراة حاسمة بين المنتخبين للتأهل إلى النهائي"
]

predictions = text_pipeline.predict(new_articles)
probabilities = text_pipeline.predict_proba(new_articles)

print("نتائج تصنيف النصوص الجديدة:")
for doc, pred, prob in zip(new_articles, predictions, probabilities):
    conf = max(prob)
    print(f"  النص: '{doc}'")
    print(f"  التصنيف المتوقع: [{pred}] بثقة: {conf:.1%}\n")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-18-ch18.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import re
from sklearn.preprocessing import FunctionTransformer

def clean_arabic(texts):
    cleaned = []
    for t in texts:
        t = re.sub(r"[إأآ]", "ا", t)
        t = re.sub(r"ى\b", "ي", t)
        t = re.sub(r"ة\b", "ه", t)
        cleaned.append(t)
    return cleaned

arabic_cleaner = FunctionTransformer(clean_arabic)
