#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
Python 3.14 — From Zero to Pro — scikit-learn

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-22-chapter-22 (lesson 22)
Code files: 33 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/python-3-14-from-zero-to-pro/code/ch-22-chapter-22.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
Python 3.14 — From Zero to Pro — scikit-learn

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-22-chapter-22 (lesson 22)
Code files: 32 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] dist/notebooks/chapter-22.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# تثبيت الحزم الناقصة فقط (سريع إن كانت مثبّتة أصلًا)
import importlib.util as _iu
_need = [m for m in ["sklearn", "pandas"] if _iu.find_spec(m) is None]
if _need:
    # %pip install -q scikit-learn pandas

# مجلد عمل معزول حتى لا تتناثر الملفات في جذر الدفتر
WORK = pathlib.Path("book_runtime")
WORK.mkdir(exist_ok=True)
os.chdir(WORK)
print("مجلد العمل:", pathlib.Path.cwd().name)

# ────────────────────────────────────────────────────────────────────────
# [002] dist/notebooks/chapter-22.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.linear_model import LinearRegression

# بيانات تدريب: مساحة المنزل (م²) -> السعر (بالآلاف)
# Features يجب أن تكون دائمًا مصفوفة ثنائية الأبعاد: [[قيمة], [قيمة], ...]
sizes = np.array([[50], [70], [90], [120], [150], [200]])
prices = np.array([150, 200, 250, 320, 400, 500])  # Target أحادي البعد

# إنشاء النموذج وتدريبه
model = LinearRegression()
model.fit(sizes, prices)  # "fit" = التدريب

# التنبؤ بسعر منزل جديد مساحته 100 م²
new_size = np.array([[100]])
predicted_price = model.predict(new_size)

print(f"السعر المتوقع لمنزل 100م²: {predicted_price[0]:.1f} ألف")

# معاملات النموذج المُتعلَّمة (المعادلة: price = slope * size + intercept)
print(f"الميل (slope): {model.coef_[0]:.2f}")
print(f"التقاطع (intercept): {model.intercept_:.2f}")

# ────────────────────────────────────────────────────────────────────────
# [003] dist/notebooks/chapter-22.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. تحميل البيانات
iris = load_iris()
X = iris.data      # Features: طول/عرض البتلة والكأس (4 أعمدة)
y = iris.target    # Target: نوع الزهرة (0, 1, 2)

print(pd.DataFrame(X, columns=iris.feature_names).head())
print("الأنواع:", iris.target_names)

# 2. تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% للاختبار
    random_state=42,     # لضمان نفس التقسيم في كل مرة (قابلية إعادة الإنتاج)
)

print(f"بيانات التدريب: {len(X_train)} عينة")
print(f"بيانات الاختبار: {len(X_test)} عينة")

# 3. اختيار الخوارزمية وتدريب النموذج
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 4. التنبؤ على بيانات الاختبار
predictions = model.predict(X_test)

# 5. تقييم الأداء
accuracy = accuracy_score(y_test, predictions)
print(f"الدقة: {accuracy:.2%}")

# ────────────────────────────────────────────────────────────────────────
# [004] dist/notebooks/chapter-22.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier

# نفس البيانات بالضبط، فقط غيّرنا الخوارزمية
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(f"دقة Random Forest: {accuracy_score(y_test, predictions):.2%}")

# ────────────────────────────────────────────────────────────────────────
# [005] dist/notebooks/chapter-22.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = pd.DataFrame({
    "المدينة": ["الرياض", "جدة", "الرياض", "الدمام"],
    "السعر": [100, 150, 120, 90],
})

# الطريقة 1: Label Encoding -- يحوّل كل فئة لرقم (مناسب للفئات المرتبة فقط!)
encoder = LabelEncoder()
df["المدينة_مشفرة"] = encoder.fit_transform(df["المدينة"])
print(df)

# الطريقة 2: One-Hot Encoding -- عمود منفصل لكل فئة (الأنسب غالبًا للفئات غير المرتبة)
df_encoded = pd.get_dummies(df, columns=["المدينة"])
print(df_encoded)

# ────────────────────────────────────────────────────────────────────────
# [006] dist/notebooks/chapter-22.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# fit_transform على بيانات التدريب فقط
X_train_scaled = scaler.fit_transform(X_train)

# transform فقط (وليس fit) على بيانات الاختبار!
X_test_scaled = scaler.transform(X_test)

# ────────────────────────────────────────────────────────────────────────
# [007] dist/notebooks/chapter-22.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# الـ Pipeline يُطبّق كل الخطوات تلقائيًا وبالترتيب الصحيح
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=3)),
])

pipeline.fit(X_train, y_train)          # يُطبّق scaler ثم classifier تلقائيًا
predictions = pipeline.predict(X_test)  # ونفس الشيء هنا -- بدون تسرّب بيانات!

# ────────────────────────────────────────────────────────────────────────
# [008] dist/notebooks/chapter-22.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

accuracy = accuracy_score(y_test, predictions)        # نسبة التنبؤات الصحيحة
precision = precision_score(y_test, predictions, average="macro")  # دقة التنبؤات الإيجابية
recall = recall_score(y_test, predictions, average="macro")        # نسبة اكتشاف الحالات الإيجابية الفعلية
f1 = f1_score(y_test, predictions, average="macro")   # متوسط توافقي بين precision وrecall

print(classification_report(y_test, predictions))  # تقرير شامل بكل المقاييس دفعة واحدة
print(confusion_matrix(y_test, predictions))        # مصفوفة توضح أنواع الأخطاء بالتفصيل

# ────────────────────────────────────────────────────────────────────────
# [009] dist/notebooks/chapter-22.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, predictions)   # متوسط الخطأ المطلق
mse = mean_squared_error(y_test, predictions)    # متوسط مربع الخطأ (يُعاقب الأخطاء الكبيرة أكثر)
r2 = r2_score(y_test, predictions)                # نسبة التباين المُفسَّر (0 إلى 1، كلما اقترب من 1 كان أفضل)

print(f"متوسط الخطأ المطلق: {mae:.2f}")
print(f"R²: {r2:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [010] dist/notebooks/chapter-22.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 1
years = np.array([[1], [2], [3], [5], [7], [10]])
salary = np.array([4000, 4500, 5200, 6500, 8000, 11000])
model = LinearRegression()
model.fit(years, salary)
print(model.predict([[4]]))

# 2
iris = load_iris()
print(iris.data.shape)
print(iris.target)
print(iris.target_names)

# 3
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)
print(len(X_train), len(X_test))

# ────────────────────────────────────────────────────────────────────────
# [011] dist/notebooks/chapter-22.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# 4
for name, model in [("KNN", KNeighborsClassifier()), ("Decision Tree", DecisionTreeClassifier())]:
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name}: {acc:.2%}")

# 5
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = KNeighborsClassifier()
model_scaled.fit(X_train_scaled, y_train)
acc_scaled = accuracy_score(y_test, model_scaled.predict(X_test_scaled))
print(f"KNN مع القياس: {acc_scaled:.2%}")

# ────────────────────────────────────────────────────────────────────────
# [012] dist/notebooks/chapter-22.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,  # 5-fold cross-validation
    scoring="r2",
)
grid_search.fit(X_train, y_train)

print("أفضل معاملات:", grid_search.best_params_)
print("أفضل نتيجة R2:", grid_search.best_score_)

best_model = grid_search.best_estimator_

# ────────────────────────────────────────────────────────────────────────
# [013] code/chapters/ch22/22-001-sizes.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.3: التثبيت وأول نموذج

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-001-sizes.py
"""

import numpy as np
from sklearn.linear_model import LinearRegression

# بيانات تدريب: مساحة المنزل (م²) -> السعر (بالآلاف)
# Features يجب أن تكون دائمًا مصفوفة ثنائية الأبعاد: [[قيمة], [قيمة], ...]
sizes = np.array([[50], [70], [90], [120], [150], [200]])
prices = np.array([150, 200, 250, 320, 400, 500])  # Target أحادي البعد

# إنشاء النموذج وتدريبه
model = LinearRegression()
model.fit(sizes, prices)  # "fit" = التدريب

# التنبؤ بسعر منزل جديد مساحته 100 م²
new_size = np.array([[100]])
predicted_price = model.predict(new_size)

print(f"السعر المتوقع لمنزل 100م²: {predicted_price[0]:.1f} ألف")

# معاملات النموذج المُتعلَّمة (المعادلة: price = slope * size + intercept)
print(f"الميل (slope): {model.coef_[0]:.2f}")
print(f"التقاطع (intercept): {model.intercept_:.2f}")

# ────────────────────────────────────────────────────────────────────────
# [014] code/chapters/ch22/22-002-iris.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.4: خط أنابيب تعلّم آلي كامل: التصنيف

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-002-iris.py
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. تحميل البيانات
iris = load_iris()
X = iris.data      # Features: طول/عرض البتلة والكأس (4 أعمدة)
y = iris.target    # Target: نوع الزهرة (0, 1, 2)

print(pd.DataFrame(X, columns=iris.feature_names).head())
print("الأنواع:", iris.target_names)

# 2. تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% للاختبار
    random_state=42,     # لضمان نفس التقسيم في كل مرة (قابلية إعادة الإنتاج)
)

print(f"بيانات التدريب: {len(X_train)} عينة")
print(f"بيانات الاختبار: {len(X_test)} عينة")

# 3. اختيار الخوارزمية وتدريب النموذج
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 4. التنبؤ على بيانات الاختبار
predictions = model.predict(X_test)

# 5. تقييم الأداء
accuracy = accuracy_score(y_test, predictions)
print(f"الدقة: {accuracy:.2%}")

# ────────────────────────────────────────────────────────────────────────
# [015] code/chapters/ch22/22-003-model.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.5: خوارزميات شائعة أخرى (نظرة سريعة)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-003-model.py
"""

from sklearn.ensemble import RandomForestClassifier

# نفس البيانات بالضبط، فقط غيّرنا الخوارزمية
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(f"دقة Random Forest: {accuracy_score(y_test, predictions):.2%}")

# ────────────────────────────────────────────────────────────────────────
# [016] code/chapters/ch22/22-004-ex22_6.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.6: تحضير البيانات: الخطوة الأهم (غالبًا)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-004-ex22_6.py
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = pd.DataFrame({
    "المدينة": ["الرياض", "جدة", "الرياض", "الدمام"],
    "السعر": [100, 150, 120, 90],
})

# الطريقة 1: Label Encoding -- يحوّل كل فئة لرقم (مناسب للفئات المرتبة فقط!)
encoder = LabelEncoder()
df["المدينة_مشفرة"] = encoder.fit_transform(df["المدينة"])
print(df)

# الطريقة 2: One-Hot Encoding -- عمود منفصل لكل فئة (الأنسب غالبًا للفئات غير المرتبة)
df_encoded = pd.get_dummies(df, columns=["المدينة"])
print(df_encoded)

# ────────────────────────────────────────────────────────────────────────
# [017] code/chapters/ch22/22-005-scaler.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.6: تحضير البيانات: الخطوة الأهم (غالبًا)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-005-scaler.py
"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# fit_transform على بيانات التدريب فقط
X_train_scaled = scaler.fit_transform(X_train)

# transform فقط (وليس fit) على بيانات الاختبار!
X_test_scaled = scaler.transform(X_test)

# ────────────────────────────────────────────────────────────────────────
# [018] code/chapters/ch22/22-006-pipeline.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.6: تحضير البيانات: الخطوة الأهم (غالبًا)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-006-pipeline.py
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# الـ Pipeline يُطبّق كل الخطوات تلقائيًا وبالترتيب الصحيح
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=3)),
])

pipeline.fit(X_train, y_train)          # يُطبّق scaler ثم classifier تلقائيًا
predictions = pipeline.predict(X_test)  # ونفس الشيء هنا -- بدون تسرّب بيانات!

# ────────────────────────────────────────────────────────────────────────
# [019] code/chapters/ch22/22-007-accuracy.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.7: مقاييس التقييم (Evaluation Metrics)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-007-accuracy.py
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

accuracy = accuracy_score(y_test, predictions)        # نسبة التنبؤات الصحيحة
precision = precision_score(y_test, predictions, average="macro")  # دقة التنبؤات الإيجابية
recall = recall_score(y_test, predictions, average="macro")        # نسبة اكتشاف الحالات الإيجابية الفعلية
f1 = f1_score(y_test, predictions, average="macro")   # متوسط توافقي بين precision وrecall

print(classification_report(y_test, predictions))  # تقرير شامل بكل المقاييس دفعة واحدة
print(confusion_matrix(y_test, predictions))        # مصفوفة توضح أنواع الأخطاء بالتفصيل

# ────────────────────────────────────────────────────────────────────────
# [020] code/chapters/ch22/22-008-mae.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.7: مقاييس التقييم (Evaluation Metrics)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-008-mae.py
"""

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, predictions)   # متوسط الخطأ المطلق
mse = mean_squared_error(y_test, predictions)    # متوسط مربع الخطأ (يُعاقب الأخطاء الكبيرة أكثر)
r2 = r2_score(y_test, predictions)                # نسبة التباين المُفسَّر (0 إلى 1، كلما اقترب من 1 كان أفضل)

print(f"متوسط الخطأ المطلق: {mae:.2f}")
print(f"R²: {r2:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [021] code/chapters/ch22/22-009-loaded_model.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.8: حفظ النموذج واستخدامه لاحقًا

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-009-loaded_model.py
"""

import joblib

# حفظ النموذج المُدرَّب
joblib.dump(pipeline, "model.joblib")

# تحميله لاحقًا (في سكربت آخر، أو حتى بعد أسابيع)
loaded_model = joblib.load("model.joblib")
new_prediction = loaded_model.predict(X_new)

# ────────────────────────────────────────────────────────────────────────
# [022] code/chapters/ch22/22-010-ex22_9.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.9: مشروع تطبيقي متكامل: نظام تنبؤ بأسعار المنازل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-010-ex22_9.py
"""

"""سكربت توليد بيانات منازل تجريبية واقعية."""

import numpy as np
import pandas as pd

np.random.seed(42)
n = 500

sizes = np.random.uniform(50, 400, n)               # المساحة بالمتر المربع
rooms = np.random.randint(1, 8, n)                   # عدد الغرف
age = np.random.uniform(0, 40, n)                    # عمر المبنى بالسنوات
city = np.random.choice(["الرياض", "جدة", "الدمام"], n, p=[0.5, 0.3, 0.2])

city_premium = {"الرياض": 1.3, "جدة": 1.15, "الدمام": 1.0}
base_price = (
    sizes * 8
    + rooms * 50
    - age * 10
    + np.random.normal(0, 100, n)  # ضوضاء عشوائية واقعية
)
price = base_price * [city_premium[c] for c in city]
price = np.clip(price, 100, None)  # لا أسعار سالبة

df = pd.DataFrame({
    "المساحة": sizes.round(1),
    "عدد_الغرف": rooms,
    "عمر_المبنى": age.round(1),
    "المدينة": city,
    "السعر": price.round(0),
})

df.to_csv("data/houses.csv", index=False, encoding="utf-8-sig")
print(f"تم توليد {n} عقار في data/houses.csv")

# ────────────────────────────────────────────────────────────────────────
# [023] code/chapters/ch22/22-011-load_data.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.9: مشروع تطبيقي متكامل: نظام تنبؤ بأسعار المنازل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-011-load_data.py
"""

"""وحدة تحضير بيانات المنازل للتدريب."""

import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = ["المساحة", "عدد_الغرف", "عمر_المبنى", "المدينة"]
TARGET_COLUMN = "السعر"

def load_data(file_path: str) -> pd.DataFrame:
    """تحميل بيانات المنازل من ملف CSV."""
    return pd.read_csv(file_path, encoding="utf-8-sig")

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """تحويل الأعمدة النصية إلى صيغة رقمية مناسبة للنموذج.

    نستخدم One-Hot Encoding للمدينة لأنها فئة غير مرتبة.
    """
    return pd.get_dummies(df[FEATURE_COLUMNS], columns=["المدينة"])

def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
):
    """تقسيم البيانات إلى مجموعتي تدريب واختبار.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = prepare_features(df)
    y = df[TARGET_COLUMN]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# ────────────────────────────────────────────────────────────────────────
# [024] code/chapters/ch22/22-012-evaluate_model.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.9: مشروع تطبيقي متكامل: نظام تنبؤ بأسعار المنازل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-012-evaluate_model.py
"""

"""وحدة تدريب ومقارنة نماذج التنبؤ بأسعار المنازل."""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor

CANDIDATE_MODELS = {
    "الانحدار الخطي": LinearRegression(),
    "شجرة القرار": DecisionTreeRegressor(random_state=42),
    "الغابة العشوائية": RandomForestRegressor(n_estimators=100, random_state=42),
}

def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    """تقييم نموذج مُدرَّب وإرجاع مقاييس الأداء."""
    predictions = model.predict(X_test)
    return {
        "MAE": mean_absolute_error(y_test, predictions),
        "R2": r2_score(y_test, predictions),
    }

def train_and_compare_models(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """تدريب عدة نماذج ومقارنة أدائها.

    Returns:
        DataFrame يلخص أداء كل نموذج، مرتّب من الأفضل للأسوأ حسب R2.
    """
    results = []

    for name, model in CANDIDATE_MODELS.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results.append({"النموذج": name, **metrics})

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
    return results_df

def get_best_model(X_train, y_train):
    """تدريب أفضل نموذج (الغابة العشوائية) وإرجاعه جاهزًا للاستخدام."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, file_path: str) -> None:
    """حفظ النموذج المُدرَّب كملف."""
    joblib.dump(model, file_path)

# ────────────────────────────────────────────────────────────────────────
# [025] code/chapters/ch22/22-013-main.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.9: مشروع تطبيقي متكامل: نظام تنبؤ بأسعار المنازل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-013-main.py
"""

"""نقطة الدخول الرئيسية: تدريب، مقارنة، حفظ، واختبار النموذج."""

import pandas as pd

from house_price_predictor.data_prep import load_data, prepare_features, split_data
from house_price_predictor.training import get_best_model, save_model, train_and_compare_models

DATA_PATH = "data/houses.csv"
MODEL_PATH = "models/house_price_model.joblib"

def main() -> None:
    print("📊 تحميل البيانات...")
    df = load_data(DATA_PATH)
    print(f"   تم تحميل {len(df)} عقار.\n")

    print("✂️ تقسيم البيانات إلى تدريب واختبار...")
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"   تدريب: {len(X_train)} | اختبار: {len(X_test)}\n")

    print("🤖 تدريب ومقارنة النماذج المرشّحة...")
    comparison = train_and_compare_models(X_train, y_train, X_test, y_test)
    print(comparison.to_string(index=False))

    print("\n🏆 تدريب النموذج الأفضل (الغابة العشوائية) على كامل بيانات التدريب...")
    best_model = get_best_model(X_train, y_train)

    print(f"💾 حفظ النموذج في {MODEL_PATH}...")
    save_model(best_model, MODEL_PATH)

    print("\n🔮 مثال تنبؤ: منزل 150م²، 4 غرف، عمر 5 سنوات، في الرياض")
    sample = pd.DataFrame({
        "المساحة": [150], "عدد_الغرف": [4], "عمر_المبنى": [5],
        "المدينة_الدمام": [0], "المدينة_الرياض": [1], "المدينة_جدة": [0],
    })
    # ملاحظة: يجب أن تتطابق أعمدة sample تمامًا مع أعمدة X_train بعد get_dummies
    sample = sample[X_train.columns]  # ضمان نفس ترتيب الأعمدة
    predicted_price = best_model.predict(sample)
    print(f"   السعر المتوقع: {predicted_price[0]:,.0f} ريال")

if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [026] code/chapters/ch22/22-014-years.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.11: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-014-years.py
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 1
years = np.array([[1], [2], [3], [5], [7], [10]])
salary = np.array([4000, 4500, 5200, 6500, 8000, 11000])
model = LinearRegression()
model.fit(years, salary)
print(model.predict([[4]]))

# 2
iris = load_iris()
print(iris.data.shape)
print(iris.target)
print(iris.target_names)

# 3
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)
print(len(X_train), len(X_test))

# ────────────────────────────────────────────────────────────────────────
# [027] code/chapters/ch22/22-015-iris.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.11: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-015-iris.py
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# 4
for name, model in [("KNN", KNeighborsClassifier()), ("Decision Tree", DecisionTreeClassifier())]:
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name}: {acc:.2%}")

# 5
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = KNeighborsClassifier()
model_scaled.fit(X_train_scaled, y_train)
acc_scaled = accuracy_score(y_test, model_scaled.predict(X_test_scaled))
print(f"KNN مع القياس: {acc_scaled:.2%}")

# ────────────────────────────────────────────────────────────────────────
# [028] code/chapters/ch22/22-016-param_grid.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.11: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-016-param_grid.py
"""

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,  # 5-fold cross-validation
    scoring="r2",
)
grid_search.fit(X_train, y_train)

print("أفضل معاملات:", grid_search.best_params_)
print("أفضل نتيجة R2:", grid_search.best_score_)

best_model = grid_search.best_estimator_

# ────────────────────────────────────────────────────────────────────────
# [029] code/chapters/ch22/22-017-model.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.13: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-017-model.py
"""

# ===== النمط الأساسي الموحّد لكل نموذج =====
from sklearn.SOME_MODULE import SomeAlgorithm
model = SomeAlgorithm()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# ===== تقسيم البيانات =====
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===== تحضير البيانات =====
import pandas as pd
pd.get_dummies(df, columns=["فئة_نصية"])        # One-Hot Encoding

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit_transform على التدريب فقط!
X_test_scaled = scaler.transform(X_test)         # transform فقط على الاختبار

# ===== Pipeline =====
from sklearn.pipeline import Pipeline
pipe = Pipeline([("scaler", StandardScaler()), ("model", SomeAlgorithm())])
pipe.fit(X_train, y_train)

# ===== التقييم =====
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
accuracy_score(y_test, predictions)        # للتصنيف
mean_absolute_error(y_test, predictions)   # للانحدار
r2_score(y_test, predictions)              # للانحدار

# ===== الحفظ والتحميل =====
import joblib
joblib.dump(model, "model.joblib")
loaded = joblib.load("model.joblib")

# ────────────────────────────────────────────────────────────────────────
# [030] dist/markdown/chapter-22.md
# ────────────────────────────────────────────────────────────────────────
> model = SomeAlgorithm()      # 1. اختر الخوارزمية
> model.fit(X_train, y_train)  # 2. درّب النموذج
> model.predict(X_new)         # 3. تنبأ ببيانات جديدة
> ```
> بمجرد إتقان هذا النمط، يمكنك استخدام عشرات الخوارزميات المختلفة بنفس الطريقة تقريبًا.

---

## 22.4 خط أنابيب تعلّم آلي كامل: التصنيف

لنبنِ الآن مثالًا أكثر واقعية: تصنيف. سنستخدم بيانات **Iris** الشهيرة (مجموعة بيانات كلاسيكية عن زهور، مدمجة في scikit-learn) للتنبؤ بنوع الزهرة بناءً على قياساتها.

# ────────────────────────────────────────────────────────────────────────
# [031] dist/markdown/chapter-22.md
# ────────────────────────────────────────────────────────────────────────
> **⚠️ تحذير: متى تستخدم Label Encoding ومتى One-Hot Encoding؟**
> `LabelEncoder` يحوّل «الرياض»→0، «جدة»→1، «الدمام»→2. المشكلة: النموذج قد **يفترض خطأً** أن هناك ترتيبًا رياضيًا بينها (كأن الدمام > جدة > الرياض)، وهذا غير منطقي لفئات لا تحمل ترتيبًا طبيعيًا (كالمدن). **القاعدة العامة**: استخدم `OneHotEncoder`/`pd.get_dummies` للفئات **غير المرتبة** (مدن، ألوان، أنواع منتجات)، واستخدم `LabelEncoder` فقط للفئات **المرتبة فعليًا** (ضعيف < متوسط < ممتاز).

### قياس البيانات (Feature Scaling)

بعض الخوارزميات (مثل KNN وSVM) حساسة جدًا لمقياس الأرقام. إذا كان عمود «السعر» يتراوح بين 0-100000 وعمود «عدد الغرف» بين 1-10، فإن الفرق الهائل في المقياس يُضلّل النموذج:

# ────────────────────────────────────────────────────────────────────────
# [032] dist/markdown/chapter-22.md
# ────────────────────────────────────────────────────────────────────────
> **⚠️ ملاحظة مهمة حول مطابقة الأعمدة:** لاحظ في `main.py` السطر `sample = sample[X_train.columns]`. هذا ضروري لأن `pd.get_dummies` قد يُنتج أعمدة بترتيب مختلف أو عددًا مختلفًا حسب البيانات، ويجب أن تتطابق أعمدة أي بيانات جديدة تمامًا (بالاسم والترتيب) مع الأعمدة التي دُرِّب عليها النموذج، وإلا سيُعطي تنبؤًا خاطئًا أو يرفع خطأ.

<details>
<summary>✅ الحل الكامل للمشروع (جميع الملفات مجتمعة)</summary>

جميع ملفات `data_prep.py`، `training.py`، `main.py` كما وردت بالكامل أعلاه تُشكّل الحل النموذجي. خطوات التشغيل الكاملة:

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-22-chapter-22.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# تثبيت الحزم الناقصة فقط (سريع إن كانت مثبّتة أصلًا)
import importlib.util as _iu
_need = [m for m in ["sklearn", "pandas"] if _iu.find_spec(m) is None]
if _need:
    # %pip install -q scikit-learn pandas

# مجلد عمل معزول حتى لا تتناثر الملفات في جذر الدفتر
WORK = pathlib.Path("book_runtime")
WORK.mkdir(exist_ok=True)
os.chdir(WORK)
print("مجلد العمل:", pathlib.Path.cwd().name)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-22-chapter-22.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.linear_model import LinearRegression

# بيانات تدريب: مساحة المنزل (م²) -> السعر (بالآلاف)
# Features يجب أن تكون دائمًا مصفوفة ثنائية الأبعاد: [[قيمة], [قيمة], ...]
sizes = np.array([[50], [70], [90], [120], [150], [200]])
prices = np.array([150, 200, 250, 320, 400, 500])  # Target أحادي البعد

# إنشاء النموذج وتدريبه
model = LinearRegression()
model.fit(sizes, prices)  # "fit" = التدريب

# التنبؤ بسعر منزل جديد مساحته 100 م²
new_size = np.array([[100]])
predicted_price = model.predict(new_size)

print(f"السعر المتوقع لمنزل 100م²: {predicted_price[0]:.1f} ألف")

# معاملات النموذج المُتعلَّمة (المعادلة: price = slope * size + intercept)
print(f"الميل (slope): {model.coef_[0]:.2f}")
print(f"التقاطع (intercept): {model.intercept_:.2f}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-22-chapter-22.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. تحميل البيانات
iris = load_iris()
X = iris.data      # Features: طول/عرض البتلة والكأس (4 أعمدة)
y = iris.target    # Target: نوع الزهرة (0, 1, 2)

print(pd.DataFrame(X, columns=iris.feature_names).head())
print("الأنواع:", iris.target_names)

# 2. تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% للاختبار
    random_state=42,     # لضمان نفس التقسيم في كل مرة (قابلية إعادة الإنتاج)
)

print(f"بيانات التدريب: {len(X_train)} عينة")
print(f"بيانات الاختبار: {len(X_test)} عينة")

# 3. اختيار الخوارزمية وتدريب النموذج
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 4. التنبؤ على بيانات الاختبار
predictions = model.predict(X_test)

# 5. تقييم الأداء
accuracy = accuracy_score(y_test, predictions)
print(f"الدقة: {accuracy:.2%}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-22-chapter-22.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier

# نفس البيانات بالضبط، فقط غيّرنا الخوارزمية
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(f"دقة Random Forest: {accuracy_score(y_test, predictions):.2%}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-22-chapter-22.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = pd.DataFrame({
    "المدينة": ["الرياض", "جدة", "الرياض", "الدمام"],
    "السعر": [100, 150, 120, 90],
})

# الطريقة 1: Label Encoding -- يحوّل كل فئة لرقم (مناسب للفئات المرتبة فقط!)
encoder = LabelEncoder()
df["المدينة_مشفرة"] = encoder.fit_transform(df["المدينة"])
print(df)

# الطريقة 2: One-Hot Encoding -- عمود منفصل لكل فئة (الأنسب غالبًا للفئات غير المرتبة)
df_encoded = pd.get_dummies(df, columns=["المدينة"])
print(df_encoded)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-22-chapter-22.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# fit_transform على بيانات التدريب فقط
X_train_scaled = scaler.fit_transform(X_train)

# transform فقط (وليس fit) على بيانات الاختبار!
X_test_scaled = scaler.transform(X_test)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-22-chapter-22.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# الـ Pipeline يُطبّق كل الخطوات تلقائيًا وبالترتيب الصحيح
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=3)),
])

pipeline.fit(X_train, y_train)          # يُطبّق scaler ثم classifier تلقائيًا
predictions = pipeline.predict(X_test)  # ونفس الشيء هنا -- بدون تسرّب بيانات!

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-22-chapter-22.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

accuracy = accuracy_score(y_test, predictions)        # نسبة التنبؤات الصحيحة
precision = precision_score(y_test, predictions, average="macro")  # دقة التنبؤات الإيجابية
recall = recall_score(y_test, predictions, average="macro")        # نسبة اكتشاف الحالات الإيجابية الفعلية
f1 = f1_score(y_test, predictions, average="macro")   # متوسط توافقي بين precision وrecall

print(classification_report(y_test, predictions))  # تقرير شامل بكل المقاييس دفعة واحدة
print(confusion_matrix(y_test, predictions))        # مصفوفة توضح أنواع الأخطاء بالتفصيل

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-22-chapter-22.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, predictions)   # متوسط الخطأ المطلق
mse = mean_squared_error(y_test, predictions)    # متوسط مربع الخطأ (يُعاقب الأخطاء الكبيرة أكثر)
r2 = r2_score(y_test, predictions)                # نسبة التباين المُفسَّر (0 إلى 1، كلما اقترب من 1 كان أفضل)

print(f"متوسط الخطأ المطلق: {mae:.2f}")
print(f"R²: {r2:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-22-chapter-22.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 1
years = np.array([[1], [2], [3], [5], [7], [10]])
salary = np.array([4000, 4500, 5200, 6500, 8000, 11000])
model = LinearRegression()
model.fit(years, salary)
print(model.predict([[4]]))

# 2
iris = load_iris()
print(iris.data.shape)
print(iris.target)
print(iris.target_names)

# 3
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)
print(len(X_train), len(X_test))

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-22-chapter-22.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# 4
for name, model in [("KNN", KNeighborsClassifier()), ("Decision Tree", DecisionTreeClassifier())]:
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name}: {acc:.2%}")

# 5
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = KNeighborsClassifier()
model_scaled.fit(X_train_scaled, y_train)
acc_scaled = accuracy_score(y_test, model_scaled.predict(X_test_scaled))
print(f"KNN مع القياس: {acc_scaled:.2%}")

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-22-chapter-22.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,  # 5-fold cross-validation
    scoring="r2",
)
grid_search.fit(X_train, y_train)

print("أفضل معاملات:", grid_search.best_params_)
print("أفضل نتيجة R2:", grid_search.best_score_)

best_model = grid_search.best_estimator_

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-22-chapter-22.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.3: التثبيت وأول نموذج

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-001-sizes.py
"""

import numpy as np
from sklearn.linear_model import LinearRegression

# بيانات تدريب: مساحة المنزل (م²) -> السعر (بالآلاف)
# Features يجب أن تكون دائمًا مصفوفة ثنائية الأبعاد: [[قيمة], [قيمة], ...]
sizes = np.array([[50], [70], [90], [120], [150], [200]])
prices = np.array([150, 200, 250, 320, 400, 500])  # Target أحادي البعد

# إنشاء النموذج وتدريبه
model = LinearRegression()
model.fit(sizes, prices)  # "fit" = التدريب

# التنبؤ بسعر منزل جديد مساحته 100 م²
new_size = np.array([[100]])
predicted_price = model.predict(new_size)

print(f"السعر المتوقع لمنزل 100م²: {predicted_price[0]:.1f} ألف")

# معاملات النموذج المُتعلَّمة (المعادلة: price = slope * size + intercept)
print(f"الميل (slope): {model.coef_[0]:.2f}")
print(f"التقاطع (intercept): {model.intercept_:.2f}")

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-22-chapter-22.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.4: خط أنابيب تعلّم آلي كامل: التصنيف

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-002-iris.py
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. تحميل البيانات
iris = load_iris()
X = iris.data      # Features: طول/عرض البتلة والكأس (4 أعمدة)
y = iris.target    # Target: نوع الزهرة (0, 1, 2)

print(pd.DataFrame(X, columns=iris.feature_names).head())
print("الأنواع:", iris.target_names)

# 2. تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% للاختبار
    random_state=42,     # لضمان نفس التقسيم في كل مرة (قابلية إعادة الإنتاج)
)

print(f"بيانات التدريب: {len(X_train)} عينة")
print(f"بيانات الاختبار: {len(X_test)} عينة")

# 3. اختيار الخوارزمية وتدريب النموذج
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 4. التنبؤ على بيانات الاختبار
predictions = model.predict(X_test)

# 5. تقييم الأداء
accuracy = accuracy_score(y_test, predictions)
print(f"الدقة: {accuracy:.2%}")

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-22-chapter-22.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.5: خوارزميات شائعة أخرى (نظرة سريعة)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-003-model.py
"""

from sklearn.ensemble import RandomForestClassifier

# نفس البيانات بالضبط، فقط غيّرنا الخوارزمية
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(f"دقة Random Forest: {accuracy_score(y_test, predictions):.2%}")

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-22-chapter-22.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.6: تحضير البيانات: الخطوة الأهم (غالبًا)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-004-ex22_6.py
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = pd.DataFrame({
    "المدينة": ["الرياض", "جدة", "الرياض", "الدمام"],
    "السعر": [100, 150, 120, 90],
})

# الطريقة 1: Label Encoding -- يحوّل كل فئة لرقم (مناسب للفئات المرتبة فقط!)
encoder = LabelEncoder()
df["المدينة_مشفرة"] = encoder.fit_transform(df["المدينة"])
print(df)

# الطريقة 2: One-Hot Encoding -- عمود منفصل لكل فئة (الأنسب غالبًا للفئات غير المرتبة)
df_encoded = pd.get_dummies(df, columns=["المدينة"])
print(df_encoded)

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-22-chapter-22.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.6: تحضير البيانات: الخطوة الأهم (غالبًا)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-005-scaler.py
"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# fit_transform على بيانات التدريب فقط
X_train_scaled = scaler.fit_transform(X_train)

# transform فقط (وليس fit) على بيانات الاختبار!
X_test_scaled = scaler.transform(X_test)

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-22-chapter-22.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.6: تحضير البيانات: الخطوة الأهم (غالبًا)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-006-pipeline.py
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# الـ Pipeline يُطبّق كل الخطوات تلقائيًا وبالترتيب الصحيح
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=3)),
])

pipeline.fit(X_train, y_train)          # يُطبّق scaler ثم classifier تلقائيًا
predictions = pipeline.predict(X_test)  # ونفس الشيء هنا -- بدون تسرّب بيانات!

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-22-chapter-22.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.7: مقاييس التقييم (Evaluation Metrics)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-007-accuracy.py
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

accuracy = accuracy_score(y_test, predictions)        # نسبة التنبؤات الصحيحة
precision = precision_score(y_test, predictions, average="macro")  # دقة التنبؤات الإيجابية
recall = recall_score(y_test, predictions, average="macro")        # نسبة اكتشاف الحالات الإيجابية الفعلية
f1 = f1_score(y_test, predictions, average="macro")   # متوسط توافقي بين precision وrecall

print(classification_report(y_test, predictions))  # تقرير شامل بكل المقاييس دفعة واحدة
print(confusion_matrix(y_test, predictions))        # مصفوفة توضح أنواع الأخطاء بالتفصيل

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-22-chapter-22.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.7: مقاييس التقييم (Evaluation Metrics)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-008-mae.py
"""

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, predictions)   # متوسط الخطأ المطلق
mse = mean_squared_error(y_test, predictions)    # متوسط مربع الخطأ (يُعاقب الأخطاء الكبيرة أكثر)
r2 = r2_score(y_test, predictions)                # نسبة التباين المُفسَّر (0 إلى 1، كلما اقترب من 1 كان أفضل)

print(f"متوسط الخطأ المطلق: {mae:.2f}")
print(f"R²: {r2:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [022] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-22-chapter-22.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.8: حفظ النموذج واستخدامه لاحقًا

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-009-loaded_model.py
"""

import joblib

# حفظ النموذج المُدرَّب
joblib.dump(pipeline, "model.joblib")

# تحميله لاحقًا (في سكربت آخر، أو حتى بعد أسابيع)
loaded_model = joblib.load("model.joblib")
new_prediction = loaded_model.predict(X_new)

# ────────────────────────────────────────────────────────────────────────
# [023] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-22-chapter-22.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.9: مشروع تطبيقي متكامل: نظام تنبؤ بأسعار المنازل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-010-ex22_9.py
"""

"""سكربت توليد بيانات منازل تجريبية واقعية."""

import numpy as np
import pandas as pd

np.random.seed(42)
n = 500

sizes = np.random.uniform(50, 400, n)               # المساحة بالمتر المربع
rooms = np.random.randint(1, 8, n)                   # عدد الغرف
age = np.random.uniform(0, 40, n)                    # عمر المبنى بالسنوات
city = np.random.choice(["الرياض", "جدة", "الدمام"], n, p=[0.5, 0.3, 0.2])

city_premium = {"الرياض": 1.3, "جدة": 1.15, "الدمام": 1.0}
base_price = (
    sizes * 8
    + rooms * 50
    - age * 10
    + np.random.normal(0, 100, n)  # ضوضاء عشوائية واقعية
)
price = base_price * [city_premium[c] for c in city]
price = np.clip(price, 100, None)  # لا أسعار سالبة

df = pd.DataFrame({
    "المساحة": sizes.round(1),
    "عدد_الغرف": rooms,
    "عمر_المبنى": age.round(1),
    "المدينة": city,
    "السعر": price.round(0),
})

df.to_csv("data/houses.csv", index=False, encoding="utf-8-sig")
print(f"تم توليد {n} عقار في data/houses.csv")

# ────────────────────────────────────────────────────────────────────────
# [024] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-22-chapter-22.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.9: مشروع تطبيقي متكامل: نظام تنبؤ بأسعار المنازل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-011-load_data.py
"""

"""وحدة تحضير بيانات المنازل للتدريب."""

import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = ["المساحة", "عدد_الغرف", "عمر_المبنى", "المدينة"]
TARGET_COLUMN = "السعر"

def load_data(file_path: str) -> pd.DataFrame:
    """تحميل بيانات المنازل من ملف CSV."""
    return pd.read_csv(file_path, encoding="utf-8-sig")

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """تحويل الأعمدة النصية إلى صيغة رقمية مناسبة للنموذج.

    نستخدم One-Hot Encoding للمدينة لأنها فئة غير مرتبة.
    """
    return pd.get_dummies(df[FEATURE_COLUMNS], columns=["المدينة"])

def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
):
    """تقسيم البيانات إلى مجموعتي تدريب واختبار.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = prepare_features(df)
    y = df[TARGET_COLUMN]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# ────────────────────────────────────────────────────────────────────────
# [025] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-22-chapter-22.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.9: مشروع تطبيقي متكامل: نظام تنبؤ بأسعار المنازل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-012-evaluate_model.py
"""

"""وحدة تدريب ومقارنة نماذج التنبؤ بأسعار المنازل."""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor

CANDIDATE_MODELS = {
    "الانحدار الخطي": LinearRegression(),
    "شجرة القرار": DecisionTreeRegressor(random_state=42),
    "الغابة العشوائية": RandomForestRegressor(n_estimators=100, random_state=42),
}

def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    """تقييم نموذج مُدرَّب وإرجاع مقاييس الأداء."""
    predictions = model.predict(X_test)
    return {
        "MAE": mean_absolute_error(y_test, predictions),
        "R2": r2_score(y_test, predictions),
    }

def train_and_compare_models(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """تدريب عدة نماذج ومقارنة أدائها.

    Returns:
        DataFrame يلخص أداء كل نموذج، مرتّب من الأفضل للأسوأ حسب R2.
    """
    results = []

    for name, model in CANDIDATE_MODELS.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results.append({"النموذج": name, **metrics})

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
    return results_df

def get_best_model(X_train, y_train):
    """تدريب أفضل نموذج (الغابة العشوائية) وإرجاعه جاهزًا للاستخدام."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, file_path: str) -> None:
    """حفظ النموذج المُدرَّب كملف."""
    joblib.dump(model, file_path)

# ────────────────────────────────────────────────────────────────────────
# [026] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-22-chapter-22.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.9: مشروع تطبيقي متكامل: نظام تنبؤ بأسعار المنازل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-013-main.py
"""

"""نقطة الدخول الرئيسية: تدريب، مقارنة، حفظ، واختبار النموذج."""

import pandas as pd

from house_price_predictor.data_prep import load_data, prepare_features, split_data
from house_price_predictor.training import get_best_model, save_model, train_and_compare_models

DATA_PATH = "data/houses.csv"
MODEL_PATH = "models/house_price_model.joblib"

def main() -> None:
    print("📊 تحميل البيانات...")
    df = load_data(DATA_PATH)
    print(f"   تم تحميل {len(df)} عقار.\n")

    print("✂️ تقسيم البيانات إلى تدريب واختبار...")
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"   تدريب: {len(X_train)} | اختبار: {len(X_test)}\n")

    print("🤖 تدريب ومقارنة النماذج المرشّحة...")
    comparison = train_and_compare_models(X_train, y_train, X_test, y_test)
    print(comparison.to_string(index=False))

    print("\n🏆 تدريب النموذج الأفضل (الغابة العشوائية) على كامل بيانات التدريب...")
    best_model = get_best_model(X_train, y_train)

    print(f"💾 حفظ النموذج في {MODEL_PATH}...")
    save_model(best_model, MODEL_PATH)

    print("\n🔮 مثال تنبؤ: منزل 150م²، 4 غرف، عمر 5 سنوات، في الرياض")
    sample = pd.DataFrame({
        "المساحة": [150], "عدد_الغرف": [4], "عمر_المبنى": [5],
        "المدينة_الدمام": [0], "المدينة_الرياض": [1], "المدينة_جدة": [0],
    })
    # ملاحظة: يجب أن تتطابق أعمدة sample تمامًا مع أعمدة X_train بعد get_dummies
    sample = sample[X_train.columns]  # ضمان نفس ترتيب الأعمدة
    predicted_price = best_model.predict(sample)
    print(f"   السعر المتوقع: {predicted_price[0]:,.0f} ريال")

if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [027] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-22-chapter-22.ipynb#cell-26
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.11: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-014-years.py
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 1
years = np.array([[1], [2], [3], [5], [7], [10]])
salary = np.array([4000, 4500, 5200, 6500, 8000, 11000])
model = LinearRegression()
model.fit(years, salary)
print(model.predict([[4]]))

# 2
iris = load_iris()
print(iris.data.shape)
print(iris.target)
print(iris.target_names)

# 3
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)
print(len(X_train), len(X_test))

# ────────────────────────────────────────────────────────────────────────
# [028] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-22-chapter-22.ipynb#cell-27
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.11: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-015-iris.py
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# 4
for name, model in [("KNN", KNeighborsClassifier()), ("Decision Tree", DecisionTreeClassifier())]:
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name}: {acc:.2%}")

# 5
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = KNeighborsClassifier()
model_scaled.fit(X_train_scaled, y_train)
acc_scaled = accuracy_score(y_test, model_scaled.predict(X_test_scaled))
print(f"KNN مع القياس: {acc_scaled:.2%}")

# ────────────────────────────────────────────────────────────────────────
# [029] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-22-chapter-22.ipynb#cell-28
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.11: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-016-param_grid.py
"""

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,  # 5-fold cross-validation
    scoring="r2",
)
grid_search.fit(X_train, y_train)

print("أفضل معاملات:", grid_search.best_params_)
print("أفضل نتيجة R2:", grid_search.best_score_)

best_model = grid_search.best_estimator_

# ────────────────────────────────────────────────────────────────────────
# [030] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-22-chapter-22.ipynb#cell-29
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.13: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-017-model.py
"""

# ===== النمط الأساسي الموحّد لكل نموذج =====
from sklearn.SOME_MODULE import SomeAlgorithm
model = SomeAlgorithm()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# ===== تقسيم البيانات =====
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===== تحضير البيانات =====
import pandas as pd
pd.get_dummies(df, columns=["فئة_نصية"])        # One-Hot Encoding

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit_transform على التدريب فقط!
X_test_scaled = scaler.transform(X_test)         # transform فقط على الاختبار

# ===== Pipeline =====
from sklearn.pipeline import Pipeline
pipe = Pipeline([("scaler", StandardScaler()), ("model", SomeAlgorithm())])
pipe.fit(X_train, y_train)

# ===== التقييم =====
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
accuracy_score(y_test, predictions)        # للتصنيف
mean_absolute_error(y_test, predictions)   # للانحدار
r2_score(y_test, predictions)              # للانحدار

# ===== الحفظ والتحميل =====
import joblib
joblib.dump(model, "model.joblib")
loaded = joblib.load("model.joblib")

# ────────────────────────────────────────────────────────────────────────
# [031] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-22-chapter-22.ipynb#cell-30
# ────────────────────────────────────────────────────────────────────────
> model = SomeAlgorithm()      # 1. اختر الخوارزمية
> model.fit(X_train, y_train)  # 2. درّب النموذج
> model.predict(X_new)         # 3. تنبأ ببيانات جديدة
> ```
> بمجرد إتقان هذا النمط، يمكنك استخدام عشرات الخوارزميات المختلفة بنفس الطريقة تقريبًا.

---

## 22.4 خط أنابيب تعلّم آلي كامل: التصنيف

لنبنِ الآن مثالًا أكثر واقعية: تصنيف. سنستخدم بيانات **Iris** الشهيرة (مجموعة بيانات كلاسيكية عن زهور، مدمجة في scikit-learn) للتنبؤ بنوع الزهرة بناءً على قياساتها.

# ────────────────────────────────────────────────────────────────────────
# [032] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-22-chapter-22.ipynb#cell-31
# ────────────────────────────────────────────────────────────────────────
> **⚠️ تحذير: متى تستخدم Label Encoding ومتى One-Hot Encoding؟**
> `LabelEncoder` يحوّل «الرياض»→0، «جدة»→1، «الدمام»→2. المشكلة: النموذج قد **يفترض خطأً** أن هناك ترتيبًا رياضيًا بينها (كأن الدمام > جدة > الرياض)، وهذا غير منطقي لفئات لا تحمل ترتيبًا طبيعيًا (كالمدن). **القاعدة العامة**: استخدم `OneHotEncoder`/`pd.get_dummies` للفئات **غير المرتبة** (مدن، ألوان، أنواع منتجات)، واستخدم `LabelEncoder` فقط للفئات **المرتبة فعليًا** (ضعيف < متوسط < ممتاز).

### قياس البيانات (Feature Scaling)

بعض الخوارزميات (مثل KNN وSVM) حساسة جدًا لمقياس الأرقام. إذا كان عمود «السعر» يتراوح بين 0-100000 وعمود «عدد الغرف» بين 1-10، فإن الفرق الهائل في المقياس يُضلّل النموذج:

# ────────────────────────────────────────────────────────────────────────
# [033] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-22-chapter-22.ipynb#cell-32
# ────────────────────────────────────────────────────────────────────────
> **⚠️ ملاحظة مهمة حول مطابقة الأعمدة:** لاحظ في `main.py` السطر `sample = sample[X_train.columns]`. هذا ضروري لأن `pd.get_dummies` قد يُنتج أعمدة بترتيب مختلف أو عددًا مختلفًا حسب البيانات، ويجب أن تتطابق أعمدة أي بيانات جديدة تمامًا (بالاسم والترتيب) مع الأعمدة التي دُرِّب عليها النموذج، وإلا سيُعطي تنبؤًا خاطئًا أو يرفع خطأ.

<details>
<summary>✅ الحل الكامل للمشروع (جميع الملفات مجتمعة)</summary>

جميع ملفات `data_prep.py`، `training.py`، `main.py` كما وردت بالكامل أعلاه تُشكّل الحل النموذجي. خطوات التشغيل الكاملة:
