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
