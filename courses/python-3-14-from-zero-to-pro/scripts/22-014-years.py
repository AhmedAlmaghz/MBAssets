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
