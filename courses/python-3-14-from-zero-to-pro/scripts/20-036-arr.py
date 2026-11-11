#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.10: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-036-arr.py
"""

# ===== NumPy =====
import numpy as np
arr = np.array([1, 2, 3])
arr.sum(); arr.mean(); arr.max(); arr.min(); arr.std()
filtered = arr[arr > 1]                      # Boolean Indexing

# ===== pandas: القراءة والاستكشاف =====
import pandas as pd
df = pd.read_csv("file.csv", encoding="utf-8")
df.head(); df.tail(); df.info(); df.describe()
df.shape; df.columns; df.dtypes
df.isna().sum()                              # القيم المفقودة

# ===== pandas: الاختيار والتصفية =====
df["col"]                                    # عمود -> Series
df[["col1", "col2"]]                         # أعمدة -> DataFrame
df.loc[0, "col"]                             # بالتسمية
df.iloc[0, 1]                                # بالموضع
df[(df["a"] > 5) & (df["b"] < 10)]           # تصفية بشروط (& | وليس and or)

# ===== pandas: التعديل =====
df["new_col"] = df["a"] * df["b"]
df = df.drop(columns=["col"])
df = df.rename(columns={"old": "new"})
df["col"] = df["col"].fillna(df["col"].mean())
df = df.dropna(subset=["col"])
df = df.sort_values("col", ascending=False)

# ===== pandas: التجميع والدمج =====
df.groupby("col")["val"].sum()
df.groupby("col")["val"].agg(["sum", "mean", "count"])
df.pivot_table(values="val", index="r", columns="c", aggfunc="sum")
pd.merge(df1, df2, on="key", how="left")
pd.concat([df1, df2], ignore_index=True)

# ===== pandas: التصدير =====
df.to_csv("out.csv", index=False, encoding="utf-8-sig")

# ===== Matplotlib =====
import matplotlib.pyplot as plt
plt.plot(x, y); plt.bar(x, y); plt.pie(values, labels=labels)
plt.scatter(x, y); plt.hist(data, bins=20)
plt.title(""); plt.xlabel(""); plt.ylabel(""); plt.legend()
plt.savefig("chart.png", dpi=200); plt.show(); plt.close()
