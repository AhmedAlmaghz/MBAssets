#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-020-customers.py
"""

customers = pd.DataFrame({
    "customer_id": [1, 2, 3],
    "اسم": ["أحمد", "سارة", "خالد"],
})

orders = pd.DataFrame({
    "order_id": [101, 102, 103, 104],
    "customer_id": [1, 2, 1, 3],
    "المبلغ": [200, 450, 120, 300],
})

# الدمج (مثل JOIN في SQL)
merged = pd.merge(orders, customers, on="customer_id", how="left")
print(merged)

# دمج/تكديس DataFrames فوق بعضها (مثل إلحاق ملفات شهر بشهر)
jan_sales = pd.DataFrame({"المنتج": ["لابتوب"], "الإجمالي": [3500]})
feb_sales = pd.DataFrame({"المنتج": ["ماوس"], "الإجمالي": [50]})
all_sales = pd.concat([jan_sales, feb_sales], ignore_index=True)
