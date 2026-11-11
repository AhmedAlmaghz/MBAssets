#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.6: مشروع تطبيقي متكامل: تحليل مبيعات متجر إلكتروني

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-030-total_revenue.py
"""

"""وحدة تحليل بيانات المبيعات."""

import pandas as pd

def total_revenue(df: pd.DataFrame) -> float:
    """حساب إجمالي الإيرادات."""
    return float(df["الإجمالي"].sum())

def revenue_by_product(df: pd.DataFrame) -> pd.Series:
    """إجمالي الإيرادات لكل منتج، مرتبة تنازليًا."""
    return df.groupby("المنتج")["الإجمالي"].sum().sort_values(ascending=False)

def revenue_by_city(df: pd.DataFrame) -> pd.Series:
    """إجمالي الإيرادات لكل مدينة، مرتبة تنازليًا."""
    return df.groupby("المدينة")["الإجمالي"].sum().sort_values(ascending=False)

def monthly_trend(df: pd.DataFrame) -> pd.Series:
    """اتجاه الإيرادات الشهري عبر العام."""
    return df.groupby("الشهر")["الإجمالي"].sum().sort_index()

def top_n_products_by_quantity(df: pd.DataFrame, n: int = 5) -> pd.Series:
    """أكثر n منتجات مبيعًا من حيث الكمية."""
    return df.groupby("المنتج")["الكمية"].sum().sort_values(ascending=False).head(n)

def average_order_value(df: pd.DataFrame) -> float:
    """متوسط قيمة الطلب الواحد."""
    return float(df["الإجمالي"].mean())

def city_product_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """جدول محوري: الإيرادات لكل منتج في كل مدينة."""
    return df.pivot_table(
        values="الإجمالي",
        index="المدينة",
        columns="المنتج",
        aggfunc="sum",
        fill_value=0,
    )

def generate_summary_report(df: pd.DataFrame) -> dict[str, float | int]:
    """توليد ملخص إحصائي شامل للبيانات."""
    return {
        "إجمالي_الإيرادات": total_revenue(df),
        "عدد_الطلبات": len(df),
        "متوسط_قيمة_الطلب": average_order_value(df),
        "أعلى_منتج_مبيعًا": str(revenue_by_product(df).index[0]),
        "أعلى_مدينة_مبيعًا": str(revenue_by_city(df).index[0]),
    }
