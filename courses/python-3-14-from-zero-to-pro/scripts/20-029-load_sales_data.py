#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.6: مشروع تطبيقي متكامل: تحليل مبيعات متجر إلكتروني

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-029-load_sales_data.py
"""

"""وحدة تحميل وتنظيف بيانات المبيعات."""

from pathlib import Path

import pandas as pd

def load_sales_data(file_path: str | Path) -> pd.DataFrame:
    """تحميل بيانات المبيعات من ملف CSV.

    Args:
        file_path: مسار ملف CSV.

    Returns:
        DataFrame يحتوي بيانات المبيعات الخام.
    """
    df = pd.read_csv(file_path, encoding="utf-8-sig", parse_dates=["التاريخ"])
    return df

def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """تنظيف بيانات المبيعات: معالجة القيم المفقودة والتكرارات.

    Args:
        df: DataFrame الخام.

    Returns:
        DataFrame نظيف جاهز للتحليل.
    """
    clean_df = df.copy()

    # تعويض المدن المفقودة بقيمة واضحة بدلاً من حذف الصفوف (نحتفظ بالبيانات)
    clean_df["المدينة"] = clean_df["المدينة"].fillna("غير محدد")

    # حذف أي صفوف مكررة بالكامل إن وُجدت
    clean_df = clean_df.drop_duplicates()

    # إضافة أعمدة مشتقة مفيدة للتحليل
    clean_df["الشهر"] = clean_df["التاريخ"].dt.month
    clean_df["اسم_الشهر"] = clean_df["التاريخ"].dt.month_name()
    clean_df["ربع_السنة"] = clean_df["التاريخ"].dt.quarter

    return clean_df
