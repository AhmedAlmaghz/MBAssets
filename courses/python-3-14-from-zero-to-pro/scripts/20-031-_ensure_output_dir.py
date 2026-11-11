#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.6: مشروع تطبيقي متكامل: تحليل مبيعات متجر إلكتروني

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-031-_ensure_output_dir.py
"""

"""وحدة توليد الرسوم البيانية لتحليل المبيعات."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = Path("output")

def _ensure_output_dir() -> None:
    """التأكد من وجود مجلد الإخراج."""
    OUTPUT_DIR.mkdir(exist_ok=True)

def plot_revenue_by_product(revenue: pd.Series) -> None:
    """رسم بياني شريطي لإيرادات المنتجات."""
    _ensure_output_dir()
    plt.figure(figsize=(10, 6))
    revenue.plot(kind="bar", color="steelblue", edgecolor="black")
    plt.title("Revenue by Product", fontsize=14, fontweight="bold")
    plt.xlabel("Product")
    plt.ylabel("Revenue (SAR)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "revenue_by_product.png", dpi=200)
    plt.close()

def plot_revenue_by_city_pie(revenue: pd.Series) -> None:
    """رسم دائري لتوزيع الإيرادات حسب المدينة."""
    _ensure_output_dir()
    plt.figure(figsize=(8, 8))
    plt.pie(revenue.values, labels=revenue.index, autopct="%1.1f%%", startangle=90)
    plt.title("Revenue Distribution by City", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "revenue_by_city.png", dpi=200)
    plt.close()

def plot_monthly_trend(trend: pd.Series) -> None:
    """رسم خطي لاتجاه المبيعات الشهري."""
    _ensure_output_dir()
    plt.figure(figsize=(10, 6))
    plt.plot(trend.index, trend.values, marker="o", linewidth=2, color="darkgreen")
    plt.title("Monthly Revenue Trend", fontsize=14, fontweight="bold")
    plt.xlabel("Month")
    plt.ylabel("Revenue (SAR)")
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 13))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "monthly_trend.png", dpi=200)
    plt.close()

def plot_city_product_heatmap(matrix: pd.DataFrame) -> None:
    """رسم Heatmap لمصفوفة المدينة/المنتج."""
    _ensure_output_dir()
    plt.figure(figsize=(10, 6))
    plt.imshow(matrix.values, cmap="YlOrRd", aspect="auto")
    plt.colorbar(label="Revenue (SAR)")
    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=45)
    plt.yticks(range(len(matrix.index)), matrix.index)
    plt.title("Revenue Heatmap: City vs Product", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "city_product_heatmap.png", dpi=200)
    plt.close()
