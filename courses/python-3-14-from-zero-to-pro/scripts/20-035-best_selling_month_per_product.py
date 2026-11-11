#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-035-best_selling_month_per_product.py
"""

# إضافة إلى analysis.py
def best_selling_month_per_product(df: pd.DataFrame) -> pd.Series:
    """الشهر الأفضل مبيعًا لكل منتج."""
    grouped = df.groupby(["المنتج", "الشهر"])["الإجمالي"].sum()
    return grouped.groupby(level="المنتج").idxmax()

# إضافة إلى visualization.py
def plot_quarterly_comparison(df: pd.DataFrame) -> None:
    """مقارنة مبيعات الأرباع الأربعة لكل مدينة."""
    _ensure_output_dir()
    pivot = df.pivot_table(
        values="الإجمالي", index="ربع_السنة", columns="المدينة",
        aggfunc="sum", fill_value=0,
    )
    pivot.plot(kind="bar", figsize=(10, 6))
    plt.title("Quarterly Revenue Comparison by City")
    plt.xlabel("Quarter")
    plt.ylabel("Revenue (SAR)")
    plt.legend(title="City")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "quarterly_comparison.png", dpi=200)
    plt.close()
