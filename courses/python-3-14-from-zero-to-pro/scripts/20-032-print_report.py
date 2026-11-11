#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.6: مشروع تطبيقي متكامل: تحليل مبيعات متجر إلكتروني

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-032-print_report.py
"""

"""نقطة الدخول الرئيسية لتحليل بيانات المبيعات."""

from sales_analysis.analysis import (
    city_product_matrix,
    generate_summary_report,
    monthly_trend,
    revenue_by_city,
    revenue_by_product,
    top_n_products_by_quantity,
)
from sales_analysis.data_loader import clean_sales_data, load_sales_data
from sales_analysis.visualization import (
    plot_city_product_heatmap,
    plot_monthly_trend,
    plot_revenue_by_city_pie,
    plot_revenue_by_product,
)

def print_report(summary: dict) -> None:
    """طباعة تقرير نصي منسّق في الطرفية."""
    print("=" * 50)
    print(" " * 15 + "تقرير تحليل المبيعات")
    print("=" * 50)
    for key, value in summary.items():
        label = key.replace("_", " ")
        if isinstance(value, float):
            print(f"{label}: {value:,.2f}")
        else:
            print(f"{label}: {value}")
    print("=" * 50)

def main() -> None:
    """تشغيل خط التحليل الكامل."""
    print("📊 تحميل البيانات...")
    raw_df = load_sales_data("data/sales_data.csv")

    print("🧹 تنظيف البيانات...")
    df = clean_sales_data(raw_df)
    print(f"   تم تحميل {len(df)} سجل بنجاح.\n")

    print("📈 تحليل البيانات...")
    summary = generate_summary_report(df)
    print_report(summary)

    print("\n🏆 أفضل 5 منتجات من حيث الكمية المباعة:")
    print(top_n_products_by_quantity(df, 5))

    print("\n🎨 توليد الرسوم البيانية...")
    plot_revenue_by_product(revenue_by_product(df))
    plot_revenue_by_city_pie(revenue_by_city(df))
    plot_monthly_trend(monthly_trend(df))
    plot_city_product_heatmap(city_product_matrix(df))

    print("✅ تم حفظ كل الرسوم البيانية في مجلد output/")

if __name__ == "__main__":
    main()
