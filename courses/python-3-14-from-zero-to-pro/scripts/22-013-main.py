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
