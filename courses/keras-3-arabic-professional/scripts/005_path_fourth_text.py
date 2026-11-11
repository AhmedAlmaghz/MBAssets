# ──────────────────────────────────────────────────────────────────────────
# الفصل 14: هندسة البيانات: من البايتات إلى الدفعات
# القسم   : المسار الرابع: النصوص
# المصدر  : src/chapters/ch14_data_pipeline.md:205
# الترتيب : 5/7
# ──────────────────────────────────────────────────────────────────────────
train_ds = keras.utils.text_dataset_from_directory(
    "data/train_text",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
)
