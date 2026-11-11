# ──────────────────────────────────────────────────────────────────────────
# الفصل 15: المعالجة المسبقة والتعزيز: هندسة المدخلات قبل الشبكة
# القسم   : 15.5 مثال محلول: مصنّف عيّنات الجلد
# المصدر  : src/chapters/ch15_preprocessing_augmentation.md:201
# الترتيب : 2/5
# ──────────────────────────────────────────────────────────────────────────
import keras

preprocess = keras.Sequential([
    keras.layers.Rescaling(1.0 / 255.0),
    keras.layers.RandomFlip("horizontal"),      # مقبول في Dermatologيا
    keras.layers.RandomRotation(0.15),
    keras.layers.RandomZoom(0.15),
    keras.layers.RandomBrightness(0.2),
    keras.layers.Normalization(),
], name="skin_preprocess")
