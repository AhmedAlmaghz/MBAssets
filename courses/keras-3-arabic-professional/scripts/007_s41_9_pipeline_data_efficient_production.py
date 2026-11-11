# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.9 خطّ البيانات الفعّال في الإنتاج
# المصدر  : src/chapters/ch41_project_vision.md:183
# الترتيب : 7/8
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. خطّ بيانات من مجلّد (في الإنتاج: صور حقيقيّة على القرص)
# train_ds = keras.utils.image_dataset_from_directory(
#     "data/train",            # مجلّد فيه مجلّد فرعيّ لكلّ صنف
#     image_size=(64, 64),
#     batch_size=32,
#     label_mode="int",
# )

# ─── 2. تحسين الأداء: cache + prefetch
# train_ds = (train_ds
#             .cache()                        # يُخزّن بعد أوّل دورة
#             .shuffle(1000)                  # خلط قبل التجميع
#             .prefetch(keras.data.AUTOTUNE)) # جلب مسبق
# النتيجة المتوقعة: GPU لا ينتظر CPU — زمن دورة أقلّ.

# ─── 3. تحقّق سريع من الشكل (محاكاة)
import numpy as np
x = np.random.rand(32, 64, 64, 3).astype("float32")
y = np.random.randint(0, 2, size=(32,))
print(x.shape, y.shape)     # (32, 64, 64, 3) (32,)
