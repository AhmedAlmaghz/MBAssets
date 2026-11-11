# ──────────────────────────────────────────────────────────────────────────
# الفصل 36: تسريع التدريب
# القسم   : 36.4 تحسين خطّ البيانات
# المصدر  : src/chapters/ch36_acceleration.md:85
# الترتيب : 3/6
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# ─── 1. بيانات كبيرة محاكاة
n = 2000
x = np.random.rand(n, 32, 32, 3).astype("float32")
y = np.random.randint(0, 10, size=(n,))

# ─── 2. خطّ بيانات محسّن
(x_train, y_train), _ = keras.datasets.mnist.load_data()   # بيانات حقيقيّة
x_train = x_train.astype("float32") / 255.0

# خطّ بيانات tf.data (خاصّ بمحرّك TensorFlow)
import tensorflow as tf
AUTOTUNE = tf.data.AUTOTUNE

batch_size = 32
ds = (
    keras.utils.timeseries_dataset_from_array(x_train, y_train,
                                              sequence_length=10)
    .cache()                 # يُخزّن في الذاكرة
    .prefetch(AUTOTUNE)      # جلب مسبق
)
# النتيجة المتوقعة: زمن الدورة أقلّ مقارنة بلا cache/prefetch.
