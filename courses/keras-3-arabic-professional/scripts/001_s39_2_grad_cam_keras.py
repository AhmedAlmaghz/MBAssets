# ──────────────────────────────────────────────────────────────────────────
# الفصل 39: التفسير والتشخيص
# القسم   : 39.2 تنفيذ Grad-CAM في Keras
# المصدر  : src/chapters/ch39_interpretability.md:32
# الترتيب : 1/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج CNN بسيط
inp = keras.Input(shape=(32, 32, 3))
x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
x = keras.layers.Conv2D(16, 3, padding="same", activation="relu", name="last_conv")(x)
x = keras.layers.GlobalAveragePooling2D()(x)
out = keras.layers.Dense(2, activation="softmax")(x)
model = keras.Model(inp, out)

# ─── 2. نموذج فرعيّ: صورة → آخر تلافيفيّ → تنبّؤ
last_conv_layer = model.get_layer("last_conv")
grad_model = keras.Model(model.input,
                         [last_conv_layer.output, model.output])

# ─── 3. حساب Grad-CAM
def gradcam(image, class_index=0):
    """خريطة Grad-CAM لصورة واحدة وصنف محدّد."""
    import tensorflow as tf          # Keras 3 لا يوحّد شريط التدرّج
    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(image[None, ...])
        loss = pred[:, class_index]
    grads = tape.gradient(loss, conv_out)
    # المتوسّط العامّ للتدرّجات
    pooled = keras.ops.mean(grads, axis=(0, 1, 2))
    # الخريطة الموزونة
    heatmap = keras.ops.sum(conv_out[0] * pooled, axis=-1)
    heatmap = keras.ops.relu(heatmap)
    # تطبيع إلى [0, 1]
    heatmap = heatmap / (keras.ops.max(heatmap) + 1e-8)
    return np.array(heatmap)

# ─── 4. تطبيق على صورة
image = np.random.rand(32, 32, 3).astype("float32")
hm = gradcam(image, class_index=0)
# النتيجة المتوقعة: خريطة بحجم (16, 16) بقيم في [0, 1].
