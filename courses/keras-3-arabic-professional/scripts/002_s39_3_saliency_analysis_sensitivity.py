# ──────────────────────────────────────────────────────────────────────────
# الفصل 39: التفسير والتشخيص
# القسم   : 39.3 تحليل الحساسية (Saliency)
# المصدر  : src/chapters/ch39_interpretability.md:84
# الترتيب : 2/5
# ──────────────────────────────────────────────────────────────────────────
def saliency(model, image, class_index=0):
    """خريطة الحساسية: التدرّج بالنسبة للمُدخل."""
    import tensorflow as tf          # شريط التدرّج من المحرّك النشط
    image = tf.Variable(tf.convert_to_tensor(image))   # يجب أن يكون موترًا يُراقَب
    with tf.GradientTape() as tape:
        tape.watch(image)
        pred = model(image[None, ...])
        loss = pred[:, class_index]
    grads = tape.gradient(loss, image)
    return np.abs(np.array(grads.numpy()))

s = saliency(model, image, class_index=0)
# النتيجة المتوقعة: أوتور (32, 32, 3) بقيم غير سالبة.
