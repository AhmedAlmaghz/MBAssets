# ──────────────────────────────────────────────────────────────────────────
# الفصل 20: شبكة CNN من الصفر
# القسم   : 20.6 تعزيز زمن الاختبار (TTA)
# المصدر  : src/chapters/ch20_cnn_from_scratch.md:137
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
def tta_predict(model, images):
    """متوسّط التنبّؤات عبر أربع نسخ معدّلة للصورة."""
    variants = [
        images,
        keras.ops.flip(images, axis=1),          # قلب أفقي
        keras.ops.flip(images, axis=2),          # قلب رأسي
        keras.ops.flip(keras.ops.flip(images, axis=1), axis=2),
    ]
    preds = [keras.ops.sigmoid(model(v, training=False)) for v in variants]
    return keras.ops.mean(keras.ops.stack(preds, axis=0), axis=0)
