# ──────────────────────────────────────────────────────────────────────────
# الفصل 18: الرياضيات خلف الالتفاف
# القسم   : حساب أشكال المخرجات يدويًا والتحقّق منها
# المصدر  : src/chapters/ch18_convolution_math.md:277
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
import keras

def conv_out(n_in, k, p, s, d=1):
    """صيغة الخرج أحادية البعد مع التمدّد."""
    k_eff = k + (k - 1) * (d - 1)          # الحجم الفعّال للنواة
    return (n_in + 2 * p - k_eff) // s + 1  # التقريب لأسفل

# حزمة طبقات: كل عنصر (نوع, معاملات)
stack = [
    ("Conv2D", dict(filters=32, kernel_size=3, padding="same", strides=1)),
    ("Conv2D", dict(filters=32, kernel_size=3, padding="same", strides=1)),
    ("MaxPooling2D", dict(pool_size=2, strides=2)),
    ("Conv2D", dict(filters=64, kernel_size=3, padding="same", strides=2)),
    ("Conv2D", dict(filters=64, kernel_size=3, padding="same", strides=1)),
    ("MaxPooling2D", dict(pool_size=2, strides=2)),
    ("Conv2D", dict(filters=128, kernel_size=3, padding="same", strides=2)),
]

# (1) بناء النموذج فعليًا في Keras
inp = keras.layers.Input((32, 32, 3))
x = inp
for kind, kw in stack:
    x = getattr(keras.layers, kind)(**kw)(x)
model = keras.Model(inp, x)

# (2) المرور اليدوي على الأشكال
h = w = c = 32
print(f"{'الطبقة':<14}{'الشكل اليدوي':<22}{'الشكل الفعلي'}")
print("-" * 62)
for kind, kw in stack:
    if kind == "Conv2D":
        h = conv_out(h, kw["kernel_size"], 1 if kw["padding"] == "same" else 0, kw["strides"])
        w = conv_out(w, kw["kernel_size"], 1 if kw["padding"] == "same" else 0, kw["strides"])
        c = kw["filters"]
    else:
        h = conv_out(h, kw["pool_size"], 0, kw["strides"])
        w = conv_out(w, kw["pool_size"], 0, kw["strides"])
    actual = model.get_layer(
        f"{kind.lower()}_{next(i for i, l in enumerate(model.layers) if l.name.startswith(kind.lower()) and False)}"
    ) if False else None  # نعرض الشكل من الكود أدناه
    print(f"{kind:<14}{str((h, w, c)):<22}")

print("\nأشكال الطبقات كما يقرأها Keras:")
for layer in model.layers:
    if hasattr(layer, "output_shape"):
        print(f"  {layer.name:<28}{layer.output_shape}")

# النتيجة المتوقعة (أشكال Keras):
# conv2d (None, 32, 32, 32)
# conv2d_1 (None, 32, 32, 32)
# max_pooling2d (None, 16, 16, 32)
# conv2d_2 (None, 8, 8, 64)
# conv2d_3 (None, 8, 8, 64)
# max_pooling2d_1 (None, 4, 4, 64)
# conv2d_4 (None, 2, 2, 128)
