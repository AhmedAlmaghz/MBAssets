# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.3 التنفيذ في Keras 3
# المصدر  : src/chapters/ch07_activations.md:178
# الترتيب : 3/9
# ──────────────────────────────────────────────────────────────────────────
import keras

# نص إلى دالة: هذه هي الآلية التي تستخدمها Dense(activation=...)
fn = keras.activations.get("gelu")
print(fn.__name__)                                  # النتيجة المتوقعة: gelu

# دالة إلى وصف قابل للحفظ: يُستخدم عند تصدير النموذج
print(keras.activations.serialize(keras.activations.gelu))
# النتيجة المتوقعة: {'module': 'keras', 'class_name': 'gelu', 'config': {}, 'registered_name': None}

# دالة تنشيط مخصّصة: دالة بايثون عادية تُقبل في Keras 3
def soft_clamp(z):
    """تنعّم القيم الكبيرة بلطف بدل قطعها (piecewise)."""
    import keras.ops as ops
    return ops.minimum(ops.maximum(z, -6.0), 6.0) * 0.9 + 0.1 * ops.tanh(z / 6.0)

model = keras.Sequential([
    keras.Input(shape=(4,)),
    keras.layers.Dense(8, activation=soft_clamp),
    keras.layers.Dense(1),
])
print('ok')  # النتيجة المتوقعة: ok
