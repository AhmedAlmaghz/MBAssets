# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : كيف تتحكّم في الرسم؟
# المصدر  : src/chapters/ch03_tensors.md:188
# الترتيب : 8/12
# ──────────────────────────────────────────────────────────────────────────
import keras


@keras.ops.custom_gradient
def cube(x):
    """f(x) = x^3 مع تدرّج صريح مخصّص."""

    def grad(*args, upstream=None):
        if upstream is None:
            (upstream,) = args
        return 3.0 * x**2 * upstream

    return x**3, grad


v = keras.ops.convert_to_tensor(2.0)
print("f(2) =", float(cube(v)))   # النتيجة المتوقعة: 8.0
