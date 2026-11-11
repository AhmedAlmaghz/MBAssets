# ──────────────────────────────────────────────────────────────────────────
# الفصل 19: طبقات الشبكة التلافيفية في Keras
# القسم   : حساب البارامترات يدويًّا والتحقّق
# المصدر  : src/chapters/ch19_cnn_layers.md:286
# الترتيب : 4/4
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. طبقة Conv2D واحدة: k²·C_in·C_out + C_out
inp = keras.Input((32, 32, 3))
conv = keras.layers.Conv2D(64, 3, padding="same", use_bias=False)
m1 = keras.Model(inp, conv(inp))
manual = 3 * 3 * 3 * 64
actual = m1.layers[-1].count_params()
print(f"manual = {manual}, actual = {actual}, equal = {manual == actual}")
# النتيجة المتوقعة: 1728 == 1728 -> True

# ─── 2. SeparableConv2D: depthwise + pointwise
sep = keras.layers.SeparableConv2D(64, 3, padding="same", use_bias=False)
m2 = keras.Model(inp, sep(inp))
depthwise = 3 * 3 * 3            # مرشّح لكل قناة مدخل
pointwise = 3 * 64               # 1×1 يخلط القنوات
manual_sep = depthwise + pointwise
actual_sep = m2.layers[-1].count_params()
print(f"manual_sep = {manual_sep}, actual = {actual_sep}")
# النتيجة المتوقعة: 219 == 219 (أقلّ من 1728 بـ 7.9×)

# ─── 3. DepthwiseConv2D
dw = keras.layers.DepthwiseConv2D(3, padding="same", use_bias=False)
m3 = keras.Model(inp, dw(inp))
manual_dw = 3 * 3 * 3
print(f"manual_dw = {manual_dw}, actual = {m3.layers[-1].count_params()}")
# النتيجة المتوقعة: 27 == 27 — رخيص جدًّا لكنه لا يخلط القنوات.
