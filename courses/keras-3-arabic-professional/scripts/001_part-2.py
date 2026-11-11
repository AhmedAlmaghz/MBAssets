# ──────────────────────────────────────────────────────────────────────────
# الفصل 13: التنظيم والقيود: كيف تمنع النموذج من الحفظ بدل التعلّم
# القسم   : التنظيمات والقيود المدمجة
# المصدر  : src/chapters/ch13_regularization.md:100
# الترتيب : 1/5
# ──────────────────────────────────────────────────────────────────────────
import keras

# ── Dense مع تنظيم L2 على الأوزان وعلى النشاط
layer = keras.layers.Dense(
    units=128,
    activation="relu",
    kernel_regularizer=keras.regularizers.L2(l2=0.001),
    bias_regularizer=keras.regularizers.L1(0.0001),
    activity_regularizer=keras.regularizers.L2(0.001),
    kernel_constraint=keras.constraints.MaxNorm(max_value=3.0),
)
