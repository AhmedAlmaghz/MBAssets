# ──────────────────────────────────────────────────────────────────────────
# الفصل 21: نقل المعرفة والتدريب الدقيق
# القسم   : 21.5 رأس التصنيف: ما الذي تستبدله؟
# المصدر  : src/chapters/ch21_transfer_learning.md:90
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
# رأس تصنيفي مناسب لمهمّة صغيرة، مبنيًّا فوق مخرجات الشبكة القاعدة
x = base.output                                  # مخرجات الـbackbone
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.3)(x)
out = keras.layers.Dense(2, activation="softmax")(x)   # عدد أصنافك
head = keras.Model(base.input, out)
print("شكل المخرجات:", head.output_shape)
