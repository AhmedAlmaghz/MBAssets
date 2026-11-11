# ──────────────────────────────────────────────────────────────────────────
# الفصل 30: منظومة KerasHub وKerasNLP
# القسم   : 30.5 الضبط الدقيق (Fine-Tuning)
# المصدر  : src/chapters/ch30_kerashub.md:86
# الترتيب : 3/5
# ──────────────────────────────────────────────────────────────────────────
# مرحلة 1: تدريب الرأس فقط (backbone مجمّد)
backbone.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

# مرحلة 2: فكّ تجميد backbone بمعدّل تعلّم أدنى بكثير
backbone.trainable = True
model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="mse")
