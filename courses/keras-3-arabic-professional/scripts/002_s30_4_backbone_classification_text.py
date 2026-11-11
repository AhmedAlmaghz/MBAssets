# ──────────────────────────────────────────────────────────────────────────
# الفصل 30: منظومة KerasHub وKerasNLP
# القسم   : 30.4 تصنيف نصّ فوق backbone
# المصدر  : src/chapters/ch30_kerashub.md:58
# الترتيب : 2/5
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. backbone مُجمّد (استخراج ميزات)
backbone = keras_hub.models.BertBackbone.from_preset("bert_base_en_uncased")
backbone.trainable = False

# ─── 2. النموذج: نصّ → backbone → رأس تصنيف
inp = keras.Input(shape=(1,), dtype="string")
# في KerasHub، النموذج الكامل يتولّى المُرمِّز داخليًّا؛
# إن بُنيت يدويًّا، مرّر معرّفات الرموز مباشرةً:
x = backbone(inp)["sequence_output"]     # (B, L, hidden)
x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dense(32, activation="relu")(x)
out = keras.layers.Dense(2, activation="softmax")(x)
model = keras.Model(inp, out)

# ─── 3. تدريب الرأس فقط
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
