#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — Keras: من الفكرة إلى النموذج
Professional TensorFlow — Keras

Book slug : tensorflow-professional-ar
Chapter   : ch-05-ch05-keras (lesson 5)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/tensorflow-professional-ar/code/ch-05-ch05-keras.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — Keras: من الفكرة إلى النموذج
Professional TensorFlow — Keras

Book slug : tensorflow-professional-ar
Chapter   : ch-05-ch05-keras (lesson 5)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/src/chapters/ch05_keras.md
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers

inputs = keras.Input(shape=(8,), name="order_features")
x = layers.Dense(32, activation="relu")(inputs)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(1, name="delivery_minutes")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="mse", metrics=[keras.metrics.MeanAbsoluteError()])
model.summary()

# ────────────────────────────────────────────────────────────────────────
# [002] book/src/chapters/ch05_keras.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras
from keras import layers

keras.utils.set_random_seed(42)
x = np.random.default_rng(42).normal(size=(256, 8)).astype("float32")
y = (2 * x[:, :1] - x[:, 1:2]).astype("float32")
model = keras.Sequential([layers.Input((8,)), layers.Dense(16, activation="relu"), layers.Dense(1)])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
history = model.fit(x, y, validation_split=0.2, epochs=5, batch_size=32, verbose=0)
print(model.evaluate(x[:32], y[:32], verbose=0))
# المتوقع: خسارة وMAE عدديان، وتتغير القيم مع التدريب.

# ────────────────────────────────────────────────────────────────────────
# [003] book/src/chapters/ch05_keras.md
# ────────────────────────────────────────────────────────────────────────
model.save("delivery.keras")
loaded = keras.models.load_model("delivery.keras")
loaded.export("delivery_savedmodel")

# ────────────────────────────────────────────────────────────────────────
# [004] book/src/chapters/ch05_keras.md
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers

distance = keras.Input(shape=(1,), name="distance_km")
vehicle = keras.Input(shape=(), dtype="string", name="vehicle_type")
lookup = layers.StringLookup(output_mode="one_hot")
lookup.adapt(["bike", "car", "van"])
category = lookup(vehicle)
x = layers.Concatenate()([distance, category])
x = layers.Dense(32, activation="relu")(x)
minutes = layers.Dense(1, name="minutes")(x)
late_probability = layers.Dense(1, activation="sigmoid", name="late")(x)
model = keras.Model({"distance_km": distance, "vehicle_type": vehicle},
                    {"minutes": minutes, "late": late_probability})

# ────────────────────────────────────────────────────────────────────────
# [005] book/src/chapters/ch05_keras.md
# ────────────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("best.keras", save_best_only=True),
]
# مرّر callbacks=callbacks إلى model.fit(...)

# ────────────────────────────────────────────────────────────────────────
# [006] book/src/chapters/ch05_keras.md
# ────────────────────────────────────────────────────────────────────────
predictions = model.predict(x[:4], verbose=0)
print(predictions.shape)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/tensorflow-professional-ar/notebooks/ch-05-ch05-keras.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers

inputs = keras.Input(shape=(8,), name="order_features")
x = layers.Dense(32, activation="relu")(inputs)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(1, name="delivery_minutes")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="mse", metrics=[keras.metrics.MeanAbsoluteError()])
model.summary()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/tensorflow-professional-ar/notebooks/ch-05-ch05-keras.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras
from keras import layers

keras.utils.set_random_seed(42)
x = np.random.default_rng(42).normal(size=(256, 8)).astype("float32")
y = (2 * x[:, :1] - x[:, 1:2]).astype("float32")
model = keras.Sequential([layers.Input((8,)), layers.Dense(16, activation="relu"), layers.Dense(1)])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
history = model.fit(x, y, validation_split=0.2, epochs=5, batch_size=32, verbose=0)
print(model.evaluate(x[:32], y[:32], verbose=0))
# المتوقع: خسارة وMAE عدديان، وتتغير القيم مع التدريب.

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/tensorflow-professional-ar/notebooks/ch-05-ch05-keras.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
model.save("delivery.keras")
loaded = keras.models.load_model("delivery.keras")
loaded.export("delivery_savedmodel")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/tensorflow-professional-ar/notebooks/ch-05-ch05-keras.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers

distance = keras.Input(shape=(1,), name="distance_km")
vehicle = keras.Input(shape=(), dtype="string", name="vehicle_type")
lookup = layers.StringLookup(output_mode="one_hot")
lookup.adapt(["bike", "car", "van"])
category = lookup(vehicle)
x = layers.Concatenate()([distance, category])
x = layers.Dense(32, activation="relu")(x)
minutes = layers.Dense(1, name="minutes")(x)
late_probability = layers.Dense(1, activation="sigmoid", name="late")(x)
model = keras.Model({"distance_km": distance, "vehicle_type": vehicle},
                    {"minutes": minutes, "late": late_probability})

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/tensorflow-professional-ar/notebooks/ch-05-ch05-keras.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("best.keras", save_best_only=True),
]
# مرّر callbacks=callbacks إلى model.fit(...)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/tensorflow-professional-ar/notebooks/ch-05-ch05-keras.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
predictions = model.predict(x[:4], verbose=0)
print(predictions.shape)
