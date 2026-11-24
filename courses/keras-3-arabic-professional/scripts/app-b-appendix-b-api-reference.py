#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — ملحق B — مرجع Keras 3 السريع
Keras 3 in Arabic — B Api Reference

Book slug : keras-3-arabic-professional
Chapter   : app-b-appendix-b-api-reference (lesson b)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/app-b-appendix-b-api-reference.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — ملحق B — مرجع Keras 3 السريع
Keras 3 in Arabic — B Api Reference

Book slug : keras-3-arabic-professional
Chapter   : app-b-appendix-b-api-reference (lesson b)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/app-b-appendix-b-api-reference.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — ملحق B — مرجع Keras 3 السريع
Keras 3 in Arabic — B Api Reference

Book slug : keras-3-arabic-professional
Chapter   : app-b-appendix-b-api-reference (lesson b)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/appendix/appendix_b_api_reference.md
# ────────────────────────────────────────────────────────────────────────
import keras

model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

history = model.fit(x_train, y_train, epochs=20, batch_size=64,
                    validation_data=(x_val, y_val),
                    callbacks=[keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=3, restore_best_weights=True)])

results = model.evaluate(x_test, y_test, return_dict=True)
probs = model.predict(x_test, verbose=0)
model.save("model.keras")
loaded = keras.saving.load_model("model.keras")

# ────────────────────────────────────────────────────────────────────────
# [002] src/appendix/appendix_b_api_reference.md
# ────────────────────────────────────────────────────────────────────────
keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0, weight_decay=1e-4)
keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
keras.optimizers.RMSprop(learning_rate=1e-3, rho=0.9)
keras.optimizers.MultiOptimizer(
    optimizer_map={"encoder": enc_opt, "decoder": dec_opt})   # جديد في 3.15

# ────────────────────────────────────────────────────────────────────────
# [003] src/appendix/appendix_b_api_reference.md
# ────────────────────────────────────────────────────────────────────────
from keras.optimizers.schedules import (
    ExponentialDecay, PolynomialDecay, CosineDecay,
    PiecewiseConstantDecay, InverseTimeDecay,
)

schedule = ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9,
)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=schedule),
    loss="categorical_crossentropy", metrics=["accuracy"])

# ────────────────────────────────────────────────────────────────────────
# [004] src/appendix/appendix_b_api_reference.md
# ────────────────────────────────────────────────────────────────────────
class MyLayer(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.w = self.add_weight(
            shape=(units,), initializer="zeros", trainable=True, name="w"
        )

    def call(self, inputs):
        return inputs * self.w

    def get_config(self):
        return {"units": self.units}

# ────────────────────────────────────────────────────────────────────────
# [005] src/appendix/appendix_b_api_reference.md
# ────────────────────────────────────────────────────────────────────────
keras.__version__          # إصدار Keras
keras.backend.backend()    # الواجهة الخلفية الحالية
keras.config.set_backend("jax")
keras.utils.set_random_seed(42)
model.count_params()
model.layers[i].get_config()
keras.saving.load_model("m.keras", compile=False)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/app-b-appendix-b-api-reference.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import keras

model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

history = model.fit(x_train, y_train, epochs=20, batch_size=64,
                    validation_data=(x_val, y_val),
                    callbacks=[keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=3, restore_best_weights=True)])

results = model.evaluate(x_test, y_test, return_dict=True)
probs = model.predict(x_test, verbose=0)
model.save("model.keras")
loaded = keras.saving.load_model("model.keras")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/app-b-appendix-b-api-reference.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0, weight_decay=1e-4)
keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
keras.optimizers.RMSprop(learning_rate=1e-3, rho=0.9)
keras.optimizers.MultiOptimizer(
    optimizer_map={"encoder": enc_opt, "decoder": dec_opt})   # جديد في 3.15

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/app-b-appendix-b-api-reference.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from keras.optimizers.schedules import (
    ExponentialDecay, PolynomialDecay, CosineDecay,
    PiecewiseConstantDecay, InverseTimeDecay,
)

schedule = ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9,
)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=schedule),
    loss="categorical_crossentropy", metrics=["accuracy"])

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/app-b-appendix-b-api-reference.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class MyLayer(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.w = self.add_weight(
            shape=(units,), initializer="zeros", trainable=True, name="w"
        )

    def call(self, inputs):
        return inputs * self.w

    def get_config(self):
        return {"units": self.units}

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/app-b-appendix-b-api-reference.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
keras.__version__          # إصدار Keras
keras.backend.backend()    # الواجهة الخلفية الحالية
keras.config.set_backend("jax")
keras.utils.set_random_seed(42)
model.count_params()
model.layers[i].get_config()
keras.saving.load_model("m.keras", compile=False)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/app-b-appendix-b-api-reference.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — ملحق B — مرجع Keras 3 السريع
Keras 3 in Arabic — B Api Reference

Book slug : keras-3-arabic-professional
Chapter   : app-b-appendix-b-api-reference (lesson b)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/appendix/appendix_b_api_reference.md
# ────────────────────────────────────────────────────────────────────────
import keras

model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

history = model.fit(x_train, y_train, epochs=20, batch_size=64,
                    validation_data=(x_val, y_val),
                    callbacks=[keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=3, restore_best_weights=True)])

results = model.evaluate(x_test, y_test, return_dict=True)
probs = model.predict(x_test, verbose=0)
model.save("model.keras")
loaded = keras.saving.load_model("model.keras")

# ────────────────────────────────────────────────────────────────────────
# [002] src/appendix/appendix_b_api_reference.md
# ────────────────────────────────────────────────────────────────────────
keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0, weight_decay=1e-4)
keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
keras.optimizers.RMSprop(learning_rate=1e-3, rho=0.9)
keras.optimizers.MultiOptimizer(
    optimizer_map={"encoder": enc_opt, "decoder": dec_opt})   # جديد في 3.15

# ────────────────────────────────────────────────────────────────────────
# [003] src/appendix/appendix_b_api_reference.md
# ────────────────────────────────────────────────────────────────────────
from keras.optimizers.schedules import (
    ExponentialDecay, PolynomialDecay, CosineDecay,
    PiecewiseConstantDecay, InverseTimeDecay,
)

schedule = ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9,
)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=schedule),
    loss="categorical_crossentropy", metrics=["accuracy"])

# ────────────────────────────────────────────────────────────────────────
# [004] src/appendix/appendix_b_api_reference.md
# ────────────────────────────────────────────────────────────────────────
class MyLayer(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.w = self.add_weight(
            shape=(units,), initializer="zeros", trainable=True, name="w"
        )

    def call(self, inputs):
        return inputs * self.w

    def get_config(self):
        return {"units": self.units}

# ────────────────────────────────────────────────────────────────────────
# [005] src/appendix/appendix_b_api_reference.md
# ────────────────────────────────────────────────────────────────────────
keras.__version__          # إصدار Keras
keras.backend.backend()    # الواجهة الخلفية الحالية
keras.config.set_backend("jax")
keras.utils.set_random_seed(42)
model.count_params()
model.layers[i].get_config()
keras.saving.load_model("m.keras", compile=False)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/app-b-appendix-b-api-reference.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import keras

model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

history = model.fit(x_train, y_train, epochs=20, batch_size=64,
                    validation_data=(x_val, y_val),
                    callbacks=[keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=3, restore_best_weights=True)])

results = model.evaluate(x_test, y_test, return_dict=True)
probs = model.predict(x_test, verbose=0)
model.save("model.keras")
loaded = keras.saving.load_model("model.keras")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/app-b-appendix-b-api-reference.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0, weight_decay=1e-4)
keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
keras.optimizers.RMSprop(learning_rate=1e-3, rho=0.9)
keras.optimizers.MultiOptimizer(
    optimizer_map={"encoder": enc_opt, "decoder": dec_opt})   # جديد في 3.15

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/app-b-appendix-b-api-reference.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from keras.optimizers.schedules import (
    ExponentialDecay, PolynomialDecay, CosineDecay,
    PiecewiseConstantDecay, InverseTimeDecay,
)

schedule = ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9,
)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=schedule),
    loss="categorical_crossentropy", metrics=["accuracy"])

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/app-b-appendix-b-api-reference.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class MyLayer(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.w = self.add_weight(
            shape=(units,), initializer="zeros", trainable=True, name="w"
        )

    def call(self, inputs):
        return inputs * self.w

    def get_config(self):
        return {"units": self.units}

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/app-b-appendix-b-api-reference.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
keras.__version__          # إصدار Keras
keras.backend.backend()    # الواجهة الخلفية الحالية
keras.config.set_backend("jax")
keras.utils.set_random_seed(42)
model.count_params()
model.layers[i].get_config()
keras.saving.load_model("m.keras", compile=False)
