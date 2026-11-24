#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — ملحق E — فهرس الأخطاء الشائعة
Keras 3 in Arabic — E Error Index

Book slug : keras-3-arabic-professional
Chapter   : app-e-appendix-e-error-index (lesson e)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/app-e-appendix-e-error-index.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — ملحق E — فهرس الأخطاء الشائعة
Keras 3 in Arabic — E Error Index

Book slug : keras-3-arabic-professional
Chapter   : app-e-appendix-e-error-index (lesson e)
Code files: 2 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/app-e-appendix-e-error-index.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — ملحق E — فهرس الأخطاء الشائعة
Keras 3 in Arabic — E Error Index

Book slug : keras-3-arabic-professional
Chapter   : app-e-appendix-e-error-index (lesson e)
Code files: 1 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/appendix/appendix_e_error_index.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

print("keras:", keras.__version__)
print("backend:", keras.backend.backend())

x = np.random.rand(8, 4).astype("float32")
y = np.random.randint(0, 2, size=8)

model = keras.Sequential([
    keras.layers.Input((4,)),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(2),
])
model.compile(optimizer="adam",
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
model.fit(x, y, epochs=1, verbose=0)
print("shape:", model.predict(x, verbose=0).shape)
print("params:", model.count_params())
print("OK ✓")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/app-e-appendix-e-error-index.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

print("keras:", keras.__version__)
print("backend:", keras.backend.backend())

x = np.random.rand(8, 4).astype("float32")
y = np.random.randint(0, 2, size=8)

model = keras.Sequential([
    keras.layers.Input((4,)),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(2),
])
model.compile(optimizer="adam",
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
model.fit(x, y, epochs=1, verbose=0)
print("shape:", model.predict(x, verbose=0).shape)
print("params:", model.count_params())
print("OK ✓")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/app-e-appendix-e-error-index.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — ملحق E — فهرس الأخطاء الشائعة
Keras 3 in Arabic — E Error Index

Book slug : keras-3-arabic-professional
Chapter   : app-e-appendix-e-error-index (lesson e)
Code files: 1 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/appendix/appendix_e_error_index.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

print("keras:", keras.__version__)
print("backend:", keras.backend.backend())

x = np.random.rand(8, 4).astype("float32")
y = np.random.randint(0, 2, size=8)

model = keras.Sequential([
    keras.layers.Input((4,)),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(2),
])
model.compile(optimizer="adam",
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
model.fit(x, y, epochs=1, verbose=0)
print("shape:", model.predict(x, verbose=0).shape)
print("params:", model.count_params())
print("OK ✓")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/app-e-appendix-e-error-index.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

print("keras:", keras.__version__)
print("backend:", keras.backend.backend())

x = np.random.rand(8, 4).astype("float32")
y = np.random.randint(0, 2, size=8)

model = keras.Sequential([
    keras.layers.Input((4,)),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(2),
])
model.compile(optimizer="adam",
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
model.fit(x, y, epochs=1, verbose=0)
print("shape:", model.predict(x, verbose=0).shape)
print("params:", model.count_params())
print("OK ✓")
