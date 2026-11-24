#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — التسلسلات: RNN/LSTM/GRU والسلاسل الزمنية
Professional TensorFlow — Sequences

Book slug : tensorflow-professional-ar
Chapter   : ch-12-ch12-sequences (lesson 12)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/tensorflow-professional-ar/code/ch-12-ch12-sequences.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — التسلسلات: RNN/LSTM/GRU والسلاسل الزمنية
Professional TensorFlow — Sequences

Book slug : tensorflow-professional-ar
Chapter   : ch-12-ch12-sequences (lesson 12)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/src/chapters/ch12_sequences.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras
from keras import layers

series = np.sin(np.arange(1200, dtype="float32") * 0.03)
window, horizon = 48, 1
x = np.stack([series[i:i+window] for i in range(len(series)-window-horizon)])
y = np.stack([series[i+window:i+window+horizon] for i in range(len(series)-window-horizon)])
split = int(len(x) * 0.8)
model = keras.Sequential([layers.Input((window, 1)), layers.LSTM(16), layers.Dense(horizon)])
model.compile(optimizer="adam", loss="mae")
model.fit(x[:split, :, None], y[:split], validation_data=(x[split:, :, None], y[split:]),
          epochs=3, batch_size=32, verbose=0)

# ────────────────────────────────────────────────────────────────────────
# [002] book/src/chapters/ch12_sequences.md
# ────────────────────────────────────────────────────────────────────────
window = 24
windows = tf.signal.frame(series, frame_length=window + 1, frame_step=1)
x, y = windows[:, :-1], windows[:, -1]

# ────────────────────────────────────────────────────────────────────────
# [003] book/src/chapters/ch12_sequences.md
# ────────────────────────────────────────────────────────────────────────
prediction = model(x_batch, training=False)
print(prediction.shape)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/tensorflow-professional-ar/notebooks/ch-12-ch12-sequences.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras
from keras import layers

series = np.sin(np.arange(1200, dtype="float32") * 0.03)
window, horizon = 48, 1
x = np.stack([series[i:i+window] for i in range(len(series)-window-horizon)])
y = np.stack([series[i+window:i+window+horizon] for i in range(len(series)-window-horizon)])
split = int(len(x) * 0.8)
model = keras.Sequential([layers.Input((window, 1)), layers.LSTM(16), layers.Dense(horizon)])
model.compile(optimizer="adam", loss="mae")
model.fit(x[:split, :, None], y[:split], validation_data=(x[split:, :, None], y[split:]),
          epochs=3, batch_size=32, verbose=0)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/tensorflow-professional-ar/notebooks/ch-12-ch12-sequences.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
window = 24
windows = tf.signal.frame(series, frame_length=window + 1, frame_step=1)
x, y = windows[:, :-1], windows[:, -1]

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/tensorflow-professional-ar/notebooks/ch-12-ch12-sequences.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
prediction = model(x_batch, training=False)
print(prediction.shape)
