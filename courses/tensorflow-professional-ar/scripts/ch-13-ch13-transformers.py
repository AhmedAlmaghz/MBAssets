#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — الانتباه والمحوّلات
Professional TensorFlow — Transformers

Book slug : tensorflow-professional-ar
Chapter   : ch-13-ch13-transformers (lesson 13)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/tensorflow-professional-ar/code/ch-13-ch13-transformers.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — الانتباه والمحوّلات
Professional TensorFlow — Transformers

Book slug : tensorflow-professional-ar
Chapter   : ch-13-ch13-transformers (lesson 13)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/src/chapters/ch13_transformers.md
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers

tokens = keras.Input((None,), dtype="int32")
x = layers.Embedding(input_dim=5000, output_dim=64, mask_zero=True)(tokens)
attention = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
x = layers.LayerNormalization()(x + attention)
ffn = layers.Dense(128, activation="gelu")(x)
ffn = layers.Dense(64)(ffn)
x = layers.LayerNormalization()(x + ffn)
pooled = layers.GlobalAveragePooling1D()(x)
outputs = layers.Dense(3, activation="softmax")(pooled)
model = keras.Model(tokens, outputs)

# ────────────────────────────────────────────────────────────────────────
# [002] book/src/chapters/ch13_transformers.md
# ────────────────────────────────────────────────────────────────────────
attention = keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)
context = attention(query=x, value=x, key=x)

# ────────────────────────────────────────────────────────────────────────
# [003] book/src/chapters/ch13_transformers.md
# ────────────────────────────────────────────────────────────────────────
attention = keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)
context = attention(x, x, use_causal_mask=True)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/tensorflow-professional-ar/notebooks/ch-13-ch13-transformers.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers

tokens = keras.Input((None,), dtype="int32")
x = layers.Embedding(input_dim=5000, output_dim=64, mask_zero=True)(tokens)
attention = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
x = layers.LayerNormalization()(x + attention)
ffn = layers.Dense(128, activation="gelu")(x)
ffn = layers.Dense(64)(ffn)
x = layers.LayerNormalization()(x + ffn)
pooled = layers.GlobalAveragePooling1D()(x)
outputs = layers.Dense(3, activation="softmax")(pooled)
model = keras.Model(tokens, outputs)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/tensorflow-professional-ar/notebooks/ch-13-ch13-transformers.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
attention = keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)
context = attention(query=x, value=x, key=x)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/tensorflow-professional-ar/notebooks/ch-13-ch13-transformers.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
attention = keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)
context = attention(x, x, use_causal_mask=True)
