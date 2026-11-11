# ──────────────────────────────────────────────────────────────────────────
# الفصل 29: المحوّلات وآلية الانتباه
# القسم   : 29.5 القناع السببيّ (Causal Masking)
# المصدر  : src/chapters/ch29_transformers.md:128
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
# في Keras 3، القناع يُمرَّر عبر use_causal_mask=True
x = keras.layers.MultiHeadAttention(
    num_heads=4, key_dim=16)(query, value, use_causal_mask=True)
