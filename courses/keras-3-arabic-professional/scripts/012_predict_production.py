# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : `predict`: إنتاج المخرجات
# المصدر  : src/chapters/ch05_first_model.md:229
# الترتيب : 12/14
# ──────────────────────────────────────────────────────────────────────────
logits = model.predict(x_test[:5], verbose=0)
print("شكل المخرجات:", logits.shape)   # (5, 10)

probs = keras.ops.softmax(logits, axis=-1)
classes = keras.ops.argmax(probs, axis=-1)

for i in range(5):
    print(f"الصورة {i}: logits={np.round(logits[i], 2)}")
    print(f"          argmax={int(classes[i])} · الثقة={float(probs[i].max()):.4f}")
