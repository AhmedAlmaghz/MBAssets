# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : قراءة ناتج `fit`
# المصدر  : src/chapters/ch05_first_model.md:188
# الترتيب : 9/14
# ──────────────────────────────────────────────────────────────────────────
print(history.history.keys())
# dict_keys(['loss', 'compile_metrics', 'val_loss', 'val_compile_metrics'])

print("أدنى خسارة تدريب :", min(history.history["loss"]))
print("أدنى خسارة تحقق :", min(history.history["val_loss"]))
