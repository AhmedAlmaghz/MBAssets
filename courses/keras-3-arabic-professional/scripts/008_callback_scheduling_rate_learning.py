# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : جدولة معدّل التعلّم من داخل الـcallback
# المصدر  : src/chapters/ch12_callbacks.md:471
# الترتيب : 8/11
# ──────────────────────────────────────────────────────────────────────────
# شطر من المثال أعلاه: قراءة معدّل التعلّم الحالي ثم كتابته
current_lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
self.model.optimizer.learning_rate = new_lr
