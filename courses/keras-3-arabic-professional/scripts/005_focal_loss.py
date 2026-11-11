# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : Focal Loss: لماذا يعمل؟
# المصدر  : src/chapters/ch17_multi_task_weights.md:244
# الترتيب : 5/9
# ──────────────────────────────────────────────────────────────────────────
import keras

loss = keras.losses.BinaryFocalCrossentropy(
    from_logits=False,
    alpha=0.25,     # وزن الصنف النادر
    gamma=2.0,      # معامل التركيز
)
model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
