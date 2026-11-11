# ──────────────────────────────────────────────────────────────────────────
# الفصل 10: المقاييس: كيف تقيس نموذجك دون أن ينخدع به الجميع
# القسم   : 10.3 التنفيذ في Keras
# المصدر  : src/chapters/ch10_metrics.md:114
# الترتيب : 2/5
# ──────────────────────────────────────────────────────────────────────────
# مهمة ثنائية: طبقة إخراج بحجم 1، خسارة BCE، ثم sigmoid عند الاستدلال
auc_roc = keras.metrics.AUC(curve="ROC", from_logits=False)
auc_pr = keras.metrics.AUC(curve="PR", from_logits=False)

# مهمة متعدّدة: إخراج after softmax
auc_multi = keras.metrics.AUC(curve="ROC", multi_label=True, num_labels=10)
