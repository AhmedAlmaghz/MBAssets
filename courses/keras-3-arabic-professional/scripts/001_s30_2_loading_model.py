# ──────────────────────────────────────────────────────────────────────────
# الفصل 30: منظومة KerasHub وKerasNLP
# القسم   : 30.2 تحميل نموذج
# المصدر  : src/chapters/ch30_kerashub.md:33
# الترتيب : 1/5
# ──────────────────────────────────────────────────────────────────────────
import keras_hub
import keras

# تحميل backbone مُدرَّب مسبقًا (لا تنزّل في بيئة معزولة)
backbone = keras_hub.models.BertBackbone.from_preset("bert_base_en_uncased")
# أو تحميل النموذج الكامل مع رأس تصنيف
classifier = keras_hub.models.BertTextClassifier.from_preset(
    "bert_base_en_uncased", num_classes=2)
