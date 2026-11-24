#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — تدريب Tokenizer مخصص
Mastering Transformers and Large Language Models — Custom Tokenizer

Book slug : mastering-llms-transformers
Chapter   : ch-19-ch19-custom-tokenizer (lesson 19)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-19-ch19-custom-tokenizer.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — تدريب Tokenizer مخصص
Mastering Transformers and Large Language Models — Custom Tokenizer

Book slug : mastering-llms-transformers
Chapter   : ch-19-ch19-custom-tokenizer (lesson 19)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-19-ch19-custom-tokenizer.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — تدريب Tokenizer مخصص
Mastering Transformers and Large Language Models — Custom Tokenizer

Book slug : mastering-llms-transformers
Chapter   : ch-19-ch19-custom-tokenizer (lesson 19)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-05-building-llms/ch19-custom-tokenizer.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# === إعداد البيئة ===
# يفحص هذا القسم الحزم المطلوبة ويحاول تثبيت الناقص منها تلقائيًا.
# شغّل هذه الخلية أولًا، ثم بقية خلايا الدفتر.
import importlib.util
import subprocess
import sys

print("Python:", sys.version.split()[0])

# (اسم وحدة الاستيراد، اسم حزمة pip)
REQUIRED = [
    ("tokenizers", "tokenizers")
]
AUTO_INSTALL = {"PIL", "bs4", "datasets", "faiss", "fasttext", "librosa", "networkx", "opentelemetry", "peft", "prometheus_client", "pydantic", "rank_bm25", "requests", "sentence_transformers", "sentencepiece", "soundfile", "tokenizers", "trafilatura", "transformers", "wandb", "yaml"}


def has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


if not REQUIRED:
    print("لا توجد مكتبات خارجية مطلوبة؛ يكفي Python القياسي.")
else:
    missing = [(module, package) for module, package in REQUIRED if not has_module(module)]
    auto_specs = []
    manual_specs = []
    for module, package in missing:
        specs = package.split()
        if module in AUTO_INSTALL:
            auto_specs.extend(specs)
        else:
            manual_specs.extend(specs)
    auto_specs = list(dict.fromkeys(auto_specs))
    manual_specs = list(dict.fromkeys(manual_specs))

    if auto_specs:
        print("⏳ تثبيت تلقائي:", " ".join(auto_specs))
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *auto_specs])
            print("✅ تم التثبيت.")
        except Exception as exc:
            print("⚠️ تعذّر التثبيت التلقائي:", exc)
            print("شغّل يدويًا: %pip install", " ".join(auto_specs))
    if manual_specs:
        print("⚠️ حزم تحتاج تثبيتًا يدويًا:", " ".join(manual_specs))
        print("شغّل: %pip install", " ".join(manual_specs))
    if not auto_specs and not manual_specs:
        print("✅ كل الحزم المطلوبة متاحة.")

    print("\nالحالة النهائية:")
    for module, package in REQUIRED:
        print(("✅" if has_module(module) else "⚠️"), package)

# ────────────────────────────────────────────────────────────────────────
# [002] content/notebooks/part-05-building-llms/ch19-custom-tokenizer.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import unicodedata

def normalize_arabic_text_nfc(text: str) -> str:
    """
    تطبيع NFC: يحوّل كل التمثيلات البديلة لنفس الحرف المرئي
    إلى تمثيل Unicode قانوني (Canonical) موحّد.
    هذه الخطوة ضرورية قبل تدريب أي Tokenizer على نصوص عربية
    لتجنب معاملة نفس الحرف كرمزين مختلفين في المفردات.
    """
    return unicodedata.normalize("NFC", text)

def demonstrate_normalization_problem():
    """توضيح عملي لمشكلة عدم التطبيع"""
    # نفس الكلمة "أحمد" لكن بتمثيلين مختلفين بايتيًا
    word_precomposed = "\u0623\u062d\u0645\u062f"  # أحمد (الألف مع الهمزة مُركَّبة)
    word_decomposed = "\u0627\u0654\u062d\u0645\u062f"  # أحمد (الألف ثم الهمزة منفصلة)

    print(f"الشكل المرئي للكلمة الأولى: {word_precomposed}")
    print(f"الشكل المرئي للكلمة الثانية: {word_decomposed}")
    print(f"هل هما متطابقتان بايتيًا قبل التطبيع؟ "
          f"{word_precomposed == word_decomposed}")  # False!

    normalized_1 = normalize_arabic_text_nfc(word_precomposed)
    normalized_2 = normalize_arabic_text_nfc(word_decomposed)
    print(f"هل هما متطابقتان بعد تطبيع NFC؟ "
          f"{normalized_1 == normalized_2}")  # True

demonstrate_normalization_problem()

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-05-building-llms/ch19-custom-tokenizer.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import re

class ArabicTextNormalizer:
    """
    خط أنابيب تطبيع شامل للنصوص العربية قبل تدريب Tokenizer.
    يجب تطبيق هذه الخطوات بثبات على كل من بيانات تدريب Tokenizer
    وبيانات تدريب النموذج لاحقًا (وإلا سيحدث عدم تطابق).
    """
    def __init__(self,
                 remove_diacritics: bool = False,
                 normalize_alef_forms: bool = False,
                 normalize_numerals: bool = True):
        """
        remove_diacritics: إزالة التشكيل (محل جدل! بعض المشاريع تحتفظ به
            للنصوص الدينية/الشعرية، لكن معظم LLMs العامة تزيله لتقليل
            حجم المفردات وزيادة الكفاءة).
        normalize_alef_forms: توحيد أ/إ/آ إلى ا (يُفقِد معلومة نحوية لكن
            يقلل تشتت المفردات؛ قرار هندسي يعتمد على حالة الاستخدام).
        normalize_numerals: توحيد الأرقام العربية-الهندية (٠١٢)
            إلى أرقام عربية-غربية (012) لتوحيد التمثيل.
        """
        self.remove_diacritics = remove_diacritics
        self.normalize_alef_forms = normalize_alef_forms
        self.normalize_numerals = normalize_numerals

        self.diacritics_pattern = re.compile(r'[\u064B-\u0652\u0670\u0640]')
        self.alef_forms_pattern = re.compile(r'[إأآا]')
        self.eastern_to_western_digits = str.maketrans(
            "٠١٢٣٤٥٦٧٨٩", "0123456789"
        )

    def normalize(self, text: str) -> str:
        # الخطوة 1: تطبيع NFC أولاً وقبل كل شيء (إلزامي دائمًا)
        text = unicodedata.normalize("NFC", text)

        # الخطوة 2: إزالة التشكيل (اختياري)
        if self.remove_diacritics:
            text = self.diacritics_pattern.sub('', text)

        # الخطوة 3: توحيد أشكال الألف (اختياري، قرار هندسي)
        if self.normalize_alef_forms:
            text = self.alef_forms_pattern.sub('ا', text)

        # الخطوة 4: توحيد الأرقام
        if self.normalize_numerals:
            text = text.translate(self.eastern_to_western_digits)

        # الخطوة 5: توحيد المسافات
        text = re.sub(r'\s+', ' ', text).strip()
        return text

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-05-building-llms/ch19-custom-tokenizer.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
from tokenizers.normalizers import NFC, Sequence
from tokenizers.processors import TemplateProcessing

def train_bpe_tokenizer_for_arabic(
    corpus_files: list,
    vocab_size: int = 50_000,
    min_frequency: int = 2,
    special_tokens: list = None,
) -> Tokenizer:
    """
    تدريب Tokenizer من نوع Byte-Level BPE مُحسَّن للنصوص العربية،
    بنفس المنهجية المستخدمة في تدريب tokenizers نماذج مثل GPT وLLaMA،
    لكن مع تطبيع NFC مُدمَج في خط الأنابيب لضمان الكفاءة مع العربية.
    """
    special_tokens = special_tokens or [
        "<|endoftext|>",   # نهاية الوثيقة
        "<|pad|>",         # الحشو
        "<|unk|>",         # غير معروف (نادر الاستخدام في BPE البايتي)
        "<|user|>",        # بداية رسالة المستخدم (لمرحلة SFT لاحقًا)
        "<|assistant|>",   # بداية رد المساعد
        "<|system|>",      # رسالة النظام
    ]

    # 1. إنشاء Tokenizer بنموذج BPE خام
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))

    # 2. تطبيع NFC كخطوة أولى إلزامية قبل أي تقطيع
    tokenizer.normalizer = Sequence([NFC()])

    # 3. Pre-tokenization بمستوى البايت (ByteLevel)
    # هذا يضمن عدم وجود رموز "unknown" إطلاقًا، لأن أي نص يمكن تفكيكه
    # لبايتات خام (256 قيمة ممكنة) حتى لو لم يظهر الحرف في بيانات التدريب
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = pre_tokenizers.ByteLevel.decoder if hasattr(
        pre_tokenizers.ByteLevel, 'decoder') else None

    # 4. إعداد المدرِّب (Trainer)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # كل البايتات الـ256
        show_progress=True,
    )

    # 5. التدريب الفعلي من ملفات الكوربص
    tokenizer.train(files=corpus_files, trainer=trainer)

    # 6. إضافة معالج القالب (Template) لإضافة رموز البداية/النهاية تلقائيًا
    tokenizer.post_processor = TemplateProcessing(
        single="$A <|endoftext|>",
        special_tokens=[("<|endoftext|>",
                          tokenizer.token_to_id("<|endoftext|>"))],
    )

    return tokenizer

def evaluate_tokenizer_fertility(tokenizer: Tokenizer, test_texts: list) -> dict:
    """
    قياس Fertility: متوسط عدد الـ Tokens لكل كلمة عبر مجموعة اختبار.
    نحسبها أيضًا بمقارنة Tokenizer جديد بـ Tokenizer مرجعي (مثل GPT-2 الأصلي)
    لتوضيح الفارق الكمي في الكفاءة.
    """
    total_words = 0
    total_tokens = 0
    total_chars = 0

    for text in test_texts:
        words = text.split()
        encoded = tokenizer.encode(text)
        total_words += len(words)
        total_tokens += len(encoded.ids)
        total_chars += len(text)

    fertility = total_tokens / max(total_words, 1)
    compression_ratio = total_chars / max(total_tokens, 1)  # أحرف لكل رمز

    return {
        "total_words": total_words,
        "total_tokens": total_tokens,
        "fertility": fertility,                 # كلما قلّ كان أفضل
        "chars_per_token": compression_ratio,    # كلما زاد كان أفضل (ضغط أعلى)
    }

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-05-building-llms/ch19-custom-tokenizer.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def calculate_embedding_parameters_tradeoff(d_model: int,
                                               vocab_sizes: list,
                                               weight_tying: bool = True):
    """
    حساب كلفة معاملات Embedding/Output Head عند أحجام مفردات مختلفة،
    لمساعدة اتخاذ القرار الهندسي عند تصميم Tokenizer.
    """
    print(f"{'حجم المفردات':<15}{'معاملات Embedding':<22}{'معاملات Output':<20}{'الإجمالي':<15}")
    print("-" * 72)
    for vocab_size in vocab_sizes:
        embedding_params = vocab_size * d_model
        # إذا استخدمنا Weight Tying (ربط أوزان Embedding وOutput Head،
        # كما في GPT-2 وLLaMA)، لا نحسب معاملات Output بشكل منفصل
        output_params = 0 if weight_tying else vocab_size * d_model
        total = embedding_params + output_params
        print(f"{vocab_size:<15,}{embedding_params:<22,}{output_params:<20,}{total:<15,}")

calculate_embedding_parameters_tradeoff(
    d_model=4096,
    vocab_sizes=[32_000, 50_000, 100_000, 128_000],
    weight_tying=True
)

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-05-building-llms/ch19-custom-tokenizer.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def compare_tokenizer_fertility_demo():
    """
    محاكاة مبسطة لتوضيح فجوة Fertility بين tokenizer عام وtokenizer عربي مخصص
    (الأرقام الفعلية تعتمد على بيانات تدريب كل tokenizer، لكن الاتجاه حقيقي
    ومُوثَّق في أدبيات المجال، مثل تقرير Jais وتقرير AraGPT2).
    """
    sentence = "تساهم نماذج المحوّلات في تطوير تطبيقات معالجة اللغات الطبيعية بشكل كبير"
    num_words = len(sentence.split())

    # قيم تقديرية واقعية من الأدبيات المنشورة لمقارنات مشابهة
    scenarios = {
        "GPT-2 Tokenizer (إنجليزي أساسًا, 50K مفردات)": 52,
        "GPT-4 Tokenizer (cl100k_base, متعدد اللغات, 100K)": 28,
        "Tokenizer عربي مخصص (50K مفردات مُدرَّب على كوربص عربي)": 13,
    }

    print(f"الجملة: {sentence}")
    print(f"عدد الكلمات: {num_words}\n")
    print(f"{'Tokenizer':<55}{'عدد Tokens':<12}{'Fertility':<10}")
    print("-" * 77)
    for name, num_tokens in scenarios.items():
        fertility = num_tokens / num_words
        print(f"{name:<55}{num_tokens:<12}{fertility:<10.2f}")

compare_tokenizer_fertility_demo()

# ────────────────────────────────────────────────────────────────────────
# [007] content/notebooks/part-05-building-llms/ch19-custom-tokenizer.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-19-ch19-custom-tokenizer.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# === إعداد البيئة ===
# يفحص هذا القسم الحزم المطلوبة ويحاول تثبيت الناقص منها تلقائيًا.
# شغّل هذه الخلية أولًا، ثم بقية خلايا الدفتر.
import importlib.util
import subprocess
import sys

print("Python:", sys.version.split()[0])

# (اسم وحدة الاستيراد، اسم حزمة pip)
REQUIRED = [
    ("tokenizers", "tokenizers")
]
AUTO_INSTALL = {"PIL", "bs4", "datasets", "faiss", "fasttext", "librosa", "networkx", "opentelemetry", "peft", "prometheus_client", "pydantic", "rank_bm25", "requests", "sentence_transformers", "sentencepiece", "soundfile", "tokenizers", "trafilatura", "transformers", "wandb", "yaml"}


def has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


if not REQUIRED:
    print("لا توجد مكتبات خارجية مطلوبة؛ يكفي Python القياسي.")
else:
    missing = [(module, package) for module, package in REQUIRED if not has_module(module)]
    auto_specs = []
    manual_specs = []
    for module, package in missing:
        specs = package.split()
        if module in AUTO_INSTALL:
            auto_specs.extend(specs)
        else:
            manual_specs.extend(specs)
    auto_specs = list(dict.fromkeys(auto_specs))
    manual_specs = list(dict.fromkeys(manual_specs))

    if auto_specs:
        print("⏳ تثبيت تلقائي:", " ".join(auto_specs))
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *auto_specs])
            print("✅ تم التثبيت.")
        except Exception as exc:
            print("⚠️ تعذّر التثبيت التلقائي:", exc)
            print("شغّل يدويًا: %pip install", " ".join(auto_specs))
    if manual_specs:
        print("⚠️ حزم تحتاج تثبيتًا يدويًا:", " ".join(manual_specs))
        print("شغّل: %pip install", " ".join(manual_specs))
    if not auto_specs and not manual_specs:
        print("✅ كل الحزم المطلوبة متاحة.")

    print("\nالحالة النهائية:")
    for module, package in REQUIRED:
        print(("✅" if has_module(module) else "⚠️"), package)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-19-ch19-custom-tokenizer.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import unicodedata

def normalize_arabic_text_nfc(text: str) -> str:
    """
    تطبيع NFC: يحوّل كل التمثيلات البديلة لنفس الحرف المرئي
    إلى تمثيل Unicode قانوني (Canonical) موحّد.
    هذه الخطوة ضرورية قبل تدريب أي Tokenizer على نصوص عربية
    لتجنب معاملة نفس الحرف كرمزين مختلفين في المفردات.
    """
    return unicodedata.normalize("NFC", text)

def demonstrate_normalization_problem():
    """توضيح عملي لمشكلة عدم التطبيع"""
    # نفس الكلمة "أحمد" لكن بتمثيلين مختلفين بايتيًا
    word_precomposed = "\u0623\u062d\u0645\u062f"  # أحمد (الألف مع الهمزة مُركَّبة)
    word_decomposed = "\u0627\u0654\u062d\u0645\u062f"  # أحمد (الألف ثم الهمزة منفصلة)

    print(f"الشكل المرئي للكلمة الأولى: {word_precomposed}")
    print(f"الشكل المرئي للكلمة الثانية: {word_decomposed}")
    print(f"هل هما متطابقتان بايتيًا قبل التطبيع؟ "
          f"{word_precomposed == word_decomposed}")  # False!

    normalized_1 = normalize_arabic_text_nfc(word_precomposed)
    normalized_2 = normalize_arabic_text_nfc(word_decomposed)
    print(f"هل هما متطابقتان بعد تطبيع NFC؟ "
          f"{normalized_1 == normalized_2}")  # True

demonstrate_normalization_problem()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-19-ch19-custom-tokenizer.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import re

class ArabicTextNormalizer:
    """
    خط أنابيب تطبيع شامل للنصوص العربية قبل تدريب Tokenizer.
    يجب تطبيق هذه الخطوات بثبات على كل من بيانات تدريب Tokenizer
    وبيانات تدريب النموذج لاحقًا (وإلا سيحدث عدم تطابق).
    """
    def __init__(self,
                 remove_diacritics: bool = False,
                 normalize_alef_forms: bool = False,
                 normalize_numerals: bool = True):
        """
        remove_diacritics: إزالة التشكيل (محل جدل! بعض المشاريع تحتفظ به
            للنصوص الدينية/الشعرية، لكن معظم LLMs العامة تزيله لتقليل
            حجم المفردات وزيادة الكفاءة).
        normalize_alef_forms: توحيد أ/إ/آ إلى ا (يُفقِد معلومة نحوية لكن
            يقلل تشتت المفردات؛ قرار هندسي يعتمد على حالة الاستخدام).
        normalize_numerals: توحيد الأرقام العربية-الهندية (٠١٢)
            إلى أرقام عربية-غربية (012) لتوحيد التمثيل.
        """
        self.remove_diacritics = remove_diacritics
        self.normalize_alef_forms = normalize_alef_forms
        self.normalize_numerals = normalize_numerals

        self.diacritics_pattern = re.compile(r'[\u064B-\u0652\u0670\u0640]')
        self.alef_forms_pattern = re.compile(r'[إأآا]')
        self.eastern_to_western_digits = str.maketrans(
            "٠١٢٣٤٥٦٧٨٩", "0123456789"
        )

    def normalize(self, text: str) -> str:
        # الخطوة 1: تطبيع NFC أولاً وقبل كل شيء (إلزامي دائمًا)
        text = unicodedata.normalize("NFC", text)

        # الخطوة 2: إزالة التشكيل (اختياري)
        if self.remove_diacritics:
            text = self.diacritics_pattern.sub('', text)

        # الخطوة 3: توحيد أشكال الألف (اختياري، قرار هندسي)
        if self.normalize_alef_forms:
            text = self.alef_forms_pattern.sub('ا', text)

        # الخطوة 4: توحيد الأرقام
        if self.normalize_numerals:
            text = text.translate(self.eastern_to_western_digits)

        # الخطوة 5: توحيد المسافات
        text = re.sub(r'\s+', ' ', text).strip()
        return text

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-19-ch19-custom-tokenizer.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
from tokenizers.normalizers import NFC, Sequence
from tokenizers.processors import TemplateProcessing

def train_bpe_tokenizer_for_arabic(
    corpus_files: list,
    vocab_size: int = 50_000,
    min_frequency: int = 2,
    special_tokens: list = None,
) -> Tokenizer:
    """
    تدريب Tokenizer من نوع Byte-Level BPE مُحسَّن للنصوص العربية،
    بنفس المنهجية المستخدمة في تدريب tokenizers نماذج مثل GPT وLLaMA،
    لكن مع تطبيع NFC مُدمَج في خط الأنابيب لضمان الكفاءة مع العربية.
    """
    special_tokens = special_tokens or [
        "<|endoftext|>",   # نهاية الوثيقة
        "<|pad|>",         # الحشو
        "<|unk|>",         # غير معروف (نادر الاستخدام في BPE البايتي)
        "<|user|>",        # بداية رسالة المستخدم (لمرحلة SFT لاحقًا)
        "<|assistant|>",   # بداية رد المساعد
        "<|system|>",      # رسالة النظام
    ]

    # 1. إنشاء Tokenizer بنموذج BPE خام
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))

    # 2. تطبيع NFC كخطوة أولى إلزامية قبل أي تقطيع
    tokenizer.normalizer = Sequence([NFC()])

    # 3. Pre-tokenization بمستوى البايت (ByteLevel)
    # هذا يضمن عدم وجود رموز "unknown" إطلاقًا، لأن أي نص يمكن تفكيكه
    # لبايتات خام (256 قيمة ممكنة) حتى لو لم يظهر الحرف في بيانات التدريب
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = pre_tokenizers.ByteLevel.decoder if hasattr(
        pre_tokenizers.ByteLevel, 'decoder') else None

    # 4. إعداد المدرِّب (Trainer)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # كل البايتات الـ256
        show_progress=True,
    )

    # 5. التدريب الفعلي من ملفات الكوربص
    tokenizer.train(files=corpus_files, trainer=trainer)

    # 6. إضافة معالج القالب (Template) لإضافة رموز البداية/النهاية تلقائيًا
    tokenizer.post_processor = TemplateProcessing(
        single="$A <|endoftext|>",
        special_tokens=[("<|endoftext|>",
                          tokenizer.token_to_id("<|endoftext|>"))],
    )

    return tokenizer

def evaluate_tokenizer_fertility(tokenizer: Tokenizer, test_texts: list) -> dict:
    """
    قياس Fertility: متوسط عدد الـ Tokens لكل كلمة عبر مجموعة اختبار.
    نحسبها أيضًا بمقارنة Tokenizer جديد بـ Tokenizer مرجعي (مثل GPT-2 الأصلي)
    لتوضيح الفارق الكمي في الكفاءة.
    """
    total_words = 0
    total_tokens = 0
    total_chars = 0

    for text in test_texts:
        words = text.split()
        encoded = tokenizer.encode(text)
        total_words += len(words)
        total_tokens += len(encoded.ids)
        total_chars += len(text)

    fertility = total_tokens / max(total_words, 1)
    compression_ratio = total_chars / max(total_tokens, 1)  # أحرف لكل رمز

    return {
        "total_words": total_words,
        "total_tokens": total_tokens,
        "fertility": fertility,                 # كلما قلّ كان أفضل
        "chars_per_token": compression_ratio,    # كلما زاد كان أفضل (ضغط أعلى)
    }

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-19-ch19-custom-tokenizer.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def calculate_embedding_parameters_tradeoff(d_model: int,
                                               vocab_sizes: list,
                                               weight_tying: bool = True):
    """
    حساب كلفة معاملات Embedding/Output Head عند أحجام مفردات مختلفة،
    لمساعدة اتخاذ القرار الهندسي عند تصميم Tokenizer.
    """
    print(f"{'حجم المفردات':<15}{'معاملات Embedding':<22}{'معاملات Output':<20}{'الإجمالي':<15}")
    print("-" * 72)
    for vocab_size in vocab_sizes:
        embedding_params = vocab_size * d_model
        # إذا استخدمنا Weight Tying (ربط أوزان Embedding وOutput Head،
        # كما في GPT-2 وLLaMA)، لا نحسب معاملات Output بشكل منفصل
        output_params = 0 if weight_tying else vocab_size * d_model
        total = embedding_params + output_params
        print(f"{vocab_size:<15,}{embedding_params:<22,}{output_params:<20,}{total:<15,}")

calculate_embedding_parameters_tradeoff(
    d_model=4096,
    vocab_sizes=[32_000, 50_000, 100_000, 128_000],
    weight_tying=True
)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-19-ch19-custom-tokenizer.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def compare_tokenizer_fertility_demo():
    """
    محاكاة مبسطة لتوضيح فجوة Fertility بين tokenizer عام وtokenizer عربي مخصص
    (الأرقام الفعلية تعتمد على بيانات تدريب كل tokenizer، لكن الاتجاه حقيقي
    ومُوثَّق في أدبيات المجال، مثل تقرير Jais وتقرير AraGPT2).
    """
    sentence = "تساهم نماذج المحوّلات في تطوير تطبيقات معالجة اللغات الطبيعية بشكل كبير"
    num_words = len(sentence.split())

    # قيم تقديرية واقعية من الأدبيات المنشورة لمقارنات مشابهة
    scenarios = {
        "GPT-2 Tokenizer (إنجليزي أساسًا, 50K مفردات)": 52,
        "GPT-4 Tokenizer (cl100k_base, متعدد اللغات, 100K)": 28,
        "Tokenizer عربي مخصص (50K مفردات مُدرَّب على كوربص عربي)": 13,
    }

    print(f"الجملة: {sentence}")
    print(f"عدد الكلمات: {num_words}\n")
    print(f"{'Tokenizer':<55}{'عدد Tokens':<12}{'Fertility':<10}")
    print("-" * 77)
    for name, num_tokens in scenarios.items():
        fertility = num_tokens / num_words
        print(f"{name:<55}{num_tokens:<12}{fertility:<10.2f}")

compare_tokenizer_fertility_demo()

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-llms-transformers/notebooks/ch-19-ch19-custom-tokenizer.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-19-ch19-custom-tokenizer.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — تدريب Tokenizer مخصص
Mastering Transformers and Large Language Models — Custom Tokenizer

Book slug : mastering-llms-transformers
Chapter   : ch-19-ch19-custom-tokenizer (lesson 19)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-05-building-llms/ch19-custom-tokenizer.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# === إعداد البيئة ===
# يفحص هذا القسم الحزم المطلوبة ويحاول تثبيت الناقص منها تلقائيًا.
# شغّل هذه الخلية أولًا، ثم بقية خلايا الدفتر.
import importlib.util
import subprocess
import sys

print("Python:", sys.version.split()[0])

# (اسم وحدة الاستيراد، اسم حزمة pip)
REQUIRED = [
    ("tokenizers", "tokenizers")
]
AUTO_INSTALL = {"PIL", "bs4", "datasets", "faiss", "fasttext", "librosa", "networkx", "opentelemetry", "peft", "prometheus_client", "pydantic", "rank_bm25", "requests", "sentence_transformers", "sentencepiece", "soundfile", "tokenizers", "trafilatura", "transformers", "wandb", "yaml"}


def has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


if not REQUIRED:
    print("لا توجد مكتبات خارجية مطلوبة؛ يكفي Python القياسي.")
else:
    missing = [(module, package) for module, package in REQUIRED if not has_module(module)]
    auto_specs = []
    manual_specs = []
    for module, package in missing:
        specs = package.split()
        if module in AUTO_INSTALL:
            auto_specs.extend(specs)
        else:
            manual_specs.extend(specs)
    auto_specs = list(dict.fromkeys(auto_specs))
    manual_specs = list(dict.fromkeys(manual_specs))

    if auto_specs:
        print("⏳ تثبيت تلقائي:", " ".join(auto_specs))
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *auto_specs])
            print("✅ تم التثبيت.")
        except Exception as exc:
            print("⚠️ تعذّر التثبيت التلقائي:", exc)
            print("شغّل يدويًا: %pip install", " ".join(auto_specs))
    if manual_specs:
        print("⚠️ حزم تحتاج تثبيتًا يدويًا:", " ".join(manual_specs))
        print("شغّل: %pip install", " ".join(manual_specs))
    if not auto_specs and not manual_specs:
        print("✅ كل الحزم المطلوبة متاحة.")

    print("\nالحالة النهائية:")
    for module, package in REQUIRED:
        print(("✅" if has_module(module) else "⚠️"), package)

# ────────────────────────────────────────────────────────────────────────
# [002] content/notebooks/part-05-building-llms/ch19-custom-tokenizer.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import unicodedata

def normalize_arabic_text_nfc(text: str) -> str:
    """
    تطبيع NFC: يحوّل كل التمثيلات البديلة لنفس الحرف المرئي
    إلى تمثيل Unicode قانوني (Canonical) موحّد.
    هذه الخطوة ضرورية قبل تدريب أي Tokenizer على نصوص عربية
    لتجنب معاملة نفس الحرف كرمزين مختلفين في المفردات.
    """
    return unicodedata.normalize("NFC", text)

def demonstrate_normalization_problem():
    """توضيح عملي لمشكلة عدم التطبيع"""
    # نفس الكلمة "أحمد" لكن بتمثيلين مختلفين بايتيًا
    word_precomposed = "\u0623\u062d\u0645\u062f"  # أحمد (الألف مع الهمزة مُركَّبة)
    word_decomposed = "\u0627\u0654\u062d\u0645\u062f"  # أحمد (الألف ثم الهمزة منفصلة)

    print(f"الشكل المرئي للكلمة الأولى: {word_precomposed}")
    print(f"الشكل المرئي للكلمة الثانية: {word_decomposed}")
    print(f"هل هما متطابقتان بايتيًا قبل التطبيع؟ "
          f"{word_precomposed == word_decomposed}")  # False!

    normalized_1 = normalize_arabic_text_nfc(word_precomposed)
    normalized_2 = normalize_arabic_text_nfc(word_decomposed)
    print(f"هل هما متطابقتان بعد تطبيع NFC؟ "
          f"{normalized_1 == normalized_2}")  # True

demonstrate_normalization_problem()

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-05-building-llms/ch19-custom-tokenizer.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import re

class ArabicTextNormalizer:
    """
    خط أنابيب تطبيع شامل للنصوص العربية قبل تدريب Tokenizer.
    يجب تطبيق هذه الخطوات بثبات على كل من بيانات تدريب Tokenizer
    وبيانات تدريب النموذج لاحقًا (وإلا سيحدث عدم تطابق).
    """
    def __init__(self,
                 remove_diacritics: bool = False,
                 normalize_alef_forms: bool = False,
                 normalize_numerals: bool = True):
        """
        remove_diacritics: إزالة التشكيل (محل جدل! بعض المشاريع تحتفظ به
            للنصوص الدينية/الشعرية، لكن معظم LLMs العامة تزيله لتقليل
            حجم المفردات وزيادة الكفاءة).
        normalize_alef_forms: توحيد أ/إ/آ إلى ا (يُفقِد معلومة نحوية لكن
            يقلل تشتت المفردات؛ قرار هندسي يعتمد على حالة الاستخدام).
        normalize_numerals: توحيد الأرقام العربية-الهندية (٠١٢)
            إلى أرقام عربية-غربية (012) لتوحيد التمثيل.
        """
        self.remove_diacritics = remove_diacritics
        self.normalize_alef_forms = normalize_alef_forms
        self.normalize_numerals = normalize_numerals

        self.diacritics_pattern = re.compile(r'[\u064B-\u0652\u0670\u0640]')
        self.alef_forms_pattern = re.compile(r'[إأآا]')
        self.eastern_to_western_digits = str.maketrans(
            "٠١٢٣٤٥٦٧٨٩", "0123456789"
        )

    def normalize(self, text: str) -> str:
        # الخطوة 1: تطبيع NFC أولاً وقبل كل شيء (إلزامي دائمًا)
        text = unicodedata.normalize("NFC", text)

        # الخطوة 2: إزالة التشكيل (اختياري)
        if self.remove_diacritics:
            text = self.diacritics_pattern.sub('', text)

        # الخطوة 3: توحيد أشكال الألف (اختياري، قرار هندسي)
        if self.normalize_alef_forms:
            text = self.alef_forms_pattern.sub('ا', text)

        # الخطوة 4: توحيد الأرقام
        if self.normalize_numerals:
            text = text.translate(self.eastern_to_western_digits)

        # الخطوة 5: توحيد المسافات
        text = re.sub(r'\s+', ' ', text).strip()
        return text

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-05-building-llms/ch19-custom-tokenizer.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
from tokenizers.normalizers import NFC, Sequence
from tokenizers.processors import TemplateProcessing

def train_bpe_tokenizer_for_arabic(
    corpus_files: list,
    vocab_size: int = 50_000,
    min_frequency: int = 2,
    special_tokens: list = None,
) -> Tokenizer:
    """
    تدريب Tokenizer من نوع Byte-Level BPE مُحسَّن للنصوص العربية،
    بنفس المنهجية المستخدمة في تدريب tokenizers نماذج مثل GPT وLLaMA،
    لكن مع تطبيع NFC مُدمَج في خط الأنابيب لضمان الكفاءة مع العربية.
    """
    special_tokens = special_tokens or [
        "<|endoftext|>",   # نهاية الوثيقة
        "<|pad|>",         # الحشو
        "<|unk|>",         # غير معروف (نادر الاستخدام في BPE البايتي)
        "<|user|>",        # بداية رسالة المستخدم (لمرحلة SFT لاحقًا)
        "<|assistant|>",   # بداية رد المساعد
        "<|system|>",      # رسالة النظام
    ]

    # 1. إنشاء Tokenizer بنموذج BPE خام
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))

    # 2. تطبيع NFC كخطوة أولى إلزامية قبل أي تقطيع
    tokenizer.normalizer = Sequence([NFC()])

    # 3. Pre-tokenization بمستوى البايت (ByteLevel)
    # هذا يضمن عدم وجود رموز "unknown" إطلاقًا، لأن أي نص يمكن تفكيكه
    # لبايتات خام (256 قيمة ممكنة) حتى لو لم يظهر الحرف في بيانات التدريب
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = pre_tokenizers.ByteLevel.decoder if hasattr(
        pre_tokenizers.ByteLevel, 'decoder') else None

    # 4. إعداد المدرِّب (Trainer)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # كل البايتات الـ256
        show_progress=True,
    )

    # 5. التدريب الفعلي من ملفات الكوربص
    tokenizer.train(files=corpus_files, trainer=trainer)

    # 6. إضافة معالج القالب (Template) لإضافة رموز البداية/النهاية تلقائيًا
    tokenizer.post_processor = TemplateProcessing(
        single="$A <|endoftext|>",
        special_tokens=[("<|endoftext|>",
                          tokenizer.token_to_id("<|endoftext|>"))],
    )

    return tokenizer

def evaluate_tokenizer_fertility(tokenizer: Tokenizer, test_texts: list) -> dict:
    """
    قياس Fertility: متوسط عدد الـ Tokens لكل كلمة عبر مجموعة اختبار.
    نحسبها أيضًا بمقارنة Tokenizer جديد بـ Tokenizer مرجعي (مثل GPT-2 الأصلي)
    لتوضيح الفارق الكمي في الكفاءة.
    """
    total_words = 0
    total_tokens = 0
    total_chars = 0

    for text in test_texts:
        words = text.split()
        encoded = tokenizer.encode(text)
        total_words += len(words)
        total_tokens += len(encoded.ids)
        total_chars += len(text)

    fertility = total_tokens / max(total_words, 1)
    compression_ratio = total_chars / max(total_tokens, 1)  # أحرف لكل رمز

    return {
        "total_words": total_words,
        "total_tokens": total_tokens,
        "fertility": fertility,                 # كلما قلّ كان أفضل
        "chars_per_token": compression_ratio,    # كلما زاد كان أفضل (ضغط أعلى)
    }

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-05-building-llms/ch19-custom-tokenizer.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def calculate_embedding_parameters_tradeoff(d_model: int,
                                               vocab_sizes: list,
                                               weight_tying: bool = True):
    """
    حساب كلفة معاملات Embedding/Output Head عند أحجام مفردات مختلفة،
    لمساعدة اتخاذ القرار الهندسي عند تصميم Tokenizer.
    """
    print(f"{'حجم المفردات':<15}{'معاملات Embedding':<22}{'معاملات Output':<20}{'الإجمالي':<15}")
    print("-" * 72)
    for vocab_size in vocab_sizes:
        embedding_params = vocab_size * d_model
        # إذا استخدمنا Weight Tying (ربط أوزان Embedding وOutput Head،
        # كما في GPT-2 وLLaMA)، لا نحسب معاملات Output بشكل منفصل
        output_params = 0 if weight_tying else vocab_size * d_model
        total = embedding_params + output_params
        print(f"{vocab_size:<15,}{embedding_params:<22,}{output_params:<20,}{total:<15,}")

calculate_embedding_parameters_tradeoff(
    d_model=4096,
    vocab_sizes=[32_000, 50_000, 100_000, 128_000],
    weight_tying=True
)

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-05-building-llms/ch19-custom-tokenizer.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def compare_tokenizer_fertility_demo():
    """
    محاكاة مبسطة لتوضيح فجوة Fertility بين tokenizer عام وtokenizer عربي مخصص
    (الأرقام الفعلية تعتمد على بيانات تدريب كل tokenizer، لكن الاتجاه حقيقي
    ومُوثَّق في أدبيات المجال، مثل تقرير Jais وتقرير AraGPT2).
    """
    sentence = "تساهم نماذج المحوّلات في تطوير تطبيقات معالجة اللغات الطبيعية بشكل كبير"
    num_words = len(sentence.split())

    # قيم تقديرية واقعية من الأدبيات المنشورة لمقارنات مشابهة
    scenarios = {
        "GPT-2 Tokenizer (إنجليزي أساسًا, 50K مفردات)": 52,
        "GPT-4 Tokenizer (cl100k_base, متعدد اللغات, 100K)": 28,
        "Tokenizer عربي مخصص (50K مفردات مُدرَّب على كوربص عربي)": 13,
    }

    print(f"الجملة: {sentence}")
    print(f"عدد الكلمات: {num_words}\n")
    print(f"{'Tokenizer':<55}{'عدد Tokens':<12}{'Fertility':<10}")
    print("-" * 77)
    for name, num_tokens in scenarios.items():
        fertility = num_tokens / num_words
        print(f"{name:<55}{num_tokens:<12}{fertility:<10.2f}")

compare_tokenizer_fertility_demo()

# ────────────────────────────────────────────────────────────────────────
# [007] content/notebooks/part-05-building-llms/ch19-custom-tokenizer.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-19-ch19-custom-tokenizer.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# === إعداد البيئة ===
# يفحص هذا القسم الحزم المطلوبة ويحاول تثبيت الناقص منها تلقائيًا.
# شغّل هذه الخلية أولًا، ثم بقية خلايا الدفتر.
import importlib.util
import subprocess
import sys

print("Python:", sys.version.split()[0])

# (اسم وحدة الاستيراد، اسم حزمة pip)
REQUIRED = [
    ("tokenizers", "tokenizers")
]
AUTO_INSTALL = {"PIL", "bs4", "datasets", "faiss", "fasttext", "librosa", "networkx", "opentelemetry", "peft", "prometheus_client", "pydantic", "rank_bm25", "requests", "sentence_transformers", "sentencepiece", "soundfile", "tokenizers", "trafilatura", "transformers", "wandb", "yaml"}


def has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


if not REQUIRED:
    print("لا توجد مكتبات خارجية مطلوبة؛ يكفي Python القياسي.")
else:
    missing = [(module, package) for module, package in REQUIRED if not has_module(module)]
    auto_specs = []
    manual_specs = []
    for module, package in missing:
        specs = package.split()
        if module in AUTO_INSTALL:
            auto_specs.extend(specs)
        else:
            manual_specs.extend(specs)
    auto_specs = list(dict.fromkeys(auto_specs))
    manual_specs = list(dict.fromkeys(manual_specs))

    if auto_specs:
        print("⏳ تثبيت تلقائي:", " ".join(auto_specs))
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *auto_specs])
            print("✅ تم التثبيت.")
        except Exception as exc:
            print("⚠️ تعذّر التثبيت التلقائي:", exc)
            print("شغّل يدويًا: %pip install", " ".join(auto_specs))
    if manual_specs:
        print("⚠️ حزم تحتاج تثبيتًا يدويًا:", " ".join(manual_specs))
        print("شغّل: %pip install", " ".join(manual_specs))
    if not auto_specs and not manual_specs:
        print("✅ كل الحزم المطلوبة متاحة.")

    print("\nالحالة النهائية:")
    for module, package in REQUIRED:
        print(("✅" if has_module(module) else "⚠️"), package)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-19-ch19-custom-tokenizer.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import unicodedata

def normalize_arabic_text_nfc(text: str) -> str:
    """
    تطبيع NFC: يحوّل كل التمثيلات البديلة لنفس الحرف المرئي
    إلى تمثيل Unicode قانوني (Canonical) موحّد.
    هذه الخطوة ضرورية قبل تدريب أي Tokenizer على نصوص عربية
    لتجنب معاملة نفس الحرف كرمزين مختلفين في المفردات.
    """
    return unicodedata.normalize("NFC", text)

def demonstrate_normalization_problem():
    """توضيح عملي لمشكلة عدم التطبيع"""
    # نفس الكلمة "أحمد" لكن بتمثيلين مختلفين بايتيًا
    word_precomposed = "\u0623\u062d\u0645\u062f"  # أحمد (الألف مع الهمزة مُركَّبة)
    word_decomposed = "\u0627\u0654\u062d\u0645\u062f"  # أحمد (الألف ثم الهمزة منفصلة)

    print(f"الشكل المرئي للكلمة الأولى: {word_precomposed}")
    print(f"الشكل المرئي للكلمة الثانية: {word_decomposed}")
    print(f"هل هما متطابقتان بايتيًا قبل التطبيع؟ "
          f"{word_precomposed == word_decomposed}")  # False!

    normalized_1 = normalize_arabic_text_nfc(word_precomposed)
    normalized_2 = normalize_arabic_text_nfc(word_decomposed)
    print(f"هل هما متطابقتان بعد تطبيع NFC؟ "
          f"{normalized_1 == normalized_2}")  # True

demonstrate_normalization_problem()

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-19-ch19-custom-tokenizer.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import re

class ArabicTextNormalizer:
    """
    خط أنابيب تطبيع شامل للنصوص العربية قبل تدريب Tokenizer.
    يجب تطبيق هذه الخطوات بثبات على كل من بيانات تدريب Tokenizer
    وبيانات تدريب النموذج لاحقًا (وإلا سيحدث عدم تطابق).
    """
    def __init__(self,
                 remove_diacritics: bool = False,
                 normalize_alef_forms: bool = False,
                 normalize_numerals: bool = True):
        """
        remove_diacritics: إزالة التشكيل (محل جدل! بعض المشاريع تحتفظ به
            للنصوص الدينية/الشعرية، لكن معظم LLMs العامة تزيله لتقليل
            حجم المفردات وزيادة الكفاءة).
        normalize_alef_forms: توحيد أ/إ/آ إلى ا (يُفقِد معلومة نحوية لكن
            يقلل تشتت المفردات؛ قرار هندسي يعتمد على حالة الاستخدام).
        normalize_numerals: توحيد الأرقام العربية-الهندية (٠١٢)
            إلى أرقام عربية-غربية (012) لتوحيد التمثيل.
        """
        self.remove_diacritics = remove_diacritics
        self.normalize_alef_forms = normalize_alef_forms
        self.normalize_numerals = normalize_numerals

        self.diacritics_pattern = re.compile(r'[\u064B-\u0652\u0670\u0640]')
        self.alef_forms_pattern = re.compile(r'[إأآا]')
        self.eastern_to_western_digits = str.maketrans(
            "٠١٢٣٤٥٦٧٨٩", "0123456789"
        )

    def normalize(self, text: str) -> str:
        # الخطوة 1: تطبيع NFC أولاً وقبل كل شيء (إلزامي دائمًا)
        text = unicodedata.normalize("NFC", text)

        # الخطوة 2: إزالة التشكيل (اختياري)
        if self.remove_diacritics:
            text = self.diacritics_pattern.sub('', text)

        # الخطوة 3: توحيد أشكال الألف (اختياري، قرار هندسي)
        if self.normalize_alef_forms:
            text = self.alef_forms_pattern.sub('ا', text)

        # الخطوة 4: توحيد الأرقام
        if self.normalize_numerals:
            text = text.translate(self.eastern_to_western_digits)

        # الخطوة 5: توحيد المسافات
        text = re.sub(r'\s+', ' ', text).strip()
        return text

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-19-ch19-custom-tokenizer.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
from tokenizers.normalizers import NFC, Sequence
from tokenizers.processors import TemplateProcessing

def train_bpe_tokenizer_for_arabic(
    corpus_files: list,
    vocab_size: int = 50_000,
    min_frequency: int = 2,
    special_tokens: list = None,
) -> Tokenizer:
    """
    تدريب Tokenizer من نوع Byte-Level BPE مُحسَّن للنصوص العربية،
    بنفس المنهجية المستخدمة في تدريب tokenizers نماذج مثل GPT وLLaMA،
    لكن مع تطبيع NFC مُدمَج في خط الأنابيب لضمان الكفاءة مع العربية.
    """
    special_tokens = special_tokens or [
        "<|endoftext|>",   # نهاية الوثيقة
        "<|pad|>",         # الحشو
        "<|unk|>",         # غير معروف (نادر الاستخدام في BPE البايتي)
        "<|user|>",        # بداية رسالة المستخدم (لمرحلة SFT لاحقًا)
        "<|assistant|>",   # بداية رد المساعد
        "<|system|>",      # رسالة النظام
    ]

    # 1. إنشاء Tokenizer بنموذج BPE خام
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))

    # 2. تطبيع NFC كخطوة أولى إلزامية قبل أي تقطيع
    tokenizer.normalizer = Sequence([NFC()])

    # 3. Pre-tokenization بمستوى البايت (ByteLevel)
    # هذا يضمن عدم وجود رموز "unknown" إطلاقًا، لأن أي نص يمكن تفكيكه
    # لبايتات خام (256 قيمة ممكنة) حتى لو لم يظهر الحرف في بيانات التدريب
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = pre_tokenizers.ByteLevel.decoder if hasattr(
        pre_tokenizers.ByteLevel, 'decoder') else None

    # 4. إعداد المدرِّب (Trainer)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # كل البايتات الـ256
        show_progress=True,
    )

    # 5. التدريب الفعلي من ملفات الكوربص
    tokenizer.train(files=corpus_files, trainer=trainer)

    # 6. إضافة معالج القالب (Template) لإضافة رموز البداية/النهاية تلقائيًا
    tokenizer.post_processor = TemplateProcessing(
        single="$A <|endoftext|>",
        special_tokens=[("<|endoftext|>",
                          tokenizer.token_to_id("<|endoftext|>"))],
    )

    return tokenizer

def evaluate_tokenizer_fertility(tokenizer: Tokenizer, test_texts: list) -> dict:
    """
    قياس Fertility: متوسط عدد الـ Tokens لكل كلمة عبر مجموعة اختبار.
    نحسبها أيضًا بمقارنة Tokenizer جديد بـ Tokenizer مرجعي (مثل GPT-2 الأصلي)
    لتوضيح الفارق الكمي في الكفاءة.
    """
    total_words = 0
    total_tokens = 0
    total_chars = 0

    for text in test_texts:
        words = text.split()
        encoded = tokenizer.encode(text)
        total_words += len(words)
        total_tokens += len(encoded.ids)
        total_chars += len(text)

    fertility = total_tokens / max(total_words, 1)
    compression_ratio = total_chars / max(total_tokens, 1)  # أحرف لكل رمز

    return {
        "total_words": total_words,
        "total_tokens": total_tokens,
        "fertility": fertility,                 # كلما قلّ كان أفضل
        "chars_per_token": compression_ratio,    # كلما زاد كان أفضل (ضغط أعلى)
    }

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-19-ch19-custom-tokenizer.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def calculate_embedding_parameters_tradeoff(d_model: int,
                                               vocab_sizes: list,
                                               weight_tying: bool = True):
    """
    حساب كلفة معاملات Embedding/Output Head عند أحجام مفردات مختلفة،
    لمساعدة اتخاذ القرار الهندسي عند تصميم Tokenizer.
    """
    print(f"{'حجم المفردات':<15}{'معاملات Embedding':<22}{'معاملات Output':<20}{'الإجمالي':<15}")
    print("-" * 72)
    for vocab_size in vocab_sizes:
        embedding_params = vocab_size * d_model
        # إذا استخدمنا Weight Tying (ربط أوزان Embedding وOutput Head،
        # كما في GPT-2 وLLaMA)، لا نحسب معاملات Output بشكل منفصل
        output_params = 0 if weight_tying else vocab_size * d_model
        total = embedding_params + output_params
        print(f"{vocab_size:<15,}{embedding_params:<22,}{output_params:<20,}{total:<15,}")

calculate_embedding_parameters_tradeoff(
    d_model=4096,
    vocab_sizes=[32_000, 50_000, 100_000, 128_000],
    weight_tying=True
)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-llms-transformers/notebooks/ch-19-ch19-custom-tokenizer.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def compare_tokenizer_fertility_demo():
    """
    محاكاة مبسطة لتوضيح فجوة Fertility بين tokenizer عام وtokenizer عربي مخصص
    (الأرقام الفعلية تعتمد على بيانات تدريب كل tokenizer، لكن الاتجاه حقيقي
    ومُوثَّق في أدبيات المجال، مثل تقرير Jais وتقرير AraGPT2).
    """
    sentence = "تساهم نماذج المحوّلات في تطوير تطبيقات معالجة اللغات الطبيعية بشكل كبير"
    num_words = len(sentence.split())

    # قيم تقديرية واقعية من الأدبيات المنشورة لمقارنات مشابهة
    scenarios = {
        "GPT-2 Tokenizer (إنجليزي أساسًا, 50K مفردات)": 52,
        "GPT-4 Tokenizer (cl100k_base, متعدد اللغات, 100K)": 28,
        "Tokenizer عربي مخصص (50K مفردات مُدرَّب على كوربص عربي)": 13,
    }

    print(f"الجملة: {sentence}")
    print(f"عدد الكلمات: {num_words}\n")
    print(f"{'Tokenizer':<55}{'عدد Tokens':<12}{'Fertility':<10}")
    print("-" * 77)
    for name, num_tokens in scenarios.items():
        fertility = num_tokens / num_words
        print(f"{name:<55}{num_tokens:<12}{fertility:<10.2f}")

compare_tokenizer_fertility_demo()

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-llms-transformers/notebooks/ch-19-ch19-custom-tokenizer.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
