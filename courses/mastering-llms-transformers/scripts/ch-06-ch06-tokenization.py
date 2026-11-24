#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — التقطيع إلى وحدات: Tokenization
Mastering Transformers and Large Language Models — Tokenization

Book slug : mastering-llms-transformers
Chapter   : ch-06-ch06-tokenization (lesson 6)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-06-ch06-tokenization.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — التقطيع إلى وحدات: Tokenization
Mastering Transformers and Large Language Models — Tokenization

Book slug : mastering-llms-transformers
Chapter   : ch-06-ch06-tokenization (lesson 6)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-06-ch06-tokenization.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — التقطيع إلى وحدات: Tokenization
Mastering Transformers and Large Language Models — Tokenization

Book slug : mastering-llms-transformers
Chapter   : ch-06-ch06-tokenization (lesson 6)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-02-language-representation/ch06-tokenization.ipynb#cell-1
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
    ("sentencepiece", "sentencepiece")
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
# [002] content/notebooks/part-02-language-representation/ch06-tokenization.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import re
from collections import defaultdict, Counter

class BPETokenizerFromScratch:
    """
    تطبيق كامل لخوارزمية Byte-Pair Encoding (Sennrich et al., 2016)
    من الصفر، بدون أي مكتبة جاهزة. هذا هو جوهر Tokenizer في
    GPT-2/GPT-3/GPT-4 وLlama (مع بعض التحسينات الهندسية الإضافية).
    """
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.merges = {}       # قاموس الدمجات المُتعلَّمة: (رمز1, رمز2) -> رمز مُدمَج
        self.vocab = {}        # قاموس المفردات النهائي: رمز -> معرّف رقمي

    def _get_word_frequencies(self, corpus: list) -> dict:
        """
        حساب تكرار كل كلمة في المجموعة النصية، مع تمثيل كل كلمة
        كتسلسل أحرف مُنفصلة + رمز نهاية الكلمة </w> (ضروري لتمييز
        "نهاية الكلمة" عن استمرارها، فـ "كتاب" و"كتابة" تشتركان
        في البادئة لكن يجب تمييز حدود الكلمة الفعلية).
        """
        word_freqs = defaultdict(int)
        for text in corpus:
            words = text.strip().split()
            for word in words:
                # تمثيل الكلمة كتسلسل أحرف منفصلة بمسافات + رمز نهاية
                char_sequence = " ".join(list(word)) + " </w>"
                word_freqs[char_sequence] += 1
        return word_freqs

    def _get_pair_frequencies(self, word_freqs: dict) -> dict:
        """حساب تكرار كل زوج أحرف/رموز متجاورة عبر كل الكلمات."""
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    def _merge_pair(self, pair: tuple, word_freqs: dict) -> dict:
        """
        دمج زوج رموز مُحدَّد في كل الكلمات التي يظهر فيها.
        مثال: دمج ('ك','ت') يُحوِّل "ك ت ا ب" إلى "كت ا ب"
        """
        new_word_freqs = {}
        bigram = re.escape(" ".join(pair))
        pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")

        for word, freq in word_freqs.items():
            new_word = pattern.sub("".join(pair), word)
            new_word_freqs[new_word] = freq
        return new_word_freqs

    def train(self, corpus: list, verbose: bool = True):
        """
        حلقة التدريب الرئيسية: نُكرِّر "إيجاد أكثر زوج تكرارًا ودمجه"
        حتى نصل لحجم المفردات المطلوب أو لا تبقى أزواج قابلة للدمج.
        """
        word_freqs = self._get_word_frequencies(corpus)

        # المفردات الابتدائية: كل حرف فريد + رمز نهاية الكلمة
        initial_chars = set()
        for word in word_freqs:
            initial_chars.update(word.split())
        self.vocab = {char: idx for idx, char in enumerate(sorted(initial_chars))}

        num_merges = self.vocab_size - len(self.vocab)
        for merge_step in range(num_merges):
            pair_freqs = self._get_pair_frequencies(word_freqs)
            if not pair_freqs:
                break  # لا توجد أزواج أخرى للدمج

            best_pair = max(pair_freqs, key=pair_freqs.get)
            word_freqs = self._merge_pair(best_pair, word_freqs)

            merged_token = "".join(best_pair)
            self.merges[best_pair] = merged_token
            self.vocab[merged_token] = len(self.vocab)

            if verbose and merge_step % 50 == 0:
                print(f"الدمج {merge_step}: {best_pair} → '{merged_token}' "
                      f"(تكرار={pair_freqs[best_pair]})")

        if verbose:
            print(f"\n✅ اكتمل التدريب: {len(self.vocab)} رمز في المفردات")

    def tokenize(self, text: str) -> list:
        """تقطيع نص جديد باستخدام الدمجات المُتعلَّمة، بنفس الترتيب الذي تعلّمناها به."""
        words = text.strip().split()
        tokens = []

        for word in words:
            symbols = list(word) + ["</w>"]
            # تطبيق الدمجات المُتعلَّمة تكراريًا بنفس ترتيب الأولوية
            while len(symbols) > 1:
                pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
                # إيجاد الدمج ذي الأولوية الأعلى (الذي تعلّمناه أولًا)
                mergeable = [p for p in pairs if p in self.merges]
                if not mergeable:
                    break
                # نختار أول دمج تعلّمناه (الأقدم زمنيًا = الأكثر عمومية/تكرارًا)
                pair_to_merge = min(mergeable, key=lambda p: list(self.merges.keys()).index(p))
                idx = pairs.index(pair_to_merge)
                symbols = (symbols[:idx] + [self.merges[pair_to_merge]] +
                           symbols[idx + 2:])
            tokens.extend(symbols)

        return tokens

    def encode(self, text: str) -> list:
        """تحويل النص إلى تسلسل معرّفات رقمية (Token IDs)."""
        tokens = self.tokenize(text)
        return [self.vocab.get(tok, self.vocab.get("<unk>", -1)) for tok in tokens]

# === تجربة عملية على نصوص عربية ===
arabic_corpus = [
    "الكتاب على الطاولة",
    "الكتابة مهارة مهمة",
    "كتب الطالب الدرس",
    "يكتب المعلم على السبورة",
    "الكاتب نشر كتابا جديدا",
] * 20  # تكرار لمحاكاة بيانات أكبر

tokenizer = BPETokenizerFromScratch(vocab_size=120)
tokenizer.train(arabic_corpus, verbose=True)

test_text = "الكاتب يكتب كتابا"
tokens = tokenizer.tokenize(test_text)
print(f"\nالنص: {test_text}")
print(f"الرموز الناتجة: {tokens}")
# الرموز الناتجة: ['الكاتب</w>', 'يكتب</w>', 'كتاب', 'ا</w>']
# لاحظ كيف أن الجذر المشترك "كتب/كتاب" يظهر كوحدة متكررة عبر عدة كلمات

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-02-language-representation/ch06-tokenization.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def explain_wordpiece_difference():
    """
    WordPiece (مُستخدَمة في BERT): تشبه BPE لكن معيار اختيار الزوج
    للدمج مختلف جوهريًا. بدلًا من "أكثر زوج تكرارًا" كما في BPE،
    تختار WordPiece الزوج الذي يُعظِّم احتمالية المجموعة التدريبية
    عند دمجه (Likelihood-based)، أي:

        score(pair) = freq(pair) / (freq(first) × freq(second))

    هذا يُفضِّل دمج الأزواج النادرة التي تظهر معًا بشكل "غير متوقع"
    (أعلى من الصدفة الإحصائية)، وليس فقط الأكثر تكرارًا بشكل مطلق.
    """
    # محاكاة الفرق: زوج ('ال', 'كتاب') قد يكون أقل تكرارًا من ('ت','ا')
    # لكن احتمالية ظهورهما معًا أعلى بكثير من المتوقع عشوائيًا
    pair_freq = 50
    first_freq, second_freq = 200, 180

    bpe_score = pair_freq  # BPE يهتم فقط بالتكرار المطلق
    wordpiece_score = pair_freq / (first_freq * second_freq)

    print(f"BPE score (تكرار مطلق): {bpe_score}")
    print(f"WordPiece score (احتمالية مُطبَّعة): {wordpiece_score:.6f}")

def explain_unigram_sentencepiece():
    """
    SentencePiece/Unigram (T5, Llama, XLNet): نهج معاكس تمامًا!
    بدلًا من البدء بأحرف والدمج تصاعديًا (Bottom-up كما في BPE/WordPiece)،
    نبدأ بمفردات كبيرة جدًا (كل الكلمات الفرعية المحتملة تقريبًا)
    ثم نُقلِّص تدريجيًا (Top-down) عبر إزالة الرموز التي تُقلِّل
    إزالتها أقل احتمالية ممكنة من إجمالي احتمالية المجموعة التدريبية
    (نموذج احتمالي Unigram Language Model كامل).

    الميزة الحاسمة: SentencePiece يُعامل النص كـ"تيار بايتات خام"
    (raw byte stream) بما في ذلك المسافات (تُمثَّل برمز خاص ▁)،
    مما يجعله مستقلًا تمامًا عن قواعد التقطيع بلغة معينة —
    ميزة بالغة الأهمية للنماذج متعددة اللغات (بما فيها العربية
    التي لا تحتاج فصل كلمات بالطريقة اللاتينية نفسها دائمًا).
    """
    example = "الكتاب جميل"
    sentencepiece_style = "▁الكتاب▁جميل"  # ▁ يُمثِّل المسافة كرمز صريح
    print(f"تمثيل SentencePiece: {sentencepiece_style}")
    print("هذا يسمح بإعادة بناء النص الأصلي بدقة تامة دون أي قواعد")
    print("خاصة باللغة لتحديد أين تقع المسافات (reversible tokenization)")

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-02-language-representation/ch06-tokenization.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import sentencepiece as spm

def train_sentencepiece_for_arabic(corpus_file: str, model_prefix: str = "arabic_tokenizer"):
    """
    التدريب الفعلي لـ SentencePiece على بيانات عربية، باستخدام
    إعدادات مُحسَّنة خصيصًا للتعامل مع خصائص العربية (التشكيل،
    الحروف المُركَّبة، النطاق الواسع من الرموز اليونيكود).
    """
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=model_prefix,
        vocab_size=32000,
        model_type="unigram",        # أو "bpe" حسب التفضيل

        # إعداد حرج جدًا للعربية: character_coverage يُحدِّد نسبة
        # الأحرف التي يجب أن يُغطّيها النموذج. للعربية (ولغات أخرى
        # غنية بالرموز مثل التشكيل والحروف الخاصة)، القيمة الافتراضية
        # 0.9995 (بدلًا من 1.0) تُهمل الأحرف النادرة جدًا (أخطاء إملائية،
        # رموز شاذة) لتجنّب تضخيم المفردات بأحرف غير مفيدة عمليًا.
        character_coverage=0.9995,

        normalization_rule_name="nfkc",  # تطبيع يونيكود موحّد (مهم جدًا للعربية
                                           # بسبب تعدد أشكال نفس الحرف Unicode)
        byte_fallback=True,   # أي رمز غير مُغطّى يُرمَّز كبايتات خام (لا OOV إطلاقًا!)

        # رموز خاصة ضرورية لأي LLM
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        user_defined_symbols=["<|system|>", "<|user|>", "<|assistant|>"],
    )
    print(f"✅ تم تدريب SentencePiece Tokenizer وحفظه بـ {model_prefix}.model")

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-02-language-representation/ch06-tokenization.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def manual_bpe_walkthrough():
    corpus = {
        "l o w </w>": 5,
        "l o w e r </w>": 2,
        "l o w e s t </w>": 6,
        "n e w e s t </w>": 3,
        "w i d e s t </w>": 2,
    }

    for step in range(3):
        pair_freqs = defaultdict(int)
        for word, freq in corpus.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair_freqs[(symbols[i], symbols[i+1])] += freq

        best_pair = max(pair_freqs, key=pair_freqs.get)
        print(f"الخطوة {step+1}: أكثر زوج تكرارًا = {best_pair} "
              f"(تكرار={pair_freqs[best_pair]})")

        new_corpus = {}
        merged = "".join(best_pair)
        for word, freq in corpus.items():
            new_word = word.replace(" ".join(best_pair), merged)
            new_corpus[new_word] = freq
        corpus = new_corpus
        print(f"   الكلمات بعد الدمج: {list(corpus.keys())}\n")

manual_bpe_walkthrough()
# الخطوة 1: أكثر زوج تكرارًا = ('e', 's') (تكرار=9)
#    الكلمات بعد الدمج: ['l o w </w>', 'l o w e r </w>', 'l o w es t </w>', ...]
# الخطوة 2: أكثر زوج تكرارًا = ('es', 't') (تكرار=9)
#    الكلمات بعد الدمج: [..., 'l o w est </w>', 'n e w est </w>', ...]
# الخطوة 3: أكثر زوج تكرارًا = ('l', 'o') (تكرار=13)
#    الكلمات بعد الدمج: ['lo w </w>', 'lo w e r </w>', 'lo w est </w>', ...]

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-02-language-representation/ch06-tokenization.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-06-ch06-tokenization.ipynb#cell-1
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
    ("sentencepiece", "sentencepiece")
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-06-ch06-tokenization.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import re
from collections import defaultdict, Counter

class BPETokenizerFromScratch:
    """
    تطبيق كامل لخوارزمية Byte-Pair Encoding (Sennrich et al., 2016)
    من الصفر، بدون أي مكتبة جاهزة. هذا هو جوهر Tokenizer في
    GPT-2/GPT-3/GPT-4 وLlama (مع بعض التحسينات الهندسية الإضافية).
    """
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.merges = {}       # قاموس الدمجات المُتعلَّمة: (رمز1, رمز2) -> رمز مُدمَج
        self.vocab = {}        # قاموس المفردات النهائي: رمز -> معرّف رقمي

    def _get_word_frequencies(self, corpus: list) -> dict:
        """
        حساب تكرار كل كلمة في المجموعة النصية، مع تمثيل كل كلمة
        كتسلسل أحرف مُنفصلة + رمز نهاية الكلمة </w> (ضروري لتمييز
        "نهاية الكلمة" عن استمرارها، فـ "كتاب" و"كتابة" تشتركان
        في البادئة لكن يجب تمييز حدود الكلمة الفعلية).
        """
        word_freqs = defaultdict(int)
        for text in corpus:
            words = text.strip().split()
            for word in words:
                # تمثيل الكلمة كتسلسل أحرف منفصلة بمسافات + رمز نهاية
                char_sequence = " ".join(list(word)) + " </w>"
                word_freqs[char_sequence] += 1
        return word_freqs

    def _get_pair_frequencies(self, word_freqs: dict) -> dict:
        """حساب تكرار كل زوج أحرف/رموز متجاورة عبر كل الكلمات."""
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    def _merge_pair(self, pair: tuple, word_freqs: dict) -> dict:
        """
        دمج زوج رموز مُحدَّد في كل الكلمات التي يظهر فيها.
        مثال: دمج ('ك','ت') يُحوِّل "ك ت ا ب" إلى "كت ا ب"
        """
        new_word_freqs = {}
        bigram = re.escape(" ".join(pair))
        pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")

        for word, freq in word_freqs.items():
            new_word = pattern.sub("".join(pair), word)
            new_word_freqs[new_word] = freq
        return new_word_freqs

    def train(self, corpus: list, verbose: bool = True):
        """
        حلقة التدريب الرئيسية: نُكرِّر "إيجاد أكثر زوج تكرارًا ودمجه"
        حتى نصل لحجم المفردات المطلوب أو لا تبقى أزواج قابلة للدمج.
        """
        word_freqs = self._get_word_frequencies(corpus)

        # المفردات الابتدائية: كل حرف فريد + رمز نهاية الكلمة
        initial_chars = set()
        for word in word_freqs:
            initial_chars.update(word.split())
        self.vocab = {char: idx for idx, char in enumerate(sorted(initial_chars))}

        num_merges = self.vocab_size - len(self.vocab)
        for merge_step in range(num_merges):
            pair_freqs = self._get_pair_frequencies(word_freqs)
            if not pair_freqs:
                break  # لا توجد أزواج أخرى للدمج

            best_pair = max(pair_freqs, key=pair_freqs.get)
            word_freqs = self._merge_pair(best_pair, word_freqs)

            merged_token = "".join(best_pair)
            self.merges[best_pair] = merged_token
            self.vocab[merged_token] = len(self.vocab)

            if verbose and merge_step % 50 == 0:
                print(f"الدمج {merge_step}: {best_pair} → '{merged_token}' "
                      f"(تكرار={pair_freqs[best_pair]})")

        if verbose:
            print(f"\n✅ اكتمل التدريب: {len(self.vocab)} رمز في المفردات")

    def tokenize(self, text: str) -> list:
        """تقطيع نص جديد باستخدام الدمجات المُتعلَّمة، بنفس الترتيب الذي تعلّمناها به."""
        words = text.strip().split()
        tokens = []

        for word in words:
            symbols = list(word) + ["</w>"]
            # تطبيق الدمجات المُتعلَّمة تكراريًا بنفس ترتيب الأولوية
            while len(symbols) > 1:
                pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
                # إيجاد الدمج ذي الأولوية الأعلى (الذي تعلّمناه أولًا)
                mergeable = [p for p in pairs if p in self.merges]
                if not mergeable:
                    break
                # نختار أول دمج تعلّمناه (الأقدم زمنيًا = الأكثر عمومية/تكرارًا)
                pair_to_merge = min(mergeable, key=lambda p: list(self.merges.keys()).index(p))
                idx = pairs.index(pair_to_merge)
                symbols = (symbols[:idx] + [self.merges[pair_to_merge]] +
                           symbols[idx + 2:])
            tokens.extend(symbols)

        return tokens

    def encode(self, text: str) -> list:
        """تحويل النص إلى تسلسل معرّفات رقمية (Token IDs)."""
        tokens = self.tokenize(text)
        return [self.vocab.get(tok, self.vocab.get("<unk>", -1)) for tok in tokens]

# === تجربة عملية على نصوص عربية ===
arabic_corpus = [
    "الكتاب على الطاولة",
    "الكتابة مهارة مهمة",
    "كتب الطالب الدرس",
    "يكتب المعلم على السبورة",
    "الكاتب نشر كتابا جديدا",
] * 20  # تكرار لمحاكاة بيانات أكبر

tokenizer = BPETokenizerFromScratch(vocab_size=120)
tokenizer.train(arabic_corpus, verbose=True)

test_text = "الكاتب يكتب كتابا"
tokens = tokenizer.tokenize(test_text)
print(f"\nالنص: {test_text}")
print(f"الرموز الناتجة: {tokens}")
# الرموز الناتجة: ['الكاتب</w>', 'يكتب</w>', 'كتاب', 'ا</w>']
# لاحظ كيف أن الجذر المشترك "كتب/كتاب" يظهر كوحدة متكررة عبر عدة كلمات

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-06-ch06-tokenization.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def explain_wordpiece_difference():
    """
    WordPiece (مُستخدَمة في BERT): تشبه BPE لكن معيار اختيار الزوج
    للدمج مختلف جوهريًا. بدلًا من "أكثر زوج تكرارًا" كما في BPE،
    تختار WordPiece الزوج الذي يُعظِّم احتمالية المجموعة التدريبية
    عند دمجه (Likelihood-based)، أي:

        score(pair) = freq(pair) / (freq(first) × freq(second))

    هذا يُفضِّل دمج الأزواج النادرة التي تظهر معًا بشكل "غير متوقع"
    (أعلى من الصدفة الإحصائية)، وليس فقط الأكثر تكرارًا بشكل مطلق.
    """
    # محاكاة الفرق: زوج ('ال', 'كتاب') قد يكون أقل تكرارًا من ('ت','ا')
    # لكن احتمالية ظهورهما معًا أعلى بكثير من المتوقع عشوائيًا
    pair_freq = 50
    first_freq, second_freq = 200, 180

    bpe_score = pair_freq  # BPE يهتم فقط بالتكرار المطلق
    wordpiece_score = pair_freq / (first_freq * second_freq)

    print(f"BPE score (تكرار مطلق): {bpe_score}")
    print(f"WordPiece score (احتمالية مُطبَّعة): {wordpiece_score:.6f}")

def explain_unigram_sentencepiece():
    """
    SentencePiece/Unigram (T5, Llama, XLNet): نهج معاكس تمامًا!
    بدلًا من البدء بأحرف والدمج تصاعديًا (Bottom-up كما في BPE/WordPiece)،
    نبدأ بمفردات كبيرة جدًا (كل الكلمات الفرعية المحتملة تقريبًا)
    ثم نُقلِّص تدريجيًا (Top-down) عبر إزالة الرموز التي تُقلِّل
    إزالتها أقل احتمالية ممكنة من إجمالي احتمالية المجموعة التدريبية
    (نموذج احتمالي Unigram Language Model كامل).

    الميزة الحاسمة: SentencePiece يُعامل النص كـ"تيار بايتات خام"
    (raw byte stream) بما في ذلك المسافات (تُمثَّل برمز خاص ▁)،
    مما يجعله مستقلًا تمامًا عن قواعد التقطيع بلغة معينة —
    ميزة بالغة الأهمية للنماذج متعددة اللغات (بما فيها العربية
    التي لا تحتاج فصل كلمات بالطريقة اللاتينية نفسها دائمًا).
    """
    example = "الكتاب جميل"
    sentencepiece_style = "▁الكتاب▁جميل"  # ▁ يُمثِّل المسافة كرمز صريح
    print(f"تمثيل SentencePiece: {sentencepiece_style}")
    print("هذا يسمح بإعادة بناء النص الأصلي بدقة تامة دون أي قواعد")
    print("خاصة باللغة لتحديد أين تقع المسافات (reversible tokenization)")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-06-ch06-tokenization.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import sentencepiece as spm

def train_sentencepiece_for_arabic(corpus_file: str, model_prefix: str = "arabic_tokenizer"):
    """
    التدريب الفعلي لـ SentencePiece على بيانات عربية، باستخدام
    إعدادات مُحسَّنة خصيصًا للتعامل مع خصائص العربية (التشكيل،
    الحروف المُركَّبة، النطاق الواسع من الرموز اليونيكود).
    """
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=model_prefix,
        vocab_size=32000,
        model_type="unigram",        # أو "bpe" حسب التفضيل

        # إعداد حرج جدًا للعربية: character_coverage يُحدِّد نسبة
        # الأحرف التي يجب أن يُغطّيها النموذج. للعربية (ولغات أخرى
        # غنية بالرموز مثل التشكيل والحروف الخاصة)، القيمة الافتراضية
        # 0.9995 (بدلًا من 1.0) تُهمل الأحرف النادرة جدًا (أخطاء إملائية،
        # رموز شاذة) لتجنّب تضخيم المفردات بأحرف غير مفيدة عمليًا.
        character_coverage=0.9995,

        normalization_rule_name="nfkc",  # تطبيع يونيكود موحّد (مهم جدًا للعربية
                                           # بسبب تعدد أشكال نفس الحرف Unicode)
        byte_fallback=True,   # أي رمز غير مُغطّى يُرمَّز كبايتات خام (لا OOV إطلاقًا!)

        # رموز خاصة ضرورية لأي LLM
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        user_defined_symbols=["<|system|>", "<|user|>", "<|assistant|>"],
    )
    print(f"✅ تم تدريب SentencePiece Tokenizer وحفظه بـ {model_prefix}.model")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-06-ch06-tokenization.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def manual_bpe_walkthrough():
    corpus = {
        "l o w </w>": 5,
        "l o w e r </w>": 2,
        "l o w e s t </w>": 6,
        "n e w e s t </w>": 3,
        "w i d e s t </w>": 2,
    }

    for step in range(3):
        pair_freqs = defaultdict(int)
        for word, freq in corpus.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair_freqs[(symbols[i], symbols[i+1])] += freq

        best_pair = max(pair_freqs, key=pair_freqs.get)
        print(f"الخطوة {step+1}: أكثر زوج تكرارًا = {best_pair} "
              f"(تكرار={pair_freqs[best_pair]})")

        new_corpus = {}
        merged = "".join(best_pair)
        for word, freq in corpus.items():
            new_word = word.replace(" ".join(best_pair), merged)
            new_corpus[new_word] = freq
        corpus = new_corpus
        print(f"   الكلمات بعد الدمج: {list(corpus.keys())}\n")

manual_bpe_walkthrough()
# الخطوة 1: أكثر زوج تكرارًا = ('e', 's') (تكرار=9)
#    الكلمات بعد الدمج: ['l o w </w>', 'l o w e r </w>', 'l o w es t </w>', ...]
# الخطوة 2: أكثر زوج تكرارًا = ('es', 't') (تكرار=9)
#    الكلمات بعد الدمج: [..., 'l o w est </w>', 'n e w est </w>', ...]
# الخطوة 3: أكثر زوج تكرارًا = ('l', 'o') (تكرار=13)
#    الكلمات بعد الدمج: ['lo w </w>', 'lo w e r </w>', 'lo w est </w>', ...]

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-06-ch06-tokenization.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-06-ch06-tokenization.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — التقطيع إلى وحدات: Tokenization
Mastering Transformers and Large Language Models — Tokenization

Book slug : mastering-llms-transformers
Chapter   : ch-06-ch06-tokenization (lesson 6)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-02-language-representation/ch06-tokenization.ipynb#cell-1
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
    ("sentencepiece", "sentencepiece")
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
# [002] content/notebooks/part-02-language-representation/ch06-tokenization.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import re
from collections import defaultdict, Counter

class BPETokenizerFromScratch:
    """
    تطبيق كامل لخوارزمية Byte-Pair Encoding (Sennrich et al., 2016)
    من الصفر، بدون أي مكتبة جاهزة. هذا هو جوهر Tokenizer في
    GPT-2/GPT-3/GPT-4 وLlama (مع بعض التحسينات الهندسية الإضافية).
    """
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.merges = {}       # قاموس الدمجات المُتعلَّمة: (رمز1, رمز2) -> رمز مُدمَج
        self.vocab = {}        # قاموس المفردات النهائي: رمز -> معرّف رقمي

    def _get_word_frequencies(self, corpus: list) -> dict:
        """
        حساب تكرار كل كلمة في المجموعة النصية، مع تمثيل كل كلمة
        كتسلسل أحرف مُنفصلة + رمز نهاية الكلمة </w> (ضروري لتمييز
        "نهاية الكلمة" عن استمرارها، فـ "كتاب" و"كتابة" تشتركان
        في البادئة لكن يجب تمييز حدود الكلمة الفعلية).
        """
        word_freqs = defaultdict(int)
        for text in corpus:
            words = text.strip().split()
            for word in words:
                # تمثيل الكلمة كتسلسل أحرف منفصلة بمسافات + رمز نهاية
                char_sequence = " ".join(list(word)) + " </w>"
                word_freqs[char_sequence] += 1
        return word_freqs

    def _get_pair_frequencies(self, word_freqs: dict) -> dict:
        """حساب تكرار كل زوج أحرف/رموز متجاورة عبر كل الكلمات."""
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    def _merge_pair(self, pair: tuple, word_freqs: dict) -> dict:
        """
        دمج زوج رموز مُحدَّد في كل الكلمات التي يظهر فيها.
        مثال: دمج ('ك','ت') يُحوِّل "ك ت ا ب" إلى "كت ا ب"
        """
        new_word_freqs = {}
        bigram = re.escape(" ".join(pair))
        pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")

        for word, freq in word_freqs.items():
            new_word = pattern.sub("".join(pair), word)
            new_word_freqs[new_word] = freq
        return new_word_freqs

    def train(self, corpus: list, verbose: bool = True):
        """
        حلقة التدريب الرئيسية: نُكرِّر "إيجاد أكثر زوج تكرارًا ودمجه"
        حتى نصل لحجم المفردات المطلوب أو لا تبقى أزواج قابلة للدمج.
        """
        word_freqs = self._get_word_frequencies(corpus)

        # المفردات الابتدائية: كل حرف فريد + رمز نهاية الكلمة
        initial_chars = set()
        for word in word_freqs:
            initial_chars.update(word.split())
        self.vocab = {char: idx for idx, char in enumerate(sorted(initial_chars))}

        num_merges = self.vocab_size - len(self.vocab)
        for merge_step in range(num_merges):
            pair_freqs = self._get_pair_frequencies(word_freqs)
            if not pair_freqs:
                break  # لا توجد أزواج أخرى للدمج

            best_pair = max(pair_freqs, key=pair_freqs.get)
            word_freqs = self._merge_pair(best_pair, word_freqs)

            merged_token = "".join(best_pair)
            self.merges[best_pair] = merged_token
            self.vocab[merged_token] = len(self.vocab)

            if verbose and merge_step % 50 == 0:
                print(f"الدمج {merge_step}: {best_pair} → '{merged_token}' "
                      f"(تكرار={pair_freqs[best_pair]})")

        if verbose:
            print(f"\n✅ اكتمل التدريب: {len(self.vocab)} رمز في المفردات")

    def tokenize(self, text: str) -> list:
        """تقطيع نص جديد باستخدام الدمجات المُتعلَّمة، بنفس الترتيب الذي تعلّمناها به."""
        words = text.strip().split()
        tokens = []

        for word in words:
            symbols = list(word) + ["</w>"]
            # تطبيق الدمجات المُتعلَّمة تكراريًا بنفس ترتيب الأولوية
            while len(symbols) > 1:
                pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
                # إيجاد الدمج ذي الأولوية الأعلى (الذي تعلّمناه أولًا)
                mergeable = [p for p in pairs if p in self.merges]
                if not mergeable:
                    break
                # نختار أول دمج تعلّمناه (الأقدم زمنيًا = الأكثر عمومية/تكرارًا)
                pair_to_merge = min(mergeable, key=lambda p: list(self.merges.keys()).index(p))
                idx = pairs.index(pair_to_merge)
                symbols = (symbols[:idx] + [self.merges[pair_to_merge]] +
                           symbols[idx + 2:])
            tokens.extend(symbols)

        return tokens

    def encode(self, text: str) -> list:
        """تحويل النص إلى تسلسل معرّفات رقمية (Token IDs)."""
        tokens = self.tokenize(text)
        return [self.vocab.get(tok, self.vocab.get("<unk>", -1)) for tok in tokens]

# === تجربة عملية على نصوص عربية ===
arabic_corpus = [
    "الكتاب على الطاولة",
    "الكتابة مهارة مهمة",
    "كتب الطالب الدرس",
    "يكتب المعلم على السبورة",
    "الكاتب نشر كتابا جديدا",
] * 20  # تكرار لمحاكاة بيانات أكبر

tokenizer = BPETokenizerFromScratch(vocab_size=120)
tokenizer.train(arabic_corpus, verbose=True)

test_text = "الكاتب يكتب كتابا"
tokens = tokenizer.tokenize(test_text)
print(f"\nالنص: {test_text}")
print(f"الرموز الناتجة: {tokens}")
# الرموز الناتجة: ['الكاتب</w>', 'يكتب</w>', 'كتاب', 'ا</w>']
# لاحظ كيف أن الجذر المشترك "كتب/كتاب" يظهر كوحدة متكررة عبر عدة كلمات

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-02-language-representation/ch06-tokenization.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def explain_wordpiece_difference():
    """
    WordPiece (مُستخدَمة في BERT): تشبه BPE لكن معيار اختيار الزوج
    للدمج مختلف جوهريًا. بدلًا من "أكثر زوج تكرارًا" كما في BPE،
    تختار WordPiece الزوج الذي يُعظِّم احتمالية المجموعة التدريبية
    عند دمجه (Likelihood-based)، أي:

        score(pair) = freq(pair) / (freq(first) × freq(second))

    هذا يُفضِّل دمج الأزواج النادرة التي تظهر معًا بشكل "غير متوقع"
    (أعلى من الصدفة الإحصائية)، وليس فقط الأكثر تكرارًا بشكل مطلق.
    """
    # محاكاة الفرق: زوج ('ال', 'كتاب') قد يكون أقل تكرارًا من ('ت','ا')
    # لكن احتمالية ظهورهما معًا أعلى بكثير من المتوقع عشوائيًا
    pair_freq = 50
    first_freq, second_freq = 200, 180

    bpe_score = pair_freq  # BPE يهتم فقط بالتكرار المطلق
    wordpiece_score = pair_freq / (first_freq * second_freq)

    print(f"BPE score (تكرار مطلق): {bpe_score}")
    print(f"WordPiece score (احتمالية مُطبَّعة): {wordpiece_score:.6f}")

def explain_unigram_sentencepiece():
    """
    SentencePiece/Unigram (T5, Llama, XLNet): نهج معاكس تمامًا!
    بدلًا من البدء بأحرف والدمج تصاعديًا (Bottom-up كما في BPE/WordPiece)،
    نبدأ بمفردات كبيرة جدًا (كل الكلمات الفرعية المحتملة تقريبًا)
    ثم نُقلِّص تدريجيًا (Top-down) عبر إزالة الرموز التي تُقلِّل
    إزالتها أقل احتمالية ممكنة من إجمالي احتمالية المجموعة التدريبية
    (نموذج احتمالي Unigram Language Model كامل).

    الميزة الحاسمة: SentencePiece يُعامل النص كـ"تيار بايتات خام"
    (raw byte stream) بما في ذلك المسافات (تُمثَّل برمز خاص ▁)،
    مما يجعله مستقلًا تمامًا عن قواعد التقطيع بلغة معينة —
    ميزة بالغة الأهمية للنماذج متعددة اللغات (بما فيها العربية
    التي لا تحتاج فصل كلمات بالطريقة اللاتينية نفسها دائمًا).
    """
    example = "الكتاب جميل"
    sentencepiece_style = "▁الكتاب▁جميل"  # ▁ يُمثِّل المسافة كرمز صريح
    print(f"تمثيل SentencePiece: {sentencepiece_style}")
    print("هذا يسمح بإعادة بناء النص الأصلي بدقة تامة دون أي قواعد")
    print("خاصة باللغة لتحديد أين تقع المسافات (reversible tokenization)")

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-02-language-representation/ch06-tokenization.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import sentencepiece as spm

def train_sentencepiece_for_arabic(corpus_file: str, model_prefix: str = "arabic_tokenizer"):
    """
    التدريب الفعلي لـ SentencePiece على بيانات عربية، باستخدام
    إعدادات مُحسَّنة خصيصًا للتعامل مع خصائص العربية (التشكيل،
    الحروف المُركَّبة، النطاق الواسع من الرموز اليونيكود).
    """
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=model_prefix,
        vocab_size=32000,
        model_type="unigram",        # أو "bpe" حسب التفضيل

        # إعداد حرج جدًا للعربية: character_coverage يُحدِّد نسبة
        # الأحرف التي يجب أن يُغطّيها النموذج. للعربية (ولغات أخرى
        # غنية بالرموز مثل التشكيل والحروف الخاصة)، القيمة الافتراضية
        # 0.9995 (بدلًا من 1.0) تُهمل الأحرف النادرة جدًا (أخطاء إملائية،
        # رموز شاذة) لتجنّب تضخيم المفردات بأحرف غير مفيدة عمليًا.
        character_coverage=0.9995,

        normalization_rule_name="nfkc",  # تطبيع يونيكود موحّد (مهم جدًا للعربية
                                           # بسبب تعدد أشكال نفس الحرف Unicode)
        byte_fallback=True,   # أي رمز غير مُغطّى يُرمَّز كبايتات خام (لا OOV إطلاقًا!)

        # رموز خاصة ضرورية لأي LLM
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        user_defined_symbols=["<|system|>", "<|user|>", "<|assistant|>"],
    )
    print(f"✅ تم تدريب SentencePiece Tokenizer وحفظه بـ {model_prefix}.model")

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-02-language-representation/ch06-tokenization.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def manual_bpe_walkthrough():
    corpus = {
        "l o w </w>": 5,
        "l o w e r </w>": 2,
        "l o w e s t </w>": 6,
        "n e w e s t </w>": 3,
        "w i d e s t </w>": 2,
    }

    for step in range(3):
        pair_freqs = defaultdict(int)
        for word, freq in corpus.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair_freqs[(symbols[i], symbols[i+1])] += freq

        best_pair = max(pair_freqs, key=pair_freqs.get)
        print(f"الخطوة {step+1}: أكثر زوج تكرارًا = {best_pair} "
              f"(تكرار={pair_freqs[best_pair]})")

        new_corpus = {}
        merged = "".join(best_pair)
        for word, freq in corpus.items():
            new_word = word.replace(" ".join(best_pair), merged)
            new_corpus[new_word] = freq
        corpus = new_corpus
        print(f"   الكلمات بعد الدمج: {list(corpus.keys())}\n")

manual_bpe_walkthrough()
# الخطوة 1: أكثر زوج تكرارًا = ('e', 's') (تكرار=9)
#    الكلمات بعد الدمج: ['l o w </w>', 'l o w e r </w>', 'l o w es t </w>', ...]
# الخطوة 2: أكثر زوج تكرارًا = ('es', 't') (تكرار=9)
#    الكلمات بعد الدمج: [..., 'l o w est </w>', 'n e w est </w>', ...]
# الخطوة 3: أكثر زوج تكرارًا = ('l', 'o') (تكرار=13)
#    الكلمات بعد الدمج: ['lo w </w>', 'lo w e r </w>', 'lo w est </w>', ...]

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-02-language-representation/ch06-tokenization.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-06-ch06-tokenization.ipynb#cell-2
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
    ("sentencepiece", "sentencepiece")
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-06-ch06-tokenization.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import re
from collections import defaultdict, Counter

class BPETokenizerFromScratch:
    """
    تطبيق كامل لخوارزمية Byte-Pair Encoding (Sennrich et al., 2016)
    من الصفر، بدون أي مكتبة جاهزة. هذا هو جوهر Tokenizer في
    GPT-2/GPT-3/GPT-4 وLlama (مع بعض التحسينات الهندسية الإضافية).
    """
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.merges = {}       # قاموس الدمجات المُتعلَّمة: (رمز1, رمز2) -> رمز مُدمَج
        self.vocab = {}        # قاموس المفردات النهائي: رمز -> معرّف رقمي

    def _get_word_frequencies(self, corpus: list) -> dict:
        """
        حساب تكرار كل كلمة في المجموعة النصية، مع تمثيل كل كلمة
        كتسلسل أحرف مُنفصلة + رمز نهاية الكلمة </w> (ضروري لتمييز
        "نهاية الكلمة" عن استمرارها، فـ "كتاب" و"كتابة" تشتركان
        في البادئة لكن يجب تمييز حدود الكلمة الفعلية).
        """
        word_freqs = defaultdict(int)
        for text in corpus:
            words = text.strip().split()
            for word in words:
                # تمثيل الكلمة كتسلسل أحرف منفصلة بمسافات + رمز نهاية
                char_sequence = " ".join(list(word)) + " </w>"
                word_freqs[char_sequence] += 1
        return word_freqs

    def _get_pair_frequencies(self, word_freqs: dict) -> dict:
        """حساب تكرار كل زوج أحرف/رموز متجاورة عبر كل الكلمات."""
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    def _merge_pair(self, pair: tuple, word_freqs: dict) -> dict:
        """
        دمج زوج رموز مُحدَّد في كل الكلمات التي يظهر فيها.
        مثال: دمج ('ك','ت') يُحوِّل "ك ت ا ب" إلى "كت ا ب"
        """
        new_word_freqs = {}
        bigram = re.escape(" ".join(pair))
        pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")

        for word, freq in word_freqs.items():
            new_word = pattern.sub("".join(pair), word)
            new_word_freqs[new_word] = freq
        return new_word_freqs

    def train(self, corpus: list, verbose: bool = True):
        """
        حلقة التدريب الرئيسية: نُكرِّر "إيجاد أكثر زوج تكرارًا ودمجه"
        حتى نصل لحجم المفردات المطلوب أو لا تبقى أزواج قابلة للدمج.
        """
        word_freqs = self._get_word_frequencies(corpus)

        # المفردات الابتدائية: كل حرف فريد + رمز نهاية الكلمة
        initial_chars = set()
        for word in word_freqs:
            initial_chars.update(word.split())
        self.vocab = {char: idx for idx, char in enumerate(sorted(initial_chars))}

        num_merges = self.vocab_size - len(self.vocab)
        for merge_step in range(num_merges):
            pair_freqs = self._get_pair_frequencies(word_freqs)
            if not pair_freqs:
                break  # لا توجد أزواج أخرى للدمج

            best_pair = max(pair_freqs, key=pair_freqs.get)
            word_freqs = self._merge_pair(best_pair, word_freqs)

            merged_token = "".join(best_pair)
            self.merges[best_pair] = merged_token
            self.vocab[merged_token] = len(self.vocab)

            if verbose and merge_step % 50 == 0:
                print(f"الدمج {merge_step}: {best_pair} → '{merged_token}' "
                      f"(تكرار={pair_freqs[best_pair]})")

        if verbose:
            print(f"\n✅ اكتمل التدريب: {len(self.vocab)} رمز في المفردات")

    def tokenize(self, text: str) -> list:
        """تقطيع نص جديد باستخدام الدمجات المُتعلَّمة، بنفس الترتيب الذي تعلّمناها به."""
        words = text.strip().split()
        tokens = []

        for word in words:
            symbols = list(word) + ["</w>"]
            # تطبيق الدمجات المُتعلَّمة تكراريًا بنفس ترتيب الأولوية
            while len(symbols) > 1:
                pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
                # إيجاد الدمج ذي الأولوية الأعلى (الذي تعلّمناه أولًا)
                mergeable = [p for p in pairs if p in self.merges]
                if not mergeable:
                    break
                # نختار أول دمج تعلّمناه (الأقدم زمنيًا = الأكثر عمومية/تكرارًا)
                pair_to_merge = min(mergeable, key=lambda p: list(self.merges.keys()).index(p))
                idx = pairs.index(pair_to_merge)
                symbols = (symbols[:idx] + [self.merges[pair_to_merge]] +
                           symbols[idx + 2:])
            tokens.extend(symbols)

        return tokens

    def encode(self, text: str) -> list:
        """تحويل النص إلى تسلسل معرّفات رقمية (Token IDs)."""
        tokens = self.tokenize(text)
        return [self.vocab.get(tok, self.vocab.get("<unk>", -1)) for tok in tokens]

# === تجربة عملية على نصوص عربية ===
arabic_corpus = [
    "الكتاب على الطاولة",
    "الكتابة مهارة مهمة",
    "كتب الطالب الدرس",
    "يكتب المعلم على السبورة",
    "الكاتب نشر كتابا جديدا",
] * 20  # تكرار لمحاكاة بيانات أكبر

tokenizer = BPETokenizerFromScratch(vocab_size=120)
tokenizer.train(arabic_corpus, verbose=True)

test_text = "الكاتب يكتب كتابا"
tokens = tokenizer.tokenize(test_text)
print(f"\nالنص: {test_text}")
print(f"الرموز الناتجة: {tokens}")
# الرموز الناتجة: ['الكاتب</w>', 'يكتب</w>', 'كتاب', 'ا</w>']
# لاحظ كيف أن الجذر المشترك "كتب/كتاب" يظهر كوحدة متكررة عبر عدة كلمات

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-06-ch06-tokenization.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def explain_wordpiece_difference():
    """
    WordPiece (مُستخدَمة في BERT): تشبه BPE لكن معيار اختيار الزوج
    للدمج مختلف جوهريًا. بدلًا من "أكثر زوج تكرارًا" كما في BPE،
    تختار WordPiece الزوج الذي يُعظِّم احتمالية المجموعة التدريبية
    عند دمجه (Likelihood-based)، أي:

        score(pair) = freq(pair) / (freq(first) × freq(second))

    هذا يُفضِّل دمج الأزواج النادرة التي تظهر معًا بشكل "غير متوقع"
    (أعلى من الصدفة الإحصائية)، وليس فقط الأكثر تكرارًا بشكل مطلق.
    """
    # محاكاة الفرق: زوج ('ال', 'كتاب') قد يكون أقل تكرارًا من ('ت','ا')
    # لكن احتمالية ظهورهما معًا أعلى بكثير من المتوقع عشوائيًا
    pair_freq = 50
    first_freq, second_freq = 200, 180

    bpe_score = pair_freq  # BPE يهتم فقط بالتكرار المطلق
    wordpiece_score = pair_freq / (first_freq * second_freq)

    print(f"BPE score (تكرار مطلق): {bpe_score}")
    print(f"WordPiece score (احتمالية مُطبَّعة): {wordpiece_score:.6f}")

def explain_unigram_sentencepiece():
    """
    SentencePiece/Unigram (T5, Llama, XLNet): نهج معاكس تمامًا!
    بدلًا من البدء بأحرف والدمج تصاعديًا (Bottom-up كما في BPE/WordPiece)،
    نبدأ بمفردات كبيرة جدًا (كل الكلمات الفرعية المحتملة تقريبًا)
    ثم نُقلِّص تدريجيًا (Top-down) عبر إزالة الرموز التي تُقلِّل
    إزالتها أقل احتمالية ممكنة من إجمالي احتمالية المجموعة التدريبية
    (نموذج احتمالي Unigram Language Model كامل).

    الميزة الحاسمة: SentencePiece يُعامل النص كـ"تيار بايتات خام"
    (raw byte stream) بما في ذلك المسافات (تُمثَّل برمز خاص ▁)،
    مما يجعله مستقلًا تمامًا عن قواعد التقطيع بلغة معينة —
    ميزة بالغة الأهمية للنماذج متعددة اللغات (بما فيها العربية
    التي لا تحتاج فصل كلمات بالطريقة اللاتينية نفسها دائمًا).
    """
    example = "الكتاب جميل"
    sentencepiece_style = "▁الكتاب▁جميل"  # ▁ يُمثِّل المسافة كرمز صريح
    print(f"تمثيل SentencePiece: {sentencepiece_style}")
    print("هذا يسمح بإعادة بناء النص الأصلي بدقة تامة دون أي قواعد")
    print("خاصة باللغة لتحديد أين تقع المسافات (reversible tokenization)")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-06-ch06-tokenization.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import sentencepiece as spm

def train_sentencepiece_for_arabic(corpus_file: str, model_prefix: str = "arabic_tokenizer"):
    """
    التدريب الفعلي لـ SentencePiece على بيانات عربية، باستخدام
    إعدادات مُحسَّنة خصيصًا للتعامل مع خصائص العربية (التشكيل،
    الحروف المُركَّبة، النطاق الواسع من الرموز اليونيكود).
    """
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=model_prefix,
        vocab_size=32000,
        model_type="unigram",        # أو "bpe" حسب التفضيل

        # إعداد حرج جدًا للعربية: character_coverage يُحدِّد نسبة
        # الأحرف التي يجب أن يُغطّيها النموذج. للعربية (ولغات أخرى
        # غنية بالرموز مثل التشكيل والحروف الخاصة)، القيمة الافتراضية
        # 0.9995 (بدلًا من 1.0) تُهمل الأحرف النادرة جدًا (أخطاء إملائية،
        # رموز شاذة) لتجنّب تضخيم المفردات بأحرف غير مفيدة عمليًا.
        character_coverage=0.9995,

        normalization_rule_name="nfkc",  # تطبيع يونيكود موحّد (مهم جدًا للعربية
                                           # بسبب تعدد أشكال نفس الحرف Unicode)
        byte_fallback=True,   # أي رمز غير مُغطّى يُرمَّز كبايتات خام (لا OOV إطلاقًا!)

        # رموز خاصة ضرورية لأي LLM
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        user_defined_symbols=["<|system|>", "<|user|>", "<|assistant|>"],
    )
    print(f"✅ تم تدريب SentencePiece Tokenizer وحفظه بـ {model_prefix}.model")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-06-ch06-tokenization.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def manual_bpe_walkthrough():
    corpus = {
        "l o w </w>": 5,
        "l o w e r </w>": 2,
        "l o w e s t </w>": 6,
        "n e w e s t </w>": 3,
        "w i d e s t </w>": 2,
    }

    for step in range(3):
        pair_freqs = defaultdict(int)
        for word, freq in corpus.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair_freqs[(symbols[i], symbols[i+1])] += freq

        best_pair = max(pair_freqs, key=pair_freqs.get)
        print(f"الخطوة {step+1}: أكثر زوج تكرارًا = {best_pair} "
              f"(تكرار={pair_freqs[best_pair]})")

        new_corpus = {}
        merged = "".join(best_pair)
        for word, freq in corpus.items():
            new_word = word.replace(" ".join(best_pair), merged)
            new_corpus[new_word] = freq
        corpus = new_corpus
        print(f"   الكلمات بعد الدمج: {list(corpus.keys())}\n")

manual_bpe_walkthrough()
# الخطوة 1: أكثر زوج تكرارًا = ('e', 's') (تكرار=9)
#    الكلمات بعد الدمج: ['l o w </w>', 'l o w e r </w>', 'l o w es t </w>', ...]
# الخطوة 2: أكثر زوج تكرارًا = ('es', 't') (تكرار=9)
#    الكلمات بعد الدمج: [..., 'l o w est </w>', 'n e w est </w>', ...]
# الخطوة 3: أكثر زوج تكرارًا = ('l', 'o') (تكرار=13)
#    الكلمات بعد الدمج: ['lo w </w>', 'lo w e r </w>', 'lo w est </w>', ...]

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-llms-transformers/notebooks/ch-06-ch06-tokenization.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
