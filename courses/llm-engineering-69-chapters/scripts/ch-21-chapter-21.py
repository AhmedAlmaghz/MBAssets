#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل الحادي والعشرون: تصميم Tokenizer مخصص على نطاق واسع
Large Language Model Engineering — Tokenizer

Book slug : llm-engineering-69-chapters
Chapter   : ch-21-chapter-21 (lesson 21)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-21-chapter-21.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل الحادي والعشرون: تصميم Tokenizer مخصص على نطاق واسع
Large Language Model Engineering — Tokenizer

Book slug : llm-engineering-69-chapters
Chapter   : ch-21-chapter-21 (lesson 21)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-21-chapter-21.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل الحادي والعشرون: تصميم Tokenizer مخصص على نطاق واسع
Large Language Model Engineering — Tokenizer

Book slug : llm-engineering-69-chapters
Chapter   : ch-21-chapter-21 (lesson 21)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_21.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 21: الفصل الحادي والعشرون: تصميم Tokenizer مخصص على نطاق واسع."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_bpe() -> None:
    words = ['low', 'lowest', 'newer', 'wider']
    symbols = [list(word) + ['</w>'] for word in words]
    pairs = Counter((pair for word in symbols for pair in zip(word, word[1:])))
    pair, count = pairs.most_common(1)[0]
    merged = ''.join(pair)
    updated = [[merged if tuple(word[i:i + 2]) == pair else word[i] for i in range(len(word)) if i + 1 >= len(word) or tuple(word[i:i + 2]) != pair] for word in symbols]
    print('most frequent pair:', pair, 'count:', count)
    print('merged symbol:', merged)
    print('vocabulary size before/after:', sum(map(len, symbols)), sum(map(len, updated)))

if __name__ == "__main__":
    print("Chapter 21 lab: Learn byte-pair merges from a tiny corpus")
    run_bpe()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-05-build-an-llm/chapter-21.md
# ────────────────────────────────────────────────────────────────────────
"""
chapter21_multilingual_tokenizer.py
تدريب Tokenizer BPE من الصفر، متوازن بين العربية والإنجليزية،
وقياس جودته عبر Fertility وCompression Ratio.

Python 3.13+
المتطلبات: pip install tokenizers datasets
"""
from __future__ import annotations

from dataclasses import dataclass

from tokenizers import Tokenizer, normalizers, pre_tokenizers, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel


# =========================================================================
# 1) بيانات تجريبية: مزيج عربي-إنجليزي (في الواقع: ملايين الوثائق من ف20)
# =========================================================================
ARABIC_CORPUS = [
    "الذكاء الاصطناعي يغير طريقة تفاعلنا مع التكنولوجيا بشكل جذري.",
    "تطورت النماذج اللغوية الكبيرة بسرعة مذهلة خلال السنوات الأخيرة.",
    "معالجة اللغة الطبيعية مجال واسع يشمل الترجمة والتلخيص والتصنيف.",
    "البيانات النظيفة عالية الجودة أهم من الكمية الهائلة من البيانات الفوضوية.",
    "يحتاج بناء نموذج لغوي ناجح إلى بنية تحتية حاسوبية قوية جدًا ومتينة.",
] * 200  # تكرار لمحاكاة حجم أكبر لأغراض العرض التوضيحي

ENGLISH_CORPUS = [
    "Artificial intelligence is transforming how we interact with technology.",
    "Large language models have evolved remarkably over the past few years.",
    "Natural language processing covers translation, summarization, and classification.",
    "Clean high-quality data matters more than raw volume of noisy data.",
    "Building a successful language model requires strong computing infrastructure.",
] * 200


@dataclass
class CorpusStats:
    name: str
    n_words: int
    n_tokens: int

    @property
    def fertility(self) -> float:
        return self.n_tokens / self.n_words if self.n_words else float("inf")


def build_balanced_training_iterator(
    arabic_docs: list[str], english_docs: list[str], arabic_boost: int = 3
) -> list[str]:
    """
    محاكاة بسيطة لـ Upsampling: نُكرر الوثائق العربية arabic_boost مرة
    لموازنة هيمنة الإنجليزية الطبيعية في بيانات الويب الحقيقية.
    """
    return arabic_docs * arabic_boost + english_docs


def train_bpe_tokenizer(
    training_corpus: list[str], vocab_size: int = 8000
) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    # التطبيع: NFKC لتوحيد أشكال الحروف المتكافئة يونيكوديًا
    tokenizer.normalizer = normalizers.Sequence([NFKC()])

    # ما قبل Tokenization: ByteLevel يضمن تمثيل أي حرف (بما فيه العربية) بلا <unk>
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>", "<mask>"]
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train_from_iterator(training_corpus, trainer=trainer)
    return tokenizer


def measure_fertility(
    tokenizer: Tokenizer, docs: list[str], language_name: str
) -> CorpusStats:
    total_words = 0
    total_tokens = 0
    for doc in docs:
        total_words += len(doc.split())
        encoding = tokenizer.encode(doc)
        total_tokens += len(encoding.ids)
    return CorpusStats(name=language_name, n_words=total_words, n_tokens=total_tokens)


def measure_compression_ratio(tokenizer: Tokenizer, docs: list[str]) -> float:
    """نسبة البايتات الخام إلى عدد الرموز الناتجة (أعلى = ضغط أفضل)."""
    total_bytes = sum(len(doc.encode("utf-8")) for doc in docs)
    total_tokens = sum(len(tokenizer.encode(doc).ids) for doc in docs)
    return total_bytes / total_tokens if total_tokens else 0.0


if __name__ == "__main__":
    # --- تجربة 1: Tokenizer غير متوازن (يحاكي الهيمنة الطبيعية للإنجليزية) ---
    print("=" * 70)
    print("تجربة 1: Tokenizer مُدرَّب على بيانات غير متوازنة (إنجليزية مهيمنة)")
    print("=" * 70)
    unbalanced_corpus = ARABIC_CORPUS[:50] + ENGLISH_CORPUS  # عربية قليلة جدًا
    tok_unbalanced = train_bpe_tokenizer(unbalanced_corpus, vocab_size=2000)

    stats_ar_unb = measure_fertility(tok_unbalanced, ARABIC_CORPUS[:100], "عربي")
    stats_en_unb = measure_fertility(tok_unbalanced, ENGLISH_CORPUS[:100], "إنجليزي")
    print(f"Fertility (عربي):    {stats_ar_unb.fertility:.2f} رمز/كلمة")
    print(f"Fertility (إنجليزي): {stats_en_unb.fertility:.2f} رمز/كلمة")
    print(f"الفجوة (عربي/إنجليزي): {stats_ar_unb.fertility / stats_en_unb.fertility:.2f}x\n")

    # --- تجربة 2: Tokenizer متوازن (عبر Upsampling للعربية) ---
    print("=" * 70)
    print("تجربة 2: Tokenizer مُدرَّب على بيانات مُعاد موازنتها (Upsampling)")
    print("=" * 70)
    balanced_corpus = build_balanced_training_iterator(
        ARABIC_CORPUS, ENGLISH_CORPUS, arabic_boost=3
    )
    tok_balanced = train_bpe_tokenizer(balanced_corpus, vocab_size=2000)

    stats_ar_bal = measure_fertility(tok_balanced, ARABIC_CORPUS[:100], "عربي")
    stats_en_bal = measure_fertility(tok_balanced, ENGLISH_CORPUS[:100], "إنجليزي")
    print(f"Fertility (عربي):    {stats_ar_bal.fertility:.2f} رمز/كلمة")
    print(f"Fertility (إنجليزي): {stats_en_bal.fertility:.2f} رمز/كلمة")
    print(f"الفجوة (عربي/إنجليزي): {stats_ar_bal.fertility / stats_en_bal.fertility:.2f}x\n")

    # --- مثال توضيحي: تفكيك جملة عربية فعليًا بكلا الـ Tokenizers ---
    sample_sentence = "الذكاء الاصطناعي يغير العالم"
    print("=" * 70)
    print(f"مثال تفكيك الجملة: \"{sample_sentence}\"\n")
    enc_unb = tok_unbalanced.encode(sample_sentence)
    enc_bal = tok_balanced.encode(sample_sentence)
    print(f"[غير متوازن] عدد الرموز: {len(enc_unb.ids)} -> {enc_unb.tokens}")
    print(f"[متوازن]     عدد الرموز: {len(enc_bal.ids)} -> {enc_bal.tokens}")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-05-build-an-llm/chapter-21.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ شائع جدًا عند المبتدئين: استخدام tokenizer جاهز دون فحص Fertility للعربية
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("gpt2")  # مُدرَّب حصريًا تقريبًا على الإنجليزية
tokens = tok.encode("الذكاء الاصطناعي رائع")
print(len(tokens))  # قد يكون 20+ رمزًا لجملة من 3 كلمات فقط!

# ✅ صحيح: افحص دائمًا Fertility قبل اعتماد أي Tokenizer جاهز لمشروعك
def quick_fertility_check(tokenizer, text: str) -> float:
    n_words = len(text.split())
    n_tokens = len(tokenizer.encode(text))
    return n_tokens / n_words

print(quick_fertility_check(tok, "الذكاء الاصطناعي رائع جدا"))  # إن كانت > 3، فكّر جديًا بإعادة التدريب

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-05-build-an-llm/chapter-21.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: اختيار حجم قاموس ضخم (256K) لنموذج صغير جدًا (125 مليون معامل)
# النتيجة: طبقة التضمين تستحوذ على نسبة غير متناسبة من إجمالي المعاملات
vocab_size = 256_000
d_model = 768
embedding_params = vocab_size * d_model  # ~196 مليون معامل!
total_model_params = 125_000_000
print(embedding_params / total_model_params)  # > 100%! غير منطقي إطلاقًا

# ✅ صحيح: اختر حجم القاموس بما يتناسب مع حجم النموذج الكلي
# (قاعدة تقريبية شائعة: نسبة معاملات التضمين يجب ألا تتجاوز 10-20% من الإجمالي)

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-05-build-an-llm/chapter-21.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: ترك BPE "يقرر" كيفية تفكيك الأرقام دون توجيه، مما ينتج تمثيلات غير متسقة
# "123" قد تُصبح رمزًا واحدًا، بينما "124" تتفكك لرمزين مختلفين تمامًا، مما يُربك
# قدرة النموذج على تعلّم الحساب والأنماط الرقمية بثبات

# ✅ صحيح: أضف قاعدة pre-tokenization تُفكّك كل رقم إلى آحاده
from tokenizers.pre_tokenizers import Digits, Sequence, ByteLevel

tokenizer.pre_tokenizer = Sequence([
    Digits(individual_digits=True),  # كل رقم منفصل كوحدة مستقلة
    ByteLevel(add_prefix_space=False),
])

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-05-build-an-llm/chapter-21.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: الاكتفاء بقياس Fertility منخفضة كدليل كافٍ على "نجاح" Tokenizer
# Fertility منخفضة ضرورية لكنها غير كافية وحدها لضمان جودة النموذج النهائي

# ✅ صحيح: درّب نموذجًا صغيرًا بكل Tokenizer مرشّح، وقارن منحنى الخسارة
# (Loss Curve) والأداء على معيار تقييم حقيقي (الفصل 46) قبل اعتماد القرار النهائي

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-05-build-an-llm/chapter-21.md
# ────────────────────────────────────────────────────────────────────────
"""
mini_project_production_tokenizer.py
بناء Tokenizer إنتاجي متوازن (32K رمز) على بيانات حقيقية من ويكيبيديا
العربية والإنجليزية، ومقارنة Fertility مع tokenizer جاهز (مثل GPT-2 أو LLaMA).

المطلوب إكماله:
1. حمّل عيّنتين من "wikimedia/wikipedia": "20231101.ar" و "20231101.en"
   (عبر streaming=True لتفادي تحميل كل البيانات).
2. طبّق استراتيجية Temperature Sampling (T=3) لموازنة نسبة اللغتين
   في بيانات تدريب Tokenizer (بضعة آلاف وثيقة من كل لغة تكفي للتجربة).
3. درّب BPE Tokenizer بحجم قاموس 32,000 باستخدام مكتبة `tokenizers`.
4. حمّل Tokenizer جاهزًا من Hugging Face Hub (مثل "gpt2" أو
   "meta-llama/Llama-2-7b-hf" إن توفّر الوصول) للمقارنة.
5. احسب Fertility وCompression Ratio لكليهما على عيّنة تقييم عربية
   وعيّنة تقييم إنجليزية منفصلتين تمامًا عن بيانات التدريب.
6. اطبع جدول مقارنة نهائي شامل.
7. (تحدٍّ إضافي) احفظ Tokenizer المُدرَّب بصيغة متوافقة مع
   Hugging Face `transformers` (PreTrainedTokenizerFast) لاستخدامه
   لاحقًا في الفصل 22 عند بناء نموذج كامل.
"""
from __future__ import annotations

from datasets import load_dataset
from tokenizers import Tokenizer


def load_balanced_sample(
    n_arabic: int = 5000, n_english: int = 5000
) -> list[str]:
    # TODO: استخدم load_dataset(..., streaming=True) لكلتا اللغتين
    # وخذ العدد المطلوب من كل لغة
    raise NotImplementedError


def main() -> None:
    corpus = load_balanced_sample()
    # TODO: درّب Tokenizer (استخدم الكود من هذا الفصل كأساس)
    # TODO: حمّل tokenizer جاهزًا من HF Hub للمقارنة
    # TODO: احسب واطبع جدول Fertility/Compression النهائي
    pass


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-21-chapter-21.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 21: الفصل الحادي والعشرون: تصميم Tokenizer مخصص على نطاق واسع."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_bpe() -> None:
    words = ['low', 'lowest', 'newer', 'wider']
    symbols = [list(word) + ['</w>'] for word in words]
    pairs = Counter((pair for word in symbols for pair in zip(word, word[1:])))
    pair, count = pairs.most_common(1)[0]
    merged = ''.join(pair)
    updated = [[merged if tuple(word[i:i + 2]) == pair else word[i] for i in range(len(word)) if i + 1 >= len(word) or tuple(word[i:i + 2]) != pair] for word in symbols]
    print('most frequent pair:', pair, 'count:', count)
    print('merged symbol:', merged)
    print('vocabulary size before/after:', sum(map(len, symbols)), sum(map(len, updated)))

if __name__ == "__main__":
    print("Chapter 21 lab: Learn byte-pair merges from a tiny corpus")
    run_bpe()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-21-chapter-21.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
chapter21_multilingual_tokenizer.py
تدريب Tokenizer BPE من الصفر، متوازن بين العربية والإنجليزية،
وقياس جودته عبر Fertility وCompression Ratio.

Python 3.13+
المتطلبات: pip install tokenizers datasets
"""
from __future__ import annotations

from dataclasses import dataclass

from tokenizers import Tokenizer, normalizers, pre_tokenizers, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel


# =========================================================================
# 1) بيانات تجريبية: مزيج عربي-إنجليزي (في الواقع: ملايين الوثائق من ف20)
# =========================================================================
ARABIC_CORPUS = [
    "الذكاء الاصطناعي يغير طريقة تفاعلنا مع التكنولوجيا بشكل جذري.",
    "تطورت النماذج اللغوية الكبيرة بسرعة مذهلة خلال السنوات الأخيرة.",
    "معالجة اللغة الطبيعية مجال واسع يشمل الترجمة والتلخيص والتصنيف.",
    "البيانات النظيفة عالية الجودة أهم من الكمية الهائلة من البيانات الفوضوية.",
    "يحتاج بناء نموذج لغوي ناجح إلى بنية تحتية حاسوبية قوية جدًا ومتينة.",
] * 200  # تكرار لمحاكاة حجم أكبر لأغراض العرض التوضيحي

ENGLISH_CORPUS = [
    "Artificial intelligence is transforming how we interact with technology.",
    "Large language models have evolved remarkably over the past few years.",
    "Natural language processing covers translation, summarization, and classification.",
    "Clean high-quality data matters more than raw volume of noisy data.",
    "Building a successful language model requires strong computing infrastructure.",
] * 200


@dataclass
class CorpusStats:
    name: str
    n_words: int
    n_tokens: int

    @property
    def fertility(self) -> float:
        return self.n_tokens / self.n_words if self.n_words else float("inf")


def build_balanced_training_iterator(
    arabic_docs: list[str], english_docs: list[str], arabic_boost: int = 3
) -> list[str]:
    """
    محاكاة بسيطة لـ Upsampling: نُكرر الوثائق العربية arabic_boost مرة
    لموازنة هيمنة الإنجليزية الطبيعية في بيانات الويب الحقيقية.
    """
    return arabic_docs * arabic_boost + english_docs


def train_bpe_tokenizer(
    training_corpus: list[str], vocab_size: int = 8000
) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    # التطبيع: NFKC لتوحيد أشكال الحروف المتكافئة يونيكوديًا
    tokenizer.normalizer = normalizers.Sequence([NFKC()])

    # ما قبل Tokenization: ByteLevel يضمن تمثيل أي حرف (بما فيه العربية) بلا <unk>
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>", "<mask>"]
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train_from_iterator(training_corpus, trainer=trainer)
    return tokenizer


def measure_fertility(
    tokenizer: Tokenizer, docs: list[str], language_name: str
) -> CorpusStats:
    total_words = 0
    total_tokens = 0
    for doc in docs:
        total_words += len(doc.split())
        encoding = tokenizer.encode(doc)
        total_tokens += len(encoding.ids)
    return CorpusStats(name=language_name, n_words=total_words, n_tokens=total_tokens)


def measure_compression_ratio(tokenizer: Tokenizer, docs: list[str]) -> float:
    """نسبة البايتات الخام إلى عدد الرموز الناتجة (أعلى = ضغط أفضل)."""
    total_bytes = sum(len(doc.encode("utf-8")) for doc in docs)
    total_tokens = sum(len(tokenizer.encode(doc).ids) for doc in docs)
    return total_bytes / total_tokens if total_tokens else 0.0


if __name__ == "__main__":
    # --- تجربة 1: Tokenizer غير متوازن (يحاكي الهيمنة الطبيعية للإنجليزية) ---
    print("=" * 70)
    print("تجربة 1: Tokenizer مُدرَّب على بيانات غير متوازنة (إنجليزية مهيمنة)")
    print("=" * 70)
    unbalanced_corpus = ARABIC_CORPUS[:50] + ENGLISH_CORPUS  # عربية قليلة جدًا
    tok_unbalanced = train_bpe_tokenizer(unbalanced_corpus, vocab_size=2000)

    stats_ar_unb = measure_fertility(tok_unbalanced, ARABIC_CORPUS[:100], "عربي")
    stats_en_unb = measure_fertility(tok_unbalanced, ENGLISH_CORPUS[:100], "إنجليزي")
    print(f"Fertility (عربي):    {stats_ar_unb.fertility:.2f} رمز/كلمة")
    print(f"Fertility (إنجليزي): {stats_en_unb.fertility:.2f} رمز/كلمة")
    print(f"الفجوة (عربي/إنجليزي): {stats_ar_unb.fertility / stats_en_unb.fertility:.2f}x\n")

    # --- تجربة 2: Tokenizer متوازن (عبر Upsampling للعربية) ---
    print("=" * 70)
    print("تجربة 2: Tokenizer مُدرَّب على بيانات مُعاد موازنتها (Upsampling)")
    print("=" * 70)
    balanced_corpus = build_balanced_training_iterator(
        ARABIC_CORPUS, ENGLISH_CORPUS, arabic_boost=3
    )
    tok_balanced = train_bpe_tokenizer(balanced_corpus, vocab_size=2000)

    stats_ar_bal = measure_fertility(tok_balanced, ARABIC_CORPUS[:100], "عربي")
    stats_en_bal = measure_fertility(tok_balanced, ENGLISH_CORPUS[:100], "إنجليزي")
    print(f"Fertility (عربي):    {stats_ar_bal.fertility:.2f} رمز/كلمة")
    print(f"Fertility (إنجليزي): {stats_en_bal.fertility:.2f} رمز/كلمة")
    print(f"الفجوة (عربي/إنجليزي): {stats_ar_bal.fertility / stats_en_bal.fertility:.2f}x\n")

    # --- مثال توضيحي: تفكيك جملة عربية فعليًا بكلا الـ Tokenizers ---
    sample_sentence = "الذكاء الاصطناعي يغير العالم"
    print("=" * 70)
    print(f"مثال تفكيك الجملة: \"{sample_sentence}\"\n")
    enc_unb = tok_unbalanced.encode(sample_sentence)
    enc_bal = tok_balanced.encode(sample_sentence)
    print(f"[غير متوازن] عدد الرموز: {len(enc_unb.ids)} -> {enc_unb.tokens}")
    print(f"[متوازن]     عدد الرموز: {len(enc_bal.ids)} -> {enc_bal.tokens}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-21-chapter-21.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ شائع جدًا عند المبتدئين: استخدام tokenizer جاهز دون فحص Fertility للعربية
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("gpt2")  # مُدرَّب حصريًا تقريبًا على الإنجليزية
tokens = tok.encode("الذكاء الاصطناعي رائع")
print(len(tokens))  # قد يكون 20+ رمزًا لجملة من 3 كلمات فقط!

# ✅ صحيح: افحص دائمًا Fertility قبل اعتماد أي Tokenizer جاهز لمشروعك
def quick_fertility_check(tokenizer, text: str) -> float:
    n_words = len(text.split())
    n_tokens = len(tokenizer.encode(text))
    return n_tokens / n_words

print(quick_fertility_check(tok, "الذكاء الاصطناعي رائع جدا"))  # إن كانت > 3، فكّر جديًا بإعادة التدريب

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-21-chapter-21.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: اختيار حجم قاموس ضخم (256K) لنموذج صغير جدًا (125 مليون معامل)
# النتيجة: طبقة التضمين تستحوذ على نسبة غير متناسبة من إجمالي المعاملات
vocab_size = 256_000
d_model = 768
embedding_params = vocab_size * d_model  # ~196 مليون معامل!
total_model_params = 125_000_000
print(embedding_params / total_model_params)  # > 100%! غير منطقي إطلاقًا

# ✅ صحيح: اختر حجم القاموس بما يتناسب مع حجم النموذج الكلي
# (قاعدة تقريبية شائعة: نسبة معاملات التضمين يجب ألا تتجاوز 10-20% من الإجمالي)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-21-chapter-21.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: ترك BPE "يقرر" كيفية تفكيك الأرقام دون توجيه، مما ينتج تمثيلات غير متسقة
# "123" قد تُصبح رمزًا واحدًا، بينما "124" تتفكك لرمزين مختلفين تمامًا، مما يُربك
# قدرة النموذج على تعلّم الحساب والأنماط الرقمية بثبات

# ✅ صحيح: أضف قاعدة pre-tokenization تُفكّك كل رقم إلى آحاده
from tokenizers.pre_tokenizers import Digits, Sequence, ByteLevel

tokenizer.pre_tokenizer = Sequence([
    Digits(individual_digits=True),  # كل رقم منفصل كوحدة مستقلة
    ByteLevel(add_prefix_space=False),
])

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-21-chapter-21.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: الاكتفاء بقياس Fertility منخفضة كدليل كافٍ على "نجاح" Tokenizer
# Fertility منخفضة ضرورية لكنها غير كافية وحدها لضمان جودة النموذج النهائي

# ✅ صحيح: درّب نموذجًا صغيرًا بكل Tokenizer مرشّح، وقارن منحنى الخسارة
# (Loss Curve) والأداء على معيار تقييم حقيقي (الفصل 46) قبل اعتماد القرار النهائي

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-21-chapter-21.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
"""
mini_project_production_tokenizer.py
بناء Tokenizer إنتاجي متوازن (32K رمز) على بيانات حقيقية من ويكيبيديا
العربية والإنجليزية، ومقارنة Fertility مع tokenizer جاهز (مثل GPT-2 أو LLaMA).

المطلوب إكماله:
1. حمّل عيّنتين من "wikimedia/wikipedia": "20231101.ar" و "20231101.en"
   (عبر streaming=True لتفادي تحميل كل البيانات).
2. طبّق استراتيجية Temperature Sampling (T=3) لموازنة نسبة اللغتين
   في بيانات تدريب Tokenizer (بضعة آلاف وثيقة من كل لغة تكفي للتجربة).
3. درّب BPE Tokenizer بحجم قاموس 32,000 باستخدام مكتبة `tokenizers`.
4. حمّل Tokenizer جاهزًا من Hugging Face Hub (مثل "gpt2" أو
   "meta-llama/Llama-2-7b-hf" إن توفّر الوصول) للمقارنة.
5. احسب Fertility وCompression Ratio لكليهما على عيّنة تقييم عربية
   وعيّنة تقييم إنجليزية منفصلتين تمامًا عن بيانات التدريب.
6. اطبع جدول مقارنة نهائي شامل.
7. (تحدٍّ إضافي) احفظ Tokenizer المُدرَّب بصيغة متوافقة مع
   Hugging Face `transformers` (PreTrainedTokenizerFast) لاستخدامه
   لاحقًا في الفصل 22 عند بناء نموذج كامل.
"""
from __future__ import annotations

from datasets import load_dataset
from tokenizers import Tokenizer


def load_balanced_sample(
    n_arabic: int = 5000, n_english: int = 5000
) -> list[str]:
    # TODO: استخدم load_dataset(..., streaming=True) لكلتا اللغتين
    # وخذ العدد المطلوب من كل لغة
    raise NotImplementedError


def main() -> None:
    corpus = load_balanced_sample()
    # TODO: درّب Tokenizer (استخدم الكود من هذا الفصل كأساس)
    # TODO: حمّل tokenizer جاهزًا من HF Hub للمقارنة
    # TODO: احسب واطبع جدول Fertility/Compression النهائي
    pass


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-21-chapter-21.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الفصل الحادي والعشرون: تصميم Tokenizer مخصص على نطاق واسع
Large Language Model Engineering — Tokenizer

Book slug : llm-engineering-69-chapters
Chapter   : ch-21-chapter-21 (lesson 21)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_21.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 21: الفصل الحادي والعشرون: تصميم Tokenizer مخصص على نطاق واسع."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_bpe() -> None:
    words = ['low', 'lowest', 'newer', 'wider']
    symbols = [list(word) + ['</w>'] for word in words]
    pairs = Counter((pair for word in symbols for pair in zip(word, word[1:])))
    pair, count = pairs.most_common(1)[0]
    merged = ''.join(pair)
    updated = [[merged if tuple(word[i:i + 2]) == pair else word[i] for i in range(len(word)) if i + 1 >= len(word) or tuple(word[i:i + 2]) != pair] for word in symbols]
    print('most frequent pair:', pair, 'count:', count)
    print('merged symbol:', merged)
    print('vocabulary size before/after:', sum(map(len, symbols)), sum(map(len, updated)))

if __name__ == "__main__":
    print("Chapter 21 lab: Learn byte-pair merges from a tiny corpus")
    run_bpe()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-05-build-an-llm/chapter-21.md
# ────────────────────────────────────────────────────────────────────────
"""
chapter21_multilingual_tokenizer.py
تدريب Tokenizer BPE من الصفر، متوازن بين العربية والإنجليزية،
وقياس جودته عبر Fertility وCompression Ratio.

Python 3.13+
المتطلبات: pip install tokenizers datasets
"""
from __future__ import annotations

from dataclasses import dataclass

from tokenizers import Tokenizer, normalizers, pre_tokenizers, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel


# =========================================================================
# 1) بيانات تجريبية: مزيج عربي-إنجليزي (في الواقع: ملايين الوثائق من ف20)
# =========================================================================
ARABIC_CORPUS = [
    "الذكاء الاصطناعي يغير طريقة تفاعلنا مع التكنولوجيا بشكل جذري.",
    "تطورت النماذج اللغوية الكبيرة بسرعة مذهلة خلال السنوات الأخيرة.",
    "معالجة اللغة الطبيعية مجال واسع يشمل الترجمة والتلخيص والتصنيف.",
    "البيانات النظيفة عالية الجودة أهم من الكمية الهائلة من البيانات الفوضوية.",
    "يحتاج بناء نموذج لغوي ناجح إلى بنية تحتية حاسوبية قوية جدًا ومتينة.",
] * 200  # تكرار لمحاكاة حجم أكبر لأغراض العرض التوضيحي

ENGLISH_CORPUS = [
    "Artificial intelligence is transforming how we interact with technology.",
    "Large language models have evolved remarkably over the past few years.",
    "Natural language processing covers translation, summarization, and classification.",
    "Clean high-quality data matters more than raw volume of noisy data.",
    "Building a successful language model requires strong computing infrastructure.",
] * 200


@dataclass
class CorpusStats:
    name: str
    n_words: int
    n_tokens: int

    @property
    def fertility(self) -> float:
        return self.n_tokens / self.n_words if self.n_words else float("inf")


def build_balanced_training_iterator(
    arabic_docs: list[str], english_docs: list[str], arabic_boost: int = 3
) -> list[str]:
    """
    محاكاة بسيطة لـ Upsampling: نُكرر الوثائق العربية arabic_boost مرة
    لموازنة هيمنة الإنجليزية الطبيعية في بيانات الويب الحقيقية.
    """
    return arabic_docs * arabic_boost + english_docs


def train_bpe_tokenizer(
    training_corpus: list[str], vocab_size: int = 8000
) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    # التطبيع: NFKC لتوحيد أشكال الحروف المتكافئة يونيكوديًا
    tokenizer.normalizer = normalizers.Sequence([NFKC()])

    # ما قبل Tokenization: ByteLevel يضمن تمثيل أي حرف (بما فيه العربية) بلا <unk>
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>", "<mask>"]
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train_from_iterator(training_corpus, trainer=trainer)
    return tokenizer


def measure_fertility(
    tokenizer: Tokenizer, docs: list[str], language_name: str
) -> CorpusStats:
    total_words = 0
    total_tokens = 0
    for doc in docs:
        total_words += len(doc.split())
        encoding = tokenizer.encode(doc)
        total_tokens += len(encoding.ids)
    return CorpusStats(name=language_name, n_words=total_words, n_tokens=total_tokens)


def measure_compression_ratio(tokenizer: Tokenizer, docs: list[str]) -> float:
    """نسبة البايتات الخام إلى عدد الرموز الناتجة (أعلى = ضغط أفضل)."""
    total_bytes = sum(len(doc.encode("utf-8")) for doc in docs)
    total_tokens = sum(len(tokenizer.encode(doc).ids) for doc in docs)
    return total_bytes / total_tokens if total_tokens else 0.0


if __name__ == "__main__":
    # --- تجربة 1: Tokenizer غير متوازن (يحاكي الهيمنة الطبيعية للإنجليزية) ---
    print("=" * 70)
    print("تجربة 1: Tokenizer مُدرَّب على بيانات غير متوازنة (إنجليزية مهيمنة)")
    print("=" * 70)
    unbalanced_corpus = ARABIC_CORPUS[:50] + ENGLISH_CORPUS  # عربية قليلة جدًا
    tok_unbalanced = train_bpe_tokenizer(unbalanced_corpus, vocab_size=2000)

    stats_ar_unb = measure_fertility(tok_unbalanced, ARABIC_CORPUS[:100], "عربي")
    stats_en_unb = measure_fertility(tok_unbalanced, ENGLISH_CORPUS[:100], "إنجليزي")
    print(f"Fertility (عربي):    {stats_ar_unb.fertility:.2f} رمز/كلمة")
    print(f"Fertility (إنجليزي): {stats_en_unb.fertility:.2f} رمز/كلمة")
    print(f"الفجوة (عربي/إنجليزي): {stats_ar_unb.fertility / stats_en_unb.fertility:.2f}x\n")

    # --- تجربة 2: Tokenizer متوازن (عبر Upsampling للعربية) ---
    print("=" * 70)
    print("تجربة 2: Tokenizer مُدرَّب على بيانات مُعاد موازنتها (Upsampling)")
    print("=" * 70)
    balanced_corpus = build_balanced_training_iterator(
        ARABIC_CORPUS, ENGLISH_CORPUS, arabic_boost=3
    )
    tok_balanced = train_bpe_tokenizer(balanced_corpus, vocab_size=2000)

    stats_ar_bal = measure_fertility(tok_balanced, ARABIC_CORPUS[:100], "عربي")
    stats_en_bal = measure_fertility(tok_balanced, ENGLISH_CORPUS[:100], "إنجليزي")
    print(f"Fertility (عربي):    {stats_ar_bal.fertility:.2f} رمز/كلمة")
    print(f"Fertility (إنجليزي): {stats_en_bal.fertility:.2f} رمز/كلمة")
    print(f"الفجوة (عربي/إنجليزي): {stats_ar_bal.fertility / stats_en_bal.fertility:.2f}x\n")

    # --- مثال توضيحي: تفكيك جملة عربية فعليًا بكلا الـ Tokenizers ---
    sample_sentence = "الذكاء الاصطناعي يغير العالم"
    print("=" * 70)
    print(f"مثال تفكيك الجملة: \"{sample_sentence}\"\n")
    enc_unb = tok_unbalanced.encode(sample_sentence)
    enc_bal = tok_balanced.encode(sample_sentence)
    print(f"[غير متوازن] عدد الرموز: {len(enc_unb.ids)} -> {enc_unb.tokens}")
    print(f"[متوازن]     عدد الرموز: {len(enc_bal.ids)} -> {enc_bal.tokens}")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-05-build-an-llm/chapter-21.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ شائع جدًا عند المبتدئين: استخدام tokenizer جاهز دون فحص Fertility للعربية
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("gpt2")  # مُدرَّب حصريًا تقريبًا على الإنجليزية
tokens = tok.encode("الذكاء الاصطناعي رائع")
print(len(tokens))  # قد يكون 20+ رمزًا لجملة من 3 كلمات فقط!

# ✅ صحيح: افحص دائمًا Fertility قبل اعتماد أي Tokenizer جاهز لمشروعك
def quick_fertility_check(tokenizer, text: str) -> float:
    n_words = len(text.split())
    n_tokens = len(tokenizer.encode(text))
    return n_tokens / n_words

print(quick_fertility_check(tok, "الذكاء الاصطناعي رائع جدا"))  # إن كانت > 3، فكّر جديًا بإعادة التدريب

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-05-build-an-llm/chapter-21.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: اختيار حجم قاموس ضخم (256K) لنموذج صغير جدًا (125 مليون معامل)
# النتيجة: طبقة التضمين تستحوذ على نسبة غير متناسبة من إجمالي المعاملات
vocab_size = 256_000
d_model = 768
embedding_params = vocab_size * d_model  # ~196 مليون معامل!
total_model_params = 125_000_000
print(embedding_params / total_model_params)  # > 100%! غير منطقي إطلاقًا

# ✅ صحيح: اختر حجم القاموس بما يتناسب مع حجم النموذج الكلي
# (قاعدة تقريبية شائعة: نسبة معاملات التضمين يجب ألا تتجاوز 10-20% من الإجمالي)

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-05-build-an-llm/chapter-21.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: ترك BPE "يقرر" كيفية تفكيك الأرقام دون توجيه، مما ينتج تمثيلات غير متسقة
# "123" قد تُصبح رمزًا واحدًا، بينما "124" تتفكك لرمزين مختلفين تمامًا، مما يُربك
# قدرة النموذج على تعلّم الحساب والأنماط الرقمية بثبات

# ✅ صحيح: أضف قاعدة pre-tokenization تُفكّك كل رقم إلى آحاده
from tokenizers.pre_tokenizers import Digits, Sequence, ByteLevel

tokenizer.pre_tokenizer = Sequence([
    Digits(individual_digits=True),  # كل رقم منفصل كوحدة مستقلة
    ByteLevel(add_prefix_space=False),
])

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-05-build-an-llm/chapter-21.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: الاكتفاء بقياس Fertility منخفضة كدليل كافٍ على "نجاح" Tokenizer
# Fertility منخفضة ضرورية لكنها غير كافية وحدها لضمان جودة النموذج النهائي

# ✅ صحيح: درّب نموذجًا صغيرًا بكل Tokenizer مرشّح، وقارن منحنى الخسارة
# (Loss Curve) والأداء على معيار تقييم حقيقي (الفصل 46) قبل اعتماد القرار النهائي

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-05-build-an-llm/chapter-21.md
# ────────────────────────────────────────────────────────────────────────
"""
mini_project_production_tokenizer.py
بناء Tokenizer إنتاجي متوازن (32K رمز) على بيانات حقيقية من ويكيبيديا
العربية والإنجليزية، ومقارنة Fertility مع tokenizer جاهز (مثل GPT-2 أو LLaMA).

المطلوب إكماله:
1. حمّل عيّنتين من "wikimedia/wikipedia": "20231101.ar" و "20231101.en"
   (عبر streaming=True لتفادي تحميل كل البيانات).
2. طبّق استراتيجية Temperature Sampling (T=3) لموازنة نسبة اللغتين
   في بيانات تدريب Tokenizer (بضعة آلاف وثيقة من كل لغة تكفي للتجربة).
3. درّب BPE Tokenizer بحجم قاموس 32,000 باستخدام مكتبة `tokenizers`.
4. حمّل Tokenizer جاهزًا من Hugging Face Hub (مثل "gpt2" أو
   "meta-llama/Llama-2-7b-hf" إن توفّر الوصول) للمقارنة.
5. احسب Fertility وCompression Ratio لكليهما على عيّنة تقييم عربية
   وعيّنة تقييم إنجليزية منفصلتين تمامًا عن بيانات التدريب.
6. اطبع جدول مقارنة نهائي شامل.
7. (تحدٍّ إضافي) احفظ Tokenizer المُدرَّب بصيغة متوافقة مع
   Hugging Face `transformers` (PreTrainedTokenizerFast) لاستخدامه
   لاحقًا في الفصل 22 عند بناء نموذج كامل.
"""
from __future__ import annotations

from datasets import load_dataset
from tokenizers import Tokenizer


def load_balanced_sample(
    n_arabic: int = 5000, n_english: int = 5000
) -> list[str]:
    # TODO: استخدم load_dataset(..., streaming=True) لكلتا اللغتين
    # وخذ العدد المطلوب من كل لغة
    raise NotImplementedError


def main() -> None:
    corpus = load_balanced_sample()
    # TODO: درّب Tokenizer (استخدم الكود من هذا الفصل كأساس)
    # TODO: حمّل tokenizer جاهزًا من HF Hub للمقارنة
    # TODO: احسب واطبع جدول Fertility/Compression النهائي
    pass


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-21-chapter-21.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 21: الفصل الحادي والعشرون: تصميم Tokenizer مخصص على نطاق واسع."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_bpe() -> None:
    words = ['low', 'lowest', 'newer', 'wider']
    symbols = [list(word) + ['</w>'] for word in words]
    pairs = Counter((pair for word in symbols for pair in zip(word, word[1:])))
    pair, count = pairs.most_common(1)[0]
    merged = ''.join(pair)
    updated = [[merged if tuple(word[i:i + 2]) == pair else word[i] for i in range(len(word)) if i + 1 >= len(word) or tuple(word[i:i + 2]) != pair] for word in symbols]
    print('most frequent pair:', pair, 'count:', count)
    print('merged symbol:', merged)
    print('vocabulary size before/after:', sum(map(len, symbols)), sum(map(len, updated)))

if __name__ == "__main__":
    print("Chapter 21 lab: Learn byte-pair merges from a tiny corpus")
    run_bpe()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-21-chapter-21.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
chapter21_multilingual_tokenizer.py
تدريب Tokenizer BPE من الصفر، متوازن بين العربية والإنجليزية،
وقياس جودته عبر Fertility وCompression Ratio.

Python 3.13+
المتطلبات: pip install tokenizers datasets
"""
from __future__ import annotations

from dataclasses import dataclass

from tokenizers import Tokenizer, normalizers, pre_tokenizers, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel


# =========================================================================
# 1) بيانات تجريبية: مزيج عربي-إنجليزي (في الواقع: ملايين الوثائق من ف20)
# =========================================================================
ARABIC_CORPUS = [
    "الذكاء الاصطناعي يغير طريقة تفاعلنا مع التكنولوجيا بشكل جذري.",
    "تطورت النماذج اللغوية الكبيرة بسرعة مذهلة خلال السنوات الأخيرة.",
    "معالجة اللغة الطبيعية مجال واسع يشمل الترجمة والتلخيص والتصنيف.",
    "البيانات النظيفة عالية الجودة أهم من الكمية الهائلة من البيانات الفوضوية.",
    "يحتاج بناء نموذج لغوي ناجح إلى بنية تحتية حاسوبية قوية جدًا ومتينة.",
] * 200  # تكرار لمحاكاة حجم أكبر لأغراض العرض التوضيحي

ENGLISH_CORPUS = [
    "Artificial intelligence is transforming how we interact with technology.",
    "Large language models have evolved remarkably over the past few years.",
    "Natural language processing covers translation, summarization, and classification.",
    "Clean high-quality data matters more than raw volume of noisy data.",
    "Building a successful language model requires strong computing infrastructure.",
] * 200


@dataclass
class CorpusStats:
    name: str
    n_words: int
    n_tokens: int

    @property
    def fertility(self) -> float:
        return self.n_tokens / self.n_words if self.n_words else float("inf")


def build_balanced_training_iterator(
    arabic_docs: list[str], english_docs: list[str], arabic_boost: int = 3
) -> list[str]:
    """
    محاكاة بسيطة لـ Upsampling: نُكرر الوثائق العربية arabic_boost مرة
    لموازنة هيمنة الإنجليزية الطبيعية في بيانات الويب الحقيقية.
    """
    return arabic_docs * arabic_boost + english_docs


def train_bpe_tokenizer(
    training_corpus: list[str], vocab_size: int = 8000
) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    # التطبيع: NFKC لتوحيد أشكال الحروف المتكافئة يونيكوديًا
    tokenizer.normalizer = normalizers.Sequence([NFKC()])

    # ما قبل Tokenization: ByteLevel يضمن تمثيل أي حرف (بما فيه العربية) بلا <unk>
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>", "<mask>"]
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train_from_iterator(training_corpus, trainer=trainer)
    return tokenizer


def measure_fertility(
    tokenizer: Tokenizer, docs: list[str], language_name: str
) -> CorpusStats:
    total_words = 0
    total_tokens = 0
    for doc in docs:
        total_words += len(doc.split())
        encoding = tokenizer.encode(doc)
        total_tokens += len(encoding.ids)
    return CorpusStats(name=language_name, n_words=total_words, n_tokens=total_tokens)


def measure_compression_ratio(tokenizer: Tokenizer, docs: list[str]) -> float:
    """نسبة البايتات الخام إلى عدد الرموز الناتجة (أعلى = ضغط أفضل)."""
    total_bytes = sum(len(doc.encode("utf-8")) for doc in docs)
    total_tokens = sum(len(tokenizer.encode(doc).ids) for doc in docs)
    return total_bytes / total_tokens if total_tokens else 0.0


if __name__ == "__main__":
    # --- تجربة 1: Tokenizer غير متوازن (يحاكي الهيمنة الطبيعية للإنجليزية) ---
    print("=" * 70)
    print("تجربة 1: Tokenizer مُدرَّب على بيانات غير متوازنة (إنجليزية مهيمنة)")
    print("=" * 70)
    unbalanced_corpus = ARABIC_CORPUS[:50] + ENGLISH_CORPUS  # عربية قليلة جدًا
    tok_unbalanced = train_bpe_tokenizer(unbalanced_corpus, vocab_size=2000)

    stats_ar_unb = measure_fertility(tok_unbalanced, ARABIC_CORPUS[:100], "عربي")
    stats_en_unb = measure_fertility(tok_unbalanced, ENGLISH_CORPUS[:100], "إنجليزي")
    print(f"Fertility (عربي):    {stats_ar_unb.fertility:.2f} رمز/كلمة")
    print(f"Fertility (إنجليزي): {stats_en_unb.fertility:.2f} رمز/كلمة")
    print(f"الفجوة (عربي/إنجليزي): {stats_ar_unb.fertility / stats_en_unb.fertility:.2f}x\n")

    # --- تجربة 2: Tokenizer متوازن (عبر Upsampling للعربية) ---
    print("=" * 70)
    print("تجربة 2: Tokenizer مُدرَّب على بيانات مُعاد موازنتها (Upsampling)")
    print("=" * 70)
    balanced_corpus = build_balanced_training_iterator(
        ARABIC_CORPUS, ENGLISH_CORPUS, arabic_boost=3
    )
    tok_balanced = train_bpe_tokenizer(balanced_corpus, vocab_size=2000)

    stats_ar_bal = measure_fertility(tok_balanced, ARABIC_CORPUS[:100], "عربي")
    stats_en_bal = measure_fertility(tok_balanced, ENGLISH_CORPUS[:100], "إنجليزي")
    print(f"Fertility (عربي):    {stats_ar_bal.fertility:.2f} رمز/كلمة")
    print(f"Fertility (إنجليزي): {stats_en_bal.fertility:.2f} رمز/كلمة")
    print(f"الفجوة (عربي/إنجليزي): {stats_ar_bal.fertility / stats_en_bal.fertility:.2f}x\n")

    # --- مثال توضيحي: تفكيك جملة عربية فعليًا بكلا الـ Tokenizers ---
    sample_sentence = "الذكاء الاصطناعي يغير العالم"
    print("=" * 70)
    print(f"مثال تفكيك الجملة: \"{sample_sentence}\"\n")
    enc_unb = tok_unbalanced.encode(sample_sentence)
    enc_bal = tok_balanced.encode(sample_sentence)
    print(f"[غير متوازن] عدد الرموز: {len(enc_unb.ids)} -> {enc_unb.tokens}")
    print(f"[متوازن]     عدد الرموز: {len(enc_bal.ids)} -> {enc_bal.tokens}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-21-chapter-21.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ شائع جدًا عند المبتدئين: استخدام tokenizer جاهز دون فحص Fertility للعربية
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("gpt2")  # مُدرَّب حصريًا تقريبًا على الإنجليزية
tokens = tok.encode("الذكاء الاصطناعي رائع")
print(len(tokens))  # قد يكون 20+ رمزًا لجملة من 3 كلمات فقط!

# ✅ صحيح: افحص دائمًا Fertility قبل اعتماد أي Tokenizer جاهز لمشروعك
def quick_fertility_check(tokenizer, text: str) -> float:
    n_words = len(text.split())
    n_tokens = len(tokenizer.encode(text))
    return n_tokens / n_words

print(quick_fertility_check(tok, "الذكاء الاصطناعي رائع جدا"))  # إن كانت > 3، فكّر جديًا بإعادة التدريب

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-21-chapter-21.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: اختيار حجم قاموس ضخم (256K) لنموذج صغير جدًا (125 مليون معامل)
# النتيجة: طبقة التضمين تستحوذ على نسبة غير متناسبة من إجمالي المعاملات
vocab_size = 256_000
d_model = 768
embedding_params = vocab_size * d_model  # ~196 مليون معامل!
total_model_params = 125_000_000
print(embedding_params / total_model_params)  # > 100%! غير منطقي إطلاقًا

# ✅ صحيح: اختر حجم القاموس بما يتناسب مع حجم النموذج الكلي
# (قاعدة تقريبية شائعة: نسبة معاملات التضمين يجب ألا تتجاوز 10-20% من الإجمالي)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-21-chapter-21.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: ترك BPE "يقرر" كيفية تفكيك الأرقام دون توجيه، مما ينتج تمثيلات غير متسقة
# "123" قد تُصبح رمزًا واحدًا، بينما "124" تتفكك لرمزين مختلفين تمامًا، مما يُربك
# قدرة النموذج على تعلّم الحساب والأنماط الرقمية بثبات

# ✅ صحيح: أضف قاعدة pre-tokenization تُفكّك كل رقم إلى آحاده
from tokenizers.pre_tokenizers import Digits, Sequence, ByteLevel

tokenizer.pre_tokenizer = Sequence([
    Digits(individual_digits=True),  # كل رقم منفصل كوحدة مستقلة
    ByteLevel(add_prefix_space=False),
])

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-21-chapter-21.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: الاكتفاء بقياس Fertility منخفضة كدليل كافٍ على "نجاح" Tokenizer
# Fertility منخفضة ضرورية لكنها غير كافية وحدها لضمان جودة النموذج النهائي

# ✅ صحيح: درّب نموذجًا صغيرًا بكل Tokenizer مرشّح، وقارن منحنى الخسارة
# (Loss Curve) والأداء على معيار تقييم حقيقي (الفصل 46) قبل اعتماد القرار النهائي

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-21-chapter-21.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
"""
mini_project_production_tokenizer.py
بناء Tokenizer إنتاجي متوازن (32K رمز) على بيانات حقيقية من ويكيبيديا
العربية والإنجليزية، ومقارنة Fertility مع tokenizer جاهز (مثل GPT-2 أو LLaMA).

المطلوب إكماله:
1. حمّل عيّنتين من "wikimedia/wikipedia": "20231101.ar" و "20231101.en"
   (عبر streaming=True لتفادي تحميل كل البيانات).
2. طبّق استراتيجية Temperature Sampling (T=3) لموازنة نسبة اللغتين
   في بيانات تدريب Tokenizer (بضعة آلاف وثيقة من كل لغة تكفي للتجربة).
3. درّب BPE Tokenizer بحجم قاموس 32,000 باستخدام مكتبة `tokenizers`.
4. حمّل Tokenizer جاهزًا من Hugging Face Hub (مثل "gpt2" أو
   "meta-llama/Llama-2-7b-hf" إن توفّر الوصول) للمقارنة.
5. احسب Fertility وCompression Ratio لكليهما على عيّنة تقييم عربية
   وعيّنة تقييم إنجليزية منفصلتين تمامًا عن بيانات التدريب.
6. اطبع جدول مقارنة نهائي شامل.
7. (تحدٍّ إضافي) احفظ Tokenizer المُدرَّب بصيغة متوافقة مع
   Hugging Face `transformers` (PreTrainedTokenizerFast) لاستخدامه
   لاحقًا في الفصل 22 عند بناء نموذج كامل.
"""
from __future__ import annotations

from datasets import load_dataset
from tokenizers import Tokenizer


def load_balanced_sample(
    n_arabic: int = 5000, n_english: int = 5000
) -> list[str]:
    # TODO: استخدم load_dataset(..., streaming=True) لكلتا اللغتين
    # وخذ العدد المطلوب من كل لغة
    raise NotImplementedError


def main() -> None:
    corpus = load_balanced_sample()
    # TODO: درّب Tokenizer (استخدم الكود من هذا الفصل كأساس)
    # TODO: حمّل tokenizer جاهزًا من HF Hub للمقارنة
    # TODO: احسب واطبع جدول Fertility/Compression النهائي
    pass


if __name__ == "__main__":
    main()
