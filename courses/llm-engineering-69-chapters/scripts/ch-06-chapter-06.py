#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — Tokenization — كيف نُحوّل اللغة إلى أرقام؟
Large Language Model Engineering — Tokenization

Book slug : llm-engineering-69-chapters
Chapter   : ch-06-chapter-06 (lesson 6)
Code files: 21 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-06-chapter-06.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — Tokenization — كيف نُحوّل اللغة إلى أرقام؟
Large Language Model Engineering — Tokenization

Book slug : llm-engineering-69-chapters
Chapter   : ch-06-chapter-06 (lesson 6)
Code files: 20 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-06-chapter-06.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — Tokenization — كيف نُحوّل اللغة إلى أرقام؟
Large Language Model Engineering — Tokenization

Book slug : llm-engineering-69-chapters
Chapter   : ch-06-chapter-06 (lesson 6)
Code files: 19 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_06.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 06: الفصل السادس: Tokenization — كيف نُحوّل اللغة إلى أرقام؟."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_tokenizer() -> None:
    text = 'small models learn useful language patterns'
    vocabulary = {word: index for index, word in enumerate(sorted(set(text.split())))}
    encoded = [vocabulary[word] for word in text.split()]
    decoded = ' '.join((next((word for word, idx in vocabulary.items() if idx == token)) for token in encoded))
    print('vocabulary:', vocabulary)
    print('token ids:', encoded)
    print('round trip:', decoded == text)

if __name__ == "__main__":
    print("Chapter 06 lab: A reversible word tokenizer")
    run_tokenizer()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
text = "القطة تنام"
tokens = list(text)  # ['ا', 'ل', 'ق', 'ط', 'ة', ' ', 'ت', 'ن', 'ا', 'م']

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
text = "القطة تنام على السرير"
tokens = text.split()  # ['القطة', 'تنام', 'على', 'السرير']

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
text = "اللامركزية"
# Word-level: ['اللامركزية']  →  غالبًا [UNK] إن كانت نادرة!
# Subword:    ['ال', 'لا', 'مركزي', 'ة']  →  كل جزء مألوف، المعنى قابل للتركيب

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
"""
bpe_from_scratch.py
تطبيق كامل وواضح لخوارزمية BPE من الصفر، بدون أي مكتبات خارجية،
لفهم الآلية الداخلية حرفيًا قبل استخدام المكتبات الجاهزة.
Python 3.13+
"""
from __future__ import annotations

from collections import Counter, defaultdict


def get_word_frequencies(corpus: list[str]) -> dict[tuple[str, ...], int]:
    """
    يُحوّل كل كلمة إلى تسلسل من الأحرف + رمز خاص لنهاية الكلمة (</w>)،
    ويحسب تكرار كل كلمة في المجموعة.
    رمز </w> ضروري ليُميّز الـ Tokenizer بين "قطة" في وسط جملة و"قطة" في نهايتها
    (فقد يختلف المعنى السياقي أو النحوي، ويساعد على إعادة بناء النص لاحقًا).
    """
    freqs: Counter[tuple[str, ...]] = Counter()
    for sentence in corpus:
        for word in sentence.split():
            char_tuple = tuple(word) + ("</w>",)
            freqs[char_tuple] += 1
    return dict(freqs)


def get_pair_frequencies(word_freqs: dict[tuple[str, ...], int]) -> Counter[tuple[str, str]]:
    """يحسب تكرار كل زوج متجاور من الرموز عبر كل الكلمات، موزونًا بتكرار الكلمة."""
    pairs: Counter[tuple[str, str]] = Counter()
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs


def merge_pair(
    pair: tuple[str, str], word_freqs: dict[tuple[str, ...], int]
) -> dict[tuple[str, ...], int]:
    """يدمج كل ظهور لزوج معيّن في كل الكلمات، منتجًا مفردات جديدة."""
    new_word_freqs: dict[tuple[str, ...], int] = {}
    merged_token = pair[0] + pair[1]

    for word, freq in word_freqs.items():
        new_word: list[str] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(merged_token)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word_freqs[tuple(new_word)] = freq

    return new_word_freqs


def train_bpe(
    corpus: list[str], num_merges: int
) -> tuple[list[tuple[str, str]], set[str]]:
    """
    التدريب الكامل: يُعيد قائمة عمليات الدمج بترتيبها (ضرورية للترميز لاحقًا)
    ومجموعة المفردات النهائية.
    """
    word_freqs = get_word_frequencies(corpus)
    merges: list[tuple[str, str]] = []

    vocabulary: set[str] = set()
    for word in word_freqs:
        vocabulary.update(word)

    for step in range(num_merges):
        pair_freqs = get_pair_frequencies(word_freqs)
        if not pair_freqs:
            print(f"توقف مبكرًا عند الخطوة {step}: لا توجد أزواج متبقية للدمج")
            break

        best_pair = max(pair_freqs, key=lambda p: pair_freqs[p])
        word_freqs = merge_pair(best_pair, word_freqs)
        merges.append(best_pair)
        vocabulary.add(best_pair[0] + best_pair[1])

        print(f"الدمج {step + 1:02d}: {best_pair} → '{best_pair[0]+best_pair[1]}' "
              f"(تكرار: {pair_freqs[best_pair]})")

    return merges, vocabulary


def apply_bpe(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """يُطبّق عمليات الدمج المُتعلَّمة على كلمة جديدة، بنفس الترتيب الذي تعلّمناه."""
    tokens = list(word) + ["</w>"]

    for pair in merges:
        new_tokens: list[str] = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append(pair[0] + pair[1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

    return tokens


if __name__ == "__main__":
    corpus = [
        "القطة تحب اللعب",
        "القطة تحب النوم",
        "الكلب يحب اللعب",
        "الكلب يحب الجري",
        "القطة والكلب يلعبان معا",
    ]

    print("=== تدريب BPE على مجموعة نصوص صغيرة ===\n")
    merges, vocab = train_bpe(corpus, num_merges=15)

    print(f"\nحجم المفردات النهائي: {len(vocab)}")
    print(f"عدد عمليات الدمج: {len(merges)}")

    print("\n=== اختبار على كلمات جديدة (بعضها لم يظهر حرفيًا من قبل) ===")
    test_words = ["القطة", "الكلاب", "يلعب"]  # "الكلاب" و"يلعب" أشكال جديدة!
    for word in test_words:
        tokens = apply_bpe(word, merges)
        print(f"'{word}' → {tokens}")

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
"""
wordpiece_scoring.py
توضيح الفرق الجوهري بين معيار اختيار الدمج في BPE مقابل WordPiece.
"""
from __future__ import annotations

from collections import Counter


def bpe_score(pair_freq: int, *_: int) -> float:
    """BPE: يختار فقط بناءً على التكرار المطلق."""
    return float(pair_freq)


def wordpiece_score(pair_freq: int, freq_a: int, freq_b: int) -> float:
    """WordPiece: يُطبّع بالتكرار الفردي لكل عنصر (شبه PMI)."""
    return pair_freq / (freq_a * freq_b)


if __name__ == "__main__":
    # سيناريو توضيحي: زوج شائع جدًا لكن عشوائيًا (حروف شائعة تظهر مع كل شيء)
    # مقابل زوج أقل تكرارًا لكن "متماسك" لغويًا بقوة
    scenarios = {
        "('ة', ' ') - شائع جدًا لكن عشوائي": (5000, 50000, 80000),
        "('مر', 'كزي') - أقل تكرارًا لكن مترابط دائمًا": (800, 900, 850),
    }

    print(f"{'السيناريو':<45}{'BPE Score':<15}{'WordPiece Score':<15}")
    for name, (pair_f, freq_a, freq_b) in scenarios.items():
        bpe_s = bpe_score(pair_f)
        wp_s = wordpiece_score(pair_f, freq_a, freq_b)
        print(f"{name:<45}{bpe_s:<15.1f}{wp_s:<15.8f}")

    print("\nلاحظ: BPE يُفضّل الزوج الأول (تكرار خام أعلى)")
    print("بينما WordPiece يُفضّل الزوج الثاني (ترابط إحصائي أقوى نسبيًا)")

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
"""
train_production_tokenizer.py
تدريب BPE Tokenizer احترافي وسريع باستخدام مكتبة tokenizers من Hugging Face
(المكتوبة بـ Rust لسرعة فائقة)، جاهز للاستخدام في نموذج حقيقي.
المتطلبات: pip install tokenizers
"""
from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFC, Sequence as NormSequence


def build_and_train_tokenizer(
    corpus_files: list[str],
    vocab_size: int = 8000,
    min_frequency: int = 2,
) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # Normalization: NFC تُوحّد الأشكال المختلفة لنفس الحرف
    # (مهم جدًا للعربية: الألف المهموزة، الهمزات المنفصلة، إلخ)
    tokenizer.normalizer = NormSequence([NFC()])

    # Pre-tokenization: تقسيم أولي بسيط على المسافات قبل تطبيق BPE داخل كل كلمة
    tokenizer.pre_tokenizer = Whitespace()

    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"]
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    tokenizer.train(files=corpus_files, trainer=trainer)
    return tokenizer


if __name__ == "__main__":
    # إعداد ملف نصي تجريبي صغير للتدريب
    sample_corpus = Path("sample_arabic_corpus.txt")
    sample_text = """القطة تحب اللعب في الحديقة كل صباح
الكلب يحب الجري خلف الكرة في الحديقة
الأطفال يلعبون مع القطة والكلب معا كل يوم
تعلم البرمجة يحتاج صبرا وممارسة مستمرة
النماذج اللغوية الكبيرة تغير طريقة تفاعلنا مع التقنية
""" * 50
    sample_corpus.write_text(sample_text, encoding="utf-8")

    tokenizer = build_and_train_tokenizer(
        corpus_files=[str(sample_corpus)], vocab_size=500
    )

    # اختبار الترميز
    test_sentence = "الأطفال والقطط يلعبون في الحدائق الجميلة"
    encoding = tokenizer.encode(test_sentence)

    print(f"الجملة الأصلية: {test_sentence}")
    print(f"عدد التوكنز   : {len(encoding.tokens)}")
    print(f"التوكنز       : {encoding.tokens}")
    print(f"المعرّفات (IDs): {encoding.ids}")

    # فك الترميز للتأكد من إعادة البناء الصحيحة
    decoded = tokenizer.decode(encoding.ids)
    print(f"\nبعد فك الترميز: {decoded}")

    # حفظ الـ Tokenizer لإعادة استخدامه لاحقًا دون إعادة تدريب
    tokenizer.save("my_arabic_tokenizer.json")
    print("\n✅ تم حفظ الـ Tokenizer في my_arabic_tokenizer.json")

    sample_corpus.unlink()  # تنظيف الملف التجريبي

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
"""
tokenization_comparison.py
مقارنة عملية شاملة بين Character-Level, Word-Level, و BPE Tokenization
على نص عربي، مع قياس كفاءة كل استراتيجية.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenizationResult:
    strategy: str
    num_tokens: int
    vocab_size: int
    handles_oov: bool
    example_tokens: list[str]


def tokenize_char_level(text: str) -> list[str]:
    return list(text.replace(" ", "▁"))  # ▁ يُمثّل المسافة بوضوح (تقليد SentencePiece)


def tokenize_word_level(text: str, vocab: set[str]) -> list[str]:
    words = text.split()
    return [w if w in vocab else "[UNK]" for w in words]


def build_word_vocab(corpus: list[str]) -> set[str]:
    vocab: set[str] = set()
    for sentence in corpus:
        vocab.update(sentence.split())
    return vocab


def run_comparison(corpus: list[str], test_sentence: str) -> list[TokenizationResult]:
    results = []

    # --- Character-Level ---
    char_tokens = tokenize_char_level(test_sentence)
    char_vocab = set()
    for s in corpus:
        char_vocab.update(tokenize_char_level(s))
    results.append(TokenizationResult(
        strategy="Character-Level",
        num_tokens=len(char_tokens),
        vocab_size=len(char_vocab),
        handles_oov=True,  # لا توجد كلمة لا يمكن تمثيلها بأحرفها
        example_tokens=char_tokens[:10],
    ))

    # --- Word-Level ---
    word_vocab = build_word_vocab(corpus)
    word_tokens = tokenize_word_level(test_sentence, word_vocab)
    results.append(TokenizationResult(
        strategy="Word-Level",
        num_tokens=len(word_tokens),
        vocab_size=len(word_vocab),
        handles_oov="[UNK]" not in word_tokens,
        example_tokens=word_tokens,
    ))

    # --- BPE (باستخدام تطبيقنا اليدوي من القسم 3.2) ---
    from bpe_from_scratch import train_bpe, apply_bpe  # يفترض وجود الملف بنفس المجلد

    merges, bpe_vocab = train_bpe(corpus, num_merges=40)
    bpe_tokens: list[str] = []
    for word in test_sentence.split():
        bpe_tokens.extend(apply_bpe(word, merges))
    results.append(TokenizationResult(
        strategy="BPE (Subword)",
        num_tokens=len(bpe_tokens),
        vocab_size=len(bpe_vocab),
        handles_oov=True,  # قادر على تركيب أي كلمة من أجزاء معروفة
        example_tokens=bpe_tokens,
    ))

    return results


if __name__ == "__main__":
    corpus = [
        "القطة تحب اللعب في الحديقة",
        "الكلب يحب الجري في الحديقة",
        "الأطفال يحبون اللعب مع الحيوانات",
        "القراءة تفتح آفاقا جديدة للعقل",
        "البرمجة مهارة مهمة في هذا العصر",
    ]
    # جملة اختبار تحتوي كلمات مُركّبة/مُصرّفة لم تظهر حرفيًا من قبل
    test_sentence = "الأطفال والقطط يلعبان في الحدائق"

    results = run_comparison(corpus, test_sentence)

    print(f"\n{'الاستراتيجية':<20}{'عدد التوكنز':<15}{'حجم المفردات':<15}{'يتعامل مع OOV؟':<15}")
    print("-" * 65)
    for r in results:
        oov_mark = "✅" if r.handles_oov else "❌"
        print(f"{r.strategy:<20}{r.num_tokens:<15}{r.vocab_size:<15}{oov_mark:<15}")

    print("\n=== أمثلة التوكنز الناتجة ===")
    for r in results:
        print(f"\n{r.strategy}:")
        print(f"  {r.example_tokens}")

# ────────────────────────────────────────────────────────────────────────
# [009] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ فادح: تحميل نموذج مُدرَّب بـ tokenizer معيّن، واستخدام tokenizer آخر معه
model = load_pretrained_model("some-llm")
tokenizer = AutoTokenizer.from_pretrained("different-model-tokenizer")  # ❌!

# ────────────────────────────────────────────────────────────────────────
# [010] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: استخدم دائمًا نفس اسم/مسار الـ checkpoint لكل من النموذج والـ tokenizer
model_name = "some-llm"
model = load_pretrained_model(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)  # ✅ متطابق تمامًا

# ────────────────────────────────────────────────────────────────────────
# [011] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
# ❌ بدون Normalization، نفس الكلمة بأشكال Unicode مختلفة = توكنز مختلفة
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()
# لا يوجد normalizer! قد تتضاعف المفردات بلا داعٍ

# ────────────────────────────────────────────────────────────────────────
# [012] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: أضف NFC Normalization دائمًا كخطوة أولى
from tokenizers.normalizers import NFC
tokenizer.normalizer = NFC()
tokenizer.pre_tokenizer = Whitespace()

# ────────────────────────────────────────────────────────────────────────
# [013] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
# ❌ نص طويل جدًا قد يتجاوز الحد الأقصى الذي يدعمه النموذج دون أي تحذير
ids = tokenizer.encode(very_long_document).ids
model(ids)  # قد يفشل أو يُعطي نتائج سيئة جدًا إن تجاوز max_position_embeddings

# ────────────────────────────────────────────────────────────────────────
# [014] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: فعّل القص الصريح (truncation) مع تحديد الحد الأقصى بوضوح
encoding = tokenizer.encode(very_long_document)
ids = encoding.ids[:MAX_SEQ_LEN]  # أو استخدم truncation المدمجة في المكتبة

# ────────────────────────────────────────────────────────────────────────
# [015] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ شائع: افتراض أن عدد التوكنز ≈ عدد الكلمات (قد يكون مضاعفًا بسهولة!)
num_words = len(text.split())
estimated_cost = num_words * cost_per_token  # تقدير غير دقيق!

# ────────────────────────────────────────────────────────────────────────
# [016] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: احسب عدد التوكنز الفعلي دائمًا باستخدام نفس الـ tokenizer المُستهدف
actual_tokens = len(tokenizer.encode(text).ids)
estimated_cost = actual_tokens * cost_per_token  # دقيق، خاصة مهم في APIs تجارية

# ────────────────────────────────────────────────────────────────────────
# [017] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
# ❌ حساب Loss على رموز PAD يُضيف ضوضاء ويُحرّف النتيجة
loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))

# ────────────────────────────────────────────────────────────────────────
# [018] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: استخدم ignore_index لتجاهل توكنز PAD تمامًا في حساب الخسارة
loss = F.cross_entropy(
    logits.view(-1, vocab_size), labels.view(-1),
    ignore_index=tokenizer.pad_token_id,
)

# ────────────────────────────────────────────────────────────────────────
# [019] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
"""
fertility_analyzer.py — هيكل بداية للتمرين
أداة لقياس وتحليل "عدالة" أي Tokenizer عبر لغات مختلفة.
"""
from __future__ import annotations

from dataclasses import dataclass

from tokenizers import Tokenizer


@dataclass(frozen=True)
class FertilityReport:
    language: str
    num_words: int
    num_tokens: int
    fertility: float  # tokens / words، كلما قلّ كان أفضل (أكثر كفاءة)


def compute_fertility(
    tokenizer: Tokenizer, text: str, language_name: str
) -> FertilityReport:
    """
    TODO:
    1. احسب عدد الكلمات في text عبر .split() (تقريب بسيط كافٍ لهذا التمرين)
    2. رمّز النص الكامل بـ tokenizer.encode(text)، واحسب عدد التوكنز الناتجة
    3. احسب fertility = num_tokens / num_words
    4. أعد FertilityReport بالقيم المحسوبة
    """
    raise NotImplementedError


def compare_languages_fairness(
    tokenizer: Tokenizer,
    texts_by_language: dict[str, str],
) -> list[FertilityReport]:
    """
    TODO: طبّق compute_fertility على كل لغة في القاموس، وأعد قائمة بكل التقارير
    مُرتّبة تنازليًا حسب fertility (الأسوأ أولًا) لتسليط الضوء على اللغات
    "المظلومة" من هذا الـ Tokenizer تحديدًا.
    """
    raise NotImplementedError


def print_fairness_report(reports: list[FertilityReport]) -> None:
    """
    TODO: اطبع جدولاً منسّقًا يُقارن fertility بين اللغات، مع تحذير واضح
    (⚠️) لأي لغة تتجاوز fertility فيها ضعف أقل لغة في القائمة
    (مؤشر على عدم توازن واضح في تمثيل الـ Tokenizer لتلك اللغة).
    """
    raise NotImplementedError


if __name__ == "__main__":
    # نصوص متكافئة معنويًا تقريبًا بلغات مختلفة لمقارنة عادلة
    texts_by_language = {
        "العربية": "الذكاء الاصطناعي يغير العالم بسرعة كبيرة جدا في هذه الايام",
        "English": "Artificial intelligence is changing the world very quickly these days",
        "Français": "L'intelligence artificielle change le monde très rapidement ces jours-ci",
    }

    # TODO: حمّل tokenizer حقيقي (مثلاً GPT-2 أو أي نموذج مفتوح عبر Hugging Face)
    # from tokenizers import Tokenizer
    # tokenizer = Tokenizer.from_pretrained("gpt2")
    #
    # reports = compare_languages_fairness(tokenizer, texts_by_language)
    # print_fairness_report(reports)

    # سؤال للتفكير بعد إتمام التمرين:
    # هل النتائج التي حصلت عليها مفاجئة؟ أي اللغات كانت الأقل "عدالة"؟
    # كيف يمكن لمطوّري نموذج جديد معالجة هذه المشكلة عند بناء Tokenizer مخصص؟
    print("أكمل الدوال أعلاه وفعّل سطور التحميل، ثم شغّل التحليل!")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 06: الفصل السادس: Tokenization — كيف نُحوّل اللغة إلى أرقام؟."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_tokenizer() -> None:
    text = 'small models learn useful language patterns'
    vocabulary = {word: index for index, word in enumerate(sorted(set(text.split())))}
    encoded = [vocabulary[word] for word in text.split()]
    decoded = ' '.join((next((word for word, idx in vocabulary.items() if idx == token)) for token in encoded))
    print('vocabulary:', vocabulary)
    print('token ids:', encoded)
    print('round trip:', decoded == text)

if __name__ == "__main__":
    print("Chapter 06 lab: A reversible word tokenizer")
    run_tokenizer()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
text = "القطة تنام"
tokens = list(text)  # ['ا', 'ل', 'ق', 'ط', 'ة', ' ', 'ت', 'ن', 'ا', 'م']

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
text = "القطة تنام على السرير"
tokens = text.split()  # ['القطة', 'تنام', 'على', 'السرير']

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
text = "اللامركزية"
# Word-level: ['اللامركزية']  →  غالبًا [UNK] إن كانت نادرة!
# Subword:    ['ال', 'لا', 'مركزي', 'ة']  →  كل جزء مألوف، المعنى قابل للتركيب

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
"""
bpe_from_scratch.py
تطبيق كامل وواضح لخوارزمية BPE من الصفر، بدون أي مكتبات خارجية،
لفهم الآلية الداخلية حرفيًا قبل استخدام المكتبات الجاهزة.
Python 3.13+
"""
from __future__ import annotations

from collections import Counter, defaultdict


def get_word_frequencies(corpus: list[str]) -> dict[tuple[str, ...], int]:
    """
    يُحوّل كل كلمة إلى تسلسل من الأحرف + رمز خاص لنهاية الكلمة (</w>)،
    ويحسب تكرار كل كلمة في المجموعة.
    رمز </w> ضروري ليُميّز الـ Tokenizer بين "قطة" في وسط جملة و"قطة" في نهايتها
    (فقد يختلف المعنى السياقي أو النحوي، ويساعد على إعادة بناء النص لاحقًا).
    """
    freqs: Counter[tuple[str, ...]] = Counter()
    for sentence in corpus:
        for word in sentence.split():
            char_tuple = tuple(word) + ("</w>",)
            freqs[char_tuple] += 1
    return dict(freqs)


def get_pair_frequencies(word_freqs: dict[tuple[str, ...], int]) -> Counter[tuple[str, str]]:
    """يحسب تكرار كل زوج متجاور من الرموز عبر كل الكلمات، موزونًا بتكرار الكلمة."""
    pairs: Counter[tuple[str, str]] = Counter()
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs


def merge_pair(
    pair: tuple[str, str], word_freqs: dict[tuple[str, ...], int]
) -> dict[tuple[str, ...], int]:
    """يدمج كل ظهور لزوج معيّن في كل الكلمات، منتجًا مفردات جديدة."""
    new_word_freqs: dict[tuple[str, ...], int] = {}
    merged_token = pair[0] + pair[1]

    for word, freq in word_freqs.items():
        new_word: list[str] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(merged_token)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word_freqs[tuple(new_word)] = freq

    return new_word_freqs


def train_bpe(
    corpus: list[str], num_merges: int
) -> tuple[list[tuple[str, str]], set[str]]:
    """
    التدريب الكامل: يُعيد قائمة عمليات الدمج بترتيبها (ضرورية للترميز لاحقًا)
    ومجموعة المفردات النهائية.
    """
    word_freqs = get_word_frequencies(corpus)
    merges: list[tuple[str, str]] = []

    vocabulary: set[str] = set()
    for word in word_freqs:
        vocabulary.update(word)

    for step in range(num_merges):
        pair_freqs = get_pair_frequencies(word_freqs)
        if not pair_freqs:
            print(f"توقف مبكرًا عند الخطوة {step}: لا توجد أزواج متبقية للدمج")
            break

        best_pair = max(pair_freqs, key=lambda p: pair_freqs[p])
        word_freqs = merge_pair(best_pair, word_freqs)
        merges.append(best_pair)
        vocabulary.add(best_pair[0] + best_pair[1])

        print(f"الدمج {step + 1:02d}: {best_pair} → '{best_pair[0]+best_pair[1]}' "
              f"(تكرار: {pair_freqs[best_pair]})")

    return merges, vocabulary


def apply_bpe(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """يُطبّق عمليات الدمج المُتعلَّمة على كلمة جديدة، بنفس الترتيب الذي تعلّمناه."""
    tokens = list(word) + ["</w>"]

    for pair in merges:
        new_tokens: list[str] = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append(pair[0] + pair[1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

    return tokens


if __name__ == "__main__":
    corpus = [
        "القطة تحب اللعب",
        "القطة تحب النوم",
        "الكلب يحب اللعب",
        "الكلب يحب الجري",
        "القطة والكلب يلعبان معا",
    ]

    print("=== تدريب BPE على مجموعة نصوص صغيرة ===\n")
    merges, vocab = train_bpe(corpus, num_merges=15)

    print(f"\nحجم المفردات النهائي: {len(vocab)}")
    print(f"عدد عمليات الدمج: {len(merges)}")

    print("\n=== اختبار على كلمات جديدة (بعضها لم يظهر حرفيًا من قبل) ===")
    test_words = ["القطة", "الكلاب", "يلعب"]  # "الكلاب" و"يلعب" أشكال جديدة!
    for word in test_words:
        tokens = apply_bpe(word, merges)
        print(f"'{word}' → {tokens}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
"""
wordpiece_scoring.py
توضيح الفرق الجوهري بين معيار اختيار الدمج في BPE مقابل WordPiece.
"""
from __future__ import annotations

from collections import Counter


def bpe_score(pair_freq: int, *_: int) -> float:
    """BPE: يختار فقط بناءً على التكرار المطلق."""
    return float(pair_freq)


def wordpiece_score(pair_freq: int, freq_a: int, freq_b: int) -> float:
    """WordPiece: يُطبّع بالتكرار الفردي لكل عنصر (شبه PMI)."""
    return pair_freq / (freq_a * freq_b)


if __name__ == "__main__":
    # سيناريو توضيحي: زوج شائع جدًا لكن عشوائيًا (حروف شائعة تظهر مع كل شيء)
    # مقابل زوج أقل تكرارًا لكن "متماسك" لغويًا بقوة
    scenarios = {
        "('ة', ' ') - شائع جدًا لكن عشوائي": (5000, 50000, 80000),
        "('مر', 'كزي') - أقل تكرارًا لكن مترابط دائمًا": (800, 900, 850),
    }

    print(f"{'السيناريو':<45}{'BPE Score':<15}{'WordPiece Score':<15}")
    for name, (pair_f, freq_a, freq_b) in scenarios.items():
        bpe_s = bpe_score(pair_f)
        wp_s = wordpiece_score(pair_f, freq_a, freq_b)
        print(f"{name:<45}{bpe_s:<15.1f}{wp_s:<15.8f}")

    print("\nلاحظ: BPE يُفضّل الزوج الأول (تكرار خام أعلى)")
    print("بينما WordPiece يُفضّل الزوج الثاني (ترابط إحصائي أقوى نسبيًا)")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
"""
train_production_tokenizer.py
تدريب BPE Tokenizer احترافي وسريع باستخدام مكتبة tokenizers من Hugging Face
(المكتوبة بـ Rust لسرعة فائقة)، جاهز للاستخدام في نموذج حقيقي.
المتطلبات: pip install tokenizers
"""
from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFC, Sequence as NormSequence


def build_and_train_tokenizer(
    corpus_files: list[str],
    vocab_size: int = 8000,
    min_frequency: int = 2,
) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # Normalization: NFC تُوحّد الأشكال المختلفة لنفس الحرف
    # (مهم جدًا للعربية: الألف المهموزة، الهمزات المنفصلة، إلخ)
    tokenizer.normalizer = NormSequence([NFC()])

    # Pre-tokenization: تقسيم أولي بسيط على المسافات قبل تطبيق BPE داخل كل كلمة
    tokenizer.pre_tokenizer = Whitespace()

    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"]
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    tokenizer.train(files=corpus_files, trainer=trainer)
    return tokenizer


if __name__ == "__main__":
    # إعداد ملف نصي تجريبي صغير للتدريب
    sample_corpus = Path("sample_arabic_corpus.txt")
    sample_text = """القطة تحب اللعب في الحديقة كل صباح
الكلب يحب الجري خلف الكرة في الحديقة
الأطفال يلعبون مع القطة والكلب معا كل يوم
تعلم البرمجة يحتاج صبرا وممارسة مستمرة
النماذج اللغوية الكبيرة تغير طريقة تفاعلنا مع التقنية
""" * 50
    sample_corpus.write_text(sample_text, encoding="utf-8")

    tokenizer = build_and_train_tokenizer(
        corpus_files=[str(sample_corpus)], vocab_size=500
    )

    # اختبار الترميز
    test_sentence = "الأطفال والقطط يلعبون في الحدائق الجميلة"
    encoding = tokenizer.encode(test_sentence)

    print(f"الجملة الأصلية: {test_sentence}")
    print(f"عدد التوكنز   : {len(encoding.tokens)}")
    print(f"التوكنز       : {encoding.tokens}")
    print(f"المعرّفات (IDs): {encoding.ids}")

    # فك الترميز للتأكد من إعادة البناء الصحيحة
    decoded = tokenizer.decode(encoding.ids)
    print(f"\nبعد فك الترميز: {decoded}")

    # حفظ الـ Tokenizer لإعادة استخدامه لاحقًا دون إعادة تدريب
    tokenizer.save("my_arabic_tokenizer.json")
    print("\n✅ تم حفظ الـ Tokenizer في my_arabic_tokenizer.json")

    sample_corpus.unlink()  # تنظيف الملف التجريبي

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
"""
tokenization_comparison.py
مقارنة عملية شاملة بين Character-Level, Word-Level, و BPE Tokenization
على نص عربي، مع قياس كفاءة كل استراتيجية.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenizationResult:
    strategy: str
    num_tokens: int
    vocab_size: int
    handles_oov: bool
    example_tokens: list[str]


def tokenize_char_level(text: str) -> list[str]:
    return list(text.replace(" ", "▁"))  # ▁ يُمثّل المسافة بوضوح (تقليد SentencePiece)


def tokenize_word_level(text: str, vocab: set[str]) -> list[str]:
    words = text.split()
    return [w if w in vocab else "[UNK]" for w in words]


def build_word_vocab(corpus: list[str]) -> set[str]:
    vocab: set[str] = set()
    for sentence in corpus:
        vocab.update(sentence.split())
    return vocab


def run_comparison(corpus: list[str], test_sentence: str) -> list[TokenizationResult]:
    results = []

    # --- Character-Level ---
    char_tokens = tokenize_char_level(test_sentence)
    char_vocab = set()
    for s in corpus:
        char_vocab.update(tokenize_char_level(s))
    results.append(TokenizationResult(
        strategy="Character-Level",
        num_tokens=len(char_tokens),
        vocab_size=len(char_vocab),
        handles_oov=True,  # لا توجد كلمة لا يمكن تمثيلها بأحرفها
        example_tokens=char_tokens[:10],
    ))

    # --- Word-Level ---
    word_vocab = build_word_vocab(corpus)
    word_tokens = tokenize_word_level(test_sentence, word_vocab)
    results.append(TokenizationResult(
        strategy="Word-Level",
        num_tokens=len(word_tokens),
        vocab_size=len(word_vocab),
        handles_oov="[UNK]" not in word_tokens,
        example_tokens=word_tokens,
    ))

    # --- BPE (باستخدام تطبيقنا اليدوي من القسم 3.2) ---
    from bpe_from_scratch import train_bpe, apply_bpe  # يفترض وجود الملف بنفس المجلد

    merges, bpe_vocab = train_bpe(corpus, num_merges=40)
    bpe_tokens: list[str] = []
    for word in test_sentence.split():
        bpe_tokens.extend(apply_bpe(word, merges))
    results.append(TokenizationResult(
        strategy="BPE (Subword)",
        num_tokens=len(bpe_tokens),
        vocab_size=len(bpe_vocab),
        handles_oov=True,  # قادر على تركيب أي كلمة من أجزاء معروفة
        example_tokens=bpe_tokens,
    ))

    return results


if __name__ == "__main__":
    corpus = [
        "القطة تحب اللعب في الحديقة",
        "الكلب يحب الجري في الحديقة",
        "الأطفال يحبون اللعب مع الحيوانات",
        "القراءة تفتح آفاقا جديدة للعقل",
        "البرمجة مهارة مهمة في هذا العصر",
    ]
    # جملة اختبار تحتوي كلمات مُركّبة/مُصرّفة لم تظهر حرفيًا من قبل
    test_sentence = "الأطفال والقطط يلعبان في الحدائق"

    results = run_comparison(corpus, test_sentence)

    print(f"\n{'الاستراتيجية':<20}{'عدد التوكنز':<15}{'حجم المفردات':<15}{'يتعامل مع OOV؟':<15}")
    print("-" * 65)
    for r in results:
        oov_mark = "✅" if r.handles_oov else "❌"
        print(f"{r.strategy:<20}{r.num_tokens:<15}{r.vocab_size:<15}{oov_mark:<15}")

    print("\n=== أمثلة التوكنز الناتجة ===")
    for r in results:
        print(f"\n{r.strategy}:")
        print(f"  {r.example_tokens}")

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ فادح: تحميل نموذج مُدرَّب بـ tokenizer معيّن، واستخدام tokenizer آخر معه
model = load_pretrained_model("some-llm")
tokenizer = AutoTokenizer.from_pretrained("different-model-tokenizer")  # ❌!

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: استخدم دائمًا نفس اسم/مسار الـ checkpoint لكل من النموذج والـ tokenizer
model_name = "some-llm"
model = load_pretrained_model(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)  # ✅ متطابق تمامًا

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# ❌ بدون Normalization، نفس الكلمة بأشكال Unicode مختلفة = توكنز مختلفة
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()
# لا يوجد normalizer! قد تتضاعف المفردات بلا داعٍ

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: أضف NFC Normalization دائمًا كخطوة أولى
from tokenizers.normalizers import NFC
tokenizer.normalizer = NFC()
tokenizer.pre_tokenizer = Whitespace()

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
# ❌ نص طويل جدًا قد يتجاوز الحد الأقصى الذي يدعمه النموذج دون أي تحذير
ids = tokenizer.encode(very_long_document).ids
model(ids)  # قد يفشل أو يُعطي نتائج سيئة جدًا إن تجاوز max_position_embeddings

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: فعّل القص الصريح (truncation) مع تحديد الحد الأقصى بوضوح
encoding = tokenizer.encode(very_long_document)
ids = encoding.ids[:MAX_SEQ_LEN]  # أو استخدم truncation المدمجة في المكتبة

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ شائع: افتراض أن عدد التوكنز ≈ عدد الكلمات (قد يكون مضاعفًا بسهولة!)
num_words = len(text.split())
estimated_cost = num_words * cost_per_token  # تقدير غير دقيق!

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: احسب عدد التوكنز الفعلي دائمًا باستخدام نفس الـ tokenizer المُستهدف
actual_tokens = len(tokenizer.encode(text).ids)
estimated_cost = actual_tokens * cost_per_token  # دقيق، خاصة مهم في APIs تجارية

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
# ❌ حساب Loss على رموز PAD يُضيف ضوضاء ويُحرّف النتيجة
loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: استخدم ignore_index لتجاهل توكنز PAD تمامًا في حساب الخسارة
loss = F.cross_entropy(
    logits.view(-1, vocab_size), labels.view(-1),
    ignore_index=tokenizer.pad_token_id,
)

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
"""
fertility_analyzer.py — هيكل بداية للتمرين
أداة لقياس وتحليل "عدالة" أي Tokenizer عبر لغات مختلفة.
"""
from __future__ import annotations

from dataclasses import dataclass

from tokenizers import Tokenizer


@dataclass(frozen=True)
class FertilityReport:
    language: str
    num_words: int
    num_tokens: int
    fertility: float  # tokens / words، كلما قلّ كان أفضل (أكثر كفاءة)


def compute_fertility(
    tokenizer: Tokenizer, text: str, language_name: str
) -> FertilityReport:
    """
    TODO:
    1. احسب عدد الكلمات في text عبر .split() (تقريب بسيط كافٍ لهذا التمرين)
    2. رمّز النص الكامل بـ tokenizer.encode(text)، واحسب عدد التوكنز الناتجة
    3. احسب fertility = num_tokens / num_words
    4. أعد FertilityReport بالقيم المحسوبة
    """
    raise NotImplementedError


def compare_languages_fairness(
    tokenizer: Tokenizer,
    texts_by_language: dict[str, str],
) -> list[FertilityReport]:
    """
    TODO: طبّق compute_fertility على كل لغة في القاموس، وأعد قائمة بكل التقارير
    مُرتّبة تنازليًا حسب fertility (الأسوأ أولًا) لتسليط الضوء على اللغات
    "المظلومة" من هذا الـ Tokenizer تحديدًا.
    """
    raise NotImplementedError


def print_fairness_report(reports: list[FertilityReport]) -> None:
    """
    TODO: اطبع جدولاً منسّقًا يُقارن fertility بين اللغات، مع تحذير واضح
    (⚠️) لأي لغة تتجاوز fertility فيها ضعف أقل لغة في القائمة
    (مؤشر على عدم توازن واضح في تمثيل الـ Tokenizer لتلك اللغة).
    """
    raise NotImplementedError


if __name__ == "__main__":
    # نصوص متكافئة معنويًا تقريبًا بلغات مختلفة لمقارنة عادلة
    texts_by_language = {
        "العربية": "الذكاء الاصطناعي يغير العالم بسرعة كبيرة جدا في هذه الايام",
        "English": "Artificial intelligence is changing the world very quickly these days",
        "Français": "L'intelligence artificielle change le monde très rapidement ces jours-ci",
    }

    # TODO: حمّل tokenizer حقيقي (مثلاً GPT-2 أو أي نموذج مفتوح عبر Hugging Face)
    # from tokenizers import Tokenizer
    # tokenizer = Tokenizer.from_pretrained("gpt2")
    #
    # reports = compare_languages_fairness(tokenizer, texts_by_language)
    # print_fairness_report(reports)

    # سؤال للتفكير بعد إتمام التمرين:
    # هل النتائج التي حصلت عليها مفاجئة؟ أي اللغات كانت الأقل "عدالة"؟
    # كيف يمكن لمطوّري نموذج جديد معالجة هذه المشكلة عند بناء Tokenizer مخصص؟
    print("أكمل الدوال أعلاه وفعّل سطور التحميل، ثم شغّل التحليل!")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — Tokenization — كيف نُحوّل اللغة إلى أرقام؟
Large Language Model Engineering — Tokenization

Book slug : llm-engineering-69-chapters
Chapter   : ch-06-chapter-06 (lesson 6)
Code files: 19 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_06.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 06: الفصل السادس: Tokenization — كيف نُحوّل اللغة إلى أرقام؟."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_tokenizer() -> None:
    text = 'small models learn useful language patterns'
    vocabulary = {word: index for index, word in enumerate(sorted(set(text.split())))}
    encoded = [vocabulary[word] for word in text.split()]
    decoded = ' '.join((next((word for word, idx in vocabulary.items() if idx == token)) for token in encoded))
    print('vocabulary:', vocabulary)
    print('token ids:', encoded)
    print('round trip:', decoded == text)

if __name__ == "__main__":
    print("Chapter 06 lab: A reversible word tokenizer")
    run_tokenizer()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
text = "القطة تنام"
tokens = list(text)  # ['ا', 'ل', 'ق', 'ط', 'ة', ' ', 'ت', 'ن', 'ا', 'م']

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
text = "القطة تنام على السرير"
tokens = text.split()  # ['القطة', 'تنام', 'على', 'السرير']

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
text = "اللامركزية"
# Word-level: ['اللامركزية']  →  غالبًا [UNK] إن كانت نادرة!
# Subword:    ['ال', 'لا', 'مركزي', 'ة']  →  كل جزء مألوف، المعنى قابل للتركيب

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
"""
bpe_from_scratch.py
تطبيق كامل وواضح لخوارزمية BPE من الصفر، بدون أي مكتبات خارجية،
لفهم الآلية الداخلية حرفيًا قبل استخدام المكتبات الجاهزة.
Python 3.13+
"""
from __future__ import annotations

from collections import Counter, defaultdict


def get_word_frequencies(corpus: list[str]) -> dict[tuple[str, ...], int]:
    """
    يُحوّل كل كلمة إلى تسلسل من الأحرف + رمز خاص لنهاية الكلمة (</w>)،
    ويحسب تكرار كل كلمة في المجموعة.
    رمز </w> ضروري ليُميّز الـ Tokenizer بين "قطة" في وسط جملة و"قطة" في نهايتها
    (فقد يختلف المعنى السياقي أو النحوي، ويساعد على إعادة بناء النص لاحقًا).
    """
    freqs: Counter[tuple[str, ...]] = Counter()
    for sentence in corpus:
        for word in sentence.split():
            char_tuple = tuple(word) + ("</w>",)
            freqs[char_tuple] += 1
    return dict(freqs)


def get_pair_frequencies(word_freqs: dict[tuple[str, ...], int]) -> Counter[tuple[str, str]]:
    """يحسب تكرار كل زوج متجاور من الرموز عبر كل الكلمات، موزونًا بتكرار الكلمة."""
    pairs: Counter[tuple[str, str]] = Counter()
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs


def merge_pair(
    pair: tuple[str, str], word_freqs: dict[tuple[str, ...], int]
) -> dict[tuple[str, ...], int]:
    """يدمج كل ظهور لزوج معيّن في كل الكلمات، منتجًا مفردات جديدة."""
    new_word_freqs: dict[tuple[str, ...], int] = {}
    merged_token = pair[0] + pair[1]

    for word, freq in word_freqs.items():
        new_word: list[str] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(merged_token)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word_freqs[tuple(new_word)] = freq

    return new_word_freqs


def train_bpe(
    corpus: list[str], num_merges: int
) -> tuple[list[tuple[str, str]], set[str]]:
    """
    التدريب الكامل: يُعيد قائمة عمليات الدمج بترتيبها (ضرورية للترميز لاحقًا)
    ومجموعة المفردات النهائية.
    """
    word_freqs = get_word_frequencies(corpus)
    merges: list[tuple[str, str]] = []

    vocabulary: set[str] = set()
    for word in word_freqs:
        vocabulary.update(word)

    for step in range(num_merges):
        pair_freqs = get_pair_frequencies(word_freqs)
        if not pair_freqs:
            print(f"توقف مبكرًا عند الخطوة {step}: لا توجد أزواج متبقية للدمج")
            break

        best_pair = max(pair_freqs, key=lambda p: pair_freqs[p])
        word_freqs = merge_pair(best_pair, word_freqs)
        merges.append(best_pair)
        vocabulary.add(best_pair[0] + best_pair[1])

        print(f"الدمج {step + 1:02d}: {best_pair} → '{best_pair[0]+best_pair[1]}' "
              f"(تكرار: {pair_freqs[best_pair]})")

    return merges, vocabulary


def apply_bpe(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """يُطبّق عمليات الدمج المُتعلَّمة على كلمة جديدة، بنفس الترتيب الذي تعلّمناه."""
    tokens = list(word) + ["</w>"]

    for pair in merges:
        new_tokens: list[str] = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append(pair[0] + pair[1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

    return tokens


if __name__ == "__main__":
    corpus = [
        "القطة تحب اللعب",
        "القطة تحب النوم",
        "الكلب يحب اللعب",
        "الكلب يحب الجري",
        "القطة والكلب يلعبان معا",
    ]

    print("=== تدريب BPE على مجموعة نصوص صغيرة ===\n")
    merges, vocab = train_bpe(corpus, num_merges=15)

    print(f"\nحجم المفردات النهائي: {len(vocab)}")
    print(f"عدد عمليات الدمج: {len(merges)}")

    print("\n=== اختبار على كلمات جديدة (بعضها لم يظهر حرفيًا من قبل) ===")
    test_words = ["القطة", "الكلاب", "يلعب"]  # "الكلاب" و"يلعب" أشكال جديدة!
    for word in test_words:
        tokens = apply_bpe(word, merges)
        print(f"'{word}' → {tokens}")

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
"""
wordpiece_scoring.py
توضيح الفرق الجوهري بين معيار اختيار الدمج في BPE مقابل WordPiece.
"""
from __future__ import annotations

from collections import Counter


def bpe_score(pair_freq: int, *_: int) -> float:
    """BPE: يختار فقط بناءً على التكرار المطلق."""
    return float(pair_freq)


def wordpiece_score(pair_freq: int, freq_a: int, freq_b: int) -> float:
    """WordPiece: يُطبّع بالتكرار الفردي لكل عنصر (شبه PMI)."""
    return pair_freq / (freq_a * freq_b)


if __name__ == "__main__":
    # سيناريو توضيحي: زوج شائع جدًا لكن عشوائيًا (حروف شائعة تظهر مع كل شيء)
    # مقابل زوج أقل تكرارًا لكن "متماسك" لغويًا بقوة
    scenarios = {
        "('ة', ' ') - شائع جدًا لكن عشوائي": (5000, 50000, 80000),
        "('مر', 'كزي') - أقل تكرارًا لكن مترابط دائمًا": (800, 900, 850),
    }

    print(f"{'السيناريو':<45}{'BPE Score':<15}{'WordPiece Score':<15}")
    for name, (pair_f, freq_a, freq_b) in scenarios.items():
        bpe_s = bpe_score(pair_f)
        wp_s = wordpiece_score(pair_f, freq_a, freq_b)
        print(f"{name:<45}{bpe_s:<15.1f}{wp_s:<15.8f}")

    print("\nلاحظ: BPE يُفضّل الزوج الأول (تكرار خام أعلى)")
    print("بينما WordPiece يُفضّل الزوج الثاني (ترابط إحصائي أقوى نسبيًا)")

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
"""
train_production_tokenizer.py
تدريب BPE Tokenizer احترافي وسريع باستخدام مكتبة tokenizers من Hugging Face
(المكتوبة بـ Rust لسرعة فائقة)، جاهز للاستخدام في نموذج حقيقي.
المتطلبات: pip install tokenizers
"""
from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFC, Sequence as NormSequence


def build_and_train_tokenizer(
    corpus_files: list[str],
    vocab_size: int = 8000,
    min_frequency: int = 2,
) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # Normalization: NFC تُوحّد الأشكال المختلفة لنفس الحرف
    # (مهم جدًا للعربية: الألف المهموزة، الهمزات المنفصلة، إلخ)
    tokenizer.normalizer = NormSequence([NFC()])

    # Pre-tokenization: تقسيم أولي بسيط على المسافات قبل تطبيق BPE داخل كل كلمة
    tokenizer.pre_tokenizer = Whitespace()

    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"]
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    tokenizer.train(files=corpus_files, trainer=trainer)
    return tokenizer


if __name__ == "__main__":
    # إعداد ملف نصي تجريبي صغير للتدريب
    sample_corpus = Path("sample_arabic_corpus.txt")
    sample_text = """القطة تحب اللعب في الحديقة كل صباح
الكلب يحب الجري خلف الكرة في الحديقة
الأطفال يلعبون مع القطة والكلب معا كل يوم
تعلم البرمجة يحتاج صبرا وممارسة مستمرة
النماذج اللغوية الكبيرة تغير طريقة تفاعلنا مع التقنية
""" * 50
    sample_corpus.write_text(sample_text, encoding="utf-8")

    tokenizer = build_and_train_tokenizer(
        corpus_files=[str(sample_corpus)], vocab_size=500
    )

    # اختبار الترميز
    test_sentence = "الأطفال والقطط يلعبون في الحدائق الجميلة"
    encoding = tokenizer.encode(test_sentence)

    print(f"الجملة الأصلية: {test_sentence}")
    print(f"عدد التوكنز   : {len(encoding.tokens)}")
    print(f"التوكنز       : {encoding.tokens}")
    print(f"المعرّفات (IDs): {encoding.ids}")

    # فك الترميز للتأكد من إعادة البناء الصحيحة
    decoded = tokenizer.decode(encoding.ids)
    print(f"\nبعد فك الترميز: {decoded}")

    # حفظ الـ Tokenizer لإعادة استخدامه لاحقًا دون إعادة تدريب
    tokenizer.save("my_arabic_tokenizer.json")
    print("\n✅ تم حفظ الـ Tokenizer في my_arabic_tokenizer.json")

    sample_corpus.unlink()  # تنظيف الملف التجريبي

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
"""
tokenization_comparison.py
مقارنة عملية شاملة بين Character-Level, Word-Level, و BPE Tokenization
على نص عربي، مع قياس كفاءة كل استراتيجية.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenizationResult:
    strategy: str
    num_tokens: int
    vocab_size: int
    handles_oov: bool
    example_tokens: list[str]


def tokenize_char_level(text: str) -> list[str]:
    return list(text.replace(" ", "▁"))  # ▁ يُمثّل المسافة بوضوح (تقليد SentencePiece)


def tokenize_word_level(text: str, vocab: set[str]) -> list[str]:
    words = text.split()
    return [w if w in vocab else "[UNK]" for w in words]


def build_word_vocab(corpus: list[str]) -> set[str]:
    vocab: set[str] = set()
    for sentence in corpus:
        vocab.update(sentence.split())
    return vocab


def run_comparison(corpus: list[str], test_sentence: str) -> list[TokenizationResult]:
    results = []

    # --- Character-Level ---
    char_tokens = tokenize_char_level(test_sentence)
    char_vocab = set()
    for s in corpus:
        char_vocab.update(tokenize_char_level(s))
    results.append(TokenizationResult(
        strategy="Character-Level",
        num_tokens=len(char_tokens),
        vocab_size=len(char_vocab),
        handles_oov=True,  # لا توجد كلمة لا يمكن تمثيلها بأحرفها
        example_tokens=char_tokens[:10],
    ))

    # --- Word-Level ---
    word_vocab = build_word_vocab(corpus)
    word_tokens = tokenize_word_level(test_sentence, word_vocab)
    results.append(TokenizationResult(
        strategy="Word-Level",
        num_tokens=len(word_tokens),
        vocab_size=len(word_vocab),
        handles_oov="[UNK]" not in word_tokens,
        example_tokens=word_tokens,
    ))

    # --- BPE (باستخدام تطبيقنا اليدوي من القسم 3.2) ---
    from bpe_from_scratch import train_bpe, apply_bpe  # يفترض وجود الملف بنفس المجلد

    merges, bpe_vocab = train_bpe(corpus, num_merges=40)
    bpe_tokens: list[str] = []
    for word in test_sentence.split():
        bpe_tokens.extend(apply_bpe(word, merges))
    results.append(TokenizationResult(
        strategy="BPE (Subword)",
        num_tokens=len(bpe_tokens),
        vocab_size=len(bpe_vocab),
        handles_oov=True,  # قادر على تركيب أي كلمة من أجزاء معروفة
        example_tokens=bpe_tokens,
    ))

    return results


if __name__ == "__main__":
    corpus = [
        "القطة تحب اللعب في الحديقة",
        "الكلب يحب الجري في الحديقة",
        "الأطفال يحبون اللعب مع الحيوانات",
        "القراءة تفتح آفاقا جديدة للعقل",
        "البرمجة مهارة مهمة في هذا العصر",
    ]
    # جملة اختبار تحتوي كلمات مُركّبة/مُصرّفة لم تظهر حرفيًا من قبل
    test_sentence = "الأطفال والقطط يلعبان في الحدائق"

    results = run_comparison(corpus, test_sentence)

    print(f"\n{'الاستراتيجية':<20}{'عدد التوكنز':<15}{'حجم المفردات':<15}{'يتعامل مع OOV؟':<15}")
    print("-" * 65)
    for r in results:
        oov_mark = "✅" if r.handles_oov else "❌"
        print(f"{r.strategy:<20}{r.num_tokens:<15}{r.vocab_size:<15}{oov_mark:<15}")

    print("\n=== أمثلة التوكنز الناتجة ===")
    for r in results:
        print(f"\n{r.strategy}:")
        print(f"  {r.example_tokens}")

# ────────────────────────────────────────────────────────────────────────
# [009] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ فادح: تحميل نموذج مُدرَّب بـ tokenizer معيّن، واستخدام tokenizer آخر معه
model = load_pretrained_model("some-llm")
tokenizer = AutoTokenizer.from_pretrained("different-model-tokenizer")  # ❌!

# ────────────────────────────────────────────────────────────────────────
# [010] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: استخدم دائمًا نفس اسم/مسار الـ checkpoint لكل من النموذج والـ tokenizer
model_name = "some-llm"
model = load_pretrained_model(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)  # ✅ متطابق تمامًا

# ────────────────────────────────────────────────────────────────────────
# [011] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
# ❌ بدون Normalization، نفس الكلمة بأشكال Unicode مختلفة = توكنز مختلفة
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()
# لا يوجد normalizer! قد تتضاعف المفردات بلا داعٍ

# ────────────────────────────────────────────────────────────────────────
# [012] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: أضف NFC Normalization دائمًا كخطوة أولى
from tokenizers.normalizers import NFC
tokenizer.normalizer = NFC()
tokenizer.pre_tokenizer = Whitespace()

# ────────────────────────────────────────────────────────────────────────
# [013] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
# ❌ نص طويل جدًا قد يتجاوز الحد الأقصى الذي يدعمه النموذج دون أي تحذير
ids = tokenizer.encode(very_long_document).ids
model(ids)  # قد يفشل أو يُعطي نتائج سيئة جدًا إن تجاوز max_position_embeddings

# ────────────────────────────────────────────────────────────────────────
# [014] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: فعّل القص الصريح (truncation) مع تحديد الحد الأقصى بوضوح
encoding = tokenizer.encode(very_long_document)
ids = encoding.ids[:MAX_SEQ_LEN]  # أو استخدم truncation المدمجة في المكتبة

# ────────────────────────────────────────────────────────────────────────
# [015] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ شائع: افتراض أن عدد التوكنز ≈ عدد الكلمات (قد يكون مضاعفًا بسهولة!)
num_words = len(text.split())
estimated_cost = num_words * cost_per_token  # تقدير غير دقيق!

# ────────────────────────────────────────────────────────────────────────
# [016] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: احسب عدد التوكنز الفعلي دائمًا باستخدام نفس الـ tokenizer المُستهدف
actual_tokens = len(tokenizer.encode(text).ids)
estimated_cost = actual_tokens * cost_per_token  # دقيق، خاصة مهم في APIs تجارية

# ────────────────────────────────────────────────────────────────────────
# [017] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
# ❌ حساب Loss على رموز PAD يُضيف ضوضاء ويُحرّف النتيجة
loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))

# ────────────────────────────────────────────────────────────────────────
# [018] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: استخدم ignore_index لتجاهل توكنز PAD تمامًا في حساب الخسارة
loss = F.cross_entropy(
    logits.view(-1, vocab_size), labels.view(-1),
    ignore_index=tokenizer.pad_token_id,
)

# ────────────────────────────────────────────────────────────────────────
# [019] book/chapters/part-02-embeddings/chapter-06.md
# ────────────────────────────────────────────────────────────────────────
"""
fertility_analyzer.py — هيكل بداية للتمرين
أداة لقياس وتحليل "عدالة" أي Tokenizer عبر لغات مختلفة.
"""
from __future__ import annotations

from dataclasses import dataclass

from tokenizers import Tokenizer


@dataclass(frozen=True)
class FertilityReport:
    language: str
    num_words: int
    num_tokens: int
    fertility: float  # tokens / words، كلما قلّ كان أفضل (أكثر كفاءة)


def compute_fertility(
    tokenizer: Tokenizer, text: str, language_name: str
) -> FertilityReport:
    """
    TODO:
    1. احسب عدد الكلمات في text عبر .split() (تقريب بسيط كافٍ لهذا التمرين)
    2. رمّز النص الكامل بـ tokenizer.encode(text)، واحسب عدد التوكنز الناتجة
    3. احسب fertility = num_tokens / num_words
    4. أعد FertilityReport بالقيم المحسوبة
    """
    raise NotImplementedError


def compare_languages_fairness(
    tokenizer: Tokenizer,
    texts_by_language: dict[str, str],
) -> list[FertilityReport]:
    """
    TODO: طبّق compute_fertility على كل لغة في القاموس، وأعد قائمة بكل التقارير
    مُرتّبة تنازليًا حسب fertility (الأسوأ أولًا) لتسليط الضوء على اللغات
    "المظلومة" من هذا الـ Tokenizer تحديدًا.
    """
    raise NotImplementedError


def print_fairness_report(reports: list[FertilityReport]) -> None:
    """
    TODO: اطبع جدولاً منسّقًا يُقارن fertility بين اللغات، مع تحذير واضح
    (⚠️) لأي لغة تتجاوز fertility فيها ضعف أقل لغة في القائمة
    (مؤشر على عدم توازن واضح في تمثيل الـ Tokenizer لتلك اللغة).
    """
    raise NotImplementedError


if __name__ == "__main__":
    # نصوص متكافئة معنويًا تقريبًا بلغات مختلفة لمقارنة عادلة
    texts_by_language = {
        "العربية": "الذكاء الاصطناعي يغير العالم بسرعة كبيرة جدا في هذه الايام",
        "English": "Artificial intelligence is changing the world very quickly these days",
        "Français": "L'intelligence artificielle change le monde très rapidement ces jours-ci",
    }

    # TODO: حمّل tokenizer حقيقي (مثلاً GPT-2 أو أي نموذج مفتوح عبر Hugging Face)
    # from tokenizers import Tokenizer
    # tokenizer = Tokenizer.from_pretrained("gpt2")
    #
    # reports = compare_languages_fairness(tokenizer, texts_by_language)
    # print_fairness_report(reports)

    # سؤال للتفكير بعد إتمام التمرين:
    # هل النتائج التي حصلت عليها مفاجئة؟ أي اللغات كانت الأقل "عدالة"؟
    # كيف يمكن لمطوّري نموذج جديد معالجة هذه المشكلة عند بناء Tokenizer مخصص؟
    print("أكمل الدوال أعلاه وفعّل سطور التحميل، ثم شغّل التحليل!")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 06: الفصل السادس: Tokenization — كيف نُحوّل اللغة إلى أرقام؟."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_tokenizer() -> None:
    text = 'small models learn useful language patterns'
    vocabulary = {word: index for index, word in enumerate(sorted(set(text.split())))}
    encoded = [vocabulary[word] for word in text.split()]
    decoded = ' '.join((next((word for word, idx in vocabulary.items() if idx == token)) for token in encoded))
    print('vocabulary:', vocabulary)
    print('token ids:', encoded)
    print('round trip:', decoded == text)

if __name__ == "__main__":
    print("Chapter 06 lab: A reversible word tokenizer")
    run_tokenizer()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
text = "القطة تنام"
tokens = list(text)  # ['ا', 'ل', 'ق', 'ط', 'ة', ' ', 'ت', 'ن', 'ا', 'م']

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
text = "القطة تنام على السرير"
tokens = text.split()  # ['القطة', 'تنام', 'على', 'السرير']

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
text = "اللامركزية"
# Word-level: ['اللامركزية']  →  غالبًا [UNK] إن كانت نادرة!
# Subword:    ['ال', 'لا', 'مركزي', 'ة']  →  كل جزء مألوف، المعنى قابل للتركيب

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
"""
bpe_from_scratch.py
تطبيق كامل وواضح لخوارزمية BPE من الصفر، بدون أي مكتبات خارجية،
لفهم الآلية الداخلية حرفيًا قبل استخدام المكتبات الجاهزة.
Python 3.13+
"""
from __future__ import annotations

from collections import Counter, defaultdict


def get_word_frequencies(corpus: list[str]) -> dict[tuple[str, ...], int]:
    """
    يُحوّل كل كلمة إلى تسلسل من الأحرف + رمز خاص لنهاية الكلمة (</w>)،
    ويحسب تكرار كل كلمة في المجموعة.
    رمز </w> ضروري ليُميّز الـ Tokenizer بين "قطة" في وسط جملة و"قطة" في نهايتها
    (فقد يختلف المعنى السياقي أو النحوي، ويساعد على إعادة بناء النص لاحقًا).
    """
    freqs: Counter[tuple[str, ...]] = Counter()
    for sentence in corpus:
        for word in sentence.split():
            char_tuple = tuple(word) + ("</w>",)
            freqs[char_tuple] += 1
    return dict(freqs)


def get_pair_frequencies(word_freqs: dict[tuple[str, ...], int]) -> Counter[tuple[str, str]]:
    """يحسب تكرار كل زوج متجاور من الرموز عبر كل الكلمات، موزونًا بتكرار الكلمة."""
    pairs: Counter[tuple[str, str]] = Counter()
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs


def merge_pair(
    pair: tuple[str, str], word_freqs: dict[tuple[str, ...], int]
) -> dict[tuple[str, ...], int]:
    """يدمج كل ظهور لزوج معيّن في كل الكلمات، منتجًا مفردات جديدة."""
    new_word_freqs: dict[tuple[str, ...], int] = {}
    merged_token = pair[0] + pair[1]

    for word, freq in word_freqs.items():
        new_word: list[str] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(merged_token)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word_freqs[tuple(new_word)] = freq

    return new_word_freqs


def train_bpe(
    corpus: list[str], num_merges: int
) -> tuple[list[tuple[str, str]], set[str]]:
    """
    التدريب الكامل: يُعيد قائمة عمليات الدمج بترتيبها (ضرورية للترميز لاحقًا)
    ومجموعة المفردات النهائية.
    """
    word_freqs = get_word_frequencies(corpus)
    merges: list[tuple[str, str]] = []

    vocabulary: set[str] = set()
    for word in word_freqs:
        vocabulary.update(word)

    for step in range(num_merges):
        pair_freqs = get_pair_frequencies(word_freqs)
        if not pair_freqs:
            print(f"توقف مبكرًا عند الخطوة {step}: لا توجد أزواج متبقية للدمج")
            break

        best_pair = max(pair_freqs, key=lambda p: pair_freqs[p])
        word_freqs = merge_pair(best_pair, word_freqs)
        merges.append(best_pair)
        vocabulary.add(best_pair[0] + best_pair[1])

        print(f"الدمج {step + 1:02d}: {best_pair} → '{best_pair[0]+best_pair[1]}' "
              f"(تكرار: {pair_freqs[best_pair]})")

    return merges, vocabulary


def apply_bpe(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """يُطبّق عمليات الدمج المُتعلَّمة على كلمة جديدة، بنفس الترتيب الذي تعلّمناه."""
    tokens = list(word) + ["</w>"]

    for pair in merges:
        new_tokens: list[str] = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append(pair[0] + pair[1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

    return tokens


if __name__ == "__main__":
    corpus = [
        "القطة تحب اللعب",
        "القطة تحب النوم",
        "الكلب يحب اللعب",
        "الكلب يحب الجري",
        "القطة والكلب يلعبان معا",
    ]

    print("=== تدريب BPE على مجموعة نصوص صغيرة ===\n")
    merges, vocab = train_bpe(corpus, num_merges=15)

    print(f"\nحجم المفردات النهائي: {len(vocab)}")
    print(f"عدد عمليات الدمج: {len(merges)}")

    print("\n=== اختبار على كلمات جديدة (بعضها لم يظهر حرفيًا من قبل) ===")
    test_words = ["القطة", "الكلاب", "يلعب"]  # "الكلاب" و"يلعب" أشكال جديدة!
    for word in test_words:
        tokens = apply_bpe(word, merges)
        print(f"'{word}' → {tokens}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
"""
wordpiece_scoring.py
توضيح الفرق الجوهري بين معيار اختيار الدمج في BPE مقابل WordPiece.
"""
from __future__ import annotations

from collections import Counter


def bpe_score(pair_freq: int, *_: int) -> float:
    """BPE: يختار فقط بناءً على التكرار المطلق."""
    return float(pair_freq)


def wordpiece_score(pair_freq: int, freq_a: int, freq_b: int) -> float:
    """WordPiece: يُطبّع بالتكرار الفردي لكل عنصر (شبه PMI)."""
    return pair_freq / (freq_a * freq_b)


if __name__ == "__main__":
    # سيناريو توضيحي: زوج شائع جدًا لكن عشوائيًا (حروف شائعة تظهر مع كل شيء)
    # مقابل زوج أقل تكرارًا لكن "متماسك" لغويًا بقوة
    scenarios = {
        "('ة', ' ') - شائع جدًا لكن عشوائي": (5000, 50000, 80000),
        "('مر', 'كزي') - أقل تكرارًا لكن مترابط دائمًا": (800, 900, 850),
    }

    print(f"{'السيناريو':<45}{'BPE Score':<15}{'WordPiece Score':<15}")
    for name, (pair_f, freq_a, freq_b) in scenarios.items():
        bpe_s = bpe_score(pair_f)
        wp_s = wordpiece_score(pair_f, freq_a, freq_b)
        print(f"{name:<45}{bpe_s:<15.1f}{wp_s:<15.8f}")

    print("\nلاحظ: BPE يُفضّل الزوج الأول (تكرار خام أعلى)")
    print("بينما WordPiece يُفضّل الزوج الثاني (ترابط إحصائي أقوى نسبيًا)")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
"""
train_production_tokenizer.py
تدريب BPE Tokenizer احترافي وسريع باستخدام مكتبة tokenizers من Hugging Face
(المكتوبة بـ Rust لسرعة فائقة)، جاهز للاستخدام في نموذج حقيقي.
المتطلبات: pip install tokenizers
"""
from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFC, Sequence as NormSequence


def build_and_train_tokenizer(
    corpus_files: list[str],
    vocab_size: int = 8000,
    min_frequency: int = 2,
) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # Normalization: NFC تُوحّد الأشكال المختلفة لنفس الحرف
    # (مهم جدًا للعربية: الألف المهموزة، الهمزات المنفصلة، إلخ)
    tokenizer.normalizer = NormSequence([NFC()])

    # Pre-tokenization: تقسيم أولي بسيط على المسافات قبل تطبيق BPE داخل كل كلمة
    tokenizer.pre_tokenizer = Whitespace()

    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"]
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    tokenizer.train(files=corpus_files, trainer=trainer)
    return tokenizer


if __name__ == "__main__":
    # إعداد ملف نصي تجريبي صغير للتدريب
    sample_corpus = Path("sample_arabic_corpus.txt")
    sample_text = """القطة تحب اللعب في الحديقة كل صباح
الكلب يحب الجري خلف الكرة في الحديقة
الأطفال يلعبون مع القطة والكلب معا كل يوم
تعلم البرمجة يحتاج صبرا وممارسة مستمرة
النماذج اللغوية الكبيرة تغير طريقة تفاعلنا مع التقنية
""" * 50
    sample_corpus.write_text(sample_text, encoding="utf-8")

    tokenizer = build_and_train_tokenizer(
        corpus_files=[str(sample_corpus)], vocab_size=500
    )

    # اختبار الترميز
    test_sentence = "الأطفال والقطط يلعبون في الحدائق الجميلة"
    encoding = tokenizer.encode(test_sentence)

    print(f"الجملة الأصلية: {test_sentence}")
    print(f"عدد التوكنز   : {len(encoding.tokens)}")
    print(f"التوكنز       : {encoding.tokens}")
    print(f"المعرّفات (IDs): {encoding.ids}")

    # فك الترميز للتأكد من إعادة البناء الصحيحة
    decoded = tokenizer.decode(encoding.ids)
    print(f"\nبعد فك الترميز: {decoded}")

    # حفظ الـ Tokenizer لإعادة استخدامه لاحقًا دون إعادة تدريب
    tokenizer.save("my_arabic_tokenizer.json")
    print("\n✅ تم حفظ الـ Tokenizer في my_arabic_tokenizer.json")

    sample_corpus.unlink()  # تنظيف الملف التجريبي

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
"""
tokenization_comparison.py
مقارنة عملية شاملة بين Character-Level, Word-Level, و BPE Tokenization
على نص عربي، مع قياس كفاءة كل استراتيجية.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenizationResult:
    strategy: str
    num_tokens: int
    vocab_size: int
    handles_oov: bool
    example_tokens: list[str]


def tokenize_char_level(text: str) -> list[str]:
    return list(text.replace(" ", "▁"))  # ▁ يُمثّل المسافة بوضوح (تقليد SentencePiece)


def tokenize_word_level(text: str, vocab: set[str]) -> list[str]:
    words = text.split()
    return [w if w in vocab else "[UNK]" for w in words]


def build_word_vocab(corpus: list[str]) -> set[str]:
    vocab: set[str] = set()
    for sentence in corpus:
        vocab.update(sentence.split())
    return vocab


def run_comparison(corpus: list[str], test_sentence: str) -> list[TokenizationResult]:
    results = []

    # --- Character-Level ---
    char_tokens = tokenize_char_level(test_sentence)
    char_vocab = set()
    for s in corpus:
        char_vocab.update(tokenize_char_level(s))
    results.append(TokenizationResult(
        strategy="Character-Level",
        num_tokens=len(char_tokens),
        vocab_size=len(char_vocab),
        handles_oov=True,  # لا توجد كلمة لا يمكن تمثيلها بأحرفها
        example_tokens=char_tokens[:10],
    ))

    # --- Word-Level ---
    word_vocab = build_word_vocab(corpus)
    word_tokens = tokenize_word_level(test_sentence, word_vocab)
    results.append(TokenizationResult(
        strategy="Word-Level",
        num_tokens=len(word_tokens),
        vocab_size=len(word_vocab),
        handles_oov="[UNK]" not in word_tokens,
        example_tokens=word_tokens,
    ))

    # --- BPE (باستخدام تطبيقنا اليدوي من القسم 3.2) ---
    from bpe_from_scratch import train_bpe, apply_bpe  # يفترض وجود الملف بنفس المجلد

    merges, bpe_vocab = train_bpe(corpus, num_merges=40)
    bpe_tokens: list[str] = []
    for word in test_sentence.split():
        bpe_tokens.extend(apply_bpe(word, merges))
    results.append(TokenizationResult(
        strategy="BPE (Subword)",
        num_tokens=len(bpe_tokens),
        vocab_size=len(bpe_vocab),
        handles_oov=True,  # قادر على تركيب أي كلمة من أجزاء معروفة
        example_tokens=bpe_tokens,
    ))

    return results


if __name__ == "__main__":
    corpus = [
        "القطة تحب اللعب في الحديقة",
        "الكلب يحب الجري في الحديقة",
        "الأطفال يحبون اللعب مع الحيوانات",
        "القراءة تفتح آفاقا جديدة للعقل",
        "البرمجة مهارة مهمة في هذا العصر",
    ]
    # جملة اختبار تحتوي كلمات مُركّبة/مُصرّفة لم تظهر حرفيًا من قبل
    test_sentence = "الأطفال والقطط يلعبان في الحدائق"

    results = run_comparison(corpus, test_sentence)

    print(f"\n{'الاستراتيجية':<20}{'عدد التوكنز':<15}{'حجم المفردات':<15}{'يتعامل مع OOV؟':<15}")
    print("-" * 65)
    for r in results:
        oov_mark = "✅" if r.handles_oov else "❌"
        print(f"{r.strategy:<20}{r.num_tokens:<15}{r.vocab_size:<15}{oov_mark:<15}")

    print("\n=== أمثلة التوكنز الناتجة ===")
    for r in results:
        print(f"\n{r.strategy}:")
        print(f"  {r.example_tokens}")

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ فادح: تحميل نموذج مُدرَّب بـ tokenizer معيّن، واستخدام tokenizer آخر معه
model = load_pretrained_model("some-llm")
tokenizer = AutoTokenizer.from_pretrained("different-model-tokenizer")  # ❌!

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: استخدم دائمًا نفس اسم/مسار الـ checkpoint لكل من النموذج والـ tokenizer
model_name = "some-llm"
model = load_pretrained_model(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)  # ✅ متطابق تمامًا

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
# ❌ بدون Normalization، نفس الكلمة بأشكال Unicode مختلفة = توكنز مختلفة
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()
# لا يوجد normalizer! قد تتضاعف المفردات بلا داعٍ

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: أضف NFC Normalization دائمًا كخطوة أولى
from tokenizers.normalizers import NFC
tokenizer.normalizer = NFC()
tokenizer.pre_tokenizer = Whitespace()

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
# ❌ نص طويل جدًا قد يتجاوز الحد الأقصى الذي يدعمه النموذج دون أي تحذير
ids = tokenizer.encode(very_long_document).ids
model(ids)  # قد يفشل أو يُعطي نتائج سيئة جدًا إن تجاوز max_position_embeddings

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: فعّل القص الصريح (truncation) مع تحديد الحد الأقصى بوضوح
encoding = tokenizer.encode(very_long_document)
ids = encoding.ids[:MAX_SEQ_LEN]  # أو استخدم truncation المدمجة في المكتبة

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ شائع: افتراض أن عدد التوكنز ≈ عدد الكلمات (قد يكون مضاعفًا بسهولة!)
num_words = len(text.split())
estimated_cost = num_words * cost_per_token  # تقدير غير دقيق!

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: احسب عدد التوكنز الفعلي دائمًا باستخدام نفس الـ tokenizer المُستهدف
actual_tokens = len(tokenizer.encode(text).ids)
estimated_cost = actual_tokens * cost_per_token  # دقيق، خاصة مهم في APIs تجارية

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
# ❌ حساب Loss على رموز PAD يُضيف ضوضاء ويُحرّف النتيجة
loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: استخدم ignore_index لتجاهل توكنز PAD تمامًا في حساب الخسارة
loss = F.cross_entropy(
    logits.view(-1, vocab_size), labels.view(-1),
    ignore_index=tokenizer.pad_token_id,
)

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-06-chapter-06.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
"""
fertility_analyzer.py — هيكل بداية للتمرين
أداة لقياس وتحليل "عدالة" أي Tokenizer عبر لغات مختلفة.
"""
from __future__ import annotations

from dataclasses import dataclass

from tokenizers import Tokenizer


@dataclass(frozen=True)
class FertilityReport:
    language: str
    num_words: int
    num_tokens: int
    fertility: float  # tokens / words، كلما قلّ كان أفضل (أكثر كفاءة)


def compute_fertility(
    tokenizer: Tokenizer, text: str, language_name: str
) -> FertilityReport:
    """
    TODO:
    1. احسب عدد الكلمات في text عبر .split() (تقريب بسيط كافٍ لهذا التمرين)
    2. رمّز النص الكامل بـ tokenizer.encode(text)، واحسب عدد التوكنز الناتجة
    3. احسب fertility = num_tokens / num_words
    4. أعد FertilityReport بالقيم المحسوبة
    """
    raise NotImplementedError


def compare_languages_fairness(
    tokenizer: Tokenizer,
    texts_by_language: dict[str, str],
) -> list[FertilityReport]:
    """
    TODO: طبّق compute_fertility على كل لغة في القاموس، وأعد قائمة بكل التقارير
    مُرتّبة تنازليًا حسب fertility (الأسوأ أولًا) لتسليط الضوء على اللغات
    "المظلومة" من هذا الـ Tokenizer تحديدًا.
    """
    raise NotImplementedError


def print_fairness_report(reports: list[FertilityReport]) -> None:
    """
    TODO: اطبع جدولاً منسّقًا يُقارن fertility بين اللغات، مع تحذير واضح
    (⚠️) لأي لغة تتجاوز fertility فيها ضعف أقل لغة في القائمة
    (مؤشر على عدم توازن واضح في تمثيل الـ Tokenizer لتلك اللغة).
    """
    raise NotImplementedError


if __name__ == "__main__":
    # نصوص متكافئة معنويًا تقريبًا بلغات مختلفة لمقارنة عادلة
    texts_by_language = {
        "العربية": "الذكاء الاصطناعي يغير العالم بسرعة كبيرة جدا في هذه الايام",
        "English": "Artificial intelligence is changing the world very quickly these days",
        "Français": "L'intelligence artificielle change le monde très rapidement ces jours-ci",
    }

    # TODO: حمّل tokenizer حقيقي (مثلاً GPT-2 أو أي نموذج مفتوح عبر Hugging Face)
    # from tokenizers import Tokenizer
    # tokenizer = Tokenizer.from_pretrained("gpt2")
    #
    # reports = compare_languages_fairness(tokenizer, texts_by_language)
    # print_fairness_report(reports)

    # سؤال للتفكير بعد إتمام التمرين:
    # هل النتائج التي حصلت عليها مفاجئة؟ أي اللغات كانت الأقل "عدالة"؟
    # كيف يمكن لمطوّري نموذج جديد معالجة هذه المشكلة عند بناء Tokenizer مخصص؟
    print("أكمل الدوال أعلاه وفعّل سطور التحميل، ثم شغّل التحليل!")
