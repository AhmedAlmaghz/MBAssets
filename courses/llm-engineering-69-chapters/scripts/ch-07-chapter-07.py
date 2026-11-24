#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — Word2Vec — كيف تتعلّم الآلة معنى الكلمات؟
Large Language Model Engineering — Word2Vec

Book slug : llm-engineering-69-chapters
Chapter   : ch-07-chapter-07 (lesson 7)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-07-chapter-07.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — Word2Vec — كيف تتعلّم الآلة معنى الكلمات؟
Large Language Model Engineering — Word2Vec

Book slug : llm-engineering-69-chapters
Chapter   : ch-07-chapter-07 (lesson 7)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-07-chapter-07.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — Word2Vec — كيف تتعلّم الآلة معنى الكلمات؟
Large Language Model Engineering — Word2Vec

Book slug : llm-engineering-69-chapters
Chapter   : ch-07-chapter-07 (lesson 7)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_07.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 07: الفصل السابع: Word2Vec — كيف تتعلّم الآلة معنى الكلمات؟."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def cosine(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum((x * x for x in left)) * sum((y * y for y in right)))
    return sum((x * y for x, y in zip(left, right))) / denominator if denominator else 0.0

def run_embeddings() -> None:
    corpus = ['the cat sat', 'the cat slept', 'the dog sat', 'a dog ran']
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for sentence in corpus:
        words = sentence.split()
        for index, word in enumerate(words):
            for neighbor in words[max(0, index - 1):index] + words[index + 1:index + 2]:
                counts[word][neighbor] += 1
    target = [counts['cat'].get(w, 0) for w in ('sat', 'slept', 'dog', 'ran')]
    related = [counts['dog'].get(w, 0) for w in ('sat', 'slept', 'dog', 'ran')]
    print('context vector(cat):', target)
    print('context vector(dog):', related)
    print('cosine(cat, dog):', round(cosine(target, related), 3))

if __name__ == "__main__":
    print("Chapter 07 lab: Context co-occurrence and cosine similarity")
    run_embeddings()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-02-embeddings/chapter-07.md
# ────────────────────────────────────────────────────────────────────────
"""
skip_gram_negative_sampling.py
تطبيق كامل لـ Skip-gram with Negative Sampling من الصفر.
Python 3.13+ | PyTorch 2.x
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

SEED = 42
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------
# 1) بناء المفردات وتوزيع السحب العشوائي للعيّنات السلبية (Noise Distribution)
# ---------------------------------------------------------------------
@dataclass
class Vocabulary:
    word2idx: dict[str, int] = field(default_factory=dict)
    idx2word: dict[int, str] = field(default_factory=dict)
    freqs: Counter[str] = field(default_factory=Counter)

    @classmethod
    def build(cls, tokenized_sentences: list[list[str]], min_count: int = 2) -> "Vocabulary":
        vocab = cls()
        counter: Counter[str] = Counter()
        for sent in tokenized_sentences:
            counter.update(sent)
        # نحتفظ فقط بالكلمات التي تكررت min_count مرة على الأقل
        kept_words = [w for w, c in counter.items() if c >= min_count]
        for idx, word in enumerate(kept_words):
            vocab.word2idx[word] = idx
            vocab.idx2word[idx] = word
        vocab.freqs = Counter({w: counter[w] for w in kept_words})
        return vocab

    def __len__(self) -> int:
        return len(self.word2idx)

    def negative_sampling_distribution(self, power: float = 0.75) -> np.ndarray:
        """يبني توزيع الاحتمال P_n(w) ∝ freq(w)^0.75 المُستخدم لسحب العيّنات السلبية."""
        freqs_sorted = np.array(
            [self.freqs[self.idx2word[i]] for i in range(len(self))], dtype=np.float64
        )
        adjusted = freqs_sorted ** power
        return adjusted / adjusted.sum()


def simple_tokenize(text: str) -> list[str]:
    """محاكاة مبسّطة لمحلّل نصوص (في الإنتاج استخدم tokenizer حقيقي من الفصل السادس)."""
    text = text.lower()
    return re.findall(r"[\w']+", text)


# ---------------------------------------------------------------------
# 2) توليد أزواج (مركز، سياق) من النصوص
# ---------------------------------------------------------------------
def generate_skipgram_pairs(
    tokenized_sentences: list[list[str]],
    vocab: Vocabulary,
    window_size: int = 2,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for sentence in tokenized_sentences:
        indices = [vocab.word2idx[w] for w in sentence if w in vocab.word2idx]
        n = len(indices)
        for center_pos in range(n):
            # حجم نافذة ديناميكي (dynamic window) -- حيلة تحسين من الورقة الأصلية:
            # النوافذ الأصغر تُعطى وزنًا أكبر ضمنيًا بسحب حجم عشوائي <= window_size
            dynamic_window = rng.integers(1, window_size + 1)
            start = max(0, center_pos - dynamic_window)
            end = min(n, center_pos + dynamic_window + 1)
            for context_pos in range(start, end):
                if context_pos != center_pos:
                    pairs.append((indices[center_pos], indices[context_pos]))
    return pairs


# ---------------------------------------------------------------------
# 3) Dataset بـ PyTorch مع Negative Sampling مدمج
# ---------------------------------------------------------------------
class SkipGramNegSamplingDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[int, int]],
        vocab_size: int,
        noise_dist: np.ndarray,
        num_negatives: int = 5,
    ) -> None:
        self.pairs = pairs
        self.vocab_size = vocab_size
        self.noise_dist = noise_dist
        self.num_negatives = num_negatives

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[int, int, np.ndarray]:
        center, context = self.pairs[idx]
        negatives = rng.choice(
            self.vocab_size, size=self.num_negatives, p=self.noise_dist, replace=True
        )
        return center, context, negatives


def collate_fn(
    batch: list[tuple[int, int, np.ndarray]]
) -> tuple[Tensor, Tensor, Tensor]:
    centers, contexts, negatives = zip(*batch)
    return (
        torch.tensor(centers, dtype=torch.long),
        torch.tensor(contexts, dtype=torch.long),
        torch.tensor(np.stack(negatives), dtype=torch.long),
    )


# ---------------------------------------------------------------------
# 4) النموذج: مصفوفتا تضمين منفصلتان (in/out) بحسب الورقة الأصلية
# ---------------------------------------------------------------------
class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 100) -> None:
        super().__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self._init_weights(embedding_dim)

    def _init_weights(self, embedding_dim: int) -> None:
        # تهيئة بتوزيع منتظم ضيّق — ممارسة شائعة في Word2Vec الأصلي
        init_range = 0.5 / embedding_dim
        nn.init.uniform_(self.center_embeddings.weight, -init_range, init_range)
        nn.init.zeros_(self.context_embeddings.weight)

    def forward(self, center: Tensor, context: Tensor, negatives: Tensor) -> Tensor:
        """
        center:    (batch,)
        context:   (batch,)
        negatives: (batch, k)
        يُعيد: الخسارة القياسية (scalar)
        """
        v_center = self.center_embeddings(center)          # (batch, d)
        v_context = self.context_embeddings(context)       # (batch, d)
        v_negatives = self.context_embeddings(negatives)   # (batch, k, d)

        # الحد الموجب: log σ(v'ᵀ_context · v_center)
        pos_score = torch.sum(v_center * v_context, dim=1)          # (batch,)
        pos_loss = torch.nn.functional.logsigmoid(pos_score)        # (batch,)

        # الحد السلبي: Σ log σ(-v'ᵀ_neg · v_center)
        neg_score = torch.bmm(v_negatives, v_center.unsqueeze(2)).squeeze(2)  # (batch, k)
        neg_loss = torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)     # (batch,)

        total_loss = -(pos_loss + neg_loss)   # نُصغّر السالب (= نُعظّم الاحتمال)
        return total_loss.mean()

    def get_word_vectors(self) -> Tensor:
        """التضمين النهائي المُستخدم عمليًا = متجهات المركز فقط (الممارسة الأشيع)."""
        return self.center_embeddings.weight.detach()


# ---------------------------------------------------------------------
# 5) حلقة التدريب الكاملة
# ---------------------------------------------------------------------
def train_skipgram(
    tokenized_sentences: list[list[str]],
    embedding_dim: int = 100,
    window_size: int = 3,
    num_negatives: int = 5,
    min_count: int = 1,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 0.01,
    device: str | None = None,
) -> tuple[SkipGramNegSampling, Vocabulary]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    vocab = Vocabulary.build(tokenized_sentences, min_count=min_count)
    pairs = generate_skipgram_pairs(tokenized_sentences, vocab, window_size)
    noise_dist = vocab.negative_sampling_distribution()

    dataset = SkipGramNegSamplingDataset(pairs, len(vocab), noise_dist, num_negatives)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    model = SkipGramNegSampling(len(vocab), embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for center, context, negatives in loader:
            center = center.to(device)
            context = context.to(device)
            negatives = negatives.to(device)

            optimizer.zero_grad()
            loss = model(center, context, negatives)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * center.size(0)

        avg_loss = total_loss / len(dataset)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f}")

    return model, vocab

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-02-embeddings/chapter-07.md
# ────────────────────────────────────────────────────────────────────────
"""
glove_cooccurrence.py
بناء مصفوفة التعايش المشترك كخطوة تمهيدية لـ GloVe (الخطوة الأغلى حسابيًا).
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np


def build_cooccurrence_matrix(
    tokenized_sentences: list[list[str]],
    vocab: "Vocabulary",
    window_size: int = 5,
) -> dict[tuple[int, int], float]:
    """
    يبني مصفوفة تعايش مرجّحة بالمسافة (أقرب كلمة للمركز = وزن أعلى).
    نستخدم dict متفرّق (sparse) لأن أغلب الأزواج X_ij = 0.
    """
    cooccurrence: dict[tuple[int, int], float] = defaultdict(float)

    for sentence in tokenized_sentences:
        indices = [vocab.word2idx[w] for w in sentence if w in vocab.word2idx]
        n = len(indices)
        for i, center in enumerate(indices):
            start, end = max(0, i - window_size), min(n, i + window_size + 1)
            for j in range(start, end):
                if j == i:
                    continue
                distance = abs(i - j)
                weight = 1.0 / distance  # كلما بعُدت الكلمة، قلّ وزن تعايشها
                cooccurrence[(center, indices[j])] += weight

    return cooccurrence


def glove_weighting_function(x: np.ndarray, x_max: float = 100.0, alpha: float = 0.75) -> np.ndarray:
    """دالة الموازنة f(X_ij) من ورقة GloVe الأصلية."""
    return np.where(x < x_max, (x / x_max) ** alpha, 1.0)

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-02-embeddings/chapter-07.md
# ────────────────────────────────────────────────────────────────────────
def cosine_similarity(a: Tensor, b: Tensor) -> Tensor:
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))


def most_similar(
    word: str, vocab: Vocabulary, embeddings: Tensor, top_k: int = 5
) -> list[tuple[str, float]]:
    if word not in vocab.word2idx:
        raise KeyError(f"الكلمة '{word}' غير موجودة في المفردات")

    word_vec = embeddings[vocab.word2idx[word]]
    similarities = torch.nn.functional.cosine_similarity(
        embeddings, word_vec.unsqueeze(0), dim=1
    )
    top_indices = torch.topk(similarities, top_k + 1).indices.tolist()

    results = [
        (vocab.idx2word[idx], similarities[idx].item())
        for idx in top_indices
        if vocab.idx2word[idx] != word
    ][:top_k]
    return results


def analogy(
    a: str, b: str, c: str, vocab: Vocabulary, embeddings: Tensor, top_k: int = 3
) -> list[tuple[str, float]]:
    """يحل المعادلة: a is to b as c is to ?  (مثال: رجل:ملك :: امرأة:؟)"""
    vec = (
        embeddings[vocab.word2idx[b]]
        - embeddings[vocab.word2idx[a]]
        + embeddings[vocab.word2idx[c]]
    )
    similarities = torch.nn.functional.cosine_similarity(
        embeddings, vec.unsqueeze(0), dim=1
    )
    excluded = {vocab.word2idx[a], vocab.word2idx[b], vocab.word2idx[c]}
    top_indices = torch.topk(similarities, top_k + len(excluded)).indices.tolist()
    results = [
        (vocab.idx2word[idx], similarities[idx].item())
        for idx in top_indices
        if idx not in excluded
    ][:top_k]
    return results


# مثال استخدام (بافتراض تدريب كافٍ على مجموعة بيانات كبيرة بما يكفي):
# model, vocab = train_skipgram(large_corpus, embedding_dim=100, epochs=100)
# vectors = model.get_word_vectors()
# print(most_similar("ملك", vocab, vectors))
# print(analogy("رجل", "ملك", "امرأة", vocab, vectors))  # نتوقع: ملكة

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-02-embeddings/chapter-07.md
# ────────────────────────────────────────────────────────────────────────
# خطأ: استخدام نفس الطبقة للمركز والسياق
self.embeddings = nn.Embedding(vocab_size, dim)
score = self.embeddings(center) @ self.embeddings(context).T  # ❌ قد يسبب تآكل تدرّج غريب

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-02-embeddings/chapter-07.md
# ────────────────────────────────────────────────────────────────────────
vocab = Vocabulary.build(sentences, min_count=1)  # ❌ يُبقي كل كلمة ظهرت مرة واحدة فقط

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-02-embeddings/chapter-07.md
# ────────────────────────────────────────────────────────────────────────
# خطأ مفاهيمي: توقع أن تضمين "بنك" يتغيّر حسب الجملة
vector_bank_river = word2vec_model["بنك"]   # ❌ نفس المتجه دائمًا!
vector_bank_money = word2vec_model["بنك"]   # ❌ متطابق تمامًا، رغم اختلاف المعنى!

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-02-embeddings/chapter-07.md
# ────────────────────────────────────────────────────────────────────────
"""
arabic_word_similarity_project.py
مشروع تطبيقي: مُكتشف تشابه دلالي عربي باستخدام Word2Vec.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# TODO 1: حمّل مجموعة نصوص عربية من ملف أو مكتبة (مثل datasets من HuggingFace)
def load_arabic_corpus(path: Path) -> list[str]:
    raise NotImplementedError("حمّل نصوصك هنا")


# TODO 2: نظّف ورمّز النصوص (استخدم simple_tokenize أو Tokenizer BPE من الفصل 6)
def preprocess(raw_texts: list[str]) -> list[list[str]]:
    raise NotImplementedError


# TODO 3: درّب النموذج باستخدام train_skipgram() المُعرَّفة في هذا الفصل
# model, vocab = train_skipgram(tokenized_sentences, embedding_dim=100, epochs=50)


# TODO 4: دالة تصوير التضمينات بعد تقليل الأبعاد
def visualize_embeddings(
    words: list[str], vocab: "Vocabulary", embeddings, output_path: str = "embeddings.png"
) -> None:
    indices = [vocab.word2idx[w] for w in words if w in vocab.word2idx]
    vectors = embeddings[indices].numpy()

    reduced = TSNE(n_components=2, random_state=42, perplexity=5).fit_transform(vectors)

    plt.figure(figsize=(10, 8))
    for i, word in enumerate(words):
        if vocab.word2idx.get(word) is None:
            continue
        x, y = reduced[i]
        plt.scatter(x, y)
        plt.annotate(word, (x, y), fontsize=12)
    plt.title("تصوّر التضمينات الدلالية (Word2Vec)")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"تم حفظ الرسم في: {output_path}")


# TODO 5 (تحدٍّ إضافي): قارن نتائج نموذجك المُدرَّب محليًا بنموذج fastText
# العربي المُدرَّب مسبقًا (Facebook AI)، وناقش الفروقات في الجودة والتغطية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-07-chapter-07.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 07: الفصل السابع: Word2Vec — كيف تتعلّم الآلة معنى الكلمات؟."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def cosine(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum((x * x for x in left)) * sum((y * y for y in right)))
    return sum((x * y for x, y in zip(left, right))) / denominator if denominator else 0.0

def run_embeddings() -> None:
    corpus = ['the cat sat', 'the cat slept', 'the dog sat', 'a dog ran']
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for sentence in corpus:
        words = sentence.split()
        for index, word in enumerate(words):
            for neighbor in words[max(0, index - 1):index] + words[index + 1:index + 2]:
                counts[word][neighbor] += 1
    target = [counts['cat'].get(w, 0) for w in ('sat', 'slept', 'dog', 'ran')]
    related = [counts['dog'].get(w, 0) for w in ('sat', 'slept', 'dog', 'ran')]
    print('context vector(cat):', target)
    print('context vector(dog):', related)
    print('cosine(cat, dog):', round(cosine(target, related), 3))

if __name__ == "__main__":
    print("Chapter 07 lab: Context co-occurrence and cosine similarity")
    run_embeddings()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-07-chapter-07.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
skip_gram_negative_sampling.py
تطبيق كامل لـ Skip-gram with Negative Sampling من الصفر.
Python 3.13+ | PyTorch 2.x
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

SEED = 42
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------
# 1) بناء المفردات وتوزيع السحب العشوائي للعيّنات السلبية (Noise Distribution)
# ---------------------------------------------------------------------
@dataclass
class Vocabulary:
    word2idx: dict[str, int] = field(default_factory=dict)
    idx2word: dict[int, str] = field(default_factory=dict)
    freqs: Counter[str] = field(default_factory=Counter)

    @classmethod
    def build(cls, tokenized_sentences: list[list[str]], min_count: int = 2) -> "Vocabulary":
        vocab = cls()
        counter: Counter[str] = Counter()
        for sent in tokenized_sentences:
            counter.update(sent)
        # نحتفظ فقط بالكلمات التي تكررت min_count مرة على الأقل
        kept_words = [w for w, c in counter.items() if c >= min_count]
        for idx, word in enumerate(kept_words):
            vocab.word2idx[word] = idx
            vocab.idx2word[idx] = word
        vocab.freqs = Counter({w: counter[w] for w in kept_words})
        return vocab

    def __len__(self) -> int:
        return len(self.word2idx)

    def negative_sampling_distribution(self, power: float = 0.75) -> np.ndarray:
        """يبني توزيع الاحتمال P_n(w) ∝ freq(w)^0.75 المُستخدم لسحب العيّنات السلبية."""
        freqs_sorted = np.array(
            [self.freqs[self.idx2word[i]] for i in range(len(self))], dtype=np.float64
        )
        adjusted = freqs_sorted ** power
        return adjusted / adjusted.sum()


def simple_tokenize(text: str) -> list[str]:
    """محاكاة مبسّطة لمحلّل نصوص (في الإنتاج استخدم tokenizer حقيقي من الفصل السادس)."""
    text = text.lower()
    return re.findall(r"[\w']+", text)


# ---------------------------------------------------------------------
# 2) توليد أزواج (مركز، سياق) من النصوص
# ---------------------------------------------------------------------
def generate_skipgram_pairs(
    tokenized_sentences: list[list[str]],
    vocab: Vocabulary,
    window_size: int = 2,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for sentence in tokenized_sentences:
        indices = [vocab.word2idx[w] for w in sentence if w in vocab.word2idx]
        n = len(indices)
        for center_pos in range(n):
            # حجم نافذة ديناميكي (dynamic window) -- حيلة تحسين من الورقة الأصلية:
            # النوافذ الأصغر تُعطى وزنًا أكبر ضمنيًا بسحب حجم عشوائي <= window_size
            dynamic_window = rng.integers(1, window_size + 1)
            start = max(0, center_pos - dynamic_window)
            end = min(n, center_pos + dynamic_window + 1)
            for context_pos in range(start, end):
                if context_pos != center_pos:
                    pairs.append((indices[center_pos], indices[context_pos]))
    return pairs


# ---------------------------------------------------------------------
# 3) Dataset بـ PyTorch مع Negative Sampling مدمج
# ---------------------------------------------------------------------
class SkipGramNegSamplingDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[int, int]],
        vocab_size: int,
        noise_dist: np.ndarray,
        num_negatives: int = 5,
    ) -> None:
        self.pairs = pairs
        self.vocab_size = vocab_size
        self.noise_dist = noise_dist
        self.num_negatives = num_negatives

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[int, int, np.ndarray]:
        center, context = self.pairs[idx]
        negatives = rng.choice(
            self.vocab_size, size=self.num_negatives, p=self.noise_dist, replace=True
        )
        return center, context, negatives


def collate_fn(
    batch: list[tuple[int, int, np.ndarray]]
) -> tuple[Tensor, Tensor, Tensor]:
    centers, contexts, negatives = zip(*batch)
    return (
        torch.tensor(centers, dtype=torch.long),
        torch.tensor(contexts, dtype=torch.long),
        torch.tensor(np.stack(negatives), dtype=torch.long),
    )


# ---------------------------------------------------------------------
# 4) النموذج: مصفوفتا تضمين منفصلتان (in/out) بحسب الورقة الأصلية
# ---------------------------------------------------------------------
class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 100) -> None:
        super().__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self._init_weights(embedding_dim)

    def _init_weights(self, embedding_dim: int) -> None:
        # تهيئة بتوزيع منتظم ضيّق — ممارسة شائعة في Word2Vec الأصلي
        init_range = 0.5 / embedding_dim
        nn.init.uniform_(self.center_embeddings.weight, -init_range, init_range)
        nn.init.zeros_(self.context_embeddings.weight)

    def forward(self, center: Tensor, context: Tensor, negatives: Tensor) -> Tensor:
        """
        center:    (batch,)
        context:   (batch,)
        negatives: (batch, k)
        يُعيد: الخسارة القياسية (scalar)
        """
        v_center = self.center_embeddings(center)          # (batch, d)
        v_context = self.context_embeddings(context)       # (batch, d)
        v_negatives = self.context_embeddings(negatives)   # (batch, k, d)

        # الحد الموجب: log σ(v'ᵀ_context · v_center)
        pos_score = torch.sum(v_center * v_context, dim=1)          # (batch,)
        pos_loss = torch.nn.functional.logsigmoid(pos_score)        # (batch,)

        # الحد السلبي: Σ log σ(-v'ᵀ_neg · v_center)
        neg_score = torch.bmm(v_negatives, v_center.unsqueeze(2)).squeeze(2)  # (batch, k)
        neg_loss = torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)     # (batch,)

        total_loss = -(pos_loss + neg_loss)   # نُصغّر السالب (= نُعظّم الاحتمال)
        return total_loss.mean()

    def get_word_vectors(self) -> Tensor:
        """التضمين النهائي المُستخدم عمليًا = متجهات المركز فقط (الممارسة الأشيع)."""
        return self.center_embeddings.weight.detach()


# ---------------------------------------------------------------------
# 5) حلقة التدريب الكاملة
# ---------------------------------------------------------------------
def train_skipgram(
    tokenized_sentences: list[list[str]],
    embedding_dim: int = 100,
    window_size: int = 3,
    num_negatives: int = 5,
    min_count: int = 1,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 0.01,
    device: str | None = None,
) -> tuple[SkipGramNegSampling, Vocabulary]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    vocab = Vocabulary.build(tokenized_sentences, min_count=min_count)
    pairs = generate_skipgram_pairs(tokenized_sentences, vocab, window_size)
    noise_dist = vocab.negative_sampling_distribution()

    dataset = SkipGramNegSamplingDataset(pairs, len(vocab), noise_dist, num_negatives)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    model = SkipGramNegSampling(len(vocab), embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for center, context, negatives in loader:
            center = center.to(device)
            context = context.to(device)
            negatives = negatives.to(device)

            optimizer.zero_grad()
            loss = model(center, context, negatives)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * center.size(0)

        avg_loss = total_loss / len(dataset)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f}")

    return model, vocab

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-07-chapter-07.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
glove_cooccurrence.py
بناء مصفوفة التعايش المشترك كخطوة تمهيدية لـ GloVe (الخطوة الأغلى حسابيًا).
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np


def build_cooccurrence_matrix(
    tokenized_sentences: list[list[str]],
    vocab: "Vocabulary",
    window_size: int = 5,
) -> dict[tuple[int, int], float]:
    """
    يبني مصفوفة تعايش مرجّحة بالمسافة (أقرب كلمة للمركز = وزن أعلى).
    نستخدم dict متفرّق (sparse) لأن أغلب الأزواج X_ij = 0.
    """
    cooccurrence: dict[tuple[int, int], float] = defaultdict(float)

    for sentence in tokenized_sentences:
        indices = [vocab.word2idx[w] for w in sentence if w in vocab.word2idx]
        n = len(indices)
        for i, center in enumerate(indices):
            start, end = max(0, i - window_size), min(n, i + window_size + 1)
            for j in range(start, end):
                if j == i:
                    continue
                distance = abs(i - j)
                weight = 1.0 / distance  # كلما بعُدت الكلمة، قلّ وزن تعايشها
                cooccurrence[(center, indices[j])] += weight

    return cooccurrence


def glove_weighting_function(x: np.ndarray, x_max: float = 100.0, alpha: float = 0.75) -> np.ndarray:
    """دالة الموازنة f(X_ij) من ورقة GloVe الأصلية."""
    return np.where(x < x_max, (x / x_max) ** alpha, 1.0)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-07-chapter-07.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def cosine_similarity(a: Tensor, b: Tensor) -> Tensor:
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))


def most_similar(
    word: str, vocab: Vocabulary, embeddings: Tensor, top_k: int = 5
) -> list[tuple[str, float]]:
    if word not in vocab.word2idx:
        raise KeyError(f"الكلمة '{word}' غير موجودة في المفردات")

    word_vec = embeddings[vocab.word2idx[word]]
    similarities = torch.nn.functional.cosine_similarity(
        embeddings, word_vec.unsqueeze(0), dim=1
    )
    top_indices = torch.topk(similarities, top_k + 1).indices.tolist()

    results = [
        (vocab.idx2word[idx], similarities[idx].item())
        for idx in top_indices
        if vocab.idx2word[idx] != word
    ][:top_k]
    return results


def analogy(
    a: str, b: str, c: str, vocab: Vocabulary, embeddings: Tensor, top_k: int = 3
) -> list[tuple[str, float]]:
    """يحل المعادلة: a is to b as c is to ?  (مثال: رجل:ملك :: امرأة:؟)"""
    vec = (
        embeddings[vocab.word2idx[b]]
        - embeddings[vocab.word2idx[a]]
        + embeddings[vocab.word2idx[c]]
    )
    similarities = torch.nn.functional.cosine_similarity(
        embeddings, vec.unsqueeze(0), dim=1
    )
    excluded = {vocab.word2idx[a], vocab.word2idx[b], vocab.word2idx[c]}
    top_indices = torch.topk(similarities, top_k + len(excluded)).indices.tolist()
    results = [
        (vocab.idx2word[idx], similarities[idx].item())
        for idx in top_indices
        if idx not in excluded
    ][:top_k]
    return results


# مثال استخدام (بافتراض تدريب كافٍ على مجموعة بيانات كبيرة بما يكفي):
# model, vocab = train_skipgram(large_corpus, embedding_dim=100, epochs=100)
# vectors = model.get_word_vectors()
# print(most_similar("ملك", vocab, vectors))
# print(analogy("رجل", "ملك", "امرأة", vocab, vectors))  # نتوقع: ملكة

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-07-chapter-07.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# خطأ: استخدام نفس الطبقة للمركز والسياق
self.embeddings = nn.Embedding(vocab_size, dim)
score = self.embeddings(center) @ self.embeddings(context).T  # ❌ قد يسبب تآكل تدرّج غريب

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-07-chapter-07.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
vocab = Vocabulary.build(sentences, min_count=1)  # ❌ يُبقي كل كلمة ظهرت مرة واحدة فقط

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-07-chapter-07.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# خطأ مفاهيمي: توقع أن تضمين "بنك" يتغيّر حسب الجملة
vector_bank_river = word2vec_model["بنك"]   # ❌ نفس المتجه دائمًا!
vector_bank_money = word2vec_model["بنك"]   # ❌ متطابق تمامًا، رغم اختلاف المعنى!

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-07-chapter-07.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
"""
arabic_word_similarity_project.py
مشروع تطبيقي: مُكتشف تشابه دلالي عربي باستخدام Word2Vec.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# TODO 1: حمّل مجموعة نصوص عربية من ملف أو مكتبة (مثل datasets من HuggingFace)
def load_arabic_corpus(path: Path) -> list[str]:
    raise NotImplementedError("حمّل نصوصك هنا")


# TODO 2: نظّف ورمّز النصوص (استخدم simple_tokenize أو Tokenizer BPE من الفصل 6)
def preprocess(raw_texts: list[str]) -> list[list[str]]:
    raise NotImplementedError


# TODO 3: درّب النموذج باستخدام train_skipgram() المُعرَّفة في هذا الفصل
# model, vocab = train_skipgram(tokenized_sentences, embedding_dim=100, epochs=50)


# TODO 4: دالة تصوير التضمينات بعد تقليل الأبعاد
def visualize_embeddings(
    words: list[str], vocab: "Vocabulary", embeddings, output_path: str = "embeddings.png"
) -> None:
    indices = [vocab.word2idx[w] for w in words if w in vocab.word2idx]
    vectors = embeddings[indices].numpy()

    reduced = TSNE(n_components=2, random_state=42, perplexity=5).fit_transform(vectors)

    plt.figure(figsize=(10, 8))
    for i, word in enumerate(words):
        if vocab.word2idx.get(word) is None:
            continue
        x, y = reduced[i]
        plt.scatter(x, y)
        plt.annotate(word, (x, y), fontsize=12)
    plt.title("تصوّر التضمينات الدلالية (Word2Vec)")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"تم حفظ الرسم في: {output_path}")


# TODO 5 (تحدٍّ إضافي): قارن نتائج نموذجك المُدرَّب محليًا بنموذج fastText
# العربي المُدرَّب مسبقًا (Facebook AI)، وناقش الفروقات في الجودة والتغطية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-07-chapter-07.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — Word2Vec — كيف تتعلّم الآلة معنى الكلمات؟
Large Language Model Engineering — Word2Vec

Book slug : llm-engineering-69-chapters
Chapter   : ch-07-chapter-07 (lesson 7)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_07.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 07: الفصل السابع: Word2Vec — كيف تتعلّم الآلة معنى الكلمات؟."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def cosine(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum((x * x for x in left)) * sum((y * y for y in right)))
    return sum((x * y for x, y in zip(left, right))) / denominator if denominator else 0.0

def run_embeddings() -> None:
    corpus = ['the cat sat', 'the cat slept', 'the dog sat', 'a dog ran']
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for sentence in corpus:
        words = sentence.split()
        for index, word in enumerate(words):
            for neighbor in words[max(0, index - 1):index] + words[index + 1:index + 2]:
                counts[word][neighbor] += 1
    target = [counts['cat'].get(w, 0) for w in ('sat', 'slept', 'dog', 'ran')]
    related = [counts['dog'].get(w, 0) for w in ('sat', 'slept', 'dog', 'ran')]
    print('context vector(cat):', target)
    print('context vector(dog):', related)
    print('cosine(cat, dog):', round(cosine(target, related), 3))

if __name__ == "__main__":
    print("Chapter 07 lab: Context co-occurrence and cosine similarity")
    run_embeddings()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-02-embeddings/chapter-07.md
# ────────────────────────────────────────────────────────────────────────
"""
skip_gram_negative_sampling.py
تطبيق كامل لـ Skip-gram with Negative Sampling من الصفر.
Python 3.13+ | PyTorch 2.x
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

SEED = 42
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------
# 1) بناء المفردات وتوزيع السحب العشوائي للعيّنات السلبية (Noise Distribution)
# ---------------------------------------------------------------------
@dataclass
class Vocabulary:
    word2idx: dict[str, int] = field(default_factory=dict)
    idx2word: dict[int, str] = field(default_factory=dict)
    freqs: Counter[str] = field(default_factory=Counter)

    @classmethod
    def build(cls, tokenized_sentences: list[list[str]], min_count: int = 2) -> "Vocabulary":
        vocab = cls()
        counter: Counter[str] = Counter()
        for sent in tokenized_sentences:
            counter.update(sent)
        # نحتفظ فقط بالكلمات التي تكررت min_count مرة على الأقل
        kept_words = [w for w, c in counter.items() if c >= min_count]
        for idx, word in enumerate(kept_words):
            vocab.word2idx[word] = idx
            vocab.idx2word[idx] = word
        vocab.freqs = Counter({w: counter[w] for w in kept_words})
        return vocab

    def __len__(self) -> int:
        return len(self.word2idx)

    def negative_sampling_distribution(self, power: float = 0.75) -> np.ndarray:
        """يبني توزيع الاحتمال P_n(w) ∝ freq(w)^0.75 المُستخدم لسحب العيّنات السلبية."""
        freqs_sorted = np.array(
            [self.freqs[self.idx2word[i]] for i in range(len(self))], dtype=np.float64
        )
        adjusted = freqs_sorted ** power
        return adjusted / adjusted.sum()


def simple_tokenize(text: str) -> list[str]:
    """محاكاة مبسّطة لمحلّل نصوص (في الإنتاج استخدم tokenizer حقيقي من الفصل السادس)."""
    text = text.lower()
    return re.findall(r"[\w']+", text)


# ---------------------------------------------------------------------
# 2) توليد أزواج (مركز، سياق) من النصوص
# ---------------------------------------------------------------------
def generate_skipgram_pairs(
    tokenized_sentences: list[list[str]],
    vocab: Vocabulary,
    window_size: int = 2,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for sentence in tokenized_sentences:
        indices = [vocab.word2idx[w] for w in sentence if w in vocab.word2idx]
        n = len(indices)
        for center_pos in range(n):
            # حجم نافذة ديناميكي (dynamic window) -- حيلة تحسين من الورقة الأصلية:
            # النوافذ الأصغر تُعطى وزنًا أكبر ضمنيًا بسحب حجم عشوائي <= window_size
            dynamic_window = rng.integers(1, window_size + 1)
            start = max(0, center_pos - dynamic_window)
            end = min(n, center_pos + dynamic_window + 1)
            for context_pos in range(start, end):
                if context_pos != center_pos:
                    pairs.append((indices[center_pos], indices[context_pos]))
    return pairs


# ---------------------------------------------------------------------
# 3) Dataset بـ PyTorch مع Negative Sampling مدمج
# ---------------------------------------------------------------------
class SkipGramNegSamplingDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[int, int]],
        vocab_size: int,
        noise_dist: np.ndarray,
        num_negatives: int = 5,
    ) -> None:
        self.pairs = pairs
        self.vocab_size = vocab_size
        self.noise_dist = noise_dist
        self.num_negatives = num_negatives

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[int, int, np.ndarray]:
        center, context = self.pairs[idx]
        negatives = rng.choice(
            self.vocab_size, size=self.num_negatives, p=self.noise_dist, replace=True
        )
        return center, context, negatives


def collate_fn(
    batch: list[tuple[int, int, np.ndarray]]
) -> tuple[Tensor, Tensor, Tensor]:
    centers, contexts, negatives = zip(*batch)
    return (
        torch.tensor(centers, dtype=torch.long),
        torch.tensor(contexts, dtype=torch.long),
        torch.tensor(np.stack(negatives), dtype=torch.long),
    )


# ---------------------------------------------------------------------
# 4) النموذج: مصفوفتا تضمين منفصلتان (in/out) بحسب الورقة الأصلية
# ---------------------------------------------------------------------
class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 100) -> None:
        super().__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self._init_weights(embedding_dim)

    def _init_weights(self, embedding_dim: int) -> None:
        # تهيئة بتوزيع منتظم ضيّق — ممارسة شائعة في Word2Vec الأصلي
        init_range = 0.5 / embedding_dim
        nn.init.uniform_(self.center_embeddings.weight, -init_range, init_range)
        nn.init.zeros_(self.context_embeddings.weight)

    def forward(self, center: Tensor, context: Tensor, negatives: Tensor) -> Tensor:
        """
        center:    (batch,)
        context:   (batch,)
        negatives: (batch, k)
        يُعيد: الخسارة القياسية (scalar)
        """
        v_center = self.center_embeddings(center)          # (batch, d)
        v_context = self.context_embeddings(context)       # (batch, d)
        v_negatives = self.context_embeddings(negatives)   # (batch, k, d)

        # الحد الموجب: log σ(v'ᵀ_context · v_center)
        pos_score = torch.sum(v_center * v_context, dim=1)          # (batch,)
        pos_loss = torch.nn.functional.logsigmoid(pos_score)        # (batch,)

        # الحد السلبي: Σ log σ(-v'ᵀ_neg · v_center)
        neg_score = torch.bmm(v_negatives, v_center.unsqueeze(2)).squeeze(2)  # (batch, k)
        neg_loss = torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)     # (batch,)

        total_loss = -(pos_loss + neg_loss)   # نُصغّر السالب (= نُعظّم الاحتمال)
        return total_loss.mean()

    def get_word_vectors(self) -> Tensor:
        """التضمين النهائي المُستخدم عمليًا = متجهات المركز فقط (الممارسة الأشيع)."""
        return self.center_embeddings.weight.detach()


# ---------------------------------------------------------------------
# 5) حلقة التدريب الكاملة
# ---------------------------------------------------------------------
def train_skipgram(
    tokenized_sentences: list[list[str]],
    embedding_dim: int = 100,
    window_size: int = 3,
    num_negatives: int = 5,
    min_count: int = 1,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 0.01,
    device: str | None = None,
) -> tuple[SkipGramNegSampling, Vocabulary]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    vocab = Vocabulary.build(tokenized_sentences, min_count=min_count)
    pairs = generate_skipgram_pairs(tokenized_sentences, vocab, window_size)
    noise_dist = vocab.negative_sampling_distribution()

    dataset = SkipGramNegSamplingDataset(pairs, len(vocab), noise_dist, num_negatives)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    model = SkipGramNegSampling(len(vocab), embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for center, context, negatives in loader:
            center = center.to(device)
            context = context.to(device)
            negatives = negatives.to(device)

            optimizer.zero_grad()
            loss = model(center, context, negatives)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * center.size(0)

        avg_loss = total_loss / len(dataset)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f}")

    return model, vocab

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-02-embeddings/chapter-07.md
# ────────────────────────────────────────────────────────────────────────
"""
glove_cooccurrence.py
بناء مصفوفة التعايش المشترك كخطوة تمهيدية لـ GloVe (الخطوة الأغلى حسابيًا).
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np


def build_cooccurrence_matrix(
    tokenized_sentences: list[list[str]],
    vocab: "Vocabulary",
    window_size: int = 5,
) -> dict[tuple[int, int], float]:
    """
    يبني مصفوفة تعايش مرجّحة بالمسافة (أقرب كلمة للمركز = وزن أعلى).
    نستخدم dict متفرّق (sparse) لأن أغلب الأزواج X_ij = 0.
    """
    cooccurrence: dict[tuple[int, int], float] = defaultdict(float)

    for sentence in tokenized_sentences:
        indices = [vocab.word2idx[w] for w in sentence if w in vocab.word2idx]
        n = len(indices)
        for i, center in enumerate(indices):
            start, end = max(0, i - window_size), min(n, i + window_size + 1)
            for j in range(start, end):
                if j == i:
                    continue
                distance = abs(i - j)
                weight = 1.0 / distance  # كلما بعُدت الكلمة، قلّ وزن تعايشها
                cooccurrence[(center, indices[j])] += weight

    return cooccurrence


def glove_weighting_function(x: np.ndarray, x_max: float = 100.0, alpha: float = 0.75) -> np.ndarray:
    """دالة الموازنة f(X_ij) من ورقة GloVe الأصلية."""
    return np.where(x < x_max, (x / x_max) ** alpha, 1.0)

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-02-embeddings/chapter-07.md
# ────────────────────────────────────────────────────────────────────────
def cosine_similarity(a: Tensor, b: Tensor) -> Tensor:
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))


def most_similar(
    word: str, vocab: Vocabulary, embeddings: Tensor, top_k: int = 5
) -> list[tuple[str, float]]:
    if word not in vocab.word2idx:
        raise KeyError(f"الكلمة '{word}' غير موجودة في المفردات")

    word_vec = embeddings[vocab.word2idx[word]]
    similarities = torch.nn.functional.cosine_similarity(
        embeddings, word_vec.unsqueeze(0), dim=1
    )
    top_indices = torch.topk(similarities, top_k + 1).indices.tolist()

    results = [
        (vocab.idx2word[idx], similarities[idx].item())
        for idx in top_indices
        if vocab.idx2word[idx] != word
    ][:top_k]
    return results


def analogy(
    a: str, b: str, c: str, vocab: Vocabulary, embeddings: Tensor, top_k: int = 3
) -> list[tuple[str, float]]:
    """يحل المعادلة: a is to b as c is to ?  (مثال: رجل:ملك :: امرأة:؟)"""
    vec = (
        embeddings[vocab.word2idx[b]]
        - embeddings[vocab.word2idx[a]]
        + embeddings[vocab.word2idx[c]]
    )
    similarities = torch.nn.functional.cosine_similarity(
        embeddings, vec.unsqueeze(0), dim=1
    )
    excluded = {vocab.word2idx[a], vocab.word2idx[b], vocab.word2idx[c]}
    top_indices = torch.topk(similarities, top_k + len(excluded)).indices.tolist()
    results = [
        (vocab.idx2word[idx], similarities[idx].item())
        for idx in top_indices
        if idx not in excluded
    ][:top_k]
    return results


# مثال استخدام (بافتراض تدريب كافٍ على مجموعة بيانات كبيرة بما يكفي):
# model, vocab = train_skipgram(large_corpus, embedding_dim=100, epochs=100)
# vectors = model.get_word_vectors()
# print(most_similar("ملك", vocab, vectors))
# print(analogy("رجل", "ملك", "امرأة", vocab, vectors))  # نتوقع: ملكة

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-02-embeddings/chapter-07.md
# ────────────────────────────────────────────────────────────────────────
# خطأ: استخدام نفس الطبقة للمركز والسياق
self.embeddings = nn.Embedding(vocab_size, dim)
score = self.embeddings(center) @ self.embeddings(context).T  # ❌ قد يسبب تآكل تدرّج غريب

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-02-embeddings/chapter-07.md
# ────────────────────────────────────────────────────────────────────────
vocab = Vocabulary.build(sentences, min_count=1)  # ❌ يُبقي كل كلمة ظهرت مرة واحدة فقط

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-02-embeddings/chapter-07.md
# ────────────────────────────────────────────────────────────────────────
# خطأ مفاهيمي: توقع أن تضمين "بنك" يتغيّر حسب الجملة
vector_bank_river = word2vec_model["بنك"]   # ❌ نفس المتجه دائمًا!
vector_bank_money = word2vec_model["بنك"]   # ❌ متطابق تمامًا، رغم اختلاف المعنى!

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-02-embeddings/chapter-07.md
# ────────────────────────────────────────────────────────────────────────
"""
arabic_word_similarity_project.py
مشروع تطبيقي: مُكتشف تشابه دلالي عربي باستخدام Word2Vec.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# TODO 1: حمّل مجموعة نصوص عربية من ملف أو مكتبة (مثل datasets من HuggingFace)
def load_arabic_corpus(path: Path) -> list[str]:
    raise NotImplementedError("حمّل نصوصك هنا")


# TODO 2: نظّف ورمّز النصوص (استخدم simple_tokenize أو Tokenizer BPE من الفصل 6)
def preprocess(raw_texts: list[str]) -> list[list[str]]:
    raise NotImplementedError


# TODO 3: درّب النموذج باستخدام train_skipgram() المُعرَّفة في هذا الفصل
# model, vocab = train_skipgram(tokenized_sentences, embedding_dim=100, epochs=50)


# TODO 4: دالة تصوير التضمينات بعد تقليل الأبعاد
def visualize_embeddings(
    words: list[str], vocab: "Vocabulary", embeddings, output_path: str = "embeddings.png"
) -> None:
    indices = [vocab.word2idx[w] for w in words if w in vocab.word2idx]
    vectors = embeddings[indices].numpy()

    reduced = TSNE(n_components=2, random_state=42, perplexity=5).fit_transform(vectors)

    plt.figure(figsize=(10, 8))
    for i, word in enumerate(words):
        if vocab.word2idx.get(word) is None:
            continue
        x, y = reduced[i]
        plt.scatter(x, y)
        plt.annotate(word, (x, y), fontsize=12)
    plt.title("تصوّر التضمينات الدلالية (Word2Vec)")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"تم حفظ الرسم في: {output_path}")


# TODO 5 (تحدٍّ إضافي): قارن نتائج نموذجك المُدرَّب محليًا بنموذج fastText
# العربي المُدرَّب مسبقًا (Facebook AI)، وناقش الفروقات في الجودة والتغطية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-07-chapter-07.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 07: الفصل السابع: Word2Vec — كيف تتعلّم الآلة معنى الكلمات؟."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def cosine(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum((x * x for x in left)) * sum((y * y for y in right)))
    return sum((x * y for x, y in zip(left, right))) / denominator if denominator else 0.0

def run_embeddings() -> None:
    corpus = ['the cat sat', 'the cat slept', 'the dog sat', 'a dog ran']
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for sentence in corpus:
        words = sentence.split()
        for index, word in enumerate(words):
            for neighbor in words[max(0, index - 1):index] + words[index + 1:index + 2]:
                counts[word][neighbor] += 1
    target = [counts['cat'].get(w, 0) for w in ('sat', 'slept', 'dog', 'ran')]
    related = [counts['dog'].get(w, 0) for w in ('sat', 'slept', 'dog', 'ran')]
    print('context vector(cat):', target)
    print('context vector(dog):', related)
    print('cosine(cat, dog):', round(cosine(target, related), 3))

if __name__ == "__main__":
    print("Chapter 07 lab: Context co-occurrence and cosine similarity")
    run_embeddings()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-07-chapter-07.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
skip_gram_negative_sampling.py
تطبيق كامل لـ Skip-gram with Negative Sampling من الصفر.
Python 3.13+ | PyTorch 2.x
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

SEED = 42
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------
# 1) بناء المفردات وتوزيع السحب العشوائي للعيّنات السلبية (Noise Distribution)
# ---------------------------------------------------------------------
@dataclass
class Vocabulary:
    word2idx: dict[str, int] = field(default_factory=dict)
    idx2word: dict[int, str] = field(default_factory=dict)
    freqs: Counter[str] = field(default_factory=Counter)

    @classmethod
    def build(cls, tokenized_sentences: list[list[str]], min_count: int = 2) -> "Vocabulary":
        vocab = cls()
        counter: Counter[str] = Counter()
        for sent in tokenized_sentences:
            counter.update(sent)
        # نحتفظ فقط بالكلمات التي تكررت min_count مرة على الأقل
        kept_words = [w for w, c in counter.items() if c >= min_count]
        for idx, word in enumerate(kept_words):
            vocab.word2idx[word] = idx
            vocab.idx2word[idx] = word
        vocab.freqs = Counter({w: counter[w] for w in kept_words})
        return vocab

    def __len__(self) -> int:
        return len(self.word2idx)

    def negative_sampling_distribution(self, power: float = 0.75) -> np.ndarray:
        """يبني توزيع الاحتمال P_n(w) ∝ freq(w)^0.75 المُستخدم لسحب العيّنات السلبية."""
        freqs_sorted = np.array(
            [self.freqs[self.idx2word[i]] for i in range(len(self))], dtype=np.float64
        )
        adjusted = freqs_sorted ** power
        return adjusted / adjusted.sum()


def simple_tokenize(text: str) -> list[str]:
    """محاكاة مبسّطة لمحلّل نصوص (في الإنتاج استخدم tokenizer حقيقي من الفصل السادس)."""
    text = text.lower()
    return re.findall(r"[\w']+", text)


# ---------------------------------------------------------------------
# 2) توليد أزواج (مركز، سياق) من النصوص
# ---------------------------------------------------------------------
def generate_skipgram_pairs(
    tokenized_sentences: list[list[str]],
    vocab: Vocabulary,
    window_size: int = 2,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for sentence in tokenized_sentences:
        indices = [vocab.word2idx[w] for w in sentence if w in vocab.word2idx]
        n = len(indices)
        for center_pos in range(n):
            # حجم نافذة ديناميكي (dynamic window) -- حيلة تحسين من الورقة الأصلية:
            # النوافذ الأصغر تُعطى وزنًا أكبر ضمنيًا بسحب حجم عشوائي <= window_size
            dynamic_window = rng.integers(1, window_size + 1)
            start = max(0, center_pos - dynamic_window)
            end = min(n, center_pos + dynamic_window + 1)
            for context_pos in range(start, end):
                if context_pos != center_pos:
                    pairs.append((indices[center_pos], indices[context_pos]))
    return pairs


# ---------------------------------------------------------------------
# 3) Dataset بـ PyTorch مع Negative Sampling مدمج
# ---------------------------------------------------------------------
class SkipGramNegSamplingDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[int, int]],
        vocab_size: int,
        noise_dist: np.ndarray,
        num_negatives: int = 5,
    ) -> None:
        self.pairs = pairs
        self.vocab_size = vocab_size
        self.noise_dist = noise_dist
        self.num_negatives = num_negatives

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[int, int, np.ndarray]:
        center, context = self.pairs[idx]
        negatives = rng.choice(
            self.vocab_size, size=self.num_negatives, p=self.noise_dist, replace=True
        )
        return center, context, negatives


def collate_fn(
    batch: list[tuple[int, int, np.ndarray]]
) -> tuple[Tensor, Tensor, Tensor]:
    centers, contexts, negatives = zip(*batch)
    return (
        torch.tensor(centers, dtype=torch.long),
        torch.tensor(contexts, dtype=torch.long),
        torch.tensor(np.stack(negatives), dtype=torch.long),
    )


# ---------------------------------------------------------------------
# 4) النموذج: مصفوفتا تضمين منفصلتان (in/out) بحسب الورقة الأصلية
# ---------------------------------------------------------------------
class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 100) -> None:
        super().__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self._init_weights(embedding_dim)

    def _init_weights(self, embedding_dim: int) -> None:
        # تهيئة بتوزيع منتظم ضيّق — ممارسة شائعة في Word2Vec الأصلي
        init_range = 0.5 / embedding_dim
        nn.init.uniform_(self.center_embeddings.weight, -init_range, init_range)
        nn.init.zeros_(self.context_embeddings.weight)

    def forward(self, center: Tensor, context: Tensor, negatives: Tensor) -> Tensor:
        """
        center:    (batch,)
        context:   (batch,)
        negatives: (batch, k)
        يُعيد: الخسارة القياسية (scalar)
        """
        v_center = self.center_embeddings(center)          # (batch, d)
        v_context = self.context_embeddings(context)       # (batch, d)
        v_negatives = self.context_embeddings(negatives)   # (batch, k, d)

        # الحد الموجب: log σ(v'ᵀ_context · v_center)
        pos_score = torch.sum(v_center * v_context, dim=1)          # (batch,)
        pos_loss = torch.nn.functional.logsigmoid(pos_score)        # (batch,)

        # الحد السلبي: Σ log σ(-v'ᵀ_neg · v_center)
        neg_score = torch.bmm(v_negatives, v_center.unsqueeze(2)).squeeze(2)  # (batch, k)
        neg_loss = torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)     # (batch,)

        total_loss = -(pos_loss + neg_loss)   # نُصغّر السالب (= نُعظّم الاحتمال)
        return total_loss.mean()

    def get_word_vectors(self) -> Tensor:
        """التضمين النهائي المُستخدم عمليًا = متجهات المركز فقط (الممارسة الأشيع)."""
        return self.center_embeddings.weight.detach()


# ---------------------------------------------------------------------
# 5) حلقة التدريب الكاملة
# ---------------------------------------------------------------------
def train_skipgram(
    tokenized_sentences: list[list[str]],
    embedding_dim: int = 100,
    window_size: int = 3,
    num_negatives: int = 5,
    min_count: int = 1,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 0.01,
    device: str | None = None,
) -> tuple[SkipGramNegSampling, Vocabulary]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    vocab = Vocabulary.build(tokenized_sentences, min_count=min_count)
    pairs = generate_skipgram_pairs(tokenized_sentences, vocab, window_size)
    noise_dist = vocab.negative_sampling_distribution()

    dataset = SkipGramNegSamplingDataset(pairs, len(vocab), noise_dist, num_negatives)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    model = SkipGramNegSampling(len(vocab), embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for center, context, negatives in loader:
            center = center.to(device)
            context = context.to(device)
            negatives = negatives.to(device)

            optimizer.zero_grad()
            loss = model(center, context, negatives)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * center.size(0)

        avg_loss = total_loss / len(dataset)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f}")

    return model, vocab

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-07-chapter-07.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
"""
glove_cooccurrence.py
بناء مصفوفة التعايش المشترك كخطوة تمهيدية لـ GloVe (الخطوة الأغلى حسابيًا).
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np


def build_cooccurrence_matrix(
    tokenized_sentences: list[list[str]],
    vocab: "Vocabulary",
    window_size: int = 5,
) -> dict[tuple[int, int], float]:
    """
    يبني مصفوفة تعايش مرجّحة بالمسافة (أقرب كلمة للمركز = وزن أعلى).
    نستخدم dict متفرّق (sparse) لأن أغلب الأزواج X_ij = 0.
    """
    cooccurrence: dict[tuple[int, int], float] = defaultdict(float)

    for sentence in tokenized_sentences:
        indices = [vocab.word2idx[w] for w in sentence if w in vocab.word2idx]
        n = len(indices)
        for i, center in enumerate(indices):
            start, end = max(0, i - window_size), min(n, i + window_size + 1)
            for j in range(start, end):
                if j == i:
                    continue
                distance = abs(i - j)
                weight = 1.0 / distance  # كلما بعُدت الكلمة، قلّ وزن تعايشها
                cooccurrence[(center, indices[j])] += weight

    return cooccurrence


def glove_weighting_function(x: np.ndarray, x_max: float = 100.0, alpha: float = 0.75) -> np.ndarray:
    """دالة الموازنة f(X_ij) من ورقة GloVe الأصلية."""
    return np.where(x < x_max, (x / x_max) ** alpha, 1.0)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-07-chapter-07.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def cosine_similarity(a: Tensor, b: Tensor) -> Tensor:
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))


def most_similar(
    word: str, vocab: Vocabulary, embeddings: Tensor, top_k: int = 5
) -> list[tuple[str, float]]:
    if word not in vocab.word2idx:
        raise KeyError(f"الكلمة '{word}' غير موجودة في المفردات")

    word_vec = embeddings[vocab.word2idx[word]]
    similarities = torch.nn.functional.cosine_similarity(
        embeddings, word_vec.unsqueeze(0), dim=1
    )
    top_indices = torch.topk(similarities, top_k + 1).indices.tolist()

    results = [
        (vocab.idx2word[idx], similarities[idx].item())
        for idx in top_indices
        if vocab.idx2word[idx] != word
    ][:top_k]
    return results


def analogy(
    a: str, b: str, c: str, vocab: Vocabulary, embeddings: Tensor, top_k: int = 3
) -> list[tuple[str, float]]:
    """يحل المعادلة: a is to b as c is to ?  (مثال: رجل:ملك :: امرأة:؟)"""
    vec = (
        embeddings[vocab.word2idx[b]]
        - embeddings[vocab.word2idx[a]]
        + embeddings[vocab.word2idx[c]]
    )
    similarities = torch.nn.functional.cosine_similarity(
        embeddings, vec.unsqueeze(0), dim=1
    )
    excluded = {vocab.word2idx[a], vocab.word2idx[b], vocab.word2idx[c]}
    top_indices = torch.topk(similarities, top_k + len(excluded)).indices.tolist()
    results = [
        (vocab.idx2word[idx], similarities[idx].item())
        for idx in top_indices
        if idx not in excluded
    ][:top_k]
    return results


# مثال استخدام (بافتراض تدريب كافٍ على مجموعة بيانات كبيرة بما يكفي):
# model, vocab = train_skipgram(large_corpus, embedding_dim=100, epochs=100)
# vectors = model.get_word_vectors()
# print(most_similar("ملك", vocab, vectors))
# print(analogy("رجل", "ملك", "امرأة", vocab, vectors))  # نتوقع: ملكة

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-07-chapter-07.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# خطأ: استخدام نفس الطبقة للمركز والسياق
self.embeddings = nn.Embedding(vocab_size, dim)
score = self.embeddings(center) @ self.embeddings(context).T  # ❌ قد يسبب تآكل تدرّج غريب

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-07-chapter-07.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
vocab = Vocabulary.build(sentences, min_count=1)  # ❌ يُبقي كل كلمة ظهرت مرة واحدة فقط

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-07-chapter-07.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# خطأ مفاهيمي: توقع أن تضمين "بنك" يتغيّر حسب الجملة
vector_bank_river = word2vec_model["بنك"]   # ❌ نفس المتجه دائمًا!
vector_bank_money = word2vec_model["بنك"]   # ❌ متطابق تمامًا، رغم اختلاف المعنى!

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-07-chapter-07.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
"""
arabic_word_similarity_project.py
مشروع تطبيقي: مُكتشف تشابه دلالي عربي باستخدام Word2Vec.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# TODO 1: حمّل مجموعة نصوص عربية من ملف أو مكتبة (مثل datasets من HuggingFace)
def load_arabic_corpus(path: Path) -> list[str]:
    raise NotImplementedError("حمّل نصوصك هنا")


# TODO 2: نظّف ورمّز النصوص (استخدم simple_tokenize أو Tokenizer BPE من الفصل 6)
def preprocess(raw_texts: list[str]) -> list[list[str]]:
    raise NotImplementedError


# TODO 3: درّب النموذج باستخدام train_skipgram() المُعرَّفة في هذا الفصل
# model, vocab = train_skipgram(tokenized_sentences, embedding_dim=100, epochs=50)


# TODO 4: دالة تصوير التضمينات بعد تقليل الأبعاد
def visualize_embeddings(
    words: list[str], vocab: "Vocabulary", embeddings, output_path: str = "embeddings.png"
) -> None:
    indices = [vocab.word2idx[w] for w in words if w in vocab.word2idx]
    vectors = embeddings[indices].numpy()

    reduced = TSNE(n_components=2, random_state=42, perplexity=5).fit_transform(vectors)

    plt.figure(figsize=(10, 8))
    for i, word in enumerate(words):
        if vocab.word2idx.get(word) is None:
            continue
        x, y = reduced[i]
        plt.scatter(x, y)
        plt.annotate(word, (x, y), fontsize=12)
    plt.title("تصوّر التضمينات الدلالية (Word2Vec)")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"تم حفظ الرسم في: {output_path}")


# TODO 5 (تحدٍّ إضافي): قارن نتائج نموذجك المُدرَّب محليًا بنموذج fastText
# العربي المُدرَّب مسبقًا (Facebook AI)، وناقش الفروقات في الجودة والتغطية.
