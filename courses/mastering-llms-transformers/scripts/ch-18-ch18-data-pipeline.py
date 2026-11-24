#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — مصادر البيانات وتنظيفها
Mastering Transformers and Large Language Models — Data Pipeline

Book slug : mastering-llms-transformers
Chapter   : ch-18-ch18-data-pipeline (lesson 18)
Code files: 13 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-18-ch18-data-pipeline.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — مصادر البيانات وتنظيفها
Mastering Transformers and Large Language Models — Data Pipeline

Book slug : mastering-llms-transformers
Chapter   : ch-18-ch18-data-pipeline (lesson 18)
Code files: 12 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-18-ch18-data-pipeline.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — مصادر البيانات وتنظيفها
Mastering Transformers and Large Language Models — Data Pipeline

Book slug : mastering-llms-transformers
Chapter   : ch-18-ch18-data-pipeline (lesson 18)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-05-building-llms/ch18-data-pipeline.ipynb#cell-1
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
    ("bs4", "beautifulsoup4"),
    ("fasttext", "fasttext-wheel"),
    ("numpy", "numpy"),
    ("trafilatura", "trafilatura")
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
# [002] content/notebooks/part-05-building-llms/ch18-data-pipeline.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import re
from bs4 import BeautifulSoup
import trafilatura  # مكتبة متخصصة في استخلاص المحتوى الرئيسي من HTML

def extract_clean_text_from_html(html_content: str) -> str:
    """
    استخلاص النص الرئيسي من صفحة HTML مع تجاهل:
    - القوائم الجانبية (navigation, sidebar)
    - الإعلانات
    - التذييل والترويسة (footer, header)
    - الوسوم البرمجية (script, style)

    نستخدم trafilatura لأنها مصممة خصيصًا لهذه المهمة وتتفوق
    على BeautifulSoup البسيط في تمييز "المحتوى الرئيسي" عن الضوضاء.
    """
    # الطريقة المتقدمة: trafilatura (تُستخدم فعليًا في بناء مجموعات مثل RefinedWeb)
    extracted = trafilatura.extract(
        html_content,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
    )
    if extracted:
        return extracted

    # طريقة احتياطية بسيطة بـ BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # تنظيف الأسطر الفارغة المتكررة
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return text

def basic_arabic_text_cleaning(text: str) -> str:
    """
    تنظيف أساسي خاص بالنصوص العربية:
    - توحيد أشكال الألف (أ، إ، آ → ا) [اختياري، يُستخدم أحيانًا للتدريب لكن يُفقِد معلومة]
    - إزالة التشكيل الزائد إذا لم يكن مطلوبًا
    - إزالة التكرار المفرط للأحرف (مثال: "جميلللللل" → "جميل")
    - توحيد المسافات المتعددة
    """
    # إزالة التشكيل (التنوين والحركات) - اختياري حسب حالة الاستخدام
    arabic_diacritics = re.compile(r'[\u064B-\u0652\u0670\u0640]')
    text = arabic_diacritics.sub('', text)

    # تقليص تكرار الأحرف الزائد عن 3 مرات متتالية إلى مرتين
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # توحيد المسافات
    text = re.sub(r'\s+', ' ', text).strip()

    # إزالة الأسطر القصيرة جدًا (أقل من 3 كلمات) التي غالبًا قوائم تنقل أو إعلانات
    lines = text.split('\n')
    lines = [l for l in lines if len(l.split()) >= 3]
    return '\n'.join(lines)

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-05-building-llms/ch18-data-pipeline.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import fasttext

class LanguageFilter:
    """
    فلترة النصوص حسب اللغة باستخدام نموذج fastText المدرَّب مسبقًا
    (lid.176.bin) القادر على تمييز 176 لغة بدقة عالية وسرعة كبيرة
    (مناسب لمعالجة تريليونات الرموز).
    """
    def __init__(self, model_path: str = "lid.176.bin",
                 target_langs: list = None,
                 confidence_threshold: float = 0.65):
        self.model = fasttext.load_model(model_path)
        self.target_langs = set(target_langs or ["ar", "en"])
        self.confidence_threshold = confidence_threshold

    def predict_language(self, text: str) -> tuple:
        """يعيد (رمز اللغة، درجة الثقة)"""
        # fastText يتوقع نصًا بدون أسطر جديدة
        clean_text = text.replace("\n", " ")[:2000]  # عيّنة كافية للتنبؤ
        predictions = self.model.predict(clean_text, k=1)
        lang_code = predictions[0][0].replace("__label__", "")
        confidence = predictions[1][0]
        return lang_code, confidence

    def should_keep(self, text: str) -> bool:
        lang, confidence = self.predict_language(text)
        return lang in self.target_langs and confidence >= self.confidence_threshold

    def filter_corpus(self, documents: list) -> list:
        kept = [doc for doc in documents if self.should_keep(doc)]
        print(f"احتُفظ بـ {len(kept)}/{len(documents)} "
              f"({100*len(kept)/max(len(documents),1):.1f}%) من الوثائق")
        return kept

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-05-building-llms/ch18-data-pipeline.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import hashlib
from collections import defaultdict

def exact_deduplication(documents: list) -> list:
    """
    إزالة التكرار التام: أي وثيقتين متطابقتين حرفيًا يُحتفظ بواحدة فقط منهما.
    نستخدم SHA-256 لأنه مقاوم للتصادم (Collision-resistant) بشكل عملي.
    """
    seen_hashes = set()
    unique_documents = []

    for doc in documents:
        # تطبيع بسيط قبل الهاش: إزالة المسافات الزائدة لتفادي اعتبار
        # وثيقتين متطابقتين فعليًا مختلفتين بسبب فراغ زائد
        normalized = " ".join(doc.split())
        doc_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

        if doc_hash not in seen_hashes:
            seen_hashes.add(doc_hash)
            unique_documents.append(doc)

    removed = len(documents) - len(unique_documents)
    print(f"Exact Dedup: أُزيلت {removed} وثيقة مكررة تمامًا "
          f"({100*removed/max(len(documents),1):.1f}%)")
    return unique_documents

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-05-building-llms/ch18-data-pipeline.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from collections import defaultdict
import hashlib

class MinHashDeduplicator:
    """
    إزالة التكرار التقريبي باستخدام MinHash + LSH.
    هذه هي التقنية الفعلية المستخدمة في بناء مجموعات مثل
    RefinedWeb و FineWeb لإزالة مليارات الوثائق شبه المكررة
    بكفاءة زمنية تقارب O(n) بدلاً من O(n²).
    """
    def __init__(self, num_hashes: int = 128, num_bands: int = 16,
                 ngram_size: int = 5, similarity_threshold: float = 0.8):
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        self.ngram_size = ngram_size
        self.similarity_threshold = similarity_threshold

        # توليد معاملات دوال الهاش العشوائية (a، b لكل دالة هاش خطية)
        # نستخدم صيغة: h_i(x) = (a_i * x + b_i) mod p
        rng = np.random.RandomState(42)
        self.max_prime = (1 << 61) - 1  # عدد أولي كبير (Mersenne Prime)
        self.a_params = rng.randint(1, self.max_prime, size=num_hashes)
        self.b_params = rng.randint(0, self.max_prime, size=num_hashes)

    def _get_ngrams(self, text: str) -> set:
        """تحويل النص إلى مجموعة n-grams على مستوى الكلمات"""
        words = text.split()
        if len(words) < self.ngram_size:
            return {tuple(words)}
        return {tuple(words[i:i + self.ngram_size])
                for i in range(len(words) - self.ngram_size + 1)}

    def _hash_ngram(self, ngram: tuple) -> int:
        """تحويل n-gram إلى عدد صحيح عبر MD5 (أسرع من SHA للأغراض الداخلية)"""
        ngram_str = " ".join(ngram)
        return int(hashlib.md5(ngram_str.encode("utf-8")).hexdigest(), 16)

    def compute_minhash_signature(self, text: str) -> np.ndarray:
        """
        حساب توقيع MinHash لوثيقة: مصفوفة من num_hashes قيمة،
        كل قيمة هي أصغر قيمة هاش بين كل الـ n-grams للوثيقة
        تحت دالة هاش معينة.
        """
        ngrams = self._get_ngrams(text)
        if not ngrams:
            return np.full(self.num_hashes, self.max_prime, dtype=np.int64)

        ngram_hashes = np.array([self._hash_ngram(ng) for ng in ngrams])
        # تطبيق كل دالة هاش خطية على كل قيم الـ ngram_hashes دفعة واحدة
        signature = np.full(self.num_hashes, self.max_prime, dtype=np.int64)
        for i in range(self.num_hashes):
            hashed_values = (self.a_params[i] * ngram_hashes +
                              self.b_params[i]) % self.max_prime
            signature[i] = hashed_values.min()
        return signature

    def estimate_jaccard_similarity(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """تقدير تشابه Jaccard من نسبة تطابق عناصر التوقيعين"""
        return float(np.mean(sig1 == sig2))

    def build_lsh_buckets(self, signatures: list) -> dict:
        """
        تقنية LSH: نقسّم كل توقيع إلى (num_bands) نطاقات (Bands)،
        كل نطاق يحتوي (rows_per_band) قيمة. وثيقتان تقعان في نفس
        الدلو (Bucket) لأي نطاق تصبحان "مرشحتين" للمقارنة الدقيقة.
        هذا يقلل عدد المقارنات من O(n²) إلى تقريبًا O(n).
        """
        buckets = defaultdict(list)
        for doc_idx, signature in enumerate(signatures):
            for band_idx in range(self.num_bands):
                start = band_idx * self.rows_per_band
                end = start + self.rows_per_band
                band_signature = tuple(signature[start:end])
                bucket_key = (band_idx, hash(band_signature))
                buckets[bucket_key].append(doc_idx)
        return buckets

    def deduplicate(self, documents: list) -> list:
        """خط الإنتاج الكامل: حساب التواقيع → بناء الدلاء → تصفية المكررات"""
        print(f"حساب تواقيع MinHash لـ {len(documents)} وثيقة...")
        signatures = [self.compute_minhash_signature(doc) for doc in documents]

        print("بناء دلاء LSH...")
        buckets = self.build_lsh_buckets(signatures)

        # تجميع أزواج الوثائق المرشحة للمقارنة (ظهرت معًا في دلو واحد على الأقل)
        candidate_pairs = set()
        for bucket_docs in buckets.values():
            if len(bucket_docs) > 1:
                for i in range(len(bucket_docs)):
                    for j in range(i + 1, len(bucket_docs)):
                        pair = tuple(sorted([bucket_docs[i], bucket_docs[j]]))
                        candidate_pairs.add(pair)

        print(f"عدد الأزواج المرشحة للمقارنة: {len(candidate_pairs)} "
              f"(بدلاً من {len(documents)*(len(documents)-1)//2} في الطريقة الساذجة)")

        # تحديد الوثائق المكررة فعليًا عبر حساب التشابه الدقيق للمرشحين فقط
        to_remove = set()
        for i, j in candidate_pairs:
            if i in to_remove or j in to_remove:
                continue
            similarity = self.estimate_jaccard_similarity(signatures[i], signatures[j])
            if similarity >= self.similarity_threshold:
                # نحتفظ بالوثيقة الأطول عادة (غالبًا الأصلية الكاملة)
                if len(documents[i]) >= len(documents[j]):
                    to_remove.add(j)
                else:
                    to_remove.add(i)

        unique_documents = [doc for idx, doc in enumerate(documents)
                             if idx not in to_remove]
        print(f"Fuzzy Dedup: أُزيلت {len(to_remove)} وثيقة شبه مكررة "
              f"({100*len(to_remove)/max(len(documents),1):.1f}%)")
        return unique_documents

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-05-building-llms/ch18-data-pipeline.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
class HeuristicQualityFilter:
    """
    مجموعة قواعد بسيطة لكن فعّالة جدًا لفلترة النصوص منخفضة الجودة،
    مستوحاة من أوراق Gopher وC4 وFineWeb. هذه القواعد رخيصة حسابيًا
    وتُطبَّق كخط دفاع أول قبل المرشّحات الأثقل (نماذج التصنيف).
    """
    def __init__(self,
                 min_words: int = 50,
                 max_words: int = 100_000,
                 min_avg_word_length: float = 2.0,
                 max_avg_word_length: float = 15.0,
                 max_symbol_to_word_ratio: float = 0.3,
                 max_repeated_line_fraction: float = 0.3):
        self.min_words = min_words
        self.max_words = max_words
        self.min_avg_word_length = min_avg_word_length
        self.max_avg_word_length = max_avg_word_length
        self.max_symbol_to_word_ratio = max_symbol_to_word_ratio
        self.max_repeated_line_fraction = max_repeated_line_fraction

    def check_document(self, text: str) -> tuple:
        """
        يعيد (يُقبَل: bool, سبب الرفض إن وُجد: str)
        نرتب الفحوصات من الأرخص للأغلى حسابيًا (fail-fast).
        """
        words = text.split()
        num_words = len(words)

        # 1. فحص طول الوثيقة
        if num_words < self.min_words:
            return False, f"قصيرة جدًا ({num_words} كلمة)"
        if num_words > self.max_words:
            return False, f"طويلة جدًا ({num_words} كلمة)"

        # 2. متوسط طول الكلمة (كلمات قصيرة جدًا = سبام/قوائم، طويلة جدًا = كود ملتصق/أخطاء)
        avg_word_len = sum(len(w) for w in words) / num_words
        if not (self.min_avg_word_length <= avg_word_len <= self.max_avg_word_length):
            return False, f"متوسط طول الكلمة غير طبيعي ({avg_word_len:.1f})"

        # 3. نسبة الرموز غير الأبجدية (علامات ترقيم، رموز خاصة) إلى الكلمات
        num_symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
        symbol_ratio = num_symbols / max(num_words, 1)
        if symbol_ratio > self.max_symbol_to_word_ratio:
            return False, f"نسبة رموز مرتفعة ({symbol_ratio:.2f})"

        # 4. نسبة الأسطر المكررة داخل نفس الوثيقة (مؤشر على قوائم/قوالب متكررة)
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if lines:
            unique_lines = len(set(lines))
            repeated_fraction = 1 - (unique_lines / len(lines))
            if repeated_fraction > self.max_repeated_line_fraction:
                return False, f"نسبة تكرار أسطر مرتفعة ({repeated_fraction:.2f})"

        # 5. فحص وجود كلمة واحدة على الأقل تحتوي حرفًا أبجديًا (رفض الوثائق الرقمية بحتة)
        has_alpha_word = any(any(c.isalpha() for c in w) for w in words)
        if not has_alpha_word:
            return False, "لا تحتوي كلمات نصية فعلية"

        return True, None

    def filter_corpus(self, documents: list) -> tuple:
        kept, rejected_reasons = [], defaultdict(int)
        for doc in documents:
            accepted, reason = self.check_document(doc)
            if accepted:
                kept.append(doc)
            else:
                rejected_reasons[reason.split("(")[0].strip()] += 1

        print(f"Heuristic Filter: احتُفظ بـ {len(kept)}/{len(documents)}")
        for reason, count in sorted(rejected_reasons.items(),
                                     key=lambda x: -x[1]):
            print(f"   - رُفضت {count} وثيقة بسبب: {reason}")
        return kept, rejected_reasons

# ────────────────────────────────────────────────────────────────────────
# [007] content/notebooks/part-05-building-llms/ch18-data-pipeline.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import fasttext

class QualityClassifierFilter:
    """
    نموذج فلترة جودة مبني على fastText، يُحاكي منهجية FineWeb-Edu:
    يُدرَّب على تمييز النصوص "التعليمية عالية القيمة" من بقية نصوص الويب.
    """
    def __init__(self, positive_samples: list, negative_samples: list):
        """
        positive_samples: نصوص عالية الجودة (ويكيبيديا، كتب، مقالات أكاديمية)
        negative_samples: عينة عشوائية من الويب الخام (غير مفلترة)
        """
        self.training_file = "_quality_classifier_train.txt"
        self._prepare_training_data(positive_samples, negative_samples)
        self.model = fasttext.train_supervised(
            input=self.training_file,
            lr=0.5, epoch=25, wordNgrams=2, dim=100,
        )

    def _prepare_training_data(self, positive, negative):
        with open(self.training_file, "w", encoding="utf-8") as f:
            for text in positive:
                clean = text.replace("\n", " ")
                f.write(f"__label__high_quality {clean}\n")
            for text in negative:
                clean = text.replace("\n", " ")
                f.write(f"__label__low_quality {clean}\n")

    def score_document(self, text: str) -> float:
        """يعيد درجة بين 0 و1، حيث 1 تعني ثقة كاملة بأنها عالية الجودة"""
        clean_text = text.replace("\n", " ")[:2000]
        labels, probs = self.model.predict(clean_text, k=1)
        label = labels[0].replace("__label__", "")
        prob = probs[0]
        return prob if label == "high_quality" else (1 - prob)

    def filter_corpus(self, documents: list, keep_top_fraction: float = 0.5) -> list:
        """الاحتفاظ بأعلى نسبة مئوية من الوثائق حسب درجة الجودة"""
        scored = [(doc, self.score_document(doc)) for doc in documents]
        scored.sort(key=lambda x: -x[1])
        num_to_keep = int(len(scored) * keep_top_fraction)
        kept = [doc for doc, score in scored[:num_to_keep]]
        print(f"Quality Classifier: احتُفظ بأعلى {keep_top_fraction*100:.0f}% "
              f"= {len(kept)} وثيقة")
        return kept

# ────────────────────────────────────────────────────────────────────────
# [008] content/notebooks/part-05-building-llms/ch18-data-pipeline.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import re

class PIIRemover:
    """
    إزالة أو إخفاء المعلومات الشخصية القابلة للتحديد (PII) من النصوص
    قبل التدريب، للامتثال للخصوصية ومنع النموذج من "حفظ" بيانات حساسة.
    نستخدم أنماط Regex للحالات الواضحة، ويمكن تعزيزها بنموذج NER
    (Named Entity Recognition) للكشف عن الأسماء الشخصية والعناوين.
    """
    def __init__(self):
        self.patterns = {
            "EMAIL": re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ),
            "PHONE_INTL": re.compile(
                r'\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}'
            ),
            "SSN_US": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "CREDIT_CARD": re.compile(
                r'\b(?:\d[ -]*?){13,16}\b'
            ),
            "IP_ADDRESS": re.compile(
                r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
            ),
            "SAUDI_NATIONAL_ID": re.compile(r'\b[12]\d{9}\b'),
        }
        self.replacement_tokens = {
            "EMAIL": "[EMAIL]",
            "PHONE_INTL": "[PHONE]",
            "SSN_US": "[SSN]",
            "CREDIT_CARD": "[CREDIT_CARD]",
            "IP_ADDRESS": "[IP_ADDRESS]",
            "SAUDI_NATIONAL_ID": "[NATIONAL_ID]",
        }

    def scrub_text(self, text: str) -> tuple:
        """يعيد (النص المنظف، عدد كل نوع تم إخفاؤه)"""
        counts = defaultdict(int)
        cleaned_text = text
        for pii_type, pattern in self.patterns.items():
            matches = pattern.findall(cleaned_text)
            counts[pii_type] = len(matches)
            cleaned_text = pattern.sub(self.replacement_tokens[pii_type],
                                         cleaned_text)
        return cleaned_text, dict(counts)

    def scrub_corpus(self, documents: list) -> list:
        total_counts = defaultdict(int)
        cleaned_docs = []
        for doc in documents:
            cleaned, counts = self.scrub_text(doc)
            cleaned_docs.append(cleaned)
            for k, v in counts.items():
                total_counts[k] += v
        print("إحصائيات إزالة PII عبر الكوربص بالكامل:")
        for pii_type, count in total_counts.items():
            if count > 0:
                print(f"   - {pii_type}: {count} حالة")
        return cleaned_docs

# ────────────────────────────────────────────────────────────────────────
# [009] content/notebooks/part-05-building-llms/ch18-data-pipeline.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
class ContaminationChecker:
    """
    يفحص ما إذا كانت وثائق التدريب تحتوي على مقاطع متطابقة أو شبه متطابقة
    من مجموعات بيانات التقييم المعروفة (benchmark test sets)، باستخدام
    تطابق n-grams طويلة (13-gram فأكثر) وهي الطريقة المعتمدة في تقارير
    GPT-4 وLLaMA الفنية.
    """
    def __init__(self, benchmark_texts: list, ngram_size: int = 13):
        self.ngram_size = ngram_size
        self.benchmark_ngrams = set()
        for text in benchmark_texts:
            self.benchmark_ngrams.update(self._get_ngrams(text))

    def _get_ngrams(self, text: str) -> set:
        words = text.lower().split()
        if len(words) < self.ngram_size:
            return {tuple(words)}
        return {tuple(words[i:i + self.ngram_size])
                for i in range(len(words) - self.ngram_size + 1)}

    def is_contaminated(self, document: str, overlap_threshold: int = 1) -> bool:
        """يعيد True إذا احتوت الوثيقة على عدد n-grams مشتركة مع benchmark
        يساوي أو يتجاوز overlap_threshold"""
        doc_ngrams = self._get_ngrams(document)
        overlap = doc_ngrams & self.benchmark_ngrams
        return len(overlap) >= overlap_threshold

    def filter_corpus(self, documents: list) -> list:
        clean_docs = [doc for doc in documents if not self.is_contaminated(doc)]
        removed = len(documents) - len(clean_docs)
        print(f"Contamination Check: أُزيلت {removed} وثيقة ملوَّثة "
              f"من مجموعات التقييم")
        return clean_docs

# ────────────────────────────────────────────────────────────────────────
# [010] content/notebooks/part-05-building-llms/ch18-data-pipeline.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
raw_documents = [
    # وثيقة 1: نص جيد عن الذكاء الاصطناعي
    "الذكاء الاصطناعي هو مجال علمي يهتم بتطوير أنظمة قادرة على أداء "
    "مهام تتطلب عادة ذكاءً بشريًا مثل الفهم والتعلم واتخاذ القرار. "
    "شهد هذا المجال تطورًا هائلاً في العقد الأخير بفضل التعلم العميق.",

    # وثيقة 2: تكرار شبه تام للوثيقة 1 (تغيير كلمة واحدة فقط)
    "الذكاء الاصطناعي هو مجال علمي يهتم بتطوير أنظمة قادرة على أداء "
    "مهام تتطلب عادة ذكاءً بشريًا مثل الفهم والتعلم واتخاذ القرارات. "
    "شهد هذا المجال تطورًا هائلاً في العقد الأخير بفضل التعلم العميق.",

    # وثيقة 3: نص قصير جدًا (سيُرفَض بفلتر الطول)
    "مرحبا بك",

    # وثيقة 4: يحتوي PII (بريد إلكتروني ورقم هاتف)
    "للتواصل معنا يرجى إرسال بريد إلكتروني إلى ahmed.sales@example.com "
    "أو الاتصال على الرقم +966-50-123-4567 خلال ساعات الدوام الرسمي من "
    "الأحد إلى الخميس للاستفسار عن خدماتنا المتنوعة في مجال البرمجيات.",

    # وثيقة 5: سبام/قائمة كلمات مفتاحية (سيُرفَض بفلتر نسبة الرموز)
    "!!!خصم خصم خصم!!! اشتري الآن ### عرض محدود $$$ سارع بالطلب !!! "
    "### اتصل الآن $$$ !!! خصم 90% ### !!!",

    # وثيقة 6: نص جيد ومختلف تمامًا
    "تُعد معالجة اللغات الطبيعية أحد أهم فروع الذكاء الاصطناعي، وتتضمن "
    "مهامًا متنوعة مثل الترجمة الآلية والتلخيص والإجابة على الأسئلة. "
    "ساهمت نماذج المحوّلات في إحداث نقلة نوعية في جودة هذه التطبيقات.",
]

print("=" * 60)
print(f"البدء بـ {len(raw_documents)} وثيقة خام")
print("=" * 60)

# الخطوة 1: فلترة الجودة Heuristic (يُزيل الوثيقة 3 والوثيقة 5)
quality_filter = HeuristicQualityFilter(min_words=5, max_symbol_to_word_ratio=0.3)
step1_docs, _ = quality_filter.filter_corpus(raw_documents)
# → يبقى: [1, 2, 4, 6]

# الخطوة 2: إزالة PII (يُنظّف الوثيقة 4 دون حذفها)
pii_remover = PIIRemover()
step2_docs = pii_remover.scrub_corpus(step1_docs)
# → الوثيقة 4 أصبحت: "... [EMAIL] ... [PHONE] ..."

# الخطوة 3: Fuzzy Deduplication (يُزيل إحدى الوثيقتين 1/2 لتشابههما الشديد)
deduplicator = MinHashDeduplicator(num_hashes=64, num_bands=8,
                                     ngram_size=3, similarity_threshold=0.7)
step3_docs = deduplicator.deduplicate(step2_docs)
# → يبقى 3 وثائق نهائية: [وثيقة 1 أو 2، الوثيقة 4 منظفة، الوثيقة 6]

print("\n" + "=" * 60)
print(f"النتيجة النهائية: {len(step3_docs)} وثيقة نظيفة من أصل {len(raw_documents)}")
print("=" * 60)
for i, doc in enumerate(step3_docs, 1):
    print(f"\n[{i}] {doc[:100]}...")

# ────────────────────────────────────────────────────────────────────────
# [011] content/notebooks/part-05-building-llms/ch18-data-pipeline.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-18-ch18-data-pipeline.ipynb#cell-1
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
    ("bs4", "beautifulsoup4"),
    ("fasttext", "fasttext-wheel"),
    ("numpy", "numpy"),
    ("trafilatura", "trafilatura")
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-18-ch18-data-pipeline.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import re
from bs4 import BeautifulSoup
import trafilatura  # مكتبة متخصصة في استخلاص المحتوى الرئيسي من HTML

def extract_clean_text_from_html(html_content: str) -> str:
    """
    استخلاص النص الرئيسي من صفحة HTML مع تجاهل:
    - القوائم الجانبية (navigation, sidebar)
    - الإعلانات
    - التذييل والترويسة (footer, header)
    - الوسوم البرمجية (script, style)

    نستخدم trafilatura لأنها مصممة خصيصًا لهذه المهمة وتتفوق
    على BeautifulSoup البسيط في تمييز "المحتوى الرئيسي" عن الضوضاء.
    """
    # الطريقة المتقدمة: trafilatura (تُستخدم فعليًا في بناء مجموعات مثل RefinedWeb)
    extracted = trafilatura.extract(
        html_content,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
    )
    if extracted:
        return extracted

    # طريقة احتياطية بسيطة بـ BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # تنظيف الأسطر الفارغة المتكررة
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return text

def basic_arabic_text_cleaning(text: str) -> str:
    """
    تنظيف أساسي خاص بالنصوص العربية:
    - توحيد أشكال الألف (أ، إ، آ → ا) [اختياري، يُستخدم أحيانًا للتدريب لكن يُفقِد معلومة]
    - إزالة التشكيل الزائد إذا لم يكن مطلوبًا
    - إزالة التكرار المفرط للأحرف (مثال: "جميلللللل" → "جميل")
    - توحيد المسافات المتعددة
    """
    # إزالة التشكيل (التنوين والحركات) - اختياري حسب حالة الاستخدام
    arabic_diacritics = re.compile(r'[\u064B-\u0652\u0670\u0640]')
    text = arabic_diacritics.sub('', text)

    # تقليص تكرار الأحرف الزائد عن 3 مرات متتالية إلى مرتين
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # توحيد المسافات
    text = re.sub(r'\s+', ' ', text).strip()

    # إزالة الأسطر القصيرة جدًا (أقل من 3 كلمات) التي غالبًا قوائم تنقل أو إعلانات
    lines = text.split('\n')
    lines = [l for l in lines if len(l.split()) >= 3]
    return '\n'.join(lines)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-18-ch18-data-pipeline.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import fasttext

class LanguageFilter:
    """
    فلترة النصوص حسب اللغة باستخدام نموذج fastText المدرَّب مسبقًا
    (lid.176.bin) القادر على تمييز 176 لغة بدقة عالية وسرعة كبيرة
    (مناسب لمعالجة تريليونات الرموز).
    """
    def __init__(self, model_path: str = "lid.176.bin",
                 target_langs: list = None,
                 confidence_threshold: float = 0.65):
        self.model = fasttext.load_model(model_path)
        self.target_langs = set(target_langs or ["ar", "en"])
        self.confidence_threshold = confidence_threshold

    def predict_language(self, text: str) -> tuple:
        """يعيد (رمز اللغة، درجة الثقة)"""
        # fastText يتوقع نصًا بدون أسطر جديدة
        clean_text = text.replace("\n", " ")[:2000]  # عيّنة كافية للتنبؤ
        predictions = self.model.predict(clean_text, k=1)
        lang_code = predictions[0][0].replace("__label__", "")
        confidence = predictions[1][0]
        return lang_code, confidence

    def should_keep(self, text: str) -> bool:
        lang, confidence = self.predict_language(text)
        return lang in self.target_langs and confidence >= self.confidence_threshold

    def filter_corpus(self, documents: list) -> list:
        kept = [doc for doc in documents if self.should_keep(doc)]
        print(f"احتُفظ بـ {len(kept)}/{len(documents)} "
              f"({100*len(kept)/max(len(documents),1):.1f}%) من الوثائق")
        return kept

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-18-ch18-data-pipeline.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import hashlib
from collections import defaultdict

def exact_deduplication(documents: list) -> list:
    """
    إزالة التكرار التام: أي وثيقتين متطابقتين حرفيًا يُحتفظ بواحدة فقط منهما.
    نستخدم SHA-256 لأنه مقاوم للتصادم (Collision-resistant) بشكل عملي.
    """
    seen_hashes = set()
    unique_documents = []

    for doc in documents:
        # تطبيع بسيط قبل الهاش: إزالة المسافات الزائدة لتفادي اعتبار
        # وثيقتين متطابقتين فعليًا مختلفتين بسبب فراغ زائد
        normalized = " ".join(doc.split())
        doc_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

        if doc_hash not in seen_hashes:
            seen_hashes.add(doc_hash)
            unique_documents.append(doc)

    removed = len(documents) - len(unique_documents)
    print(f"Exact Dedup: أُزيلت {removed} وثيقة مكررة تمامًا "
          f"({100*removed/max(len(documents),1):.1f}%)")
    return unique_documents

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-18-ch18-data-pipeline.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from collections import defaultdict
import hashlib

class MinHashDeduplicator:
    """
    إزالة التكرار التقريبي باستخدام MinHash + LSH.
    هذه هي التقنية الفعلية المستخدمة في بناء مجموعات مثل
    RefinedWeb و FineWeb لإزالة مليارات الوثائق شبه المكررة
    بكفاءة زمنية تقارب O(n) بدلاً من O(n²).
    """
    def __init__(self, num_hashes: int = 128, num_bands: int = 16,
                 ngram_size: int = 5, similarity_threshold: float = 0.8):
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        self.ngram_size = ngram_size
        self.similarity_threshold = similarity_threshold

        # توليد معاملات دوال الهاش العشوائية (a، b لكل دالة هاش خطية)
        # نستخدم صيغة: h_i(x) = (a_i * x + b_i) mod p
        rng = np.random.RandomState(42)
        self.max_prime = (1 << 61) - 1  # عدد أولي كبير (Mersenne Prime)
        self.a_params = rng.randint(1, self.max_prime, size=num_hashes)
        self.b_params = rng.randint(0, self.max_prime, size=num_hashes)

    def _get_ngrams(self, text: str) -> set:
        """تحويل النص إلى مجموعة n-grams على مستوى الكلمات"""
        words = text.split()
        if len(words) < self.ngram_size:
            return {tuple(words)}
        return {tuple(words[i:i + self.ngram_size])
                for i in range(len(words) - self.ngram_size + 1)}

    def _hash_ngram(self, ngram: tuple) -> int:
        """تحويل n-gram إلى عدد صحيح عبر MD5 (أسرع من SHA للأغراض الداخلية)"""
        ngram_str = " ".join(ngram)
        return int(hashlib.md5(ngram_str.encode("utf-8")).hexdigest(), 16)

    def compute_minhash_signature(self, text: str) -> np.ndarray:
        """
        حساب توقيع MinHash لوثيقة: مصفوفة من num_hashes قيمة،
        كل قيمة هي أصغر قيمة هاش بين كل الـ n-grams للوثيقة
        تحت دالة هاش معينة.
        """
        ngrams = self._get_ngrams(text)
        if not ngrams:
            return np.full(self.num_hashes, self.max_prime, dtype=np.int64)

        ngram_hashes = np.array([self._hash_ngram(ng) for ng in ngrams])
        # تطبيق كل دالة هاش خطية على كل قيم الـ ngram_hashes دفعة واحدة
        signature = np.full(self.num_hashes, self.max_prime, dtype=np.int64)
        for i in range(self.num_hashes):
            hashed_values = (self.a_params[i] * ngram_hashes +
                              self.b_params[i]) % self.max_prime
            signature[i] = hashed_values.min()
        return signature

    def estimate_jaccard_similarity(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """تقدير تشابه Jaccard من نسبة تطابق عناصر التوقيعين"""
        return float(np.mean(sig1 == sig2))

    def build_lsh_buckets(self, signatures: list) -> dict:
        """
        تقنية LSH: نقسّم كل توقيع إلى (num_bands) نطاقات (Bands)،
        كل نطاق يحتوي (rows_per_band) قيمة. وثيقتان تقعان في نفس
        الدلو (Bucket) لأي نطاق تصبحان "مرشحتين" للمقارنة الدقيقة.
        هذا يقلل عدد المقارنات من O(n²) إلى تقريبًا O(n).
        """
        buckets = defaultdict(list)
        for doc_idx, signature in enumerate(signatures):
            for band_idx in range(self.num_bands):
                start = band_idx * self.rows_per_band
                end = start + self.rows_per_band
                band_signature = tuple(signature[start:end])
                bucket_key = (band_idx, hash(band_signature))
                buckets[bucket_key].append(doc_idx)
        return buckets

    def deduplicate(self, documents: list) -> list:
        """خط الإنتاج الكامل: حساب التواقيع → بناء الدلاء → تصفية المكررات"""
        print(f"حساب تواقيع MinHash لـ {len(documents)} وثيقة...")
        signatures = [self.compute_minhash_signature(doc) for doc in documents]

        print("بناء دلاء LSH...")
        buckets = self.build_lsh_buckets(signatures)

        # تجميع أزواج الوثائق المرشحة للمقارنة (ظهرت معًا في دلو واحد على الأقل)
        candidate_pairs = set()
        for bucket_docs in buckets.values():
            if len(bucket_docs) > 1:
                for i in range(len(bucket_docs)):
                    for j in range(i + 1, len(bucket_docs)):
                        pair = tuple(sorted([bucket_docs[i], bucket_docs[j]]))
                        candidate_pairs.add(pair)

        print(f"عدد الأزواج المرشحة للمقارنة: {len(candidate_pairs)} "
              f"(بدلاً من {len(documents)*(len(documents)-1)//2} في الطريقة الساذجة)")

        # تحديد الوثائق المكررة فعليًا عبر حساب التشابه الدقيق للمرشحين فقط
        to_remove = set()
        for i, j in candidate_pairs:
            if i in to_remove or j in to_remove:
                continue
            similarity = self.estimate_jaccard_similarity(signatures[i], signatures[j])
            if similarity >= self.similarity_threshold:
                # نحتفظ بالوثيقة الأطول عادة (غالبًا الأصلية الكاملة)
                if len(documents[i]) >= len(documents[j]):
                    to_remove.add(j)
                else:
                    to_remove.add(i)

        unique_documents = [doc for idx, doc in enumerate(documents)
                             if idx not in to_remove]
        print(f"Fuzzy Dedup: أُزيلت {len(to_remove)} وثيقة شبه مكررة "
              f"({100*len(to_remove)/max(len(documents),1):.1f}%)")
        return unique_documents

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-18-ch18-data-pipeline.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
class HeuristicQualityFilter:
    """
    مجموعة قواعد بسيطة لكن فعّالة جدًا لفلترة النصوص منخفضة الجودة،
    مستوحاة من أوراق Gopher وC4 وFineWeb. هذه القواعد رخيصة حسابيًا
    وتُطبَّق كخط دفاع أول قبل المرشّحات الأثقل (نماذج التصنيف).
    """
    def __init__(self,
                 min_words: int = 50,
                 max_words: int = 100_000,
                 min_avg_word_length: float = 2.0,
                 max_avg_word_length: float = 15.0,
                 max_symbol_to_word_ratio: float = 0.3,
                 max_repeated_line_fraction: float = 0.3):
        self.min_words = min_words
        self.max_words = max_words
        self.min_avg_word_length = min_avg_word_length
        self.max_avg_word_length = max_avg_word_length
        self.max_symbol_to_word_ratio = max_symbol_to_word_ratio
        self.max_repeated_line_fraction = max_repeated_line_fraction

    def check_document(self, text: str) -> tuple:
        """
        يعيد (يُقبَل: bool, سبب الرفض إن وُجد: str)
        نرتب الفحوصات من الأرخص للأغلى حسابيًا (fail-fast).
        """
        words = text.split()
        num_words = len(words)

        # 1. فحص طول الوثيقة
        if num_words < self.min_words:
            return False, f"قصيرة جدًا ({num_words} كلمة)"
        if num_words > self.max_words:
            return False, f"طويلة جدًا ({num_words} كلمة)"

        # 2. متوسط طول الكلمة (كلمات قصيرة جدًا = سبام/قوائم، طويلة جدًا = كود ملتصق/أخطاء)
        avg_word_len = sum(len(w) for w in words) / num_words
        if not (self.min_avg_word_length <= avg_word_len <= self.max_avg_word_length):
            return False, f"متوسط طول الكلمة غير طبيعي ({avg_word_len:.1f})"

        # 3. نسبة الرموز غير الأبجدية (علامات ترقيم، رموز خاصة) إلى الكلمات
        num_symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
        symbol_ratio = num_symbols / max(num_words, 1)
        if symbol_ratio > self.max_symbol_to_word_ratio:
            return False, f"نسبة رموز مرتفعة ({symbol_ratio:.2f})"

        # 4. نسبة الأسطر المكررة داخل نفس الوثيقة (مؤشر على قوائم/قوالب متكررة)
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if lines:
            unique_lines = len(set(lines))
            repeated_fraction = 1 - (unique_lines / len(lines))
            if repeated_fraction > self.max_repeated_line_fraction:
                return False, f"نسبة تكرار أسطر مرتفعة ({repeated_fraction:.2f})"

        # 5. فحص وجود كلمة واحدة على الأقل تحتوي حرفًا أبجديًا (رفض الوثائق الرقمية بحتة)
        has_alpha_word = any(any(c.isalpha() for c in w) for w in words)
        if not has_alpha_word:
            return False, "لا تحتوي كلمات نصية فعلية"

        return True, None

    def filter_corpus(self, documents: list) -> tuple:
        kept, rejected_reasons = [], defaultdict(int)
        for doc in documents:
            accepted, reason = self.check_document(doc)
            if accepted:
                kept.append(doc)
            else:
                rejected_reasons[reason.split("(")[0].strip()] += 1

        print(f"Heuristic Filter: احتُفظ بـ {len(kept)}/{len(documents)}")
        for reason, count in sorted(rejected_reasons.items(),
                                     key=lambda x: -x[1]):
            print(f"   - رُفضت {count} وثيقة بسبب: {reason}")
        return kept, rejected_reasons

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-llms-transformers/notebooks/ch-18-ch18-data-pipeline.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import fasttext

class QualityClassifierFilter:
    """
    نموذج فلترة جودة مبني على fastText، يُحاكي منهجية FineWeb-Edu:
    يُدرَّب على تمييز النصوص "التعليمية عالية القيمة" من بقية نصوص الويب.
    """
    def __init__(self, positive_samples: list, negative_samples: list):
        """
        positive_samples: نصوص عالية الجودة (ويكيبيديا، كتب، مقالات أكاديمية)
        negative_samples: عينة عشوائية من الويب الخام (غير مفلترة)
        """
        self.training_file = "_quality_classifier_train.txt"
        self._prepare_training_data(positive_samples, negative_samples)
        self.model = fasttext.train_supervised(
            input=self.training_file,
            lr=0.5, epoch=25, wordNgrams=2, dim=100,
        )

    def _prepare_training_data(self, positive, negative):
        with open(self.training_file, "w", encoding="utf-8") as f:
            for text in positive:
                clean = text.replace("\n", " ")
                f.write(f"__label__high_quality {clean}\n")
            for text in negative:
                clean = text.replace("\n", " ")
                f.write(f"__label__low_quality {clean}\n")

    def score_document(self, text: str) -> float:
        """يعيد درجة بين 0 و1، حيث 1 تعني ثقة كاملة بأنها عالية الجودة"""
        clean_text = text.replace("\n", " ")[:2000]
        labels, probs = self.model.predict(clean_text, k=1)
        label = labels[0].replace("__label__", "")
        prob = probs[0]
        return prob if label == "high_quality" else (1 - prob)

    def filter_corpus(self, documents: list, keep_top_fraction: float = 0.5) -> list:
        """الاحتفاظ بأعلى نسبة مئوية من الوثائق حسب درجة الجودة"""
        scored = [(doc, self.score_document(doc)) for doc in documents]
        scored.sort(key=lambda x: -x[1])
        num_to_keep = int(len(scored) * keep_top_fraction)
        kept = [doc for doc, score in scored[:num_to_keep]]
        print(f"Quality Classifier: احتُفظ بأعلى {keep_top_fraction*100:.0f}% "
              f"= {len(kept)} وثيقة")
        return kept

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-llms-transformers/notebooks/ch-18-ch18-data-pipeline.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import re

class PIIRemover:
    """
    إزالة أو إخفاء المعلومات الشخصية القابلة للتحديد (PII) من النصوص
    قبل التدريب، للامتثال للخصوصية ومنع النموذج من "حفظ" بيانات حساسة.
    نستخدم أنماط Regex للحالات الواضحة، ويمكن تعزيزها بنموذج NER
    (Named Entity Recognition) للكشف عن الأسماء الشخصية والعناوين.
    """
    def __init__(self):
        self.patterns = {
            "EMAIL": re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ),
            "PHONE_INTL": re.compile(
                r'\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}'
            ),
            "SSN_US": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "CREDIT_CARD": re.compile(
                r'\b(?:\d[ -]*?){13,16}\b'
            ),
            "IP_ADDRESS": re.compile(
                r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
            ),
            "SAUDI_NATIONAL_ID": re.compile(r'\b[12]\d{9}\b'),
        }
        self.replacement_tokens = {
            "EMAIL": "[EMAIL]",
            "PHONE_INTL": "[PHONE]",
            "SSN_US": "[SSN]",
            "CREDIT_CARD": "[CREDIT_CARD]",
            "IP_ADDRESS": "[IP_ADDRESS]",
            "SAUDI_NATIONAL_ID": "[NATIONAL_ID]",
        }

    def scrub_text(self, text: str) -> tuple:
        """يعيد (النص المنظف، عدد كل نوع تم إخفاؤه)"""
        counts = defaultdict(int)
        cleaned_text = text
        for pii_type, pattern in self.patterns.items():
            matches = pattern.findall(cleaned_text)
            counts[pii_type] = len(matches)
            cleaned_text = pattern.sub(self.replacement_tokens[pii_type],
                                         cleaned_text)
        return cleaned_text, dict(counts)

    def scrub_corpus(self, documents: list) -> list:
        total_counts = defaultdict(int)
        cleaned_docs = []
        for doc in documents:
            cleaned, counts = self.scrub_text(doc)
            cleaned_docs.append(cleaned)
            for k, v in counts.items():
                total_counts[k] += v
        print("إحصائيات إزالة PII عبر الكوربص بالكامل:")
        for pii_type, count in total_counts.items():
            if count > 0:
                print(f"   - {pii_type}: {count} حالة")
        return cleaned_docs

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/mastering-llms-transformers/notebooks/ch-18-ch18-data-pipeline.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
class ContaminationChecker:
    """
    يفحص ما إذا كانت وثائق التدريب تحتوي على مقاطع متطابقة أو شبه متطابقة
    من مجموعات بيانات التقييم المعروفة (benchmark test sets)، باستخدام
    تطابق n-grams طويلة (13-gram فأكثر) وهي الطريقة المعتمدة في تقارير
    GPT-4 وLLaMA الفنية.
    """
    def __init__(self, benchmark_texts: list, ngram_size: int = 13):
        self.ngram_size = ngram_size
        self.benchmark_ngrams = set()
        for text in benchmark_texts:
            self.benchmark_ngrams.update(self._get_ngrams(text))

    def _get_ngrams(self, text: str) -> set:
        words = text.lower().split()
        if len(words) < self.ngram_size:
            return {tuple(words)}
        return {tuple(words[i:i + self.ngram_size])
                for i in range(len(words) - self.ngram_size + 1)}

    def is_contaminated(self, document: str, overlap_threshold: int = 1) -> bool:
        """يعيد True إذا احتوت الوثيقة على عدد n-grams مشتركة مع benchmark
        يساوي أو يتجاوز overlap_threshold"""
        doc_ngrams = self._get_ngrams(document)
        overlap = doc_ngrams & self.benchmark_ngrams
        return len(overlap) >= overlap_threshold

    def filter_corpus(self, documents: list) -> list:
        clean_docs = [doc for doc in documents if not self.is_contaminated(doc)]
        removed = len(documents) - len(clean_docs)
        print(f"Contamination Check: أُزيلت {removed} وثيقة ملوَّثة "
              f"من مجموعات التقييم")
        return clean_docs

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/mastering-llms-transformers/notebooks/ch-18-ch18-data-pipeline.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
raw_documents = [
    # وثيقة 1: نص جيد عن الذكاء الاصطناعي
    "الذكاء الاصطناعي هو مجال علمي يهتم بتطوير أنظمة قادرة على أداء "
    "مهام تتطلب عادة ذكاءً بشريًا مثل الفهم والتعلم واتخاذ القرار. "
    "شهد هذا المجال تطورًا هائلاً في العقد الأخير بفضل التعلم العميق.",

    # وثيقة 2: تكرار شبه تام للوثيقة 1 (تغيير كلمة واحدة فقط)
    "الذكاء الاصطناعي هو مجال علمي يهتم بتطوير أنظمة قادرة على أداء "
    "مهام تتطلب عادة ذكاءً بشريًا مثل الفهم والتعلم واتخاذ القرارات. "
    "شهد هذا المجال تطورًا هائلاً في العقد الأخير بفضل التعلم العميق.",

    # وثيقة 3: نص قصير جدًا (سيُرفَض بفلتر الطول)
    "مرحبا بك",

    # وثيقة 4: يحتوي PII (بريد إلكتروني ورقم هاتف)
    "للتواصل معنا يرجى إرسال بريد إلكتروني إلى ahmed.sales@example.com "
    "أو الاتصال على الرقم +966-50-123-4567 خلال ساعات الدوام الرسمي من "
    "الأحد إلى الخميس للاستفسار عن خدماتنا المتنوعة في مجال البرمجيات.",

    # وثيقة 5: سبام/قائمة كلمات مفتاحية (سيُرفَض بفلتر نسبة الرموز)
    "!!!خصم خصم خصم!!! اشتري الآن ### عرض محدود $$$ سارع بالطلب !!! "
    "### اتصل الآن $$$ !!! خصم 90% ### !!!",

    # وثيقة 6: نص جيد ومختلف تمامًا
    "تُعد معالجة اللغات الطبيعية أحد أهم فروع الذكاء الاصطناعي، وتتضمن "
    "مهامًا متنوعة مثل الترجمة الآلية والتلخيص والإجابة على الأسئلة. "
    "ساهمت نماذج المحوّلات في إحداث نقلة نوعية في جودة هذه التطبيقات.",
]

print("=" * 60)
print(f"البدء بـ {len(raw_documents)} وثيقة خام")
print("=" * 60)

# الخطوة 1: فلترة الجودة Heuristic (يُزيل الوثيقة 3 والوثيقة 5)
quality_filter = HeuristicQualityFilter(min_words=5, max_symbol_to_word_ratio=0.3)
step1_docs, _ = quality_filter.filter_corpus(raw_documents)
# → يبقى: [1, 2, 4, 6]

# الخطوة 2: إزالة PII (يُنظّف الوثيقة 4 دون حذفها)
pii_remover = PIIRemover()
step2_docs = pii_remover.scrub_corpus(step1_docs)
# → الوثيقة 4 أصبحت: "... [EMAIL] ... [PHONE] ..."

# الخطوة 3: Fuzzy Deduplication (يُزيل إحدى الوثيقتين 1/2 لتشابههما الشديد)
deduplicator = MinHashDeduplicator(num_hashes=64, num_bands=8,
                                     ngram_size=3, similarity_threshold=0.7)
step3_docs = deduplicator.deduplicate(step2_docs)
# → يبقى 3 وثائق نهائية: [وثيقة 1 أو 2، الوثيقة 4 منظفة، الوثيقة 6]

print("\n" + "=" * 60)
print(f"النتيجة النهائية: {len(step3_docs)} وثيقة نظيفة من أصل {len(raw_documents)}")
print("=" * 60)
for i, doc in enumerate(step3_docs, 1):
    print(f"\n[{i}] {doc[:100]}...")

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/mastering-llms-transformers/notebooks/ch-18-ch18-data-pipeline.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-18-ch18-data-pipeline.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — مصادر البيانات وتنظيفها
Mastering Transformers and Large Language Models — Data Pipeline

Book slug : mastering-llms-transformers
Chapter   : ch-18-ch18-data-pipeline (lesson 18)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-05-building-llms/ch18-data-pipeline.ipynb#cell-1
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
    ("bs4", "beautifulsoup4"),
    ("fasttext", "fasttext-wheel"),
    ("numpy", "numpy"),
    ("trafilatura", "trafilatura")
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
# [002] content/notebooks/part-05-building-llms/ch18-data-pipeline.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import re
from bs4 import BeautifulSoup
import trafilatura  # مكتبة متخصصة في استخلاص المحتوى الرئيسي من HTML

def extract_clean_text_from_html(html_content: str) -> str:
    """
    استخلاص النص الرئيسي من صفحة HTML مع تجاهل:
    - القوائم الجانبية (navigation, sidebar)
    - الإعلانات
    - التذييل والترويسة (footer, header)
    - الوسوم البرمجية (script, style)

    نستخدم trafilatura لأنها مصممة خصيصًا لهذه المهمة وتتفوق
    على BeautifulSoup البسيط في تمييز "المحتوى الرئيسي" عن الضوضاء.
    """
    # الطريقة المتقدمة: trafilatura (تُستخدم فعليًا في بناء مجموعات مثل RefinedWeb)
    extracted = trafilatura.extract(
        html_content,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
    )
    if extracted:
        return extracted

    # طريقة احتياطية بسيطة بـ BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # تنظيف الأسطر الفارغة المتكررة
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return text

def basic_arabic_text_cleaning(text: str) -> str:
    """
    تنظيف أساسي خاص بالنصوص العربية:
    - توحيد أشكال الألف (أ، إ، آ → ا) [اختياري، يُستخدم أحيانًا للتدريب لكن يُفقِد معلومة]
    - إزالة التشكيل الزائد إذا لم يكن مطلوبًا
    - إزالة التكرار المفرط للأحرف (مثال: "جميلللللل" → "جميل")
    - توحيد المسافات المتعددة
    """
    # إزالة التشكيل (التنوين والحركات) - اختياري حسب حالة الاستخدام
    arabic_diacritics = re.compile(r'[\u064B-\u0652\u0670\u0640]')
    text = arabic_diacritics.sub('', text)

    # تقليص تكرار الأحرف الزائد عن 3 مرات متتالية إلى مرتين
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # توحيد المسافات
    text = re.sub(r'\s+', ' ', text).strip()

    # إزالة الأسطر القصيرة جدًا (أقل من 3 كلمات) التي غالبًا قوائم تنقل أو إعلانات
    lines = text.split('\n')
    lines = [l for l in lines if len(l.split()) >= 3]
    return '\n'.join(lines)

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-05-building-llms/ch18-data-pipeline.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import fasttext

class LanguageFilter:
    """
    فلترة النصوص حسب اللغة باستخدام نموذج fastText المدرَّب مسبقًا
    (lid.176.bin) القادر على تمييز 176 لغة بدقة عالية وسرعة كبيرة
    (مناسب لمعالجة تريليونات الرموز).
    """
    def __init__(self, model_path: str = "lid.176.bin",
                 target_langs: list = None,
                 confidence_threshold: float = 0.65):
        self.model = fasttext.load_model(model_path)
        self.target_langs = set(target_langs or ["ar", "en"])
        self.confidence_threshold = confidence_threshold

    def predict_language(self, text: str) -> tuple:
        """يعيد (رمز اللغة، درجة الثقة)"""
        # fastText يتوقع نصًا بدون أسطر جديدة
        clean_text = text.replace("\n", " ")[:2000]  # عيّنة كافية للتنبؤ
        predictions = self.model.predict(clean_text, k=1)
        lang_code = predictions[0][0].replace("__label__", "")
        confidence = predictions[1][0]
        return lang_code, confidence

    def should_keep(self, text: str) -> bool:
        lang, confidence = self.predict_language(text)
        return lang in self.target_langs and confidence >= self.confidence_threshold

    def filter_corpus(self, documents: list) -> list:
        kept = [doc for doc in documents if self.should_keep(doc)]
        print(f"احتُفظ بـ {len(kept)}/{len(documents)} "
              f"({100*len(kept)/max(len(documents),1):.1f}%) من الوثائق")
        return kept

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-05-building-llms/ch18-data-pipeline.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import hashlib
from collections import defaultdict

def exact_deduplication(documents: list) -> list:
    """
    إزالة التكرار التام: أي وثيقتين متطابقتين حرفيًا يُحتفظ بواحدة فقط منهما.
    نستخدم SHA-256 لأنه مقاوم للتصادم (Collision-resistant) بشكل عملي.
    """
    seen_hashes = set()
    unique_documents = []

    for doc in documents:
        # تطبيع بسيط قبل الهاش: إزالة المسافات الزائدة لتفادي اعتبار
        # وثيقتين متطابقتين فعليًا مختلفتين بسبب فراغ زائد
        normalized = " ".join(doc.split())
        doc_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

        if doc_hash not in seen_hashes:
            seen_hashes.add(doc_hash)
            unique_documents.append(doc)

    removed = len(documents) - len(unique_documents)
    print(f"Exact Dedup: أُزيلت {removed} وثيقة مكررة تمامًا "
          f"({100*removed/max(len(documents),1):.1f}%)")
    return unique_documents

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-05-building-llms/ch18-data-pipeline.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from collections import defaultdict
import hashlib

class MinHashDeduplicator:
    """
    إزالة التكرار التقريبي باستخدام MinHash + LSH.
    هذه هي التقنية الفعلية المستخدمة في بناء مجموعات مثل
    RefinedWeb و FineWeb لإزالة مليارات الوثائق شبه المكررة
    بكفاءة زمنية تقارب O(n) بدلاً من O(n²).
    """
    def __init__(self, num_hashes: int = 128, num_bands: int = 16,
                 ngram_size: int = 5, similarity_threshold: float = 0.8):
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        self.ngram_size = ngram_size
        self.similarity_threshold = similarity_threshold

        # توليد معاملات دوال الهاش العشوائية (a، b لكل دالة هاش خطية)
        # نستخدم صيغة: h_i(x) = (a_i * x + b_i) mod p
        rng = np.random.RandomState(42)
        self.max_prime = (1 << 61) - 1  # عدد أولي كبير (Mersenne Prime)
        self.a_params = rng.randint(1, self.max_prime, size=num_hashes)
        self.b_params = rng.randint(0, self.max_prime, size=num_hashes)

    def _get_ngrams(self, text: str) -> set:
        """تحويل النص إلى مجموعة n-grams على مستوى الكلمات"""
        words = text.split()
        if len(words) < self.ngram_size:
            return {tuple(words)}
        return {tuple(words[i:i + self.ngram_size])
                for i in range(len(words) - self.ngram_size + 1)}

    def _hash_ngram(self, ngram: tuple) -> int:
        """تحويل n-gram إلى عدد صحيح عبر MD5 (أسرع من SHA للأغراض الداخلية)"""
        ngram_str = " ".join(ngram)
        return int(hashlib.md5(ngram_str.encode("utf-8")).hexdigest(), 16)

    def compute_minhash_signature(self, text: str) -> np.ndarray:
        """
        حساب توقيع MinHash لوثيقة: مصفوفة من num_hashes قيمة،
        كل قيمة هي أصغر قيمة هاش بين كل الـ n-grams للوثيقة
        تحت دالة هاش معينة.
        """
        ngrams = self._get_ngrams(text)
        if not ngrams:
            return np.full(self.num_hashes, self.max_prime, dtype=np.int64)

        ngram_hashes = np.array([self._hash_ngram(ng) for ng in ngrams])
        # تطبيق كل دالة هاش خطية على كل قيم الـ ngram_hashes دفعة واحدة
        signature = np.full(self.num_hashes, self.max_prime, dtype=np.int64)
        for i in range(self.num_hashes):
            hashed_values = (self.a_params[i] * ngram_hashes +
                              self.b_params[i]) % self.max_prime
            signature[i] = hashed_values.min()
        return signature

    def estimate_jaccard_similarity(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """تقدير تشابه Jaccard من نسبة تطابق عناصر التوقيعين"""
        return float(np.mean(sig1 == sig2))

    def build_lsh_buckets(self, signatures: list) -> dict:
        """
        تقنية LSH: نقسّم كل توقيع إلى (num_bands) نطاقات (Bands)،
        كل نطاق يحتوي (rows_per_band) قيمة. وثيقتان تقعان في نفس
        الدلو (Bucket) لأي نطاق تصبحان "مرشحتين" للمقارنة الدقيقة.
        هذا يقلل عدد المقارنات من O(n²) إلى تقريبًا O(n).
        """
        buckets = defaultdict(list)
        for doc_idx, signature in enumerate(signatures):
            for band_idx in range(self.num_bands):
                start = band_idx * self.rows_per_band
                end = start + self.rows_per_band
                band_signature = tuple(signature[start:end])
                bucket_key = (band_idx, hash(band_signature))
                buckets[bucket_key].append(doc_idx)
        return buckets

    def deduplicate(self, documents: list) -> list:
        """خط الإنتاج الكامل: حساب التواقيع → بناء الدلاء → تصفية المكررات"""
        print(f"حساب تواقيع MinHash لـ {len(documents)} وثيقة...")
        signatures = [self.compute_minhash_signature(doc) for doc in documents]

        print("بناء دلاء LSH...")
        buckets = self.build_lsh_buckets(signatures)

        # تجميع أزواج الوثائق المرشحة للمقارنة (ظهرت معًا في دلو واحد على الأقل)
        candidate_pairs = set()
        for bucket_docs in buckets.values():
            if len(bucket_docs) > 1:
                for i in range(len(bucket_docs)):
                    for j in range(i + 1, len(bucket_docs)):
                        pair = tuple(sorted([bucket_docs[i], bucket_docs[j]]))
                        candidate_pairs.add(pair)

        print(f"عدد الأزواج المرشحة للمقارنة: {len(candidate_pairs)} "
              f"(بدلاً من {len(documents)*(len(documents)-1)//2} في الطريقة الساذجة)")

        # تحديد الوثائق المكررة فعليًا عبر حساب التشابه الدقيق للمرشحين فقط
        to_remove = set()
        for i, j in candidate_pairs:
            if i in to_remove or j in to_remove:
                continue
            similarity = self.estimate_jaccard_similarity(signatures[i], signatures[j])
            if similarity >= self.similarity_threshold:
                # نحتفظ بالوثيقة الأطول عادة (غالبًا الأصلية الكاملة)
                if len(documents[i]) >= len(documents[j]):
                    to_remove.add(j)
                else:
                    to_remove.add(i)

        unique_documents = [doc for idx, doc in enumerate(documents)
                             if idx not in to_remove]
        print(f"Fuzzy Dedup: أُزيلت {len(to_remove)} وثيقة شبه مكررة "
              f"({100*len(to_remove)/max(len(documents),1):.1f}%)")
        return unique_documents

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-05-building-llms/ch18-data-pipeline.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
class HeuristicQualityFilter:
    """
    مجموعة قواعد بسيطة لكن فعّالة جدًا لفلترة النصوص منخفضة الجودة،
    مستوحاة من أوراق Gopher وC4 وFineWeb. هذه القواعد رخيصة حسابيًا
    وتُطبَّق كخط دفاع أول قبل المرشّحات الأثقل (نماذج التصنيف).
    """
    def __init__(self,
                 min_words: int = 50,
                 max_words: int = 100_000,
                 min_avg_word_length: float = 2.0,
                 max_avg_word_length: float = 15.0,
                 max_symbol_to_word_ratio: float = 0.3,
                 max_repeated_line_fraction: float = 0.3):
        self.min_words = min_words
        self.max_words = max_words
        self.min_avg_word_length = min_avg_word_length
        self.max_avg_word_length = max_avg_word_length
        self.max_symbol_to_word_ratio = max_symbol_to_word_ratio
        self.max_repeated_line_fraction = max_repeated_line_fraction

    def check_document(self, text: str) -> tuple:
        """
        يعيد (يُقبَل: bool, سبب الرفض إن وُجد: str)
        نرتب الفحوصات من الأرخص للأغلى حسابيًا (fail-fast).
        """
        words = text.split()
        num_words = len(words)

        # 1. فحص طول الوثيقة
        if num_words < self.min_words:
            return False, f"قصيرة جدًا ({num_words} كلمة)"
        if num_words > self.max_words:
            return False, f"طويلة جدًا ({num_words} كلمة)"

        # 2. متوسط طول الكلمة (كلمات قصيرة جدًا = سبام/قوائم، طويلة جدًا = كود ملتصق/أخطاء)
        avg_word_len = sum(len(w) for w in words) / num_words
        if not (self.min_avg_word_length <= avg_word_len <= self.max_avg_word_length):
            return False, f"متوسط طول الكلمة غير طبيعي ({avg_word_len:.1f})"

        # 3. نسبة الرموز غير الأبجدية (علامات ترقيم، رموز خاصة) إلى الكلمات
        num_symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
        symbol_ratio = num_symbols / max(num_words, 1)
        if symbol_ratio > self.max_symbol_to_word_ratio:
            return False, f"نسبة رموز مرتفعة ({symbol_ratio:.2f})"

        # 4. نسبة الأسطر المكررة داخل نفس الوثيقة (مؤشر على قوائم/قوالب متكررة)
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if lines:
            unique_lines = len(set(lines))
            repeated_fraction = 1 - (unique_lines / len(lines))
            if repeated_fraction > self.max_repeated_line_fraction:
                return False, f"نسبة تكرار أسطر مرتفعة ({repeated_fraction:.2f})"

        # 5. فحص وجود كلمة واحدة على الأقل تحتوي حرفًا أبجديًا (رفض الوثائق الرقمية بحتة)
        has_alpha_word = any(any(c.isalpha() for c in w) for w in words)
        if not has_alpha_word:
            return False, "لا تحتوي كلمات نصية فعلية"

        return True, None

    def filter_corpus(self, documents: list) -> tuple:
        kept, rejected_reasons = [], defaultdict(int)
        for doc in documents:
            accepted, reason = self.check_document(doc)
            if accepted:
                kept.append(doc)
            else:
                rejected_reasons[reason.split("(")[0].strip()] += 1

        print(f"Heuristic Filter: احتُفظ بـ {len(kept)}/{len(documents)}")
        for reason, count in sorted(rejected_reasons.items(),
                                     key=lambda x: -x[1]):
            print(f"   - رُفضت {count} وثيقة بسبب: {reason}")
        return kept, rejected_reasons

# ────────────────────────────────────────────────────────────────────────
# [007] content/notebooks/part-05-building-llms/ch18-data-pipeline.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import fasttext

class QualityClassifierFilter:
    """
    نموذج فلترة جودة مبني على fastText، يُحاكي منهجية FineWeb-Edu:
    يُدرَّب على تمييز النصوص "التعليمية عالية القيمة" من بقية نصوص الويب.
    """
    def __init__(self, positive_samples: list, negative_samples: list):
        """
        positive_samples: نصوص عالية الجودة (ويكيبيديا، كتب، مقالات أكاديمية)
        negative_samples: عينة عشوائية من الويب الخام (غير مفلترة)
        """
        self.training_file = "_quality_classifier_train.txt"
        self._prepare_training_data(positive_samples, negative_samples)
        self.model = fasttext.train_supervised(
            input=self.training_file,
            lr=0.5, epoch=25, wordNgrams=2, dim=100,
        )

    def _prepare_training_data(self, positive, negative):
        with open(self.training_file, "w", encoding="utf-8") as f:
            for text in positive:
                clean = text.replace("\n", " ")
                f.write(f"__label__high_quality {clean}\n")
            for text in negative:
                clean = text.replace("\n", " ")
                f.write(f"__label__low_quality {clean}\n")

    def score_document(self, text: str) -> float:
        """يعيد درجة بين 0 و1، حيث 1 تعني ثقة كاملة بأنها عالية الجودة"""
        clean_text = text.replace("\n", " ")[:2000]
        labels, probs = self.model.predict(clean_text, k=1)
        label = labels[0].replace("__label__", "")
        prob = probs[0]
        return prob if label == "high_quality" else (1 - prob)

    def filter_corpus(self, documents: list, keep_top_fraction: float = 0.5) -> list:
        """الاحتفاظ بأعلى نسبة مئوية من الوثائق حسب درجة الجودة"""
        scored = [(doc, self.score_document(doc)) for doc in documents]
        scored.sort(key=lambda x: -x[1])
        num_to_keep = int(len(scored) * keep_top_fraction)
        kept = [doc for doc, score in scored[:num_to_keep]]
        print(f"Quality Classifier: احتُفظ بأعلى {keep_top_fraction*100:.0f}% "
              f"= {len(kept)} وثيقة")
        return kept

# ────────────────────────────────────────────────────────────────────────
# [008] content/notebooks/part-05-building-llms/ch18-data-pipeline.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import re

class PIIRemover:
    """
    إزالة أو إخفاء المعلومات الشخصية القابلة للتحديد (PII) من النصوص
    قبل التدريب، للامتثال للخصوصية ومنع النموذج من "حفظ" بيانات حساسة.
    نستخدم أنماط Regex للحالات الواضحة، ويمكن تعزيزها بنموذج NER
    (Named Entity Recognition) للكشف عن الأسماء الشخصية والعناوين.
    """
    def __init__(self):
        self.patterns = {
            "EMAIL": re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ),
            "PHONE_INTL": re.compile(
                r'\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}'
            ),
            "SSN_US": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "CREDIT_CARD": re.compile(
                r'\b(?:\d[ -]*?){13,16}\b'
            ),
            "IP_ADDRESS": re.compile(
                r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
            ),
            "SAUDI_NATIONAL_ID": re.compile(r'\b[12]\d{9}\b'),
        }
        self.replacement_tokens = {
            "EMAIL": "[EMAIL]",
            "PHONE_INTL": "[PHONE]",
            "SSN_US": "[SSN]",
            "CREDIT_CARD": "[CREDIT_CARD]",
            "IP_ADDRESS": "[IP_ADDRESS]",
            "SAUDI_NATIONAL_ID": "[NATIONAL_ID]",
        }

    def scrub_text(self, text: str) -> tuple:
        """يعيد (النص المنظف، عدد كل نوع تم إخفاؤه)"""
        counts = defaultdict(int)
        cleaned_text = text
        for pii_type, pattern in self.patterns.items():
            matches = pattern.findall(cleaned_text)
            counts[pii_type] = len(matches)
            cleaned_text = pattern.sub(self.replacement_tokens[pii_type],
                                         cleaned_text)
        return cleaned_text, dict(counts)

    def scrub_corpus(self, documents: list) -> list:
        total_counts = defaultdict(int)
        cleaned_docs = []
        for doc in documents:
            cleaned, counts = self.scrub_text(doc)
            cleaned_docs.append(cleaned)
            for k, v in counts.items():
                total_counts[k] += v
        print("إحصائيات إزالة PII عبر الكوربص بالكامل:")
        for pii_type, count in total_counts.items():
            if count > 0:
                print(f"   - {pii_type}: {count} حالة")
        return cleaned_docs

# ────────────────────────────────────────────────────────────────────────
# [009] content/notebooks/part-05-building-llms/ch18-data-pipeline.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
class ContaminationChecker:
    """
    يفحص ما إذا كانت وثائق التدريب تحتوي على مقاطع متطابقة أو شبه متطابقة
    من مجموعات بيانات التقييم المعروفة (benchmark test sets)، باستخدام
    تطابق n-grams طويلة (13-gram فأكثر) وهي الطريقة المعتمدة في تقارير
    GPT-4 وLLaMA الفنية.
    """
    def __init__(self, benchmark_texts: list, ngram_size: int = 13):
        self.ngram_size = ngram_size
        self.benchmark_ngrams = set()
        for text in benchmark_texts:
            self.benchmark_ngrams.update(self._get_ngrams(text))

    def _get_ngrams(self, text: str) -> set:
        words = text.lower().split()
        if len(words) < self.ngram_size:
            return {tuple(words)}
        return {tuple(words[i:i + self.ngram_size])
                for i in range(len(words) - self.ngram_size + 1)}

    def is_contaminated(self, document: str, overlap_threshold: int = 1) -> bool:
        """يعيد True إذا احتوت الوثيقة على عدد n-grams مشتركة مع benchmark
        يساوي أو يتجاوز overlap_threshold"""
        doc_ngrams = self._get_ngrams(document)
        overlap = doc_ngrams & self.benchmark_ngrams
        return len(overlap) >= overlap_threshold

    def filter_corpus(self, documents: list) -> list:
        clean_docs = [doc for doc in documents if not self.is_contaminated(doc)]
        removed = len(documents) - len(clean_docs)
        print(f"Contamination Check: أُزيلت {removed} وثيقة ملوَّثة "
              f"من مجموعات التقييم")
        return clean_docs

# ────────────────────────────────────────────────────────────────────────
# [010] content/notebooks/part-05-building-llms/ch18-data-pipeline.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
raw_documents = [
    # وثيقة 1: نص جيد عن الذكاء الاصطناعي
    "الذكاء الاصطناعي هو مجال علمي يهتم بتطوير أنظمة قادرة على أداء "
    "مهام تتطلب عادة ذكاءً بشريًا مثل الفهم والتعلم واتخاذ القرار. "
    "شهد هذا المجال تطورًا هائلاً في العقد الأخير بفضل التعلم العميق.",

    # وثيقة 2: تكرار شبه تام للوثيقة 1 (تغيير كلمة واحدة فقط)
    "الذكاء الاصطناعي هو مجال علمي يهتم بتطوير أنظمة قادرة على أداء "
    "مهام تتطلب عادة ذكاءً بشريًا مثل الفهم والتعلم واتخاذ القرارات. "
    "شهد هذا المجال تطورًا هائلاً في العقد الأخير بفضل التعلم العميق.",

    # وثيقة 3: نص قصير جدًا (سيُرفَض بفلتر الطول)
    "مرحبا بك",

    # وثيقة 4: يحتوي PII (بريد إلكتروني ورقم هاتف)
    "للتواصل معنا يرجى إرسال بريد إلكتروني إلى ahmed.sales@example.com "
    "أو الاتصال على الرقم +966-50-123-4567 خلال ساعات الدوام الرسمي من "
    "الأحد إلى الخميس للاستفسار عن خدماتنا المتنوعة في مجال البرمجيات.",

    # وثيقة 5: سبام/قائمة كلمات مفتاحية (سيُرفَض بفلتر نسبة الرموز)
    "!!!خصم خصم خصم!!! اشتري الآن ### عرض محدود $$$ سارع بالطلب !!! "
    "### اتصل الآن $$$ !!! خصم 90% ### !!!",

    # وثيقة 6: نص جيد ومختلف تمامًا
    "تُعد معالجة اللغات الطبيعية أحد أهم فروع الذكاء الاصطناعي، وتتضمن "
    "مهامًا متنوعة مثل الترجمة الآلية والتلخيص والإجابة على الأسئلة. "
    "ساهمت نماذج المحوّلات في إحداث نقلة نوعية في جودة هذه التطبيقات.",
]

print("=" * 60)
print(f"البدء بـ {len(raw_documents)} وثيقة خام")
print("=" * 60)

# الخطوة 1: فلترة الجودة Heuristic (يُزيل الوثيقة 3 والوثيقة 5)
quality_filter = HeuristicQualityFilter(min_words=5, max_symbol_to_word_ratio=0.3)
step1_docs, _ = quality_filter.filter_corpus(raw_documents)
# → يبقى: [1, 2, 4, 6]

# الخطوة 2: إزالة PII (يُنظّف الوثيقة 4 دون حذفها)
pii_remover = PIIRemover()
step2_docs = pii_remover.scrub_corpus(step1_docs)
# → الوثيقة 4 أصبحت: "... [EMAIL] ... [PHONE] ..."

# الخطوة 3: Fuzzy Deduplication (يُزيل إحدى الوثيقتين 1/2 لتشابههما الشديد)
deduplicator = MinHashDeduplicator(num_hashes=64, num_bands=8,
                                     ngram_size=3, similarity_threshold=0.7)
step3_docs = deduplicator.deduplicate(step2_docs)
# → يبقى 3 وثائق نهائية: [وثيقة 1 أو 2، الوثيقة 4 منظفة، الوثيقة 6]

print("\n" + "=" * 60)
print(f"النتيجة النهائية: {len(step3_docs)} وثيقة نظيفة من أصل {len(raw_documents)}")
print("=" * 60)
for i, doc in enumerate(step3_docs, 1):
    print(f"\n[{i}] {doc[:100]}...")

# ────────────────────────────────────────────────────────────────────────
# [011] content/notebooks/part-05-building-llms/ch18-data-pipeline.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-18-ch18-data-pipeline.ipynb#cell-2
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
    ("bs4", "beautifulsoup4"),
    ("fasttext", "fasttext-wheel"),
    ("numpy", "numpy"),
    ("trafilatura", "trafilatura")
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-18-ch18-data-pipeline.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import re
from bs4 import BeautifulSoup
import trafilatura  # مكتبة متخصصة في استخلاص المحتوى الرئيسي من HTML

def extract_clean_text_from_html(html_content: str) -> str:
    """
    استخلاص النص الرئيسي من صفحة HTML مع تجاهل:
    - القوائم الجانبية (navigation, sidebar)
    - الإعلانات
    - التذييل والترويسة (footer, header)
    - الوسوم البرمجية (script, style)

    نستخدم trafilatura لأنها مصممة خصيصًا لهذه المهمة وتتفوق
    على BeautifulSoup البسيط في تمييز "المحتوى الرئيسي" عن الضوضاء.
    """
    # الطريقة المتقدمة: trafilatura (تُستخدم فعليًا في بناء مجموعات مثل RefinedWeb)
    extracted = trafilatura.extract(
        html_content,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
    )
    if extracted:
        return extracted

    # طريقة احتياطية بسيطة بـ BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # تنظيف الأسطر الفارغة المتكررة
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return text

def basic_arabic_text_cleaning(text: str) -> str:
    """
    تنظيف أساسي خاص بالنصوص العربية:
    - توحيد أشكال الألف (أ، إ، آ → ا) [اختياري، يُستخدم أحيانًا للتدريب لكن يُفقِد معلومة]
    - إزالة التشكيل الزائد إذا لم يكن مطلوبًا
    - إزالة التكرار المفرط للأحرف (مثال: "جميلللللل" → "جميل")
    - توحيد المسافات المتعددة
    """
    # إزالة التشكيل (التنوين والحركات) - اختياري حسب حالة الاستخدام
    arabic_diacritics = re.compile(r'[\u064B-\u0652\u0670\u0640]')
    text = arabic_diacritics.sub('', text)

    # تقليص تكرار الأحرف الزائد عن 3 مرات متتالية إلى مرتين
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # توحيد المسافات
    text = re.sub(r'\s+', ' ', text).strip()

    # إزالة الأسطر القصيرة جدًا (أقل من 3 كلمات) التي غالبًا قوائم تنقل أو إعلانات
    lines = text.split('\n')
    lines = [l for l in lines if len(l.split()) >= 3]
    return '\n'.join(lines)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-18-ch18-data-pipeline.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import fasttext

class LanguageFilter:
    """
    فلترة النصوص حسب اللغة باستخدام نموذج fastText المدرَّب مسبقًا
    (lid.176.bin) القادر على تمييز 176 لغة بدقة عالية وسرعة كبيرة
    (مناسب لمعالجة تريليونات الرموز).
    """
    def __init__(self, model_path: str = "lid.176.bin",
                 target_langs: list = None,
                 confidence_threshold: float = 0.65):
        self.model = fasttext.load_model(model_path)
        self.target_langs = set(target_langs or ["ar", "en"])
        self.confidence_threshold = confidence_threshold

    def predict_language(self, text: str) -> tuple:
        """يعيد (رمز اللغة، درجة الثقة)"""
        # fastText يتوقع نصًا بدون أسطر جديدة
        clean_text = text.replace("\n", " ")[:2000]  # عيّنة كافية للتنبؤ
        predictions = self.model.predict(clean_text, k=1)
        lang_code = predictions[0][0].replace("__label__", "")
        confidence = predictions[1][0]
        return lang_code, confidence

    def should_keep(self, text: str) -> bool:
        lang, confidence = self.predict_language(text)
        return lang in self.target_langs and confidence >= self.confidence_threshold

    def filter_corpus(self, documents: list) -> list:
        kept = [doc for doc in documents if self.should_keep(doc)]
        print(f"احتُفظ بـ {len(kept)}/{len(documents)} "
              f"({100*len(kept)/max(len(documents),1):.1f}%) من الوثائق")
        return kept

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-18-ch18-data-pipeline.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import hashlib
from collections import defaultdict

def exact_deduplication(documents: list) -> list:
    """
    إزالة التكرار التام: أي وثيقتين متطابقتين حرفيًا يُحتفظ بواحدة فقط منهما.
    نستخدم SHA-256 لأنه مقاوم للتصادم (Collision-resistant) بشكل عملي.
    """
    seen_hashes = set()
    unique_documents = []

    for doc in documents:
        # تطبيع بسيط قبل الهاش: إزالة المسافات الزائدة لتفادي اعتبار
        # وثيقتين متطابقتين فعليًا مختلفتين بسبب فراغ زائد
        normalized = " ".join(doc.split())
        doc_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

        if doc_hash not in seen_hashes:
            seen_hashes.add(doc_hash)
            unique_documents.append(doc)

    removed = len(documents) - len(unique_documents)
    print(f"Exact Dedup: أُزيلت {removed} وثيقة مكررة تمامًا "
          f"({100*removed/max(len(documents),1):.1f}%)")
    return unique_documents

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-18-ch18-data-pipeline.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from collections import defaultdict
import hashlib

class MinHashDeduplicator:
    """
    إزالة التكرار التقريبي باستخدام MinHash + LSH.
    هذه هي التقنية الفعلية المستخدمة في بناء مجموعات مثل
    RefinedWeb و FineWeb لإزالة مليارات الوثائق شبه المكررة
    بكفاءة زمنية تقارب O(n) بدلاً من O(n²).
    """
    def __init__(self, num_hashes: int = 128, num_bands: int = 16,
                 ngram_size: int = 5, similarity_threshold: float = 0.8):
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        self.ngram_size = ngram_size
        self.similarity_threshold = similarity_threshold

        # توليد معاملات دوال الهاش العشوائية (a، b لكل دالة هاش خطية)
        # نستخدم صيغة: h_i(x) = (a_i * x + b_i) mod p
        rng = np.random.RandomState(42)
        self.max_prime = (1 << 61) - 1  # عدد أولي كبير (Mersenne Prime)
        self.a_params = rng.randint(1, self.max_prime, size=num_hashes)
        self.b_params = rng.randint(0, self.max_prime, size=num_hashes)

    def _get_ngrams(self, text: str) -> set:
        """تحويل النص إلى مجموعة n-grams على مستوى الكلمات"""
        words = text.split()
        if len(words) < self.ngram_size:
            return {tuple(words)}
        return {tuple(words[i:i + self.ngram_size])
                for i in range(len(words) - self.ngram_size + 1)}

    def _hash_ngram(self, ngram: tuple) -> int:
        """تحويل n-gram إلى عدد صحيح عبر MD5 (أسرع من SHA للأغراض الداخلية)"""
        ngram_str = " ".join(ngram)
        return int(hashlib.md5(ngram_str.encode("utf-8")).hexdigest(), 16)

    def compute_minhash_signature(self, text: str) -> np.ndarray:
        """
        حساب توقيع MinHash لوثيقة: مصفوفة من num_hashes قيمة،
        كل قيمة هي أصغر قيمة هاش بين كل الـ n-grams للوثيقة
        تحت دالة هاش معينة.
        """
        ngrams = self._get_ngrams(text)
        if not ngrams:
            return np.full(self.num_hashes, self.max_prime, dtype=np.int64)

        ngram_hashes = np.array([self._hash_ngram(ng) for ng in ngrams])
        # تطبيق كل دالة هاش خطية على كل قيم الـ ngram_hashes دفعة واحدة
        signature = np.full(self.num_hashes, self.max_prime, dtype=np.int64)
        for i in range(self.num_hashes):
            hashed_values = (self.a_params[i] * ngram_hashes +
                              self.b_params[i]) % self.max_prime
            signature[i] = hashed_values.min()
        return signature

    def estimate_jaccard_similarity(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """تقدير تشابه Jaccard من نسبة تطابق عناصر التوقيعين"""
        return float(np.mean(sig1 == sig2))

    def build_lsh_buckets(self, signatures: list) -> dict:
        """
        تقنية LSH: نقسّم كل توقيع إلى (num_bands) نطاقات (Bands)،
        كل نطاق يحتوي (rows_per_band) قيمة. وثيقتان تقعان في نفس
        الدلو (Bucket) لأي نطاق تصبحان "مرشحتين" للمقارنة الدقيقة.
        هذا يقلل عدد المقارنات من O(n²) إلى تقريبًا O(n).
        """
        buckets = defaultdict(list)
        for doc_idx, signature in enumerate(signatures):
            for band_idx in range(self.num_bands):
                start = band_idx * self.rows_per_band
                end = start + self.rows_per_band
                band_signature = tuple(signature[start:end])
                bucket_key = (band_idx, hash(band_signature))
                buckets[bucket_key].append(doc_idx)
        return buckets

    def deduplicate(self, documents: list) -> list:
        """خط الإنتاج الكامل: حساب التواقيع → بناء الدلاء → تصفية المكررات"""
        print(f"حساب تواقيع MinHash لـ {len(documents)} وثيقة...")
        signatures = [self.compute_minhash_signature(doc) for doc in documents]

        print("بناء دلاء LSH...")
        buckets = self.build_lsh_buckets(signatures)

        # تجميع أزواج الوثائق المرشحة للمقارنة (ظهرت معًا في دلو واحد على الأقل)
        candidate_pairs = set()
        for bucket_docs in buckets.values():
            if len(bucket_docs) > 1:
                for i in range(len(bucket_docs)):
                    for j in range(i + 1, len(bucket_docs)):
                        pair = tuple(sorted([bucket_docs[i], bucket_docs[j]]))
                        candidate_pairs.add(pair)

        print(f"عدد الأزواج المرشحة للمقارنة: {len(candidate_pairs)} "
              f"(بدلاً من {len(documents)*(len(documents)-1)//2} في الطريقة الساذجة)")

        # تحديد الوثائق المكررة فعليًا عبر حساب التشابه الدقيق للمرشحين فقط
        to_remove = set()
        for i, j in candidate_pairs:
            if i in to_remove or j in to_remove:
                continue
            similarity = self.estimate_jaccard_similarity(signatures[i], signatures[j])
            if similarity >= self.similarity_threshold:
                # نحتفظ بالوثيقة الأطول عادة (غالبًا الأصلية الكاملة)
                if len(documents[i]) >= len(documents[j]):
                    to_remove.add(j)
                else:
                    to_remove.add(i)

        unique_documents = [doc for idx, doc in enumerate(documents)
                             if idx not in to_remove]
        print(f"Fuzzy Dedup: أُزيلت {len(to_remove)} وثيقة شبه مكررة "
              f"({100*len(to_remove)/max(len(documents),1):.1f}%)")
        return unique_documents

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-llms-transformers/notebooks/ch-18-ch18-data-pipeline.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
class HeuristicQualityFilter:
    """
    مجموعة قواعد بسيطة لكن فعّالة جدًا لفلترة النصوص منخفضة الجودة،
    مستوحاة من أوراق Gopher وC4 وFineWeb. هذه القواعد رخيصة حسابيًا
    وتُطبَّق كخط دفاع أول قبل المرشّحات الأثقل (نماذج التصنيف).
    """
    def __init__(self,
                 min_words: int = 50,
                 max_words: int = 100_000,
                 min_avg_word_length: float = 2.0,
                 max_avg_word_length: float = 15.0,
                 max_symbol_to_word_ratio: float = 0.3,
                 max_repeated_line_fraction: float = 0.3):
        self.min_words = min_words
        self.max_words = max_words
        self.min_avg_word_length = min_avg_word_length
        self.max_avg_word_length = max_avg_word_length
        self.max_symbol_to_word_ratio = max_symbol_to_word_ratio
        self.max_repeated_line_fraction = max_repeated_line_fraction

    def check_document(self, text: str) -> tuple:
        """
        يعيد (يُقبَل: bool, سبب الرفض إن وُجد: str)
        نرتب الفحوصات من الأرخص للأغلى حسابيًا (fail-fast).
        """
        words = text.split()
        num_words = len(words)

        # 1. فحص طول الوثيقة
        if num_words < self.min_words:
            return False, f"قصيرة جدًا ({num_words} كلمة)"
        if num_words > self.max_words:
            return False, f"طويلة جدًا ({num_words} كلمة)"

        # 2. متوسط طول الكلمة (كلمات قصيرة جدًا = سبام/قوائم، طويلة جدًا = كود ملتصق/أخطاء)
        avg_word_len = sum(len(w) for w in words) / num_words
        if not (self.min_avg_word_length <= avg_word_len <= self.max_avg_word_length):
            return False, f"متوسط طول الكلمة غير طبيعي ({avg_word_len:.1f})"

        # 3. نسبة الرموز غير الأبجدية (علامات ترقيم، رموز خاصة) إلى الكلمات
        num_symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
        symbol_ratio = num_symbols / max(num_words, 1)
        if symbol_ratio > self.max_symbol_to_word_ratio:
            return False, f"نسبة رموز مرتفعة ({symbol_ratio:.2f})"

        # 4. نسبة الأسطر المكررة داخل نفس الوثيقة (مؤشر على قوائم/قوالب متكررة)
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if lines:
            unique_lines = len(set(lines))
            repeated_fraction = 1 - (unique_lines / len(lines))
            if repeated_fraction > self.max_repeated_line_fraction:
                return False, f"نسبة تكرار أسطر مرتفعة ({repeated_fraction:.2f})"

        # 5. فحص وجود كلمة واحدة على الأقل تحتوي حرفًا أبجديًا (رفض الوثائق الرقمية بحتة)
        has_alpha_word = any(any(c.isalpha() for c in w) for w in words)
        if not has_alpha_word:
            return False, "لا تحتوي كلمات نصية فعلية"

        return True, None

    def filter_corpus(self, documents: list) -> tuple:
        kept, rejected_reasons = [], defaultdict(int)
        for doc in documents:
            accepted, reason = self.check_document(doc)
            if accepted:
                kept.append(doc)
            else:
                rejected_reasons[reason.split("(")[0].strip()] += 1

        print(f"Heuristic Filter: احتُفظ بـ {len(kept)}/{len(documents)}")
        for reason, count in sorted(rejected_reasons.items(),
                                     key=lambda x: -x[1]):
            print(f"   - رُفضت {count} وثيقة بسبب: {reason}")
        return kept, rejected_reasons

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-llms-transformers/notebooks/ch-18-ch18-data-pipeline.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import fasttext

class QualityClassifierFilter:
    """
    نموذج فلترة جودة مبني على fastText، يُحاكي منهجية FineWeb-Edu:
    يُدرَّب على تمييز النصوص "التعليمية عالية القيمة" من بقية نصوص الويب.
    """
    def __init__(self, positive_samples: list, negative_samples: list):
        """
        positive_samples: نصوص عالية الجودة (ويكيبيديا، كتب، مقالات أكاديمية)
        negative_samples: عينة عشوائية من الويب الخام (غير مفلترة)
        """
        self.training_file = "_quality_classifier_train.txt"
        self._prepare_training_data(positive_samples, negative_samples)
        self.model = fasttext.train_supervised(
            input=self.training_file,
            lr=0.5, epoch=25, wordNgrams=2, dim=100,
        )

    def _prepare_training_data(self, positive, negative):
        with open(self.training_file, "w", encoding="utf-8") as f:
            for text in positive:
                clean = text.replace("\n", " ")
                f.write(f"__label__high_quality {clean}\n")
            for text in negative:
                clean = text.replace("\n", " ")
                f.write(f"__label__low_quality {clean}\n")

    def score_document(self, text: str) -> float:
        """يعيد درجة بين 0 و1، حيث 1 تعني ثقة كاملة بأنها عالية الجودة"""
        clean_text = text.replace("\n", " ")[:2000]
        labels, probs = self.model.predict(clean_text, k=1)
        label = labels[0].replace("__label__", "")
        prob = probs[0]
        return prob if label == "high_quality" else (1 - prob)

    def filter_corpus(self, documents: list, keep_top_fraction: float = 0.5) -> list:
        """الاحتفاظ بأعلى نسبة مئوية من الوثائق حسب درجة الجودة"""
        scored = [(doc, self.score_document(doc)) for doc in documents]
        scored.sort(key=lambda x: -x[1])
        num_to_keep = int(len(scored) * keep_top_fraction)
        kept = [doc for doc, score in scored[:num_to_keep]]
        print(f"Quality Classifier: احتُفظ بأعلى {keep_top_fraction*100:.0f}% "
              f"= {len(kept)} وثيقة")
        return kept

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/mastering-llms-transformers/notebooks/ch-18-ch18-data-pipeline.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import re

class PIIRemover:
    """
    إزالة أو إخفاء المعلومات الشخصية القابلة للتحديد (PII) من النصوص
    قبل التدريب، للامتثال للخصوصية ومنع النموذج من "حفظ" بيانات حساسة.
    نستخدم أنماط Regex للحالات الواضحة، ويمكن تعزيزها بنموذج NER
    (Named Entity Recognition) للكشف عن الأسماء الشخصية والعناوين.
    """
    def __init__(self):
        self.patterns = {
            "EMAIL": re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ),
            "PHONE_INTL": re.compile(
                r'\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}'
            ),
            "SSN_US": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "CREDIT_CARD": re.compile(
                r'\b(?:\d[ -]*?){13,16}\b'
            ),
            "IP_ADDRESS": re.compile(
                r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
            ),
            "SAUDI_NATIONAL_ID": re.compile(r'\b[12]\d{9}\b'),
        }
        self.replacement_tokens = {
            "EMAIL": "[EMAIL]",
            "PHONE_INTL": "[PHONE]",
            "SSN_US": "[SSN]",
            "CREDIT_CARD": "[CREDIT_CARD]",
            "IP_ADDRESS": "[IP_ADDRESS]",
            "SAUDI_NATIONAL_ID": "[NATIONAL_ID]",
        }

    def scrub_text(self, text: str) -> tuple:
        """يعيد (النص المنظف، عدد كل نوع تم إخفاؤه)"""
        counts = defaultdict(int)
        cleaned_text = text
        for pii_type, pattern in self.patterns.items():
            matches = pattern.findall(cleaned_text)
            counts[pii_type] = len(matches)
            cleaned_text = pattern.sub(self.replacement_tokens[pii_type],
                                         cleaned_text)
        return cleaned_text, dict(counts)

    def scrub_corpus(self, documents: list) -> list:
        total_counts = defaultdict(int)
        cleaned_docs = []
        for doc in documents:
            cleaned, counts = self.scrub_text(doc)
            cleaned_docs.append(cleaned)
            for k, v in counts.items():
                total_counts[k] += v
        print("إحصائيات إزالة PII عبر الكوربص بالكامل:")
        for pii_type, count in total_counts.items():
            if count > 0:
                print(f"   - {pii_type}: {count} حالة")
        return cleaned_docs

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/mastering-llms-transformers/notebooks/ch-18-ch18-data-pipeline.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
class ContaminationChecker:
    """
    يفحص ما إذا كانت وثائق التدريب تحتوي على مقاطع متطابقة أو شبه متطابقة
    من مجموعات بيانات التقييم المعروفة (benchmark test sets)، باستخدام
    تطابق n-grams طويلة (13-gram فأكثر) وهي الطريقة المعتمدة في تقارير
    GPT-4 وLLaMA الفنية.
    """
    def __init__(self, benchmark_texts: list, ngram_size: int = 13):
        self.ngram_size = ngram_size
        self.benchmark_ngrams = set()
        for text in benchmark_texts:
            self.benchmark_ngrams.update(self._get_ngrams(text))

    def _get_ngrams(self, text: str) -> set:
        words = text.lower().split()
        if len(words) < self.ngram_size:
            return {tuple(words)}
        return {tuple(words[i:i + self.ngram_size])
                for i in range(len(words) - self.ngram_size + 1)}

    def is_contaminated(self, document: str, overlap_threshold: int = 1) -> bool:
        """يعيد True إذا احتوت الوثيقة على عدد n-grams مشتركة مع benchmark
        يساوي أو يتجاوز overlap_threshold"""
        doc_ngrams = self._get_ngrams(document)
        overlap = doc_ngrams & self.benchmark_ngrams
        return len(overlap) >= overlap_threshold

    def filter_corpus(self, documents: list) -> list:
        clean_docs = [doc for doc in documents if not self.is_contaminated(doc)]
        removed = len(documents) - len(clean_docs)
        print(f"Contamination Check: أُزيلت {removed} وثيقة ملوَّثة "
              f"من مجموعات التقييم")
        return clean_docs

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/mastering-llms-transformers/notebooks/ch-18-ch18-data-pipeline.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
raw_documents = [
    # وثيقة 1: نص جيد عن الذكاء الاصطناعي
    "الذكاء الاصطناعي هو مجال علمي يهتم بتطوير أنظمة قادرة على أداء "
    "مهام تتطلب عادة ذكاءً بشريًا مثل الفهم والتعلم واتخاذ القرار. "
    "شهد هذا المجال تطورًا هائلاً في العقد الأخير بفضل التعلم العميق.",

    # وثيقة 2: تكرار شبه تام للوثيقة 1 (تغيير كلمة واحدة فقط)
    "الذكاء الاصطناعي هو مجال علمي يهتم بتطوير أنظمة قادرة على أداء "
    "مهام تتطلب عادة ذكاءً بشريًا مثل الفهم والتعلم واتخاذ القرارات. "
    "شهد هذا المجال تطورًا هائلاً في العقد الأخير بفضل التعلم العميق.",

    # وثيقة 3: نص قصير جدًا (سيُرفَض بفلتر الطول)
    "مرحبا بك",

    # وثيقة 4: يحتوي PII (بريد إلكتروني ورقم هاتف)
    "للتواصل معنا يرجى إرسال بريد إلكتروني إلى ahmed.sales@example.com "
    "أو الاتصال على الرقم +966-50-123-4567 خلال ساعات الدوام الرسمي من "
    "الأحد إلى الخميس للاستفسار عن خدماتنا المتنوعة في مجال البرمجيات.",

    # وثيقة 5: سبام/قائمة كلمات مفتاحية (سيُرفَض بفلتر نسبة الرموز)
    "!!!خصم خصم خصم!!! اشتري الآن ### عرض محدود $$$ سارع بالطلب !!! "
    "### اتصل الآن $$$ !!! خصم 90% ### !!!",

    # وثيقة 6: نص جيد ومختلف تمامًا
    "تُعد معالجة اللغات الطبيعية أحد أهم فروع الذكاء الاصطناعي، وتتضمن "
    "مهامًا متنوعة مثل الترجمة الآلية والتلخيص والإجابة على الأسئلة. "
    "ساهمت نماذج المحوّلات في إحداث نقلة نوعية في جودة هذه التطبيقات.",
]

print("=" * 60)
print(f"البدء بـ {len(raw_documents)} وثيقة خام")
print("=" * 60)

# الخطوة 1: فلترة الجودة Heuristic (يُزيل الوثيقة 3 والوثيقة 5)
quality_filter = HeuristicQualityFilter(min_words=5, max_symbol_to_word_ratio=0.3)
step1_docs, _ = quality_filter.filter_corpus(raw_documents)
# → يبقى: [1, 2, 4, 6]

# الخطوة 2: إزالة PII (يُنظّف الوثيقة 4 دون حذفها)
pii_remover = PIIRemover()
step2_docs = pii_remover.scrub_corpus(step1_docs)
# → الوثيقة 4 أصبحت: "... [EMAIL] ... [PHONE] ..."

# الخطوة 3: Fuzzy Deduplication (يُزيل إحدى الوثيقتين 1/2 لتشابههما الشديد)
deduplicator = MinHashDeduplicator(num_hashes=64, num_bands=8,
                                     ngram_size=3, similarity_threshold=0.7)
step3_docs = deduplicator.deduplicate(step2_docs)
# → يبقى 3 وثائق نهائية: [وثيقة 1 أو 2، الوثيقة 4 منظفة، الوثيقة 6]

print("\n" + "=" * 60)
print(f"النتيجة النهائية: {len(step3_docs)} وثيقة نظيفة من أصل {len(raw_documents)}")
print("=" * 60)
for i, doc in enumerate(step3_docs, 1):
    print(f"\n[{i}] {doc[:100]}...")

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/mastering-llms-transformers/notebooks/ch-18-ch18-data-pipeline.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
