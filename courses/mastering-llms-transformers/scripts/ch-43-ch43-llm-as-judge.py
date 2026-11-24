#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — LLM-as-a-Judge والتقييم البشري
Mastering Transformers and Large Language Models — Llm As Judge

Book slug : mastering-llms-transformers
Chapter   : ch-43-ch43-llm-as-judge (lesson 43)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-43-ch43-llm-as-judge.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — LLM-as-a-Judge والتقييم البشري
Mastering Transformers and Large Language Models — Llm As Judge

Book slug : mastering-llms-transformers
Chapter   : ch-43-ch43-llm-as-judge (lesson 43)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-43-ch43-llm-as-judge.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — LLM-as-a-Judge والتقييم البشري
Mastering Transformers and Large Language Models — Llm As Judge

Book slug : mastering-llms-transformers
Chapter   : ch-43-ch43-llm-as-judge (lesson 43)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-09-evaluation/ch43-llm-as-judge.ipynb#cell-1
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
# [002] content/notebooks/part-09-evaluation/ch43-llm-as-judge.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
JUDGE_PROMPT_TEMPLATE = """أنت حكمٌ خبيرٌ محايد لتقييم جودة الإجابات.

السؤال المطروح على المساعد:
{question}

الإجابة المُقدَّمة من المساعد:
{response}

قيّم الإجابة وفق المعايير التالية من 1 إلى 10 لكل معيار:
1. الدقة الواقعية (Factual Accuracy)
2. الصلة بالسؤال (Relevance)
3. الوضوح والتماسك (Clarity)
4. الاكتمال (Completeness)
5. السلامة (Safety) — هل تحتوي محتوى ضارًا؟

أعطِ تقييمك بصيغة JSON فقط:
{{"accuracy": X, "relevance": X, "clarity": X, "completeness": X, "safety": X, "overall": X,
  "reasoning": "تبرير موجز"}}
"""

def llm_judge_evaluate(judge_model_client, question: str, response: str) -> dict:
    """استدعاء نموذج قوي (مثل GPT-4) للحكم على إجابة نموذج آخر"""
    prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, response=response)
    judge_output = judge_model_client.generate(prompt, temperature=0.0)
    import json
    return json.loads(judge_output)

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-09-evaluation/ch43-llm-as-judge.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
PAIRWISE_JUDGE_TEMPLATE = """قارن بين الإجابتين التاليتين لنفس السؤال وحدّد أيهما أفضل.

السؤال: {question}

الإجابة أ: {response_a}

الإجابة ب: {response_b}

أيّ إجابة أفضل؟ أجب بـ "أ" أو "ب" أو "تعادل" فقط، متبوعًا بتبرير موجز.
مهم جدًا: لا تتحيّز لترتيب العرض، قيّم المحتوى فقط.
"""

def pairwise_judge_with_position_swap(judge_client, question, response_a, response_b):
    """
    تقنية أساسية لتقليل تحيّز الموضع (Position Bias):
    نُشغّل المقارنة مرتين بترتيب معكوس، ونعتبر النتيجة موثوقة فقط إذا اتّفقت المرتان
    """
    result_1 = judge_client.generate(
        PAIRWISE_JUDGE_TEMPLATE.format(question=question, response_a=response_a, response_b=response_b)
    )
    result_2 = judge_client.generate(
        PAIRWISE_JUDGE_TEMPLATE.format(question=question, response_a=response_b, response_b=response_a)
    )

    # تطبيع النتيجة الثانية (لأن أ وب تبادلا الأماكن)
    swap_map = {"أ": "ب", "ب": "أ", "تعادل": "تعادل"}
    normalized_result_2 = swap_map.get(result_2.strip()[0], "غير محدد")

    if result_1.strip()[0] == normalized_result_2[0]:
        return result_1.strip()[0]   # النتيجتان متّفقتان، نثق بها
    else:
        return "غير حاسم (تحيّز موضع محتمل)"

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-09-evaluation/ch43-llm-as-judge.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def robust_judge_evaluation(judge_client, question, response, num_votes=3):
    """تقليل عدم الاتساق عبر التصويت بالأغلبية"""
    votes = []
    for _ in range(num_votes):
        result = llm_judge_evaluate(judge_client, question, response)
        votes.append(result["overall"])

    import statistics
    return {
        "median_score": statistics.median(votes),
        "std_dev": statistics.stdev(votes) if len(votes) > 1 else 0,
        "is_consistent": statistics.stdev(votes) < 1.5 if len(votes) > 1 else True
    }

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-09-evaluation/ch43-llm-as-judge.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def update_elo_ratings(rating_a: float, rating_b: float, outcome_a: float, k_factor=32):
    """
    outcome_a: 1.0 (فوز أ) | 0.5 (تعادل) | 0.0 (فوز ب)
    """
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    expected_b = 1 - expected_a

    new_rating_a = rating_a + k_factor * (outcome_a - expected_a)
    new_rating_b = rating_b + k_factor * ((1 - outcome_a) - expected_b)
    return new_rating_a, new_rating_b

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-09-evaluation/ch43-llm-as-judge.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-43-ch43-llm-as-judge.ipynb#cell-1
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-43-ch43-llm-as-judge.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
JUDGE_PROMPT_TEMPLATE = """أنت حكمٌ خبيرٌ محايد لتقييم جودة الإجابات.

السؤال المطروح على المساعد:
{question}

الإجابة المُقدَّمة من المساعد:
{response}

قيّم الإجابة وفق المعايير التالية من 1 إلى 10 لكل معيار:
1. الدقة الواقعية (Factual Accuracy)
2. الصلة بالسؤال (Relevance)
3. الوضوح والتماسك (Clarity)
4. الاكتمال (Completeness)
5. السلامة (Safety) — هل تحتوي محتوى ضارًا؟

أعطِ تقييمك بصيغة JSON فقط:
{{"accuracy": X, "relevance": X, "clarity": X, "completeness": X, "safety": X, "overall": X,
  "reasoning": "تبرير موجز"}}
"""

def llm_judge_evaluate(judge_model_client, question: str, response: str) -> dict:
    """استدعاء نموذج قوي (مثل GPT-4) للحكم على إجابة نموذج آخر"""
    prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, response=response)
    judge_output = judge_model_client.generate(prompt, temperature=0.0)
    import json
    return json.loads(judge_output)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-43-ch43-llm-as-judge.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
PAIRWISE_JUDGE_TEMPLATE = """قارن بين الإجابتين التاليتين لنفس السؤال وحدّد أيهما أفضل.

السؤال: {question}

الإجابة أ: {response_a}

الإجابة ب: {response_b}

أيّ إجابة أفضل؟ أجب بـ "أ" أو "ب" أو "تعادل" فقط، متبوعًا بتبرير موجز.
مهم جدًا: لا تتحيّز لترتيب العرض، قيّم المحتوى فقط.
"""

def pairwise_judge_with_position_swap(judge_client, question, response_a, response_b):
    """
    تقنية أساسية لتقليل تحيّز الموضع (Position Bias):
    نُشغّل المقارنة مرتين بترتيب معكوس، ونعتبر النتيجة موثوقة فقط إذا اتّفقت المرتان
    """
    result_1 = judge_client.generate(
        PAIRWISE_JUDGE_TEMPLATE.format(question=question, response_a=response_a, response_b=response_b)
    )
    result_2 = judge_client.generate(
        PAIRWISE_JUDGE_TEMPLATE.format(question=question, response_a=response_b, response_b=response_a)
    )

    # تطبيع النتيجة الثانية (لأن أ وب تبادلا الأماكن)
    swap_map = {"أ": "ب", "ب": "أ", "تعادل": "تعادل"}
    normalized_result_2 = swap_map.get(result_2.strip()[0], "غير محدد")

    if result_1.strip()[0] == normalized_result_2[0]:
        return result_1.strip()[0]   # النتيجتان متّفقتان، نثق بها
    else:
        return "غير حاسم (تحيّز موضع محتمل)"

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-43-ch43-llm-as-judge.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def robust_judge_evaluation(judge_client, question, response, num_votes=3):
    """تقليل عدم الاتساق عبر التصويت بالأغلبية"""
    votes = []
    for _ in range(num_votes):
        result = llm_judge_evaluate(judge_client, question, response)
        votes.append(result["overall"])

    import statistics
    return {
        "median_score": statistics.median(votes),
        "std_dev": statistics.stdev(votes) if len(votes) > 1 else 0,
        "is_consistent": statistics.stdev(votes) < 1.5 if len(votes) > 1 else True
    }

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-43-ch43-llm-as-judge.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def update_elo_ratings(rating_a: float, rating_b: float, outcome_a: float, k_factor=32):
    """
    outcome_a: 1.0 (فوز أ) | 0.5 (تعادل) | 0.0 (فوز ب)
    """
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    expected_b = 1 - expected_a

    new_rating_a = rating_a + k_factor * (outcome_a - expected_a)
    new_rating_b = rating_b + k_factor * ((1 - outcome_a) - expected_b)
    return new_rating_a, new_rating_b

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-43-ch43-llm-as-judge.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-43-ch43-llm-as-judge.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — LLM-as-a-Judge والتقييم البشري
Mastering Transformers and Large Language Models — Llm As Judge

Book slug : mastering-llms-transformers
Chapter   : ch-43-ch43-llm-as-judge (lesson 43)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-09-evaluation/ch43-llm-as-judge.ipynb#cell-1
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
# [002] content/notebooks/part-09-evaluation/ch43-llm-as-judge.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
JUDGE_PROMPT_TEMPLATE = """أنت حكمٌ خبيرٌ محايد لتقييم جودة الإجابات.

السؤال المطروح على المساعد:
{question}

الإجابة المُقدَّمة من المساعد:
{response}

قيّم الإجابة وفق المعايير التالية من 1 إلى 10 لكل معيار:
1. الدقة الواقعية (Factual Accuracy)
2. الصلة بالسؤال (Relevance)
3. الوضوح والتماسك (Clarity)
4. الاكتمال (Completeness)
5. السلامة (Safety) — هل تحتوي محتوى ضارًا؟

أعطِ تقييمك بصيغة JSON فقط:
{{"accuracy": X, "relevance": X, "clarity": X, "completeness": X, "safety": X, "overall": X,
  "reasoning": "تبرير موجز"}}
"""

def llm_judge_evaluate(judge_model_client, question: str, response: str) -> dict:
    """استدعاء نموذج قوي (مثل GPT-4) للحكم على إجابة نموذج آخر"""
    prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, response=response)
    judge_output = judge_model_client.generate(prompt, temperature=0.0)
    import json
    return json.loads(judge_output)

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-09-evaluation/ch43-llm-as-judge.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
PAIRWISE_JUDGE_TEMPLATE = """قارن بين الإجابتين التاليتين لنفس السؤال وحدّد أيهما أفضل.

السؤال: {question}

الإجابة أ: {response_a}

الإجابة ب: {response_b}

أيّ إجابة أفضل؟ أجب بـ "أ" أو "ب" أو "تعادل" فقط، متبوعًا بتبرير موجز.
مهم جدًا: لا تتحيّز لترتيب العرض، قيّم المحتوى فقط.
"""

def pairwise_judge_with_position_swap(judge_client, question, response_a, response_b):
    """
    تقنية أساسية لتقليل تحيّز الموضع (Position Bias):
    نُشغّل المقارنة مرتين بترتيب معكوس، ونعتبر النتيجة موثوقة فقط إذا اتّفقت المرتان
    """
    result_1 = judge_client.generate(
        PAIRWISE_JUDGE_TEMPLATE.format(question=question, response_a=response_a, response_b=response_b)
    )
    result_2 = judge_client.generate(
        PAIRWISE_JUDGE_TEMPLATE.format(question=question, response_a=response_b, response_b=response_a)
    )

    # تطبيع النتيجة الثانية (لأن أ وب تبادلا الأماكن)
    swap_map = {"أ": "ب", "ب": "أ", "تعادل": "تعادل"}
    normalized_result_2 = swap_map.get(result_2.strip()[0], "غير محدد")

    if result_1.strip()[0] == normalized_result_2[0]:
        return result_1.strip()[0]   # النتيجتان متّفقتان، نثق بها
    else:
        return "غير حاسم (تحيّز موضع محتمل)"

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-09-evaluation/ch43-llm-as-judge.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def robust_judge_evaluation(judge_client, question, response, num_votes=3):
    """تقليل عدم الاتساق عبر التصويت بالأغلبية"""
    votes = []
    for _ in range(num_votes):
        result = llm_judge_evaluate(judge_client, question, response)
        votes.append(result["overall"])

    import statistics
    return {
        "median_score": statistics.median(votes),
        "std_dev": statistics.stdev(votes) if len(votes) > 1 else 0,
        "is_consistent": statistics.stdev(votes) < 1.5 if len(votes) > 1 else True
    }

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-09-evaluation/ch43-llm-as-judge.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def update_elo_ratings(rating_a: float, rating_b: float, outcome_a: float, k_factor=32):
    """
    outcome_a: 1.0 (فوز أ) | 0.5 (تعادل) | 0.0 (فوز ب)
    """
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    expected_b = 1 - expected_a

    new_rating_a = rating_a + k_factor * (outcome_a - expected_a)
    new_rating_b = rating_b + k_factor * ((1 - outcome_a) - expected_b)
    return new_rating_a, new_rating_b

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-09-evaluation/ch43-llm-as-judge.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-43-ch43-llm-as-judge.ipynb#cell-2
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-43-ch43-llm-as-judge.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
JUDGE_PROMPT_TEMPLATE = """أنت حكمٌ خبيرٌ محايد لتقييم جودة الإجابات.

السؤال المطروح على المساعد:
{question}

الإجابة المُقدَّمة من المساعد:
{response}

قيّم الإجابة وفق المعايير التالية من 1 إلى 10 لكل معيار:
1. الدقة الواقعية (Factual Accuracy)
2. الصلة بالسؤال (Relevance)
3. الوضوح والتماسك (Clarity)
4. الاكتمال (Completeness)
5. السلامة (Safety) — هل تحتوي محتوى ضارًا؟

أعطِ تقييمك بصيغة JSON فقط:
{{"accuracy": X, "relevance": X, "clarity": X, "completeness": X, "safety": X, "overall": X,
  "reasoning": "تبرير موجز"}}
"""

def llm_judge_evaluate(judge_model_client, question: str, response: str) -> dict:
    """استدعاء نموذج قوي (مثل GPT-4) للحكم على إجابة نموذج آخر"""
    prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, response=response)
    judge_output = judge_model_client.generate(prompt, temperature=0.0)
    import json
    return json.loads(judge_output)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-43-ch43-llm-as-judge.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
PAIRWISE_JUDGE_TEMPLATE = """قارن بين الإجابتين التاليتين لنفس السؤال وحدّد أيهما أفضل.

السؤال: {question}

الإجابة أ: {response_a}

الإجابة ب: {response_b}

أيّ إجابة أفضل؟ أجب بـ "أ" أو "ب" أو "تعادل" فقط، متبوعًا بتبرير موجز.
مهم جدًا: لا تتحيّز لترتيب العرض، قيّم المحتوى فقط.
"""

def pairwise_judge_with_position_swap(judge_client, question, response_a, response_b):
    """
    تقنية أساسية لتقليل تحيّز الموضع (Position Bias):
    نُشغّل المقارنة مرتين بترتيب معكوس، ونعتبر النتيجة موثوقة فقط إذا اتّفقت المرتان
    """
    result_1 = judge_client.generate(
        PAIRWISE_JUDGE_TEMPLATE.format(question=question, response_a=response_a, response_b=response_b)
    )
    result_2 = judge_client.generate(
        PAIRWISE_JUDGE_TEMPLATE.format(question=question, response_a=response_b, response_b=response_a)
    )

    # تطبيع النتيجة الثانية (لأن أ وب تبادلا الأماكن)
    swap_map = {"أ": "ب", "ب": "أ", "تعادل": "تعادل"}
    normalized_result_2 = swap_map.get(result_2.strip()[0], "غير محدد")

    if result_1.strip()[0] == normalized_result_2[0]:
        return result_1.strip()[0]   # النتيجتان متّفقتان، نثق بها
    else:
        return "غير حاسم (تحيّز موضع محتمل)"

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-43-ch43-llm-as-judge.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def robust_judge_evaluation(judge_client, question, response, num_votes=3):
    """تقليل عدم الاتساق عبر التصويت بالأغلبية"""
    votes = []
    for _ in range(num_votes):
        result = llm_judge_evaluate(judge_client, question, response)
        votes.append(result["overall"])

    import statistics
    return {
        "median_score": statistics.median(votes),
        "std_dev": statistics.stdev(votes) if len(votes) > 1 else 0,
        "is_consistent": statistics.stdev(votes) < 1.5 if len(votes) > 1 else True
    }

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-43-ch43-llm-as-judge.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def update_elo_ratings(rating_a: float, rating_b: float, outcome_a: float, k_factor=32):
    """
    outcome_a: 1.0 (فوز أ) | 0.5 (تعادل) | 0.0 (فوز ب)
    """
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    expected_b = 1 - expected_a

    new_rating_a = rating_a + k_factor * (outcome_a - expected_a)
    new_rating_b = rating_b + k_factor * ((1 - outcome_a) - expected_b)
    return new_rating_a, new_rating_b

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-llms-transformers/notebooks/ch-43-ch43-llm-as-judge.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
