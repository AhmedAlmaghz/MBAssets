#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — معايير القياس المرجعية: Benchmarks
Mastering Transformers and Large Language Models — Benchmarks

Book slug : mastering-llms-transformers
Chapter   : ch-42-ch42-benchmarks (lesson 42)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-42-ch42-benchmarks.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — معايير القياس المرجعية: Benchmarks
Mastering Transformers and Large Language Models — Benchmarks

Book slug : mastering-llms-transformers
Chapter   : ch-42-ch42-benchmarks (lesson 42)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-42-ch42-benchmarks.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — معايير القياس المرجعية: Benchmarks
Mastering Transformers and Large Language Models — Benchmarks

Book slug : mastering-llms-transformers
Chapter   : ch-42-ch42-benchmarks (lesson 42)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-09-evaluation/ch42-benchmarks.ipynb#cell-1
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
# [002] content/notebooks/part-09-evaluation/ch42-benchmarks.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
def evaluate_mmlu_style(model, tokenizer, questions: list, choices_list: list, answers: list):
    """
    التقييم القياسي لأسئلة الاختيار من متعدد:
    نقارن log-likelihood لكل خيار، ونختار الأعلى (وليس التوليد الحر!)
    """
    correct = 0
    for question, choices, correct_answer_idx in zip(questions, choices_list, answers):
        choice_scores = []
        for choice in choices:
            prompt = f"{question}\nالإجابة: {choice}"
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
            # نستخدم negative loss كـ score (أعلى = احتمال أعلى)
            choice_scores.append(-outputs.loss.item())

        predicted_idx = choice_scores.index(max(choice_scores))
        if predicted_idx == correct_answer_idx:
            correct += 1

    return correct / len(questions)

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-09-evaluation/ch42-benchmarks.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def detect_contamination_via_perplexity_outlier(model, tokenizer, benchmark_samples: list,
                                                   reference_corpus_samples: list):
    """
    فكرة: إذا كان نموذج قد "حفظ" عيّنات benchmark أثناء التدريب،
    فإن perplexity عليها ستكون أقل بشكل غير طبيعي مقارنة بنصوص مشابهة لم يرها
    """
    def avg_ppl(samples):
        ppls = []
        for text in samples:
            input_ids = tokenizer(text, return_tensors="pt").input_ids
            with torch.no_grad():
                loss = model(input_ids, labels=input_ids).loss
            ppls.append(torch.exp(loss).item())
        return sum(ppls) / len(ppls)

    benchmark_ppl = avg_ppl(benchmark_samples)
    reference_ppl = avg_ppl(reference_corpus_samples)

    contamination_ratio = reference_ppl / benchmark_ppl
    # إذا كانت > 1.5-2x، هذا مؤشر قوي على احتمال التلوث (حفظ، وليس تعميم حقيقي)
    return {
        "benchmark_ppl": benchmark_ppl,
        "reference_ppl": reference_ppl,
        "contamination_suspicion_ratio": contamination_ratio
    }

def ngram_overlap_contamination_check(training_corpus_sample: list, benchmark_texts: list, n=13):
    """
    الطريقة الأكثر شيوعًا في التقارير الرسمية (مثل GPT-4 technical report):
    فحص تطابق n-grams طويلة (13 كلمة فأكثر) بين بيانات التدريب والـ benchmark
    """
    def get_ngrams(text, n):
        tokens = text.split()
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    contaminated_count = 0
    for benchmark_text in benchmark_texts:
        benchmark_ngrams = get_ngrams(benchmark_text, n)
        for train_text in training_corpus_sample:
            train_ngrams = get_ngrams(train_text, n)
            if benchmark_ngrams & train_ngrams:   # أي تقاطع يُعتبر تلوثًا محتملًا
                contaminated_count += 1
                break

    return contaminated_count / len(benchmark_texts)

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-09-evaluation/ch42-benchmarks.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def evaluate_with_chain_of_thought(model, tokenizer, question: str, use_reasoning_tokens=True):
    """
    تقييم أسلوب نماذج الاستدلال الحديثة: السماح بـ "مساحة تفكير" قبل الإجابة النهائية
    """
    if use_reasoning_tokens:
        prompt = f"""السؤال: {question}
فكّر خطوة بخطوة بعناية شديدة قبل الإجابة النهائية.
<تفكير>"""
    else:
        prompt = f"السؤال: {question}\nالإجابة المباشرة:"

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output_ids = model.generate(
        input_ids, max_new_tokens=2048,   # مساحة أكبر بكثير لنماذج الاستدلال
        temperature=0.0, do_sample=False
    )
    full_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # استخراج الإجابة النهائية بعد علامة الانتقال (حسب تنسيق النموذج)
    if "<إجابة_نهائية>" in full_response:
        final_answer = full_response.split("<إجابة_نهائية>")[-1].strip()
    else:
        final_answer = full_response.strip()
    return full_response, final_answer

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-09-evaluation/ch42-benchmarks.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-42-ch42-benchmarks.ipynb#cell-1
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-42-ch42-benchmarks.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
def evaluate_mmlu_style(model, tokenizer, questions: list, choices_list: list, answers: list):
    """
    التقييم القياسي لأسئلة الاختيار من متعدد:
    نقارن log-likelihood لكل خيار، ونختار الأعلى (وليس التوليد الحر!)
    """
    correct = 0
    for question, choices, correct_answer_idx in zip(questions, choices_list, answers):
        choice_scores = []
        for choice in choices:
            prompt = f"{question}\nالإجابة: {choice}"
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
            # نستخدم negative loss كـ score (أعلى = احتمال أعلى)
            choice_scores.append(-outputs.loss.item())

        predicted_idx = choice_scores.index(max(choice_scores))
        if predicted_idx == correct_answer_idx:
            correct += 1

    return correct / len(questions)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-42-ch42-benchmarks.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def detect_contamination_via_perplexity_outlier(model, tokenizer, benchmark_samples: list,
                                                   reference_corpus_samples: list):
    """
    فكرة: إذا كان نموذج قد "حفظ" عيّنات benchmark أثناء التدريب،
    فإن perplexity عليها ستكون أقل بشكل غير طبيعي مقارنة بنصوص مشابهة لم يرها
    """
    def avg_ppl(samples):
        ppls = []
        for text in samples:
            input_ids = tokenizer(text, return_tensors="pt").input_ids
            with torch.no_grad():
                loss = model(input_ids, labels=input_ids).loss
            ppls.append(torch.exp(loss).item())
        return sum(ppls) / len(ppls)

    benchmark_ppl = avg_ppl(benchmark_samples)
    reference_ppl = avg_ppl(reference_corpus_samples)

    contamination_ratio = reference_ppl / benchmark_ppl
    # إذا كانت > 1.5-2x، هذا مؤشر قوي على احتمال التلوث (حفظ، وليس تعميم حقيقي)
    return {
        "benchmark_ppl": benchmark_ppl,
        "reference_ppl": reference_ppl,
        "contamination_suspicion_ratio": contamination_ratio
    }

def ngram_overlap_contamination_check(training_corpus_sample: list, benchmark_texts: list, n=13):
    """
    الطريقة الأكثر شيوعًا في التقارير الرسمية (مثل GPT-4 technical report):
    فحص تطابق n-grams طويلة (13 كلمة فأكثر) بين بيانات التدريب والـ benchmark
    """
    def get_ngrams(text, n):
        tokens = text.split()
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    contaminated_count = 0
    for benchmark_text in benchmark_texts:
        benchmark_ngrams = get_ngrams(benchmark_text, n)
        for train_text in training_corpus_sample:
            train_ngrams = get_ngrams(train_text, n)
            if benchmark_ngrams & train_ngrams:   # أي تقاطع يُعتبر تلوثًا محتملًا
                contaminated_count += 1
                break

    return contaminated_count / len(benchmark_texts)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-42-ch42-benchmarks.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def evaluate_with_chain_of_thought(model, tokenizer, question: str, use_reasoning_tokens=True):
    """
    تقييم أسلوب نماذج الاستدلال الحديثة: السماح بـ "مساحة تفكير" قبل الإجابة النهائية
    """
    if use_reasoning_tokens:
        prompt = f"""السؤال: {question}
فكّر خطوة بخطوة بعناية شديدة قبل الإجابة النهائية.
<تفكير>"""
    else:
        prompt = f"السؤال: {question}\nالإجابة المباشرة:"

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output_ids = model.generate(
        input_ids, max_new_tokens=2048,   # مساحة أكبر بكثير لنماذج الاستدلال
        temperature=0.0, do_sample=False
    )
    full_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # استخراج الإجابة النهائية بعد علامة الانتقال (حسب تنسيق النموذج)
    if "<إجابة_نهائية>" in full_response:
        final_answer = full_response.split("<إجابة_نهائية>")[-1].strip()
    else:
        final_answer = full_response.strip()
    return full_response, final_answer

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-42-ch42-benchmarks.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-42-ch42-benchmarks.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — معايير القياس المرجعية: Benchmarks
Mastering Transformers and Large Language Models — Benchmarks

Book slug : mastering-llms-transformers
Chapter   : ch-42-ch42-benchmarks (lesson 42)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-09-evaluation/ch42-benchmarks.ipynb#cell-1
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
# [002] content/notebooks/part-09-evaluation/ch42-benchmarks.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
def evaluate_mmlu_style(model, tokenizer, questions: list, choices_list: list, answers: list):
    """
    التقييم القياسي لأسئلة الاختيار من متعدد:
    نقارن log-likelihood لكل خيار، ونختار الأعلى (وليس التوليد الحر!)
    """
    correct = 0
    for question, choices, correct_answer_idx in zip(questions, choices_list, answers):
        choice_scores = []
        for choice in choices:
            prompt = f"{question}\nالإجابة: {choice}"
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
            # نستخدم negative loss كـ score (أعلى = احتمال أعلى)
            choice_scores.append(-outputs.loss.item())

        predicted_idx = choice_scores.index(max(choice_scores))
        if predicted_idx == correct_answer_idx:
            correct += 1

    return correct / len(questions)

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-09-evaluation/ch42-benchmarks.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def detect_contamination_via_perplexity_outlier(model, tokenizer, benchmark_samples: list,
                                                   reference_corpus_samples: list):
    """
    فكرة: إذا كان نموذج قد "حفظ" عيّنات benchmark أثناء التدريب،
    فإن perplexity عليها ستكون أقل بشكل غير طبيعي مقارنة بنصوص مشابهة لم يرها
    """
    def avg_ppl(samples):
        ppls = []
        for text in samples:
            input_ids = tokenizer(text, return_tensors="pt").input_ids
            with torch.no_grad():
                loss = model(input_ids, labels=input_ids).loss
            ppls.append(torch.exp(loss).item())
        return sum(ppls) / len(ppls)

    benchmark_ppl = avg_ppl(benchmark_samples)
    reference_ppl = avg_ppl(reference_corpus_samples)

    contamination_ratio = reference_ppl / benchmark_ppl
    # إذا كانت > 1.5-2x، هذا مؤشر قوي على احتمال التلوث (حفظ، وليس تعميم حقيقي)
    return {
        "benchmark_ppl": benchmark_ppl,
        "reference_ppl": reference_ppl,
        "contamination_suspicion_ratio": contamination_ratio
    }

def ngram_overlap_contamination_check(training_corpus_sample: list, benchmark_texts: list, n=13):
    """
    الطريقة الأكثر شيوعًا في التقارير الرسمية (مثل GPT-4 technical report):
    فحص تطابق n-grams طويلة (13 كلمة فأكثر) بين بيانات التدريب والـ benchmark
    """
    def get_ngrams(text, n):
        tokens = text.split()
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    contaminated_count = 0
    for benchmark_text in benchmark_texts:
        benchmark_ngrams = get_ngrams(benchmark_text, n)
        for train_text in training_corpus_sample:
            train_ngrams = get_ngrams(train_text, n)
            if benchmark_ngrams & train_ngrams:   # أي تقاطع يُعتبر تلوثًا محتملًا
                contaminated_count += 1
                break

    return contaminated_count / len(benchmark_texts)

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-09-evaluation/ch42-benchmarks.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def evaluate_with_chain_of_thought(model, tokenizer, question: str, use_reasoning_tokens=True):
    """
    تقييم أسلوب نماذج الاستدلال الحديثة: السماح بـ "مساحة تفكير" قبل الإجابة النهائية
    """
    if use_reasoning_tokens:
        prompt = f"""السؤال: {question}
فكّر خطوة بخطوة بعناية شديدة قبل الإجابة النهائية.
<تفكير>"""
    else:
        prompt = f"السؤال: {question}\nالإجابة المباشرة:"

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output_ids = model.generate(
        input_ids, max_new_tokens=2048,   # مساحة أكبر بكثير لنماذج الاستدلال
        temperature=0.0, do_sample=False
    )
    full_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # استخراج الإجابة النهائية بعد علامة الانتقال (حسب تنسيق النموذج)
    if "<إجابة_نهائية>" in full_response:
        final_answer = full_response.split("<إجابة_نهائية>")[-1].strip()
    else:
        final_answer = full_response.strip()
    return full_response, final_answer

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-09-evaluation/ch42-benchmarks.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-42-ch42-benchmarks.ipynb#cell-2
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-42-ch42-benchmarks.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def evaluate_mmlu_style(model, tokenizer, questions: list, choices_list: list, answers: list):
    """
    التقييم القياسي لأسئلة الاختيار من متعدد:
    نقارن log-likelihood لكل خيار، ونختار الأعلى (وليس التوليد الحر!)
    """
    correct = 0
    for question, choices, correct_answer_idx in zip(questions, choices_list, answers):
        choice_scores = []
        for choice in choices:
            prompt = f"{question}\nالإجابة: {choice}"
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
            # نستخدم negative loss كـ score (أعلى = احتمال أعلى)
            choice_scores.append(-outputs.loss.item())

        predicted_idx = choice_scores.index(max(choice_scores))
        if predicted_idx == correct_answer_idx:
            correct += 1

    return correct / len(questions)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-42-ch42-benchmarks.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def detect_contamination_via_perplexity_outlier(model, tokenizer, benchmark_samples: list,
                                                   reference_corpus_samples: list):
    """
    فكرة: إذا كان نموذج قد "حفظ" عيّنات benchmark أثناء التدريب،
    فإن perplexity عليها ستكون أقل بشكل غير طبيعي مقارنة بنصوص مشابهة لم يرها
    """
    def avg_ppl(samples):
        ppls = []
        for text in samples:
            input_ids = tokenizer(text, return_tensors="pt").input_ids
            with torch.no_grad():
                loss = model(input_ids, labels=input_ids).loss
            ppls.append(torch.exp(loss).item())
        return sum(ppls) / len(ppls)

    benchmark_ppl = avg_ppl(benchmark_samples)
    reference_ppl = avg_ppl(reference_corpus_samples)

    contamination_ratio = reference_ppl / benchmark_ppl
    # إذا كانت > 1.5-2x، هذا مؤشر قوي على احتمال التلوث (حفظ، وليس تعميم حقيقي)
    return {
        "benchmark_ppl": benchmark_ppl,
        "reference_ppl": reference_ppl,
        "contamination_suspicion_ratio": contamination_ratio
    }

def ngram_overlap_contamination_check(training_corpus_sample: list, benchmark_texts: list, n=13):
    """
    الطريقة الأكثر شيوعًا في التقارير الرسمية (مثل GPT-4 technical report):
    فحص تطابق n-grams طويلة (13 كلمة فأكثر) بين بيانات التدريب والـ benchmark
    """
    def get_ngrams(text, n):
        tokens = text.split()
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    contaminated_count = 0
    for benchmark_text in benchmark_texts:
        benchmark_ngrams = get_ngrams(benchmark_text, n)
        for train_text in training_corpus_sample:
            train_ngrams = get_ngrams(train_text, n)
            if benchmark_ngrams & train_ngrams:   # أي تقاطع يُعتبر تلوثًا محتملًا
                contaminated_count += 1
                break

    return contaminated_count / len(benchmark_texts)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-42-ch42-benchmarks.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def evaluate_with_chain_of_thought(model, tokenizer, question: str, use_reasoning_tokens=True):
    """
    تقييم أسلوب نماذج الاستدلال الحديثة: السماح بـ "مساحة تفكير" قبل الإجابة النهائية
    """
    if use_reasoning_tokens:
        prompt = f"""السؤال: {question}
فكّر خطوة بخطوة بعناية شديدة قبل الإجابة النهائية.
<تفكير>"""
    else:
        prompt = f"السؤال: {question}\nالإجابة المباشرة:"

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output_ids = model.generate(
        input_ids, max_new_tokens=2048,   # مساحة أكبر بكثير لنماذج الاستدلال
        temperature=0.0, do_sample=False
    )
    full_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # استخراج الإجابة النهائية بعد علامة الانتقال (حسب تنسيق النموذج)
    if "<إجابة_نهائية>" in full_response:
        final_answer = full_response.split("<إجابة_نهائية>")[-1].strip()
    else:
        final_answer = full_response.strip()
    return full_response, final_answer

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-42-ch42-benchmarks.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
