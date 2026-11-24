#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — معايير القياس الكبرى (Major Benchmarks: MMLU, HellaSwag, GSM8K, HumanEval)
Large Language Model Engineering — Major

Book slug : llm-engineering-69-chapters
Chapter   : ch-46-chapter-46 (lesson 46)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-46-chapter-46.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — معايير القياس الكبرى (Major Benchmarks: MMLU, HellaSwag, GSM8K, HumanEval)
Large Language Model Engineering — Major

Book slug : llm-engineering-69-chapters
Chapter   : ch-46-chapter-46 (lesson 46)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-46-chapter-46.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — معايير القياس الكبرى (Major Benchmarks: MMLU, HellaSwag, GSM8K, HumanEval)
Large Language Model Engineering — Major

Book slug : llm-engineering-69-chapters
Chapter   : ch-46-chapter-46 (lesson 46)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_46.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 46: الفصل 46: معايير القياس الكبرى (Major Benchmarks: MMLU, HellaSwag, GSM8K, HumanEval)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_benchmark() -> None:
    examples = [{'id': 'a', 'expected': 'yes', 'output': 'yes'}, {'id': 'b', 'expected': 'no', 'output': 'no'}, {'id': 'c', 'expected': 'yes', 'output': 'no'}]
    results = [{'id': item['id'], 'correct': item['expected'] == item['output']} for item in examples]
    print('per-example results:', results)
    print('accuracy:', round(sum((item['correct'] for item in results)) / len(results), 3))
    fingerprint = hashlib.sha256('|'.join((item['id'] for item in examples)).encode()).hexdigest()[:12]
    print('dataset fingerprint:', fingerprint)

if __name__ == "__main__":
    print("Chapter 46 lab: Run a deterministic miniature benchmark harness")
    run_benchmark()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-09-evaluation-testing/chapter-46.md
# ────────────────────────────────────────────────────────────────────────
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ تحقق مما إذا كان هناك رقمان في القائمة أقرب من threshold لبعضهما """
    # <-- النموذج يُكمِّل من هنا

METADATA = {...}
def check(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
    # ... المزيد من الاختبارات الخفية

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-09-evaluation-testing/chapter-46.md
# ────────────────────────────────────────────────────────────────────────
"""
Benchmark Evaluation Harness: MMLU + HellaSwag + GSM8K + HumanEval
يستخدم: PyTorch, transformers, datasets (Hugging Face)
تنبيه أمان: تنفيذ كود HumanEval يتطلب عزلًا (sandboxing) في الإنتاج الحقيقي.
"""

import re
import signal
import math
import itertools
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "gpt2"  # للتوضيح فقط؛ في الواقع تُستخدم نماذج أكبر بكثير (7B+)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
model.eval()


# ============================================================
# الجزء 1: أداة مساعدة لحساب log-likelihood لأي استكمال نصي
# ============================================================

@torch.no_grad()
def compute_log_likelihood(context: str, continuation: str) -> float:
    """
    يحسب مجموع log P(continuation | context) تحت النموذج.
    هذا هو حجر الأساس لتقييم MMLU وHellaSwag (الاختيار من متعدد عبر الاحتمال).
    """
    full_text = context + continuation
    context_ids = tokenizer(context, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)

    context_len = context_ids.size(1)
    continuation_len = full_ids.size(1) - context_len
    if continuation_len <= 0:
        return float("-inf")

    outputs = model(full_ids)
    logits = outputs.logits[0]  # (seq_len, vocab_size)

    log_probs = F.log_softmax(logits, dim=-1)

    total_log_prob = 0.0
    for i in range(context_len, full_ids.size(1)):
        token_id = full_ids[0, i]
        # logits[i-1] يتنبأ بالرمز رقم i (إزاحة نموذج اللغة المعتادة)
        total_log_prob += log_probs[i - 1, token_id].item()

    return total_log_prob


# ============================================================
# الجزء 2: MMLU — الاختيار من متعدد عبر log-likelihood (5-shot)
# ============================================================

MMLU_FEWSHOT_EXAMPLES = """The following are multiple choice questions about elementary mathematics.

Question: What is 15% of 200?
A) 20
B) 30
C) 40
D) 50
Answer: B

Question: If x + 5 = 12, what is x?
A) 5
B) 6
C) 7
D) 8
Answer: C

"""


def evaluate_mmlu_sample(question: str, choices: list[str], correct_idx: int) -> bool:
    """
    يُقيّم سؤال MMLU واحد: يحسب log-likelihood لكل خيار (A/B/C/D)
    ويختار الأعلى، تطبيقًا للصيغة الرياضية في القسم 🟡 أعلاه.
    """
    letters = ["A", "B", "C", "D"]
    prompt = MMLU_FEWSHOT_EXAMPLES + f"Question: {question}\n"
    for letter, choice in zip(letters, choices):
        prompt += f"{letter}) {choice}\n"
    prompt += "Answer:"

    log_likelihoods = []
    for letter in letters:
        ll = compute_log_likelihood(prompt, f" {letter}")
        log_likelihoods.append(ll)

    predicted_idx = log_likelihoods.index(max(log_likelihoods))
    return predicted_idx == correct_idx


def run_mmlu_eval(n_samples: int = 20) -> float:
    """يحمّل عينة من MMLU (مادة الفيزياء التشريحية كمثال) ويحسب الدقة."""
    dataset = load_dataset("cais/mmlu", "anatomy", split="test")
    dataset = dataset.select(range(min(n_samples, len(dataset))))

    correct = 0
    for example in dataset:
        is_correct = evaluate_mmlu_sample(
            question=example["question"],
            choices=example["choices"],
            correct_idx=example["answer"],
        )
        correct += int(is_correct)

    accuracy = correct / len(dataset)
    print(f"[MMLU - anatomy] الدقة على {len(dataset)} عينة: {accuracy:.2%}")
    return accuracy


# ============================================================
# الجزء 3: HellaSwag — log-likelihood مُطبَّع بالطول
# ============================================================

def evaluate_hellaswag_sample(context: str, endings: list[str], correct_idx: int) -> bool:
    """
    يُقيّم مثال HellaSwag واحد: log-likelihood مُطبَّع بطول كل خيار
    (الصيغة الرياضية في القسم 🟡: القسمة على |continuation|).
    """
    normalized_scores = []
    for ending in endings:
        ll = compute_log_likelihood(context, " " + ending)
        normalized_score = ll / max(len(ending), 1)  # تطبيع بطول الخيار بالأحرف
        normalized_scores.append(normalized_score)

    predicted_idx = normalized_scores.index(max(normalized_scores))
    return predicted_idx == correct_idx


def run_hellaswag_eval(n_samples: int = 20) -> float:
    dataset = load_dataset("Rowan/hellaswag", split="validation")
    dataset = dataset.select(range(min(n_samples, len(dataset))))

    correct = 0
    for example in dataset:
        is_correct = evaluate_hellaswag_sample(
            context=example["ctx"],
            endings=example["endings"],
            correct_idx=int(example["label"]),
        )
        correct += int(is_correct)

    accuracy = correct / len(dataset)
    print(f"[HellaSwag] الدقة على {len(dataset)} عينة: {accuracy:.2%}")
    return accuracy


# ============================================================
# الجزء 4: GSM8K — توليد حر + Chain-of-Thought + استخراج رقمي
# ============================================================

GSM8K_FEWSHOT_EXAMPLES = """Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: Natalia sold 48/2 = 24 clips in May.
Natalia sold 48+24 = 72 clips altogether in April and May.
#### 72

Question: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
Answer: It takes 2/2 = 1 bolt of white fiber.
So the total amount of fabric is 2+1 = 3 bolts.
#### 3

"""


def extract_final_number(text: str) -> str | None:
    """يستخرج الرقم النهائي بعد علامة '####' القياسية في GSM8K، أو آخر رقم في النص كبديل."""
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    return numbers[-1].replace(",", "") if numbers else None


@torch.no_grad()
def generate_gsm8k_answer(question: str, max_new_tokens: int = 256) -> str:
    prompt = GSM8K_FEWSHOT_EXAMPLES + f"Question: {question}\nAnswer:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # Greedy للحصول على نتائج قابلة لإعادة الإنتاج
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_text = tokenizer.decode(output_ids[0, input_ids.size(1):], skip_special_tokens=True)
    # نقف عند أول "Question:" جديد (بداية مثال تالٍ وهمي قد يولّده النموذج)
    generated_text = generated_text.split("Question:")[0]
    return generated_text


def run_gsm8k_eval(n_samples: int = 10) -> float:
    dataset = load_dataset("gsm8k", "main", split="test")
    dataset = dataset.select(range(min(n_samples, len(dataset))))

    correct = 0
    for example in dataset:
        generated = generate_gsm8k_answer(example["question"])
        predicted_number = extract_final_number(generated)
        gold_number = extract_final_number(example["answer"])

        is_correct = (predicted_number is not None and
                      gold_number is not None and
                      float(predicted_number) == float(gold_number))
        correct += int(is_correct)

    accuracy = correct / len(dataset)
    print(f"[GSM8K] الدقة (Exact Match) على {len(dataset)} عينة: {accuracy:.2%}")
    return accuracy


# ============================================================
# الجزء 5: HumanEval — تنفيذ فعلي للكود وحساب pass@k
# ============================================================

class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds: float):
    """حارس زمني لمنع الكود المولَّد من التعليق في حلقة لا نهائية."""
    def signal_handler(signum, frame):
        raise TimeoutException("انتهى الوقت المحدد لتنفيذ الكود")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)


def execute_code_safely(code: str, test_code: str, timeout: float = 5.0) -> bool:
    """
    ⚠️ تحذير أمني: هذا تنفيذ توضيحي مبسّط فقط.
    في الإنتاج الحقيقي، يجب تنفيذ الكود داخل حاوية Docker معزولة تمامًا
    (sandbox) بدون وصول للشبكة أو نظام الملفات، أبدًا exec() مباشرة
    على كود غير موثوق في بيئة الإنتاج.
    """
    full_program = code + "\n" + test_code
    try:
        with time_limit(timeout):
            exec_globals: dict = {}
            exec(full_program, exec_globals)
        return True
    except Exception:
        return False


@torch.no_grad()
def generate_code_completions(prompt: str, n_samples: int = 5,
                                temperature: float = 0.8) -> list[str]:
    """يولّد n إكمالات مختلفة لنفس مسألة البرمجة عبر Temperature Sampling (فصل 38)."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    completions = []
    for _ in range(n_samples):
        output_ids = model.generate(
            input_ids,
            max_new_tokens=128,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
        completion = tokenizer.decode(output_ids[0, input_ids.size(1):], skip_special_tokens=True)
        completions.append(completion)

    return completions


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    المُقدِّر غير المتحيّز الرسمي من ورقة Codex (Chen et al., 2021).
    n: إجمالي العينات المولَّدة، c: عدد العينات الناجحة، k: حجم العينة الفرعية المُفترضة.
    """
    if n - c < k:
        return 1.0  # يستحيل اختيار k عينة فاشلة بالكامل
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def run_humaneval_eval(n_problems: int = 5, n_samples_per_problem: int = 5) -> dict:
    """يقيّم عددًا من مسائل HumanEval ويحسب pass@1 وpass@5."""
    dataset = load_dataset("openai_humaneval", split="test")
    dataset = dataset.select(range(min(n_problems, len(dataset))))

    results = []
    for problem in dataset:
        prompt = problem["prompt"]
        test_code = problem["test"] + f"\ncheck({problem['entry_point']})"

        completions = generate_code_completions(prompt, n_samples=n_samples_per_problem)

        n_correct = 0
        for completion in completions:
            full_code = prompt + completion
            if execute_code_safely(full_code, test_code):
                n_correct += 1

        results.append({"n": n_samples_per_problem, "c": n_correct})
        print(f"  المسألة '{problem['entry_point']}': {n_correct}/{n_samples_per_problem} نجحت")

    pass_1 = sum(pass_at_k(r["n"], r["c"], 1) for r in results) / len(results)
    pass_5 = sum(pass_at_k(r["n"], r["c"], min(5, r["n"])) for r in results) / len(results)

    print(f"[HumanEval] pass@1: {pass_1:.2%} | pass@5: {pass_5:.2%}")
    return {"pass@1": pass_1, "pass@5": pass_5}


# ============================================================
# الجزء 6: تشغيل كل المعايير وإنتاج تقرير موحّد
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print(f"تقييم النموذج '{MODEL_NAME}' على المعايير الأربعة الكبرى")
    print("(ملاحظة: gpt2 صغير جدًا؛ الدرجات ستكون متواضعة جدًا كمثال توضيحي)")
    print("=" * 70)

    report = {}

    print("\n--- MMLU ---")
    report["MMLU (anatomy)"] = run_mmlu_eval(n_samples=10)

    print("\n--- HellaSwag ---")
    report["HellaSwag"] = run_hellaswag_eval(n_samples=10)

    print("\n--- GSM8K ---")
    report["GSM8K"] = run_gsm8k_eval(n_samples=5)

    print("\n--- HumanEval ---")
    humaneval_results = run_humaneval_eval(n_problems=3, n_samples_per_problem=5)
    report["HumanEval pass@1"] = humaneval_results["pass@1"]

    print("\n" + "=" * 70)
    print("التقرير النهائي الموحّد")
    print("=" * 70)
    for metric_name, score in report.items():
        print(f"  {metric_name:25s}: {score:.2%}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-46-chapter-46.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 46: الفصل 46: معايير القياس الكبرى (Major Benchmarks: MMLU, HellaSwag, GSM8K, HumanEval)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_benchmark() -> None:
    examples = [{'id': 'a', 'expected': 'yes', 'output': 'yes'}, {'id': 'b', 'expected': 'no', 'output': 'no'}, {'id': 'c', 'expected': 'yes', 'output': 'no'}]
    results = [{'id': item['id'], 'correct': item['expected'] == item['output']} for item in examples]
    print('per-example results:', results)
    print('accuracy:', round(sum((item['correct'] for item in results)) / len(results), 3))
    fingerprint = hashlib.sha256('|'.join((item['id'] for item in examples)).encode()).hexdigest()[:12]
    print('dataset fingerprint:', fingerprint)

if __name__ == "__main__":
    print("Chapter 46 lab: Run a deterministic miniature benchmark harness")
    run_benchmark()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-46-chapter-46.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ تحقق مما إذا كان هناك رقمان في القائمة أقرب من threshold لبعضهما """
    # <-- النموذج يُكمِّل من هنا

METADATA = {...}
def check(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
    # ... المزيد من الاختبارات الخفية

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-46-chapter-46.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
Benchmark Evaluation Harness: MMLU + HellaSwag + GSM8K + HumanEval
يستخدم: PyTorch, transformers, datasets (Hugging Face)
تنبيه أمان: تنفيذ كود HumanEval يتطلب عزلًا (sandboxing) في الإنتاج الحقيقي.
"""

import re
import signal
import math
import itertools
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "gpt2"  # للتوضيح فقط؛ في الواقع تُستخدم نماذج أكبر بكثير (7B+)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
model.eval()


# ============================================================
# الجزء 1: أداة مساعدة لحساب log-likelihood لأي استكمال نصي
# ============================================================

@torch.no_grad()
def compute_log_likelihood(context: str, continuation: str) -> float:
    """
    يحسب مجموع log P(continuation | context) تحت النموذج.
    هذا هو حجر الأساس لتقييم MMLU وHellaSwag (الاختيار من متعدد عبر الاحتمال).
    """
    full_text = context + continuation
    context_ids = tokenizer(context, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)

    context_len = context_ids.size(1)
    continuation_len = full_ids.size(1) - context_len
    if continuation_len <= 0:
        return float("-inf")

    outputs = model(full_ids)
    logits = outputs.logits[0]  # (seq_len, vocab_size)

    log_probs = F.log_softmax(logits, dim=-1)

    total_log_prob = 0.0
    for i in range(context_len, full_ids.size(1)):
        token_id = full_ids[0, i]
        # logits[i-1] يتنبأ بالرمز رقم i (إزاحة نموذج اللغة المعتادة)
        total_log_prob += log_probs[i - 1, token_id].item()

    return total_log_prob


# ============================================================
# الجزء 2: MMLU — الاختيار من متعدد عبر log-likelihood (5-shot)
# ============================================================

MMLU_FEWSHOT_EXAMPLES = """The following are multiple choice questions about elementary mathematics.

Question: What is 15% of 200?
A) 20
B) 30
C) 40
D) 50
Answer: B

Question: If x + 5 = 12, what is x?
A) 5
B) 6
C) 7
D) 8
Answer: C

"""


def evaluate_mmlu_sample(question: str, choices: list[str], correct_idx: int) -> bool:
    """
    يُقيّم سؤال MMLU واحد: يحسب log-likelihood لكل خيار (A/B/C/D)
    ويختار الأعلى، تطبيقًا للصيغة الرياضية في القسم 🟡 أعلاه.
    """
    letters = ["A", "B", "C", "D"]
    prompt = MMLU_FEWSHOT_EXAMPLES + f"Question: {question}\n"
    for letter, choice in zip(letters, choices):
        prompt += f"{letter}) {choice}\n"
    prompt += "Answer:"

    log_likelihoods = []
    for letter in letters:
        ll = compute_log_likelihood(prompt, f" {letter}")
        log_likelihoods.append(ll)

    predicted_idx = log_likelihoods.index(max(log_likelihoods))
    return predicted_idx == correct_idx


def run_mmlu_eval(n_samples: int = 20) -> float:
    """يحمّل عينة من MMLU (مادة الفيزياء التشريحية كمثال) ويحسب الدقة."""
    dataset = load_dataset("cais/mmlu", "anatomy", split="test")
    dataset = dataset.select(range(min(n_samples, len(dataset))))

    correct = 0
    for example in dataset:
        is_correct = evaluate_mmlu_sample(
            question=example["question"],
            choices=example["choices"],
            correct_idx=example["answer"],
        )
        correct += int(is_correct)

    accuracy = correct / len(dataset)
    print(f"[MMLU - anatomy] الدقة على {len(dataset)} عينة: {accuracy:.2%}")
    return accuracy


# ============================================================
# الجزء 3: HellaSwag — log-likelihood مُطبَّع بالطول
# ============================================================

def evaluate_hellaswag_sample(context: str, endings: list[str], correct_idx: int) -> bool:
    """
    يُقيّم مثال HellaSwag واحد: log-likelihood مُطبَّع بطول كل خيار
    (الصيغة الرياضية في القسم 🟡: القسمة على |continuation|).
    """
    normalized_scores = []
    for ending in endings:
        ll = compute_log_likelihood(context, " " + ending)
        normalized_score = ll / max(len(ending), 1)  # تطبيع بطول الخيار بالأحرف
        normalized_scores.append(normalized_score)

    predicted_idx = normalized_scores.index(max(normalized_scores))
    return predicted_idx == correct_idx


def run_hellaswag_eval(n_samples: int = 20) -> float:
    dataset = load_dataset("Rowan/hellaswag", split="validation")
    dataset = dataset.select(range(min(n_samples, len(dataset))))

    correct = 0
    for example in dataset:
        is_correct = evaluate_hellaswag_sample(
            context=example["ctx"],
            endings=example["endings"],
            correct_idx=int(example["label"]),
        )
        correct += int(is_correct)

    accuracy = correct / len(dataset)
    print(f"[HellaSwag] الدقة على {len(dataset)} عينة: {accuracy:.2%}")
    return accuracy


# ============================================================
# الجزء 4: GSM8K — توليد حر + Chain-of-Thought + استخراج رقمي
# ============================================================

GSM8K_FEWSHOT_EXAMPLES = """Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: Natalia sold 48/2 = 24 clips in May.
Natalia sold 48+24 = 72 clips altogether in April and May.
#### 72

Question: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
Answer: It takes 2/2 = 1 bolt of white fiber.
So the total amount of fabric is 2+1 = 3 bolts.
#### 3

"""


def extract_final_number(text: str) -> str | None:
    """يستخرج الرقم النهائي بعد علامة '####' القياسية في GSM8K، أو آخر رقم في النص كبديل."""
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    return numbers[-1].replace(",", "") if numbers else None


@torch.no_grad()
def generate_gsm8k_answer(question: str, max_new_tokens: int = 256) -> str:
    prompt = GSM8K_FEWSHOT_EXAMPLES + f"Question: {question}\nAnswer:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # Greedy للحصول على نتائج قابلة لإعادة الإنتاج
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_text = tokenizer.decode(output_ids[0, input_ids.size(1):], skip_special_tokens=True)
    # نقف عند أول "Question:" جديد (بداية مثال تالٍ وهمي قد يولّده النموذج)
    generated_text = generated_text.split("Question:")[0]
    return generated_text


def run_gsm8k_eval(n_samples: int = 10) -> float:
    dataset = load_dataset("gsm8k", "main", split="test")
    dataset = dataset.select(range(min(n_samples, len(dataset))))

    correct = 0
    for example in dataset:
        generated = generate_gsm8k_answer(example["question"])
        predicted_number = extract_final_number(generated)
        gold_number = extract_final_number(example["answer"])

        is_correct = (predicted_number is not None and
                      gold_number is not None and
                      float(predicted_number) == float(gold_number))
        correct += int(is_correct)

    accuracy = correct / len(dataset)
    print(f"[GSM8K] الدقة (Exact Match) على {len(dataset)} عينة: {accuracy:.2%}")
    return accuracy


# ============================================================
# الجزء 5: HumanEval — تنفيذ فعلي للكود وحساب pass@k
# ============================================================

class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds: float):
    """حارس زمني لمنع الكود المولَّد من التعليق في حلقة لا نهائية."""
    def signal_handler(signum, frame):
        raise TimeoutException("انتهى الوقت المحدد لتنفيذ الكود")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)


def execute_code_safely(code: str, test_code: str, timeout: float = 5.0) -> bool:
    """
    ⚠️ تحذير أمني: هذا تنفيذ توضيحي مبسّط فقط.
    في الإنتاج الحقيقي، يجب تنفيذ الكود داخل حاوية Docker معزولة تمامًا
    (sandbox) بدون وصول للشبكة أو نظام الملفات، أبدًا exec() مباشرة
    على كود غير موثوق في بيئة الإنتاج.
    """
    full_program = code + "\n" + test_code
    try:
        with time_limit(timeout):
            exec_globals: dict = {}
            exec(full_program, exec_globals)
        return True
    except Exception:
        return False


@torch.no_grad()
def generate_code_completions(prompt: str, n_samples: int = 5,
                                temperature: float = 0.8) -> list[str]:
    """يولّد n إكمالات مختلفة لنفس مسألة البرمجة عبر Temperature Sampling (فصل 38)."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    completions = []
    for _ in range(n_samples):
        output_ids = model.generate(
            input_ids,
            max_new_tokens=128,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
        completion = tokenizer.decode(output_ids[0, input_ids.size(1):], skip_special_tokens=True)
        completions.append(completion)

    return completions


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    المُقدِّر غير المتحيّز الرسمي من ورقة Codex (Chen et al., 2021).
    n: إجمالي العينات المولَّدة، c: عدد العينات الناجحة، k: حجم العينة الفرعية المُفترضة.
    """
    if n - c < k:
        return 1.0  # يستحيل اختيار k عينة فاشلة بالكامل
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def run_humaneval_eval(n_problems: int = 5, n_samples_per_problem: int = 5) -> dict:
    """يقيّم عددًا من مسائل HumanEval ويحسب pass@1 وpass@5."""
    dataset = load_dataset("openai_humaneval", split="test")
    dataset = dataset.select(range(min(n_problems, len(dataset))))

    results = []
    for problem in dataset:
        prompt = problem["prompt"]
        test_code = problem["test"] + f"\ncheck({problem['entry_point']})"

        completions = generate_code_completions(prompt, n_samples=n_samples_per_problem)

        n_correct = 0
        for completion in completions:
            full_code = prompt + completion
            if execute_code_safely(full_code, test_code):
                n_correct += 1

        results.append({"n": n_samples_per_problem, "c": n_correct})
        print(f"  المسألة '{problem['entry_point']}': {n_correct}/{n_samples_per_problem} نجحت")

    pass_1 = sum(pass_at_k(r["n"], r["c"], 1) for r in results) / len(results)
    pass_5 = sum(pass_at_k(r["n"], r["c"], min(5, r["n"])) for r in results) / len(results)

    print(f"[HumanEval] pass@1: {pass_1:.2%} | pass@5: {pass_5:.2%}")
    return {"pass@1": pass_1, "pass@5": pass_5}


# ============================================================
# الجزء 6: تشغيل كل المعايير وإنتاج تقرير موحّد
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print(f"تقييم النموذج '{MODEL_NAME}' على المعايير الأربعة الكبرى")
    print("(ملاحظة: gpt2 صغير جدًا؛ الدرجات ستكون متواضعة جدًا كمثال توضيحي)")
    print("=" * 70)

    report = {}

    print("\n--- MMLU ---")
    report["MMLU (anatomy)"] = run_mmlu_eval(n_samples=10)

    print("\n--- HellaSwag ---")
    report["HellaSwag"] = run_hellaswag_eval(n_samples=10)

    print("\n--- GSM8K ---")
    report["GSM8K"] = run_gsm8k_eval(n_samples=5)

    print("\n--- HumanEval ---")
    humaneval_results = run_humaneval_eval(n_problems=3, n_samples_per_problem=5)
    report["HumanEval pass@1"] = humaneval_results["pass@1"]

    print("\n" + "=" * 70)
    print("التقرير النهائي الموحّد")
    print("=" * 70)
    for metric_name, score in report.items():
        print(f"  {metric_name:25s}: {score:.2%}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-46-chapter-46.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — معايير القياس الكبرى (Major Benchmarks: MMLU, HellaSwag, GSM8K, HumanEval)
Large Language Model Engineering — Major

Book slug : llm-engineering-69-chapters
Chapter   : ch-46-chapter-46 (lesson 46)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_46.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 46: الفصل 46: معايير القياس الكبرى (Major Benchmarks: MMLU, HellaSwag, GSM8K, HumanEval)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_benchmark() -> None:
    examples = [{'id': 'a', 'expected': 'yes', 'output': 'yes'}, {'id': 'b', 'expected': 'no', 'output': 'no'}, {'id': 'c', 'expected': 'yes', 'output': 'no'}]
    results = [{'id': item['id'], 'correct': item['expected'] == item['output']} for item in examples]
    print('per-example results:', results)
    print('accuracy:', round(sum((item['correct'] for item in results)) / len(results), 3))
    fingerprint = hashlib.sha256('|'.join((item['id'] for item in examples)).encode()).hexdigest()[:12]
    print('dataset fingerprint:', fingerprint)

if __name__ == "__main__":
    print("Chapter 46 lab: Run a deterministic miniature benchmark harness")
    run_benchmark()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-09-evaluation-testing/chapter-46.md
# ────────────────────────────────────────────────────────────────────────
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ تحقق مما إذا كان هناك رقمان في القائمة أقرب من threshold لبعضهما """
    # <-- النموذج يُكمِّل من هنا

METADATA = {...}
def check(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
    # ... المزيد من الاختبارات الخفية

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-09-evaluation-testing/chapter-46.md
# ────────────────────────────────────────────────────────────────────────
"""
Benchmark Evaluation Harness: MMLU + HellaSwag + GSM8K + HumanEval
يستخدم: PyTorch, transformers, datasets (Hugging Face)
تنبيه أمان: تنفيذ كود HumanEval يتطلب عزلًا (sandboxing) في الإنتاج الحقيقي.
"""

import re
import signal
import math
import itertools
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "gpt2"  # للتوضيح فقط؛ في الواقع تُستخدم نماذج أكبر بكثير (7B+)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
model.eval()


# ============================================================
# الجزء 1: أداة مساعدة لحساب log-likelihood لأي استكمال نصي
# ============================================================

@torch.no_grad()
def compute_log_likelihood(context: str, continuation: str) -> float:
    """
    يحسب مجموع log P(continuation | context) تحت النموذج.
    هذا هو حجر الأساس لتقييم MMLU وHellaSwag (الاختيار من متعدد عبر الاحتمال).
    """
    full_text = context + continuation
    context_ids = tokenizer(context, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)

    context_len = context_ids.size(1)
    continuation_len = full_ids.size(1) - context_len
    if continuation_len <= 0:
        return float("-inf")

    outputs = model(full_ids)
    logits = outputs.logits[0]  # (seq_len, vocab_size)

    log_probs = F.log_softmax(logits, dim=-1)

    total_log_prob = 0.0
    for i in range(context_len, full_ids.size(1)):
        token_id = full_ids[0, i]
        # logits[i-1] يتنبأ بالرمز رقم i (إزاحة نموذج اللغة المعتادة)
        total_log_prob += log_probs[i - 1, token_id].item()

    return total_log_prob


# ============================================================
# الجزء 2: MMLU — الاختيار من متعدد عبر log-likelihood (5-shot)
# ============================================================

MMLU_FEWSHOT_EXAMPLES = """The following are multiple choice questions about elementary mathematics.

Question: What is 15% of 200?
A) 20
B) 30
C) 40
D) 50
Answer: B

Question: If x + 5 = 12, what is x?
A) 5
B) 6
C) 7
D) 8
Answer: C

"""


def evaluate_mmlu_sample(question: str, choices: list[str], correct_idx: int) -> bool:
    """
    يُقيّم سؤال MMLU واحد: يحسب log-likelihood لكل خيار (A/B/C/D)
    ويختار الأعلى، تطبيقًا للصيغة الرياضية في القسم 🟡 أعلاه.
    """
    letters = ["A", "B", "C", "D"]
    prompt = MMLU_FEWSHOT_EXAMPLES + f"Question: {question}\n"
    for letter, choice in zip(letters, choices):
        prompt += f"{letter}) {choice}\n"
    prompt += "Answer:"

    log_likelihoods = []
    for letter in letters:
        ll = compute_log_likelihood(prompt, f" {letter}")
        log_likelihoods.append(ll)

    predicted_idx = log_likelihoods.index(max(log_likelihoods))
    return predicted_idx == correct_idx


def run_mmlu_eval(n_samples: int = 20) -> float:
    """يحمّل عينة من MMLU (مادة الفيزياء التشريحية كمثال) ويحسب الدقة."""
    dataset = load_dataset("cais/mmlu", "anatomy", split="test")
    dataset = dataset.select(range(min(n_samples, len(dataset))))

    correct = 0
    for example in dataset:
        is_correct = evaluate_mmlu_sample(
            question=example["question"],
            choices=example["choices"],
            correct_idx=example["answer"],
        )
        correct += int(is_correct)

    accuracy = correct / len(dataset)
    print(f"[MMLU - anatomy] الدقة على {len(dataset)} عينة: {accuracy:.2%}")
    return accuracy


# ============================================================
# الجزء 3: HellaSwag — log-likelihood مُطبَّع بالطول
# ============================================================

def evaluate_hellaswag_sample(context: str, endings: list[str], correct_idx: int) -> bool:
    """
    يُقيّم مثال HellaSwag واحد: log-likelihood مُطبَّع بطول كل خيار
    (الصيغة الرياضية في القسم 🟡: القسمة على |continuation|).
    """
    normalized_scores = []
    for ending in endings:
        ll = compute_log_likelihood(context, " " + ending)
        normalized_score = ll / max(len(ending), 1)  # تطبيع بطول الخيار بالأحرف
        normalized_scores.append(normalized_score)

    predicted_idx = normalized_scores.index(max(normalized_scores))
    return predicted_idx == correct_idx


def run_hellaswag_eval(n_samples: int = 20) -> float:
    dataset = load_dataset("Rowan/hellaswag", split="validation")
    dataset = dataset.select(range(min(n_samples, len(dataset))))

    correct = 0
    for example in dataset:
        is_correct = evaluate_hellaswag_sample(
            context=example["ctx"],
            endings=example["endings"],
            correct_idx=int(example["label"]),
        )
        correct += int(is_correct)

    accuracy = correct / len(dataset)
    print(f"[HellaSwag] الدقة على {len(dataset)} عينة: {accuracy:.2%}")
    return accuracy


# ============================================================
# الجزء 4: GSM8K — توليد حر + Chain-of-Thought + استخراج رقمي
# ============================================================

GSM8K_FEWSHOT_EXAMPLES = """Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: Natalia sold 48/2 = 24 clips in May.
Natalia sold 48+24 = 72 clips altogether in April and May.
#### 72

Question: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
Answer: It takes 2/2 = 1 bolt of white fiber.
So the total amount of fabric is 2+1 = 3 bolts.
#### 3

"""


def extract_final_number(text: str) -> str | None:
    """يستخرج الرقم النهائي بعد علامة '####' القياسية في GSM8K، أو آخر رقم في النص كبديل."""
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    return numbers[-1].replace(",", "") if numbers else None


@torch.no_grad()
def generate_gsm8k_answer(question: str, max_new_tokens: int = 256) -> str:
    prompt = GSM8K_FEWSHOT_EXAMPLES + f"Question: {question}\nAnswer:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # Greedy للحصول على نتائج قابلة لإعادة الإنتاج
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_text = tokenizer.decode(output_ids[0, input_ids.size(1):], skip_special_tokens=True)
    # نقف عند أول "Question:" جديد (بداية مثال تالٍ وهمي قد يولّده النموذج)
    generated_text = generated_text.split("Question:")[0]
    return generated_text


def run_gsm8k_eval(n_samples: int = 10) -> float:
    dataset = load_dataset("gsm8k", "main", split="test")
    dataset = dataset.select(range(min(n_samples, len(dataset))))

    correct = 0
    for example in dataset:
        generated = generate_gsm8k_answer(example["question"])
        predicted_number = extract_final_number(generated)
        gold_number = extract_final_number(example["answer"])

        is_correct = (predicted_number is not None and
                      gold_number is not None and
                      float(predicted_number) == float(gold_number))
        correct += int(is_correct)

    accuracy = correct / len(dataset)
    print(f"[GSM8K] الدقة (Exact Match) على {len(dataset)} عينة: {accuracy:.2%}")
    return accuracy


# ============================================================
# الجزء 5: HumanEval — تنفيذ فعلي للكود وحساب pass@k
# ============================================================

class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds: float):
    """حارس زمني لمنع الكود المولَّد من التعليق في حلقة لا نهائية."""
    def signal_handler(signum, frame):
        raise TimeoutException("انتهى الوقت المحدد لتنفيذ الكود")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)


def execute_code_safely(code: str, test_code: str, timeout: float = 5.0) -> bool:
    """
    ⚠️ تحذير أمني: هذا تنفيذ توضيحي مبسّط فقط.
    في الإنتاج الحقيقي، يجب تنفيذ الكود داخل حاوية Docker معزولة تمامًا
    (sandbox) بدون وصول للشبكة أو نظام الملفات، أبدًا exec() مباشرة
    على كود غير موثوق في بيئة الإنتاج.
    """
    full_program = code + "\n" + test_code
    try:
        with time_limit(timeout):
            exec_globals: dict = {}
            exec(full_program, exec_globals)
        return True
    except Exception:
        return False


@torch.no_grad()
def generate_code_completions(prompt: str, n_samples: int = 5,
                                temperature: float = 0.8) -> list[str]:
    """يولّد n إكمالات مختلفة لنفس مسألة البرمجة عبر Temperature Sampling (فصل 38)."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    completions = []
    for _ in range(n_samples):
        output_ids = model.generate(
            input_ids,
            max_new_tokens=128,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
        completion = tokenizer.decode(output_ids[0, input_ids.size(1):], skip_special_tokens=True)
        completions.append(completion)

    return completions


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    المُقدِّر غير المتحيّز الرسمي من ورقة Codex (Chen et al., 2021).
    n: إجمالي العينات المولَّدة، c: عدد العينات الناجحة، k: حجم العينة الفرعية المُفترضة.
    """
    if n - c < k:
        return 1.0  # يستحيل اختيار k عينة فاشلة بالكامل
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def run_humaneval_eval(n_problems: int = 5, n_samples_per_problem: int = 5) -> dict:
    """يقيّم عددًا من مسائل HumanEval ويحسب pass@1 وpass@5."""
    dataset = load_dataset("openai_humaneval", split="test")
    dataset = dataset.select(range(min(n_problems, len(dataset))))

    results = []
    for problem in dataset:
        prompt = problem["prompt"]
        test_code = problem["test"] + f"\ncheck({problem['entry_point']})"

        completions = generate_code_completions(prompt, n_samples=n_samples_per_problem)

        n_correct = 0
        for completion in completions:
            full_code = prompt + completion
            if execute_code_safely(full_code, test_code):
                n_correct += 1

        results.append({"n": n_samples_per_problem, "c": n_correct})
        print(f"  المسألة '{problem['entry_point']}': {n_correct}/{n_samples_per_problem} نجحت")

    pass_1 = sum(pass_at_k(r["n"], r["c"], 1) for r in results) / len(results)
    pass_5 = sum(pass_at_k(r["n"], r["c"], min(5, r["n"])) for r in results) / len(results)

    print(f"[HumanEval] pass@1: {pass_1:.2%} | pass@5: {pass_5:.2%}")
    return {"pass@1": pass_1, "pass@5": pass_5}


# ============================================================
# الجزء 6: تشغيل كل المعايير وإنتاج تقرير موحّد
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print(f"تقييم النموذج '{MODEL_NAME}' على المعايير الأربعة الكبرى")
    print("(ملاحظة: gpt2 صغير جدًا؛ الدرجات ستكون متواضعة جدًا كمثال توضيحي)")
    print("=" * 70)

    report = {}

    print("\n--- MMLU ---")
    report["MMLU (anatomy)"] = run_mmlu_eval(n_samples=10)

    print("\n--- HellaSwag ---")
    report["HellaSwag"] = run_hellaswag_eval(n_samples=10)

    print("\n--- GSM8K ---")
    report["GSM8K"] = run_gsm8k_eval(n_samples=5)

    print("\n--- HumanEval ---")
    humaneval_results = run_humaneval_eval(n_problems=3, n_samples_per_problem=5)
    report["HumanEval pass@1"] = humaneval_results["pass@1"]

    print("\n" + "=" * 70)
    print("التقرير النهائي الموحّد")
    print("=" * 70)
    for metric_name, score in report.items():
        print(f"  {metric_name:25s}: {score:.2%}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-46-chapter-46.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 46: الفصل 46: معايير القياس الكبرى (Major Benchmarks: MMLU, HellaSwag, GSM8K, HumanEval)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_benchmark() -> None:
    examples = [{'id': 'a', 'expected': 'yes', 'output': 'yes'}, {'id': 'b', 'expected': 'no', 'output': 'no'}, {'id': 'c', 'expected': 'yes', 'output': 'no'}]
    results = [{'id': item['id'], 'correct': item['expected'] == item['output']} for item in examples]
    print('per-example results:', results)
    print('accuracy:', round(sum((item['correct'] for item in results)) / len(results), 3))
    fingerprint = hashlib.sha256('|'.join((item['id'] for item in examples)).encode()).hexdigest()[:12]
    print('dataset fingerprint:', fingerprint)

if __name__ == "__main__":
    print("Chapter 46 lab: Run a deterministic miniature benchmark harness")
    run_benchmark()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-46-chapter-46.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ تحقق مما إذا كان هناك رقمان في القائمة أقرب من threshold لبعضهما """
    # <-- النموذج يُكمِّل من هنا

METADATA = {...}
def check(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
    # ... المزيد من الاختبارات الخفية

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-46-chapter-46.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
"""
Benchmark Evaluation Harness: MMLU + HellaSwag + GSM8K + HumanEval
يستخدم: PyTorch, transformers, datasets (Hugging Face)
تنبيه أمان: تنفيذ كود HumanEval يتطلب عزلًا (sandboxing) في الإنتاج الحقيقي.
"""

import re
import signal
import math
import itertools
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "gpt2"  # للتوضيح فقط؛ في الواقع تُستخدم نماذج أكبر بكثير (7B+)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
model.eval()


# ============================================================
# الجزء 1: أداة مساعدة لحساب log-likelihood لأي استكمال نصي
# ============================================================

@torch.no_grad()
def compute_log_likelihood(context: str, continuation: str) -> float:
    """
    يحسب مجموع log P(continuation | context) تحت النموذج.
    هذا هو حجر الأساس لتقييم MMLU وHellaSwag (الاختيار من متعدد عبر الاحتمال).
    """
    full_text = context + continuation
    context_ids = tokenizer(context, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)

    context_len = context_ids.size(1)
    continuation_len = full_ids.size(1) - context_len
    if continuation_len <= 0:
        return float("-inf")

    outputs = model(full_ids)
    logits = outputs.logits[0]  # (seq_len, vocab_size)

    log_probs = F.log_softmax(logits, dim=-1)

    total_log_prob = 0.0
    for i in range(context_len, full_ids.size(1)):
        token_id = full_ids[0, i]
        # logits[i-1] يتنبأ بالرمز رقم i (إزاحة نموذج اللغة المعتادة)
        total_log_prob += log_probs[i - 1, token_id].item()

    return total_log_prob


# ============================================================
# الجزء 2: MMLU — الاختيار من متعدد عبر log-likelihood (5-shot)
# ============================================================

MMLU_FEWSHOT_EXAMPLES = """The following are multiple choice questions about elementary mathematics.

Question: What is 15% of 200?
A) 20
B) 30
C) 40
D) 50
Answer: B

Question: If x + 5 = 12, what is x?
A) 5
B) 6
C) 7
D) 8
Answer: C

"""


def evaluate_mmlu_sample(question: str, choices: list[str], correct_idx: int) -> bool:
    """
    يُقيّم سؤال MMLU واحد: يحسب log-likelihood لكل خيار (A/B/C/D)
    ويختار الأعلى، تطبيقًا للصيغة الرياضية في القسم 🟡 أعلاه.
    """
    letters = ["A", "B", "C", "D"]
    prompt = MMLU_FEWSHOT_EXAMPLES + f"Question: {question}\n"
    for letter, choice in zip(letters, choices):
        prompt += f"{letter}) {choice}\n"
    prompt += "Answer:"

    log_likelihoods = []
    for letter in letters:
        ll = compute_log_likelihood(prompt, f" {letter}")
        log_likelihoods.append(ll)

    predicted_idx = log_likelihoods.index(max(log_likelihoods))
    return predicted_idx == correct_idx


def run_mmlu_eval(n_samples: int = 20) -> float:
    """يحمّل عينة من MMLU (مادة الفيزياء التشريحية كمثال) ويحسب الدقة."""
    dataset = load_dataset("cais/mmlu", "anatomy", split="test")
    dataset = dataset.select(range(min(n_samples, len(dataset))))

    correct = 0
    for example in dataset:
        is_correct = evaluate_mmlu_sample(
            question=example["question"],
            choices=example["choices"],
            correct_idx=example["answer"],
        )
        correct += int(is_correct)

    accuracy = correct / len(dataset)
    print(f"[MMLU - anatomy] الدقة على {len(dataset)} عينة: {accuracy:.2%}")
    return accuracy


# ============================================================
# الجزء 3: HellaSwag — log-likelihood مُطبَّع بالطول
# ============================================================

def evaluate_hellaswag_sample(context: str, endings: list[str], correct_idx: int) -> bool:
    """
    يُقيّم مثال HellaSwag واحد: log-likelihood مُطبَّع بطول كل خيار
    (الصيغة الرياضية في القسم 🟡: القسمة على |continuation|).
    """
    normalized_scores = []
    for ending in endings:
        ll = compute_log_likelihood(context, " " + ending)
        normalized_score = ll / max(len(ending), 1)  # تطبيع بطول الخيار بالأحرف
        normalized_scores.append(normalized_score)

    predicted_idx = normalized_scores.index(max(normalized_scores))
    return predicted_idx == correct_idx


def run_hellaswag_eval(n_samples: int = 20) -> float:
    dataset = load_dataset("Rowan/hellaswag", split="validation")
    dataset = dataset.select(range(min(n_samples, len(dataset))))

    correct = 0
    for example in dataset:
        is_correct = evaluate_hellaswag_sample(
            context=example["ctx"],
            endings=example["endings"],
            correct_idx=int(example["label"]),
        )
        correct += int(is_correct)

    accuracy = correct / len(dataset)
    print(f"[HellaSwag] الدقة على {len(dataset)} عينة: {accuracy:.2%}")
    return accuracy


# ============================================================
# الجزء 4: GSM8K — توليد حر + Chain-of-Thought + استخراج رقمي
# ============================================================

GSM8K_FEWSHOT_EXAMPLES = """Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: Natalia sold 48/2 = 24 clips in May.
Natalia sold 48+24 = 72 clips altogether in April and May.
#### 72

Question: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
Answer: It takes 2/2 = 1 bolt of white fiber.
So the total amount of fabric is 2+1 = 3 bolts.
#### 3

"""


def extract_final_number(text: str) -> str | None:
    """يستخرج الرقم النهائي بعد علامة '####' القياسية في GSM8K، أو آخر رقم في النص كبديل."""
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    return numbers[-1].replace(",", "") if numbers else None


@torch.no_grad()
def generate_gsm8k_answer(question: str, max_new_tokens: int = 256) -> str:
    prompt = GSM8K_FEWSHOT_EXAMPLES + f"Question: {question}\nAnswer:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # Greedy للحصول على نتائج قابلة لإعادة الإنتاج
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_text = tokenizer.decode(output_ids[0, input_ids.size(1):], skip_special_tokens=True)
    # نقف عند أول "Question:" جديد (بداية مثال تالٍ وهمي قد يولّده النموذج)
    generated_text = generated_text.split("Question:")[0]
    return generated_text


def run_gsm8k_eval(n_samples: int = 10) -> float:
    dataset = load_dataset("gsm8k", "main", split="test")
    dataset = dataset.select(range(min(n_samples, len(dataset))))

    correct = 0
    for example in dataset:
        generated = generate_gsm8k_answer(example["question"])
        predicted_number = extract_final_number(generated)
        gold_number = extract_final_number(example["answer"])

        is_correct = (predicted_number is not None and
                      gold_number is not None and
                      float(predicted_number) == float(gold_number))
        correct += int(is_correct)

    accuracy = correct / len(dataset)
    print(f"[GSM8K] الدقة (Exact Match) على {len(dataset)} عينة: {accuracy:.2%}")
    return accuracy


# ============================================================
# الجزء 5: HumanEval — تنفيذ فعلي للكود وحساب pass@k
# ============================================================

class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds: float):
    """حارس زمني لمنع الكود المولَّد من التعليق في حلقة لا نهائية."""
    def signal_handler(signum, frame):
        raise TimeoutException("انتهى الوقت المحدد لتنفيذ الكود")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)


def execute_code_safely(code: str, test_code: str, timeout: float = 5.0) -> bool:
    """
    ⚠️ تحذير أمني: هذا تنفيذ توضيحي مبسّط فقط.
    في الإنتاج الحقيقي، يجب تنفيذ الكود داخل حاوية Docker معزولة تمامًا
    (sandbox) بدون وصول للشبكة أو نظام الملفات، أبدًا exec() مباشرة
    على كود غير موثوق في بيئة الإنتاج.
    """
    full_program = code + "\n" + test_code
    try:
        with time_limit(timeout):
            exec_globals: dict = {}
            exec(full_program, exec_globals)
        return True
    except Exception:
        return False


@torch.no_grad()
def generate_code_completions(prompt: str, n_samples: int = 5,
                                temperature: float = 0.8) -> list[str]:
    """يولّد n إكمالات مختلفة لنفس مسألة البرمجة عبر Temperature Sampling (فصل 38)."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    completions = []
    for _ in range(n_samples):
        output_ids = model.generate(
            input_ids,
            max_new_tokens=128,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
        completion = tokenizer.decode(output_ids[0, input_ids.size(1):], skip_special_tokens=True)
        completions.append(completion)

    return completions


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    المُقدِّر غير المتحيّز الرسمي من ورقة Codex (Chen et al., 2021).
    n: إجمالي العينات المولَّدة، c: عدد العينات الناجحة، k: حجم العينة الفرعية المُفترضة.
    """
    if n - c < k:
        return 1.0  # يستحيل اختيار k عينة فاشلة بالكامل
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def run_humaneval_eval(n_problems: int = 5, n_samples_per_problem: int = 5) -> dict:
    """يقيّم عددًا من مسائل HumanEval ويحسب pass@1 وpass@5."""
    dataset = load_dataset("openai_humaneval", split="test")
    dataset = dataset.select(range(min(n_problems, len(dataset))))

    results = []
    for problem in dataset:
        prompt = problem["prompt"]
        test_code = problem["test"] + f"\ncheck({problem['entry_point']})"

        completions = generate_code_completions(prompt, n_samples=n_samples_per_problem)

        n_correct = 0
        for completion in completions:
            full_code = prompt + completion
            if execute_code_safely(full_code, test_code):
                n_correct += 1

        results.append({"n": n_samples_per_problem, "c": n_correct})
        print(f"  المسألة '{problem['entry_point']}': {n_correct}/{n_samples_per_problem} نجحت")

    pass_1 = sum(pass_at_k(r["n"], r["c"], 1) for r in results) / len(results)
    pass_5 = sum(pass_at_k(r["n"], r["c"], min(5, r["n"])) for r in results) / len(results)

    print(f"[HumanEval] pass@1: {pass_1:.2%} | pass@5: {pass_5:.2%}")
    return {"pass@1": pass_1, "pass@5": pass_5}


# ============================================================
# الجزء 6: تشغيل كل المعايير وإنتاج تقرير موحّد
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print(f"تقييم النموذج '{MODEL_NAME}' على المعايير الأربعة الكبرى")
    print("(ملاحظة: gpt2 صغير جدًا؛ الدرجات ستكون متواضعة جدًا كمثال توضيحي)")
    print("=" * 70)

    report = {}

    print("\n--- MMLU ---")
    report["MMLU (anatomy)"] = run_mmlu_eval(n_samples=10)

    print("\n--- HellaSwag ---")
    report["HellaSwag"] = run_hellaswag_eval(n_samples=10)

    print("\n--- GSM8K ---")
    report["GSM8K"] = run_gsm8k_eval(n_samples=5)

    print("\n--- HumanEval ---")
    humaneval_results = run_humaneval_eval(n_problems=3, n_samples_per_problem=5)
    report["HumanEval pass@1"] = humaneval_results["pass@1"]

    print("\n" + "=" * 70)
    print("التقرير النهائي الموحّد")
    print("=" * 70)
    for metric_name, score in report.items():
        print(f"  {metric_name:25s}: {score:.2%}")
