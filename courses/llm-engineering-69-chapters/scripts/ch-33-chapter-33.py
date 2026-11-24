#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — هندسة الأوامر (Prompt Engineering)
Large Language Model Engineering — Prompt

Book slug : llm-engineering-69-chapters
Chapter   : ch-33-chapter-33 (lesson 33)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-33-chapter-33.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — هندسة الأوامر (Prompt Engineering)
Large Language Model Engineering — Prompt

Book slug : llm-engineering-69-chapters
Chapter   : ch-33-chapter-33 (lesson 33)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-33-chapter-33.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — هندسة الأوامر (Prompt Engineering)
Large Language Model Engineering — Prompt

Book slug : llm-engineering-69-chapters
Chapter   : ch-33-chapter-33 (lesson 33)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_33.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 33: الفصل 33: هندسة الأوامر (Prompt Engineering)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_prompt() -> None:
    prompt = {'role': 'assistant', 'task': 'summarize', 'constraints': ['under 30 words', 'cite supplied text'], 'input': 'Attention combines token representations.'}
    serialized = json.dumps(prompt, ensure_ascii=False, indent=2)
    parsed = json.loads(serialized)
    assert parsed['role'] == 'assistant' and len(parsed['constraints']) == 2
    print(serialized)
    print('required fields validated:', sorted(parsed))

if __name__ == "__main__":
    print("Chapter 33 lab: Build and validate a structured prompt")
    run_prompt()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-07-fine-tuning-alignment/chapter-33.md
# ────────────────────────────────────────────────────────────────────────
"""
نظام هندسة الأوامر الشامل
Python 3.13+ | PyTorch | HuggingFace Transformers
"""

from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import Counter
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ------------------------------------------------------------------
# 1) واجهة موحّدة لأي نموذج لغوي (Abstraction Layer)
# ------------------------------------------------------------------

class LLMClient(ABC):
    """واجهة موحّدة تفصل منطق هندسة الأوامر عن تفاصيل النموذج المُستخدَم."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        num_return_sequences: int = 1,
    ) -> list[str]:
        ...


class HFLocalClient(LLMClient):
    """تطبيق فعلي باستخدام نموذج محلي من HuggingFace."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        num_return_sequences: int = 1,
    ) -> list[str]:
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        do_sample = temperature > 0.0
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-4),
                top_p=0.95 if do_sample else None,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        results = []
        for out in outputs:
            generated = out[input_ids.shape[-1]:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            results.append(text.strip())
        return results


# ------------------------------------------------------------------
# 2) بنية الـ Prompt المحترف: عناصر قابلة للتركيب
# ------------------------------------------------------------------

@dataclass
class FewShotExample:
    """مثال واحد محلول يُستخدم في In-Context Learning."""
    input_text: str
    reasoning: str | None = None   # خطوات التفكير (للـ CoT)
    output_text: str = ""


@dataclass
class PromptSpec:
    """المواصفة الكاملة لأي Prompt — تفصل المحتوى عن طريقة العرض."""
    system_instruction: str
    task_description: str
    examples: list[FewShotExample] = field(default_factory=list)
    use_cot: bool = False
    output_format_hint: str | None = None


class PromptBuilder:
    """يُحوّل PromptSpec إلى نص Prompt فعلي جاهز للإرسال للنموذج."""

    COT_TRIGGER = "لنفكّر في هذا خطوة بخطوة قبل الوصول للإجابة النهائية."

    @staticmethod
    def build(spec: PromptSpec, query: str) -> str:
        parts: list[str] = []

        # 1. التعليمة النظامية
        parts.append(f"### التعليمات\n{spec.system_instruction}\n")

        # 2. وصف المهمة
        parts.append(f"### المهمة\n{spec.task_description}\n")

        # 3. الأمثلة (Few-shot)
        if spec.examples:
            parts.append("### أمثلة محلولة")
            for i, ex in enumerate(spec.examples, 1):
                parts.append(f"\nمثال {i}:")
                parts.append(f"السؤال: {ex.input_text}")
                if spec.use_cot and ex.reasoning:
                    parts.append(f"التفكير: {ex.reasoning}")
                parts.append(f"الإجابة: {ex.output_text}")
            parts.append("")

        # 4. تنسيق المخرجات المطلوب
        if spec.output_format_hint:
            parts.append(f"### تنسيق الإجابة المطلوب\n{spec.output_format_hint}\n")

        # 5. السؤال الفعلي
        parts.append("### الآن حلّ هذا السؤال")
        parts.append(f"السؤال: {query}")

        if spec.use_cot:
            parts.append(f"التفكير: {PromptBuilder.COT_TRIGGER}")
        else:
            parts.append("الإجابة:")

        return "\n".join(parts)


# ------------------------------------------------------------------
# 3) استخراج الإجابة النهائية من مخرجات CoT
# ------------------------------------------------------------------

def extract_final_answer(generated_text: str) -> str:
    """
    يبحث عن نمط 'الإجابة النهائية: X' أو آخر رقم/جملة في النص.
    هذا جزء حسّاس عمليًا — استخراج إجابة موثوق من نص حر هو تحدٍّ هندسي حقيقي.
    """
    match = re.search(r"الإجابة النهائية\s*[:：]\s*(.+)", generated_text)
    if match:
        return match.group(1).strip().split("\n")[0]

    # fallback: آخر رقم في النص (مفيد لمسائل رياضية)
    numbers = re.findall(r"-?\d+\.?\d*", generated_text)
    if numbers:
        return numbers[-1]

    return generated_text.strip().split("\n")[-1]


# ------------------------------------------------------------------
# 4) Self-Consistency: تصويت على مسارات تفكير متعددة
# ------------------------------------------------------------------

class SelfConsistencyEngine:
    """ينفّذ Self-Consistency: توليد عدة إجابات بـ Temperature > 0، ثم التصويت."""

    def __init__(self, client: LLMClient, num_samples: int = 5, temperature: float = 0.7):
        self.client = client
        self.num_samples = num_samples
        self.temperature = temperature

    def solve(self, prompt: str, max_new_tokens: int = 300) -> dict:
        raw_outputs = self.client.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=self.temperature,
            num_return_sequences=self.num_samples,
        )

        answers = [extract_final_answer(o) for o in raw_outputs]
        vote_counts = Counter(answers)
        majority_answer, majority_count = vote_counts.most_common(1)[0]

        return {
            "final_answer": majority_answer,
            "confidence": majority_count / self.num_samples,
            "all_answers": answers,
            "all_reasoning_paths": raw_outputs,
            "vote_distribution": dict(vote_counts),
        }

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-07-fine-tuning-alignment/chapter-33.md
# ────────────────────────────────────────────────────────────────────────
# ------------------------------------------------------------------
# مثال End-to-End: مسألة رياضية تتطلب استدلالًا متعدد الخطوات
# ------------------------------------------------------------------

QUESTION = (
    "اشترى أحمد 3 صناديق من التفاح، كل صندوق يحتوي على 24 تفاحة. "
    "باع ثلث الكمية الكلية في اليوم الأول، ثم باع 15 تفاحة إضافية في اليوم الثاني. "
    "كم تفاحة تبقّت لديه؟"
)

client = HFLocalClient(model_name="Qwen/Qwen2.5-1.5B-Instruct")

# --- 1) Zero-shot: بلا أمثلة وبلا تفكير صريح ---
zero_shot_spec = PromptSpec(
    system_instruction="أنت مساعد رياضي دقيق.",
    task_description="أجب عن السؤال التالي مباشرة برقم واحد فقط.",
    use_cot=False,
)
zs_prompt = PromptBuilder.build(zero_shot_spec, QUESTION)
zs_output = client.generate(zs_prompt, temperature=0.0)[0]
print("=== Zero-shot ===")
print(zs_output)
print(f"الإجابة المستخرجة: {extract_final_answer(zs_output)}\n")

# --- 2) Zero-shot CoT: فقط إضافة "لنفكر خطوة بخطوة" ---
zero_cot_spec = PromptSpec(
    system_instruction="أنت مساعد رياضي دقيق.",
    task_description="أجب عن السؤال التالي، واذكر الإجابة النهائية بصيغة: 'الإجابة النهائية: X'",
    use_cot=True,
)
zc_prompt = PromptBuilder.build(zero_cot_spec, QUESTION)
zc_output = client.generate(zc_prompt, temperature=0.0)[0]
print("=== Zero-shot CoT ===")
print(zc_output)
print(f"الإجابة المستخرجة: {extract_final_answer(zc_output)}\n")

# --- 3) Few-shot CoT: مع أمثلة محلولة توضح أسلوب التفكير المطلوب ---
few_shot_examples = [
    FewShotExample(
        input_text="لدى سارة 5 صناديق، كل صندوق فيه 10 أقلام. استخدمت 12 قلمًا. كم تبقى؟",
        reasoning=(
            "العدد الكلي = 5 × 10 = 50 قلمًا. "
            "بعد الاستخدام = 50 - 12 = 38 قلمًا."
        ),
        output_text="الإجابة النهائية: 38",
    ),
    FewShotExample(
        input_text="مصنع ينتج 200 قطعة يوميًا. باع نصف الإنتاج في يوم، ثم 30 قطعة إضافية. كم تبقى؟",
        reasoning=(
            "نصف الإنتاج المباع = 200 ÷ 2 = 100 قطعة. "
            "المتبقي بعد البيع الأول = 200 - 100 = 100. "
            "المتبقي بعد البيع الثاني = 100 - 30 = 70 قطعة."
        ),
        output_text="الإجابة النهائية: 70",
    ),
]

few_shot_cot_spec = PromptSpec(
    system_instruction="أنت مساعد رياضي دقيق يحل المسائل خطوة بخطوة.",
    task_description="اتبع أسلوب الأمثلة التالية في التفكير، ثم اذكر الإجابة النهائية بنفس الصيغة.",
    examples=few_shot_examples,
    use_cot=True,
    output_format_hint="الإجابة النهائية: <رقم فقط>",
)
fc_prompt = PromptBuilder.build(few_shot_cot_spec, QUESTION)
fc_output = client.generate(fc_prompt, temperature=0.0)[0]
print("=== Few-shot CoT ===")
print(fc_output)
print(f"الإجابة المستخرجة: {extract_final_answer(fc_output)}\n")

# --- 4) Self-Consistency: توليد 5 مسارات تفكير والتصويت ---
sc_engine = SelfConsistencyEngine(client, num_samples=5, temperature=0.7)
sc_result = sc_engine.solve(fc_prompt)
print("=== Self-Consistency (5 عيّنات) ===")
print(f"الإجابة النهائية (تصويت الأغلبية): {sc_result['final_answer']}")
print(f"مستوى الثقة: {sc_result['confidence']:.0%}")
print(f"توزيع الأصوات: {sc_result['vote_distribution']}")

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-07-fine-tuning-alignment/chapter-33.md
# ────────────────────────────────────────────────────────────────────────
# ------------------------------------------------------------------
# 5) ReAct: تفكير + فعل (استدعاء أدوات خارجية)
# ------------------------------------------------------------------

import ast
import operator


def safe_calculator(expression: str) -> str:
    """آلة حاسبة آمنة تدعم العمليات الأساسية فقط (لا eval() خطير)."""
    ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg,
    }

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](_eval(node.operand))
        raise ValueError("تعبير غير مدعوم")

    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval(tree.body))
    except Exception as e:
        return f"خطأ في الحساب: {e}"


REACT_SYSTEM_PROMPT = """أنت وكيل يحل المسائل باستخدام نمط التفكير والفعل.
في كل خطوة اكتب بالضبط أحد الأنماط التالية:

Thought: <تفكيرك في الخطوة التالية>
Action: calculator[<تعبير حسابي>]

أو إذا وصلت للإجابة النهائية:
Thought: <تفكيرك الأخير>
Final Answer: <الرقم النهائي فقط>

لا تكتب أي شيء آخر غير هذه الأنماط."""


def run_react_loop(client: LLMClient, question: str, max_steps: int = 5) -> str:
    """حلقة ReAct: فكّر → نفّذ أداة → لاحظ → كرّر حتى الوصول لإجابة نهائية."""
    transcript = f"{REACT_SYSTEM_PROMPT}\n\nQuestion: {question}\n"

    for step in range(max_steps):
        output = client.generate(transcript, max_new_tokens=150, temperature=0.0)[0]
        transcript += output + "\n"
        print(f"--- خطوة {step + 1} ---\n{output}\n")

        final_match = re.search(r"Final Answer\s*[:：]\s*(.+)", output)
        if final_match:
            return final_match.group(1).strip()

        action_match = re.search(r"Action\s*[:：]\s*calculator\[(.+?)\]", output)
        if action_match:
            expression = action_match.group(1)
            observation = safe_calculator(expression)
            transcript += f"Observation: {observation}\n"
            print(f"Observation: {observation}\n")

    return "لم يتم الوصول لإجابة نهائية ضمن الحد الأقصى من الخطوات"


# تشغيل الحلقة على نفس المسألة
final = run_react_loop(client, QUESTION)
print(f"\n🎯 الإجابة النهائية من ReAct: {final}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-33-chapter-33.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 33: الفصل 33: هندسة الأوامر (Prompt Engineering)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_prompt() -> None:
    prompt = {'role': 'assistant', 'task': 'summarize', 'constraints': ['under 30 words', 'cite supplied text'], 'input': 'Attention combines token representations.'}
    serialized = json.dumps(prompt, ensure_ascii=False, indent=2)
    parsed = json.loads(serialized)
    assert parsed['role'] == 'assistant' and len(parsed['constraints']) == 2
    print(serialized)
    print('required fields validated:', sorted(parsed))

if __name__ == "__main__":
    print("Chapter 33 lab: Build and validate a structured prompt")
    run_prompt()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-33-chapter-33.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
نظام هندسة الأوامر الشامل
Python 3.13+ | PyTorch | HuggingFace Transformers
"""

from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import Counter
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ------------------------------------------------------------------
# 1) واجهة موحّدة لأي نموذج لغوي (Abstraction Layer)
# ------------------------------------------------------------------

class LLMClient(ABC):
    """واجهة موحّدة تفصل منطق هندسة الأوامر عن تفاصيل النموذج المُستخدَم."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        num_return_sequences: int = 1,
    ) -> list[str]:
        ...


class HFLocalClient(LLMClient):
    """تطبيق فعلي باستخدام نموذج محلي من HuggingFace."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        num_return_sequences: int = 1,
    ) -> list[str]:
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        do_sample = temperature > 0.0
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-4),
                top_p=0.95 if do_sample else None,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        results = []
        for out in outputs:
            generated = out[input_ids.shape[-1]:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            results.append(text.strip())
        return results


# ------------------------------------------------------------------
# 2) بنية الـ Prompt المحترف: عناصر قابلة للتركيب
# ------------------------------------------------------------------

@dataclass
class FewShotExample:
    """مثال واحد محلول يُستخدم في In-Context Learning."""
    input_text: str
    reasoning: str | None = None   # خطوات التفكير (للـ CoT)
    output_text: str = ""


@dataclass
class PromptSpec:
    """المواصفة الكاملة لأي Prompt — تفصل المحتوى عن طريقة العرض."""
    system_instruction: str
    task_description: str
    examples: list[FewShotExample] = field(default_factory=list)
    use_cot: bool = False
    output_format_hint: str | None = None


class PromptBuilder:
    """يُحوّل PromptSpec إلى نص Prompt فعلي جاهز للإرسال للنموذج."""

    COT_TRIGGER = "لنفكّر في هذا خطوة بخطوة قبل الوصول للإجابة النهائية."

    @staticmethod
    def build(spec: PromptSpec, query: str) -> str:
        parts: list[str] = []

        # 1. التعليمة النظامية
        parts.append(f"### التعليمات\n{spec.system_instruction}\n")

        # 2. وصف المهمة
        parts.append(f"### المهمة\n{spec.task_description}\n")

        # 3. الأمثلة (Few-shot)
        if spec.examples:
            parts.append("### أمثلة محلولة")
            for i, ex in enumerate(spec.examples, 1):
                parts.append(f"\nمثال {i}:")
                parts.append(f"السؤال: {ex.input_text}")
                if spec.use_cot and ex.reasoning:
                    parts.append(f"التفكير: {ex.reasoning}")
                parts.append(f"الإجابة: {ex.output_text}")
            parts.append("")

        # 4. تنسيق المخرجات المطلوب
        if spec.output_format_hint:
            parts.append(f"### تنسيق الإجابة المطلوب\n{spec.output_format_hint}\n")

        # 5. السؤال الفعلي
        parts.append("### الآن حلّ هذا السؤال")
        parts.append(f"السؤال: {query}")

        if spec.use_cot:
            parts.append(f"التفكير: {PromptBuilder.COT_TRIGGER}")
        else:
            parts.append("الإجابة:")

        return "\n".join(parts)


# ------------------------------------------------------------------
# 3) استخراج الإجابة النهائية من مخرجات CoT
# ------------------------------------------------------------------

def extract_final_answer(generated_text: str) -> str:
    """
    يبحث عن نمط 'الإجابة النهائية: X' أو آخر رقم/جملة في النص.
    هذا جزء حسّاس عمليًا — استخراج إجابة موثوق من نص حر هو تحدٍّ هندسي حقيقي.
    """
    match = re.search(r"الإجابة النهائية\s*[:：]\s*(.+)", generated_text)
    if match:
        return match.group(1).strip().split("\n")[0]

    # fallback: آخر رقم في النص (مفيد لمسائل رياضية)
    numbers = re.findall(r"-?\d+\.?\d*", generated_text)
    if numbers:
        return numbers[-1]

    return generated_text.strip().split("\n")[-1]


# ------------------------------------------------------------------
# 4) Self-Consistency: تصويت على مسارات تفكير متعددة
# ------------------------------------------------------------------

class SelfConsistencyEngine:
    """ينفّذ Self-Consistency: توليد عدة إجابات بـ Temperature > 0، ثم التصويت."""

    def __init__(self, client: LLMClient, num_samples: int = 5, temperature: float = 0.7):
        self.client = client
        self.num_samples = num_samples
        self.temperature = temperature

    def solve(self, prompt: str, max_new_tokens: int = 300) -> dict:
        raw_outputs = self.client.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=self.temperature,
            num_return_sequences=self.num_samples,
        )

        answers = [extract_final_answer(o) for o in raw_outputs]
        vote_counts = Counter(answers)
        majority_answer, majority_count = vote_counts.most_common(1)[0]

        return {
            "final_answer": majority_answer,
            "confidence": majority_count / self.num_samples,
            "all_answers": answers,
            "all_reasoning_paths": raw_outputs,
            "vote_distribution": dict(vote_counts),
        }

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-33-chapter-33.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ------------------------------------------------------------------
# مثال End-to-End: مسألة رياضية تتطلب استدلالًا متعدد الخطوات
# ------------------------------------------------------------------

QUESTION = (
    "اشترى أحمد 3 صناديق من التفاح، كل صندوق يحتوي على 24 تفاحة. "
    "باع ثلث الكمية الكلية في اليوم الأول، ثم باع 15 تفاحة إضافية في اليوم الثاني. "
    "كم تفاحة تبقّت لديه؟"
)

client = HFLocalClient(model_name="Qwen/Qwen2.5-1.5B-Instruct")

# --- 1) Zero-shot: بلا أمثلة وبلا تفكير صريح ---
zero_shot_spec = PromptSpec(
    system_instruction="أنت مساعد رياضي دقيق.",
    task_description="أجب عن السؤال التالي مباشرة برقم واحد فقط.",
    use_cot=False,
)
zs_prompt = PromptBuilder.build(zero_shot_spec, QUESTION)
zs_output = client.generate(zs_prompt, temperature=0.0)[0]
print("=== Zero-shot ===")
print(zs_output)
print(f"الإجابة المستخرجة: {extract_final_answer(zs_output)}\n")

# --- 2) Zero-shot CoT: فقط إضافة "لنفكر خطوة بخطوة" ---
zero_cot_spec = PromptSpec(
    system_instruction="أنت مساعد رياضي دقيق.",
    task_description="أجب عن السؤال التالي، واذكر الإجابة النهائية بصيغة: 'الإجابة النهائية: X'",
    use_cot=True,
)
zc_prompt = PromptBuilder.build(zero_cot_spec, QUESTION)
zc_output = client.generate(zc_prompt, temperature=0.0)[0]
print("=== Zero-shot CoT ===")
print(zc_output)
print(f"الإجابة المستخرجة: {extract_final_answer(zc_output)}\n")

# --- 3) Few-shot CoT: مع أمثلة محلولة توضح أسلوب التفكير المطلوب ---
few_shot_examples = [
    FewShotExample(
        input_text="لدى سارة 5 صناديق، كل صندوق فيه 10 أقلام. استخدمت 12 قلمًا. كم تبقى؟",
        reasoning=(
            "العدد الكلي = 5 × 10 = 50 قلمًا. "
            "بعد الاستخدام = 50 - 12 = 38 قلمًا."
        ),
        output_text="الإجابة النهائية: 38",
    ),
    FewShotExample(
        input_text="مصنع ينتج 200 قطعة يوميًا. باع نصف الإنتاج في يوم، ثم 30 قطعة إضافية. كم تبقى؟",
        reasoning=(
            "نصف الإنتاج المباع = 200 ÷ 2 = 100 قطعة. "
            "المتبقي بعد البيع الأول = 200 - 100 = 100. "
            "المتبقي بعد البيع الثاني = 100 - 30 = 70 قطعة."
        ),
        output_text="الإجابة النهائية: 70",
    ),
]

few_shot_cot_spec = PromptSpec(
    system_instruction="أنت مساعد رياضي دقيق يحل المسائل خطوة بخطوة.",
    task_description="اتبع أسلوب الأمثلة التالية في التفكير، ثم اذكر الإجابة النهائية بنفس الصيغة.",
    examples=few_shot_examples,
    use_cot=True,
    output_format_hint="الإجابة النهائية: <رقم فقط>",
)
fc_prompt = PromptBuilder.build(few_shot_cot_spec, QUESTION)
fc_output = client.generate(fc_prompt, temperature=0.0)[0]
print("=== Few-shot CoT ===")
print(fc_output)
print(f"الإجابة المستخرجة: {extract_final_answer(fc_output)}\n")

# --- 4) Self-Consistency: توليد 5 مسارات تفكير والتصويت ---
sc_engine = SelfConsistencyEngine(client, num_samples=5, temperature=0.7)
sc_result = sc_engine.solve(fc_prompt)
print("=== Self-Consistency (5 عيّنات) ===")
print(f"الإجابة النهائية (تصويت الأغلبية): {sc_result['final_answer']}")
print(f"مستوى الثقة: {sc_result['confidence']:.0%}")
print(f"توزيع الأصوات: {sc_result['vote_distribution']}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-33-chapter-33.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ------------------------------------------------------------------
# 5) ReAct: تفكير + فعل (استدعاء أدوات خارجية)
# ------------------------------------------------------------------

import ast
import operator


def safe_calculator(expression: str) -> str:
    """آلة حاسبة آمنة تدعم العمليات الأساسية فقط (لا eval() خطير)."""
    ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg,
    }

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](_eval(node.operand))
        raise ValueError("تعبير غير مدعوم")

    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval(tree.body))
    except Exception as e:
        return f"خطأ في الحساب: {e}"


REACT_SYSTEM_PROMPT = """أنت وكيل يحل المسائل باستخدام نمط التفكير والفعل.
في كل خطوة اكتب بالضبط أحد الأنماط التالية:

Thought: <تفكيرك في الخطوة التالية>
Action: calculator[<تعبير حسابي>]

أو إذا وصلت للإجابة النهائية:
Thought: <تفكيرك الأخير>
Final Answer: <الرقم النهائي فقط>

لا تكتب أي شيء آخر غير هذه الأنماط."""


def run_react_loop(client: LLMClient, question: str, max_steps: int = 5) -> str:
    """حلقة ReAct: فكّر → نفّذ أداة → لاحظ → كرّر حتى الوصول لإجابة نهائية."""
    transcript = f"{REACT_SYSTEM_PROMPT}\n\nQuestion: {question}\n"

    for step in range(max_steps):
        output = client.generate(transcript, max_new_tokens=150, temperature=0.0)[0]
        transcript += output + "\n"
        print(f"--- خطوة {step + 1} ---\n{output}\n")

        final_match = re.search(r"Final Answer\s*[:：]\s*(.+)", output)
        if final_match:
            return final_match.group(1).strip()

        action_match = re.search(r"Action\s*[:：]\s*calculator\[(.+?)\]", output)
        if action_match:
            expression = action_match.group(1)
            observation = safe_calculator(expression)
            transcript += f"Observation: {observation}\n"
            print(f"Observation: {observation}\n")

    return "لم يتم الوصول لإجابة نهائية ضمن الحد الأقصى من الخطوات"


# تشغيل الحلقة على نفس المسألة
final = run_react_loop(client, QUESTION)
print(f"\n🎯 الإجابة النهائية من ReAct: {final}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-33-chapter-33.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — هندسة الأوامر (Prompt Engineering)
Large Language Model Engineering — Prompt

Book slug : llm-engineering-69-chapters
Chapter   : ch-33-chapter-33 (lesson 33)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_33.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 33: الفصل 33: هندسة الأوامر (Prompt Engineering)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_prompt() -> None:
    prompt = {'role': 'assistant', 'task': 'summarize', 'constraints': ['under 30 words', 'cite supplied text'], 'input': 'Attention combines token representations.'}
    serialized = json.dumps(prompt, ensure_ascii=False, indent=2)
    parsed = json.loads(serialized)
    assert parsed['role'] == 'assistant' and len(parsed['constraints']) == 2
    print(serialized)
    print('required fields validated:', sorted(parsed))

if __name__ == "__main__":
    print("Chapter 33 lab: Build and validate a structured prompt")
    run_prompt()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-07-fine-tuning-alignment/chapter-33.md
# ────────────────────────────────────────────────────────────────────────
"""
نظام هندسة الأوامر الشامل
Python 3.13+ | PyTorch | HuggingFace Transformers
"""

from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import Counter
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ------------------------------------------------------------------
# 1) واجهة موحّدة لأي نموذج لغوي (Abstraction Layer)
# ------------------------------------------------------------------

class LLMClient(ABC):
    """واجهة موحّدة تفصل منطق هندسة الأوامر عن تفاصيل النموذج المُستخدَم."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        num_return_sequences: int = 1,
    ) -> list[str]:
        ...


class HFLocalClient(LLMClient):
    """تطبيق فعلي باستخدام نموذج محلي من HuggingFace."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        num_return_sequences: int = 1,
    ) -> list[str]:
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        do_sample = temperature > 0.0
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-4),
                top_p=0.95 if do_sample else None,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        results = []
        for out in outputs:
            generated = out[input_ids.shape[-1]:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            results.append(text.strip())
        return results


# ------------------------------------------------------------------
# 2) بنية الـ Prompt المحترف: عناصر قابلة للتركيب
# ------------------------------------------------------------------

@dataclass
class FewShotExample:
    """مثال واحد محلول يُستخدم في In-Context Learning."""
    input_text: str
    reasoning: str | None = None   # خطوات التفكير (للـ CoT)
    output_text: str = ""


@dataclass
class PromptSpec:
    """المواصفة الكاملة لأي Prompt — تفصل المحتوى عن طريقة العرض."""
    system_instruction: str
    task_description: str
    examples: list[FewShotExample] = field(default_factory=list)
    use_cot: bool = False
    output_format_hint: str | None = None


class PromptBuilder:
    """يُحوّل PromptSpec إلى نص Prompt فعلي جاهز للإرسال للنموذج."""

    COT_TRIGGER = "لنفكّر في هذا خطوة بخطوة قبل الوصول للإجابة النهائية."

    @staticmethod
    def build(spec: PromptSpec, query: str) -> str:
        parts: list[str] = []

        # 1. التعليمة النظامية
        parts.append(f"### التعليمات\n{spec.system_instruction}\n")

        # 2. وصف المهمة
        parts.append(f"### المهمة\n{spec.task_description}\n")

        # 3. الأمثلة (Few-shot)
        if spec.examples:
            parts.append("### أمثلة محلولة")
            for i, ex in enumerate(spec.examples, 1):
                parts.append(f"\nمثال {i}:")
                parts.append(f"السؤال: {ex.input_text}")
                if spec.use_cot and ex.reasoning:
                    parts.append(f"التفكير: {ex.reasoning}")
                parts.append(f"الإجابة: {ex.output_text}")
            parts.append("")

        # 4. تنسيق المخرجات المطلوب
        if spec.output_format_hint:
            parts.append(f"### تنسيق الإجابة المطلوب\n{spec.output_format_hint}\n")

        # 5. السؤال الفعلي
        parts.append("### الآن حلّ هذا السؤال")
        parts.append(f"السؤال: {query}")

        if spec.use_cot:
            parts.append(f"التفكير: {PromptBuilder.COT_TRIGGER}")
        else:
            parts.append("الإجابة:")

        return "\n".join(parts)


# ------------------------------------------------------------------
# 3) استخراج الإجابة النهائية من مخرجات CoT
# ------------------------------------------------------------------

def extract_final_answer(generated_text: str) -> str:
    """
    يبحث عن نمط 'الإجابة النهائية: X' أو آخر رقم/جملة في النص.
    هذا جزء حسّاس عمليًا — استخراج إجابة موثوق من نص حر هو تحدٍّ هندسي حقيقي.
    """
    match = re.search(r"الإجابة النهائية\s*[:：]\s*(.+)", generated_text)
    if match:
        return match.group(1).strip().split("\n")[0]

    # fallback: آخر رقم في النص (مفيد لمسائل رياضية)
    numbers = re.findall(r"-?\d+\.?\d*", generated_text)
    if numbers:
        return numbers[-1]

    return generated_text.strip().split("\n")[-1]


# ------------------------------------------------------------------
# 4) Self-Consistency: تصويت على مسارات تفكير متعددة
# ------------------------------------------------------------------

class SelfConsistencyEngine:
    """ينفّذ Self-Consistency: توليد عدة إجابات بـ Temperature > 0، ثم التصويت."""

    def __init__(self, client: LLMClient, num_samples: int = 5, temperature: float = 0.7):
        self.client = client
        self.num_samples = num_samples
        self.temperature = temperature

    def solve(self, prompt: str, max_new_tokens: int = 300) -> dict:
        raw_outputs = self.client.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=self.temperature,
            num_return_sequences=self.num_samples,
        )

        answers = [extract_final_answer(o) for o in raw_outputs]
        vote_counts = Counter(answers)
        majority_answer, majority_count = vote_counts.most_common(1)[0]

        return {
            "final_answer": majority_answer,
            "confidence": majority_count / self.num_samples,
            "all_answers": answers,
            "all_reasoning_paths": raw_outputs,
            "vote_distribution": dict(vote_counts),
        }

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-07-fine-tuning-alignment/chapter-33.md
# ────────────────────────────────────────────────────────────────────────
# ------------------------------------------------------------------
# مثال End-to-End: مسألة رياضية تتطلب استدلالًا متعدد الخطوات
# ------------------------------------------------------------------

QUESTION = (
    "اشترى أحمد 3 صناديق من التفاح، كل صندوق يحتوي على 24 تفاحة. "
    "باع ثلث الكمية الكلية في اليوم الأول، ثم باع 15 تفاحة إضافية في اليوم الثاني. "
    "كم تفاحة تبقّت لديه؟"
)

client = HFLocalClient(model_name="Qwen/Qwen2.5-1.5B-Instruct")

# --- 1) Zero-shot: بلا أمثلة وبلا تفكير صريح ---
zero_shot_spec = PromptSpec(
    system_instruction="أنت مساعد رياضي دقيق.",
    task_description="أجب عن السؤال التالي مباشرة برقم واحد فقط.",
    use_cot=False,
)
zs_prompt = PromptBuilder.build(zero_shot_spec, QUESTION)
zs_output = client.generate(zs_prompt, temperature=0.0)[0]
print("=== Zero-shot ===")
print(zs_output)
print(f"الإجابة المستخرجة: {extract_final_answer(zs_output)}\n")

# --- 2) Zero-shot CoT: فقط إضافة "لنفكر خطوة بخطوة" ---
zero_cot_spec = PromptSpec(
    system_instruction="أنت مساعد رياضي دقيق.",
    task_description="أجب عن السؤال التالي، واذكر الإجابة النهائية بصيغة: 'الإجابة النهائية: X'",
    use_cot=True,
)
zc_prompt = PromptBuilder.build(zero_cot_spec, QUESTION)
zc_output = client.generate(zc_prompt, temperature=0.0)[0]
print("=== Zero-shot CoT ===")
print(zc_output)
print(f"الإجابة المستخرجة: {extract_final_answer(zc_output)}\n")

# --- 3) Few-shot CoT: مع أمثلة محلولة توضح أسلوب التفكير المطلوب ---
few_shot_examples = [
    FewShotExample(
        input_text="لدى سارة 5 صناديق، كل صندوق فيه 10 أقلام. استخدمت 12 قلمًا. كم تبقى؟",
        reasoning=(
            "العدد الكلي = 5 × 10 = 50 قلمًا. "
            "بعد الاستخدام = 50 - 12 = 38 قلمًا."
        ),
        output_text="الإجابة النهائية: 38",
    ),
    FewShotExample(
        input_text="مصنع ينتج 200 قطعة يوميًا. باع نصف الإنتاج في يوم، ثم 30 قطعة إضافية. كم تبقى؟",
        reasoning=(
            "نصف الإنتاج المباع = 200 ÷ 2 = 100 قطعة. "
            "المتبقي بعد البيع الأول = 200 - 100 = 100. "
            "المتبقي بعد البيع الثاني = 100 - 30 = 70 قطعة."
        ),
        output_text="الإجابة النهائية: 70",
    ),
]

few_shot_cot_spec = PromptSpec(
    system_instruction="أنت مساعد رياضي دقيق يحل المسائل خطوة بخطوة.",
    task_description="اتبع أسلوب الأمثلة التالية في التفكير، ثم اذكر الإجابة النهائية بنفس الصيغة.",
    examples=few_shot_examples,
    use_cot=True,
    output_format_hint="الإجابة النهائية: <رقم فقط>",
)
fc_prompt = PromptBuilder.build(few_shot_cot_spec, QUESTION)
fc_output = client.generate(fc_prompt, temperature=0.0)[0]
print("=== Few-shot CoT ===")
print(fc_output)
print(f"الإجابة المستخرجة: {extract_final_answer(fc_output)}\n")

# --- 4) Self-Consistency: توليد 5 مسارات تفكير والتصويت ---
sc_engine = SelfConsistencyEngine(client, num_samples=5, temperature=0.7)
sc_result = sc_engine.solve(fc_prompt)
print("=== Self-Consistency (5 عيّنات) ===")
print(f"الإجابة النهائية (تصويت الأغلبية): {sc_result['final_answer']}")
print(f"مستوى الثقة: {sc_result['confidence']:.0%}")
print(f"توزيع الأصوات: {sc_result['vote_distribution']}")

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-07-fine-tuning-alignment/chapter-33.md
# ────────────────────────────────────────────────────────────────────────
# ------------------------------------------------------------------
# 5) ReAct: تفكير + فعل (استدعاء أدوات خارجية)
# ------------------------------------------------------------------

import ast
import operator


def safe_calculator(expression: str) -> str:
    """آلة حاسبة آمنة تدعم العمليات الأساسية فقط (لا eval() خطير)."""
    ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg,
    }

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](_eval(node.operand))
        raise ValueError("تعبير غير مدعوم")

    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval(tree.body))
    except Exception as e:
        return f"خطأ في الحساب: {e}"


REACT_SYSTEM_PROMPT = """أنت وكيل يحل المسائل باستخدام نمط التفكير والفعل.
في كل خطوة اكتب بالضبط أحد الأنماط التالية:

Thought: <تفكيرك في الخطوة التالية>
Action: calculator[<تعبير حسابي>]

أو إذا وصلت للإجابة النهائية:
Thought: <تفكيرك الأخير>
Final Answer: <الرقم النهائي فقط>

لا تكتب أي شيء آخر غير هذه الأنماط."""


def run_react_loop(client: LLMClient, question: str, max_steps: int = 5) -> str:
    """حلقة ReAct: فكّر → نفّذ أداة → لاحظ → كرّر حتى الوصول لإجابة نهائية."""
    transcript = f"{REACT_SYSTEM_PROMPT}\n\nQuestion: {question}\n"

    for step in range(max_steps):
        output = client.generate(transcript, max_new_tokens=150, temperature=0.0)[0]
        transcript += output + "\n"
        print(f"--- خطوة {step + 1} ---\n{output}\n")

        final_match = re.search(r"Final Answer\s*[:：]\s*(.+)", output)
        if final_match:
            return final_match.group(1).strip()

        action_match = re.search(r"Action\s*[:：]\s*calculator\[(.+?)\]", output)
        if action_match:
            expression = action_match.group(1)
            observation = safe_calculator(expression)
            transcript += f"Observation: {observation}\n"
            print(f"Observation: {observation}\n")

    return "لم يتم الوصول لإجابة نهائية ضمن الحد الأقصى من الخطوات"


# تشغيل الحلقة على نفس المسألة
final = run_react_loop(client, QUESTION)
print(f"\n🎯 الإجابة النهائية من ReAct: {final}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-33-chapter-33.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 33: الفصل 33: هندسة الأوامر (Prompt Engineering)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_prompt() -> None:
    prompt = {'role': 'assistant', 'task': 'summarize', 'constraints': ['under 30 words', 'cite supplied text'], 'input': 'Attention combines token representations.'}
    serialized = json.dumps(prompt, ensure_ascii=False, indent=2)
    parsed = json.loads(serialized)
    assert parsed['role'] == 'assistant' and len(parsed['constraints']) == 2
    print(serialized)
    print('required fields validated:', sorted(parsed))

if __name__ == "__main__":
    print("Chapter 33 lab: Build and validate a structured prompt")
    run_prompt()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-33-chapter-33.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
نظام هندسة الأوامر الشامل
Python 3.13+ | PyTorch | HuggingFace Transformers
"""

from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import Counter
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ------------------------------------------------------------------
# 1) واجهة موحّدة لأي نموذج لغوي (Abstraction Layer)
# ------------------------------------------------------------------

class LLMClient(ABC):
    """واجهة موحّدة تفصل منطق هندسة الأوامر عن تفاصيل النموذج المُستخدَم."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        num_return_sequences: int = 1,
    ) -> list[str]:
        ...


class HFLocalClient(LLMClient):
    """تطبيق فعلي باستخدام نموذج محلي من HuggingFace."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        num_return_sequences: int = 1,
    ) -> list[str]:
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        do_sample = temperature > 0.0
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-4),
                top_p=0.95 if do_sample else None,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        results = []
        for out in outputs:
            generated = out[input_ids.shape[-1]:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            results.append(text.strip())
        return results


# ------------------------------------------------------------------
# 2) بنية الـ Prompt المحترف: عناصر قابلة للتركيب
# ------------------------------------------------------------------

@dataclass
class FewShotExample:
    """مثال واحد محلول يُستخدم في In-Context Learning."""
    input_text: str
    reasoning: str | None = None   # خطوات التفكير (للـ CoT)
    output_text: str = ""


@dataclass
class PromptSpec:
    """المواصفة الكاملة لأي Prompt — تفصل المحتوى عن طريقة العرض."""
    system_instruction: str
    task_description: str
    examples: list[FewShotExample] = field(default_factory=list)
    use_cot: bool = False
    output_format_hint: str | None = None


class PromptBuilder:
    """يُحوّل PromptSpec إلى نص Prompt فعلي جاهز للإرسال للنموذج."""

    COT_TRIGGER = "لنفكّر في هذا خطوة بخطوة قبل الوصول للإجابة النهائية."

    @staticmethod
    def build(spec: PromptSpec, query: str) -> str:
        parts: list[str] = []

        # 1. التعليمة النظامية
        parts.append(f"### التعليمات\n{spec.system_instruction}\n")

        # 2. وصف المهمة
        parts.append(f"### المهمة\n{spec.task_description}\n")

        # 3. الأمثلة (Few-shot)
        if spec.examples:
            parts.append("### أمثلة محلولة")
            for i, ex in enumerate(spec.examples, 1):
                parts.append(f"\nمثال {i}:")
                parts.append(f"السؤال: {ex.input_text}")
                if spec.use_cot and ex.reasoning:
                    parts.append(f"التفكير: {ex.reasoning}")
                parts.append(f"الإجابة: {ex.output_text}")
            parts.append("")

        # 4. تنسيق المخرجات المطلوب
        if spec.output_format_hint:
            parts.append(f"### تنسيق الإجابة المطلوب\n{spec.output_format_hint}\n")

        # 5. السؤال الفعلي
        parts.append("### الآن حلّ هذا السؤال")
        parts.append(f"السؤال: {query}")

        if spec.use_cot:
            parts.append(f"التفكير: {PromptBuilder.COT_TRIGGER}")
        else:
            parts.append("الإجابة:")

        return "\n".join(parts)


# ------------------------------------------------------------------
# 3) استخراج الإجابة النهائية من مخرجات CoT
# ------------------------------------------------------------------

def extract_final_answer(generated_text: str) -> str:
    """
    يبحث عن نمط 'الإجابة النهائية: X' أو آخر رقم/جملة في النص.
    هذا جزء حسّاس عمليًا — استخراج إجابة موثوق من نص حر هو تحدٍّ هندسي حقيقي.
    """
    match = re.search(r"الإجابة النهائية\s*[:：]\s*(.+)", generated_text)
    if match:
        return match.group(1).strip().split("\n")[0]

    # fallback: آخر رقم في النص (مفيد لمسائل رياضية)
    numbers = re.findall(r"-?\d+\.?\d*", generated_text)
    if numbers:
        return numbers[-1]

    return generated_text.strip().split("\n")[-1]


# ------------------------------------------------------------------
# 4) Self-Consistency: تصويت على مسارات تفكير متعددة
# ------------------------------------------------------------------

class SelfConsistencyEngine:
    """ينفّذ Self-Consistency: توليد عدة إجابات بـ Temperature > 0، ثم التصويت."""

    def __init__(self, client: LLMClient, num_samples: int = 5, temperature: float = 0.7):
        self.client = client
        self.num_samples = num_samples
        self.temperature = temperature

    def solve(self, prompt: str, max_new_tokens: int = 300) -> dict:
        raw_outputs = self.client.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=self.temperature,
            num_return_sequences=self.num_samples,
        )

        answers = [extract_final_answer(o) for o in raw_outputs]
        vote_counts = Counter(answers)
        majority_answer, majority_count = vote_counts.most_common(1)[0]

        return {
            "final_answer": majority_answer,
            "confidence": majority_count / self.num_samples,
            "all_answers": answers,
            "all_reasoning_paths": raw_outputs,
            "vote_distribution": dict(vote_counts),
        }

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-33-chapter-33.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ------------------------------------------------------------------
# مثال End-to-End: مسألة رياضية تتطلب استدلالًا متعدد الخطوات
# ------------------------------------------------------------------

QUESTION = (
    "اشترى أحمد 3 صناديق من التفاح، كل صندوق يحتوي على 24 تفاحة. "
    "باع ثلث الكمية الكلية في اليوم الأول، ثم باع 15 تفاحة إضافية في اليوم الثاني. "
    "كم تفاحة تبقّت لديه؟"
)

client = HFLocalClient(model_name="Qwen/Qwen2.5-1.5B-Instruct")

# --- 1) Zero-shot: بلا أمثلة وبلا تفكير صريح ---
zero_shot_spec = PromptSpec(
    system_instruction="أنت مساعد رياضي دقيق.",
    task_description="أجب عن السؤال التالي مباشرة برقم واحد فقط.",
    use_cot=False,
)
zs_prompt = PromptBuilder.build(zero_shot_spec, QUESTION)
zs_output = client.generate(zs_prompt, temperature=0.0)[0]
print("=== Zero-shot ===")
print(zs_output)
print(f"الإجابة المستخرجة: {extract_final_answer(zs_output)}\n")

# --- 2) Zero-shot CoT: فقط إضافة "لنفكر خطوة بخطوة" ---
zero_cot_spec = PromptSpec(
    system_instruction="أنت مساعد رياضي دقيق.",
    task_description="أجب عن السؤال التالي، واذكر الإجابة النهائية بصيغة: 'الإجابة النهائية: X'",
    use_cot=True,
)
zc_prompt = PromptBuilder.build(zero_cot_spec, QUESTION)
zc_output = client.generate(zc_prompt, temperature=0.0)[0]
print("=== Zero-shot CoT ===")
print(zc_output)
print(f"الإجابة المستخرجة: {extract_final_answer(zc_output)}\n")

# --- 3) Few-shot CoT: مع أمثلة محلولة توضح أسلوب التفكير المطلوب ---
few_shot_examples = [
    FewShotExample(
        input_text="لدى سارة 5 صناديق، كل صندوق فيه 10 أقلام. استخدمت 12 قلمًا. كم تبقى؟",
        reasoning=(
            "العدد الكلي = 5 × 10 = 50 قلمًا. "
            "بعد الاستخدام = 50 - 12 = 38 قلمًا."
        ),
        output_text="الإجابة النهائية: 38",
    ),
    FewShotExample(
        input_text="مصنع ينتج 200 قطعة يوميًا. باع نصف الإنتاج في يوم، ثم 30 قطعة إضافية. كم تبقى؟",
        reasoning=(
            "نصف الإنتاج المباع = 200 ÷ 2 = 100 قطعة. "
            "المتبقي بعد البيع الأول = 200 - 100 = 100. "
            "المتبقي بعد البيع الثاني = 100 - 30 = 70 قطعة."
        ),
        output_text="الإجابة النهائية: 70",
    ),
]

few_shot_cot_spec = PromptSpec(
    system_instruction="أنت مساعد رياضي دقيق يحل المسائل خطوة بخطوة.",
    task_description="اتبع أسلوب الأمثلة التالية في التفكير، ثم اذكر الإجابة النهائية بنفس الصيغة.",
    examples=few_shot_examples,
    use_cot=True,
    output_format_hint="الإجابة النهائية: <رقم فقط>",
)
fc_prompt = PromptBuilder.build(few_shot_cot_spec, QUESTION)
fc_output = client.generate(fc_prompt, temperature=0.0)[0]
print("=== Few-shot CoT ===")
print(fc_output)
print(f"الإجابة المستخرجة: {extract_final_answer(fc_output)}\n")

# --- 4) Self-Consistency: توليد 5 مسارات تفكير والتصويت ---
sc_engine = SelfConsistencyEngine(client, num_samples=5, temperature=0.7)
sc_result = sc_engine.solve(fc_prompt)
print("=== Self-Consistency (5 عيّنات) ===")
print(f"الإجابة النهائية (تصويت الأغلبية): {sc_result['final_answer']}")
print(f"مستوى الثقة: {sc_result['confidence']:.0%}")
print(f"توزيع الأصوات: {sc_result['vote_distribution']}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-33-chapter-33.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ------------------------------------------------------------------
# 5) ReAct: تفكير + فعل (استدعاء أدوات خارجية)
# ------------------------------------------------------------------

import ast
import operator


def safe_calculator(expression: str) -> str:
    """آلة حاسبة آمنة تدعم العمليات الأساسية فقط (لا eval() خطير)."""
    ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg,
    }

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](_eval(node.operand))
        raise ValueError("تعبير غير مدعوم")

    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval(tree.body))
    except Exception as e:
        return f"خطأ في الحساب: {e}"


REACT_SYSTEM_PROMPT = """أنت وكيل يحل المسائل باستخدام نمط التفكير والفعل.
في كل خطوة اكتب بالضبط أحد الأنماط التالية:

Thought: <تفكيرك في الخطوة التالية>
Action: calculator[<تعبير حسابي>]

أو إذا وصلت للإجابة النهائية:
Thought: <تفكيرك الأخير>
Final Answer: <الرقم النهائي فقط>

لا تكتب أي شيء آخر غير هذه الأنماط."""


def run_react_loop(client: LLMClient, question: str, max_steps: int = 5) -> str:
    """حلقة ReAct: فكّر → نفّذ أداة → لاحظ → كرّر حتى الوصول لإجابة نهائية."""
    transcript = f"{REACT_SYSTEM_PROMPT}\n\nQuestion: {question}\n"

    for step in range(max_steps):
        output = client.generate(transcript, max_new_tokens=150, temperature=0.0)[0]
        transcript += output + "\n"
        print(f"--- خطوة {step + 1} ---\n{output}\n")

        final_match = re.search(r"Final Answer\s*[:：]\s*(.+)", output)
        if final_match:
            return final_match.group(1).strip()

        action_match = re.search(r"Action\s*[:：]\s*calculator\[(.+?)\]", output)
        if action_match:
            expression = action_match.group(1)
            observation = safe_calculator(expression)
            transcript += f"Observation: {observation}\n"
            print(f"Observation: {observation}\n")

    return "لم يتم الوصول لإجابة نهائية ضمن الحد الأقصى من الخطوات"


# تشغيل الحلقة على نفس المسألة
final = run_react_loop(client, QUESTION)
print(f"\n🎯 الإجابة النهائية من ReAct: {final}")
