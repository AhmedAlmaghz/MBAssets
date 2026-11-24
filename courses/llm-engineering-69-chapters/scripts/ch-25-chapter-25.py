#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — من Base إلى نموذج قابل للاستخدام — Supervised Fine-Tuning والبيانات الاصطناعية
Large Language Model Engineering — Base

Book slug : llm-engineering-69-chapters
Chapter   : ch-25-chapter-25 (lesson 25)
Code files: 13 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-25-chapter-25.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — من Base إلى نموذج قابل للاستخدام — Supervised Fine-Tuning والبيانات الاصطناعية
Large Language Model Engineering — Base

Book slug : llm-engineering-69-chapters
Chapter   : ch-25-chapter-25 (lesson 25)
Code files: 12 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-25-chapter-25.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — من Base إلى نموذج قابل للاستخدام — Supervised Fine-Tuning والبيانات الاصطناعية
Large Language Model Engineering — Base

Book slug : llm-engineering-69-chapters
Chapter   : ch-25-chapter-25 (lesson 25)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_25.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 25: الفصل 25: من Base إلى نموذج قابل للاستخدام — Supervised Fine-Tuning والبيانات الاصطناعية."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_sft() -> None:
    prompt, answer = ('User: define attention\nAssistant:', ' attention weights combine context')
    tokens = (prompt + answer).split()
    labels = [-100] * len(prompt.split()) + list(range(len(prompt.split()), len(tokens)))
    print('tokens:', tokens)
    print('loss-masked labels:', labels)
    print('supervised completion tokens:', sum((label != -100 for label in labels)))

if __name__ == "__main__":
    print("Chapter 25 lab: Mask prompt tokens in supervised fine-tuning")
    run_sft()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-05-build-an-llm/chapter-25.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChatMessage:
    role: str   # "system" | "user" | "assistant"
    content: str


@dataclass
class ChatTemplate:
    """تطبيق مبسّط لتنسيق على نمط ChatML، قابل للتخصيص لتنسيقات أخرى
    (Llama-3, Mistral, Alpaca) بتغيير الرموز الخاصة فقط."""

    bos_token: str = "<|im_start|>"
    eos_token: str = "<|im_end|>"

    def format_conversation(self, messages: list[ChatMessage]) -> str:
        formatted_parts: list[str] = []
        for msg in messages:
            formatted_parts.append(
                f"{self.bos_token}{msg.role}\n{msg.content}{self.eos_token}\n"
            )
        return "".join(formatted_parts)

    def format_for_training(
        self, messages: list[ChatMessage]
    ) -> tuple[str, list[tuple[int, int]]]:
        """يُرجع النص الكامل بالإضافة إلى نطاقات (start, end) بالأحرف
        لكل جزء "assistant" فقط — ضروري لاحقًا لبناء قناع الخسارة (Loss Mask)."""
        full_text = ""
        assistant_spans: list[tuple[int, int]] = []

        for msg in messages:
            prefix = f"{self.bos_token}{msg.role}\n"
            content_start = len(full_text) + len(prefix)
            segment = f"{prefix}{msg.content}{self.eos_token}\n"

            if msg.role == "assistant":
                content_end = content_start + len(msg.content) + len(self.eos_token)
                assistant_spans.append((content_start, content_end))

            full_text += segment

        return full_text, assistant_spans

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-05-build-an-llm/chapter-25.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase


def build_training_example(
    messages: list[ChatMessage],
    tokenizer: PreTrainedTokenizerBase,
    template: ChatTemplate,
    max_length: int = 2048,
) -> dict[str, torch.Tensor]:
    """بناء مثال تدريب كامل: input_ids + labels مع قناع الخسارة الصحيح
    على كل شيء عدا رموز استجابة المساعد."""

    full_text, assistant_char_spans = template.format_for_training(messages)

    encoding = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        return_offsets_mapping=True,  # ضروري لربط كل رمز بموضعه الحرفي الأصلي
    )

    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    # نبدأ بتقنيع كل شيء (-100)، ثم "نفتح" فقط رموز استجابة المساعد
    labels = [-100] * len(input_ids)

    for token_idx, (char_start, char_end) in enumerate(offsets):
        if char_start == char_end:  # رمز خاص بلا نطاق حرفي حقيقي
            continue
        for span_start, span_end in assistant_char_spans:
            if char_start >= span_start and char_end <= span_end:
                labels[token_idx] = input_ids[token_idx]
                break

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-05-build-an-llm/chapter-25.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import difflib
import random
from dataclasses import dataclass


@dataclass
class InstructionExample:
    instruction: str
    input_text: str = ""
    output: str = ""


class SelfInstructPipeline:
    """تطبيق مبسّط تعليمي لخوارزمية Self-Instruct — في الإنتاج الفعلي
    تُستبدل دالة التوليد بنداء حقيقي لـ API نموذج معلّم قوي (GPT-4/Claude)."""

    def __init__(
        self,
        seed_tasks: list[InstructionExample],
        similarity_threshold: float = 0.7,
        num_few_shot_examples: int = 6,
    ) -> None:
        self.task_pool: list[InstructionExample] = list(seed_tasks)
        self.similarity_threshold = similarity_threshold
        self.num_few_shot_examples = num_few_shot_examples
        self._rng = random.Random(42)

    def _build_few_shot_prompt(self) -> str:
        """بناء Prompt يحتوي عينة عشوائية من المهام الموجودة كأمثلة توجيهية
        للنموذج المعلّم، لتوليد تعليمة جديدة مشابهة في الأسلوب والتنوع."""
        samples = self._rng.sample(
            self.task_pool, min(self.num_few_shot_examples, len(self.task_pool))
        )
        prompt_lines = ["وَلِّد تعليمة جديدة متنوعة على نمط الأمثلة التالية:\n"]
        for i, ex in enumerate(samples, start=1):
            prompt_lines.append(f"{i}. {ex.instruction}")
        prompt_lines.append(f"{len(samples) + 1}.")
        return "\n".join(prompt_lines)

    def _is_too_similar(self, new_instruction: str) -> bool:
        """فحص تشابه بسيط (ROUGE-L مبسّط عبر SequenceMatcher) لمنع
        تكرار تعليمات شبه متطابقة في مجموعة البيانات النهائية."""
        for existing in self.task_pool:
            similarity = difflib.SequenceMatcher(
                None, new_instruction.lower(), existing.instruction.lower()
            ).ratio()
            if similarity > self.similarity_threshold:
                return True
        return False

    def generate_new_instruction(self, teacher_model_call: callable) -> str | None:
        """teacher_model_call: دالة تستدعي نموذج معلّم حقيقي (API) وتُرجع نصًا.
        هنا نُحاكي واجهتها فقط لأغراض توضيحية."""
        prompt = self._build_few_shot_prompt()
        candidate_instruction = teacher_model_call(prompt).strip()

        if self._is_too_similar(candidate_instruction):
            return None  # رُفض بسبب التشابه الزائد — دورة أخرى مطلوبة

        return candidate_instruction

    def generate_response(
        self, instruction: str, teacher_model_call: callable
    ) -> str:
        response_prompt = f"أجب على التعليمة التالية بدقة ووضوح:\n{instruction}"
        return teacher_model_call(response_prompt).strip()

    def run_bootstrap_round(
        self, teacher_model_call: callable, num_new_examples: int = 50
    ) -> list[InstructionExample]:
        new_examples: list[InstructionExample] = []
        attempts = 0
        max_attempts = num_new_examples * 3  # هامش أمان لمحاولات التشابه الفاشلة

        while len(new_examples) < num_new_examples and attempts < max_attempts:
            attempts += 1
            instruction = self.generate_new_instruction(teacher_model_call)
            if instruction is None:
                continue

            output = self.generate_response(instruction, teacher_model_call)
            example = InstructionExample(instruction=instruction, output=output)
            new_examples.append(example)
            self.task_pool.append(example)  # يُغذّي الجولات التالية بمزيد من التنوع

        return new_examples

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-05-build-an-llm/chapter-25.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    """مجموعة بيانات SFT بسيطة تطبّق Padding بدلاً من Packing —
    أبسط للفهم والتصحيح، وإن كانت أقل كفاءة حسابية قليلاً من Packing
    المتقدّم الذي يتطلب قناع انتباه حدودي (Block-Diagonal Attention Mask)."""

    def __init__(
        self,
        examples: list[list[ChatMessage]],
        tokenizer,
        template: ChatTemplate,
        max_length: int = 2048,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.template = template
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        messages = self.examples[idx]
        return build_training_example(
            messages, self.tokenizer, self.template, self.max_length
        )


def sft_collate_fn(
    batch: list[dict[str, torch.Tensor]], pad_token_id: int
) -> dict[str, torch.Tensor]:
    """تجميع دفعة مع Padding ديناميكي (إلى أطول عيّنة في الدفعة فقط،
    وليس إلى max_length الثابت) لتوفير حوسبة غير ضرورية."""
    max_len_in_batch = max(item["input_ids"].size(0) for item in batch)

    input_ids_batch = []
    labels_batch = []
    attention_mask_batch = []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len_in_batch - seq_len

        input_ids_batch.append(
            torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id)])
        )
        # -100 دائمًا للحشو — لا نريد حساب خسارة على رموز Padding أبدًا
        labels_batch.append(
            torch.cat([item["labels"], torch.full((pad_len,), -100)])
        )
        attention_mask_batch.append(
            torch.cat([torch.ones(seq_len), torch.zeros(pad_len)])
        )

    return {
        "input_ids": torch.stack(input_ids_batch).long(),
        "labels": torch.stack(labels_batch).long(),
        "attention_mask": torch.stack(attention_mask_batch).long(),
    }


def sft_training_step(
    model, batch: dict[str, torch.Tensor], optimizer: torch.optim.Optimizer
) -> float:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],  # معظم مكتبات HF تحسب الخسارة المُقنَّعة داخليًا تلقائيًا
    )
    loss = outputs.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.item()

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-05-build-an-llm/chapter-25.md
# ────────────────────────────────────────────────────────────────────────
messages = [
    ChatMessage(role="system", content="أنت مساعد مفيد."),
    ChatMessage(role="user", content="ما نتيجة 2+2؟"),
    ChatMessage(role="assistant", content="نتيجة 2+2 هي 4."),
]

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-05-build-an-llm/chapter-25.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: النموذج "يتعلّم" أيضًا كيف يُعيد كتابة أسئلة المستخدمين
labels = input_ids.clone()  # بلا أي تقنيع!
loss = cross_entropy(logits, labels)

# ✅ صحيح: قناع صريح يحصر الخسارة على استجابة المساعد فقط
labels = build_masked_labels(input_ids, assistant_spans)
loss = cross_entropy(logits, labels, ignore_index=-100)

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-05-build-an-llm/chapter-25.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: معدل تعلّم عالٍ جدًا لـ SFT يمحو جزءًا كبيرًا من المعرفة
# المُكتسبة في Pretraining (ظاهرة تُسمى "Catastrophic Forgetting")
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)  # مناسب لـ Pretraining فقط

# ✅ صحيح: معدل تعلّم أقل بعشرة أضعاف تقريبًا لـ SFT
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# ────────────────────────────────────────────────────────────────────────
# [009] book/chapters/part-05-build-an-llm/chapter-25.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: 10 حقبات على 50,000 مثال فقط يؤدي لـ Overfitting حاد،
# والنموذج يبدأ "بحفظ" الاستجابات بدلاً من تعميم السلوك
for epoch in range(10):
    train_one_epoch(model, sft_dataloader)

# ✅ صحيح: 2-3 حقبات عادة كافية تمامًا، مع مراقبة Validation Loss
# للتوقف المبكر (Early Stopping) إذا بدأ بالتدهور
for epoch in range(3):
    train_one_epoch(model, sft_dataloader)
    val_loss = evaluate(model, val_dataloader)
    if val_loss > best_val_loss:
        break  # إشارة Overfitting مبكرة

# ────────────────────────────────────────────────────────────────────────
# [010] book/chapters/part-05-build-an-llm/chapter-25.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطير: دمج عدة محادثات في تسلسل واحد لتحسين الكفاءة، لكن بدون منع
# الانتباه عبر حدود المحادثات — النموذج "يرى" محادثة أخرى غير متعلقة!
packed_sequence = concat([conv1_tokens, conv2_tokens, conv3_tokens])
# بدون قناع خاص، Self-Attention يسمح لرموز conv2 برؤية رموز conv1 بالكامل

# ✅ صحيح: استخدام قناع انتباه Block-Diagonal يعزل كل محادثة عن الأخرى
# (مدعوم أصلاً في تطبيقات حديثة مثل flash_attn_varlen_func)
attention_mask = build_block_diagonal_mask(sequence_boundaries)

# ────────────────────────────────────────────────────────────────────────
# [011] book/chapters/part-05-build-an-llm/chapter-25.md
# ────────────────────────────────────────────────────────────────────────
"""
مشروع تطبيقي: خط أنابيب SFT كامل من البداية للنهاية

المطلوب إكماله:
1. استخدم نموذجًا صغيرًا مُدرَّبًا مسبقًا (مثل "gpt2" أو "distilgpt2" من Hugging Face)
   كـ"نموذج أساسي" توضيحي (بالطبع أصغر بكثير من LLM حقيقي، لكن المبدأ واحد)
2. اصنع يدويًا 30-50 زوج (تعليمة، استجابة) بسيطة جدًا (رياضيات، حقائق عامة)
3. طبّق ChatTemplate وbuild_training_example لتحضير بيانات التدريب
4. درّب النموذج لـ 3 حقبات بمعدل تعلّم منخفض (2e-5)
5. قارن سلوك النموذج قبل وبعد SFT على نفس مجموعة أسئلة اختبار
   (هل يُجيب مباشرة بعد SFT بدلاً من "إكمال نص" عشوائي؟)
"""

from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer


def load_base_model_and_tokenizer(
    model_name: str = "distilgpt2",
) -> tuple:
    """TODO: حمّل النموذج والـ tokenizer، وأضف الرموز الخاصة
    (<|im_start|>, <|im_end|>) إلى الـ tokenizer، ثم وسّع embedding
    الخاص بالنموذج ليتوافق مع المفردات الجديدة
    (model.resize_token_embeddings)."""
    raise NotImplementedError


def create_toy_sft_dataset() -> list[list[ChatMessage]]:
    """TODO: أنشئ يدويًا 30-50 محادثة بسيطة (تعليمة + استجابة مباشرة)
    تغطي: حقائق رياضية بسيطة، أسئلة عامة، تعليمات بسيطة
    (مثال: "اكتب جملة عن الشمس" -> استجابة قصيرة مباشرة)."""
    raise NotImplementedError


def evaluate_before_after(
    model, tokenizer, test_prompts: list[str]
) -> None:
    """TODO: ولّد استجابات للنموذج على test_prompts قبل وبعد SFT،
    واطبع المقارنة جنبًا إلى جنب. راقب: هل يتوقف النموذج بشكل مناسب
    بعد SFT بدلاً من الاستمرار في توليد نص عشوائي؟"""
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: اربط كل الدوال في تجربة متكاملة، وسجّل ملاحظاتك
    # حول الفرق الملموس في السلوك قبل وبعد SFT حتى على نموذج صغير جدًا
    pass

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-25-chapter-25.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 25: الفصل 25: من Base إلى نموذج قابل للاستخدام — Supervised Fine-Tuning والبيانات الاصطناعية."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_sft() -> None:
    prompt, answer = ('User: define attention\nAssistant:', ' attention weights combine context')
    tokens = (prompt + answer).split()
    labels = [-100] * len(prompt.split()) + list(range(len(prompt.split()), len(tokens)))
    print('tokens:', tokens)
    print('loss-masked labels:', labels)
    print('supervised completion tokens:', sum((label != -100 for label in labels)))

if __name__ == "__main__":
    print("Chapter 25 lab: Mask prompt tokens in supervised fine-tuning")
    run_sft()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-25-chapter-25.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChatMessage:
    role: str   # "system" | "user" | "assistant"
    content: str


@dataclass
class ChatTemplate:
    """تطبيق مبسّط لتنسيق على نمط ChatML، قابل للتخصيص لتنسيقات أخرى
    (Llama-3, Mistral, Alpaca) بتغيير الرموز الخاصة فقط."""

    bos_token: str = "<|im_start|>"
    eos_token: str = "<|im_end|>"

    def format_conversation(self, messages: list[ChatMessage]) -> str:
        formatted_parts: list[str] = []
        for msg in messages:
            formatted_parts.append(
                f"{self.bos_token}{msg.role}\n{msg.content}{self.eos_token}\n"
            )
        return "".join(formatted_parts)

    def format_for_training(
        self, messages: list[ChatMessage]
    ) -> tuple[str, list[tuple[int, int]]]:
        """يُرجع النص الكامل بالإضافة إلى نطاقات (start, end) بالأحرف
        لكل جزء "assistant" فقط — ضروري لاحقًا لبناء قناع الخسارة (Loss Mask)."""
        full_text = ""
        assistant_spans: list[tuple[int, int]] = []

        for msg in messages:
            prefix = f"{self.bos_token}{msg.role}\n"
            content_start = len(full_text) + len(prefix)
            segment = f"{prefix}{msg.content}{self.eos_token}\n"

            if msg.role == "assistant":
                content_end = content_start + len(msg.content) + len(self.eos_token)
                assistant_spans.append((content_start, content_end))

            full_text += segment

        return full_text, assistant_spans

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-25-chapter-25.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase


def build_training_example(
    messages: list[ChatMessage],
    tokenizer: PreTrainedTokenizerBase,
    template: ChatTemplate,
    max_length: int = 2048,
) -> dict[str, torch.Tensor]:
    """بناء مثال تدريب كامل: input_ids + labels مع قناع الخسارة الصحيح
    على كل شيء عدا رموز استجابة المساعد."""

    full_text, assistant_char_spans = template.format_for_training(messages)

    encoding = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        return_offsets_mapping=True,  # ضروري لربط كل رمز بموضعه الحرفي الأصلي
    )

    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    # نبدأ بتقنيع كل شيء (-100)، ثم "نفتح" فقط رموز استجابة المساعد
    labels = [-100] * len(input_ids)

    for token_idx, (char_start, char_end) in enumerate(offsets):
        if char_start == char_end:  # رمز خاص بلا نطاق حرفي حقيقي
            continue
        for span_start, span_end in assistant_char_spans:
            if char_start >= span_start and char_end <= span_end:
                labels[token_idx] = input_ids[token_idx]
                break

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-25-chapter-25.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import difflib
import random
from dataclasses import dataclass


@dataclass
class InstructionExample:
    instruction: str
    input_text: str = ""
    output: str = ""


class SelfInstructPipeline:
    """تطبيق مبسّط تعليمي لخوارزمية Self-Instruct — في الإنتاج الفعلي
    تُستبدل دالة التوليد بنداء حقيقي لـ API نموذج معلّم قوي (GPT-4/Claude)."""

    def __init__(
        self,
        seed_tasks: list[InstructionExample],
        similarity_threshold: float = 0.7,
        num_few_shot_examples: int = 6,
    ) -> None:
        self.task_pool: list[InstructionExample] = list(seed_tasks)
        self.similarity_threshold = similarity_threshold
        self.num_few_shot_examples = num_few_shot_examples
        self._rng = random.Random(42)

    def _build_few_shot_prompt(self) -> str:
        """بناء Prompt يحتوي عينة عشوائية من المهام الموجودة كأمثلة توجيهية
        للنموذج المعلّم، لتوليد تعليمة جديدة مشابهة في الأسلوب والتنوع."""
        samples = self._rng.sample(
            self.task_pool, min(self.num_few_shot_examples, len(self.task_pool))
        )
        prompt_lines = ["وَلِّد تعليمة جديدة متنوعة على نمط الأمثلة التالية:\n"]
        for i, ex in enumerate(samples, start=1):
            prompt_lines.append(f"{i}. {ex.instruction}")
        prompt_lines.append(f"{len(samples) + 1}.")
        return "\n".join(prompt_lines)

    def _is_too_similar(self, new_instruction: str) -> bool:
        """فحص تشابه بسيط (ROUGE-L مبسّط عبر SequenceMatcher) لمنع
        تكرار تعليمات شبه متطابقة في مجموعة البيانات النهائية."""
        for existing in self.task_pool:
            similarity = difflib.SequenceMatcher(
                None, new_instruction.lower(), existing.instruction.lower()
            ).ratio()
            if similarity > self.similarity_threshold:
                return True
        return False

    def generate_new_instruction(self, teacher_model_call: callable) -> str | None:
        """teacher_model_call: دالة تستدعي نموذج معلّم حقيقي (API) وتُرجع نصًا.
        هنا نُحاكي واجهتها فقط لأغراض توضيحية."""
        prompt = self._build_few_shot_prompt()
        candidate_instruction = teacher_model_call(prompt).strip()

        if self._is_too_similar(candidate_instruction):
            return None  # رُفض بسبب التشابه الزائد — دورة أخرى مطلوبة

        return candidate_instruction

    def generate_response(
        self, instruction: str, teacher_model_call: callable
    ) -> str:
        response_prompt = f"أجب على التعليمة التالية بدقة ووضوح:\n{instruction}"
        return teacher_model_call(response_prompt).strip()

    def run_bootstrap_round(
        self, teacher_model_call: callable, num_new_examples: int = 50
    ) -> list[InstructionExample]:
        new_examples: list[InstructionExample] = []
        attempts = 0
        max_attempts = num_new_examples * 3  # هامش أمان لمحاولات التشابه الفاشلة

        while len(new_examples) < num_new_examples and attempts < max_attempts:
            attempts += 1
            instruction = self.generate_new_instruction(teacher_model_call)
            if instruction is None:
                continue

            output = self.generate_response(instruction, teacher_model_call)
            example = InstructionExample(instruction=instruction, output=output)
            new_examples.append(example)
            self.task_pool.append(example)  # يُغذّي الجولات التالية بمزيد من التنوع

        return new_examples

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-25-chapter-25.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    """مجموعة بيانات SFT بسيطة تطبّق Padding بدلاً من Packing —
    أبسط للفهم والتصحيح، وإن كانت أقل كفاءة حسابية قليلاً من Packing
    المتقدّم الذي يتطلب قناع انتباه حدودي (Block-Diagonal Attention Mask)."""

    def __init__(
        self,
        examples: list[list[ChatMessage]],
        tokenizer,
        template: ChatTemplate,
        max_length: int = 2048,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.template = template
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        messages = self.examples[idx]
        return build_training_example(
            messages, self.tokenizer, self.template, self.max_length
        )


def sft_collate_fn(
    batch: list[dict[str, torch.Tensor]], pad_token_id: int
) -> dict[str, torch.Tensor]:
    """تجميع دفعة مع Padding ديناميكي (إلى أطول عيّنة في الدفعة فقط،
    وليس إلى max_length الثابت) لتوفير حوسبة غير ضرورية."""
    max_len_in_batch = max(item["input_ids"].size(0) for item in batch)

    input_ids_batch = []
    labels_batch = []
    attention_mask_batch = []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len_in_batch - seq_len

        input_ids_batch.append(
            torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id)])
        )
        # -100 دائمًا للحشو — لا نريد حساب خسارة على رموز Padding أبدًا
        labels_batch.append(
            torch.cat([item["labels"], torch.full((pad_len,), -100)])
        )
        attention_mask_batch.append(
            torch.cat([torch.ones(seq_len), torch.zeros(pad_len)])
        )

    return {
        "input_ids": torch.stack(input_ids_batch).long(),
        "labels": torch.stack(labels_batch).long(),
        "attention_mask": torch.stack(attention_mask_batch).long(),
    }


def sft_training_step(
    model, batch: dict[str, torch.Tensor], optimizer: torch.optim.Optimizer
) -> float:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],  # معظم مكتبات HF تحسب الخسارة المُقنَّعة داخليًا تلقائيًا
    )
    loss = outputs.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.item()

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-25-chapter-25.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
messages = [
    ChatMessage(role="system", content="أنت مساعد مفيد."),
    ChatMessage(role="user", content="ما نتيجة 2+2؟"),
    ChatMessage(role="assistant", content="نتيجة 2+2 هي 4."),
]

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-25-chapter-25.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: النموذج "يتعلّم" أيضًا كيف يُعيد كتابة أسئلة المستخدمين
labels = input_ids.clone()  # بلا أي تقنيع!
loss = cross_entropy(logits, labels)

# ✅ صحيح: قناع صريح يحصر الخسارة على استجابة المساعد فقط
labels = build_masked_labels(input_ids, assistant_spans)
loss = cross_entropy(logits, labels, ignore_index=-100)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-25-chapter-25.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: معدل تعلّم عالٍ جدًا لـ SFT يمحو جزءًا كبيرًا من المعرفة
# المُكتسبة في Pretraining (ظاهرة تُسمى "Catastrophic Forgetting")
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)  # مناسب لـ Pretraining فقط

# ✅ صحيح: معدل تعلّم أقل بعشرة أضعاف تقريبًا لـ SFT
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-25-chapter-25.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: 10 حقبات على 50,000 مثال فقط يؤدي لـ Overfitting حاد،
# والنموذج يبدأ "بحفظ" الاستجابات بدلاً من تعميم السلوك
for epoch in range(10):
    train_one_epoch(model, sft_dataloader)

# ✅ صحيح: 2-3 حقبات عادة كافية تمامًا، مع مراقبة Validation Loss
# للتوقف المبكر (Early Stopping) إذا بدأ بالتدهور
for epoch in range(3):
    train_one_epoch(model, sft_dataloader)
    val_loss = evaluate(model, val_dataloader)
    if val_loss > best_val_loss:
        break  # إشارة Overfitting مبكرة

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-25-chapter-25.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# ❌ خطير: دمج عدة محادثات في تسلسل واحد لتحسين الكفاءة، لكن بدون منع
# الانتباه عبر حدود المحادثات — النموذج "يرى" محادثة أخرى غير متعلقة!
packed_sequence = concat([conv1_tokens, conv2_tokens, conv3_tokens])
# بدون قناع خاص، Self-Attention يسمح لرموز conv2 برؤية رموز conv1 بالكامل

# ✅ صحيح: استخدام قناع انتباه Block-Diagonal يعزل كل محادثة عن الأخرى
# (مدعوم أصلاً في تطبيقات حديثة مثل flash_attn_varlen_func)
attention_mask = build_block_diagonal_mask(sequence_boundaries)

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-25-chapter-25.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
"""
مشروع تطبيقي: خط أنابيب SFT كامل من البداية للنهاية

المطلوب إكماله:
1. استخدم نموذجًا صغيرًا مُدرَّبًا مسبقًا (مثل "gpt2" أو "distilgpt2" من Hugging Face)
   كـ"نموذج أساسي" توضيحي (بالطبع أصغر بكثير من LLM حقيقي، لكن المبدأ واحد)
2. اصنع يدويًا 30-50 زوج (تعليمة، استجابة) بسيطة جدًا (رياضيات، حقائق عامة)
3. طبّق ChatTemplate وbuild_training_example لتحضير بيانات التدريب
4. درّب النموذج لـ 3 حقبات بمعدل تعلّم منخفض (2e-5)
5. قارن سلوك النموذج قبل وبعد SFT على نفس مجموعة أسئلة اختبار
   (هل يُجيب مباشرة بعد SFT بدلاً من "إكمال نص" عشوائي؟)
"""

from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer


def load_base_model_and_tokenizer(
    model_name: str = "distilgpt2",
) -> tuple:
    """TODO: حمّل النموذج والـ tokenizer، وأضف الرموز الخاصة
    (<|im_start|>, <|im_end|>) إلى الـ tokenizer، ثم وسّع embedding
    الخاص بالنموذج ليتوافق مع المفردات الجديدة
    (model.resize_token_embeddings)."""
    raise NotImplementedError


def create_toy_sft_dataset() -> list[list[ChatMessage]]:
    """TODO: أنشئ يدويًا 30-50 محادثة بسيطة (تعليمة + استجابة مباشرة)
    تغطي: حقائق رياضية بسيطة، أسئلة عامة، تعليمات بسيطة
    (مثال: "اكتب جملة عن الشمس" -> استجابة قصيرة مباشرة)."""
    raise NotImplementedError


def evaluate_before_after(
    model, tokenizer, test_prompts: list[str]
) -> None:
    """TODO: ولّد استجابات للنموذج على test_prompts قبل وبعد SFT،
    واطبع المقارنة جنبًا إلى جنب. راقب: هل يتوقف النموذج بشكل مناسب
    بعد SFT بدلاً من الاستمرار في توليد نص عشوائي؟"""
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: اربط كل الدوال في تجربة متكاملة، وسجّل ملاحظاتك
    # حول الفرق الملموس في السلوك قبل وبعد SFT حتى على نموذج صغير جدًا
    pass

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-25-chapter-25.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — من Base إلى نموذج قابل للاستخدام — Supervised Fine-Tuning والبيانات الاصطناعية
Large Language Model Engineering — Base

Book slug : llm-engineering-69-chapters
Chapter   : ch-25-chapter-25 (lesson 25)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_25.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 25: الفصل 25: من Base إلى نموذج قابل للاستخدام — Supervised Fine-Tuning والبيانات الاصطناعية."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_sft() -> None:
    prompt, answer = ('User: define attention\nAssistant:', ' attention weights combine context')
    tokens = (prompt + answer).split()
    labels = [-100] * len(prompt.split()) + list(range(len(prompt.split()), len(tokens)))
    print('tokens:', tokens)
    print('loss-masked labels:', labels)
    print('supervised completion tokens:', sum((label != -100 for label in labels)))

if __name__ == "__main__":
    print("Chapter 25 lab: Mask prompt tokens in supervised fine-tuning")
    run_sft()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-05-build-an-llm/chapter-25.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChatMessage:
    role: str   # "system" | "user" | "assistant"
    content: str


@dataclass
class ChatTemplate:
    """تطبيق مبسّط لتنسيق على نمط ChatML، قابل للتخصيص لتنسيقات أخرى
    (Llama-3, Mistral, Alpaca) بتغيير الرموز الخاصة فقط."""

    bos_token: str = "<|im_start|>"
    eos_token: str = "<|im_end|>"

    def format_conversation(self, messages: list[ChatMessage]) -> str:
        formatted_parts: list[str] = []
        for msg in messages:
            formatted_parts.append(
                f"{self.bos_token}{msg.role}\n{msg.content}{self.eos_token}\n"
            )
        return "".join(formatted_parts)

    def format_for_training(
        self, messages: list[ChatMessage]
    ) -> tuple[str, list[tuple[int, int]]]:
        """يُرجع النص الكامل بالإضافة إلى نطاقات (start, end) بالأحرف
        لكل جزء "assistant" فقط — ضروري لاحقًا لبناء قناع الخسارة (Loss Mask)."""
        full_text = ""
        assistant_spans: list[tuple[int, int]] = []

        for msg in messages:
            prefix = f"{self.bos_token}{msg.role}\n"
            content_start = len(full_text) + len(prefix)
            segment = f"{prefix}{msg.content}{self.eos_token}\n"

            if msg.role == "assistant":
                content_end = content_start + len(msg.content) + len(self.eos_token)
                assistant_spans.append((content_start, content_end))

            full_text += segment

        return full_text, assistant_spans

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-05-build-an-llm/chapter-25.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase


def build_training_example(
    messages: list[ChatMessage],
    tokenizer: PreTrainedTokenizerBase,
    template: ChatTemplate,
    max_length: int = 2048,
) -> dict[str, torch.Tensor]:
    """بناء مثال تدريب كامل: input_ids + labels مع قناع الخسارة الصحيح
    على كل شيء عدا رموز استجابة المساعد."""

    full_text, assistant_char_spans = template.format_for_training(messages)

    encoding = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        return_offsets_mapping=True,  # ضروري لربط كل رمز بموضعه الحرفي الأصلي
    )

    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    # نبدأ بتقنيع كل شيء (-100)، ثم "نفتح" فقط رموز استجابة المساعد
    labels = [-100] * len(input_ids)

    for token_idx, (char_start, char_end) in enumerate(offsets):
        if char_start == char_end:  # رمز خاص بلا نطاق حرفي حقيقي
            continue
        for span_start, span_end in assistant_char_spans:
            if char_start >= span_start and char_end <= span_end:
                labels[token_idx] = input_ids[token_idx]
                break

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-05-build-an-llm/chapter-25.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import difflib
import random
from dataclasses import dataclass


@dataclass
class InstructionExample:
    instruction: str
    input_text: str = ""
    output: str = ""


class SelfInstructPipeline:
    """تطبيق مبسّط تعليمي لخوارزمية Self-Instruct — في الإنتاج الفعلي
    تُستبدل دالة التوليد بنداء حقيقي لـ API نموذج معلّم قوي (GPT-4/Claude)."""

    def __init__(
        self,
        seed_tasks: list[InstructionExample],
        similarity_threshold: float = 0.7,
        num_few_shot_examples: int = 6,
    ) -> None:
        self.task_pool: list[InstructionExample] = list(seed_tasks)
        self.similarity_threshold = similarity_threshold
        self.num_few_shot_examples = num_few_shot_examples
        self._rng = random.Random(42)

    def _build_few_shot_prompt(self) -> str:
        """بناء Prompt يحتوي عينة عشوائية من المهام الموجودة كأمثلة توجيهية
        للنموذج المعلّم، لتوليد تعليمة جديدة مشابهة في الأسلوب والتنوع."""
        samples = self._rng.sample(
            self.task_pool, min(self.num_few_shot_examples, len(self.task_pool))
        )
        prompt_lines = ["وَلِّد تعليمة جديدة متنوعة على نمط الأمثلة التالية:\n"]
        for i, ex in enumerate(samples, start=1):
            prompt_lines.append(f"{i}. {ex.instruction}")
        prompt_lines.append(f"{len(samples) + 1}.")
        return "\n".join(prompt_lines)

    def _is_too_similar(self, new_instruction: str) -> bool:
        """فحص تشابه بسيط (ROUGE-L مبسّط عبر SequenceMatcher) لمنع
        تكرار تعليمات شبه متطابقة في مجموعة البيانات النهائية."""
        for existing in self.task_pool:
            similarity = difflib.SequenceMatcher(
                None, new_instruction.lower(), existing.instruction.lower()
            ).ratio()
            if similarity > self.similarity_threshold:
                return True
        return False

    def generate_new_instruction(self, teacher_model_call: callable) -> str | None:
        """teacher_model_call: دالة تستدعي نموذج معلّم حقيقي (API) وتُرجع نصًا.
        هنا نُحاكي واجهتها فقط لأغراض توضيحية."""
        prompt = self._build_few_shot_prompt()
        candidate_instruction = teacher_model_call(prompt).strip()

        if self._is_too_similar(candidate_instruction):
            return None  # رُفض بسبب التشابه الزائد — دورة أخرى مطلوبة

        return candidate_instruction

    def generate_response(
        self, instruction: str, teacher_model_call: callable
    ) -> str:
        response_prompt = f"أجب على التعليمة التالية بدقة ووضوح:\n{instruction}"
        return teacher_model_call(response_prompt).strip()

    def run_bootstrap_round(
        self, teacher_model_call: callable, num_new_examples: int = 50
    ) -> list[InstructionExample]:
        new_examples: list[InstructionExample] = []
        attempts = 0
        max_attempts = num_new_examples * 3  # هامش أمان لمحاولات التشابه الفاشلة

        while len(new_examples) < num_new_examples and attempts < max_attempts:
            attempts += 1
            instruction = self.generate_new_instruction(teacher_model_call)
            if instruction is None:
                continue

            output = self.generate_response(instruction, teacher_model_call)
            example = InstructionExample(instruction=instruction, output=output)
            new_examples.append(example)
            self.task_pool.append(example)  # يُغذّي الجولات التالية بمزيد من التنوع

        return new_examples

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-05-build-an-llm/chapter-25.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    """مجموعة بيانات SFT بسيطة تطبّق Padding بدلاً من Packing —
    أبسط للفهم والتصحيح، وإن كانت أقل كفاءة حسابية قليلاً من Packing
    المتقدّم الذي يتطلب قناع انتباه حدودي (Block-Diagonal Attention Mask)."""

    def __init__(
        self,
        examples: list[list[ChatMessage]],
        tokenizer,
        template: ChatTemplate,
        max_length: int = 2048,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.template = template
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        messages = self.examples[idx]
        return build_training_example(
            messages, self.tokenizer, self.template, self.max_length
        )


def sft_collate_fn(
    batch: list[dict[str, torch.Tensor]], pad_token_id: int
) -> dict[str, torch.Tensor]:
    """تجميع دفعة مع Padding ديناميكي (إلى أطول عيّنة في الدفعة فقط،
    وليس إلى max_length الثابت) لتوفير حوسبة غير ضرورية."""
    max_len_in_batch = max(item["input_ids"].size(0) for item in batch)

    input_ids_batch = []
    labels_batch = []
    attention_mask_batch = []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len_in_batch - seq_len

        input_ids_batch.append(
            torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id)])
        )
        # -100 دائمًا للحشو — لا نريد حساب خسارة على رموز Padding أبدًا
        labels_batch.append(
            torch.cat([item["labels"], torch.full((pad_len,), -100)])
        )
        attention_mask_batch.append(
            torch.cat([torch.ones(seq_len), torch.zeros(pad_len)])
        )

    return {
        "input_ids": torch.stack(input_ids_batch).long(),
        "labels": torch.stack(labels_batch).long(),
        "attention_mask": torch.stack(attention_mask_batch).long(),
    }


def sft_training_step(
    model, batch: dict[str, torch.Tensor], optimizer: torch.optim.Optimizer
) -> float:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],  # معظم مكتبات HF تحسب الخسارة المُقنَّعة داخليًا تلقائيًا
    )
    loss = outputs.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.item()

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-05-build-an-llm/chapter-25.md
# ────────────────────────────────────────────────────────────────────────
messages = [
    ChatMessage(role="system", content="أنت مساعد مفيد."),
    ChatMessage(role="user", content="ما نتيجة 2+2؟"),
    ChatMessage(role="assistant", content="نتيجة 2+2 هي 4."),
]

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-05-build-an-llm/chapter-25.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: النموذج "يتعلّم" أيضًا كيف يُعيد كتابة أسئلة المستخدمين
labels = input_ids.clone()  # بلا أي تقنيع!
loss = cross_entropy(logits, labels)

# ✅ صحيح: قناع صريح يحصر الخسارة على استجابة المساعد فقط
labels = build_masked_labels(input_ids, assistant_spans)
loss = cross_entropy(logits, labels, ignore_index=-100)

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-05-build-an-llm/chapter-25.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: معدل تعلّم عالٍ جدًا لـ SFT يمحو جزءًا كبيرًا من المعرفة
# المُكتسبة في Pretraining (ظاهرة تُسمى "Catastrophic Forgetting")
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)  # مناسب لـ Pretraining فقط

# ✅ صحيح: معدل تعلّم أقل بعشرة أضعاف تقريبًا لـ SFT
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# ────────────────────────────────────────────────────────────────────────
# [009] book/chapters/part-05-build-an-llm/chapter-25.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: 10 حقبات على 50,000 مثال فقط يؤدي لـ Overfitting حاد،
# والنموذج يبدأ "بحفظ" الاستجابات بدلاً من تعميم السلوك
for epoch in range(10):
    train_one_epoch(model, sft_dataloader)

# ✅ صحيح: 2-3 حقبات عادة كافية تمامًا، مع مراقبة Validation Loss
# للتوقف المبكر (Early Stopping) إذا بدأ بالتدهور
for epoch in range(3):
    train_one_epoch(model, sft_dataloader)
    val_loss = evaluate(model, val_dataloader)
    if val_loss > best_val_loss:
        break  # إشارة Overfitting مبكرة

# ────────────────────────────────────────────────────────────────────────
# [010] book/chapters/part-05-build-an-llm/chapter-25.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطير: دمج عدة محادثات في تسلسل واحد لتحسين الكفاءة، لكن بدون منع
# الانتباه عبر حدود المحادثات — النموذج "يرى" محادثة أخرى غير متعلقة!
packed_sequence = concat([conv1_tokens, conv2_tokens, conv3_tokens])
# بدون قناع خاص، Self-Attention يسمح لرموز conv2 برؤية رموز conv1 بالكامل

# ✅ صحيح: استخدام قناع انتباه Block-Diagonal يعزل كل محادثة عن الأخرى
# (مدعوم أصلاً في تطبيقات حديثة مثل flash_attn_varlen_func)
attention_mask = build_block_diagonal_mask(sequence_boundaries)

# ────────────────────────────────────────────────────────────────────────
# [011] book/chapters/part-05-build-an-llm/chapter-25.md
# ────────────────────────────────────────────────────────────────────────
"""
مشروع تطبيقي: خط أنابيب SFT كامل من البداية للنهاية

المطلوب إكماله:
1. استخدم نموذجًا صغيرًا مُدرَّبًا مسبقًا (مثل "gpt2" أو "distilgpt2" من Hugging Face)
   كـ"نموذج أساسي" توضيحي (بالطبع أصغر بكثير من LLM حقيقي، لكن المبدأ واحد)
2. اصنع يدويًا 30-50 زوج (تعليمة، استجابة) بسيطة جدًا (رياضيات، حقائق عامة)
3. طبّق ChatTemplate وbuild_training_example لتحضير بيانات التدريب
4. درّب النموذج لـ 3 حقبات بمعدل تعلّم منخفض (2e-5)
5. قارن سلوك النموذج قبل وبعد SFT على نفس مجموعة أسئلة اختبار
   (هل يُجيب مباشرة بعد SFT بدلاً من "إكمال نص" عشوائي؟)
"""

from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer


def load_base_model_and_tokenizer(
    model_name: str = "distilgpt2",
) -> tuple:
    """TODO: حمّل النموذج والـ tokenizer، وأضف الرموز الخاصة
    (<|im_start|>, <|im_end|>) إلى الـ tokenizer، ثم وسّع embedding
    الخاص بالنموذج ليتوافق مع المفردات الجديدة
    (model.resize_token_embeddings)."""
    raise NotImplementedError


def create_toy_sft_dataset() -> list[list[ChatMessage]]:
    """TODO: أنشئ يدويًا 30-50 محادثة بسيطة (تعليمة + استجابة مباشرة)
    تغطي: حقائق رياضية بسيطة، أسئلة عامة، تعليمات بسيطة
    (مثال: "اكتب جملة عن الشمس" -> استجابة قصيرة مباشرة)."""
    raise NotImplementedError


def evaluate_before_after(
    model, tokenizer, test_prompts: list[str]
) -> None:
    """TODO: ولّد استجابات للنموذج على test_prompts قبل وبعد SFT،
    واطبع المقارنة جنبًا إلى جنب. راقب: هل يتوقف النموذج بشكل مناسب
    بعد SFT بدلاً من الاستمرار في توليد نص عشوائي؟"""
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: اربط كل الدوال في تجربة متكاملة، وسجّل ملاحظاتك
    # حول الفرق الملموس في السلوك قبل وبعد SFT حتى على نموذج صغير جدًا
    pass

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-25-chapter-25.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 25: الفصل 25: من Base إلى نموذج قابل للاستخدام — Supervised Fine-Tuning والبيانات الاصطناعية."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_sft() -> None:
    prompt, answer = ('User: define attention\nAssistant:', ' attention weights combine context')
    tokens = (prompt + answer).split()
    labels = [-100] * len(prompt.split()) + list(range(len(prompt.split()), len(tokens)))
    print('tokens:', tokens)
    print('loss-masked labels:', labels)
    print('supervised completion tokens:', sum((label != -100 for label in labels)))

if __name__ == "__main__":
    print("Chapter 25 lab: Mask prompt tokens in supervised fine-tuning")
    run_sft()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-25-chapter-25.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChatMessage:
    role: str   # "system" | "user" | "assistant"
    content: str


@dataclass
class ChatTemplate:
    """تطبيق مبسّط لتنسيق على نمط ChatML، قابل للتخصيص لتنسيقات أخرى
    (Llama-3, Mistral, Alpaca) بتغيير الرموز الخاصة فقط."""

    bos_token: str = "<|im_start|>"
    eos_token: str = "<|im_end|>"

    def format_conversation(self, messages: list[ChatMessage]) -> str:
        formatted_parts: list[str] = []
        for msg in messages:
            formatted_parts.append(
                f"{self.bos_token}{msg.role}\n{msg.content}{self.eos_token}\n"
            )
        return "".join(formatted_parts)

    def format_for_training(
        self, messages: list[ChatMessage]
    ) -> tuple[str, list[tuple[int, int]]]:
        """يُرجع النص الكامل بالإضافة إلى نطاقات (start, end) بالأحرف
        لكل جزء "assistant" فقط — ضروري لاحقًا لبناء قناع الخسارة (Loss Mask)."""
        full_text = ""
        assistant_spans: list[tuple[int, int]] = []

        for msg in messages:
            prefix = f"{self.bos_token}{msg.role}\n"
            content_start = len(full_text) + len(prefix)
            segment = f"{prefix}{msg.content}{self.eos_token}\n"

            if msg.role == "assistant":
                content_end = content_start + len(msg.content) + len(self.eos_token)
                assistant_spans.append((content_start, content_end))

            full_text += segment

        return full_text, assistant_spans

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-25-chapter-25.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase


def build_training_example(
    messages: list[ChatMessage],
    tokenizer: PreTrainedTokenizerBase,
    template: ChatTemplate,
    max_length: int = 2048,
) -> dict[str, torch.Tensor]:
    """بناء مثال تدريب كامل: input_ids + labels مع قناع الخسارة الصحيح
    على كل شيء عدا رموز استجابة المساعد."""

    full_text, assistant_char_spans = template.format_for_training(messages)

    encoding = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        return_offsets_mapping=True,  # ضروري لربط كل رمز بموضعه الحرفي الأصلي
    )

    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    # نبدأ بتقنيع كل شيء (-100)، ثم "نفتح" فقط رموز استجابة المساعد
    labels = [-100] * len(input_ids)

    for token_idx, (char_start, char_end) in enumerate(offsets):
        if char_start == char_end:  # رمز خاص بلا نطاق حرفي حقيقي
            continue
        for span_start, span_end in assistant_char_spans:
            if char_start >= span_start and char_end <= span_end:
                labels[token_idx] = input_ids[token_idx]
                break

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-25-chapter-25.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import difflib
import random
from dataclasses import dataclass


@dataclass
class InstructionExample:
    instruction: str
    input_text: str = ""
    output: str = ""


class SelfInstructPipeline:
    """تطبيق مبسّط تعليمي لخوارزمية Self-Instruct — في الإنتاج الفعلي
    تُستبدل دالة التوليد بنداء حقيقي لـ API نموذج معلّم قوي (GPT-4/Claude)."""

    def __init__(
        self,
        seed_tasks: list[InstructionExample],
        similarity_threshold: float = 0.7,
        num_few_shot_examples: int = 6,
    ) -> None:
        self.task_pool: list[InstructionExample] = list(seed_tasks)
        self.similarity_threshold = similarity_threshold
        self.num_few_shot_examples = num_few_shot_examples
        self._rng = random.Random(42)

    def _build_few_shot_prompt(self) -> str:
        """بناء Prompt يحتوي عينة عشوائية من المهام الموجودة كأمثلة توجيهية
        للنموذج المعلّم، لتوليد تعليمة جديدة مشابهة في الأسلوب والتنوع."""
        samples = self._rng.sample(
            self.task_pool, min(self.num_few_shot_examples, len(self.task_pool))
        )
        prompt_lines = ["وَلِّد تعليمة جديدة متنوعة على نمط الأمثلة التالية:\n"]
        for i, ex in enumerate(samples, start=1):
            prompt_lines.append(f"{i}. {ex.instruction}")
        prompt_lines.append(f"{len(samples) + 1}.")
        return "\n".join(prompt_lines)

    def _is_too_similar(self, new_instruction: str) -> bool:
        """فحص تشابه بسيط (ROUGE-L مبسّط عبر SequenceMatcher) لمنع
        تكرار تعليمات شبه متطابقة في مجموعة البيانات النهائية."""
        for existing in self.task_pool:
            similarity = difflib.SequenceMatcher(
                None, new_instruction.lower(), existing.instruction.lower()
            ).ratio()
            if similarity > self.similarity_threshold:
                return True
        return False

    def generate_new_instruction(self, teacher_model_call: callable) -> str | None:
        """teacher_model_call: دالة تستدعي نموذج معلّم حقيقي (API) وتُرجع نصًا.
        هنا نُحاكي واجهتها فقط لأغراض توضيحية."""
        prompt = self._build_few_shot_prompt()
        candidate_instruction = teacher_model_call(prompt).strip()

        if self._is_too_similar(candidate_instruction):
            return None  # رُفض بسبب التشابه الزائد — دورة أخرى مطلوبة

        return candidate_instruction

    def generate_response(
        self, instruction: str, teacher_model_call: callable
    ) -> str:
        response_prompt = f"أجب على التعليمة التالية بدقة ووضوح:\n{instruction}"
        return teacher_model_call(response_prompt).strip()

    def run_bootstrap_round(
        self, teacher_model_call: callable, num_new_examples: int = 50
    ) -> list[InstructionExample]:
        new_examples: list[InstructionExample] = []
        attempts = 0
        max_attempts = num_new_examples * 3  # هامش أمان لمحاولات التشابه الفاشلة

        while len(new_examples) < num_new_examples and attempts < max_attempts:
            attempts += 1
            instruction = self.generate_new_instruction(teacher_model_call)
            if instruction is None:
                continue

            output = self.generate_response(instruction, teacher_model_call)
            example = InstructionExample(instruction=instruction, output=output)
            new_examples.append(example)
            self.task_pool.append(example)  # يُغذّي الجولات التالية بمزيد من التنوع

        return new_examples

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-25-chapter-25.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    """مجموعة بيانات SFT بسيطة تطبّق Padding بدلاً من Packing —
    أبسط للفهم والتصحيح، وإن كانت أقل كفاءة حسابية قليلاً من Packing
    المتقدّم الذي يتطلب قناع انتباه حدودي (Block-Diagonal Attention Mask)."""

    def __init__(
        self,
        examples: list[list[ChatMessage]],
        tokenizer,
        template: ChatTemplate,
        max_length: int = 2048,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.template = template
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        messages = self.examples[idx]
        return build_training_example(
            messages, self.tokenizer, self.template, self.max_length
        )


def sft_collate_fn(
    batch: list[dict[str, torch.Tensor]], pad_token_id: int
) -> dict[str, torch.Tensor]:
    """تجميع دفعة مع Padding ديناميكي (إلى أطول عيّنة في الدفعة فقط،
    وليس إلى max_length الثابت) لتوفير حوسبة غير ضرورية."""
    max_len_in_batch = max(item["input_ids"].size(0) for item in batch)

    input_ids_batch = []
    labels_batch = []
    attention_mask_batch = []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len_in_batch - seq_len

        input_ids_batch.append(
            torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id)])
        )
        # -100 دائمًا للحشو — لا نريد حساب خسارة على رموز Padding أبدًا
        labels_batch.append(
            torch.cat([item["labels"], torch.full((pad_len,), -100)])
        )
        attention_mask_batch.append(
            torch.cat([torch.ones(seq_len), torch.zeros(pad_len)])
        )

    return {
        "input_ids": torch.stack(input_ids_batch).long(),
        "labels": torch.stack(labels_batch).long(),
        "attention_mask": torch.stack(attention_mask_batch).long(),
    }


def sft_training_step(
    model, batch: dict[str, torch.Tensor], optimizer: torch.optim.Optimizer
) -> float:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],  # معظم مكتبات HF تحسب الخسارة المُقنَّعة داخليًا تلقائيًا
    )
    loss = outputs.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.item()

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-25-chapter-25.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
messages = [
    ChatMessage(role="system", content="أنت مساعد مفيد."),
    ChatMessage(role="user", content="ما نتيجة 2+2؟"),
    ChatMessage(role="assistant", content="نتيجة 2+2 هي 4."),
]

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-25-chapter-25.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: النموذج "يتعلّم" أيضًا كيف يُعيد كتابة أسئلة المستخدمين
labels = input_ids.clone()  # بلا أي تقنيع!
loss = cross_entropy(logits, labels)

# ✅ صحيح: قناع صريح يحصر الخسارة على استجابة المساعد فقط
labels = build_masked_labels(input_ids, assistant_spans)
loss = cross_entropy(logits, labels, ignore_index=-100)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-25-chapter-25.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: معدل تعلّم عالٍ جدًا لـ SFT يمحو جزءًا كبيرًا من المعرفة
# المُكتسبة في Pretraining (ظاهرة تُسمى "Catastrophic Forgetting")
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)  # مناسب لـ Pretraining فقط

# ✅ صحيح: معدل تعلّم أقل بعشرة أضعاف تقريبًا لـ SFT
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-25-chapter-25.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: 10 حقبات على 50,000 مثال فقط يؤدي لـ Overfitting حاد،
# والنموذج يبدأ "بحفظ" الاستجابات بدلاً من تعميم السلوك
for epoch in range(10):
    train_one_epoch(model, sft_dataloader)

# ✅ صحيح: 2-3 حقبات عادة كافية تمامًا، مع مراقبة Validation Loss
# للتوقف المبكر (Early Stopping) إذا بدأ بالتدهور
for epoch in range(3):
    train_one_epoch(model, sft_dataloader)
    val_loss = evaluate(model, val_dataloader)
    if val_loss > best_val_loss:
        break  # إشارة Overfitting مبكرة

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-25-chapter-25.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# ❌ خطير: دمج عدة محادثات في تسلسل واحد لتحسين الكفاءة، لكن بدون منع
# الانتباه عبر حدود المحادثات — النموذج "يرى" محادثة أخرى غير متعلقة!
packed_sequence = concat([conv1_tokens, conv2_tokens, conv3_tokens])
# بدون قناع خاص، Self-Attention يسمح لرموز conv2 برؤية رموز conv1 بالكامل

# ✅ صحيح: استخدام قناع انتباه Block-Diagonal يعزل كل محادثة عن الأخرى
# (مدعوم أصلاً في تطبيقات حديثة مثل flash_attn_varlen_func)
attention_mask = build_block_diagonal_mask(sequence_boundaries)

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-25-chapter-25.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
"""
مشروع تطبيقي: خط أنابيب SFT كامل من البداية للنهاية

المطلوب إكماله:
1. استخدم نموذجًا صغيرًا مُدرَّبًا مسبقًا (مثل "gpt2" أو "distilgpt2" من Hugging Face)
   كـ"نموذج أساسي" توضيحي (بالطبع أصغر بكثير من LLM حقيقي، لكن المبدأ واحد)
2. اصنع يدويًا 30-50 زوج (تعليمة، استجابة) بسيطة جدًا (رياضيات، حقائق عامة)
3. طبّق ChatTemplate وbuild_training_example لتحضير بيانات التدريب
4. درّب النموذج لـ 3 حقبات بمعدل تعلّم منخفض (2e-5)
5. قارن سلوك النموذج قبل وبعد SFT على نفس مجموعة أسئلة اختبار
   (هل يُجيب مباشرة بعد SFT بدلاً من "إكمال نص" عشوائي؟)
"""

from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer


def load_base_model_and_tokenizer(
    model_name: str = "distilgpt2",
) -> tuple:
    """TODO: حمّل النموذج والـ tokenizer، وأضف الرموز الخاصة
    (<|im_start|>, <|im_end|>) إلى الـ tokenizer، ثم وسّع embedding
    الخاص بالنموذج ليتوافق مع المفردات الجديدة
    (model.resize_token_embeddings)."""
    raise NotImplementedError


def create_toy_sft_dataset() -> list[list[ChatMessage]]:
    """TODO: أنشئ يدويًا 30-50 محادثة بسيطة (تعليمة + استجابة مباشرة)
    تغطي: حقائق رياضية بسيطة، أسئلة عامة، تعليمات بسيطة
    (مثال: "اكتب جملة عن الشمس" -> استجابة قصيرة مباشرة)."""
    raise NotImplementedError


def evaluate_before_after(
    model, tokenizer, test_prompts: list[str]
) -> None:
    """TODO: ولّد استجابات للنموذج على test_prompts قبل وبعد SFT،
    واطبع المقارنة جنبًا إلى جنب. راقب: هل يتوقف النموذج بشكل مناسب
    بعد SFT بدلاً من الاستمرار في توليد نص عشوائي؟"""
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: اربط كل الدوال في تجربة متكاملة، وسجّل ملاحظاتك
    # حول الفرق الملموس في السلوك قبل وبعد SFT حتى على نموذج صغير جدًا
    pass
