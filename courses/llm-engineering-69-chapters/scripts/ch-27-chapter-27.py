#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — Transfer Learning & Continued Pretraining
Large Language Model Engineering — Transfer Learning

Book slug : llm-engineering-69-chapters
Chapter   : ch-27-chapter-27 (lesson 27)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-27-chapter-27.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — Transfer Learning & Continued Pretraining
Large Language Model Engineering — Transfer Learning

Book slug : llm-engineering-69-chapters
Chapter   : ch-27-chapter-27 (lesson 27)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-27-chapter-27.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — Transfer Learning & Continued Pretraining
Large Language Model Engineering — Transfer Learning

Book slug : llm-engineering-69-chapters
Chapter   : ch-27-chapter-27 (lesson 27)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_27.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 27: الفصل 27: Transfer Learning & Continued Pretraining."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_transfer() -> None:
    general_loss, domain_loss, adapted_domain_loss = (0.42, 0.75, 0.39)
    retained_general_loss = 0.46
    print('domain loss before/after:', domain_loss, '->', adapted_domain_loss)
    print('general loss before/after:', general_loss, '->', retained_general_loss)
    print('domain gain:', round(domain_loss - adapted_domain_loss, 3))
    print('general regression:', round(retained_general_loss - general_loss, 3))

if __name__ == "__main__":
    print("Chapter 27 lab: Measure domain adaptation without forgetting")
    run_transfer()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-06-model-reuse-adaptation/chapter-27.md
# ────────────────────────────────────────────────────────────────────────
"""
chapter_27_continued_pretraining.py
=====================================
مشروع متكامل: Continued Pretraining لنموذج GPT-2
لتحويله إلى نموذج متخصص في النصوص الطبية العربية
المتطلبات: PyTorch 2.5+, Transformers 4.46+, Python 3.13+
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup
)
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import random
import math
from pathlib import Path

# ضبط البذرة العشوائية للتكرارية (Reproducibility)
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# استخدام أفضل جهاز متاح (دعم MPS لأجهزة Apple Silicon أيضًا)
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"🖥️  الجهاز المستخدم: {device}")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-06-model-reuse-adaptation/chapter-27.md
# ────────────────────────────────────────────────────────────────────────
@dataclass
class ContinuedPretrainingConfig:
    """
    تكوين شامل لعملية الـ Continued Pretraining
    كل معامل هنا له تأثير مباشر على مكافحة Catastrophic Forgetting
    """
    # إعدادات النموذج
    base_model_name: str = "openai-community/gpt2"
    max_seq_length: int = 512

    # إعدادات مكافحة النسيان الكارثي (الأهم في هذا الفصل!)
    replay_ratio: float = 0.15  # 15% بيانات عامة تُمزج مع البيانات الطبية
    learning_rate: float = 5e-6  # صغير جدًا مقارنة بالـ Pretraining الأصلي (عادة 1e-4)

    # إعدادات التدريب
    batch_size: int = 4
    gradient_accumulation_steps: int = 8  # Effective batch = 32
    num_epochs: int = 3
    warmup_ratio: float = 0.1  # نسبة خطوات الـ Warmup من إجمالي الخطوات
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0  # Gradient Clipping لمنع الانفجار

    # إعدادات المراقبة
    eval_every_n_steps: int = 50
    save_every_n_steps: int = 200
    output_dir: str = "./medical_arabic_gpt2"

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


config = ContinuedPretrainingConfig()

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-06-model-reuse-adaptation/chapter-27.md
# ────────────────────────────────────────────────────────────────────────
class ReplayMixedDataset(Dataset):
    """
    Dataset يطبّق استراتيجية Replay Buffer:
    يمزج بيانات النطاق الجديد (طبي) مع عينة من البيانات العامة الأصلية
    لمنع Catastrophic Forgetting

    🟡 هذه هي آلية العمل الأساسية التي ناقشناها في القسم 27.5
    """

    def __init__(
        self,
        domain_texts: List[str],      # النصوص الطبية (النطاق الجديد)
        general_texts: List[str],      # عينة من بيانات عامة (Replay buffer)
        tokenizer,
        max_length: int = 512,
        replay_ratio: float = 0.15
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # حساب عدد عينات الـ Replay المطلوبة
        n_domain = len(domain_texts)
        n_replay = int(n_domain * replay_ratio / (1 - replay_ratio))

        # أخذ عينة عشوائية من البيانات العامة (مع التكرار إن لزم)
        if n_replay > len(general_texts):
            sampled_general = random.choices(general_texts, k=n_replay)
        else:
            sampled_general = random.sample(general_texts, n_replay)

        # دمج وخلط البيانات
        self.texts = domain_texts + sampled_general
        self.labels = (
            ["domain"] * n_domain + ["replay"] * len(sampled_general)
        )

        # خلط الترتيب (مهم جدًا! بدون هذا، سيرى النموذج كل البيانات
        # الطبية أولًا ثم العامة، مما يُفاقم النسيان قرب نهاية كل Epoch)
        combined = list(zip(self.texts, self.labels))
        random.shuffle(combined)
        self.texts, self.labels = zip(*combined)

        print(f"📊 إحصائيات الـ Dataset:")
        print(f"   - عينات النطاق الطبي: {n_domain}")
        print(f"   - عينات الـ Replay (عامة): {len(sampled_general)}")
        print(f"   - النسبة الفعلية للـ Replay: "
              f"{len(sampled_general)/len(self.texts):.1%}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # في Causal LM، الـ labels هي نفس input_ids
        # (النموذج يتعلم التنبؤ بالتوكن التالي)
        labels = input_ids.clone()
        # تجاهل الـ padding tokens في حساب الخسارة
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "source": self.labels[idx]  # لتتبع الأداء حسب المصدر لاحقًا
        }

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-06-model-reuse-adaptation/chapter-27.md
# ────────────────────────────────────────────────────────────────────────
def prepare_sample_datasets():
    """
    تحضير عينات بيانات للتوضيح:
    - domain_texts: نصوص طبية عربية (محاكاة)
    - general_texts: نصوص عامة عربية (لـ Replay Buffer)
    """

    # عينة من نصوص طبية عربية (في الواقع: آلاف/ملايين الأمثلة)
    domain_texts = [
        "يُعرّف احتشاء عضلة القلب بأنه موت جزء من أنسجة القلب "
        "بسبب انقطاع تدفق الدم الغني بالأكسجين إلى تلك المنطقة.",

        "تشير الدراسات الحديثة إلى أن ارتفاع ضغط الدم المزمن "
        "يُعد عامل خطر رئيسي للإصابة بالسكتة الدماغية والفشل الكلوي.",

        "يتم تشخيص السكري من النوع الثاني عادة من خلال فحص "
        "مستوى السكر التراكمي HbA1c، حيث تشير النسبة فوق 6.5% إلى الإصابة.",

        "العلاج الكيميائي يعمل عن طريق استهداف الخلايا سريعة "
        "الانقسام، مما يفسر الآثار الجانبية المرتبطة بالشعر والجهاز الهضمي.",

        "تعتبر المضادات الحيوية غير فعالة ضد الالتهابات الفيروسية، "
        "ويُعد الاستخدام غير الصحيح لها سببًا رئيسيًا لمقاومة البكتيريا.",

        # ... في مشروع حقيقي: آلاف الأمثلة الإضافية
    ] * 50  # تكرار للمحاكاة فقط (لا تفعل هذا في بيانات حقيقية!)

    # عينة من نصوص عامة عربية (لمنع نسيان القدرات العامة)
    general_texts = [
        "تُعتبر مدينة القاهرة من أقدم المدن المأهولة في العالم، "
        "حيث يعود تاريخها إلى آلاف السنين.",

        "الذكاء الاصطناعي يُحدث تحولًا جذريًا في طريقة تعاملنا "
        "مع التكنولوجيا في مختلف مناحي الحياة اليومية.",

        "يُعد الأدب العربي أحد أغنى الآداب العالمية، بما يحتويه "
        "من شعر ونثر يمتد عبر قرون طويلة من الإبداع.",

        "كرة القدم هي الرياضة الأكثر شعبية في العالم، حيث يتابعها "
        "ملايين المشجعين في كل قارة.",

        "الطهي العربي التقليدي يتميز بتنوعه الكبير، ويعكس تراث "
        "كل منطقة وثقافتها الخاصة بالتوابل والمكونات.",

        # ... في مشروع حقيقي: عينة ممثلة من بيانات Pretraining الأصلية
    ] * 50

    return domain_texts, general_texts

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-06-model-reuse-adaptation/chapter-27.md
# ────────────────────────────────────────────────────────────────────────
class ContinuedPretrainer:
    """
    مُدرّب متكامل لتطبيق Continued Pretraining
    مع تطبيق كل أفضل الممارسات لمكافحة Catastrophic Forgetting
    """

    def __init__(self, config: ContinuedPretrainingConfig):
        self.config = config

        print(f"🔄 تحميل النموذج الأساسي: {config.base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name
        ).to(device)

        # 🎯 خزن نسخة من المعاملات الأصلية لقياس "الانحراف" لاحقًا
        self.original_state_dict = {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }

        self.training_history = {
            "step": [], "loss": [], "domain_loss": [],
            "replay_loss": [], "param_drift": []
        }

    def compute_param_drift(self) -> float:
        """
        🔴 صندوق تعمّق: قياس مدى ابتعاد المعاملات الحالية
        عن المعاملات الأصلية (مؤشر مباشر على خطر النسيان الكارثي)

        نستخدم L2 norm للفرق بين المعاملات الحالية والأصلية،
        مُطبّعة بعدد المعاملات لتكون قابلة للمقارنة عبر النماذج
        """
        total_drift = 0.0
        total_params = 0

        current_state = self.model.state_dict()
        for key in self.original_state_dict:
            diff = (current_state[key] - self.original_state_dict[key])
            total_drift += torch.sum(diff ** 2).item()
            total_params += diff.numel()

        return math.sqrt(total_drift / total_params)

    def evaluate_forgetting(self, general_eval_texts: List[str]) -> float:
        """
        قياس Perplexity على عينة من البيانات العامة
        للتحقق من مدى حدوث النسيان الكارثي
        (نفس المفهوم من الفصل 4: Perplexity كمقياس للتعميم)
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for text in general_eval_texts:
                inputs = self.tokenizer(
                    text, return_tensors="pt",
                    truncation=True, max_length=512
                ).to(device)

                outputs = self.model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

        self.model.train()
        return perplexity

    def train(self, dataset: ReplayMixedDataset, eval_texts: List[str]):
        """حلقة التدريب الرئيسية الكاملة"""

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        total_steps = (
            len(dataloader) * self.config.num_epochs
            // self.config.gradient_accumulation_steps
        )
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        # 💡 استخدام AdamW مع LR منخفض جدًا - أساسي لمنع الانحراف الكبير
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Perplexity الأولي (خط الأساس قبل أي تدريب)
        baseline_ppl = self.evaluate_forgetting(eval_texts)
        print(f"\n📏 Perplexity الأساسي (قبل التدريب): {baseline_ppl:.2f}\n")

        self.model.train()
        global_step = 0

        print("🚀 بدء Continued Pretraining...\n")
        print("=" * 70)

        for epoch in range(self.config.num_epochs):
            domain_losses, replay_losses = [], []

            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                sources = batch["source"]

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss / self.config.gradient_accumulation_steps
                loss.backward()

                # تتبع الخسارة منفصلة حسب المصدر (طبي vs عام)
                for i, src in enumerate(sources):
                    item_loss = outputs.loss.item()
                    if src == "domain":
                        domain_losses.append(item_loss)
                    else:
                        replay_losses.append(item_loss)

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # ⚠️ Gradient Clipping - أساسي لمنع القفزات الكبيرة
                    # التي قد تسبب نسيانًا سريعًا ومفاجئًا
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % self.config.eval_every_n_steps == 0:
                        drift = self.compute_param_drift()
                        current_ppl = self.evaluate_forgetting(eval_texts)

                        avg_domain = np.mean(domain_losses[-50:]) if domain_losses else 0
                        avg_replay = np.mean(replay_losses[-50:]) if replay_losses else 0

                        print(
                            f"⚡ الخطوة {global_step:4d} | "
                            f"Loss(طبي)={avg_domain:.3f} | "
                            f"Loss(عام)={avg_replay:.3f} | "
                            f"Drift={drift:.5f} | "
                            f"PPL(عام)={current_ppl:.2f} "
                            f"({'📈' if current_ppl > baseline_ppl * 1.1 else '✅'})"
                        )

                        self.training_history["step"].append(global_step)
                        self.training_history["loss"].append(outputs.loss.item())
                        self.training_history["param_drift"].append(drift)

                        # ⚠️ تحذير تلقائي عند اكتشاف نسيان كارثي محتمل
                        if current_ppl > baseline_ppl * 1.5:
                            print(
                                "   🚨 تحذير: ارتفاع كبير في Perplexity على "
                                "البيانات العامة! قد يكون هناك نسيان كارثي. "
                                "فكّر في تقليل LR أو زيادة replay_ratio."
                            )

            print(f"\n✅ انتهت Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"   متوسط Loss طبي: {np.mean(domain_losses):.4f}")
            print(f"   متوسط Loss عام: {np.mean(replay_losses):.4f}")
            print("-" * 70)

        print("\n" + "=" * 70)
        print("🎉 اكتمل Continued Pretraining بنجاح!")
        final_ppl = self.evaluate_forgetting(eval_texts)
        print(f"📊 Perplexity النهائي على بيانات عامة: {final_ppl:.2f} "
              f"(الأساسي كان: {baseline_ppl:.2f})")

        if final_ppl < baseline_ppl * 1.15:
            print("✅ ممتاز! النموذج احتفظ بقدراته العامة بشكل جيد جدًا")
        else:
            print("⚠️  يُنصح بمراجعة استراتيجية مكافحة النسيان الكارثي")

    def save_model(self):
        """حفظ النموذج والـ Tokenizer"""
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        print(f"💾 تم حفظ النموذج في: {self.config.output_dir}")

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-06-model-reuse-adaptation/chapter-27.md
# ────────────────────────────────────────────────────────────────────────
def main():
    """نقطة الدخول الرئيسية - تشغيل التجربة كاملة"""

    # 1. تحضير البيانات
    print("📂 تحضير البيانات...")
    domain_texts, general_texts = prepare_sample_datasets()

    # تقسيم بيانات عامة إضافية للتقييم (منفصلة عن التدريب)
    eval_general_texts = general_texts[:20]
    train_general_texts = general_texts[20:]

    # 2. تهيئة المُدرّب
    pretrainer = ContinuedPretrainer(config)

    # 3. بناء Dataset مع استراتيجية Replay
    train_dataset = ReplayMixedDataset(
        domain_texts=domain_texts,
        general_texts=train_general_texts,
        tokenizer=pretrainer.tokenizer,
        max_length=config.max_seq_length,
        replay_ratio=config.replay_ratio
    )

    # 4. تشغيل التدريب
    pretrainer.train(train_dataset, eval_general_texts)

    # 5. حفظ النموذج النهائي
    pretrainer.save_model()

    # 6. اختبار نوعي سريع - مقارنة قبل وبعد
    print("\n" + "=" * 70)
    print("🧪 اختبار نوعي: توليد نص طبي")
    print("=" * 70)

    test_prompt = "العلاج الأمثل لارتفاع ضغط الدم يتضمن"
    inputs = pretrainer.tokenizer(test_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = pretrainer.model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=pretrainer.tokenizer.eos_token_id
        )

    generated_text = pretrainer.tokenizer.decode(
        output[0], skip_special_tokens=True
    )
    print(f"📝 النص المُولّد:\n{generated_text}")


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-27-chapter-27.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 27: الفصل 27: Transfer Learning & Continued Pretraining."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_transfer() -> None:
    general_loss, domain_loss, adapted_domain_loss = (0.42, 0.75, 0.39)
    retained_general_loss = 0.46
    print('domain loss before/after:', domain_loss, '->', adapted_domain_loss)
    print('general loss before/after:', general_loss, '->', retained_general_loss)
    print('domain gain:', round(domain_loss - adapted_domain_loss, 3))
    print('general regression:', round(retained_general_loss - general_loss, 3))

if __name__ == "__main__":
    print("Chapter 27 lab: Measure domain adaptation without forgetting")
    run_transfer()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-27-chapter-27.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
chapter_27_continued_pretraining.py
=====================================
مشروع متكامل: Continued Pretraining لنموذج GPT-2
لتحويله إلى نموذج متخصص في النصوص الطبية العربية
المتطلبات: PyTorch 2.5+, Transformers 4.46+, Python 3.13+
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup
)
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import random
import math
from pathlib import Path

# ضبط البذرة العشوائية للتكرارية (Reproducibility)
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# استخدام أفضل جهاز متاح (دعم MPS لأجهزة Apple Silicon أيضًا)
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"🖥️  الجهاز المستخدم: {device}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-27-chapter-27.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
@dataclass
class ContinuedPretrainingConfig:
    """
    تكوين شامل لعملية الـ Continued Pretraining
    كل معامل هنا له تأثير مباشر على مكافحة Catastrophic Forgetting
    """
    # إعدادات النموذج
    base_model_name: str = "openai-community/gpt2"
    max_seq_length: int = 512

    # إعدادات مكافحة النسيان الكارثي (الأهم في هذا الفصل!)
    replay_ratio: float = 0.15  # 15% بيانات عامة تُمزج مع البيانات الطبية
    learning_rate: float = 5e-6  # صغير جدًا مقارنة بالـ Pretraining الأصلي (عادة 1e-4)

    # إعدادات التدريب
    batch_size: int = 4
    gradient_accumulation_steps: int = 8  # Effective batch = 32
    num_epochs: int = 3
    warmup_ratio: float = 0.1  # نسبة خطوات الـ Warmup من إجمالي الخطوات
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0  # Gradient Clipping لمنع الانفجار

    # إعدادات المراقبة
    eval_every_n_steps: int = 50
    save_every_n_steps: int = 200
    output_dir: str = "./medical_arabic_gpt2"

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


config = ContinuedPretrainingConfig()

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-27-chapter-27.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class ReplayMixedDataset(Dataset):
    """
    Dataset يطبّق استراتيجية Replay Buffer:
    يمزج بيانات النطاق الجديد (طبي) مع عينة من البيانات العامة الأصلية
    لمنع Catastrophic Forgetting

    🟡 هذه هي آلية العمل الأساسية التي ناقشناها في القسم 27.5
    """

    def __init__(
        self,
        domain_texts: List[str],      # النصوص الطبية (النطاق الجديد)
        general_texts: List[str],      # عينة من بيانات عامة (Replay buffer)
        tokenizer,
        max_length: int = 512,
        replay_ratio: float = 0.15
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # حساب عدد عينات الـ Replay المطلوبة
        n_domain = len(domain_texts)
        n_replay = int(n_domain * replay_ratio / (1 - replay_ratio))

        # أخذ عينة عشوائية من البيانات العامة (مع التكرار إن لزم)
        if n_replay > len(general_texts):
            sampled_general = random.choices(general_texts, k=n_replay)
        else:
            sampled_general = random.sample(general_texts, n_replay)

        # دمج وخلط البيانات
        self.texts = domain_texts + sampled_general
        self.labels = (
            ["domain"] * n_domain + ["replay"] * len(sampled_general)
        )

        # خلط الترتيب (مهم جدًا! بدون هذا، سيرى النموذج كل البيانات
        # الطبية أولًا ثم العامة، مما يُفاقم النسيان قرب نهاية كل Epoch)
        combined = list(zip(self.texts, self.labels))
        random.shuffle(combined)
        self.texts, self.labels = zip(*combined)

        print(f"📊 إحصائيات الـ Dataset:")
        print(f"   - عينات النطاق الطبي: {n_domain}")
        print(f"   - عينات الـ Replay (عامة): {len(sampled_general)}")
        print(f"   - النسبة الفعلية للـ Replay: "
              f"{len(sampled_general)/len(self.texts):.1%}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # في Causal LM، الـ labels هي نفس input_ids
        # (النموذج يتعلم التنبؤ بالتوكن التالي)
        labels = input_ids.clone()
        # تجاهل الـ padding tokens في حساب الخسارة
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "source": self.labels[idx]  # لتتبع الأداء حسب المصدر لاحقًا
        }

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-27-chapter-27.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def prepare_sample_datasets():
    """
    تحضير عينات بيانات للتوضيح:
    - domain_texts: نصوص طبية عربية (محاكاة)
    - general_texts: نصوص عامة عربية (لـ Replay Buffer)
    """

    # عينة من نصوص طبية عربية (في الواقع: آلاف/ملايين الأمثلة)
    domain_texts = [
        "يُعرّف احتشاء عضلة القلب بأنه موت جزء من أنسجة القلب "
        "بسبب انقطاع تدفق الدم الغني بالأكسجين إلى تلك المنطقة.",

        "تشير الدراسات الحديثة إلى أن ارتفاع ضغط الدم المزمن "
        "يُعد عامل خطر رئيسي للإصابة بالسكتة الدماغية والفشل الكلوي.",

        "يتم تشخيص السكري من النوع الثاني عادة من خلال فحص "
        "مستوى السكر التراكمي HbA1c، حيث تشير النسبة فوق 6.5% إلى الإصابة.",

        "العلاج الكيميائي يعمل عن طريق استهداف الخلايا سريعة "
        "الانقسام، مما يفسر الآثار الجانبية المرتبطة بالشعر والجهاز الهضمي.",

        "تعتبر المضادات الحيوية غير فعالة ضد الالتهابات الفيروسية، "
        "ويُعد الاستخدام غير الصحيح لها سببًا رئيسيًا لمقاومة البكتيريا.",

        # ... في مشروع حقيقي: آلاف الأمثلة الإضافية
    ] * 50  # تكرار للمحاكاة فقط (لا تفعل هذا في بيانات حقيقية!)

    # عينة من نصوص عامة عربية (لمنع نسيان القدرات العامة)
    general_texts = [
        "تُعتبر مدينة القاهرة من أقدم المدن المأهولة في العالم، "
        "حيث يعود تاريخها إلى آلاف السنين.",

        "الذكاء الاصطناعي يُحدث تحولًا جذريًا في طريقة تعاملنا "
        "مع التكنولوجيا في مختلف مناحي الحياة اليومية.",

        "يُعد الأدب العربي أحد أغنى الآداب العالمية، بما يحتويه "
        "من شعر ونثر يمتد عبر قرون طويلة من الإبداع.",

        "كرة القدم هي الرياضة الأكثر شعبية في العالم، حيث يتابعها "
        "ملايين المشجعين في كل قارة.",

        "الطهي العربي التقليدي يتميز بتنوعه الكبير، ويعكس تراث "
        "كل منطقة وثقافتها الخاصة بالتوابل والمكونات.",

        # ... في مشروع حقيقي: عينة ممثلة من بيانات Pretraining الأصلية
    ] * 50

    return domain_texts, general_texts

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-27-chapter-27.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
class ContinuedPretrainer:
    """
    مُدرّب متكامل لتطبيق Continued Pretraining
    مع تطبيق كل أفضل الممارسات لمكافحة Catastrophic Forgetting
    """

    def __init__(self, config: ContinuedPretrainingConfig):
        self.config = config

        print(f"🔄 تحميل النموذج الأساسي: {config.base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name
        ).to(device)

        # 🎯 خزن نسخة من المعاملات الأصلية لقياس "الانحراف" لاحقًا
        self.original_state_dict = {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }

        self.training_history = {
            "step": [], "loss": [], "domain_loss": [],
            "replay_loss": [], "param_drift": []
        }

    def compute_param_drift(self) -> float:
        """
        🔴 صندوق تعمّق: قياس مدى ابتعاد المعاملات الحالية
        عن المعاملات الأصلية (مؤشر مباشر على خطر النسيان الكارثي)

        نستخدم L2 norm للفرق بين المعاملات الحالية والأصلية،
        مُطبّعة بعدد المعاملات لتكون قابلة للمقارنة عبر النماذج
        """
        total_drift = 0.0
        total_params = 0

        current_state = self.model.state_dict()
        for key in self.original_state_dict:
            diff = (current_state[key] - self.original_state_dict[key])
            total_drift += torch.sum(diff ** 2).item()
            total_params += diff.numel()

        return math.sqrt(total_drift / total_params)

    def evaluate_forgetting(self, general_eval_texts: List[str]) -> float:
        """
        قياس Perplexity على عينة من البيانات العامة
        للتحقق من مدى حدوث النسيان الكارثي
        (نفس المفهوم من الفصل 4: Perplexity كمقياس للتعميم)
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for text in general_eval_texts:
                inputs = self.tokenizer(
                    text, return_tensors="pt",
                    truncation=True, max_length=512
                ).to(device)

                outputs = self.model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

        self.model.train()
        return perplexity

    def train(self, dataset: ReplayMixedDataset, eval_texts: List[str]):
        """حلقة التدريب الرئيسية الكاملة"""

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        total_steps = (
            len(dataloader) * self.config.num_epochs
            // self.config.gradient_accumulation_steps
        )
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        # 💡 استخدام AdamW مع LR منخفض جدًا - أساسي لمنع الانحراف الكبير
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Perplexity الأولي (خط الأساس قبل أي تدريب)
        baseline_ppl = self.evaluate_forgetting(eval_texts)
        print(f"\n📏 Perplexity الأساسي (قبل التدريب): {baseline_ppl:.2f}\n")

        self.model.train()
        global_step = 0

        print("🚀 بدء Continued Pretraining...\n")
        print("=" * 70)

        for epoch in range(self.config.num_epochs):
            domain_losses, replay_losses = [], []

            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                sources = batch["source"]

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss / self.config.gradient_accumulation_steps
                loss.backward()

                # تتبع الخسارة منفصلة حسب المصدر (طبي vs عام)
                for i, src in enumerate(sources):
                    item_loss = outputs.loss.item()
                    if src == "domain":
                        domain_losses.append(item_loss)
                    else:
                        replay_losses.append(item_loss)

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # ⚠️ Gradient Clipping - أساسي لمنع القفزات الكبيرة
                    # التي قد تسبب نسيانًا سريعًا ومفاجئًا
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % self.config.eval_every_n_steps == 0:
                        drift = self.compute_param_drift()
                        current_ppl = self.evaluate_forgetting(eval_texts)

                        avg_domain = np.mean(domain_losses[-50:]) if domain_losses else 0
                        avg_replay = np.mean(replay_losses[-50:]) if replay_losses else 0

                        print(
                            f"⚡ الخطوة {global_step:4d} | "
                            f"Loss(طبي)={avg_domain:.3f} | "
                            f"Loss(عام)={avg_replay:.3f} | "
                            f"Drift={drift:.5f} | "
                            f"PPL(عام)={current_ppl:.2f} "
                            f"({'📈' if current_ppl > baseline_ppl * 1.1 else '✅'})"
                        )

                        self.training_history["step"].append(global_step)
                        self.training_history["loss"].append(outputs.loss.item())
                        self.training_history["param_drift"].append(drift)

                        # ⚠️ تحذير تلقائي عند اكتشاف نسيان كارثي محتمل
                        if current_ppl > baseline_ppl * 1.5:
                            print(
                                "   🚨 تحذير: ارتفاع كبير في Perplexity على "
                                "البيانات العامة! قد يكون هناك نسيان كارثي. "
                                "فكّر في تقليل LR أو زيادة replay_ratio."
                            )

            print(f"\n✅ انتهت Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"   متوسط Loss طبي: {np.mean(domain_losses):.4f}")
            print(f"   متوسط Loss عام: {np.mean(replay_losses):.4f}")
            print("-" * 70)

        print("\n" + "=" * 70)
        print("🎉 اكتمل Continued Pretraining بنجاح!")
        final_ppl = self.evaluate_forgetting(eval_texts)
        print(f"📊 Perplexity النهائي على بيانات عامة: {final_ppl:.2f} "
              f"(الأساسي كان: {baseline_ppl:.2f})")

        if final_ppl < baseline_ppl * 1.15:
            print("✅ ممتاز! النموذج احتفظ بقدراته العامة بشكل جيد جدًا")
        else:
            print("⚠️  يُنصح بمراجعة استراتيجية مكافحة النسيان الكارثي")

    def save_model(self):
        """حفظ النموذج والـ Tokenizer"""
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        print(f"💾 تم حفظ النموذج في: {self.config.output_dir}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-27-chapter-27.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def main():
    """نقطة الدخول الرئيسية - تشغيل التجربة كاملة"""

    # 1. تحضير البيانات
    print("📂 تحضير البيانات...")
    domain_texts, general_texts = prepare_sample_datasets()

    # تقسيم بيانات عامة إضافية للتقييم (منفصلة عن التدريب)
    eval_general_texts = general_texts[:20]
    train_general_texts = general_texts[20:]

    # 2. تهيئة المُدرّب
    pretrainer = ContinuedPretrainer(config)

    # 3. بناء Dataset مع استراتيجية Replay
    train_dataset = ReplayMixedDataset(
        domain_texts=domain_texts,
        general_texts=train_general_texts,
        tokenizer=pretrainer.tokenizer,
        max_length=config.max_seq_length,
        replay_ratio=config.replay_ratio
    )

    # 4. تشغيل التدريب
    pretrainer.train(train_dataset, eval_general_texts)

    # 5. حفظ النموذج النهائي
    pretrainer.save_model()

    # 6. اختبار نوعي سريع - مقارنة قبل وبعد
    print("\n" + "=" * 70)
    print("🧪 اختبار نوعي: توليد نص طبي")
    print("=" * 70)

    test_prompt = "العلاج الأمثل لارتفاع ضغط الدم يتضمن"
    inputs = pretrainer.tokenizer(test_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = pretrainer.model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=pretrainer.tokenizer.eos_token_id
        )

    generated_text = pretrainer.tokenizer.decode(
        output[0], skip_special_tokens=True
    )
    print(f"📝 النص المُولّد:\n{generated_text}")


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-27-chapter-27.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — Transfer Learning & Continued Pretraining
Large Language Model Engineering — Transfer Learning

Book slug : llm-engineering-69-chapters
Chapter   : ch-27-chapter-27 (lesson 27)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_27.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 27: الفصل 27: Transfer Learning & Continued Pretraining."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_transfer() -> None:
    general_loss, domain_loss, adapted_domain_loss = (0.42, 0.75, 0.39)
    retained_general_loss = 0.46
    print('domain loss before/after:', domain_loss, '->', adapted_domain_loss)
    print('general loss before/after:', general_loss, '->', retained_general_loss)
    print('domain gain:', round(domain_loss - adapted_domain_loss, 3))
    print('general regression:', round(retained_general_loss - general_loss, 3))

if __name__ == "__main__":
    print("Chapter 27 lab: Measure domain adaptation without forgetting")
    run_transfer()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-06-model-reuse-adaptation/chapter-27.md
# ────────────────────────────────────────────────────────────────────────
"""
chapter_27_continued_pretraining.py
=====================================
مشروع متكامل: Continued Pretraining لنموذج GPT-2
لتحويله إلى نموذج متخصص في النصوص الطبية العربية
المتطلبات: PyTorch 2.5+, Transformers 4.46+, Python 3.13+
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup
)
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import random
import math
from pathlib import Path

# ضبط البذرة العشوائية للتكرارية (Reproducibility)
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# استخدام أفضل جهاز متاح (دعم MPS لأجهزة Apple Silicon أيضًا)
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"🖥️  الجهاز المستخدم: {device}")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-06-model-reuse-adaptation/chapter-27.md
# ────────────────────────────────────────────────────────────────────────
@dataclass
class ContinuedPretrainingConfig:
    """
    تكوين شامل لعملية الـ Continued Pretraining
    كل معامل هنا له تأثير مباشر على مكافحة Catastrophic Forgetting
    """
    # إعدادات النموذج
    base_model_name: str = "openai-community/gpt2"
    max_seq_length: int = 512

    # إعدادات مكافحة النسيان الكارثي (الأهم في هذا الفصل!)
    replay_ratio: float = 0.15  # 15% بيانات عامة تُمزج مع البيانات الطبية
    learning_rate: float = 5e-6  # صغير جدًا مقارنة بالـ Pretraining الأصلي (عادة 1e-4)

    # إعدادات التدريب
    batch_size: int = 4
    gradient_accumulation_steps: int = 8  # Effective batch = 32
    num_epochs: int = 3
    warmup_ratio: float = 0.1  # نسبة خطوات الـ Warmup من إجمالي الخطوات
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0  # Gradient Clipping لمنع الانفجار

    # إعدادات المراقبة
    eval_every_n_steps: int = 50
    save_every_n_steps: int = 200
    output_dir: str = "./medical_arabic_gpt2"

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


config = ContinuedPretrainingConfig()

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-06-model-reuse-adaptation/chapter-27.md
# ────────────────────────────────────────────────────────────────────────
class ReplayMixedDataset(Dataset):
    """
    Dataset يطبّق استراتيجية Replay Buffer:
    يمزج بيانات النطاق الجديد (طبي) مع عينة من البيانات العامة الأصلية
    لمنع Catastrophic Forgetting

    🟡 هذه هي آلية العمل الأساسية التي ناقشناها في القسم 27.5
    """

    def __init__(
        self,
        domain_texts: List[str],      # النصوص الطبية (النطاق الجديد)
        general_texts: List[str],      # عينة من بيانات عامة (Replay buffer)
        tokenizer,
        max_length: int = 512,
        replay_ratio: float = 0.15
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # حساب عدد عينات الـ Replay المطلوبة
        n_domain = len(domain_texts)
        n_replay = int(n_domain * replay_ratio / (1 - replay_ratio))

        # أخذ عينة عشوائية من البيانات العامة (مع التكرار إن لزم)
        if n_replay > len(general_texts):
            sampled_general = random.choices(general_texts, k=n_replay)
        else:
            sampled_general = random.sample(general_texts, n_replay)

        # دمج وخلط البيانات
        self.texts = domain_texts + sampled_general
        self.labels = (
            ["domain"] * n_domain + ["replay"] * len(sampled_general)
        )

        # خلط الترتيب (مهم جدًا! بدون هذا، سيرى النموذج كل البيانات
        # الطبية أولًا ثم العامة، مما يُفاقم النسيان قرب نهاية كل Epoch)
        combined = list(zip(self.texts, self.labels))
        random.shuffle(combined)
        self.texts, self.labels = zip(*combined)

        print(f"📊 إحصائيات الـ Dataset:")
        print(f"   - عينات النطاق الطبي: {n_domain}")
        print(f"   - عينات الـ Replay (عامة): {len(sampled_general)}")
        print(f"   - النسبة الفعلية للـ Replay: "
              f"{len(sampled_general)/len(self.texts):.1%}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # في Causal LM، الـ labels هي نفس input_ids
        # (النموذج يتعلم التنبؤ بالتوكن التالي)
        labels = input_ids.clone()
        # تجاهل الـ padding tokens في حساب الخسارة
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "source": self.labels[idx]  # لتتبع الأداء حسب المصدر لاحقًا
        }

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-06-model-reuse-adaptation/chapter-27.md
# ────────────────────────────────────────────────────────────────────────
def prepare_sample_datasets():
    """
    تحضير عينات بيانات للتوضيح:
    - domain_texts: نصوص طبية عربية (محاكاة)
    - general_texts: نصوص عامة عربية (لـ Replay Buffer)
    """

    # عينة من نصوص طبية عربية (في الواقع: آلاف/ملايين الأمثلة)
    domain_texts = [
        "يُعرّف احتشاء عضلة القلب بأنه موت جزء من أنسجة القلب "
        "بسبب انقطاع تدفق الدم الغني بالأكسجين إلى تلك المنطقة.",

        "تشير الدراسات الحديثة إلى أن ارتفاع ضغط الدم المزمن "
        "يُعد عامل خطر رئيسي للإصابة بالسكتة الدماغية والفشل الكلوي.",

        "يتم تشخيص السكري من النوع الثاني عادة من خلال فحص "
        "مستوى السكر التراكمي HbA1c، حيث تشير النسبة فوق 6.5% إلى الإصابة.",

        "العلاج الكيميائي يعمل عن طريق استهداف الخلايا سريعة "
        "الانقسام، مما يفسر الآثار الجانبية المرتبطة بالشعر والجهاز الهضمي.",

        "تعتبر المضادات الحيوية غير فعالة ضد الالتهابات الفيروسية، "
        "ويُعد الاستخدام غير الصحيح لها سببًا رئيسيًا لمقاومة البكتيريا.",

        # ... في مشروع حقيقي: آلاف الأمثلة الإضافية
    ] * 50  # تكرار للمحاكاة فقط (لا تفعل هذا في بيانات حقيقية!)

    # عينة من نصوص عامة عربية (لمنع نسيان القدرات العامة)
    general_texts = [
        "تُعتبر مدينة القاهرة من أقدم المدن المأهولة في العالم، "
        "حيث يعود تاريخها إلى آلاف السنين.",

        "الذكاء الاصطناعي يُحدث تحولًا جذريًا في طريقة تعاملنا "
        "مع التكنولوجيا في مختلف مناحي الحياة اليومية.",

        "يُعد الأدب العربي أحد أغنى الآداب العالمية، بما يحتويه "
        "من شعر ونثر يمتد عبر قرون طويلة من الإبداع.",

        "كرة القدم هي الرياضة الأكثر شعبية في العالم، حيث يتابعها "
        "ملايين المشجعين في كل قارة.",

        "الطهي العربي التقليدي يتميز بتنوعه الكبير، ويعكس تراث "
        "كل منطقة وثقافتها الخاصة بالتوابل والمكونات.",

        # ... في مشروع حقيقي: عينة ممثلة من بيانات Pretraining الأصلية
    ] * 50

    return domain_texts, general_texts

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-06-model-reuse-adaptation/chapter-27.md
# ────────────────────────────────────────────────────────────────────────
class ContinuedPretrainer:
    """
    مُدرّب متكامل لتطبيق Continued Pretraining
    مع تطبيق كل أفضل الممارسات لمكافحة Catastrophic Forgetting
    """

    def __init__(self, config: ContinuedPretrainingConfig):
        self.config = config

        print(f"🔄 تحميل النموذج الأساسي: {config.base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name
        ).to(device)

        # 🎯 خزن نسخة من المعاملات الأصلية لقياس "الانحراف" لاحقًا
        self.original_state_dict = {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }

        self.training_history = {
            "step": [], "loss": [], "domain_loss": [],
            "replay_loss": [], "param_drift": []
        }

    def compute_param_drift(self) -> float:
        """
        🔴 صندوق تعمّق: قياس مدى ابتعاد المعاملات الحالية
        عن المعاملات الأصلية (مؤشر مباشر على خطر النسيان الكارثي)

        نستخدم L2 norm للفرق بين المعاملات الحالية والأصلية،
        مُطبّعة بعدد المعاملات لتكون قابلة للمقارنة عبر النماذج
        """
        total_drift = 0.0
        total_params = 0

        current_state = self.model.state_dict()
        for key in self.original_state_dict:
            diff = (current_state[key] - self.original_state_dict[key])
            total_drift += torch.sum(diff ** 2).item()
            total_params += diff.numel()

        return math.sqrt(total_drift / total_params)

    def evaluate_forgetting(self, general_eval_texts: List[str]) -> float:
        """
        قياس Perplexity على عينة من البيانات العامة
        للتحقق من مدى حدوث النسيان الكارثي
        (نفس المفهوم من الفصل 4: Perplexity كمقياس للتعميم)
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for text in general_eval_texts:
                inputs = self.tokenizer(
                    text, return_tensors="pt",
                    truncation=True, max_length=512
                ).to(device)

                outputs = self.model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

        self.model.train()
        return perplexity

    def train(self, dataset: ReplayMixedDataset, eval_texts: List[str]):
        """حلقة التدريب الرئيسية الكاملة"""

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        total_steps = (
            len(dataloader) * self.config.num_epochs
            // self.config.gradient_accumulation_steps
        )
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        # 💡 استخدام AdamW مع LR منخفض جدًا - أساسي لمنع الانحراف الكبير
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Perplexity الأولي (خط الأساس قبل أي تدريب)
        baseline_ppl = self.evaluate_forgetting(eval_texts)
        print(f"\n📏 Perplexity الأساسي (قبل التدريب): {baseline_ppl:.2f}\n")

        self.model.train()
        global_step = 0

        print("🚀 بدء Continued Pretraining...\n")
        print("=" * 70)

        for epoch in range(self.config.num_epochs):
            domain_losses, replay_losses = [], []

            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                sources = batch["source"]

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss / self.config.gradient_accumulation_steps
                loss.backward()

                # تتبع الخسارة منفصلة حسب المصدر (طبي vs عام)
                for i, src in enumerate(sources):
                    item_loss = outputs.loss.item()
                    if src == "domain":
                        domain_losses.append(item_loss)
                    else:
                        replay_losses.append(item_loss)

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # ⚠️ Gradient Clipping - أساسي لمنع القفزات الكبيرة
                    # التي قد تسبب نسيانًا سريعًا ومفاجئًا
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % self.config.eval_every_n_steps == 0:
                        drift = self.compute_param_drift()
                        current_ppl = self.evaluate_forgetting(eval_texts)

                        avg_domain = np.mean(domain_losses[-50:]) if domain_losses else 0
                        avg_replay = np.mean(replay_losses[-50:]) if replay_losses else 0

                        print(
                            f"⚡ الخطوة {global_step:4d} | "
                            f"Loss(طبي)={avg_domain:.3f} | "
                            f"Loss(عام)={avg_replay:.3f} | "
                            f"Drift={drift:.5f} | "
                            f"PPL(عام)={current_ppl:.2f} "
                            f"({'📈' if current_ppl > baseline_ppl * 1.1 else '✅'})"
                        )

                        self.training_history["step"].append(global_step)
                        self.training_history["loss"].append(outputs.loss.item())
                        self.training_history["param_drift"].append(drift)

                        # ⚠️ تحذير تلقائي عند اكتشاف نسيان كارثي محتمل
                        if current_ppl > baseline_ppl * 1.5:
                            print(
                                "   🚨 تحذير: ارتفاع كبير في Perplexity على "
                                "البيانات العامة! قد يكون هناك نسيان كارثي. "
                                "فكّر في تقليل LR أو زيادة replay_ratio."
                            )

            print(f"\n✅ انتهت Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"   متوسط Loss طبي: {np.mean(domain_losses):.4f}")
            print(f"   متوسط Loss عام: {np.mean(replay_losses):.4f}")
            print("-" * 70)

        print("\n" + "=" * 70)
        print("🎉 اكتمل Continued Pretraining بنجاح!")
        final_ppl = self.evaluate_forgetting(eval_texts)
        print(f"📊 Perplexity النهائي على بيانات عامة: {final_ppl:.2f} "
              f"(الأساسي كان: {baseline_ppl:.2f})")

        if final_ppl < baseline_ppl * 1.15:
            print("✅ ممتاز! النموذج احتفظ بقدراته العامة بشكل جيد جدًا")
        else:
            print("⚠️  يُنصح بمراجعة استراتيجية مكافحة النسيان الكارثي")

    def save_model(self):
        """حفظ النموذج والـ Tokenizer"""
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        print(f"💾 تم حفظ النموذج في: {self.config.output_dir}")

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-06-model-reuse-adaptation/chapter-27.md
# ────────────────────────────────────────────────────────────────────────
def main():
    """نقطة الدخول الرئيسية - تشغيل التجربة كاملة"""

    # 1. تحضير البيانات
    print("📂 تحضير البيانات...")
    domain_texts, general_texts = prepare_sample_datasets()

    # تقسيم بيانات عامة إضافية للتقييم (منفصلة عن التدريب)
    eval_general_texts = general_texts[:20]
    train_general_texts = general_texts[20:]

    # 2. تهيئة المُدرّب
    pretrainer = ContinuedPretrainer(config)

    # 3. بناء Dataset مع استراتيجية Replay
    train_dataset = ReplayMixedDataset(
        domain_texts=domain_texts,
        general_texts=train_general_texts,
        tokenizer=pretrainer.tokenizer,
        max_length=config.max_seq_length,
        replay_ratio=config.replay_ratio
    )

    # 4. تشغيل التدريب
    pretrainer.train(train_dataset, eval_general_texts)

    # 5. حفظ النموذج النهائي
    pretrainer.save_model()

    # 6. اختبار نوعي سريع - مقارنة قبل وبعد
    print("\n" + "=" * 70)
    print("🧪 اختبار نوعي: توليد نص طبي")
    print("=" * 70)

    test_prompt = "العلاج الأمثل لارتفاع ضغط الدم يتضمن"
    inputs = pretrainer.tokenizer(test_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = pretrainer.model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=pretrainer.tokenizer.eos_token_id
        )

    generated_text = pretrainer.tokenizer.decode(
        output[0], skip_special_tokens=True
    )
    print(f"📝 النص المُولّد:\n{generated_text}")


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-27-chapter-27.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 27: الفصل 27: Transfer Learning & Continued Pretraining."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_transfer() -> None:
    general_loss, domain_loss, adapted_domain_loss = (0.42, 0.75, 0.39)
    retained_general_loss = 0.46
    print('domain loss before/after:', domain_loss, '->', adapted_domain_loss)
    print('general loss before/after:', general_loss, '->', retained_general_loss)
    print('domain gain:', round(domain_loss - adapted_domain_loss, 3))
    print('general regression:', round(retained_general_loss - general_loss, 3))

if __name__ == "__main__":
    print("Chapter 27 lab: Measure domain adaptation without forgetting")
    run_transfer()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-27-chapter-27.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
chapter_27_continued_pretraining.py
=====================================
مشروع متكامل: Continued Pretraining لنموذج GPT-2
لتحويله إلى نموذج متخصص في النصوص الطبية العربية
المتطلبات: PyTorch 2.5+, Transformers 4.46+, Python 3.13+
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup
)
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import random
import math
from pathlib import Path

# ضبط البذرة العشوائية للتكرارية (Reproducibility)
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# استخدام أفضل جهاز متاح (دعم MPS لأجهزة Apple Silicon أيضًا)
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"🖥️  الجهاز المستخدم: {device}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-27-chapter-27.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
@dataclass
class ContinuedPretrainingConfig:
    """
    تكوين شامل لعملية الـ Continued Pretraining
    كل معامل هنا له تأثير مباشر على مكافحة Catastrophic Forgetting
    """
    # إعدادات النموذج
    base_model_name: str = "openai-community/gpt2"
    max_seq_length: int = 512

    # إعدادات مكافحة النسيان الكارثي (الأهم في هذا الفصل!)
    replay_ratio: float = 0.15  # 15% بيانات عامة تُمزج مع البيانات الطبية
    learning_rate: float = 5e-6  # صغير جدًا مقارنة بالـ Pretraining الأصلي (عادة 1e-4)

    # إعدادات التدريب
    batch_size: int = 4
    gradient_accumulation_steps: int = 8  # Effective batch = 32
    num_epochs: int = 3
    warmup_ratio: float = 0.1  # نسبة خطوات الـ Warmup من إجمالي الخطوات
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0  # Gradient Clipping لمنع الانفجار

    # إعدادات المراقبة
    eval_every_n_steps: int = 50
    save_every_n_steps: int = 200
    output_dir: str = "./medical_arabic_gpt2"

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


config = ContinuedPretrainingConfig()

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-27-chapter-27.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class ReplayMixedDataset(Dataset):
    """
    Dataset يطبّق استراتيجية Replay Buffer:
    يمزج بيانات النطاق الجديد (طبي) مع عينة من البيانات العامة الأصلية
    لمنع Catastrophic Forgetting

    🟡 هذه هي آلية العمل الأساسية التي ناقشناها في القسم 27.5
    """

    def __init__(
        self,
        domain_texts: List[str],      # النصوص الطبية (النطاق الجديد)
        general_texts: List[str],      # عينة من بيانات عامة (Replay buffer)
        tokenizer,
        max_length: int = 512,
        replay_ratio: float = 0.15
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # حساب عدد عينات الـ Replay المطلوبة
        n_domain = len(domain_texts)
        n_replay = int(n_domain * replay_ratio / (1 - replay_ratio))

        # أخذ عينة عشوائية من البيانات العامة (مع التكرار إن لزم)
        if n_replay > len(general_texts):
            sampled_general = random.choices(general_texts, k=n_replay)
        else:
            sampled_general = random.sample(general_texts, n_replay)

        # دمج وخلط البيانات
        self.texts = domain_texts + sampled_general
        self.labels = (
            ["domain"] * n_domain + ["replay"] * len(sampled_general)
        )

        # خلط الترتيب (مهم جدًا! بدون هذا، سيرى النموذج كل البيانات
        # الطبية أولًا ثم العامة، مما يُفاقم النسيان قرب نهاية كل Epoch)
        combined = list(zip(self.texts, self.labels))
        random.shuffle(combined)
        self.texts, self.labels = zip(*combined)

        print(f"📊 إحصائيات الـ Dataset:")
        print(f"   - عينات النطاق الطبي: {n_domain}")
        print(f"   - عينات الـ Replay (عامة): {len(sampled_general)}")
        print(f"   - النسبة الفعلية للـ Replay: "
              f"{len(sampled_general)/len(self.texts):.1%}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # في Causal LM، الـ labels هي نفس input_ids
        # (النموذج يتعلم التنبؤ بالتوكن التالي)
        labels = input_ids.clone()
        # تجاهل الـ padding tokens في حساب الخسارة
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "source": self.labels[idx]  # لتتبع الأداء حسب المصدر لاحقًا
        }

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-27-chapter-27.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def prepare_sample_datasets():
    """
    تحضير عينات بيانات للتوضيح:
    - domain_texts: نصوص طبية عربية (محاكاة)
    - general_texts: نصوص عامة عربية (لـ Replay Buffer)
    """

    # عينة من نصوص طبية عربية (في الواقع: آلاف/ملايين الأمثلة)
    domain_texts = [
        "يُعرّف احتشاء عضلة القلب بأنه موت جزء من أنسجة القلب "
        "بسبب انقطاع تدفق الدم الغني بالأكسجين إلى تلك المنطقة.",

        "تشير الدراسات الحديثة إلى أن ارتفاع ضغط الدم المزمن "
        "يُعد عامل خطر رئيسي للإصابة بالسكتة الدماغية والفشل الكلوي.",

        "يتم تشخيص السكري من النوع الثاني عادة من خلال فحص "
        "مستوى السكر التراكمي HbA1c، حيث تشير النسبة فوق 6.5% إلى الإصابة.",

        "العلاج الكيميائي يعمل عن طريق استهداف الخلايا سريعة "
        "الانقسام، مما يفسر الآثار الجانبية المرتبطة بالشعر والجهاز الهضمي.",

        "تعتبر المضادات الحيوية غير فعالة ضد الالتهابات الفيروسية، "
        "ويُعد الاستخدام غير الصحيح لها سببًا رئيسيًا لمقاومة البكتيريا.",

        # ... في مشروع حقيقي: آلاف الأمثلة الإضافية
    ] * 50  # تكرار للمحاكاة فقط (لا تفعل هذا في بيانات حقيقية!)

    # عينة من نصوص عامة عربية (لمنع نسيان القدرات العامة)
    general_texts = [
        "تُعتبر مدينة القاهرة من أقدم المدن المأهولة في العالم، "
        "حيث يعود تاريخها إلى آلاف السنين.",

        "الذكاء الاصطناعي يُحدث تحولًا جذريًا في طريقة تعاملنا "
        "مع التكنولوجيا في مختلف مناحي الحياة اليومية.",

        "يُعد الأدب العربي أحد أغنى الآداب العالمية، بما يحتويه "
        "من شعر ونثر يمتد عبر قرون طويلة من الإبداع.",

        "كرة القدم هي الرياضة الأكثر شعبية في العالم، حيث يتابعها "
        "ملايين المشجعين في كل قارة.",

        "الطهي العربي التقليدي يتميز بتنوعه الكبير، ويعكس تراث "
        "كل منطقة وثقافتها الخاصة بالتوابل والمكونات.",

        # ... في مشروع حقيقي: عينة ممثلة من بيانات Pretraining الأصلية
    ] * 50

    return domain_texts, general_texts

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-27-chapter-27.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
class ContinuedPretrainer:
    """
    مُدرّب متكامل لتطبيق Continued Pretraining
    مع تطبيق كل أفضل الممارسات لمكافحة Catastrophic Forgetting
    """

    def __init__(self, config: ContinuedPretrainingConfig):
        self.config = config

        print(f"🔄 تحميل النموذج الأساسي: {config.base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name
        ).to(device)

        # 🎯 خزن نسخة من المعاملات الأصلية لقياس "الانحراف" لاحقًا
        self.original_state_dict = {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }

        self.training_history = {
            "step": [], "loss": [], "domain_loss": [],
            "replay_loss": [], "param_drift": []
        }

    def compute_param_drift(self) -> float:
        """
        🔴 صندوق تعمّق: قياس مدى ابتعاد المعاملات الحالية
        عن المعاملات الأصلية (مؤشر مباشر على خطر النسيان الكارثي)

        نستخدم L2 norm للفرق بين المعاملات الحالية والأصلية،
        مُطبّعة بعدد المعاملات لتكون قابلة للمقارنة عبر النماذج
        """
        total_drift = 0.0
        total_params = 0

        current_state = self.model.state_dict()
        for key in self.original_state_dict:
            diff = (current_state[key] - self.original_state_dict[key])
            total_drift += torch.sum(diff ** 2).item()
            total_params += diff.numel()

        return math.sqrt(total_drift / total_params)

    def evaluate_forgetting(self, general_eval_texts: List[str]) -> float:
        """
        قياس Perplexity على عينة من البيانات العامة
        للتحقق من مدى حدوث النسيان الكارثي
        (نفس المفهوم من الفصل 4: Perplexity كمقياس للتعميم)
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for text in general_eval_texts:
                inputs = self.tokenizer(
                    text, return_tensors="pt",
                    truncation=True, max_length=512
                ).to(device)

                outputs = self.model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

        self.model.train()
        return perplexity

    def train(self, dataset: ReplayMixedDataset, eval_texts: List[str]):
        """حلقة التدريب الرئيسية الكاملة"""

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        total_steps = (
            len(dataloader) * self.config.num_epochs
            // self.config.gradient_accumulation_steps
        )
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        # 💡 استخدام AdamW مع LR منخفض جدًا - أساسي لمنع الانحراف الكبير
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Perplexity الأولي (خط الأساس قبل أي تدريب)
        baseline_ppl = self.evaluate_forgetting(eval_texts)
        print(f"\n📏 Perplexity الأساسي (قبل التدريب): {baseline_ppl:.2f}\n")

        self.model.train()
        global_step = 0

        print("🚀 بدء Continued Pretraining...\n")
        print("=" * 70)

        for epoch in range(self.config.num_epochs):
            domain_losses, replay_losses = [], []

            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                sources = batch["source"]

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss / self.config.gradient_accumulation_steps
                loss.backward()

                # تتبع الخسارة منفصلة حسب المصدر (طبي vs عام)
                for i, src in enumerate(sources):
                    item_loss = outputs.loss.item()
                    if src == "domain":
                        domain_losses.append(item_loss)
                    else:
                        replay_losses.append(item_loss)

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # ⚠️ Gradient Clipping - أساسي لمنع القفزات الكبيرة
                    # التي قد تسبب نسيانًا سريعًا ومفاجئًا
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % self.config.eval_every_n_steps == 0:
                        drift = self.compute_param_drift()
                        current_ppl = self.evaluate_forgetting(eval_texts)

                        avg_domain = np.mean(domain_losses[-50:]) if domain_losses else 0
                        avg_replay = np.mean(replay_losses[-50:]) if replay_losses else 0

                        print(
                            f"⚡ الخطوة {global_step:4d} | "
                            f"Loss(طبي)={avg_domain:.3f} | "
                            f"Loss(عام)={avg_replay:.3f} | "
                            f"Drift={drift:.5f} | "
                            f"PPL(عام)={current_ppl:.2f} "
                            f"({'📈' if current_ppl > baseline_ppl * 1.1 else '✅'})"
                        )

                        self.training_history["step"].append(global_step)
                        self.training_history["loss"].append(outputs.loss.item())
                        self.training_history["param_drift"].append(drift)

                        # ⚠️ تحذير تلقائي عند اكتشاف نسيان كارثي محتمل
                        if current_ppl > baseline_ppl * 1.5:
                            print(
                                "   🚨 تحذير: ارتفاع كبير في Perplexity على "
                                "البيانات العامة! قد يكون هناك نسيان كارثي. "
                                "فكّر في تقليل LR أو زيادة replay_ratio."
                            )

            print(f"\n✅ انتهت Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"   متوسط Loss طبي: {np.mean(domain_losses):.4f}")
            print(f"   متوسط Loss عام: {np.mean(replay_losses):.4f}")
            print("-" * 70)

        print("\n" + "=" * 70)
        print("🎉 اكتمل Continued Pretraining بنجاح!")
        final_ppl = self.evaluate_forgetting(eval_texts)
        print(f"📊 Perplexity النهائي على بيانات عامة: {final_ppl:.2f} "
              f"(الأساسي كان: {baseline_ppl:.2f})")

        if final_ppl < baseline_ppl * 1.15:
            print("✅ ممتاز! النموذج احتفظ بقدراته العامة بشكل جيد جدًا")
        else:
            print("⚠️  يُنصح بمراجعة استراتيجية مكافحة النسيان الكارثي")

    def save_model(self):
        """حفظ النموذج والـ Tokenizer"""
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        print(f"💾 تم حفظ النموذج في: {self.config.output_dir}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-27-chapter-27.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
def main():
    """نقطة الدخول الرئيسية - تشغيل التجربة كاملة"""

    # 1. تحضير البيانات
    print("📂 تحضير البيانات...")
    domain_texts, general_texts = prepare_sample_datasets()

    # تقسيم بيانات عامة إضافية للتقييم (منفصلة عن التدريب)
    eval_general_texts = general_texts[:20]
    train_general_texts = general_texts[20:]

    # 2. تهيئة المُدرّب
    pretrainer = ContinuedPretrainer(config)

    # 3. بناء Dataset مع استراتيجية Replay
    train_dataset = ReplayMixedDataset(
        domain_texts=domain_texts,
        general_texts=train_general_texts,
        tokenizer=pretrainer.tokenizer,
        max_length=config.max_seq_length,
        replay_ratio=config.replay_ratio
    )

    # 4. تشغيل التدريب
    pretrainer.train(train_dataset, eval_general_texts)

    # 5. حفظ النموذج النهائي
    pretrainer.save_model()

    # 6. اختبار نوعي سريع - مقارنة قبل وبعد
    print("\n" + "=" * 70)
    print("🧪 اختبار نوعي: توليد نص طبي")
    print("=" * 70)

    test_prompt = "العلاج الأمثل لارتفاع ضغط الدم يتضمن"
    inputs = pretrainer.tokenizer(test_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = pretrainer.model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=pretrainer.tokenizer.eos_token_id
        )

    generated_text = pretrainer.tokenizer.decode(
        output[0], skip_special_tokens=True
    )
    print(f"📝 النص المُولّد:\n{generated_text}")


if __name__ == "__main__":
    main()
