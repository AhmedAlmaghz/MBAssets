#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — التدريب المسبق المستمر: CPT
Mastering Transformers and Large Language Models — Continued Pretraining

Book slug : mastering-llms-transformers
Chapter   : ch-24-ch24-continued-pretraining (lesson 24)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-24-ch24-continued-pretraining.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — التدريب المسبق المستمر: CPT
Mastering Transformers and Large Language Models — Continued Pretraining

Book slug : mastering-llms-transformers
Chapter   : ch-24-ch24-continued-pretraining (lesson 24)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-24-ch24-continued-pretraining.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — التدريب المسبق المستمر: CPT
Mastering Transformers and Large Language Models — Continued Pretraining

Book slug : mastering-llms-transformers
Chapter   : ch-24-ch24-continued-pretraining (lesson 24)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-06-model-reuse/ch24-continued-pretraining.ipynb#cell-1
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
    ("torch", "torch"),
    ("transformers", "transformers")
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
# [002] content/notebooks/part-06-model-reuse/ch24-continued-pretraining.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

class ContinuedPretrainingPipeline:
    """
    خط أنابيب كامل للتدريب المسبق المستمر (CPT) على نموذج أساسي موجود،
    مع إدارة دقيقة لمعدل التعلّم (LR Re-warming) ومزج البيانات لتفادي
    النسيان الكارثي (القسم 3.3 أدناه).
    """
    def __init__(self, base_model_name: str, domain_corpus_path: str,
                 general_corpus_path: str = None, device: str = "cuda"):
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.device = device
        self.domain_corpus_path = domain_corpus_path
        self.general_corpus_path = general_corpus_path  # لمزج "تذكير" (replay)

    def build_mixed_dataset(self, domain_replay_ratio: float = 0.7):
        """
        بناء مجموعة بيانات مُمزوجة: نسبة كبيرة من بيانات المجال الجديد
        + نسبة أصغر من بيانات عامة "تذكيرية" (Replay Buffer) لتثبيت
        القدرات العامة الأصلية ومنع نسيانها (راجع القسم 3.3).
        """
        domain_texts = self._load_corpus(self.domain_corpus_path)
        general_texts = (
            self._load_corpus(self.general_corpus_path)
            if self.general_corpus_path else []
        )

        n_domain = int(len(domain_texts))
        n_general_needed = int(
            n_domain * (1 - domain_replay_ratio) / domain_replay_ratio
        )
        general_sample = general_texts[:n_general_needed] if general_texts else []

        mixed = domain_texts + general_sample
        import random
        random.shuffle(mixed)  # خلط عشوائي - مهم جدًا لمنع تجميع نوع واحد
        return mixed

    def _load_corpus(self, path):
        with open(path, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def get_cpt_optimizer_and_scheduler(self, train_steps: int,
                                         peak_lr: float = 5e-6,
                                         warmup_ratio: float = 0.03):
        """
        نقطة حاسمة في CPT: معدل التعلّم الذروي (peak_lr) يجب أن يكون
        أصغر بكثير من peak_lr الأصلي للتدريب المسبق الأول (عادة أقل
        بـ 5-10 مرات)، لأن النموذج بالفعل في منطقة جيدة من فضاء
        الأوزان ولا نريد "تدميرها" بتحديثات كبيرة. يُنصح أيضًا
        بـ warmup قصير (LR Re-warming) بدل البدء المباشر من peak_lr.
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=peak_lr, weight_decay=0.01,
            betas=(0.9, 0.95)  # betas معدّلة شائعة في تدريب LLM الحديث
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(train_steps * warmup_ratio),
            num_training_steps=train_steps,
        )
        return optimizer, scheduler

    def train(self, mixed_dataset, num_epochs: int = 1, batch_size: int = 4,
               max_seq_len: int = 2048, gradient_accumulation_steps: int = 8):
        """حلقة تدريب CPT كاملة مع كل الاحتياطات اللازمة."""
        tokenized = [
            self.tokenizer(t, truncation=True, max_length=max_seq_len,
                            return_tensors="pt")["input_ids"][0]
            for t in mixed_dataset
        ]
        total_steps = (len(tokenized) // batch_size) * num_epochs
        optimizer, scheduler = self.get_cpt_optimizer_and_scheduler(total_steps)

        self.model.train()
        step = 0
        for epoch in range(num_epochs):
            for i in range(0, len(tokenized) - batch_size, batch_size):
                batch = tokenized[i:i + batch_size]
                batch_padded = torch.nn.utils.rnn.pad_sequence(
                    batch, batch_first=True, padding_value=self.tokenizer.pad_token_id
                ).to(self.device)

                outputs = self.model(input_ids=batch_padded, labels=batch_padded)
                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if step % 50 == 0:
                    print(f"خطوة {step}: الخسارة = {outputs.loss.item():.4f}, "
                          f"LR = {scheduler.get_last_lr()[0]:.2e}")
                step += 1

        return self.model

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-06-model-reuse/ch24-continued-pretraining.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def measure_catastrophic_forgetting(
    original_model, cpt_model, general_benchmark_suite: dict,
    domain_benchmark_suite: dict,
) -> dict:
    """
    قياس كمي لمدى النسيان الكارثي: مقارنة أداء النموذج قبل وبعد CPT
    على مجموعتين من المعايير: عامة (يجب ألا تنخفض كثيرًا) ومجال
    مُتخصِّص (يجب أن ترتفع بوضوح - وإلا فإن CPT لم يحقق هدفه أصلًا).
    """
    results = {"general": {}, "domain": {}}

    for benchmark_name, benchmark_data in general_benchmark_suite.items():
        original_score = evaluate_on_benchmark(original_model, benchmark_data)
        cpt_score = evaluate_on_benchmark(cpt_model, benchmark_data)
        forgetting = original_score - cpt_score  # رقم موجب = نسيان حقيقي
        results["general"][benchmark_name] = {
            "before": original_score, "after": cpt_score,
            "forgetting": round(forgetting, 4),
            "status": "⚠️ نسيان ملحوظ" if forgetting > 0.05 else "✅ مستقر",
        }

    for benchmark_name, benchmark_data in domain_benchmark_suite.items():
        original_score = evaluate_on_benchmark(original_model, benchmark_data)
        cpt_score = evaluate_on_benchmark(cpt_model, benchmark_data)
        improvement = cpt_score - original_score  # رقم موجب = تحسّن مرغوب
        results["domain"][benchmark_name] = {
            "before": original_score, "after": cpt_score,
            "improvement": round(improvement, 4),
            "status": "✅ تحسّن جيد" if improvement > 0.05 else "⚠️ تحسّن ضعيف",
        }

    return results

def evaluate_on_benchmark(model, benchmark_data) -> float:
    # دالة تقييم مرجعية - راجع الجزء التاسع (فصل 41-42) للتفاصيل الكاملة
    raise NotImplementedError("يُطبَّق باستخدام أدوات التقييم من الفصل 41-42")

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-06-model-reuse/ch24-continued-pretraining.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class ElasticWeightConsolidation:
    """
    تطبيق EWC لتقليل النسيان الكارثي: نُقدِّر "أهمية" كل وزن بالنسبة
    للمهمة الأصلية عبر مصفوفة معلومات فيشر (تقريب قطري)، ثم نعاقب
    أي انحراف كبير عن القيمة الأصلية للأوزان المهمة أثناء CPT.
    """
    def __init__(self, model, ewc_lambda: float = 1000.0):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.fisher_info = {}
        self.original_params = {}

    def compute_fisher_information(self, general_dataloader):
        """تقدير مصفوفة فيشر القطرية باستخدام بيانات المهمة الأصلية."""
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()
                   if p.requires_grad}

        for batch in general_dataloader:
            self.model.zero_grad()
            outputs = self.model(**batch, labels=batch["input_ids"])
            outputs.loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data ** 2 / len(general_dataloader)

        self.fisher_info = fisher
        self.original_params = {
            n: p.clone().detach() for n, p in self.model.named_parameters()
        }

    def ewc_penalty(self) -> torch.Tensor:
        """حساب عقوبة EWC لإضافتها لخسارة CPT الأساسية."""
        penalty = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for n, p in self.model.named_parameters():
            if n in self.fisher_info:
                penalty += (
                    self.fisher_info[n] * (p - self.original_params[n]) ** 2
                ).sum()
        return self.ewc_lambda * penalty

    def combined_loss(self, cpt_task_loss: torch.Tensor) -> torch.Tensor:
        return cpt_task_loss + self.ewc_penalty()

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-06-model-reuse/ch24-continued-pretraining.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# الخطوة 1: تجهيز بيانات المجال (نصوص طبية عربية خام - أبحاث، كتب، تقارير)
pipeline = ContinuedPretrainingPipeline(
    base_model_name="meta-llama/Meta-Llama-3-8B",
    domain_corpus_path="arabic_medical_corpus.txt",  # ~2 مليار رمز
    general_corpus_path="arabic_general_corpus_sample.txt",  # للـ Replay
)

# الخطوة 2: بناء مجموعة بيانات ممزوجة (80% طبي، 20% عام للتذكير)
mixed_data = pipeline.build_mixed_dataset(domain_replay_ratio=0.8)

# الخطوة 3: التدريب بمعدل تعلّم محافظ جدًا وتمرير واحد فقط
trained_model = pipeline.train(
    mixed_dataset=mixed_data, num_epochs=1, batch_size=4,
    max_seq_len=4096, gradient_accumulation_steps=16,
)

# الخطوة 4: قياس النسيان الكارثي قبل المتابعة لمرحلة SFT
forgetting_report = measure_catastrophic_forgetting(
    original_model=pipeline.model,  # يُفترض نسخة أصلية محفوظة مسبقًا
    cpt_model=trained_model,
    general_benchmark_suite={"MMLU_general": ..., "HellaSwag": ...},
    domain_benchmark_suite={"Arabic_Medical_QA": ..., "MedMCQA_Arabic": ...},
)
# توقع: ارتفاع واضح في Arabic_Medical_QA، انخفاض طفيف جدًا (<3%) في MMLU_general

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-06-model-reuse/ch24-continued-pretraining.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-24-ch24-continued-pretraining.ipynb#cell-1
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
    ("torch", "torch"),
    ("transformers", "transformers")
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-24-ch24-continued-pretraining.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

class ContinuedPretrainingPipeline:
    """
    خط أنابيب كامل للتدريب المسبق المستمر (CPT) على نموذج أساسي موجود،
    مع إدارة دقيقة لمعدل التعلّم (LR Re-warming) ومزج البيانات لتفادي
    النسيان الكارثي (القسم 3.3 أدناه).
    """
    def __init__(self, base_model_name: str, domain_corpus_path: str,
                 general_corpus_path: str = None, device: str = "cuda"):
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.device = device
        self.domain_corpus_path = domain_corpus_path
        self.general_corpus_path = general_corpus_path  # لمزج "تذكير" (replay)

    def build_mixed_dataset(self, domain_replay_ratio: float = 0.7):
        """
        بناء مجموعة بيانات مُمزوجة: نسبة كبيرة من بيانات المجال الجديد
        + نسبة أصغر من بيانات عامة "تذكيرية" (Replay Buffer) لتثبيت
        القدرات العامة الأصلية ومنع نسيانها (راجع القسم 3.3).
        """
        domain_texts = self._load_corpus(self.domain_corpus_path)
        general_texts = (
            self._load_corpus(self.general_corpus_path)
            if self.general_corpus_path else []
        )

        n_domain = int(len(domain_texts))
        n_general_needed = int(
            n_domain * (1 - domain_replay_ratio) / domain_replay_ratio
        )
        general_sample = general_texts[:n_general_needed] if general_texts else []

        mixed = domain_texts + general_sample
        import random
        random.shuffle(mixed)  # خلط عشوائي - مهم جدًا لمنع تجميع نوع واحد
        return mixed

    def _load_corpus(self, path):
        with open(path, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def get_cpt_optimizer_and_scheduler(self, train_steps: int,
                                         peak_lr: float = 5e-6,
                                         warmup_ratio: float = 0.03):
        """
        نقطة حاسمة في CPT: معدل التعلّم الذروي (peak_lr) يجب أن يكون
        أصغر بكثير من peak_lr الأصلي للتدريب المسبق الأول (عادة أقل
        بـ 5-10 مرات)، لأن النموذج بالفعل في منطقة جيدة من فضاء
        الأوزان ولا نريد "تدميرها" بتحديثات كبيرة. يُنصح أيضًا
        بـ warmup قصير (LR Re-warming) بدل البدء المباشر من peak_lr.
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=peak_lr, weight_decay=0.01,
            betas=(0.9, 0.95)  # betas معدّلة شائعة في تدريب LLM الحديث
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(train_steps * warmup_ratio),
            num_training_steps=train_steps,
        )
        return optimizer, scheduler

    def train(self, mixed_dataset, num_epochs: int = 1, batch_size: int = 4,
               max_seq_len: int = 2048, gradient_accumulation_steps: int = 8):
        """حلقة تدريب CPT كاملة مع كل الاحتياطات اللازمة."""
        tokenized = [
            self.tokenizer(t, truncation=True, max_length=max_seq_len,
                            return_tensors="pt")["input_ids"][0]
            for t in mixed_dataset
        ]
        total_steps = (len(tokenized) // batch_size) * num_epochs
        optimizer, scheduler = self.get_cpt_optimizer_and_scheduler(total_steps)

        self.model.train()
        step = 0
        for epoch in range(num_epochs):
            for i in range(0, len(tokenized) - batch_size, batch_size):
                batch = tokenized[i:i + batch_size]
                batch_padded = torch.nn.utils.rnn.pad_sequence(
                    batch, batch_first=True, padding_value=self.tokenizer.pad_token_id
                ).to(self.device)

                outputs = self.model(input_ids=batch_padded, labels=batch_padded)
                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if step % 50 == 0:
                    print(f"خطوة {step}: الخسارة = {outputs.loss.item():.4f}, "
                          f"LR = {scheduler.get_last_lr()[0]:.2e}")
                step += 1

        return self.model

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-24-ch24-continued-pretraining.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def measure_catastrophic_forgetting(
    original_model, cpt_model, general_benchmark_suite: dict,
    domain_benchmark_suite: dict,
) -> dict:
    """
    قياس كمي لمدى النسيان الكارثي: مقارنة أداء النموذج قبل وبعد CPT
    على مجموعتين من المعايير: عامة (يجب ألا تنخفض كثيرًا) ومجال
    مُتخصِّص (يجب أن ترتفع بوضوح - وإلا فإن CPT لم يحقق هدفه أصلًا).
    """
    results = {"general": {}, "domain": {}}

    for benchmark_name, benchmark_data in general_benchmark_suite.items():
        original_score = evaluate_on_benchmark(original_model, benchmark_data)
        cpt_score = evaluate_on_benchmark(cpt_model, benchmark_data)
        forgetting = original_score - cpt_score  # رقم موجب = نسيان حقيقي
        results["general"][benchmark_name] = {
            "before": original_score, "after": cpt_score,
            "forgetting": round(forgetting, 4),
            "status": "⚠️ نسيان ملحوظ" if forgetting > 0.05 else "✅ مستقر",
        }

    for benchmark_name, benchmark_data in domain_benchmark_suite.items():
        original_score = evaluate_on_benchmark(original_model, benchmark_data)
        cpt_score = evaluate_on_benchmark(cpt_model, benchmark_data)
        improvement = cpt_score - original_score  # رقم موجب = تحسّن مرغوب
        results["domain"][benchmark_name] = {
            "before": original_score, "after": cpt_score,
            "improvement": round(improvement, 4),
            "status": "✅ تحسّن جيد" if improvement > 0.05 else "⚠️ تحسّن ضعيف",
        }

    return results

def evaluate_on_benchmark(model, benchmark_data) -> float:
    # دالة تقييم مرجعية - راجع الجزء التاسع (فصل 41-42) للتفاصيل الكاملة
    raise NotImplementedError("يُطبَّق باستخدام أدوات التقييم من الفصل 41-42")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-24-ch24-continued-pretraining.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class ElasticWeightConsolidation:
    """
    تطبيق EWC لتقليل النسيان الكارثي: نُقدِّر "أهمية" كل وزن بالنسبة
    للمهمة الأصلية عبر مصفوفة معلومات فيشر (تقريب قطري)، ثم نعاقب
    أي انحراف كبير عن القيمة الأصلية للأوزان المهمة أثناء CPT.
    """
    def __init__(self, model, ewc_lambda: float = 1000.0):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.fisher_info = {}
        self.original_params = {}

    def compute_fisher_information(self, general_dataloader):
        """تقدير مصفوفة فيشر القطرية باستخدام بيانات المهمة الأصلية."""
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()
                   if p.requires_grad}

        for batch in general_dataloader:
            self.model.zero_grad()
            outputs = self.model(**batch, labels=batch["input_ids"])
            outputs.loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data ** 2 / len(general_dataloader)

        self.fisher_info = fisher
        self.original_params = {
            n: p.clone().detach() for n, p in self.model.named_parameters()
        }

    def ewc_penalty(self) -> torch.Tensor:
        """حساب عقوبة EWC لإضافتها لخسارة CPT الأساسية."""
        penalty = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for n, p in self.model.named_parameters():
            if n in self.fisher_info:
                penalty += (
                    self.fisher_info[n] * (p - self.original_params[n]) ** 2
                ).sum()
        return self.ewc_lambda * penalty

    def combined_loss(self, cpt_task_loss: torch.Tensor) -> torch.Tensor:
        return cpt_task_loss + self.ewc_penalty()

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-24-ch24-continued-pretraining.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# الخطوة 1: تجهيز بيانات المجال (نصوص طبية عربية خام - أبحاث، كتب، تقارير)
pipeline = ContinuedPretrainingPipeline(
    base_model_name="meta-llama/Meta-Llama-3-8B",
    domain_corpus_path="arabic_medical_corpus.txt",  # ~2 مليار رمز
    general_corpus_path="arabic_general_corpus_sample.txt",  # للـ Replay
)

# الخطوة 2: بناء مجموعة بيانات ممزوجة (80% طبي، 20% عام للتذكير)
mixed_data = pipeline.build_mixed_dataset(domain_replay_ratio=0.8)

# الخطوة 3: التدريب بمعدل تعلّم محافظ جدًا وتمرير واحد فقط
trained_model = pipeline.train(
    mixed_dataset=mixed_data, num_epochs=1, batch_size=4,
    max_seq_len=4096, gradient_accumulation_steps=16,
)

# الخطوة 4: قياس النسيان الكارثي قبل المتابعة لمرحلة SFT
forgetting_report = measure_catastrophic_forgetting(
    original_model=pipeline.model,  # يُفترض نسخة أصلية محفوظة مسبقًا
    cpt_model=trained_model,
    general_benchmark_suite={"MMLU_general": ..., "HellaSwag": ...},
    domain_benchmark_suite={"Arabic_Medical_QA": ..., "MedMCQA_Arabic": ...},
)
# توقع: ارتفاع واضح في Arabic_Medical_QA، انخفاض طفيف جدًا (<3%) في MMLU_general

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-24-ch24-continued-pretraining.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-24-ch24-continued-pretraining.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — التدريب المسبق المستمر: CPT
Mastering Transformers and Large Language Models — Continued Pretraining

Book slug : mastering-llms-transformers
Chapter   : ch-24-ch24-continued-pretraining (lesson 24)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-06-model-reuse/ch24-continued-pretraining.ipynb#cell-1
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
    ("torch", "torch"),
    ("transformers", "transformers")
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
# [002] content/notebooks/part-06-model-reuse/ch24-continued-pretraining.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

class ContinuedPretrainingPipeline:
    """
    خط أنابيب كامل للتدريب المسبق المستمر (CPT) على نموذج أساسي موجود،
    مع إدارة دقيقة لمعدل التعلّم (LR Re-warming) ومزج البيانات لتفادي
    النسيان الكارثي (القسم 3.3 أدناه).
    """
    def __init__(self, base_model_name: str, domain_corpus_path: str,
                 general_corpus_path: str = None, device: str = "cuda"):
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.device = device
        self.domain_corpus_path = domain_corpus_path
        self.general_corpus_path = general_corpus_path  # لمزج "تذكير" (replay)

    def build_mixed_dataset(self, domain_replay_ratio: float = 0.7):
        """
        بناء مجموعة بيانات مُمزوجة: نسبة كبيرة من بيانات المجال الجديد
        + نسبة أصغر من بيانات عامة "تذكيرية" (Replay Buffer) لتثبيت
        القدرات العامة الأصلية ومنع نسيانها (راجع القسم 3.3).
        """
        domain_texts = self._load_corpus(self.domain_corpus_path)
        general_texts = (
            self._load_corpus(self.general_corpus_path)
            if self.general_corpus_path else []
        )

        n_domain = int(len(domain_texts))
        n_general_needed = int(
            n_domain * (1 - domain_replay_ratio) / domain_replay_ratio
        )
        general_sample = general_texts[:n_general_needed] if general_texts else []

        mixed = domain_texts + general_sample
        import random
        random.shuffle(mixed)  # خلط عشوائي - مهم جدًا لمنع تجميع نوع واحد
        return mixed

    def _load_corpus(self, path):
        with open(path, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def get_cpt_optimizer_and_scheduler(self, train_steps: int,
                                         peak_lr: float = 5e-6,
                                         warmup_ratio: float = 0.03):
        """
        نقطة حاسمة في CPT: معدل التعلّم الذروي (peak_lr) يجب أن يكون
        أصغر بكثير من peak_lr الأصلي للتدريب المسبق الأول (عادة أقل
        بـ 5-10 مرات)، لأن النموذج بالفعل في منطقة جيدة من فضاء
        الأوزان ولا نريد "تدميرها" بتحديثات كبيرة. يُنصح أيضًا
        بـ warmup قصير (LR Re-warming) بدل البدء المباشر من peak_lr.
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=peak_lr, weight_decay=0.01,
            betas=(0.9, 0.95)  # betas معدّلة شائعة في تدريب LLM الحديث
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(train_steps * warmup_ratio),
            num_training_steps=train_steps,
        )
        return optimizer, scheduler

    def train(self, mixed_dataset, num_epochs: int = 1, batch_size: int = 4,
               max_seq_len: int = 2048, gradient_accumulation_steps: int = 8):
        """حلقة تدريب CPT كاملة مع كل الاحتياطات اللازمة."""
        tokenized = [
            self.tokenizer(t, truncation=True, max_length=max_seq_len,
                            return_tensors="pt")["input_ids"][0]
            for t in mixed_dataset
        ]
        total_steps = (len(tokenized) // batch_size) * num_epochs
        optimizer, scheduler = self.get_cpt_optimizer_and_scheduler(total_steps)

        self.model.train()
        step = 0
        for epoch in range(num_epochs):
            for i in range(0, len(tokenized) - batch_size, batch_size):
                batch = tokenized[i:i + batch_size]
                batch_padded = torch.nn.utils.rnn.pad_sequence(
                    batch, batch_first=True, padding_value=self.tokenizer.pad_token_id
                ).to(self.device)

                outputs = self.model(input_ids=batch_padded, labels=batch_padded)
                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if step % 50 == 0:
                    print(f"خطوة {step}: الخسارة = {outputs.loss.item():.4f}, "
                          f"LR = {scheduler.get_last_lr()[0]:.2e}")
                step += 1

        return self.model

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-06-model-reuse/ch24-continued-pretraining.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def measure_catastrophic_forgetting(
    original_model, cpt_model, general_benchmark_suite: dict,
    domain_benchmark_suite: dict,
) -> dict:
    """
    قياس كمي لمدى النسيان الكارثي: مقارنة أداء النموذج قبل وبعد CPT
    على مجموعتين من المعايير: عامة (يجب ألا تنخفض كثيرًا) ومجال
    مُتخصِّص (يجب أن ترتفع بوضوح - وإلا فإن CPT لم يحقق هدفه أصلًا).
    """
    results = {"general": {}, "domain": {}}

    for benchmark_name, benchmark_data in general_benchmark_suite.items():
        original_score = evaluate_on_benchmark(original_model, benchmark_data)
        cpt_score = evaluate_on_benchmark(cpt_model, benchmark_data)
        forgetting = original_score - cpt_score  # رقم موجب = نسيان حقيقي
        results["general"][benchmark_name] = {
            "before": original_score, "after": cpt_score,
            "forgetting": round(forgetting, 4),
            "status": "⚠️ نسيان ملحوظ" if forgetting > 0.05 else "✅ مستقر",
        }

    for benchmark_name, benchmark_data in domain_benchmark_suite.items():
        original_score = evaluate_on_benchmark(original_model, benchmark_data)
        cpt_score = evaluate_on_benchmark(cpt_model, benchmark_data)
        improvement = cpt_score - original_score  # رقم موجب = تحسّن مرغوب
        results["domain"][benchmark_name] = {
            "before": original_score, "after": cpt_score,
            "improvement": round(improvement, 4),
            "status": "✅ تحسّن جيد" if improvement > 0.05 else "⚠️ تحسّن ضعيف",
        }

    return results

def evaluate_on_benchmark(model, benchmark_data) -> float:
    # دالة تقييم مرجعية - راجع الجزء التاسع (فصل 41-42) للتفاصيل الكاملة
    raise NotImplementedError("يُطبَّق باستخدام أدوات التقييم من الفصل 41-42")

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-06-model-reuse/ch24-continued-pretraining.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class ElasticWeightConsolidation:
    """
    تطبيق EWC لتقليل النسيان الكارثي: نُقدِّر "أهمية" كل وزن بالنسبة
    للمهمة الأصلية عبر مصفوفة معلومات فيشر (تقريب قطري)، ثم نعاقب
    أي انحراف كبير عن القيمة الأصلية للأوزان المهمة أثناء CPT.
    """
    def __init__(self, model, ewc_lambda: float = 1000.0):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.fisher_info = {}
        self.original_params = {}

    def compute_fisher_information(self, general_dataloader):
        """تقدير مصفوفة فيشر القطرية باستخدام بيانات المهمة الأصلية."""
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()
                   if p.requires_grad}

        for batch in general_dataloader:
            self.model.zero_grad()
            outputs = self.model(**batch, labels=batch["input_ids"])
            outputs.loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data ** 2 / len(general_dataloader)

        self.fisher_info = fisher
        self.original_params = {
            n: p.clone().detach() for n, p in self.model.named_parameters()
        }

    def ewc_penalty(self) -> torch.Tensor:
        """حساب عقوبة EWC لإضافتها لخسارة CPT الأساسية."""
        penalty = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for n, p in self.model.named_parameters():
            if n in self.fisher_info:
                penalty += (
                    self.fisher_info[n] * (p - self.original_params[n]) ** 2
                ).sum()
        return self.ewc_lambda * penalty

    def combined_loss(self, cpt_task_loss: torch.Tensor) -> torch.Tensor:
        return cpt_task_loss + self.ewc_penalty()

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-06-model-reuse/ch24-continued-pretraining.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# الخطوة 1: تجهيز بيانات المجال (نصوص طبية عربية خام - أبحاث، كتب، تقارير)
pipeline = ContinuedPretrainingPipeline(
    base_model_name="meta-llama/Meta-Llama-3-8B",
    domain_corpus_path="arabic_medical_corpus.txt",  # ~2 مليار رمز
    general_corpus_path="arabic_general_corpus_sample.txt",  # للـ Replay
)

# الخطوة 2: بناء مجموعة بيانات ممزوجة (80% طبي، 20% عام للتذكير)
mixed_data = pipeline.build_mixed_dataset(domain_replay_ratio=0.8)

# الخطوة 3: التدريب بمعدل تعلّم محافظ جدًا وتمرير واحد فقط
trained_model = pipeline.train(
    mixed_dataset=mixed_data, num_epochs=1, batch_size=4,
    max_seq_len=4096, gradient_accumulation_steps=16,
)

# الخطوة 4: قياس النسيان الكارثي قبل المتابعة لمرحلة SFT
forgetting_report = measure_catastrophic_forgetting(
    original_model=pipeline.model,  # يُفترض نسخة أصلية محفوظة مسبقًا
    cpt_model=trained_model,
    general_benchmark_suite={"MMLU_general": ..., "HellaSwag": ...},
    domain_benchmark_suite={"Arabic_Medical_QA": ..., "MedMCQA_Arabic": ...},
)
# توقع: ارتفاع واضح في Arabic_Medical_QA، انخفاض طفيف جدًا (<3%) في MMLU_general

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-06-model-reuse/ch24-continued-pretraining.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-24-ch24-continued-pretraining.ipynb#cell-2
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
    ("torch", "torch"),
    ("transformers", "transformers")
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-24-ch24-continued-pretraining.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

class ContinuedPretrainingPipeline:
    """
    خط أنابيب كامل للتدريب المسبق المستمر (CPT) على نموذج أساسي موجود،
    مع إدارة دقيقة لمعدل التعلّم (LR Re-warming) ومزج البيانات لتفادي
    النسيان الكارثي (القسم 3.3 أدناه).
    """
    def __init__(self, base_model_name: str, domain_corpus_path: str,
                 general_corpus_path: str = None, device: str = "cuda"):
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.device = device
        self.domain_corpus_path = domain_corpus_path
        self.general_corpus_path = general_corpus_path  # لمزج "تذكير" (replay)

    def build_mixed_dataset(self, domain_replay_ratio: float = 0.7):
        """
        بناء مجموعة بيانات مُمزوجة: نسبة كبيرة من بيانات المجال الجديد
        + نسبة أصغر من بيانات عامة "تذكيرية" (Replay Buffer) لتثبيت
        القدرات العامة الأصلية ومنع نسيانها (راجع القسم 3.3).
        """
        domain_texts = self._load_corpus(self.domain_corpus_path)
        general_texts = (
            self._load_corpus(self.general_corpus_path)
            if self.general_corpus_path else []
        )

        n_domain = int(len(domain_texts))
        n_general_needed = int(
            n_domain * (1 - domain_replay_ratio) / domain_replay_ratio
        )
        general_sample = general_texts[:n_general_needed] if general_texts else []

        mixed = domain_texts + general_sample
        import random
        random.shuffle(mixed)  # خلط عشوائي - مهم جدًا لمنع تجميع نوع واحد
        return mixed

    def _load_corpus(self, path):
        with open(path, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def get_cpt_optimizer_and_scheduler(self, train_steps: int,
                                         peak_lr: float = 5e-6,
                                         warmup_ratio: float = 0.03):
        """
        نقطة حاسمة في CPT: معدل التعلّم الذروي (peak_lr) يجب أن يكون
        أصغر بكثير من peak_lr الأصلي للتدريب المسبق الأول (عادة أقل
        بـ 5-10 مرات)، لأن النموذج بالفعل في منطقة جيدة من فضاء
        الأوزان ولا نريد "تدميرها" بتحديثات كبيرة. يُنصح أيضًا
        بـ warmup قصير (LR Re-warming) بدل البدء المباشر من peak_lr.
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=peak_lr, weight_decay=0.01,
            betas=(0.9, 0.95)  # betas معدّلة شائعة في تدريب LLM الحديث
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(train_steps * warmup_ratio),
            num_training_steps=train_steps,
        )
        return optimizer, scheduler

    def train(self, mixed_dataset, num_epochs: int = 1, batch_size: int = 4,
               max_seq_len: int = 2048, gradient_accumulation_steps: int = 8):
        """حلقة تدريب CPT كاملة مع كل الاحتياطات اللازمة."""
        tokenized = [
            self.tokenizer(t, truncation=True, max_length=max_seq_len,
                            return_tensors="pt")["input_ids"][0]
            for t in mixed_dataset
        ]
        total_steps = (len(tokenized) // batch_size) * num_epochs
        optimizer, scheduler = self.get_cpt_optimizer_and_scheduler(total_steps)

        self.model.train()
        step = 0
        for epoch in range(num_epochs):
            for i in range(0, len(tokenized) - batch_size, batch_size):
                batch = tokenized[i:i + batch_size]
                batch_padded = torch.nn.utils.rnn.pad_sequence(
                    batch, batch_first=True, padding_value=self.tokenizer.pad_token_id
                ).to(self.device)

                outputs = self.model(input_ids=batch_padded, labels=batch_padded)
                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if step % 50 == 0:
                    print(f"خطوة {step}: الخسارة = {outputs.loss.item():.4f}, "
                          f"LR = {scheduler.get_last_lr()[0]:.2e}")
                step += 1

        return self.model

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-24-ch24-continued-pretraining.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def measure_catastrophic_forgetting(
    original_model, cpt_model, general_benchmark_suite: dict,
    domain_benchmark_suite: dict,
) -> dict:
    """
    قياس كمي لمدى النسيان الكارثي: مقارنة أداء النموذج قبل وبعد CPT
    على مجموعتين من المعايير: عامة (يجب ألا تنخفض كثيرًا) ومجال
    مُتخصِّص (يجب أن ترتفع بوضوح - وإلا فإن CPT لم يحقق هدفه أصلًا).
    """
    results = {"general": {}, "domain": {}}

    for benchmark_name, benchmark_data in general_benchmark_suite.items():
        original_score = evaluate_on_benchmark(original_model, benchmark_data)
        cpt_score = evaluate_on_benchmark(cpt_model, benchmark_data)
        forgetting = original_score - cpt_score  # رقم موجب = نسيان حقيقي
        results["general"][benchmark_name] = {
            "before": original_score, "after": cpt_score,
            "forgetting": round(forgetting, 4),
            "status": "⚠️ نسيان ملحوظ" if forgetting > 0.05 else "✅ مستقر",
        }

    for benchmark_name, benchmark_data in domain_benchmark_suite.items():
        original_score = evaluate_on_benchmark(original_model, benchmark_data)
        cpt_score = evaluate_on_benchmark(cpt_model, benchmark_data)
        improvement = cpt_score - original_score  # رقم موجب = تحسّن مرغوب
        results["domain"][benchmark_name] = {
            "before": original_score, "after": cpt_score,
            "improvement": round(improvement, 4),
            "status": "✅ تحسّن جيد" if improvement > 0.05 else "⚠️ تحسّن ضعيف",
        }

    return results

def evaluate_on_benchmark(model, benchmark_data) -> float:
    # دالة تقييم مرجعية - راجع الجزء التاسع (فصل 41-42) للتفاصيل الكاملة
    raise NotImplementedError("يُطبَّق باستخدام أدوات التقييم من الفصل 41-42")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-24-ch24-continued-pretraining.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class ElasticWeightConsolidation:
    """
    تطبيق EWC لتقليل النسيان الكارثي: نُقدِّر "أهمية" كل وزن بالنسبة
    للمهمة الأصلية عبر مصفوفة معلومات فيشر (تقريب قطري)، ثم نعاقب
    أي انحراف كبير عن القيمة الأصلية للأوزان المهمة أثناء CPT.
    """
    def __init__(self, model, ewc_lambda: float = 1000.0):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.fisher_info = {}
        self.original_params = {}

    def compute_fisher_information(self, general_dataloader):
        """تقدير مصفوفة فيشر القطرية باستخدام بيانات المهمة الأصلية."""
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()
                   if p.requires_grad}

        for batch in general_dataloader:
            self.model.zero_grad()
            outputs = self.model(**batch, labels=batch["input_ids"])
            outputs.loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data ** 2 / len(general_dataloader)

        self.fisher_info = fisher
        self.original_params = {
            n: p.clone().detach() for n, p in self.model.named_parameters()
        }

    def ewc_penalty(self) -> torch.Tensor:
        """حساب عقوبة EWC لإضافتها لخسارة CPT الأساسية."""
        penalty = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for n, p in self.model.named_parameters():
            if n in self.fisher_info:
                penalty += (
                    self.fisher_info[n] * (p - self.original_params[n]) ** 2
                ).sum()
        return self.ewc_lambda * penalty

    def combined_loss(self, cpt_task_loss: torch.Tensor) -> torch.Tensor:
        return cpt_task_loss + self.ewc_penalty()

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-24-ch24-continued-pretraining.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# الخطوة 1: تجهيز بيانات المجال (نصوص طبية عربية خام - أبحاث، كتب، تقارير)
pipeline = ContinuedPretrainingPipeline(
    base_model_name="meta-llama/Meta-Llama-3-8B",
    domain_corpus_path="arabic_medical_corpus.txt",  # ~2 مليار رمز
    general_corpus_path="arabic_general_corpus_sample.txt",  # للـ Replay
)

# الخطوة 2: بناء مجموعة بيانات ممزوجة (80% طبي، 20% عام للتذكير)
mixed_data = pipeline.build_mixed_dataset(domain_replay_ratio=0.8)

# الخطوة 3: التدريب بمعدل تعلّم محافظ جدًا وتمرير واحد فقط
trained_model = pipeline.train(
    mixed_dataset=mixed_data, num_epochs=1, batch_size=4,
    max_seq_len=4096, gradient_accumulation_steps=16,
)

# الخطوة 4: قياس النسيان الكارثي قبل المتابعة لمرحلة SFT
forgetting_report = measure_catastrophic_forgetting(
    original_model=pipeline.model,  # يُفترض نسخة أصلية محفوظة مسبقًا
    cpt_model=trained_model,
    general_benchmark_suite={"MMLU_general": ..., "HellaSwag": ...},
    domain_benchmark_suite={"Arabic_Medical_QA": ..., "MedMCQA_Arabic": ...},
)
# توقع: ارتفاع واضح في Arabic_Medical_QA، انخفاض طفيف جدًا (<3%) في MMLU_general

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-llms-transformers/notebooks/ch-24-ch24-continued-pretraining.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
