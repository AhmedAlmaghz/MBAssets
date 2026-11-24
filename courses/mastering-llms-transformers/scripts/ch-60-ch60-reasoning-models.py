#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — نماذج الاستدلال وTest-Time Compute
Mastering Transformers and Large Language Models — Reasoning Models

Book slug : mastering-llms-transformers
Chapter   : ch-60-ch60-reasoning-models (lesson 60)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-60-ch60-reasoning-models.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — نماذج الاستدلال وTest-Time Compute
Mastering Transformers and Large Language Models — Reasoning Models

Book slug : mastering-llms-transformers
Chapter   : ch-60-ch60-reasoning-models (lesson 60)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-60-ch60-reasoning-models.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — نماذج الاستدلال وTest-Time Compute
Mastering Transformers and Large Language Models — Reasoning Models

Book slug : mastering-llms-transformers
Chapter   : ch-60-ch60-reasoning-models (lesson 60)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-12-future/ch60-reasoning-models.ipynb#cell-1
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
    ("torch", "torch")
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
# [002] content/notebooks/part-12-future/ch60-reasoning-models.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Callable

@dataclass
class GRPOConfig:
    group_size: int = 16          # عدد الاستجابات لكل سؤال
    clip_epsilon: float = 0.2     # قص PPO-style
    kl_coefficient: float = 0.04  # عقوبة الابتعاد عن النموذج المرجعي
    learning_rate: float = 1e-6

class GRPOTrainer:
    """
    تطبيق مبسّط ومفاهيمي لخوارزمية GRPO المستخدمة في تدريب DeepSeek-R1.
    الفكرة الجوهرية: استبدال Value Model بمتوسط مكافآت المجموعة.
    """
    def __init__(self, policy_model, ref_model, reward_fn: Callable,
                 config: GRPOConfig):
        self.policy_model = policy_model
        self.ref_model = ref_model  # نسخة مجمّدة من النموذج قبل RL (لـ KL)
        self.reward_fn = reward_fn  # دالة مكافأة قابلة للتحقق (مثل: صحة الإجابة الرياضية)
        self.config = config
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(), lr=config.learning_rate
        )

    def sample_group(self, prompt_ids, tokenizer, max_new_tokens=2048):
        """توليد G استجابات مختلفة لنفس السؤال (بعينات Sampling عشوائية)."""
        group_size = self.config.group_size
        responses = []
        for _ in range(group_size):
            with torch.no_grad():
                output_ids = self.policy_model.generate(
                    prompt_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                )
            responses.append(output_ids)
        return responses

    def compute_group_advantages(self, rewards: List[float]) -> torch.Tensor:
        """حساب الـ Advantage النسبي داخل المجموعة (بديل Value Model)."""
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        mean_r = rewards_t.mean()
        std_r = rewards_t.std() + 1e-8  # تفادي القسمة على صفر
        advantages = (rewards_t - mean_r) / std_r
        return advantages

    def compute_grpo_loss(self, log_probs_new, log_probs_old,
                           log_probs_ref, advantages):
        """
        حساب خسارة GRPO: PPO-style clipped objective + عقوبة KL صريحة
        (بخلاف PPO التي تدمج KL داخل المكافأة أحيانًا).
        """
        # نسبة الاحتمالات (policy ratio)
        ratio = torch.exp(log_probs_new - log_probs_old)

        # الهدف المقصوص (clipped surrogate objective) كما في PPO
        unclipped = ratio * advantages
        clipped = torch.clamp(
            ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon
        ) * advantages
        policy_loss = -torch.min(unclipped, clipped).mean()

        # عقوبة KL صريحة بين السياسة الحالية والمرجعية (منع الانحراف الكبير)
        kl_penalty = (
            torch.exp(log_probs_ref - log_probs_new)
            - (log_probs_ref - log_probs_new) - 1
        ).mean()  # تقدير KL غير المتحيّز (k3 estimator)

        total_loss = policy_loss + self.config.kl_coefficient * kl_penalty
        return total_loss, policy_loss.item(), kl_penalty.item()

    def train_step(self, prompt_ids, tokenizer, ground_truth_answer):
        """خطوة تدريب واحدة كاملة بمنهجية GRPO."""
        # 1. توليد مجموعة من الاستجابات
        responses = self.sample_group(prompt_ids, tokenizer)

        # 2. حساب المكافأة القابلة للتحقق لكل استجابة
        #    (مثال: 1.0 إذا كانت الإجابة الرياضية صحيحة، 0.0 إذا خاطئة
        #     + مكافأة تنسيق صغيرة إذا استُخدمت وسوم <think></think>)
        rewards = [
            self.reward_fn(resp, ground_truth_answer) for resp in responses
        ]

        # 3. حساب الـ Advantages النسبية (بدون Value Model!)
        advantages = self.compute_group_advantages(rewards)

        # 4. حساب log-probabilities من السياسة الحالية والمرجعية
        log_probs_new = self._compute_log_probs(self.policy_model, responses)
        log_probs_old = log_probs_new.detach()  # من نفس التوليد (on-policy تقريبًا)
        log_probs_ref = self._compute_log_probs(self.ref_model, responses)

        # 5. حساب الخسارة والتحديث
        loss, policy_loss, kl = self.compute_grpo_loss(
            log_probs_new, log_probs_old, log_probs_ref, advantages
        )
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.optimizer.step()

        return {"loss": loss.item(), "policy_loss": policy_loss,
                "kl": kl, "mean_reward": sum(rewards) / len(rewards)}

    def _compute_log_probs(self, model, responses):
        # دالة مساعدة مفاهيمية — تحسب مجموع log-probs للتسلسل بالكامل
        raise NotImplementedError("تُطبَّق بحسب واجهة النموذج المحدد")

def verifiable_math_reward(response_text: str, ground_truth: str) -> float:
    """
    دالة مكافأة قابلة للتحقق (Rule-Based Reward) — جوهر نجاح R1-Zero.
    بخلاف Reward Model مُدرَّب (قد يُخدَع بـ Reward Hacking)، هذه الدالة
    حتمية 100% ولا يمكن التلاعب بها.
    """
    import re
    # استخراج الإجابة النهائية من داخل \boxed{} أو بعد "الإجابة النهائية:"
    match = re.search(r"\\boxed\{([^}]+)\}", response_text)
    if not match:
        match = re.search(r"الإجابة النهائية[:\s]*([^\n]+)", response_text)
    if not match:
        return 0.0
    extracted = match.group(1).strip()

    # مكافأة الدقة
    correctness_reward = 1.0 if extracted == ground_truth.strip() else 0.0

    # مكافأة تنسيق إضافية صغيرة (تشجيع استخدام وسوم التفكير المنظّمة)
    format_reward = 0.1 if "<think>" in response_text and "</think>" in response_text else 0.0

    return correctness_reward + format_reward

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-12-future/ch60-reasoning-models.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from enum import Enum

class TestTimeComputeStrategy(Enum):
    """الاستراتيجيات الأربع الرئيسية لاستغلال حوسبة وقت الاستدلال."""
    SEQUENTIAL_REVISION = "sequential_revision"      # النموذج يراجع إجابته تكراريًا
    PARALLEL_SAMPLING = "parallel_sampling"           # توليد N إجابات واختيار الأفضل
    TREE_SEARCH = "tree_search"                        # بحث شجري موجّه (ToT، MCTS)
    LEARNED_LONG_CHAIN = "learned_long_chain"          # RL على سلاسل تفكير طويلة (o1/R1)

def best_of_n_with_prm(model, prm_scorer, prompt: str, n: int = 32) -> str:
    """
    استراتيجية Best-of-N الموجّهة بـ PRM: وَلِّد N حلول، قيّم كل خطوة بـ PRM،
    اختر الحل ذا أعلى درجة تراكمية. أرخص من RL لكن أكثر تكلفة وقت الاستدلال
    من استدلال واحد مباشر.
    """
    candidates = [model.generate(prompt, temperature=0.8) for _ in range(n)]

    scored_candidates = []
    for candidate in candidates:
        steps = split_into_reasoning_steps(candidate)
        step_scores = [prm_scorer.score_step(prompt, steps[:i+1])
                       for i in range(len(steps))]
        # الدرجة التراكمية = أدنى درجة عبر الخطوات (أضعف حلقة في السلسلة)
        # أو حاصل الضرب — كلا الاستراتيجيتين مُستخدم في الأدبيات
        aggregate_score = min(step_scores) if step_scores else 0.0
        scored_candidates.append((candidate, aggregate_score))

    best_candidate = max(scored_candidates, key=lambda x: x[1])
    return best_candidate[0]

def split_into_reasoning_steps(text: str) -> list:
    """تقسيم نص التفكير إلى خطوات منفصلة (بناءً على فواصل الأسطر/الترقيم)."""
    import re
    steps = re.split(r'\n(?=\d+[\.\)]|الخطوة)', text)
    return [s.strip() for s in steps if s.strip()]

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-12-future/ch60-reasoning-models.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-60-ch60-reasoning-models.ipynb#cell-1
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
    ("torch", "torch")
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-60-ch60-reasoning-models.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Callable

@dataclass
class GRPOConfig:
    group_size: int = 16          # عدد الاستجابات لكل سؤال
    clip_epsilon: float = 0.2     # قص PPO-style
    kl_coefficient: float = 0.04  # عقوبة الابتعاد عن النموذج المرجعي
    learning_rate: float = 1e-6

class GRPOTrainer:
    """
    تطبيق مبسّط ومفاهيمي لخوارزمية GRPO المستخدمة في تدريب DeepSeek-R1.
    الفكرة الجوهرية: استبدال Value Model بمتوسط مكافآت المجموعة.
    """
    def __init__(self, policy_model, ref_model, reward_fn: Callable,
                 config: GRPOConfig):
        self.policy_model = policy_model
        self.ref_model = ref_model  # نسخة مجمّدة من النموذج قبل RL (لـ KL)
        self.reward_fn = reward_fn  # دالة مكافأة قابلة للتحقق (مثل: صحة الإجابة الرياضية)
        self.config = config
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(), lr=config.learning_rate
        )

    def sample_group(self, prompt_ids, tokenizer, max_new_tokens=2048):
        """توليد G استجابات مختلفة لنفس السؤال (بعينات Sampling عشوائية)."""
        group_size = self.config.group_size
        responses = []
        for _ in range(group_size):
            with torch.no_grad():
                output_ids = self.policy_model.generate(
                    prompt_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                )
            responses.append(output_ids)
        return responses

    def compute_group_advantages(self, rewards: List[float]) -> torch.Tensor:
        """حساب الـ Advantage النسبي داخل المجموعة (بديل Value Model)."""
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        mean_r = rewards_t.mean()
        std_r = rewards_t.std() + 1e-8  # تفادي القسمة على صفر
        advantages = (rewards_t - mean_r) / std_r
        return advantages

    def compute_grpo_loss(self, log_probs_new, log_probs_old,
                           log_probs_ref, advantages):
        """
        حساب خسارة GRPO: PPO-style clipped objective + عقوبة KL صريحة
        (بخلاف PPO التي تدمج KL داخل المكافأة أحيانًا).
        """
        # نسبة الاحتمالات (policy ratio)
        ratio = torch.exp(log_probs_new - log_probs_old)

        # الهدف المقصوص (clipped surrogate objective) كما في PPO
        unclipped = ratio * advantages
        clipped = torch.clamp(
            ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon
        ) * advantages
        policy_loss = -torch.min(unclipped, clipped).mean()

        # عقوبة KL صريحة بين السياسة الحالية والمرجعية (منع الانحراف الكبير)
        kl_penalty = (
            torch.exp(log_probs_ref - log_probs_new)
            - (log_probs_ref - log_probs_new) - 1
        ).mean()  # تقدير KL غير المتحيّز (k3 estimator)

        total_loss = policy_loss + self.config.kl_coefficient * kl_penalty
        return total_loss, policy_loss.item(), kl_penalty.item()

    def train_step(self, prompt_ids, tokenizer, ground_truth_answer):
        """خطوة تدريب واحدة كاملة بمنهجية GRPO."""
        # 1. توليد مجموعة من الاستجابات
        responses = self.sample_group(prompt_ids, tokenizer)

        # 2. حساب المكافأة القابلة للتحقق لكل استجابة
        #    (مثال: 1.0 إذا كانت الإجابة الرياضية صحيحة، 0.0 إذا خاطئة
        #     + مكافأة تنسيق صغيرة إذا استُخدمت وسوم <think></think>)
        rewards = [
            self.reward_fn(resp, ground_truth_answer) for resp in responses
        ]

        # 3. حساب الـ Advantages النسبية (بدون Value Model!)
        advantages = self.compute_group_advantages(rewards)

        # 4. حساب log-probabilities من السياسة الحالية والمرجعية
        log_probs_new = self._compute_log_probs(self.policy_model, responses)
        log_probs_old = log_probs_new.detach()  # من نفس التوليد (on-policy تقريبًا)
        log_probs_ref = self._compute_log_probs(self.ref_model, responses)

        # 5. حساب الخسارة والتحديث
        loss, policy_loss, kl = self.compute_grpo_loss(
            log_probs_new, log_probs_old, log_probs_ref, advantages
        )
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.optimizer.step()

        return {"loss": loss.item(), "policy_loss": policy_loss,
                "kl": kl, "mean_reward": sum(rewards) / len(rewards)}

    def _compute_log_probs(self, model, responses):
        # دالة مساعدة مفاهيمية — تحسب مجموع log-probs للتسلسل بالكامل
        raise NotImplementedError("تُطبَّق بحسب واجهة النموذج المحدد")

def verifiable_math_reward(response_text: str, ground_truth: str) -> float:
    """
    دالة مكافأة قابلة للتحقق (Rule-Based Reward) — جوهر نجاح R1-Zero.
    بخلاف Reward Model مُدرَّب (قد يُخدَع بـ Reward Hacking)، هذه الدالة
    حتمية 100% ولا يمكن التلاعب بها.
    """
    import re
    # استخراج الإجابة النهائية من داخل \boxed{} أو بعد "الإجابة النهائية:"
    match = re.search(r"\\boxed\{([^}]+)\}", response_text)
    if not match:
        match = re.search(r"الإجابة النهائية[:\s]*([^\n]+)", response_text)
    if not match:
        return 0.0
    extracted = match.group(1).strip()

    # مكافأة الدقة
    correctness_reward = 1.0 if extracted == ground_truth.strip() else 0.0

    # مكافأة تنسيق إضافية صغيرة (تشجيع استخدام وسوم التفكير المنظّمة)
    format_reward = 0.1 if "<think>" in response_text and "</think>" in response_text else 0.0

    return correctness_reward + format_reward

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-60-ch60-reasoning-models.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from enum import Enum

class TestTimeComputeStrategy(Enum):
    """الاستراتيجيات الأربع الرئيسية لاستغلال حوسبة وقت الاستدلال."""
    SEQUENTIAL_REVISION = "sequential_revision"      # النموذج يراجع إجابته تكراريًا
    PARALLEL_SAMPLING = "parallel_sampling"           # توليد N إجابات واختيار الأفضل
    TREE_SEARCH = "tree_search"                        # بحث شجري موجّه (ToT، MCTS)
    LEARNED_LONG_CHAIN = "learned_long_chain"          # RL على سلاسل تفكير طويلة (o1/R1)

def best_of_n_with_prm(model, prm_scorer, prompt: str, n: int = 32) -> str:
    """
    استراتيجية Best-of-N الموجّهة بـ PRM: وَلِّد N حلول، قيّم كل خطوة بـ PRM،
    اختر الحل ذا أعلى درجة تراكمية. أرخص من RL لكن أكثر تكلفة وقت الاستدلال
    من استدلال واحد مباشر.
    """
    candidates = [model.generate(prompt, temperature=0.8) for _ in range(n)]

    scored_candidates = []
    for candidate in candidates:
        steps = split_into_reasoning_steps(candidate)
        step_scores = [prm_scorer.score_step(prompt, steps[:i+1])
                       for i in range(len(steps))]
        # الدرجة التراكمية = أدنى درجة عبر الخطوات (أضعف حلقة في السلسلة)
        # أو حاصل الضرب — كلا الاستراتيجيتين مُستخدم في الأدبيات
        aggregate_score = min(step_scores) if step_scores else 0.0
        scored_candidates.append((candidate, aggregate_score))

    best_candidate = max(scored_candidates, key=lambda x: x[1])
    return best_candidate[0]

def split_into_reasoning_steps(text: str) -> list:
    """تقسيم نص التفكير إلى خطوات منفصلة (بناءً على فواصل الأسطر/الترقيم)."""
    import re
    steps = re.split(r'\n(?=\d+[\.\)]|الخطوة)', text)
    return [s.strip() for s in steps if s.strip()]

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-60-ch60-reasoning-models.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-60-ch60-reasoning-models.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — نماذج الاستدلال وTest-Time Compute
Mastering Transformers and Large Language Models — Reasoning Models

Book slug : mastering-llms-transformers
Chapter   : ch-60-ch60-reasoning-models (lesson 60)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-12-future/ch60-reasoning-models.ipynb#cell-1
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
    ("torch", "torch")
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
# [002] content/notebooks/part-12-future/ch60-reasoning-models.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Callable

@dataclass
class GRPOConfig:
    group_size: int = 16          # عدد الاستجابات لكل سؤال
    clip_epsilon: float = 0.2     # قص PPO-style
    kl_coefficient: float = 0.04  # عقوبة الابتعاد عن النموذج المرجعي
    learning_rate: float = 1e-6

class GRPOTrainer:
    """
    تطبيق مبسّط ومفاهيمي لخوارزمية GRPO المستخدمة في تدريب DeepSeek-R1.
    الفكرة الجوهرية: استبدال Value Model بمتوسط مكافآت المجموعة.
    """
    def __init__(self, policy_model, ref_model, reward_fn: Callable,
                 config: GRPOConfig):
        self.policy_model = policy_model
        self.ref_model = ref_model  # نسخة مجمّدة من النموذج قبل RL (لـ KL)
        self.reward_fn = reward_fn  # دالة مكافأة قابلة للتحقق (مثل: صحة الإجابة الرياضية)
        self.config = config
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(), lr=config.learning_rate
        )

    def sample_group(self, prompt_ids, tokenizer, max_new_tokens=2048):
        """توليد G استجابات مختلفة لنفس السؤال (بعينات Sampling عشوائية)."""
        group_size = self.config.group_size
        responses = []
        for _ in range(group_size):
            with torch.no_grad():
                output_ids = self.policy_model.generate(
                    prompt_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                )
            responses.append(output_ids)
        return responses

    def compute_group_advantages(self, rewards: List[float]) -> torch.Tensor:
        """حساب الـ Advantage النسبي داخل المجموعة (بديل Value Model)."""
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        mean_r = rewards_t.mean()
        std_r = rewards_t.std() + 1e-8  # تفادي القسمة على صفر
        advantages = (rewards_t - mean_r) / std_r
        return advantages

    def compute_grpo_loss(self, log_probs_new, log_probs_old,
                           log_probs_ref, advantages):
        """
        حساب خسارة GRPO: PPO-style clipped objective + عقوبة KL صريحة
        (بخلاف PPO التي تدمج KL داخل المكافأة أحيانًا).
        """
        # نسبة الاحتمالات (policy ratio)
        ratio = torch.exp(log_probs_new - log_probs_old)

        # الهدف المقصوص (clipped surrogate objective) كما في PPO
        unclipped = ratio * advantages
        clipped = torch.clamp(
            ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon
        ) * advantages
        policy_loss = -torch.min(unclipped, clipped).mean()

        # عقوبة KL صريحة بين السياسة الحالية والمرجعية (منع الانحراف الكبير)
        kl_penalty = (
            torch.exp(log_probs_ref - log_probs_new)
            - (log_probs_ref - log_probs_new) - 1
        ).mean()  # تقدير KL غير المتحيّز (k3 estimator)

        total_loss = policy_loss + self.config.kl_coefficient * kl_penalty
        return total_loss, policy_loss.item(), kl_penalty.item()

    def train_step(self, prompt_ids, tokenizer, ground_truth_answer):
        """خطوة تدريب واحدة كاملة بمنهجية GRPO."""
        # 1. توليد مجموعة من الاستجابات
        responses = self.sample_group(prompt_ids, tokenizer)

        # 2. حساب المكافأة القابلة للتحقق لكل استجابة
        #    (مثال: 1.0 إذا كانت الإجابة الرياضية صحيحة، 0.0 إذا خاطئة
        #     + مكافأة تنسيق صغيرة إذا استُخدمت وسوم <think></think>)
        rewards = [
            self.reward_fn(resp, ground_truth_answer) for resp in responses
        ]

        # 3. حساب الـ Advantages النسبية (بدون Value Model!)
        advantages = self.compute_group_advantages(rewards)

        # 4. حساب log-probabilities من السياسة الحالية والمرجعية
        log_probs_new = self._compute_log_probs(self.policy_model, responses)
        log_probs_old = log_probs_new.detach()  # من نفس التوليد (on-policy تقريبًا)
        log_probs_ref = self._compute_log_probs(self.ref_model, responses)

        # 5. حساب الخسارة والتحديث
        loss, policy_loss, kl = self.compute_grpo_loss(
            log_probs_new, log_probs_old, log_probs_ref, advantages
        )
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.optimizer.step()

        return {"loss": loss.item(), "policy_loss": policy_loss,
                "kl": kl, "mean_reward": sum(rewards) / len(rewards)}

    def _compute_log_probs(self, model, responses):
        # دالة مساعدة مفاهيمية — تحسب مجموع log-probs للتسلسل بالكامل
        raise NotImplementedError("تُطبَّق بحسب واجهة النموذج المحدد")

def verifiable_math_reward(response_text: str, ground_truth: str) -> float:
    """
    دالة مكافأة قابلة للتحقق (Rule-Based Reward) — جوهر نجاح R1-Zero.
    بخلاف Reward Model مُدرَّب (قد يُخدَع بـ Reward Hacking)، هذه الدالة
    حتمية 100% ولا يمكن التلاعب بها.
    """
    import re
    # استخراج الإجابة النهائية من داخل \boxed{} أو بعد "الإجابة النهائية:"
    match = re.search(r"\\boxed\{([^}]+)\}", response_text)
    if not match:
        match = re.search(r"الإجابة النهائية[:\s]*([^\n]+)", response_text)
    if not match:
        return 0.0
    extracted = match.group(1).strip()

    # مكافأة الدقة
    correctness_reward = 1.0 if extracted == ground_truth.strip() else 0.0

    # مكافأة تنسيق إضافية صغيرة (تشجيع استخدام وسوم التفكير المنظّمة)
    format_reward = 0.1 if "<think>" in response_text and "</think>" in response_text else 0.0

    return correctness_reward + format_reward

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-12-future/ch60-reasoning-models.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from enum import Enum

class TestTimeComputeStrategy(Enum):
    """الاستراتيجيات الأربع الرئيسية لاستغلال حوسبة وقت الاستدلال."""
    SEQUENTIAL_REVISION = "sequential_revision"      # النموذج يراجع إجابته تكراريًا
    PARALLEL_SAMPLING = "parallel_sampling"           # توليد N إجابات واختيار الأفضل
    TREE_SEARCH = "tree_search"                        # بحث شجري موجّه (ToT، MCTS)
    LEARNED_LONG_CHAIN = "learned_long_chain"          # RL على سلاسل تفكير طويلة (o1/R1)

def best_of_n_with_prm(model, prm_scorer, prompt: str, n: int = 32) -> str:
    """
    استراتيجية Best-of-N الموجّهة بـ PRM: وَلِّد N حلول، قيّم كل خطوة بـ PRM،
    اختر الحل ذا أعلى درجة تراكمية. أرخص من RL لكن أكثر تكلفة وقت الاستدلال
    من استدلال واحد مباشر.
    """
    candidates = [model.generate(prompt, temperature=0.8) for _ in range(n)]

    scored_candidates = []
    for candidate in candidates:
        steps = split_into_reasoning_steps(candidate)
        step_scores = [prm_scorer.score_step(prompt, steps[:i+1])
                       for i in range(len(steps))]
        # الدرجة التراكمية = أدنى درجة عبر الخطوات (أضعف حلقة في السلسلة)
        # أو حاصل الضرب — كلا الاستراتيجيتين مُستخدم في الأدبيات
        aggregate_score = min(step_scores) if step_scores else 0.0
        scored_candidates.append((candidate, aggregate_score))

    best_candidate = max(scored_candidates, key=lambda x: x[1])
    return best_candidate[0]

def split_into_reasoning_steps(text: str) -> list:
    """تقسيم نص التفكير إلى خطوات منفصلة (بناءً على فواصل الأسطر/الترقيم)."""
    import re
    steps = re.split(r'\n(?=\d+[\.\)]|الخطوة)', text)
    return [s.strip() for s in steps if s.strip()]

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-12-future/ch60-reasoning-models.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-60-ch60-reasoning-models.ipynb#cell-2
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
    ("torch", "torch")
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-60-ch60-reasoning-models.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Callable

@dataclass
class GRPOConfig:
    group_size: int = 16          # عدد الاستجابات لكل سؤال
    clip_epsilon: float = 0.2     # قص PPO-style
    kl_coefficient: float = 0.04  # عقوبة الابتعاد عن النموذج المرجعي
    learning_rate: float = 1e-6

class GRPOTrainer:
    """
    تطبيق مبسّط ومفاهيمي لخوارزمية GRPO المستخدمة في تدريب DeepSeek-R1.
    الفكرة الجوهرية: استبدال Value Model بمتوسط مكافآت المجموعة.
    """
    def __init__(self, policy_model, ref_model, reward_fn: Callable,
                 config: GRPOConfig):
        self.policy_model = policy_model
        self.ref_model = ref_model  # نسخة مجمّدة من النموذج قبل RL (لـ KL)
        self.reward_fn = reward_fn  # دالة مكافأة قابلة للتحقق (مثل: صحة الإجابة الرياضية)
        self.config = config
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(), lr=config.learning_rate
        )

    def sample_group(self, prompt_ids, tokenizer, max_new_tokens=2048):
        """توليد G استجابات مختلفة لنفس السؤال (بعينات Sampling عشوائية)."""
        group_size = self.config.group_size
        responses = []
        for _ in range(group_size):
            with torch.no_grad():
                output_ids = self.policy_model.generate(
                    prompt_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                )
            responses.append(output_ids)
        return responses

    def compute_group_advantages(self, rewards: List[float]) -> torch.Tensor:
        """حساب الـ Advantage النسبي داخل المجموعة (بديل Value Model)."""
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        mean_r = rewards_t.mean()
        std_r = rewards_t.std() + 1e-8  # تفادي القسمة على صفر
        advantages = (rewards_t - mean_r) / std_r
        return advantages

    def compute_grpo_loss(self, log_probs_new, log_probs_old,
                           log_probs_ref, advantages):
        """
        حساب خسارة GRPO: PPO-style clipped objective + عقوبة KL صريحة
        (بخلاف PPO التي تدمج KL داخل المكافأة أحيانًا).
        """
        # نسبة الاحتمالات (policy ratio)
        ratio = torch.exp(log_probs_new - log_probs_old)

        # الهدف المقصوص (clipped surrogate objective) كما في PPO
        unclipped = ratio * advantages
        clipped = torch.clamp(
            ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon
        ) * advantages
        policy_loss = -torch.min(unclipped, clipped).mean()

        # عقوبة KL صريحة بين السياسة الحالية والمرجعية (منع الانحراف الكبير)
        kl_penalty = (
            torch.exp(log_probs_ref - log_probs_new)
            - (log_probs_ref - log_probs_new) - 1
        ).mean()  # تقدير KL غير المتحيّز (k3 estimator)

        total_loss = policy_loss + self.config.kl_coefficient * kl_penalty
        return total_loss, policy_loss.item(), kl_penalty.item()

    def train_step(self, prompt_ids, tokenizer, ground_truth_answer):
        """خطوة تدريب واحدة كاملة بمنهجية GRPO."""
        # 1. توليد مجموعة من الاستجابات
        responses = self.sample_group(prompt_ids, tokenizer)

        # 2. حساب المكافأة القابلة للتحقق لكل استجابة
        #    (مثال: 1.0 إذا كانت الإجابة الرياضية صحيحة، 0.0 إذا خاطئة
        #     + مكافأة تنسيق صغيرة إذا استُخدمت وسوم <think></think>)
        rewards = [
            self.reward_fn(resp, ground_truth_answer) for resp in responses
        ]

        # 3. حساب الـ Advantages النسبية (بدون Value Model!)
        advantages = self.compute_group_advantages(rewards)

        # 4. حساب log-probabilities من السياسة الحالية والمرجعية
        log_probs_new = self._compute_log_probs(self.policy_model, responses)
        log_probs_old = log_probs_new.detach()  # من نفس التوليد (on-policy تقريبًا)
        log_probs_ref = self._compute_log_probs(self.ref_model, responses)

        # 5. حساب الخسارة والتحديث
        loss, policy_loss, kl = self.compute_grpo_loss(
            log_probs_new, log_probs_old, log_probs_ref, advantages
        )
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.optimizer.step()

        return {"loss": loss.item(), "policy_loss": policy_loss,
                "kl": kl, "mean_reward": sum(rewards) / len(rewards)}

    def _compute_log_probs(self, model, responses):
        # دالة مساعدة مفاهيمية — تحسب مجموع log-probs للتسلسل بالكامل
        raise NotImplementedError("تُطبَّق بحسب واجهة النموذج المحدد")

def verifiable_math_reward(response_text: str, ground_truth: str) -> float:
    """
    دالة مكافأة قابلة للتحقق (Rule-Based Reward) — جوهر نجاح R1-Zero.
    بخلاف Reward Model مُدرَّب (قد يُخدَع بـ Reward Hacking)، هذه الدالة
    حتمية 100% ولا يمكن التلاعب بها.
    """
    import re
    # استخراج الإجابة النهائية من داخل \boxed{} أو بعد "الإجابة النهائية:"
    match = re.search(r"\\boxed\{([^}]+)\}", response_text)
    if not match:
        match = re.search(r"الإجابة النهائية[:\s]*([^\n]+)", response_text)
    if not match:
        return 0.0
    extracted = match.group(1).strip()

    # مكافأة الدقة
    correctness_reward = 1.0 if extracted == ground_truth.strip() else 0.0

    # مكافأة تنسيق إضافية صغيرة (تشجيع استخدام وسوم التفكير المنظّمة)
    format_reward = 0.1 if "<think>" in response_text and "</think>" in response_text else 0.0

    return correctness_reward + format_reward

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-60-ch60-reasoning-models.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from enum import Enum

class TestTimeComputeStrategy(Enum):
    """الاستراتيجيات الأربع الرئيسية لاستغلال حوسبة وقت الاستدلال."""
    SEQUENTIAL_REVISION = "sequential_revision"      # النموذج يراجع إجابته تكراريًا
    PARALLEL_SAMPLING = "parallel_sampling"           # توليد N إجابات واختيار الأفضل
    TREE_SEARCH = "tree_search"                        # بحث شجري موجّه (ToT، MCTS)
    LEARNED_LONG_CHAIN = "learned_long_chain"          # RL على سلاسل تفكير طويلة (o1/R1)

def best_of_n_with_prm(model, prm_scorer, prompt: str, n: int = 32) -> str:
    """
    استراتيجية Best-of-N الموجّهة بـ PRM: وَلِّد N حلول، قيّم كل خطوة بـ PRM،
    اختر الحل ذا أعلى درجة تراكمية. أرخص من RL لكن أكثر تكلفة وقت الاستدلال
    من استدلال واحد مباشر.
    """
    candidates = [model.generate(prompt, temperature=0.8) for _ in range(n)]

    scored_candidates = []
    for candidate in candidates:
        steps = split_into_reasoning_steps(candidate)
        step_scores = [prm_scorer.score_step(prompt, steps[:i+1])
                       for i in range(len(steps))]
        # الدرجة التراكمية = أدنى درجة عبر الخطوات (أضعف حلقة في السلسلة)
        # أو حاصل الضرب — كلا الاستراتيجيتين مُستخدم في الأدبيات
        aggregate_score = min(step_scores) if step_scores else 0.0
        scored_candidates.append((candidate, aggregate_score))

    best_candidate = max(scored_candidates, key=lambda x: x[1])
    return best_candidate[0]

def split_into_reasoning_steps(text: str) -> list:
    """تقسيم نص التفكير إلى خطوات منفصلة (بناءً على فواصل الأسطر/الترقيم)."""
    import re
    steps = re.split(r'\n(?=\d+[\.\)]|الخطوة)', text)
    return [s.strip() for s in steps if s.strip()]

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-60-ch60-reasoning-models.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
