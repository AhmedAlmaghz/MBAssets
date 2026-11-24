#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — التعلّم المعزَّز من التغذية الراجعة البشرية (RLHF)
Large Language Model Engineering — RLHF

Book slug : llm-engineering-69-chapters
Chapter   : ch-35-chapter-35 (lesson 35)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-35-chapter-35.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — التعلّم المعزَّز من التغذية الراجعة البشرية (RLHF)
Large Language Model Engineering — RLHF

Book slug : llm-engineering-69-chapters
Chapter   : ch-35-chapter-35 (lesson 35)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-35-chapter-35.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — التعلّم المعزَّز من التغذية الراجعة البشرية (RLHF)
Large Language Model Engineering — RLHF

Book slug : llm-engineering-69-chapters
Chapter   : ch-35-chapter-35 (lesson 35)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_35.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 35: الفصل 35: التعلّم المعزَّز من التغذية الراجعة البشرية (RLHF)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_rlhf() -> None:
    rewards, kl_penalty = ({'helpful': 1.4, 'unsafe': -1.0}, 0.2)
    reference_drift = {'helpful': 0.3, 'unsafe': 0.8}
    objective = {name: reward - kl_penalty * reference_drift[name] for name, reward in rewards.items()}
    print('reward scores:', rewards)
    print('KL-regularized objective:', {key: round(value, 3) for key, value in objective.items()})

if __name__ == "__main__":
    print("Chapter 35 lab: Compare preference rewards and a KL constraint")
    run_rlhf()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-07-fine-tuning-alignment/chapter-35.md
# ────────────────────────────────────────────────────────────────────────
"""
تدريب نموذج المكافأة (Reward Model) من بيانات تفضيل بشرية
Python 3.13+ | PyTorch 2.x | transformers
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader


class RewardModel(nn.Module):
    """
    نموذج مكافأة: يُغلّف نموذجًا لغويًا أساسيًا ويستبدل رأس توليد
    اللغة برأس Scalar واحد يُنتج درجة تقييم واحدة لكل تسلسل.
    """

    def __init__(self, base_model_name: str):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.backbone.config.hidden_size
        # رأس المكافأة: يُسقط التمثيل المخفي النهائي إلى رقم واحد
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)

        # نأخذ تمثيل آخر توكن فعلي (غير Padding) لكل تسلسل
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_hidden = hidden_states[batch_indices, sequence_lengths]  # (batch, hidden)

        reward = self.reward_head(last_hidden).squeeze(-1)  # (batch,)
        return reward


@dataclass
class PreferencePair:
    prompt: str
    chosen: str      # الاستجابة المفضَّلة (y_w)
    rejected: str     # الاستجابة المرفوضة (y_l)


class PreferenceDataset(Dataset):
    def __init__(self, pairs: list[PreferencePair], tokenizer, max_length: int = 512):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        chosen_text = f"{pair.prompt}\n{pair.chosen}"
        rejected_text = f"{pair.prompt}\n{pair.rejected}"

        chosen_enc = self.tokenizer(
            chosen_text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        rejected_enc = self.tokenizer(
            rejected_text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen_enc.input_ids.squeeze(0),
            "chosen_attention_mask": chosen_enc.attention_mask.squeeze(0),
            "rejected_input_ids": rejected_enc.input_ids.squeeze(0),
            "rejected_attention_mask": rejected_enc.attention_mask.squeeze(0),
        }


def reward_model_loss(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
    """
    دالة خسارة Bradley-Terry:
    L = -log(sigmoid(r_chosen - r_rejected))
    """
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()


def train_reward_model(
    model: RewardModel,
    dataloader: DataLoader,
    num_epochs: int = 3,
    learning_rate: float = 1e-5,
    device: str = "cuda",
) -> RewardModel:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        total_loss, correct_predictions, total_samples = 0.0, 0, 0

        for batch in dataloader:
            chosen_rewards = model(
                batch["chosen_input_ids"].to(device),
                batch["chosen_attention_mask"].to(device),
            )
            rejected_rewards = model(
                batch["rejected_input_ids"].to(device),
                batch["rejected_attention_mask"].to(device),
            )

            loss = reward_model_loss(chosen_rewards, rejected_rewards)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            # دقة: كم مرة r_chosen > r_rejected فعليًا (مؤشر صحة مباشر)
            correct_predictions += (chosen_rewards > rejected_rewards).sum().item()
            total_samples += chosen_rewards.size(0)

        accuracy = correct_predictions / total_samples
        print(f"Epoch {epoch+1}: Loss={total_loss/len(dataloader):.4f}, "
              f"Preference Accuracy={accuracy:.2%}")

    return model

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-07-fine-tuning-alignment/chapter-35.md
# ────────────────────────────────────────────────────────────────────────
"""
تطبيق مبسّط لحلقة PPO لتحسين سياسة نموذج لغوي
يُظهر الآلية الكاملة: 4 نماذج تعمل معًا
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy


@dataclass
class PPOConfig:
    kl_coef: float = 0.1          # β في معادلة المكافأة المُعدَّلة
    clip_epsilon: float = 0.2     # ε في Clipped Objective
    gamma: float = 1.0            # عامل الخصم (عادة 1.0 لمهام توليد النص القصيرة)
    gae_lambda: float = 0.95      # λ في GAE
    ppo_epochs: int = 4           # عدد تكرارات التحديث على نفس الدفعة
    learning_rate: float = 1e-6
    value_loss_coef: float = 0.5


class PPOTrainerFromScratch:
    """
    تطبيق تعليمي مبسّط لـ PPO يُظهر الأربعة نماذج والعلاقة بينها.
    (في الإنتاج الفعلي، استخدم مكتبة trl.PPOTrainer المُحسَّنة والمُختبَرة جيدًا)
    """

    def __init__(
        self,
        policy_model: AutoModelForCausalLM,   # السياسة القابلة للتدريب π_θ
        reward_model,                          # نموذج المكافأة r_φ (مجمَّد)
        tokenizer,
        config: PPOConfig,
        device: str = "cuda",
    ):
        self.device = device
        self.config = config
        self.tokenizer = tokenizer

        self.policy = policy_model.to(device)

        # Reference Model: نسخة مجمَّدة تمامًا من السياسة الأصلية (بعد SFT)
        self.reference_model = copy.deepcopy(policy_model).to(device)
        for p in self.reference_model.parameters():
            p.requires_grad = False
        self.reference_model.eval()

        self.reward_model = reward_model.to(device)
        for p in self.reward_model.parameters():
            p.requires_grad = False
        self.reward_model.eval()

        # Value Head: رأس إضافي بسيط فوق نفس backbone السياسة لتقدير V(s)
        hidden_size = self.policy.config.hidden_size
        self.value_head = torch.nn.Linear(hidden_size, 1).to(device)

        self.optimizer = torch.optim.AdamW(
            list(self.policy.parameters()) + list(self.value_head.parameters()),
            lr=config.learning_rate,
        )

    def generate_rollouts(self, prompts: list[str], max_new_tokens: int = 64) -> dict:
        """المرحلة 1 من كل تكرار: توليد استجابات من السياسة الحالية."""
        self.policy.eval()
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            generated = self.policy.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=1.0, top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        full_texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return {"input_ids": generated, "attention_mask": (generated != self.tokenizer.pad_token_id).long(),
                "texts": full_texts}

    def compute_rewards_and_kl(self, rollout: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """المرحلة 2: حساب المكافأة الخام من Reward Model + عقوبة KL من Reference Model."""
        input_ids = rollout["input_ids"]
        attention_mask = rollout["attention_mask"]

        with torch.no_grad():
            # مكافأة سكالر واحدة لكل تسلسل كامل من Reward Model
            raw_rewards = self.reward_model(input_ids, attention_mask)

            # احتمالات log لكل توكن من السياسة الحالية والـ Reference
            policy_logits = self.policy(input_ids, attention_mask=attention_mask).logits
            ref_logits = self.reference_model(input_ids, attention_mask=attention_mask).logits

            policy_log_probs = F.log_softmax(policy_logits, dim=-1)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)

            # نأخذ log-prob للتوكن الفعلي المُولَّد في كل موضع
            token_policy_lp = torch.gather(
                policy_log_probs[:, :-1], 2, input_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            token_ref_lp = torch.gather(
                ref_log_probs[:, :-1], 2, input_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)

            # تقدير KL لكل توكن (Schulman's k1 approximation)
            per_token_kl = token_policy_lp - token_ref_lp

        return raw_rewards, per_token_kl

    def compute_advantages_gae(
        self, rewards: torch.Tensor, values: torch.Tensor, per_token_kl: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """المرحلة 3: دمج المكافأة مع عقوبة KL، ثم حساب Advantage عبر GAE."""
        # المكافأة النهائية: فقط على آخر توكن + عقوبة KL موزَّعة على كل التوكنز
        seq_len = per_token_kl.size(1)
        token_rewards = -self.config.kl_coef * per_token_kl
        token_rewards[:, -1] += rewards  # المكافأة الكلية تُضاف فقط عند نهاية التسلسل

        advantages = torch.zeros_like(token_rewards)
        last_advantage = 0.0
        last_value = 0.0

        for t in reversed(range(seq_len)):
            next_value = values[:, t + 1] if t < seq_len - 1 else last_value
            delta = token_rewards[:, t] + self.config.gamma * next_value - values[:, t]
            last_advantage = delta + self.config.gamma * self.config.gae_lambda * last_advantage
            advantages[:, t] = last_advantage

        returns = advantages + values
        # تطبيع Advantage (ممارسة قياسية لتحسين استقرار التدريب)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def ppo_update_step(
        self, rollout: dict, old_log_probs: torch.Tensor,
        advantages: torch.Tensor, returns: torch.Tensor,
    ) -> dict:
        """المرحلة 4: التحديث الفعلي بالدالة الهدف المُقتَصَّة (Clipped Objective)."""
        self.policy.train()
        input_ids = rollout["input_ids"]
        attention_mask = rollout["attention_mask"]

        for _ in range(self.config.ppo_epochs):
            outputs = self.policy(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)
            new_log_probs = torch.gather(
                log_probs[:, :-1], 2, input_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)

            # القيمة المُقدَّرة الجديدة من Value Head
            hidden = outputs.hidden_states[-1][:, :-1]
            new_values = self.value_head(hidden).squeeze(-1)

            # نسبة الاحتمال r_t(θ)
            ratio = torch.exp(new_log_probs - old_log_probs)

            # الدالة الهدف المُقتَصَّة
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon
            ) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # خسارة Value Head (تدريب Critic ليُقدّر العوائد بدقة)
            value_loss = F.mse_loss(new_values, returns)

            total_loss = policy_loss + self.config.value_loss_coef * value_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.policy.parameters()) + list(self.value_head.parameters()), max_norm=1.0
            )
            self.optimizer.step()

        return {"policy_loss": policy_loss.item(), "value_loss": value_loss.item()}

    def train_step(self, prompts: list[str]) -> dict:
        """تكرار كامل واحد من RLHF: Rollout → Reward → Advantage → Update."""
        rollout = self.generate_rollouts(prompts)
        rewards, per_token_kl = self.compute_rewards_and_kl(rollout)

        with torch.no_grad():
            outputs = self.policy(rollout["input_ids"], output_hidden_states=True)
            old_log_probs_full = F.log_softmax(outputs.logits, dim=-1)
            old_log_probs = torch.gather(
                old_log_probs_full[:, :-1], 2, rollout["input_ids"][:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            old_values = self.value_head(outputs.hidden_states[-1][:, :-1]).squeeze(-1)

        advantages, returns = self.compute_advantages_gae(rewards, old_values, per_token_kl)
        stats = self.ppo_update_step(rollout, old_log_probs, advantages, returns)
        stats["mean_reward"] = rewards.mean().item()
        stats["mean_kl"] = per_token_kl.mean().item()
        return stats

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-07-fine-tuning-alignment/chapter-35.md
# ────────────────────────────────────────────────────────────────────────
"""
استخدام مكتبة trl لتطبيق RLHF كامل في الإنتاج
pip install trl transformers peft
"""

from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
import torch


MODEL_NAME = "./sft-model"  # نموذج تم ضبطه بـ SFT مسبقًا (الفصل 34)

# النموذج مع Value Head مدمج تلقائيًا (بدل بنائه يدويًا كما فعلنا أعلاه)
policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME)
reference_model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

ppo_config = PPOConfig(
    learning_rate=1.41e-5,
    batch_size=64,
    mini_batch_size=8,
    ppo_epochs=4,
    kl_penalty="kl",       # نوع تقدير الـ KL
    init_kl_coef=0.2,      # β الأولي، يُعدَّل تلقائيًا (Adaptive KL Controller)
    target_kl=6.0,         # القيمة المستهدفة لـ KL — يُضبط β تلقائيًا للبقاء قربها
    gamma=1.0,
    lam=0.95,              # λ في GAE
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=policy_model,
    ref_model=reference_model,
    tokenizer=tokenizer,
)

# حلقة تدريب مبسّطة (في الإنتاج: آلاف التكرارات على بيانات prompts متنوعة)
prompts = ["اشرح لي أهمية الذكاء الاصطناعي", "كيف أتعلم البرمجة بسرعة؟"]

for epoch in range(10):
    query_tensors = [tokenizer(p, return_tensors="pt").input_ids[0] for p in prompts]

    # 1) توليد استجابات
    response_tensors = ppo_trainer.generate(
        query_tensors, max_new_tokens=64, do_sample=True, top_p=0.9,
    )
    responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

    # 2) حساب المكافآت من Reward Model المدرَّب مسبقًا (دالة خارجية في مشروع حقيقي)
    rewards = [compute_reward_from_model(q, r) for q, r in zip(prompts, responses)]
    reward_tensors = [torch.tensor(r) for r in rewards]

    # 3) خطوة PPO واحدة — المكتبة تُدير GAE، Clipping، KL Control داخليًا
    stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
    ppo_trainer.log_stats(stats, {"query": prompts, "response": responses}, rewards)

    print(f"Epoch {epoch}: mean_reward={sum(rewards)/len(rewards):.3f}, "
          f"kl={stats.get('objective/kl', 0):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-35-chapter-35.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 35: الفصل 35: التعلّم المعزَّز من التغذية الراجعة البشرية (RLHF)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_rlhf() -> None:
    rewards, kl_penalty = ({'helpful': 1.4, 'unsafe': -1.0}, 0.2)
    reference_drift = {'helpful': 0.3, 'unsafe': 0.8}
    objective = {name: reward - kl_penalty * reference_drift[name] for name, reward in rewards.items()}
    print('reward scores:', rewards)
    print('KL-regularized objective:', {key: round(value, 3) for key, value in objective.items()})

if __name__ == "__main__":
    print("Chapter 35 lab: Compare preference rewards and a KL constraint")
    run_rlhf()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-35-chapter-35.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
تدريب نموذج المكافأة (Reward Model) من بيانات تفضيل بشرية
Python 3.13+ | PyTorch 2.x | transformers
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader


class RewardModel(nn.Module):
    """
    نموذج مكافأة: يُغلّف نموذجًا لغويًا أساسيًا ويستبدل رأس توليد
    اللغة برأس Scalar واحد يُنتج درجة تقييم واحدة لكل تسلسل.
    """

    def __init__(self, base_model_name: str):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.backbone.config.hidden_size
        # رأس المكافأة: يُسقط التمثيل المخفي النهائي إلى رقم واحد
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)

        # نأخذ تمثيل آخر توكن فعلي (غير Padding) لكل تسلسل
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_hidden = hidden_states[batch_indices, sequence_lengths]  # (batch, hidden)

        reward = self.reward_head(last_hidden).squeeze(-1)  # (batch,)
        return reward


@dataclass
class PreferencePair:
    prompt: str
    chosen: str      # الاستجابة المفضَّلة (y_w)
    rejected: str     # الاستجابة المرفوضة (y_l)


class PreferenceDataset(Dataset):
    def __init__(self, pairs: list[PreferencePair], tokenizer, max_length: int = 512):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        chosen_text = f"{pair.prompt}\n{pair.chosen}"
        rejected_text = f"{pair.prompt}\n{pair.rejected}"

        chosen_enc = self.tokenizer(
            chosen_text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        rejected_enc = self.tokenizer(
            rejected_text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen_enc.input_ids.squeeze(0),
            "chosen_attention_mask": chosen_enc.attention_mask.squeeze(0),
            "rejected_input_ids": rejected_enc.input_ids.squeeze(0),
            "rejected_attention_mask": rejected_enc.attention_mask.squeeze(0),
        }


def reward_model_loss(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
    """
    دالة خسارة Bradley-Terry:
    L = -log(sigmoid(r_chosen - r_rejected))
    """
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()


def train_reward_model(
    model: RewardModel,
    dataloader: DataLoader,
    num_epochs: int = 3,
    learning_rate: float = 1e-5,
    device: str = "cuda",
) -> RewardModel:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        total_loss, correct_predictions, total_samples = 0.0, 0, 0

        for batch in dataloader:
            chosen_rewards = model(
                batch["chosen_input_ids"].to(device),
                batch["chosen_attention_mask"].to(device),
            )
            rejected_rewards = model(
                batch["rejected_input_ids"].to(device),
                batch["rejected_attention_mask"].to(device),
            )

            loss = reward_model_loss(chosen_rewards, rejected_rewards)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            # دقة: كم مرة r_chosen > r_rejected فعليًا (مؤشر صحة مباشر)
            correct_predictions += (chosen_rewards > rejected_rewards).sum().item()
            total_samples += chosen_rewards.size(0)

        accuracy = correct_predictions / total_samples
        print(f"Epoch {epoch+1}: Loss={total_loss/len(dataloader):.4f}, "
              f"Preference Accuracy={accuracy:.2%}")

    return model

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-35-chapter-35.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
تطبيق مبسّط لحلقة PPO لتحسين سياسة نموذج لغوي
يُظهر الآلية الكاملة: 4 نماذج تعمل معًا
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy


@dataclass
class PPOConfig:
    kl_coef: float = 0.1          # β في معادلة المكافأة المُعدَّلة
    clip_epsilon: float = 0.2     # ε في Clipped Objective
    gamma: float = 1.0            # عامل الخصم (عادة 1.0 لمهام توليد النص القصيرة)
    gae_lambda: float = 0.95      # λ في GAE
    ppo_epochs: int = 4           # عدد تكرارات التحديث على نفس الدفعة
    learning_rate: float = 1e-6
    value_loss_coef: float = 0.5


class PPOTrainerFromScratch:
    """
    تطبيق تعليمي مبسّط لـ PPO يُظهر الأربعة نماذج والعلاقة بينها.
    (في الإنتاج الفعلي، استخدم مكتبة trl.PPOTrainer المُحسَّنة والمُختبَرة جيدًا)
    """

    def __init__(
        self,
        policy_model: AutoModelForCausalLM,   # السياسة القابلة للتدريب π_θ
        reward_model,                          # نموذج المكافأة r_φ (مجمَّد)
        tokenizer,
        config: PPOConfig,
        device: str = "cuda",
    ):
        self.device = device
        self.config = config
        self.tokenizer = tokenizer

        self.policy = policy_model.to(device)

        # Reference Model: نسخة مجمَّدة تمامًا من السياسة الأصلية (بعد SFT)
        self.reference_model = copy.deepcopy(policy_model).to(device)
        for p in self.reference_model.parameters():
            p.requires_grad = False
        self.reference_model.eval()

        self.reward_model = reward_model.to(device)
        for p in self.reward_model.parameters():
            p.requires_grad = False
        self.reward_model.eval()

        # Value Head: رأس إضافي بسيط فوق نفس backbone السياسة لتقدير V(s)
        hidden_size = self.policy.config.hidden_size
        self.value_head = torch.nn.Linear(hidden_size, 1).to(device)

        self.optimizer = torch.optim.AdamW(
            list(self.policy.parameters()) + list(self.value_head.parameters()),
            lr=config.learning_rate,
        )

    def generate_rollouts(self, prompts: list[str], max_new_tokens: int = 64) -> dict:
        """المرحلة 1 من كل تكرار: توليد استجابات من السياسة الحالية."""
        self.policy.eval()
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            generated = self.policy.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=1.0, top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        full_texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return {"input_ids": generated, "attention_mask": (generated != self.tokenizer.pad_token_id).long(),
                "texts": full_texts}

    def compute_rewards_and_kl(self, rollout: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """المرحلة 2: حساب المكافأة الخام من Reward Model + عقوبة KL من Reference Model."""
        input_ids = rollout["input_ids"]
        attention_mask = rollout["attention_mask"]

        with torch.no_grad():
            # مكافأة سكالر واحدة لكل تسلسل كامل من Reward Model
            raw_rewards = self.reward_model(input_ids, attention_mask)

            # احتمالات log لكل توكن من السياسة الحالية والـ Reference
            policy_logits = self.policy(input_ids, attention_mask=attention_mask).logits
            ref_logits = self.reference_model(input_ids, attention_mask=attention_mask).logits

            policy_log_probs = F.log_softmax(policy_logits, dim=-1)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)

            # نأخذ log-prob للتوكن الفعلي المُولَّد في كل موضع
            token_policy_lp = torch.gather(
                policy_log_probs[:, :-1], 2, input_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            token_ref_lp = torch.gather(
                ref_log_probs[:, :-1], 2, input_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)

            # تقدير KL لكل توكن (Schulman's k1 approximation)
            per_token_kl = token_policy_lp - token_ref_lp

        return raw_rewards, per_token_kl

    def compute_advantages_gae(
        self, rewards: torch.Tensor, values: torch.Tensor, per_token_kl: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """المرحلة 3: دمج المكافأة مع عقوبة KL، ثم حساب Advantage عبر GAE."""
        # المكافأة النهائية: فقط على آخر توكن + عقوبة KL موزَّعة على كل التوكنز
        seq_len = per_token_kl.size(1)
        token_rewards = -self.config.kl_coef * per_token_kl
        token_rewards[:, -1] += rewards  # المكافأة الكلية تُضاف فقط عند نهاية التسلسل

        advantages = torch.zeros_like(token_rewards)
        last_advantage = 0.0
        last_value = 0.0

        for t in reversed(range(seq_len)):
            next_value = values[:, t + 1] if t < seq_len - 1 else last_value
            delta = token_rewards[:, t] + self.config.gamma * next_value - values[:, t]
            last_advantage = delta + self.config.gamma * self.config.gae_lambda * last_advantage
            advantages[:, t] = last_advantage

        returns = advantages + values
        # تطبيع Advantage (ممارسة قياسية لتحسين استقرار التدريب)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def ppo_update_step(
        self, rollout: dict, old_log_probs: torch.Tensor,
        advantages: torch.Tensor, returns: torch.Tensor,
    ) -> dict:
        """المرحلة 4: التحديث الفعلي بالدالة الهدف المُقتَصَّة (Clipped Objective)."""
        self.policy.train()
        input_ids = rollout["input_ids"]
        attention_mask = rollout["attention_mask"]

        for _ in range(self.config.ppo_epochs):
            outputs = self.policy(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)
            new_log_probs = torch.gather(
                log_probs[:, :-1], 2, input_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)

            # القيمة المُقدَّرة الجديدة من Value Head
            hidden = outputs.hidden_states[-1][:, :-1]
            new_values = self.value_head(hidden).squeeze(-1)

            # نسبة الاحتمال r_t(θ)
            ratio = torch.exp(new_log_probs - old_log_probs)

            # الدالة الهدف المُقتَصَّة
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon
            ) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # خسارة Value Head (تدريب Critic ليُقدّر العوائد بدقة)
            value_loss = F.mse_loss(new_values, returns)

            total_loss = policy_loss + self.config.value_loss_coef * value_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.policy.parameters()) + list(self.value_head.parameters()), max_norm=1.0
            )
            self.optimizer.step()

        return {"policy_loss": policy_loss.item(), "value_loss": value_loss.item()}

    def train_step(self, prompts: list[str]) -> dict:
        """تكرار كامل واحد من RLHF: Rollout → Reward → Advantage → Update."""
        rollout = self.generate_rollouts(prompts)
        rewards, per_token_kl = self.compute_rewards_and_kl(rollout)

        with torch.no_grad():
            outputs = self.policy(rollout["input_ids"], output_hidden_states=True)
            old_log_probs_full = F.log_softmax(outputs.logits, dim=-1)
            old_log_probs = torch.gather(
                old_log_probs_full[:, :-1], 2, rollout["input_ids"][:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            old_values = self.value_head(outputs.hidden_states[-1][:, :-1]).squeeze(-1)

        advantages, returns = self.compute_advantages_gae(rewards, old_values, per_token_kl)
        stats = self.ppo_update_step(rollout, old_log_probs, advantages, returns)
        stats["mean_reward"] = rewards.mean().item()
        stats["mean_kl"] = per_token_kl.mean().item()
        return stats

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-35-chapter-35.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
"""
استخدام مكتبة trl لتطبيق RLHF كامل في الإنتاج
pip install trl transformers peft
"""

from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
import torch


MODEL_NAME = "./sft-model"  # نموذج تم ضبطه بـ SFT مسبقًا (الفصل 34)

# النموذج مع Value Head مدمج تلقائيًا (بدل بنائه يدويًا كما فعلنا أعلاه)
policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME)
reference_model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

ppo_config = PPOConfig(
    learning_rate=1.41e-5,
    batch_size=64,
    mini_batch_size=8,
    ppo_epochs=4,
    kl_penalty="kl",       # نوع تقدير الـ KL
    init_kl_coef=0.2,      # β الأولي، يُعدَّل تلقائيًا (Adaptive KL Controller)
    target_kl=6.0,         # القيمة المستهدفة لـ KL — يُضبط β تلقائيًا للبقاء قربها
    gamma=1.0,
    lam=0.95,              # λ في GAE
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=policy_model,
    ref_model=reference_model,
    tokenizer=tokenizer,
)

# حلقة تدريب مبسّطة (في الإنتاج: آلاف التكرارات على بيانات prompts متنوعة)
prompts = ["اشرح لي أهمية الذكاء الاصطناعي", "كيف أتعلم البرمجة بسرعة؟"]

for epoch in range(10):
    query_tensors = [tokenizer(p, return_tensors="pt").input_ids[0] for p in prompts]

    # 1) توليد استجابات
    response_tensors = ppo_trainer.generate(
        query_tensors, max_new_tokens=64, do_sample=True, top_p=0.9,
    )
    responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

    # 2) حساب المكافآت من Reward Model المدرَّب مسبقًا (دالة خارجية في مشروع حقيقي)
    rewards = [compute_reward_from_model(q, r) for q, r in zip(prompts, responses)]
    reward_tensors = [torch.tensor(r) for r in rewards]

    # 3) خطوة PPO واحدة — المكتبة تُدير GAE، Clipping، KL Control داخليًا
    stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
    ppo_trainer.log_stats(stats, {"query": prompts, "response": responses}, rewards)

    print(f"Epoch {epoch}: mean_reward={sum(rewards)/len(rewards):.3f}, "
          f"kl={stats.get('objective/kl', 0):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-35-chapter-35.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — التعلّم المعزَّز من التغذية الراجعة البشرية (RLHF)
Large Language Model Engineering — RLHF

Book slug : llm-engineering-69-chapters
Chapter   : ch-35-chapter-35 (lesson 35)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_35.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 35: الفصل 35: التعلّم المعزَّز من التغذية الراجعة البشرية (RLHF)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_rlhf() -> None:
    rewards, kl_penalty = ({'helpful': 1.4, 'unsafe': -1.0}, 0.2)
    reference_drift = {'helpful': 0.3, 'unsafe': 0.8}
    objective = {name: reward - kl_penalty * reference_drift[name] for name, reward in rewards.items()}
    print('reward scores:', rewards)
    print('KL-regularized objective:', {key: round(value, 3) for key, value in objective.items()})

if __name__ == "__main__":
    print("Chapter 35 lab: Compare preference rewards and a KL constraint")
    run_rlhf()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-07-fine-tuning-alignment/chapter-35.md
# ────────────────────────────────────────────────────────────────────────
"""
تدريب نموذج المكافأة (Reward Model) من بيانات تفضيل بشرية
Python 3.13+ | PyTorch 2.x | transformers
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader


class RewardModel(nn.Module):
    """
    نموذج مكافأة: يُغلّف نموذجًا لغويًا أساسيًا ويستبدل رأس توليد
    اللغة برأس Scalar واحد يُنتج درجة تقييم واحدة لكل تسلسل.
    """

    def __init__(self, base_model_name: str):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.backbone.config.hidden_size
        # رأس المكافأة: يُسقط التمثيل المخفي النهائي إلى رقم واحد
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)

        # نأخذ تمثيل آخر توكن فعلي (غير Padding) لكل تسلسل
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_hidden = hidden_states[batch_indices, sequence_lengths]  # (batch, hidden)

        reward = self.reward_head(last_hidden).squeeze(-1)  # (batch,)
        return reward


@dataclass
class PreferencePair:
    prompt: str
    chosen: str      # الاستجابة المفضَّلة (y_w)
    rejected: str     # الاستجابة المرفوضة (y_l)


class PreferenceDataset(Dataset):
    def __init__(self, pairs: list[PreferencePair], tokenizer, max_length: int = 512):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        chosen_text = f"{pair.prompt}\n{pair.chosen}"
        rejected_text = f"{pair.prompt}\n{pair.rejected}"

        chosen_enc = self.tokenizer(
            chosen_text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        rejected_enc = self.tokenizer(
            rejected_text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen_enc.input_ids.squeeze(0),
            "chosen_attention_mask": chosen_enc.attention_mask.squeeze(0),
            "rejected_input_ids": rejected_enc.input_ids.squeeze(0),
            "rejected_attention_mask": rejected_enc.attention_mask.squeeze(0),
        }


def reward_model_loss(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
    """
    دالة خسارة Bradley-Terry:
    L = -log(sigmoid(r_chosen - r_rejected))
    """
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()


def train_reward_model(
    model: RewardModel,
    dataloader: DataLoader,
    num_epochs: int = 3,
    learning_rate: float = 1e-5,
    device: str = "cuda",
) -> RewardModel:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        total_loss, correct_predictions, total_samples = 0.0, 0, 0

        for batch in dataloader:
            chosen_rewards = model(
                batch["chosen_input_ids"].to(device),
                batch["chosen_attention_mask"].to(device),
            )
            rejected_rewards = model(
                batch["rejected_input_ids"].to(device),
                batch["rejected_attention_mask"].to(device),
            )

            loss = reward_model_loss(chosen_rewards, rejected_rewards)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            # دقة: كم مرة r_chosen > r_rejected فعليًا (مؤشر صحة مباشر)
            correct_predictions += (chosen_rewards > rejected_rewards).sum().item()
            total_samples += chosen_rewards.size(0)

        accuracy = correct_predictions / total_samples
        print(f"Epoch {epoch+1}: Loss={total_loss/len(dataloader):.4f}, "
              f"Preference Accuracy={accuracy:.2%}")

    return model

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-07-fine-tuning-alignment/chapter-35.md
# ────────────────────────────────────────────────────────────────────────
"""
تطبيق مبسّط لحلقة PPO لتحسين سياسة نموذج لغوي
يُظهر الآلية الكاملة: 4 نماذج تعمل معًا
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy


@dataclass
class PPOConfig:
    kl_coef: float = 0.1          # β في معادلة المكافأة المُعدَّلة
    clip_epsilon: float = 0.2     # ε في Clipped Objective
    gamma: float = 1.0            # عامل الخصم (عادة 1.0 لمهام توليد النص القصيرة)
    gae_lambda: float = 0.95      # λ في GAE
    ppo_epochs: int = 4           # عدد تكرارات التحديث على نفس الدفعة
    learning_rate: float = 1e-6
    value_loss_coef: float = 0.5


class PPOTrainerFromScratch:
    """
    تطبيق تعليمي مبسّط لـ PPO يُظهر الأربعة نماذج والعلاقة بينها.
    (في الإنتاج الفعلي، استخدم مكتبة trl.PPOTrainer المُحسَّنة والمُختبَرة جيدًا)
    """

    def __init__(
        self,
        policy_model: AutoModelForCausalLM,   # السياسة القابلة للتدريب π_θ
        reward_model,                          # نموذج المكافأة r_φ (مجمَّد)
        tokenizer,
        config: PPOConfig,
        device: str = "cuda",
    ):
        self.device = device
        self.config = config
        self.tokenizer = tokenizer

        self.policy = policy_model.to(device)

        # Reference Model: نسخة مجمَّدة تمامًا من السياسة الأصلية (بعد SFT)
        self.reference_model = copy.deepcopy(policy_model).to(device)
        for p in self.reference_model.parameters():
            p.requires_grad = False
        self.reference_model.eval()

        self.reward_model = reward_model.to(device)
        for p in self.reward_model.parameters():
            p.requires_grad = False
        self.reward_model.eval()

        # Value Head: رأس إضافي بسيط فوق نفس backbone السياسة لتقدير V(s)
        hidden_size = self.policy.config.hidden_size
        self.value_head = torch.nn.Linear(hidden_size, 1).to(device)

        self.optimizer = torch.optim.AdamW(
            list(self.policy.parameters()) + list(self.value_head.parameters()),
            lr=config.learning_rate,
        )

    def generate_rollouts(self, prompts: list[str], max_new_tokens: int = 64) -> dict:
        """المرحلة 1 من كل تكرار: توليد استجابات من السياسة الحالية."""
        self.policy.eval()
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            generated = self.policy.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=1.0, top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        full_texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return {"input_ids": generated, "attention_mask": (generated != self.tokenizer.pad_token_id).long(),
                "texts": full_texts}

    def compute_rewards_and_kl(self, rollout: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """المرحلة 2: حساب المكافأة الخام من Reward Model + عقوبة KL من Reference Model."""
        input_ids = rollout["input_ids"]
        attention_mask = rollout["attention_mask"]

        with torch.no_grad():
            # مكافأة سكالر واحدة لكل تسلسل كامل من Reward Model
            raw_rewards = self.reward_model(input_ids, attention_mask)

            # احتمالات log لكل توكن من السياسة الحالية والـ Reference
            policy_logits = self.policy(input_ids, attention_mask=attention_mask).logits
            ref_logits = self.reference_model(input_ids, attention_mask=attention_mask).logits

            policy_log_probs = F.log_softmax(policy_logits, dim=-1)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)

            # نأخذ log-prob للتوكن الفعلي المُولَّد في كل موضع
            token_policy_lp = torch.gather(
                policy_log_probs[:, :-1], 2, input_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            token_ref_lp = torch.gather(
                ref_log_probs[:, :-1], 2, input_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)

            # تقدير KL لكل توكن (Schulman's k1 approximation)
            per_token_kl = token_policy_lp - token_ref_lp

        return raw_rewards, per_token_kl

    def compute_advantages_gae(
        self, rewards: torch.Tensor, values: torch.Tensor, per_token_kl: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """المرحلة 3: دمج المكافأة مع عقوبة KL، ثم حساب Advantage عبر GAE."""
        # المكافأة النهائية: فقط على آخر توكن + عقوبة KL موزَّعة على كل التوكنز
        seq_len = per_token_kl.size(1)
        token_rewards = -self.config.kl_coef * per_token_kl
        token_rewards[:, -1] += rewards  # المكافأة الكلية تُضاف فقط عند نهاية التسلسل

        advantages = torch.zeros_like(token_rewards)
        last_advantage = 0.0
        last_value = 0.0

        for t in reversed(range(seq_len)):
            next_value = values[:, t + 1] if t < seq_len - 1 else last_value
            delta = token_rewards[:, t] + self.config.gamma * next_value - values[:, t]
            last_advantage = delta + self.config.gamma * self.config.gae_lambda * last_advantage
            advantages[:, t] = last_advantage

        returns = advantages + values
        # تطبيع Advantage (ممارسة قياسية لتحسين استقرار التدريب)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def ppo_update_step(
        self, rollout: dict, old_log_probs: torch.Tensor,
        advantages: torch.Tensor, returns: torch.Tensor,
    ) -> dict:
        """المرحلة 4: التحديث الفعلي بالدالة الهدف المُقتَصَّة (Clipped Objective)."""
        self.policy.train()
        input_ids = rollout["input_ids"]
        attention_mask = rollout["attention_mask"]

        for _ in range(self.config.ppo_epochs):
            outputs = self.policy(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)
            new_log_probs = torch.gather(
                log_probs[:, :-1], 2, input_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)

            # القيمة المُقدَّرة الجديدة من Value Head
            hidden = outputs.hidden_states[-1][:, :-1]
            new_values = self.value_head(hidden).squeeze(-1)

            # نسبة الاحتمال r_t(θ)
            ratio = torch.exp(new_log_probs - old_log_probs)

            # الدالة الهدف المُقتَصَّة
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon
            ) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # خسارة Value Head (تدريب Critic ليُقدّر العوائد بدقة)
            value_loss = F.mse_loss(new_values, returns)

            total_loss = policy_loss + self.config.value_loss_coef * value_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.policy.parameters()) + list(self.value_head.parameters()), max_norm=1.0
            )
            self.optimizer.step()

        return {"policy_loss": policy_loss.item(), "value_loss": value_loss.item()}

    def train_step(self, prompts: list[str]) -> dict:
        """تكرار كامل واحد من RLHF: Rollout → Reward → Advantage → Update."""
        rollout = self.generate_rollouts(prompts)
        rewards, per_token_kl = self.compute_rewards_and_kl(rollout)

        with torch.no_grad():
            outputs = self.policy(rollout["input_ids"], output_hidden_states=True)
            old_log_probs_full = F.log_softmax(outputs.logits, dim=-1)
            old_log_probs = torch.gather(
                old_log_probs_full[:, :-1], 2, rollout["input_ids"][:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            old_values = self.value_head(outputs.hidden_states[-1][:, :-1]).squeeze(-1)

        advantages, returns = self.compute_advantages_gae(rewards, old_values, per_token_kl)
        stats = self.ppo_update_step(rollout, old_log_probs, advantages, returns)
        stats["mean_reward"] = rewards.mean().item()
        stats["mean_kl"] = per_token_kl.mean().item()
        return stats

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-07-fine-tuning-alignment/chapter-35.md
# ────────────────────────────────────────────────────────────────────────
"""
استخدام مكتبة trl لتطبيق RLHF كامل في الإنتاج
pip install trl transformers peft
"""

from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
import torch


MODEL_NAME = "./sft-model"  # نموذج تم ضبطه بـ SFT مسبقًا (الفصل 34)

# النموذج مع Value Head مدمج تلقائيًا (بدل بنائه يدويًا كما فعلنا أعلاه)
policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME)
reference_model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

ppo_config = PPOConfig(
    learning_rate=1.41e-5,
    batch_size=64,
    mini_batch_size=8,
    ppo_epochs=4,
    kl_penalty="kl",       # نوع تقدير الـ KL
    init_kl_coef=0.2,      # β الأولي، يُعدَّل تلقائيًا (Adaptive KL Controller)
    target_kl=6.0,         # القيمة المستهدفة لـ KL — يُضبط β تلقائيًا للبقاء قربها
    gamma=1.0,
    lam=0.95,              # λ في GAE
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=policy_model,
    ref_model=reference_model,
    tokenizer=tokenizer,
)

# حلقة تدريب مبسّطة (في الإنتاج: آلاف التكرارات على بيانات prompts متنوعة)
prompts = ["اشرح لي أهمية الذكاء الاصطناعي", "كيف أتعلم البرمجة بسرعة؟"]

for epoch in range(10):
    query_tensors = [tokenizer(p, return_tensors="pt").input_ids[0] for p in prompts]

    # 1) توليد استجابات
    response_tensors = ppo_trainer.generate(
        query_tensors, max_new_tokens=64, do_sample=True, top_p=0.9,
    )
    responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

    # 2) حساب المكافآت من Reward Model المدرَّب مسبقًا (دالة خارجية في مشروع حقيقي)
    rewards = [compute_reward_from_model(q, r) for q, r in zip(prompts, responses)]
    reward_tensors = [torch.tensor(r) for r in rewards]

    # 3) خطوة PPO واحدة — المكتبة تُدير GAE، Clipping، KL Control داخليًا
    stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
    ppo_trainer.log_stats(stats, {"query": prompts, "response": responses}, rewards)

    print(f"Epoch {epoch}: mean_reward={sum(rewards)/len(rewards):.3f}, "
          f"kl={stats.get('objective/kl', 0):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-35-chapter-35.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 35: الفصل 35: التعلّم المعزَّز من التغذية الراجعة البشرية (RLHF)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_rlhf() -> None:
    rewards, kl_penalty = ({'helpful': 1.4, 'unsafe': -1.0}, 0.2)
    reference_drift = {'helpful': 0.3, 'unsafe': 0.8}
    objective = {name: reward - kl_penalty * reference_drift[name] for name, reward in rewards.items()}
    print('reward scores:', rewards)
    print('KL-regularized objective:', {key: round(value, 3) for key, value in objective.items()})

if __name__ == "__main__":
    print("Chapter 35 lab: Compare preference rewards and a KL constraint")
    run_rlhf()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-35-chapter-35.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
تدريب نموذج المكافأة (Reward Model) من بيانات تفضيل بشرية
Python 3.13+ | PyTorch 2.x | transformers
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader


class RewardModel(nn.Module):
    """
    نموذج مكافأة: يُغلّف نموذجًا لغويًا أساسيًا ويستبدل رأس توليد
    اللغة برأس Scalar واحد يُنتج درجة تقييم واحدة لكل تسلسل.
    """

    def __init__(self, base_model_name: str):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.backbone.config.hidden_size
        # رأس المكافأة: يُسقط التمثيل المخفي النهائي إلى رقم واحد
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)

        # نأخذ تمثيل آخر توكن فعلي (غير Padding) لكل تسلسل
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_hidden = hidden_states[batch_indices, sequence_lengths]  # (batch, hidden)

        reward = self.reward_head(last_hidden).squeeze(-1)  # (batch,)
        return reward


@dataclass
class PreferencePair:
    prompt: str
    chosen: str      # الاستجابة المفضَّلة (y_w)
    rejected: str     # الاستجابة المرفوضة (y_l)


class PreferenceDataset(Dataset):
    def __init__(self, pairs: list[PreferencePair], tokenizer, max_length: int = 512):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        chosen_text = f"{pair.prompt}\n{pair.chosen}"
        rejected_text = f"{pair.prompt}\n{pair.rejected}"

        chosen_enc = self.tokenizer(
            chosen_text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        rejected_enc = self.tokenizer(
            rejected_text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen_enc.input_ids.squeeze(0),
            "chosen_attention_mask": chosen_enc.attention_mask.squeeze(0),
            "rejected_input_ids": rejected_enc.input_ids.squeeze(0),
            "rejected_attention_mask": rejected_enc.attention_mask.squeeze(0),
        }


def reward_model_loss(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
    """
    دالة خسارة Bradley-Terry:
    L = -log(sigmoid(r_chosen - r_rejected))
    """
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()


def train_reward_model(
    model: RewardModel,
    dataloader: DataLoader,
    num_epochs: int = 3,
    learning_rate: float = 1e-5,
    device: str = "cuda",
) -> RewardModel:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        total_loss, correct_predictions, total_samples = 0.0, 0, 0

        for batch in dataloader:
            chosen_rewards = model(
                batch["chosen_input_ids"].to(device),
                batch["chosen_attention_mask"].to(device),
            )
            rejected_rewards = model(
                batch["rejected_input_ids"].to(device),
                batch["rejected_attention_mask"].to(device),
            )

            loss = reward_model_loss(chosen_rewards, rejected_rewards)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            # دقة: كم مرة r_chosen > r_rejected فعليًا (مؤشر صحة مباشر)
            correct_predictions += (chosen_rewards > rejected_rewards).sum().item()
            total_samples += chosen_rewards.size(0)

        accuracy = correct_predictions / total_samples
        print(f"Epoch {epoch+1}: Loss={total_loss/len(dataloader):.4f}, "
              f"Preference Accuracy={accuracy:.2%}")

    return model

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-35-chapter-35.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
"""
تطبيق مبسّط لحلقة PPO لتحسين سياسة نموذج لغوي
يُظهر الآلية الكاملة: 4 نماذج تعمل معًا
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy


@dataclass
class PPOConfig:
    kl_coef: float = 0.1          # β في معادلة المكافأة المُعدَّلة
    clip_epsilon: float = 0.2     # ε في Clipped Objective
    gamma: float = 1.0            # عامل الخصم (عادة 1.0 لمهام توليد النص القصيرة)
    gae_lambda: float = 0.95      # λ في GAE
    ppo_epochs: int = 4           # عدد تكرارات التحديث على نفس الدفعة
    learning_rate: float = 1e-6
    value_loss_coef: float = 0.5


class PPOTrainerFromScratch:
    """
    تطبيق تعليمي مبسّط لـ PPO يُظهر الأربعة نماذج والعلاقة بينها.
    (في الإنتاج الفعلي، استخدم مكتبة trl.PPOTrainer المُحسَّنة والمُختبَرة جيدًا)
    """

    def __init__(
        self,
        policy_model: AutoModelForCausalLM,   # السياسة القابلة للتدريب π_θ
        reward_model,                          # نموذج المكافأة r_φ (مجمَّد)
        tokenizer,
        config: PPOConfig,
        device: str = "cuda",
    ):
        self.device = device
        self.config = config
        self.tokenizer = tokenizer

        self.policy = policy_model.to(device)

        # Reference Model: نسخة مجمَّدة تمامًا من السياسة الأصلية (بعد SFT)
        self.reference_model = copy.deepcopy(policy_model).to(device)
        for p in self.reference_model.parameters():
            p.requires_grad = False
        self.reference_model.eval()

        self.reward_model = reward_model.to(device)
        for p in self.reward_model.parameters():
            p.requires_grad = False
        self.reward_model.eval()

        # Value Head: رأس إضافي بسيط فوق نفس backbone السياسة لتقدير V(s)
        hidden_size = self.policy.config.hidden_size
        self.value_head = torch.nn.Linear(hidden_size, 1).to(device)

        self.optimizer = torch.optim.AdamW(
            list(self.policy.parameters()) + list(self.value_head.parameters()),
            lr=config.learning_rate,
        )

    def generate_rollouts(self, prompts: list[str], max_new_tokens: int = 64) -> dict:
        """المرحلة 1 من كل تكرار: توليد استجابات من السياسة الحالية."""
        self.policy.eval()
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            generated = self.policy.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=1.0, top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        full_texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return {"input_ids": generated, "attention_mask": (generated != self.tokenizer.pad_token_id).long(),
                "texts": full_texts}

    def compute_rewards_and_kl(self, rollout: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """المرحلة 2: حساب المكافأة الخام من Reward Model + عقوبة KL من Reference Model."""
        input_ids = rollout["input_ids"]
        attention_mask = rollout["attention_mask"]

        with torch.no_grad():
            # مكافأة سكالر واحدة لكل تسلسل كامل من Reward Model
            raw_rewards = self.reward_model(input_ids, attention_mask)

            # احتمالات log لكل توكن من السياسة الحالية والـ Reference
            policy_logits = self.policy(input_ids, attention_mask=attention_mask).logits
            ref_logits = self.reference_model(input_ids, attention_mask=attention_mask).logits

            policy_log_probs = F.log_softmax(policy_logits, dim=-1)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)

            # نأخذ log-prob للتوكن الفعلي المُولَّد في كل موضع
            token_policy_lp = torch.gather(
                policy_log_probs[:, :-1], 2, input_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            token_ref_lp = torch.gather(
                ref_log_probs[:, :-1], 2, input_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)

            # تقدير KL لكل توكن (Schulman's k1 approximation)
            per_token_kl = token_policy_lp - token_ref_lp

        return raw_rewards, per_token_kl

    def compute_advantages_gae(
        self, rewards: torch.Tensor, values: torch.Tensor, per_token_kl: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """المرحلة 3: دمج المكافأة مع عقوبة KL، ثم حساب Advantage عبر GAE."""
        # المكافأة النهائية: فقط على آخر توكن + عقوبة KL موزَّعة على كل التوكنز
        seq_len = per_token_kl.size(1)
        token_rewards = -self.config.kl_coef * per_token_kl
        token_rewards[:, -1] += rewards  # المكافأة الكلية تُضاف فقط عند نهاية التسلسل

        advantages = torch.zeros_like(token_rewards)
        last_advantage = 0.0
        last_value = 0.0

        for t in reversed(range(seq_len)):
            next_value = values[:, t + 1] if t < seq_len - 1 else last_value
            delta = token_rewards[:, t] + self.config.gamma * next_value - values[:, t]
            last_advantage = delta + self.config.gamma * self.config.gae_lambda * last_advantage
            advantages[:, t] = last_advantage

        returns = advantages + values
        # تطبيع Advantage (ممارسة قياسية لتحسين استقرار التدريب)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def ppo_update_step(
        self, rollout: dict, old_log_probs: torch.Tensor,
        advantages: torch.Tensor, returns: torch.Tensor,
    ) -> dict:
        """المرحلة 4: التحديث الفعلي بالدالة الهدف المُقتَصَّة (Clipped Objective)."""
        self.policy.train()
        input_ids = rollout["input_ids"]
        attention_mask = rollout["attention_mask"]

        for _ in range(self.config.ppo_epochs):
            outputs = self.policy(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)
            new_log_probs = torch.gather(
                log_probs[:, :-1], 2, input_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)

            # القيمة المُقدَّرة الجديدة من Value Head
            hidden = outputs.hidden_states[-1][:, :-1]
            new_values = self.value_head(hidden).squeeze(-1)

            # نسبة الاحتمال r_t(θ)
            ratio = torch.exp(new_log_probs - old_log_probs)

            # الدالة الهدف المُقتَصَّة
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon
            ) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # خسارة Value Head (تدريب Critic ليُقدّر العوائد بدقة)
            value_loss = F.mse_loss(new_values, returns)

            total_loss = policy_loss + self.config.value_loss_coef * value_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.policy.parameters()) + list(self.value_head.parameters()), max_norm=1.0
            )
            self.optimizer.step()

        return {"policy_loss": policy_loss.item(), "value_loss": value_loss.item()}

    def train_step(self, prompts: list[str]) -> dict:
        """تكرار كامل واحد من RLHF: Rollout → Reward → Advantage → Update."""
        rollout = self.generate_rollouts(prompts)
        rewards, per_token_kl = self.compute_rewards_and_kl(rollout)

        with torch.no_grad():
            outputs = self.policy(rollout["input_ids"], output_hidden_states=True)
            old_log_probs_full = F.log_softmax(outputs.logits, dim=-1)
            old_log_probs = torch.gather(
                old_log_probs_full[:, :-1], 2, rollout["input_ids"][:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            old_values = self.value_head(outputs.hidden_states[-1][:, :-1]).squeeze(-1)

        advantages, returns = self.compute_advantages_gae(rewards, old_values, per_token_kl)
        stats = self.ppo_update_step(rollout, old_log_probs, advantages, returns)
        stats["mean_reward"] = rewards.mean().item()
        stats["mean_kl"] = per_token_kl.mean().item()
        return stats

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-35-chapter-35.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
"""
استخدام مكتبة trl لتطبيق RLHF كامل في الإنتاج
pip install trl transformers peft
"""

from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
import torch


MODEL_NAME = "./sft-model"  # نموذج تم ضبطه بـ SFT مسبقًا (الفصل 34)

# النموذج مع Value Head مدمج تلقائيًا (بدل بنائه يدويًا كما فعلنا أعلاه)
policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME)
reference_model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

ppo_config = PPOConfig(
    learning_rate=1.41e-5,
    batch_size=64,
    mini_batch_size=8,
    ppo_epochs=4,
    kl_penalty="kl",       # نوع تقدير الـ KL
    init_kl_coef=0.2,      # β الأولي، يُعدَّل تلقائيًا (Adaptive KL Controller)
    target_kl=6.0,         # القيمة المستهدفة لـ KL — يُضبط β تلقائيًا للبقاء قربها
    gamma=1.0,
    lam=0.95,              # λ في GAE
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=policy_model,
    ref_model=reference_model,
    tokenizer=tokenizer,
)

# حلقة تدريب مبسّطة (في الإنتاج: آلاف التكرارات على بيانات prompts متنوعة)
prompts = ["اشرح لي أهمية الذكاء الاصطناعي", "كيف أتعلم البرمجة بسرعة؟"]

for epoch in range(10):
    query_tensors = [tokenizer(p, return_tensors="pt").input_ids[0] for p in prompts]

    # 1) توليد استجابات
    response_tensors = ppo_trainer.generate(
        query_tensors, max_new_tokens=64, do_sample=True, top_p=0.9,
    )
    responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

    # 2) حساب المكافآت من Reward Model المدرَّب مسبقًا (دالة خارجية في مشروع حقيقي)
    rewards = [compute_reward_from_model(q, r) for q, r in zip(prompts, responses)]
    reward_tensors = [torch.tensor(r) for r in rewards]

    # 3) خطوة PPO واحدة — المكتبة تُدير GAE، Clipping، KL Control داخليًا
    stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
    ppo_trainer.log_stats(stats, {"query": prompts, "response": responses}, rewards)

    print(f"Epoch {epoch}: mean_reward={sum(rewards)/len(rewards):.3f}, "
          f"kl={stats.get('objective/kl', 0):.3f}")
