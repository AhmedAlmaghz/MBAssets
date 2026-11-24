#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — RLHF وPPO بعمق
Mastering Transformers and Large Language Models — Rlhf Ppo

Book slug : mastering-llms-transformers
Chapter   : ch-33-ch33-rlhf-ppo (lesson 33)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-33-ch33-rlhf-ppo.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — RLHF وPPO بعمق
Mastering Transformers and Large Language Models — Rlhf Ppo

Book slug : mastering-llms-transformers
Chapter   : ch-33-ch33-rlhf-ppo (lesson 33)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-33-ch33-rlhf-ppo.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — RLHF وPPO بعمق
Mastering Transformers and Large Language Models — Rlhf Ppo

Book slug : mastering-llms-transformers
Chapter   : ch-33-ch33-rlhf-ppo (lesson 33)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-07-peft-alignment/ch33-rlhf-ppo.ipynb#cell-1
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
# [002] content/notebooks/part-07-peft-alignment/ch33-rlhf-ppo.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel(nn.Module):
    """
    نموذج المكافأة: يأخذ نفس معمارية LLM الأساسي (عادة نفس SFT model)
    لكن نستبدل رأس اللغة (LM Head) برأس انحدار قياسي (scalar head)
    يُخرِج رقمًا واحدًا يُمثِّل "جودة" الاستجابة الكاملة.
    """
    def __init__(self, base_transformer: nn.Module, d_model: int):
        super().__init__()
        self.transformer = base_transformer  # عادة مُهيَّأ من نموذج SFT
        self.reward_head = nn.Linear(d_model, 1)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state  # (batch, seq_len, d_model)

        # أخذ تمثيل آخر رمز فعلي (غير padding) لكل تسلسل
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(input_ids.shape[0])
        last_token_hidden = hidden_states[batch_indices, seq_lengths]

        reward = self.reward_head(last_token_hidden)  # (batch, 1)
        return reward.squeeze(-1)  # (batch,)

def bradley_terry_reward_loss(reward_chosen: torch.Tensor,
                                reward_rejected: torch.Tensor) -> torch.Tensor:
    """
    دالة خسارة نموذج المكافأة المبنية على نموذج Bradley-Terry الإحصائي
    لاحتمالية التفضيل الزوجي:

        P(y_w ≻ y_l | x) = σ(r(x, y_w) - r(x, y_l))

    حيث y_w هي الاستجابة المُفضَّلة (winner) وy_l هي المرفوضة (loser).
    نُعظِّم log-likelihood هذا الاحتمال، أو بشكل مكافئ، نُصغِّر:
    """
    loss = -F.logsigmoid(reward_chosen - reward_rejected).mean()
    return loss

class PreferenceDataCollector:
    """
    محاكاة عملية جمع بيانات التفضيل البشري، كما في منهجية
    InstructGPT: يُعرَض على المُقيِّم البشري عدة استجابات (k=4 إلى 9)
    لنفس التعليمة، ويُرتِّبها من الأفضل إلى الأسوأ.
    من الترتيب الكلي، نستخرج أزواج (chosen, rejected) للتدريب.
    """
    @staticmethod
    def rankings_to_pairs(ranked_responses: list) -> list:
        """
        تحويل ترتيب كلي (مثال: [r1 > r3 > r2 > r4]) إلى كل الأزواج الممكنة
        (C(k,2) زوجًا)، كما فعلت ورقة InstructGPT لزيادة كفاءة البيانات.
        """
        pairs = []
        for i in range(len(ranked_responses)):
            for j in range(i + 1, len(ranked_responses)):
                pairs.append({
                    "chosen": ranked_responses[i],
                    "rejected": ranked_responses[j],
                })
        return pairs

def train_reward_model_step(reward_model: RewardModel, optimizer,
                              chosen_batch: dict, rejected_batch: dict):
    """خطوة تدريب واحدة كاملة لنموذج المكافأة."""
    optimizer.zero_grad()

    reward_chosen = reward_model(
        chosen_batch["input_ids"], chosen_batch["attention_mask"]
    )
    reward_rejected = reward_model(
        rejected_batch["input_ids"], rejected_batch["attention_mask"]
    )

    loss = bradley_terry_reward_loss(reward_chosen, reward_rejected)
    loss.backward()
    optimizer.step()

    # مقياس دقة مفيد للمراقبة: هل RM يُرتِّب الأزواج بشكل صحيح؟
    accuracy = (reward_chosen > reward_rejected).float().mean()
    return {"loss": loss.item(), "accuracy": accuracy.item()}

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-07-peft-alignment/ch33-rlhf-ppo.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class PPOTrainerForLLM:
    """
    تطبيق مبسّط تعليميًا لحلقة تدريب PPO المُستخدَمة في RLHF.
    يتضمن أربعة نماذج يجب تشغيلها معًا:
    1. Policy Model (π_θ) — النموذج الذي نُحسِّنه (مُهيَّأ من SFT)
    2. Reference Model (π_ref) — نسخة مجمّدة من SFT (لحساب KL)
    3. Reward Model (RM) — مُجمَّد، يُقيِّم جودة الاستجابات
    4. Value Model (V) — يُقدِّر "القيمة المتوقعة" لحالة معينة (Critic)
    """
    def __init__(self, policy_model, reference_model, reward_model,
                 value_model, tokenizer, kl_coef: float = 0.1,
                 clip_epsilon: float = 0.2):
        self.policy = policy_model
        self.reference = reference_model  # مُجمَّد تمامًا، لا تدرجات
        self.reward_model = reward_model  # مُجمَّد تمامًا
        self.value_model = value_model
        self.tokenizer = tokenizer
        self.kl_coef = kl_coef
        self.clip_epsilon = clip_epsilon

        for p in self.reference.parameters():
            p.requires_grad = False
        for p in self.reward_model.parameters():
            p.requires_grad = False

    def compute_rewards_with_kl_penalty(self, prompts, responses,
                                          response_logprobs, ref_logprobs):
        """
        المكافأة الفعلية المُستخدَمة في PPO ليست فقط مخرج RM،
        بل RM مطروحًا منها عقوبة KL-divergence عن السياسة المرجعية.
        هذا يمنع النموذج من الانحراف الجذري بعيدًا عن سلوك SFT
        الآمن والمعقول سعيًا وراء "خداع" نموذج المكافأة:

            R(x,y) = RM(x,y) - β · KL(π_θ(y|x) || π_ref(y|x))

        حيث β هو kl_coef، وKL هنا تُقدَّر تجريبيًا كفرق log-probabilities.
        """
        full_sequences = [p + r for p, r in zip(prompts, responses)]
        inputs = self.tokenizer(full_sequences, return_tensors="pt",
                                  padding=True, truncation=True)

        with torch.no_grad():
            rm_scores = self.reward_model(
                inputs["input_ids"], inputs["attention_mask"]
            )

        # تقدير KL لكل رمز: log π_θ(token) - log π_ref(token)
        kl_per_token = response_logprobs - ref_logprobs
        kl_penalty = self.kl_coef * kl_per_token.sum(dim=-1)

        total_reward = rm_scores - kl_penalty
        return total_reward, kl_penalty

    def compute_advantages_gae(self, rewards, values, gamma=0.99, lam=0.95):
        """
        Generalized Advantage Estimation (GAE): تقدير "الميزة" (Advantage)
        لكل خطوة زمنية (رمز)، يُوازن بين الانحياز والتباين في تقدير
        مدى "جودة" إجراء مُعيَّن مقارنة بالمتوسط المتوقع.

            δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
            A_t = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...
        """
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t < len(rewards) - 1 else 0
            delta = rewards[t] + gamma * next_value - values[t]
            last_gae = delta + gamma * lam * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        return advantages, returns

    def ppo_clipped_objective(self, old_logprobs, new_logprobs, advantages):
        """
        دالة الهدف المُقيَّدة لـ PPO (الابتكار الجوهري للخوارزمية):
        تمنع التحديثات الكبيرة جدًا للسياسة في خطوة واحدة عبر
        "قصّ" نسبة الاحتمالات الجديدة/القديمة ضمن [1-ε, 1+ε]،
        مما يضمن استقرار التدريب (Proximal = "قريب من السياسة القديمة").

            L^CLIP(θ) = E[min(r_t(θ)·A_t, clip(r_t(θ), 1-ε, 1+ε)·A_t)]

            حيث r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
        """
        ratio = torch.exp(new_logprobs - old_logprobs)  # نسبة الاحتمالات

        unclipped = ratio * advantages
        clipped = torch.clamp(
            ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
        ) * advantages

        # نأخذ min لضمان أننا لا نُحسِّن الهدف أكثر من اللازم
        # (pessimistic bound يضمن استقرار تحديثات السياسة)
        policy_loss = -torch.min(unclipped, clipped).mean()
        return policy_loss

    def training_step(self, prompts: list):
        """خطوة PPO كاملة واحدة (مُبسَّطة تعليميًا)."""
        # 1. توليد استجابات من السياسة الحالية
        responses, response_logprobs = self._generate_with_logprobs(prompts)

        # 2. حساب log-probs من النموذج المرجعي لنفس الاستجابات (لـ KL)
        with torch.no_grad():
            ref_logprobs = self._compute_logprobs(
                self.reference, prompts, responses
            )

        # 3. حساب المكافأة مع عقوبة KL
        rewards, kl_penalty = self.compute_rewards_with_kl_penalty(
            prompts, responses, response_logprobs, ref_logprobs
        )

        # 4. حساب القيم (Critic) والميزات (Advantages)
        values = self.value_model(prompts, responses)
        advantages, returns = self.compute_advantages_gae(rewards, values)

        # 5. عدة تكرارات PPO على نفس الدفعة (عادة 4 epochs داخلية)
        for _ in range(4):
            new_logprobs = self._compute_logprobs(
                self.policy, prompts, responses
            )
            policy_loss = self.ppo_clipped_objective(
                response_logprobs.detach(), new_logprobs, advantages.detach()
            )
            value_loss = F.mse_loss(
                self.value_model(prompts, responses), returns.detach()
            )
            total_loss = policy_loss + 0.5 * value_loss
            total_loss.backward()
            # optimizer.step() ...

        return {
            "policy_loss": policy_loss.item(),
            "mean_reward": rewards.mean().item(),
            "mean_kl": kl_penalty.mean().item(),
        }

    def _generate_with_logprobs(self, prompts):
        raise NotImplementedError("تفاصيل التوليد الفعلي تعتمد على المكتبة")

    def _compute_logprobs(self, model, prompts, responses):
        raise NotImplementedError("حساب log P(token) لكل رمز في الاستجابة")

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-07-peft-alignment/ch33-rlhf-ppo.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
reward_pairs = [
    {"chosen": 2.3, "rejected": 0.8},   # RM واثق وصحيح
    {"chosen": 1.1, "rejected": 1.4},   # RM مُخطئ (الترتيب معكوس!)
    {"chosen": 0.2, "rejected": -0.1},  # RM صحيح لكن بثقة منخفضة
]

for i, pair in enumerate(reward_pairs):
    r_chosen = torch.tensor(pair["chosen"])
    r_rejected = torch.tensor(pair["rejected"])
    loss = -F.logsigmoid(r_chosen - r_rejected)
    correct = r_chosen > r_rejected
    print(f"زوج {i+1}: loss={loss.item():.4f}, "
          f"ترتيب صحيح؟ {correct.item()}")

# زوج 1: loss=0.1690, ترتيب صحيح؟ True
# زوج 2: loss=0.8544, ترتيب صحيح؟ False   <- خسارة أعلى بكثير لأن الترتيب خاطئ
# زوج 3: loss=0.5981, ترتيب صحيح؟ True    <- صحيح لكن الفارق صغير فالخسارة متوسطة

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-07-peft-alignment/ch33-rlhf-ppo.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-33-ch33-rlhf-ppo.ipynb#cell-1
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-33-ch33-rlhf-ppo.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel(nn.Module):
    """
    نموذج المكافأة: يأخذ نفس معمارية LLM الأساسي (عادة نفس SFT model)
    لكن نستبدل رأس اللغة (LM Head) برأس انحدار قياسي (scalar head)
    يُخرِج رقمًا واحدًا يُمثِّل "جودة" الاستجابة الكاملة.
    """
    def __init__(self, base_transformer: nn.Module, d_model: int):
        super().__init__()
        self.transformer = base_transformer  # عادة مُهيَّأ من نموذج SFT
        self.reward_head = nn.Linear(d_model, 1)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state  # (batch, seq_len, d_model)

        # أخذ تمثيل آخر رمز فعلي (غير padding) لكل تسلسل
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(input_ids.shape[0])
        last_token_hidden = hidden_states[batch_indices, seq_lengths]

        reward = self.reward_head(last_token_hidden)  # (batch, 1)
        return reward.squeeze(-1)  # (batch,)

def bradley_terry_reward_loss(reward_chosen: torch.Tensor,
                                reward_rejected: torch.Tensor) -> torch.Tensor:
    """
    دالة خسارة نموذج المكافأة المبنية على نموذج Bradley-Terry الإحصائي
    لاحتمالية التفضيل الزوجي:

        P(y_w ≻ y_l | x) = σ(r(x, y_w) - r(x, y_l))

    حيث y_w هي الاستجابة المُفضَّلة (winner) وy_l هي المرفوضة (loser).
    نُعظِّم log-likelihood هذا الاحتمال، أو بشكل مكافئ، نُصغِّر:
    """
    loss = -F.logsigmoid(reward_chosen - reward_rejected).mean()
    return loss

class PreferenceDataCollector:
    """
    محاكاة عملية جمع بيانات التفضيل البشري، كما في منهجية
    InstructGPT: يُعرَض على المُقيِّم البشري عدة استجابات (k=4 إلى 9)
    لنفس التعليمة، ويُرتِّبها من الأفضل إلى الأسوأ.
    من الترتيب الكلي، نستخرج أزواج (chosen, rejected) للتدريب.
    """
    @staticmethod
    def rankings_to_pairs(ranked_responses: list) -> list:
        """
        تحويل ترتيب كلي (مثال: [r1 > r3 > r2 > r4]) إلى كل الأزواج الممكنة
        (C(k,2) زوجًا)، كما فعلت ورقة InstructGPT لزيادة كفاءة البيانات.
        """
        pairs = []
        for i in range(len(ranked_responses)):
            for j in range(i + 1, len(ranked_responses)):
                pairs.append({
                    "chosen": ranked_responses[i],
                    "rejected": ranked_responses[j],
                })
        return pairs

def train_reward_model_step(reward_model: RewardModel, optimizer,
                              chosen_batch: dict, rejected_batch: dict):
    """خطوة تدريب واحدة كاملة لنموذج المكافأة."""
    optimizer.zero_grad()

    reward_chosen = reward_model(
        chosen_batch["input_ids"], chosen_batch["attention_mask"]
    )
    reward_rejected = reward_model(
        rejected_batch["input_ids"], rejected_batch["attention_mask"]
    )

    loss = bradley_terry_reward_loss(reward_chosen, reward_rejected)
    loss.backward()
    optimizer.step()

    # مقياس دقة مفيد للمراقبة: هل RM يُرتِّب الأزواج بشكل صحيح؟
    accuracy = (reward_chosen > reward_rejected).float().mean()
    return {"loss": loss.item(), "accuracy": accuracy.item()}

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-33-ch33-rlhf-ppo.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class PPOTrainerForLLM:
    """
    تطبيق مبسّط تعليميًا لحلقة تدريب PPO المُستخدَمة في RLHF.
    يتضمن أربعة نماذج يجب تشغيلها معًا:
    1. Policy Model (π_θ) — النموذج الذي نُحسِّنه (مُهيَّأ من SFT)
    2. Reference Model (π_ref) — نسخة مجمّدة من SFT (لحساب KL)
    3. Reward Model (RM) — مُجمَّد، يُقيِّم جودة الاستجابات
    4. Value Model (V) — يُقدِّر "القيمة المتوقعة" لحالة معينة (Critic)
    """
    def __init__(self, policy_model, reference_model, reward_model,
                 value_model, tokenizer, kl_coef: float = 0.1,
                 clip_epsilon: float = 0.2):
        self.policy = policy_model
        self.reference = reference_model  # مُجمَّد تمامًا، لا تدرجات
        self.reward_model = reward_model  # مُجمَّد تمامًا
        self.value_model = value_model
        self.tokenizer = tokenizer
        self.kl_coef = kl_coef
        self.clip_epsilon = clip_epsilon

        for p in self.reference.parameters():
            p.requires_grad = False
        for p in self.reward_model.parameters():
            p.requires_grad = False

    def compute_rewards_with_kl_penalty(self, prompts, responses,
                                          response_logprobs, ref_logprobs):
        """
        المكافأة الفعلية المُستخدَمة في PPO ليست فقط مخرج RM،
        بل RM مطروحًا منها عقوبة KL-divergence عن السياسة المرجعية.
        هذا يمنع النموذج من الانحراف الجذري بعيدًا عن سلوك SFT
        الآمن والمعقول سعيًا وراء "خداع" نموذج المكافأة:

            R(x,y) = RM(x,y) - β · KL(π_θ(y|x) || π_ref(y|x))

        حيث β هو kl_coef، وKL هنا تُقدَّر تجريبيًا كفرق log-probabilities.
        """
        full_sequences = [p + r for p, r in zip(prompts, responses)]
        inputs = self.tokenizer(full_sequences, return_tensors="pt",
                                  padding=True, truncation=True)

        with torch.no_grad():
            rm_scores = self.reward_model(
                inputs["input_ids"], inputs["attention_mask"]
            )

        # تقدير KL لكل رمز: log π_θ(token) - log π_ref(token)
        kl_per_token = response_logprobs - ref_logprobs
        kl_penalty = self.kl_coef * kl_per_token.sum(dim=-1)

        total_reward = rm_scores - kl_penalty
        return total_reward, kl_penalty

    def compute_advantages_gae(self, rewards, values, gamma=0.99, lam=0.95):
        """
        Generalized Advantage Estimation (GAE): تقدير "الميزة" (Advantage)
        لكل خطوة زمنية (رمز)، يُوازن بين الانحياز والتباين في تقدير
        مدى "جودة" إجراء مُعيَّن مقارنة بالمتوسط المتوقع.

            δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
            A_t = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...
        """
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t < len(rewards) - 1 else 0
            delta = rewards[t] + gamma * next_value - values[t]
            last_gae = delta + gamma * lam * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        return advantages, returns

    def ppo_clipped_objective(self, old_logprobs, new_logprobs, advantages):
        """
        دالة الهدف المُقيَّدة لـ PPO (الابتكار الجوهري للخوارزمية):
        تمنع التحديثات الكبيرة جدًا للسياسة في خطوة واحدة عبر
        "قصّ" نسبة الاحتمالات الجديدة/القديمة ضمن [1-ε, 1+ε]،
        مما يضمن استقرار التدريب (Proximal = "قريب من السياسة القديمة").

            L^CLIP(θ) = E[min(r_t(θ)·A_t, clip(r_t(θ), 1-ε, 1+ε)·A_t)]

            حيث r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
        """
        ratio = torch.exp(new_logprobs - old_logprobs)  # نسبة الاحتمالات

        unclipped = ratio * advantages
        clipped = torch.clamp(
            ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
        ) * advantages

        # نأخذ min لضمان أننا لا نُحسِّن الهدف أكثر من اللازم
        # (pessimistic bound يضمن استقرار تحديثات السياسة)
        policy_loss = -torch.min(unclipped, clipped).mean()
        return policy_loss

    def training_step(self, prompts: list):
        """خطوة PPO كاملة واحدة (مُبسَّطة تعليميًا)."""
        # 1. توليد استجابات من السياسة الحالية
        responses, response_logprobs = self._generate_with_logprobs(prompts)

        # 2. حساب log-probs من النموذج المرجعي لنفس الاستجابات (لـ KL)
        with torch.no_grad():
            ref_logprobs = self._compute_logprobs(
                self.reference, prompts, responses
            )

        # 3. حساب المكافأة مع عقوبة KL
        rewards, kl_penalty = self.compute_rewards_with_kl_penalty(
            prompts, responses, response_logprobs, ref_logprobs
        )

        # 4. حساب القيم (Critic) والميزات (Advantages)
        values = self.value_model(prompts, responses)
        advantages, returns = self.compute_advantages_gae(rewards, values)

        # 5. عدة تكرارات PPO على نفس الدفعة (عادة 4 epochs داخلية)
        for _ in range(4):
            new_logprobs = self._compute_logprobs(
                self.policy, prompts, responses
            )
            policy_loss = self.ppo_clipped_objective(
                response_logprobs.detach(), new_logprobs, advantages.detach()
            )
            value_loss = F.mse_loss(
                self.value_model(prompts, responses), returns.detach()
            )
            total_loss = policy_loss + 0.5 * value_loss
            total_loss.backward()
            # optimizer.step() ...

        return {
            "policy_loss": policy_loss.item(),
            "mean_reward": rewards.mean().item(),
            "mean_kl": kl_penalty.mean().item(),
        }

    def _generate_with_logprobs(self, prompts):
        raise NotImplementedError("تفاصيل التوليد الفعلي تعتمد على المكتبة")

    def _compute_logprobs(self, model, prompts, responses):
        raise NotImplementedError("حساب log P(token) لكل رمز في الاستجابة")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-33-ch33-rlhf-ppo.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
reward_pairs = [
    {"chosen": 2.3, "rejected": 0.8},   # RM واثق وصحيح
    {"chosen": 1.1, "rejected": 1.4},   # RM مُخطئ (الترتيب معكوس!)
    {"chosen": 0.2, "rejected": -0.1},  # RM صحيح لكن بثقة منخفضة
]

for i, pair in enumerate(reward_pairs):
    r_chosen = torch.tensor(pair["chosen"])
    r_rejected = torch.tensor(pair["rejected"])
    loss = -F.logsigmoid(r_chosen - r_rejected)
    correct = r_chosen > r_rejected
    print(f"زوج {i+1}: loss={loss.item():.4f}, "
          f"ترتيب صحيح؟ {correct.item()}")

# زوج 1: loss=0.1690, ترتيب صحيح؟ True
# زوج 2: loss=0.8544, ترتيب صحيح؟ False   <- خسارة أعلى بكثير لأن الترتيب خاطئ
# زوج 3: loss=0.5981, ترتيب صحيح؟ True    <- صحيح لكن الفارق صغير فالخسارة متوسطة

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-33-ch33-rlhf-ppo.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-33-ch33-rlhf-ppo.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — RLHF وPPO بعمق
Mastering Transformers and Large Language Models — Rlhf Ppo

Book slug : mastering-llms-transformers
Chapter   : ch-33-ch33-rlhf-ppo (lesson 33)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-07-peft-alignment/ch33-rlhf-ppo.ipynb#cell-1
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
# [002] content/notebooks/part-07-peft-alignment/ch33-rlhf-ppo.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel(nn.Module):
    """
    نموذج المكافأة: يأخذ نفس معمارية LLM الأساسي (عادة نفس SFT model)
    لكن نستبدل رأس اللغة (LM Head) برأس انحدار قياسي (scalar head)
    يُخرِج رقمًا واحدًا يُمثِّل "جودة" الاستجابة الكاملة.
    """
    def __init__(self, base_transformer: nn.Module, d_model: int):
        super().__init__()
        self.transformer = base_transformer  # عادة مُهيَّأ من نموذج SFT
        self.reward_head = nn.Linear(d_model, 1)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state  # (batch, seq_len, d_model)

        # أخذ تمثيل آخر رمز فعلي (غير padding) لكل تسلسل
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(input_ids.shape[0])
        last_token_hidden = hidden_states[batch_indices, seq_lengths]

        reward = self.reward_head(last_token_hidden)  # (batch, 1)
        return reward.squeeze(-1)  # (batch,)

def bradley_terry_reward_loss(reward_chosen: torch.Tensor,
                                reward_rejected: torch.Tensor) -> torch.Tensor:
    """
    دالة خسارة نموذج المكافأة المبنية على نموذج Bradley-Terry الإحصائي
    لاحتمالية التفضيل الزوجي:

        P(y_w ≻ y_l | x) = σ(r(x, y_w) - r(x, y_l))

    حيث y_w هي الاستجابة المُفضَّلة (winner) وy_l هي المرفوضة (loser).
    نُعظِّم log-likelihood هذا الاحتمال، أو بشكل مكافئ، نُصغِّر:
    """
    loss = -F.logsigmoid(reward_chosen - reward_rejected).mean()
    return loss

class PreferenceDataCollector:
    """
    محاكاة عملية جمع بيانات التفضيل البشري، كما في منهجية
    InstructGPT: يُعرَض على المُقيِّم البشري عدة استجابات (k=4 إلى 9)
    لنفس التعليمة، ويُرتِّبها من الأفضل إلى الأسوأ.
    من الترتيب الكلي، نستخرج أزواج (chosen, rejected) للتدريب.
    """
    @staticmethod
    def rankings_to_pairs(ranked_responses: list) -> list:
        """
        تحويل ترتيب كلي (مثال: [r1 > r3 > r2 > r4]) إلى كل الأزواج الممكنة
        (C(k,2) زوجًا)، كما فعلت ورقة InstructGPT لزيادة كفاءة البيانات.
        """
        pairs = []
        for i in range(len(ranked_responses)):
            for j in range(i + 1, len(ranked_responses)):
                pairs.append({
                    "chosen": ranked_responses[i],
                    "rejected": ranked_responses[j],
                })
        return pairs

def train_reward_model_step(reward_model: RewardModel, optimizer,
                              chosen_batch: dict, rejected_batch: dict):
    """خطوة تدريب واحدة كاملة لنموذج المكافأة."""
    optimizer.zero_grad()

    reward_chosen = reward_model(
        chosen_batch["input_ids"], chosen_batch["attention_mask"]
    )
    reward_rejected = reward_model(
        rejected_batch["input_ids"], rejected_batch["attention_mask"]
    )

    loss = bradley_terry_reward_loss(reward_chosen, reward_rejected)
    loss.backward()
    optimizer.step()

    # مقياس دقة مفيد للمراقبة: هل RM يُرتِّب الأزواج بشكل صحيح؟
    accuracy = (reward_chosen > reward_rejected).float().mean()
    return {"loss": loss.item(), "accuracy": accuracy.item()}

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-07-peft-alignment/ch33-rlhf-ppo.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class PPOTrainerForLLM:
    """
    تطبيق مبسّط تعليميًا لحلقة تدريب PPO المُستخدَمة في RLHF.
    يتضمن أربعة نماذج يجب تشغيلها معًا:
    1. Policy Model (π_θ) — النموذج الذي نُحسِّنه (مُهيَّأ من SFT)
    2. Reference Model (π_ref) — نسخة مجمّدة من SFT (لحساب KL)
    3. Reward Model (RM) — مُجمَّد، يُقيِّم جودة الاستجابات
    4. Value Model (V) — يُقدِّر "القيمة المتوقعة" لحالة معينة (Critic)
    """
    def __init__(self, policy_model, reference_model, reward_model,
                 value_model, tokenizer, kl_coef: float = 0.1,
                 clip_epsilon: float = 0.2):
        self.policy = policy_model
        self.reference = reference_model  # مُجمَّد تمامًا، لا تدرجات
        self.reward_model = reward_model  # مُجمَّد تمامًا
        self.value_model = value_model
        self.tokenizer = tokenizer
        self.kl_coef = kl_coef
        self.clip_epsilon = clip_epsilon

        for p in self.reference.parameters():
            p.requires_grad = False
        for p in self.reward_model.parameters():
            p.requires_grad = False

    def compute_rewards_with_kl_penalty(self, prompts, responses,
                                          response_logprobs, ref_logprobs):
        """
        المكافأة الفعلية المُستخدَمة في PPO ليست فقط مخرج RM،
        بل RM مطروحًا منها عقوبة KL-divergence عن السياسة المرجعية.
        هذا يمنع النموذج من الانحراف الجذري بعيدًا عن سلوك SFT
        الآمن والمعقول سعيًا وراء "خداع" نموذج المكافأة:

            R(x,y) = RM(x,y) - β · KL(π_θ(y|x) || π_ref(y|x))

        حيث β هو kl_coef، وKL هنا تُقدَّر تجريبيًا كفرق log-probabilities.
        """
        full_sequences = [p + r for p, r in zip(prompts, responses)]
        inputs = self.tokenizer(full_sequences, return_tensors="pt",
                                  padding=True, truncation=True)

        with torch.no_grad():
            rm_scores = self.reward_model(
                inputs["input_ids"], inputs["attention_mask"]
            )

        # تقدير KL لكل رمز: log π_θ(token) - log π_ref(token)
        kl_per_token = response_logprobs - ref_logprobs
        kl_penalty = self.kl_coef * kl_per_token.sum(dim=-1)

        total_reward = rm_scores - kl_penalty
        return total_reward, kl_penalty

    def compute_advantages_gae(self, rewards, values, gamma=0.99, lam=0.95):
        """
        Generalized Advantage Estimation (GAE): تقدير "الميزة" (Advantage)
        لكل خطوة زمنية (رمز)، يُوازن بين الانحياز والتباين في تقدير
        مدى "جودة" إجراء مُعيَّن مقارنة بالمتوسط المتوقع.

            δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
            A_t = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...
        """
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t < len(rewards) - 1 else 0
            delta = rewards[t] + gamma * next_value - values[t]
            last_gae = delta + gamma * lam * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        return advantages, returns

    def ppo_clipped_objective(self, old_logprobs, new_logprobs, advantages):
        """
        دالة الهدف المُقيَّدة لـ PPO (الابتكار الجوهري للخوارزمية):
        تمنع التحديثات الكبيرة جدًا للسياسة في خطوة واحدة عبر
        "قصّ" نسبة الاحتمالات الجديدة/القديمة ضمن [1-ε, 1+ε]،
        مما يضمن استقرار التدريب (Proximal = "قريب من السياسة القديمة").

            L^CLIP(θ) = E[min(r_t(θ)·A_t, clip(r_t(θ), 1-ε, 1+ε)·A_t)]

            حيث r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
        """
        ratio = torch.exp(new_logprobs - old_logprobs)  # نسبة الاحتمالات

        unclipped = ratio * advantages
        clipped = torch.clamp(
            ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
        ) * advantages

        # نأخذ min لضمان أننا لا نُحسِّن الهدف أكثر من اللازم
        # (pessimistic bound يضمن استقرار تحديثات السياسة)
        policy_loss = -torch.min(unclipped, clipped).mean()
        return policy_loss

    def training_step(self, prompts: list):
        """خطوة PPO كاملة واحدة (مُبسَّطة تعليميًا)."""
        # 1. توليد استجابات من السياسة الحالية
        responses, response_logprobs = self._generate_with_logprobs(prompts)

        # 2. حساب log-probs من النموذج المرجعي لنفس الاستجابات (لـ KL)
        with torch.no_grad():
            ref_logprobs = self._compute_logprobs(
                self.reference, prompts, responses
            )

        # 3. حساب المكافأة مع عقوبة KL
        rewards, kl_penalty = self.compute_rewards_with_kl_penalty(
            prompts, responses, response_logprobs, ref_logprobs
        )

        # 4. حساب القيم (Critic) والميزات (Advantages)
        values = self.value_model(prompts, responses)
        advantages, returns = self.compute_advantages_gae(rewards, values)

        # 5. عدة تكرارات PPO على نفس الدفعة (عادة 4 epochs داخلية)
        for _ in range(4):
            new_logprobs = self._compute_logprobs(
                self.policy, prompts, responses
            )
            policy_loss = self.ppo_clipped_objective(
                response_logprobs.detach(), new_logprobs, advantages.detach()
            )
            value_loss = F.mse_loss(
                self.value_model(prompts, responses), returns.detach()
            )
            total_loss = policy_loss + 0.5 * value_loss
            total_loss.backward()
            # optimizer.step() ...

        return {
            "policy_loss": policy_loss.item(),
            "mean_reward": rewards.mean().item(),
            "mean_kl": kl_penalty.mean().item(),
        }

    def _generate_with_logprobs(self, prompts):
        raise NotImplementedError("تفاصيل التوليد الفعلي تعتمد على المكتبة")

    def _compute_logprobs(self, model, prompts, responses):
        raise NotImplementedError("حساب log P(token) لكل رمز في الاستجابة")

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-07-peft-alignment/ch33-rlhf-ppo.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
reward_pairs = [
    {"chosen": 2.3, "rejected": 0.8},   # RM واثق وصحيح
    {"chosen": 1.1, "rejected": 1.4},   # RM مُخطئ (الترتيب معكوس!)
    {"chosen": 0.2, "rejected": -0.1},  # RM صحيح لكن بثقة منخفضة
]

for i, pair in enumerate(reward_pairs):
    r_chosen = torch.tensor(pair["chosen"])
    r_rejected = torch.tensor(pair["rejected"])
    loss = -F.logsigmoid(r_chosen - r_rejected)
    correct = r_chosen > r_rejected
    print(f"زوج {i+1}: loss={loss.item():.4f}, "
          f"ترتيب صحيح؟ {correct.item()}")

# زوج 1: loss=0.1690, ترتيب صحيح؟ True
# زوج 2: loss=0.8544, ترتيب صحيح؟ False   <- خسارة أعلى بكثير لأن الترتيب خاطئ
# زوج 3: loss=0.5981, ترتيب صحيح؟ True    <- صحيح لكن الفارق صغير فالخسارة متوسطة

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-07-peft-alignment/ch33-rlhf-ppo.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-33-ch33-rlhf-ppo.ipynb#cell-2
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-33-ch33-rlhf-ppo.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel(nn.Module):
    """
    نموذج المكافأة: يأخذ نفس معمارية LLM الأساسي (عادة نفس SFT model)
    لكن نستبدل رأس اللغة (LM Head) برأس انحدار قياسي (scalar head)
    يُخرِج رقمًا واحدًا يُمثِّل "جودة" الاستجابة الكاملة.
    """
    def __init__(self, base_transformer: nn.Module, d_model: int):
        super().__init__()
        self.transformer = base_transformer  # عادة مُهيَّأ من نموذج SFT
        self.reward_head = nn.Linear(d_model, 1)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state  # (batch, seq_len, d_model)

        # أخذ تمثيل آخر رمز فعلي (غير padding) لكل تسلسل
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(input_ids.shape[0])
        last_token_hidden = hidden_states[batch_indices, seq_lengths]

        reward = self.reward_head(last_token_hidden)  # (batch, 1)
        return reward.squeeze(-1)  # (batch,)

def bradley_terry_reward_loss(reward_chosen: torch.Tensor,
                                reward_rejected: torch.Tensor) -> torch.Tensor:
    """
    دالة خسارة نموذج المكافأة المبنية على نموذج Bradley-Terry الإحصائي
    لاحتمالية التفضيل الزوجي:

        P(y_w ≻ y_l | x) = σ(r(x, y_w) - r(x, y_l))

    حيث y_w هي الاستجابة المُفضَّلة (winner) وy_l هي المرفوضة (loser).
    نُعظِّم log-likelihood هذا الاحتمال، أو بشكل مكافئ، نُصغِّر:
    """
    loss = -F.logsigmoid(reward_chosen - reward_rejected).mean()
    return loss

class PreferenceDataCollector:
    """
    محاكاة عملية جمع بيانات التفضيل البشري، كما في منهجية
    InstructGPT: يُعرَض على المُقيِّم البشري عدة استجابات (k=4 إلى 9)
    لنفس التعليمة، ويُرتِّبها من الأفضل إلى الأسوأ.
    من الترتيب الكلي، نستخرج أزواج (chosen, rejected) للتدريب.
    """
    @staticmethod
    def rankings_to_pairs(ranked_responses: list) -> list:
        """
        تحويل ترتيب كلي (مثال: [r1 > r3 > r2 > r4]) إلى كل الأزواج الممكنة
        (C(k,2) زوجًا)، كما فعلت ورقة InstructGPT لزيادة كفاءة البيانات.
        """
        pairs = []
        for i in range(len(ranked_responses)):
            for j in range(i + 1, len(ranked_responses)):
                pairs.append({
                    "chosen": ranked_responses[i],
                    "rejected": ranked_responses[j],
                })
        return pairs

def train_reward_model_step(reward_model: RewardModel, optimizer,
                              chosen_batch: dict, rejected_batch: dict):
    """خطوة تدريب واحدة كاملة لنموذج المكافأة."""
    optimizer.zero_grad()

    reward_chosen = reward_model(
        chosen_batch["input_ids"], chosen_batch["attention_mask"]
    )
    reward_rejected = reward_model(
        rejected_batch["input_ids"], rejected_batch["attention_mask"]
    )

    loss = bradley_terry_reward_loss(reward_chosen, reward_rejected)
    loss.backward()
    optimizer.step()

    # مقياس دقة مفيد للمراقبة: هل RM يُرتِّب الأزواج بشكل صحيح؟
    accuracy = (reward_chosen > reward_rejected).float().mean()
    return {"loss": loss.item(), "accuracy": accuracy.item()}

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-33-ch33-rlhf-ppo.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class PPOTrainerForLLM:
    """
    تطبيق مبسّط تعليميًا لحلقة تدريب PPO المُستخدَمة في RLHF.
    يتضمن أربعة نماذج يجب تشغيلها معًا:
    1. Policy Model (π_θ) — النموذج الذي نُحسِّنه (مُهيَّأ من SFT)
    2. Reference Model (π_ref) — نسخة مجمّدة من SFT (لحساب KL)
    3. Reward Model (RM) — مُجمَّد، يُقيِّم جودة الاستجابات
    4. Value Model (V) — يُقدِّر "القيمة المتوقعة" لحالة معينة (Critic)
    """
    def __init__(self, policy_model, reference_model, reward_model,
                 value_model, tokenizer, kl_coef: float = 0.1,
                 clip_epsilon: float = 0.2):
        self.policy = policy_model
        self.reference = reference_model  # مُجمَّد تمامًا، لا تدرجات
        self.reward_model = reward_model  # مُجمَّد تمامًا
        self.value_model = value_model
        self.tokenizer = tokenizer
        self.kl_coef = kl_coef
        self.clip_epsilon = clip_epsilon

        for p in self.reference.parameters():
            p.requires_grad = False
        for p in self.reward_model.parameters():
            p.requires_grad = False

    def compute_rewards_with_kl_penalty(self, prompts, responses,
                                          response_logprobs, ref_logprobs):
        """
        المكافأة الفعلية المُستخدَمة في PPO ليست فقط مخرج RM،
        بل RM مطروحًا منها عقوبة KL-divergence عن السياسة المرجعية.
        هذا يمنع النموذج من الانحراف الجذري بعيدًا عن سلوك SFT
        الآمن والمعقول سعيًا وراء "خداع" نموذج المكافأة:

            R(x,y) = RM(x,y) - β · KL(π_θ(y|x) || π_ref(y|x))

        حيث β هو kl_coef، وKL هنا تُقدَّر تجريبيًا كفرق log-probabilities.
        """
        full_sequences = [p + r for p, r in zip(prompts, responses)]
        inputs = self.tokenizer(full_sequences, return_tensors="pt",
                                  padding=True, truncation=True)

        with torch.no_grad():
            rm_scores = self.reward_model(
                inputs["input_ids"], inputs["attention_mask"]
            )

        # تقدير KL لكل رمز: log π_θ(token) - log π_ref(token)
        kl_per_token = response_logprobs - ref_logprobs
        kl_penalty = self.kl_coef * kl_per_token.sum(dim=-1)

        total_reward = rm_scores - kl_penalty
        return total_reward, kl_penalty

    def compute_advantages_gae(self, rewards, values, gamma=0.99, lam=0.95):
        """
        Generalized Advantage Estimation (GAE): تقدير "الميزة" (Advantage)
        لكل خطوة زمنية (رمز)، يُوازن بين الانحياز والتباين في تقدير
        مدى "جودة" إجراء مُعيَّن مقارنة بالمتوسط المتوقع.

            δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
            A_t = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...
        """
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t < len(rewards) - 1 else 0
            delta = rewards[t] + gamma * next_value - values[t]
            last_gae = delta + gamma * lam * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        return advantages, returns

    def ppo_clipped_objective(self, old_logprobs, new_logprobs, advantages):
        """
        دالة الهدف المُقيَّدة لـ PPO (الابتكار الجوهري للخوارزمية):
        تمنع التحديثات الكبيرة جدًا للسياسة في خطوة واحدة عبر
        "قصّ" نسبة الاحتمالات الجديدة/القديمة ضمن [1-ε, 1+ε]،
        مما يضمن استقرار التدريب (Proximal = "قريب من السياسة القديمة").

            L^CLIP(θ) = E[min(r_t(θ)·A_t, clip(r_t(θ), 1-ε, 1+ε)·A_t)]

            حيث r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
        """
        ratio = torch.exp(new_logprobs - old_logprobs)  # نسبة الاحتمالات

        unclipped = ratio * advantages
        clipped = torch.clamp(
            ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
        ) * advantages

        # نأخذ min لضمان أننا لا نُحسِّن الهدف أكثر من اللازم
        # (pessimistic bound يضمن استقرار تحديثات السياسة)
        policy_loss = -torch.min(unclipped, clipped).mean()
        return policy_loss

    def training_step(self, prompts: list):
        """خطوة PPO كاملة واحدة (مُبسَّطة تعليميًا)."""
        # 1. توليد استجابات من السياسة الحالية
        responses, response_logprobs = self._generate_with_logprobs(prompts)

        # 2. حساب log-probs من النموذج المرجعي لنفس الاستجابات (لـ KL)
        with torch.no_grad():
            ref_logprobs = self._compute_logprobs(
                self.reference, prompts, responses
            )

        # 3. حساب المكافأة مع عقوبة KL
        rewards, kl_penalty = self.compute_rewards_with_kl_penalty(
            prompts, responses, response_logprobs, ref_logprobs
        )

        # 4. حساب القيم (Critic) والميزات (Advantages)
        values = self.value_model(prompts, responses)
        advantages, returns = self.compute_advantages_gae(rewards, values)

        # 5. عدة تكرارات PPO على نفس الدفعة (عادة 4 epochs داخلية)
        for _ in range(4):
            new_logprobs = self._compute_logprobs(
                self.policy, prompts, responses
            )
            policy_loss = self.ppo_clipped_objective(
                response_logprobs.detach(), new_logprobs, advantages.detach()
            )
            value_loss = F.mse_loss(
                self.value_model(prompts, responses), returns.detach()
            )
            total_loss = policy_loss + 0.5 * value_loss
            total_loss.backward()
            # optimizer.step() ...

        return {
            "policy_loss": policy_loss.item(),
            "mean_reward": rewards.mean().item(),
            "mean_kl": kl_penalty.mean().item(),
        }

    def _generate_with_logprobs(self, prompts):
        raise NotImplementedError("تفاصيل التوليد الفعلي تعتمد على المكتبة")

    def _compute_logprobs(self, model, prompts, responses):
        raise NotImplementedError("حساب log P(token) لكل رمز في الاستجابة")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-33-ch33-rlhf-ppo.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
reward_pairs = [
    {"chosen": 2.3, "rejected": 0.8},   # RM واثق وصحيح
    {"chosen": 1.1, "rejected": 1.4},   # RM مُخطئ (الترتيب معكوس!)
    {"chosen": 0.2, "rejected": -0.1},  # RM صحيح لكن بثقة منخفضة
]

for i, pair in enumerate(reward_pairs):
    r_chosen = torch.tensor(pair["chosen"])
    r_rejected = torch.tensor(pair["rejected"])
    loss = -F.logsigmoid(r_chosen - r_rejected)
    correct = r_chosen > r_rejected
    print(f"زوج {i+1}: loss={loss.item():.4f}, "
          f"ترتيب صحيح؟ {correct.item()}")

# زوج 1: loss=0.1690, ترتيب صحيح؟ True
# زوج 2: loss=0.8544, ترتيب صحيح؟ False   <- خسارة أعلى بكثير لأن الترتيب خاطئ
# زوج 3: loss=0.5981, ترتيب صحيح؟ True    <- صحيح لكن الفارق صغير فالخسارة متوسطة

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-33-ch33-rlhf-ppo.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
