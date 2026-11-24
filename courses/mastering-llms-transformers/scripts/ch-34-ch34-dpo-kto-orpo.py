#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — DPO وKTO وORPO: ما بعد RLHF
Mastering Transformers and Large Language Models — Dpo Kto Orpo

Book slug : mastering-llms-transformers
Chapter   : ch-34-ch34-dpo-kto-orpo (lesson 34)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-34-ch34-dpo-kto-orpo.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — DPO وKTO وORPO: ما بعد RLHF
Mastering Transformers and Large Language Models — Dpo Kto Orpo

Book slug : mastering-llms-transformers
Chapter   : ch-34-ch34-dpo-kto-orpo (lesson 34)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-34-ch34-dpo-kto-orpo.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — DPO وKTO وORPO: ما بعد RLHF
Mastering Transformers and Large Language Models — Dpo Kto Orpo

Book slug : mastering-llms-transformers
Chapter   : ch-34-ch34-dpo-kto-orpo (lesson 34)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-07-peft-alignment/ch34-dpo-kto-orpo.ipynb#cell-1
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
# [002] content/notebooks/part-07-peft-alignment/ch34-dpo-kto-orpo.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_sequence_logprobs(model, input_ids: torch.Tensor,
                                attention_mask: torch.Tensor,
                                labels: torch.Tensor) -> torch.Tensor:
    """
    حساب مجموع log-probabilities لتسلسل كامل (الاستجابة فقط،
    وليس التعليمة/prompt — يجب إخفاء التعليمة بـ labels=-100).
    هذا الـ building block مُستخدَم في كل من DPO/KTO/ORPO.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]   # إزاحة بمقدار واحد (next-token prediction)
    labels = labels[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)

    # إخفاء مواضع الـ prompt/padding (labels == -100) من حساب الخسارة
    mask = (labels != -100)
    labels_clamped = labels.clamp(min=0)  # لتفادي index error مؤقتًا
    token_logprobs = torch.gather(
        log_probs, dim=2, index=labels_clamped.unsqueeze(-1)
    ).squeeze(-1)

    token_logprobs = token_logprobs * mask  # تصفير المواضع المُخفاة
    sequence_logprobs = token_logprobs.sum(dim=1)  # مجموع log-probs للاستجابة كاملة
    return sequence_logprobs

class DPOTrainer:
    """
    تطبيق كامل لـ Direct Preference Optimization (Rafailov et al., 2023).
    يتطلب فقط نموذجين: Policy (قابل للتدريب) وReference (مُجمَّد).
    """
    def __init__(self, policy_model: nn.Module, reference_model: nn.Module,
                 beta: float = 0.1):
        self.policy = policy_model
        self.reference = reference_model
        self.beta = beta

        for p in self.reference.parameters():
            p.requires_grad = False
        self.reference.eval()

    def compute_dpo_loss(self, batch: dict) -> dict:
        """
        batch يحتوي على: chosen_input_ids, chosen_attention_mask, chosen_labels,
                          rejected_input_ids, rejected_attention_mask, rejected_labels
        """
        # Log-probs من السياسة الحالية (قابلة للتدريب)
        policy_chosen_logps = compute_sequence_logprobs(
            self.policy, batch["chosen_input_ids"],
            batch["chosen_attention_mask"], batch["chosen_labels"]
        )
        policy_rejected_logps = compute_sequence_logprobs(
            self.policy, batch["rejected_input_ids"],
            batch["rejected_attention_mask"], batch["rejected_labels"]
        )

        # Log-probs من النموذج المرجعي (بدون تدرجات)
        with torch.no_grad():
            ref_chosen_logps = compute_sequence_logprobs(
                self.reference, batch["chosen_input_ids"],
                batch["chosen_attention_mask"], batch["chosen_labels"]
            )
            ref_rejected_logps = compute_sequence_logprobs(
                self.reference, batch["rejected_input_ids"],
                batch["rejected_attention_mask"], batch["rejected_labels"]
            )

        # النسب اللوغاريتمية (implicit reward) لكل من المُفضَّل والمرفوض
        chosen_log_ratio = policy_chosen_logps - ref_chosen_logps
        rejected_log_ratio = policy_rejected_logps - ref_rejected_logps

        # دالة خسارة DPO النهائية (معادلة الاشتقاق أعلاه)
        logits = self.beta * (chosen_log_ratio - rejected_log_ratio)
        loss = -F.logsigmoid(logits).mean()

        # مقاييس مفيدة للمراقبة أثناء التدريب
        with torch.no_grad():
            implicit_reward_chosen = self.beta * chosen_log_ratio
            implicit_reward_rejected = self.beta * rejected_log_ratio
            accuracy = (implicit_reward_chosen > implicit_reward_rejected).float().mean()
            reward_margin = (implicit_reward_chosen - implicit_reward_rejected).mean()

        return {
            "loss": loss,
            "accuracy": accuracy.item(),
            "reward_margin": reward_margin.item(),
            "chosen_reward": implicit_reward_chosen.mean().item(),
            "rejected_reward": implicit_reward_rejected.mean().item(),
        }

    def training_step(self, batch: dict, optimizer):
        optimizer.zero_grad()
        metrics = self.compute_dpo_loss(batch)
        metrics["loss"].backward()
        optimizer.step()
        return {k: (v if isinstance(v, float) else v.item())
                for k, v in metrics.items()}

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-07-peft-alignment/ch34-dpo-kto-orpo.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def ipo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta: float = 0.1):
    """
    IPO (Azar et al., 2023): بدلًا من logsigmoid غير المحدودة،
    نستخدم خسارة تربيعية (MSE-style) مع هدف ثابت = 1/(2β).
    هذا يمنع "الثقة المفرطة" (overconfidence) ويُحسِّن التعميم،
    خصوصًا عندما تحتوي البيانات على تفضيلات قوية جدًا أو حتمية.
    """
    chosen_log_ratio = policy_chosen_logps - ref_chosen_logps
    rejected_log_ratio = policy_rejected_logps - ref_rejected_logps

    h = chosen_log_ratio - rejected_log_ratio
    # الهدف: h يجب أن يقترب من 1/(2β) بالضبط، وليس "أكبر ما يمكن"
    loss = (h - 1 / (2 * beta)) ** 2
    return loss.mean()

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-07-peft-alignment/ch34-dpo-kto-orpo.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def kto_loss(policy_logps: torch.Tensor, ref_logps: torch.Tensor,
             is_desirable: torch.Tensor, beta: float = 0.1,
             desirable_weight: float = 1.0, undesirable_weight: float = 1.0):
    """
    KTO (Ethayarajh et al., 2024): لا يحتاج أزواج (chosen, rejected)!
    كل عينة بيانات مستقلة، فقط بعلامة ثنائية: مرغوبة (1) أو غير مرغوبة (0).

    مُستوحى من نظرية "النفور من الخسارة" (loss aversion) البشرية:
    البشر يتفاعلون بشكل غير متماثل مع المكاسب والخسائر، لذا KTO
    تستخدم دالة قيمة (value function) غير خطية وغير متماثلة.
    """
    log_ratio = policy_logps - ref_logps
    kl_estimate = log_ratio.mean().detach()  # تقدير KL على مستوى الدفعة

    # دالة القيمة غير المتماثلة (مستوحاة من Prospect Theory)
    desirable_losses = 1 - torch.sigmoid(
        beta * (log_ratio - kl_estimate)
    )
    undesirable_losses = 1 - torch.sigmoid(
        beta * (kl_estimate - log_ratio)
    )

    losses = torch.where(
        is_desirable.bool(),
        desirable_weight * desirable_losses,
        undesirable_weight * undesirable_losses,
    )
    return losses.mean()

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-07-peft-alignment/ch34-dpo-kto-orpo.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def orpo_loss(model, chosen_batch: dict, rejected_batch: dict,
              lambda_weight: float = 0.1):
    """
    ORPO (Hong et al., 2024): خسارة واحدة مُركَّبة من جزأين:

    1. SFT Loss (Cross-Entropy) على الاستجابة المُفضَّلة فقط
       (نفس هدف التدريب الخاضع للإشراف العادي)

    2. Odds Ratio Loss: تعظيم نسبة الأرجحية (odds) للاستجابة
       المُفضَّلة على المرفوضة، دون الحاجة لأي reference model!
       (الفارق الجوهري عن DPO: لا نموذج مرجعي مُجمَّد مطلوب)
    """
    chosen_outputs = model(input_ids=chosen_batch["input_ids"],
                             attention_mask=chosen_batch["attention_mask"])
    rejected_outputs = model(input_ids=rejected_batch["input_ids"],
                               attention_mask=rejected_batch["attention_mask"])

    # الجزء الأول: SFT Loss قياسية على الاستجابة المُفضَّلة
    sft_loss = F.cross_entropy(
        chosen_outputs.logits[:, :-1].reshape(-1, chosen_outputs.logits.size(-1)),
        chosen_batch["labels"][:, 1:].reshape(-1),
        ignore_index=-100,
    )

    # الجزء الثاني: Odds Ratio بين الاستجابتين
    def compute_avg_logprob(logits, labels):
        log_probs = F.log_softmax(logits[:, :-1], dim=-1)
        labels_shifted = labels[:, 1:]
        mask = (labels_shifted != -100)
        labels_clamped = labels_shifted.clamp(min=0)
        token_logps = torch.gather(
            log_probs, 2, labels_clamped.unsqueeze(-1)
        ).squeeze(-1) * mask
        # متوسط (وليس مجموع) لتطبيع طول التسلسل - مهم لصيغة Odds Ratio
        return token_logps.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    chosen_logp = compute_avg_logprob(chosen_outputs.logits, chosen_batch["labels"])
    rejected_logp = compute_avg_logprob(rejected_outputs.logits, rejected_batch["labels"])

    # log odds = log(p / (1-p))، حيث p = exp(avg_logprob)
    log_odds_chosen = chosen_logp - torch.log1p(-torch.exp(chosen_logp).clamp(max=1 - 1e-6))
    log_odds_rejected = rejected_logp - torch.log1p(-torch.exp(rejected_logp).clamp(max=1 - 1e-6))

    odds_ratio_loss = -F.logsigmoid(log_odds_chosen - log_odds_rejected).mean()

    total_loss = sft_loss + lambda_weight * odds_ratio_loss
    return {
        "total_loss": total_loss,
        "sft_loss": sft_loss.item(),
        "odds_ratio_loss": odds_ratio_loss.item(),
    }

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-07-peft-alignment/ch34-dpo-kto-orpo.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def manual_dpo_example():
    # افتراضات مبسّطة لقيم log-probability (مجموع log-probs للاستجابة كاملة)
    policy_chosen_logp = torch.tensor(-12.5)    # النموذج الحالي "واثق نسبيًا" من المُفضَّل
    policy_rejected_logp = torch.tensor(-15.0)  # وأقل ثقة في المرفوض (جيد!)
    ref_chosen_logp = torch.tensor(-13.0)       # النموذج المرجعي (قبل أي تحديث)
    ref_rejected_logp = torch.tensor(-13.5)

    for beta in [0.05, 0.1, 0.5]:
        chosen_ratio = policy_chosen_logp - ref_chosen_logp     # 0.5
        rejected_ratio = policy_rejected_logp - ref_rejected_logp  # -1.5

        logits = beta * (chosen_ratio - rejected_ratio)  # beta * 2.0
        loss = -F.logsigmoid(logits)

        print(f"β={beta}: implicit_logits={logits.item():.3f}, "
              f"loss={loss.item():.4f}")

manual_dpo_example()
# β=0.05: implicit_logits=0.100, loss=0.6444
# β=0.1:  implicit_logits=0.200, loss=0.5981
# β=0.5:  implicit_logits=1.000, loss=0.3133

# ────────────────────────────────────────────────────────────────────────
# [007] content/notebooks/part-07-peft-alignment/ch34-dpo-kto-orpo.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-34-ch34-dpo-kto-orpo.ipynb#cell-1
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-34-ch34-dpo-kto-orpo.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_sequence_logprobs(model, input_ids: torch.Tensor,
                                attention_mask: torch.Tensor,
                                labels: torch.Tensor) -> torch.Tensor:
    """
    حساب مجموع log-probabilities لتسلسل كامل (الاستجابة فقط،
    وليس التعليمة/prompt — يجب إخفاء التعليمة بـ labels=-100).
    هذا الـ building block مُستخدَم في كل من DPO/KTO/ORPO.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]   # إزاحة بمقدار واحد (next-token prediction)
    labels = labels[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)

    # إخفاء مواضع الـ prompt/padding (labels == -100) من حساب الخسارة
    mask = (labels != -100)
    labels_clamped = labels.clamp(min=0)  # لتفادي index error مؤقتًا
    token_logprobs = torch.gather(
        log_probs, dim=2, index=labels_clamped.unsqueeze(-1)
    ).squeeze(-1)

    token_logprobs = token_logprobs * mask  # تصفير المواضع المُخفاة
    sequence_logprobs = token_logprobs.sum(dim=1)  # مجموع log-probs للاستجابة كاملة
    return sequence_logprobs

class DPOTrainer:
    """
    تطبيق كامل لـ Direct Preference Optimization (Rafailov et al., 2023).
    يتطلب فقط نموذجين: Policy (قابل للتدريب) وReference (مُجمَّد).
    """
    def __init__(self, policy_model: nn.Module, reference_model: nn.Module,
                 beta: float = 0.1):
        self.policy = policy_model
        self.reference = reference_model
        self.beta = beta

        for p in self.reference.parameters():
            p.requires_grad = False
        self.reference.eval()

    def compute_dpo_loss(self, batch: dict) -> dict:
        """
        batch يحتوي على: chosen_input_ids, chosen_attention_mask, chosen_labels,
                          rejected_input_ids, rejected_attention_mask, rejected_labels
        """
        # Log-probs من السياسة الحالية (قابلة للتدريب)
        policy_chosen_logps = compute_sequence_logprobs(
            self.policy, batch["chosen_input_ids"],
            batch["chosen_attention_mask"], batch["chosen_labels"]
        )
        policy_rejected_logps = compute_sequence_logprobs(
            self.policy, batch["rejected_input_ids"],
            batch["rejected_attention_mask"], batch["rejected_labels"]
        )

        # Log-probs من النموذج المرجعي (بدون تدرجات)
        with torch.no_grad():
            ref_chosen_logps = compute_sequence_logprobs(
                self.reference, batch["chosen_input_ids"],
                batch["chosen_attention_mask"], batch["chosen_labels"]
            )
            ref_rejected_logps = compute_sequence_logprobs(
                self.reference, batch["rejected_input_ids"],
                batch["rejected_attention_mask"], batch["rejected_labels"]
            )

        # النسب اللوغاريتمية (implicit reward) لكل من المُفضَّل والمرفوض
        chosen_log_ratio = policy_chosen_logps - ref_chosen_logps
        rejected_log_ratio = policy_rejected_logps - ref_rejected_logps

        # دالة خسارة DPO النهائية (معادلة الاشتقاق أعلاه)
        logits = self.beta * (chosen_log_ratio - rejected_log_ratio)
        loss = -F.logsigmoid(logits).mean()

        # مقاييس مفيدة للمراقبة أثناء التدريب
        with torch.no_grad():
            implicit_reward_chosen = self.beta * chosen_log_ratio
            implicit_reward_rejected = self.beta * rejected_log_ratio
            accuracy = (implicit_reward_chosen > implicit_reward_rejected).float().mean()
            reward_margin = (implicit_reward_chosen - implicit_reward_rejected).mean()

        return {
            "loss": loss,
            "accuracy": accuracy.item(),
            "reward_margin": reward_margin.item(),
            "chosen_reward": implicit_reward_chosen.mean().item(),
            "rejected_reward": implicit_reward_rejected.mean().item(),
        }

    def training_step(self, batch: dict, optimizer):
        optimizer.zero_grad()
        metrics = self.compute_dpo_loss(batch)
        metrics["loss"].backward()
        optimizer.step()
        return {k: (v if isinstance(v, float) else v.item())
                for k, v in metrics.items()}

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-34-ch34-dpo-kto-orpo.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def ipo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta: float = 0.1):
    """
    IPO (Azar et al., 2023): بدلًا من logsigmoid غير المحدودة،
    نستخدم خسارة تربيعية (MSE-style) مع هدف ثابت = 1/(2β).
    هذا يمنع "الثقة المفرطة" (overconfidence) ويُحسِّن التعميم،
    خصوصًا عندما تحتوي البيانات على تفضيلات قوية جدًا أو حتمية.
    """
    chosen_log_ratio = policy_chosen_logps - ref_chosen_logps
    rejected_log_ratio = policy_rejected_logps - ref_rejected_logps

    h = chosen_log_ratio - rejected_log_ratio
    # الهدف: h يجب أن يقترب من 1/(2β) بالضبط، وليس "أكبر ما يمكن"
    loss = (h - 1 / (2 * beta)) ** 2
    return loss.mean()

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-34-ch34-dpo-kto-orpo.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def kto_loss(policy_logps: torch.Tensor, ref_logps: torch.Tensor,
             is_desirable: torch.Tensor, beta: float = 0.1,
             desirable_weight: float = 1.0, undesirable_weight: float = 1.0):
    """
    KTO (Ethayarajh et al., 2024): لا يحتاج أزواج (chosen, rejected)!
    كل عينة بيانات مستقلة، فقط بعلامة ثنائية: مرغوبة (1) أو غير مرغوبة (0).

    مُستوحى من نظرية "النفور من الخسارة" (loss aversion) البشرية:
    البشر يتفاعلون بشكل غير متماثل مع المكاسب والخسائر، لذا KTO
    تستخدم دالة قيمة (value function) غير خطية وغير متماثلة.
    """
    log_ratio = policy_logps - ref_logps
    kl_estimate = log_ratio.mean().detach()  # تقدير KL على مستوى الدفعة

    # دالة القيمة غير المتماثلة (مستوحاة من Prospect Theory)
    desirable_losses = 1 - torch.sigmoid(
        beta * (log_ratio - kl_estimate)
    )
    undesirable_losses = 1 - torch.sigmoid(
        beta * (kl_estimate - log_ratio)
    )

    losses = torch.where(
        is_desirable.bool(),
        desirable_weight * desirable_losses,
        undesirable_weight * undesirable_losses,
    )
    return losses.mean()

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-34-ch34-dpo-kto-orpo.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def orpo_loss(model, chosen_batch: dict, rejected_batch: dict,
              lambda_weight: float = 0.1):
    """
    ORPO (Hong et al., 2024): خسارة واحدة مُركَّبة من جزأين:

    1. SFT Loss (Cross-Entropy) على الاستجابة المُفضَّلة فقط
       (نفس هدف التدريب الخاضع للإشراف العادي)

    2. Odds Ratio Loss: تعظيم نسبة الأرجحية (odds) للاستجابة
       المُفضَّلة على المرفوضة، دون الحاجة لأي reference model!
       (الفارق الجوهري عن DPO: لا نموذج مرجعي مُجمَّد مطلوب)
    """
    chosen_outputs = model(input_ids=chosen_batch["input_ids"],
                             attention_mask=chosen_batch["attention_mask"])
    rejected_outputs = model(input_ids=rejected_batch["input_ids"],
                               attention_mask=rejected_batch["attention_mask"])

    # الجزء الأول: SFT Loss قياسية على الاستجابة المُفضَّلة
    sft_loss = F.cross_entropy(
        chosen_outputs.logits[:, :-1].reshape(-1, chosen_outputs.logits.size(-1)),
        chosen_batch["labels"][:, 1:].reshape(-1),
        ignore_index=-100,
    )

    # الجزء الثاني: Odds Ratio بين الاستجابتين
    def compute_avg_logprob(logits, labels):
        log_probs = F.log_softmax(logits[:, :-1], dim=-1)
        labels_shifted = labels[:, 1:]
        mask = (labels_shifted != -100)
        labels_clamped = labels_shifted.clamp(min=0)
        token_logps = torch.gather(
            log_probs, 2, labels_clamped.unsqueeze(-1)
        ).squeeze(-1) * mask
        # متوسط (وليس مجموع) لتطبيع طول التسلسل - مهم لصيغة Odds Ratio
        return token_logps.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    chosen_logp = compute_avg_logprob(chosen_outputs.logits, chosen_batch["labels"])
    rejected_logp = compute_avg_logprob(rejected_outputs.logits, rejected_batch["labels"])

    # log odds = log(p / (1-p))، حيث p = exp(avg_logprob)
    log_odds_chosen = chosen_logp - torch.log1p(-torch.exp(chosen_logp).clamp(max=1 - 1e-6))
    log_odds_rejected = rejected_logp - torch.log1p(-torch.exp(rejected_logp).clamp(max=1 - 1e-6))

    odds_ratio_loss = -F.logsigmoid(log_odds_chosen - log_odds_rejected).mean()

    total_loss = sft_loss + lambda_weight * odds_ratio_loss
    return {
        "total_loss": total_loss,
        "sft_loss": sft_loss.item(),
        "odds_ratio_loss": odds_ratio_loss.item(),
    }

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-34-ch34-dpo-kto-orpo.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def manual_dpo_example():
    # افتراضات مبسّطة لقيم log-probability (مجموع log-probs للاستجابة كاملة)
    policy_chosen_logp = torch.tensor(-12.5)    # النموذج الحالي "واثق نسبيًا" من المُفضَّل
    policy_rejected_logp = torch.tensor(-15.0)  # وأقل ثقة في المرفوض (جيد!)
    ref_chosen_logp = torch.tensor(-13.0)       # النموذج المرجعي (قبل أي تحديث)
    ref_rejected_logp = torch.tensor(-13.5)

    for beta in [0.05, 0.1, 0.5]:
        chosen_ratio = policy_chosen_logp - ref_chosen_logp     # 0.5
        rejected_ratio = policy_rejected_logp - ref_rejected_logp  # -1.5

        logits = beta * (chosen_ratio - rejected_ratio)  # beta * 2.0
        loss = -F.logsigmoid(logits)

        print(f"β={beta}: implicit_logits={logits.item():.3f}, "
              f"loss={loss.item():.4f}")

manual_dpo_example()
# β=0.05: implicit_logits=0.100, loss=0.6444
# β=0.1:  implicit_logits=0.200, loss=0.5981
# β=0.5:  implicit_logits=1.000, loss=0.3133

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-llms-transformers/notebooks/ch-34-ch34-dpo-kto-orpo.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-34-ch34-dpo-kto-orpo.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — DPO وKTO وORPO: ما بعد RLHF
Mastering Transformers and Large Language Models — Dpo Kto Orpo

Book slug : mastering-llms-transformers
Chapter   : ch-34-ch34-dpo-kto-orpo (lesson 34)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-07-peft-alignment/ch34-dpo-kto-orpo.ipynb#cell-1
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
# [002] content/notebooks/part-07-peft-alignment/ch34-dpo-kto-orpo.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_sequence_logprobs(model, input_ids: torch.Tensor,
                                attention_mask: torch.Tensor,
                                labels: torch.Tensor) -> torch.Tensor:
    """
    حساب مجموع log-probabilities لتسلسل كامل (الاستجابة فقط،
    وليس التعليمة/prompt — يجب إخفاء التعليمة بـ labels=-100).
    هذا الـ building block مُستخدَم في كل من DPO/KTO/ORPO.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]   # إزاحة بمقدار واحد (next-token prediction)
    labels = labels[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)

    # إخفاء مواضع الـ prompt/padding (labels == -100) من حساب الخسارة
    mask = (labels != -100)
    labels_clamped = labels.clamp(min=0)  # لتفادي index error مؤقتًا
    token_logprobs = torch.gather(
        log_probs, dim=2, index=labels_clamped.unsqueeze(-1)
    ).squeeze(-1)

    token_logprobs = token_logprobs * mask  # تصفير المواضع المُخفاة
    sequence_logprobs = token_logprobs.sum(dim=1)  # مجموع log-probs للاستجابة كاملة
    return sequence_logprobs

class DPOTrainer:
    """
    تطبيق كامل لـ Direct Preference Optimization (Rafailov et al., 2023).
    يتطلب فقط نموذجين: Policy (قابل للتدريب) وReference (مُجمَّد).
    """
    def __init__(self, policy_model: nn.Module, reference_model: nn.Module,
                 beta: float = 0.1):
        self.policy = policy_model
        self.reference = reference_model
        self.beta = beta

        for p in self.reference.parameters():
            p.requires_grad = False
        self.reference.eval()

    def compute_dpo_loss(self, batch: dict) -> dict:
        """
        batch يحتوي على: chosen_input_ids, chosen_attention_mask, chosen_labels,
                          rejected_input_ids, rejected_attention_mask, rejected_labels
        """
        # Log-probs من السياسة الحالية (قابلة للتدريب)
        policy_chosen_logps = compute_sequence_logprobs(
            self.policy, batch["chosen_input_ids"],
            batch["chosen_attention_mask"], batch["chosen_labels"]
        )
        policy_rejected_logps = compute_sequence_logprobs(
            self.policy, batch["rejected_input_ids"],
            batch["rejected_attention_mask"], batch["rejected_labels"]
        )

        # Log-probs من النموذج المرجعي (بدون تدرجات)
        with torch.no_grad():
            ref_chosen_logps = compute_sequence_logprobs(
                self.reference, batch["chosen_input_ids"],
                batch["chosen_attention_mask"], batch["chosen_labels"]
            )
            ref_rejected_logps = compute_sequence_logprobs(
                self.reference, batch["rejected_input_ids"],
                batch["rejected_attention_mask"], batch["rejected_labels"]
            )

        # النسب اللوغاريتمية (implicit reward) لكل من المُفضَّل والمرفوض
        chosen_log_ratio = policy_chosen_logps - ref_chosen_logps
        rejected_log_ratio = policy_rejected_logps - ref_rejected_logps

        # دالة خسارة DPO النهائية (معادلة الاشتقاق أعلاه)
        logits = self.beta * (chosen_log_ratio - rejected_log_ratio)
        loss = -F.logsigmoid(logits).mean()

        # مقاييس مفيدة للمراقبة أثناء التدريب
        with torch.no_grad():
            implicit_reward_chosen = self.beta * chosen_log_ratio
            implicit_reward_rejected = self.beta * rejected_log_ratio
            accuracy = (implicit_reward_chosen > implicit_reward_rejected).float().mean()
            reward_margin = (implicit_reward_chosen - implicit_reward_rejected).mean()

        return {
            "loss": loss,
            "accuracy": accuracy.item(),
            "reward_margin": reward_margin.item(),
            "chosen_reward": implicit_reward_chosen.mean().item(),
            "rejected_reward": implicit_reward_rejected.mean().item(),
        }

    def training_step(self, batch: dict, optimizer):
        optimizer.zero_grad()
        metrics = self.compute_dpo_loss(batch)
        metrics["loss"].backward()
        optimizer.step()
        return {k: (v if isinstance(v, float) else v.item())
                for k, v in metrics.items()}

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-07-peft-alignment/ch34-dpo-kto-orpo.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def ipo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta: float = 0.1):
    """
    IPO (Azar et al., 2023): بدلًا من logsigmoid غير المحدودة،
    نستخدم خسارة تربيعية (MSE-style) مع هدف ثابت = 1/(2β).
    هذا يمنع "الثقة المفرطة" (overconfidence) ويُحسِّن التعميم،
    خصوصًا عندما تحتوي البيانات على تفضيلات قوية جدًا أو حتمية.
    """
    chosen_log_ratio = policy_chosen_logps - ref_chosen_logps
    rejected_log_ratio = policy_rejected_logps - ref_rejected_logps

    h = chosen_log_ratio - rejected_log_ratio
    # الهدف: h يجب أن يقترب من 1/(2β) بالضبط، وليس "أكبر ما يمكن"
    loss = (h - 1 / (2 * beta)) ** 2
    return loss.mean()

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-07-peft-alignment/ch34-dpo-kto-orpo.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def kto_loss(policy_logps: torch.Tensor, ref_logps: torch.Tensor,
             is_desirable: torch.Tensor, beta: float = 0.1,
             desirable_weight: float = 1.0, undesirable_weight: float = 1.0):
    """
    KTO (Ethayarajh et al., 2024): لا يحتاج أزواج (chosen, rejected)!
    كل عينة بيانات مستقلة، فقط بعلامة ثنائية: مرغوبة (1) أو غير مرغوبة (0).

    مُستوحى من نظرية "النفور من الخسارة" (loss aversion) البشرية:
    البشر يتفاعلون بشكل غير متماثل مع المكاسب والخسائر، لذا KTO
    تستخدم دالة قيمة (value function) غير خطية وغير متماثلة.
    """
    log_ratio = policy_logps - ref_logps
    kl_estimate = log_ratio.mean().detach()  # تقدير KL على مستوى الدفعة

    # دالة القيمة غير المتماثلة (مستوحاة من Prospect Theory)
    desirable_losses = 1 - torch.sigmoid(
        beta * (log_ratio - kl_estimate)
    )
    undesirable_losses = 1 - torch.sigmoid(
        beta * (kl_estimate - log_ratio)
    )

    losses = torch.where(
        is_desirable.bool(),
        desirable_weight * desirable_losses,
        undesirable_weight * undesirable_losses,
    )
    return losses.mean()

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-07-peft-alignment/ch34-dpo-kto-orpo.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def orpo_loss(model, chosen_batch: dict, rejected_batch: dict,
              lambda_weight: float = 0.1):
    """
    ORPO (Hong et al., 2024): خسارة واحدة مُركَّبة من جزأين:

    1. SFT Loss (Cross-Entropy) على الاستجابة المُفضَّلة فقط
       (نفس هدف التدريب الخاضع للإشراف العادي)

    2. Odds Ratio Loss: تعظيم نسبة الأرجحية (odds) للاستجابة
       المُفضَّلة على المرفوضة، دون الحاجة لأي reference model!
       (الفارق الجوهري عن DPO: لا نموذج مرجعي مُجمَّد مطلوب)
    """
    chosen_outputs = model(input_ids=chosen_batch["input_ids"],
                             attention_mask=chosen_batch["attention_mask"])
    rejected_outputs = model(input_ids=rejected_batch["input_ids"],
                               attention_mask=rejected_batch["attention_mask"])

    # الجزء الأول: SFT Loss قياسية على الاستجابة المُفضَّلة
    sft_loss = F.cross_entropy(
        chosen_outputs.logits[:, :-1].reshape(-1, chosen_outputs.logits.size(-1)),
        chosen_batch["labels"][:, 1:].reshape(-1),
        ignore_index=-100,
    )

    # الجزء الثاني: Odds Ratio بين الاستجابتين
    def compute_avg_logprob(logits, labels):
        log_probs = F.log_softmax(logits[:, :-1], dim=-1)
        labels_shifted = labels[:, 1:]
        mask = (labels_shifted != -100)
        labels_clamped = labels_shifted.clamp(min=0)
        token_logps = torch.gather(
            log_probs, 2, labels_clamped.unsqueeze(-1)
        ).squeeze(-1) * mask
        # متوسط (وليس مجموع) لتطبيع طول التسلسل - مهم لصيغة Odds Ratio
        return token_logps.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    chosen_logp = compute_avg_logprob(chosen_outputs.logits, chosen_batch["labels"])
    rejected_logp = compute_avg_logprob(rejected_outputs.logits, rejected_batch["labels"])

    # log odds = log(p / (1-p))، حيث p = exp(avg_logprob)
    log_odds_chosen = chosen_logp - torch.log1p(-torch.exp(chosen_logp).clamp(max=1 - 1e-6))
    log_odds_rejected = rejected_logp - torch.log1p(-torch.exp(rejected_logp).clamp(max=1 - 1e-6))

    odds_ratio_loss = -F.logsigmoid(log_odds_chosen - log_odds_rejected).mean()

    total_loss = sft_loss + lambda_weight * odds_ratio_loss
    return {
        "total_loss": total_loss,
        "sft_loss": sft_loss.item(),
        "odds_ratio_loss": odds_ratio_loss.item(),
    }

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-07-peft-alignment/ch34-dpo-kto-orpo.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def manual_dpo_example():
    # افتراضات مبسّطة لقيم log-probability (مجموع log-probs للاستجابة كاملة)
    policy_chosen_logp = torch.tensor(-12.5)    # النموذج الحالي "واثق نسبيًا" من المُفضَّل
    policy_rejected_logp = torch.tensor(-15.0)  # وأقل ثقة في المرفوض (جيد!)
    ref_chosen_logp = torch.tensor(-13.0)       # النموذج المرجعي (قبل أي تحديث)
    ref_rejected_logp = torch.tensor(-13.5)

    for beta in [0.05, 0.1, 0.5]:
        chosen_ratio = policy_chosen_logp - ref_chosen_logp     # 0.5
        rejected_ratio = policy_rejected_logp - ref_rejected_logp  # -1.5

        logits = beta * (chosen_ratio - rejected_ratio)  # beta * 2.0
        loss = -F.logsigmoid(logits)

        print(f"β={beta}: implicit_logits={logits.item():.3f}, "
              f"loss={loss.item():.4f}")

manual_dpo_example()
# β=0.05: implicit_logits=0.100, loss=0.6444
# β=0.1:  implicit_logits=0.200, loss=0.5981
# β=0.5:  implicit_logits=1.000, loss=0.3133

# ────────────────────────────────────────────────────────────────────────
# [007] content/notebooks/part-07-peft-alignment/ch34-dpo-kto-orpo.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-34-ch34-dpo-kto-orpo.ipynb#cell-2
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-34-ch34-dpo-kto-orpo.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_sequence_logprobs(model, input_ids: torch.Tensor,
                                attention_mask: torch.Tensor,
                                labels: torch.Tensor) -> torch.Tensor:
    """
    حساب مجموع log-probabilities لتسلسل كامل (الاستجابة فقط،
    وليس التعليمة/prompt — يجب إخفاء التعليمة بـ labels=-100).
    هذا الـ building block مُستخدَم في كل من DPO/KTO/ORPO.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]   # إزاحة بمقدار واحد (next-token prediction)
    labels = labels[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)

    # إخفاء مواضع الـ prompt/padding (labels == -100) من حساب الخسارة
    mask = (labels != -100)
    labels_clamped = labels.clamp(min=0)  # لتفادي index error مؤقتًا
    token_logprobs = torch.gather(
        log_probs, dim=2, index=labels_clamped.unsqueeze(-1)
    ).squeeze(-1)

    token_logprobs = token_logprobs * mask  # تصفير المواضع المُخفاة
    sequence_logprobs = token_logprobs.sum(dim=1)  # مجموع log-probs للاستجابة كاملة
    return sequence_logprobs

class DPOTrainer:
    """
    تطبيق كامل لـ Direct Preference Optimization (Rafailov et al., 2023).
    يتطلب فقط نموذجين: Policy (قابل للتدريب) وReference (مُجمَّد).
    """
    def __init__(self, policy_model: nn.Module, reference_model: nn.Module,
                 beta: float = 0.1):
        self.policy = policy_model
        self.reference = reference_model
        self.beta = beta

        for p in self.reference.parameters():
            p.requires_grad = False
        self.reference.eval()

    def compute_dpo_loss(self, batch: dict) -> dict:
        """
        batch يحتوي على: chosen_input_ids, chosen_attention_mask, chosen_labels,
                          rejected_input_ids, rejected_attention_mask, rejected_labels
        """
        # Log-probs من السياسة الحالية (قابلة للتدريب)
        policy_chosen_logps = compute_sequence_logprobs(
            self.policy, batch["chosen_input_ids"],
            batch["chosen_attention_mask"], batch["chosen_labels"]
        )
        policy_rejected_logps = compute_sequence_logprobs(
            self.policy, batch["rejected_input_ids"],
            batch["rejected_attention_mask"], batch["rejected_labels"]
        )

        # Log-probs من النموذج المرجعي (بدون تدرجات)
        with torch.no_grad():
            ref_chosen_logps = compute_sequence_logprobs(
                self.reference, batch["chosen_input_ids"],
                batch["chosen_attention_mask"], batch["chosen_labels"]
            )
            ref_rejected_logps = compute_sequence_logprobs(
                self.reference, batch["rejected_input_ids"],
                batch["rejected_attention_mask"], batch["rejected_labels"]
            )

        # النسب اللوغاريتمية (implicit reward) لكل من المُفضَّل والمرفوض
        chosen_log_ratio = policy_chosen_logps - ref_chosen_logps
        rejected_log_ratio = policy_rejected_logps - ref_rejected_logps

        # دالة خسارة DPO النهائية (معادلة الاشتقاق أعلاه)
        logits = self.beta * (chosen_log_ratio - rejected_log_ratio)
        loss = -F.logsigmoid(logits).mean()

        # مقاييس مفيدة للمراقبة أثناء التدريب
        with torch.no_grad():
            implicit_reward_chosen = self.beta * chosen_log_ratio
            implicit_reward_rejected = self.beta * rejected_log_ratio
            accuracy = (implicit_reward_chosen > implicit_reward_rejected).float().mean()
            reward_margin = (implicit_reward_chosen - implicit_reward_rejected).mean()

        return {
            "loss": loss,
            "accuracy": accuracy.item(),
            "reward_margin": reward_margin.item(),
            "chosen_reward": implicit_reward_chosen.mean().item(),
            "rejected_reward": implicit_reward_rejected.mean().item(),
        }

    def training_step(self, batch: dict, optimizer):
        optimizer.zero_grad()
        metrics = self.compute_dpo_loss(batch)
        metrics["loss"].backward()
        optimizer.step()
        return {k: (v if isinstance(v, float) else v.item())
                for k, v in metrics.items()}

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-34-ch34-dpo-kto-orpo.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def ipo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta: float = 0.1):
    """
    IPO (Azar et al., 2023): بدلًا من logsigmoid غير المحدودة،
    نستخدم خسارة تربيعية (MSE-style) مع هدف ثابت = 1/(2β).
    هذا يمنع "الثقة المفرطة" (overconfidence) ويُحسِّن التعميم،
    خصوصًا عندما تحتوي البيانات على تفضيلات قوية جدًا أو حتمية.
    """
    chosen_log_ratio = policy_chosen_logps - ref_chosen_logps
    rejected_log_ratio = policy_rejected_logps - ref_rejected_logps

    h = chosen_log_ratio - rejected_log_ratio
    # الهدف: h يجب أن يقترب من 1/(2β) بالضبط، وليس "أكبر ما يمكن"
    loss = (h - 1 / (2 * beta)) ** 2
    return loss.mean()

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-34-ch34-dpo-kto-orpo.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def kto_loss(policy_logps: torch.Tensor, ref_logps: torch.Tensor,
             is_desirable: torch.Tensor, beta: float = 0.1,
             desirable_weight: float = 1.0, undesirable_weight: float = 1.0):
    """
    KTO (Ethayarajh et al., 2024): لا يحتاج أزواج (chosen, rejected)!
    كل عينة بيانات مستقلة، فقط بعلامة ثنائية: مرغوبة (1) أو غير مرغوبة (0).

    مُستوحى من نظرية "النفور من الخسارة" (loss aversion) البشرية:
    البشر يتفاعلون بشكل غير متماثل مع المكاسب والخسائر، لذا KTO
    تستخدم دالة قيمة (value function) غير خطية وغير متماثلة.
    """
    log_ratio = policy_logps - ref_logps
    kl_estimate = log_ratio.mean().detach()  # تقدير KL على مستوى الدفعة

    # دالة القيمة غير المتماثلة (مستوحاة من Prospect Theory)
    desirable_losses = 1 - torch.sigmoid(
        beta * (log_ratio - kl_estimate)
    )
    undesirable_losses = 1 - torch.sigmoid(
        beta * (kl_estimate - log_ratio)
    )

    losses = torch.where(
        is_desirable.bool(),
        desirable_weight * desirable_losses,
        undesirable_weight * undesirable_losses,
    )
    return losses.mean()

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-34-ch34-dpo-kto-orpo.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def orpo_loss(model, chosen_batch: dict, rejected_batch: dict,
              lambda_weight: float = 0.1):
    """
    ORPO (Hong et al., 2024): خسارة واحدة مُركَّبة من جزأين:

    1. SFT Loss (Cross-Entropy) على الاستجابة المُفضَّلة فقط
       (نفس هدف التدريب الخاضع للإشراف العادي)

    2. Odds Ratio Loss: تعظيم نسبة الأرجحية (odds) للاستجابة
       المُفضَّلة على المرفوضة، دون الحاجة لأي reference model!
       (الفارق الجوهري عن DPO: لا نموذج مرجعي مُجمَّد مطلوب)
    """
    chosen_outputs = model(input_ids=chosen_batch["input_ids"],
                             attention_mask=chosen_batch["attention_mask"])
    rejected_outputs = model(input_ids=rejected_batch["input_ids"],
                               attention_mask=rejected_batch["attention_mask"])

    # الجزء الأول: SFT Loss قياسية على الاستجابة المُفضَّلة
    sft_loss = F.cross_entropy(
        chosen_outputs.logits[:, :-1].reshape(-1, chosen_outputs.logits.size(-1)),
        chosen_batch["labels"][:, 1:].reshape(-1),
        ignore_index=-100,
    )

    # الجزء الثاني: Odds Ratio بين الاستجابتين
    def compute_avg_logprob(logits, labels):
        log_probs = F.log_softmax(logits[:, :-1], dim=-1)
        labels_shifted = labels[:, 1:]
        mask = (labels_shifted != -100)
        labels_clamped = labels_shifted.clamp(min=0)
        token_logps = torch.gather(
            log_probs, 2, labels_clamped.unsqueeze(-1)
        ).squeeze(-1) * mask
        # متوسط (وليس مجموع) لتطبيع طول التسلسل - مهم لصيغة Odds Ratio
        return token_logps.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    chosen_logp = compute_avg_logprob(chosen_outputs.logits, chosen_batch["labels"])
    rejected_logp = compute_avg_logprob(rejected_outputs.logits, rejected_batch["labels"])

    # log odds = log(p / (1-p))، حيث p = exp(avg_logprob)
    log_odds_chosen = chosen_logp - torch.log1p(-torch.exp(chosen_logp).clamp(max=1 - 1e-6))
    log_odds_rejected = rejected_logp - torch.log1p(-torch.exp(rejected_logp).clamp(max=1 - 1e-6))

    odds_ratio_loss = -F.logsigmoid(log_odds_chosen - log_odds_rejected).mean()

    total_loss = sft_loss + lambda_weight * odds_ratio_loss
    return {
        "total_loss": total_loss,
        "sft_loss": sft_loss.item(),
        "odds_ratio_loss": odds_ratio_loss.item(),
    }

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-llms-transformers/notebooks/ch-34-ch34-dpo-kto-orpo.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def manual_dpo_example():
    # افتراضات مبسّطة لقيم log-probability (مجموع log-probs للاستجابة كاملة)
    policy_chosen_logp = torch.tensor(-12.5)    # النموذج الحالي "واثق نسبيًا" من المُفضَّل
    policy_rejected_logp = torch.tensor(-15.0)  # وأقل ثقة في المرفوض (جيد!)
    ref_chosen_logp = torch.tensor(-13.0)       # النموذج المرجعي (قبل أي تحديث)
    ref_rejected_logp = torch.tensor(-13.5)

    for beta in [0.05, 0.1, 0.5]:
        chosen_ratio = policy_chosen_logp - ref_chosen_logp     # 0.5
        rejected_ratio = policy_rejected_logp - ref_rejected_logp  # -1.5

        logits = beta * (chosen_ratio - rejected_ratio)  # beta * 2.0
        loss = -F.logsigmoid(logits)

        print(f"β={beta}: implicit_logits={logits.item():.3f}, "
              f"loss={loss.item():.4f}")

manual_dpo_example()
# β=0.05: implicit_logits=0.100, loss=0.6444
# β=0.1:  implicit_logits=0.200, loss=0.5981
# β=0.5:  implicit_logits=1.000, loss=0.3133

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-llms-transformers/notebooks/ch-34-ch34-dpo-kto-orpo.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
