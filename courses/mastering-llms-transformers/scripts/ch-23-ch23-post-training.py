#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — خط إنتاج ما بعد التدريب
Mastering Transformers and Large Language Models — Post Training

Book slug : mastering-llms-transformers
Chapter   : ch-23-ch23-post-training (lesson 23)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-23-ch23-post-training.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — خط إنتاج ما بعد التدريب
Mastering Transformers and Large Language Models — Post Training

Book slug : mastering-llms-transformers
Chapter   : ch-23-ch23-post-training (lesson 23)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-23-ch23-post-training.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — خط إنتاج ما بعد التدريب
Mastering Transformers and Large Language Models — Post Training

Book slug : mastering-llms-transformers
Chapter   : ch-23-ch23-post-training (lesson 23)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-05-building-llms/ch23-post-training.ipynb#cell-1
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
# [002] content/notebooks/part-05-building-llms/ch23-post-training.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class InstructionTuningDataset(Dataset):
    """
    مجموعة بيانات SFT: كل عينة هي محادثة مُنسَّقة بقالب موحَّد،
    حيث تُحسَب دالة الخسارة فقط على جزء "استجابة المساعد"
    (Response Masking)، وليس على نص التعليمة نفسها، لأن هدفنا
    تعليم النموذج "كيف يستجيب" وليس "كيف يكتب تعليمات".
    """
    def __init__(self, conversations: list, tokenizer, max_length: int = 2048):
        """
        conversations: قائمة من القواميس بالشكل:
        {"system": "...", "user": "...", "assistant": "..."}
        """
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conv = self.conversations[idx]

        # بناء القالب الكامل بالرموز الخاصة التي أضفناها عند تدريب
        # Tokenizer في الفصل 19 (<|system|>, <|user|>, <|assistant|>)
        system_part = f"<|system|>{conv.get('system', 'أنت مساعد ذكي ومفيد.')}"
        user_part = f"<|user|>{conv['user']}"
        assistant_part = f"<|assistant|>{conv['assistant']}<|endoftext|>"

        prompt_text = system_part + user_part + "<|assistant|>"
        full_text = system_part + user_part + assistant_part

        prompt_ids = self.tokenizer.encode(prompt_text).ids
        full_ids = self.tokenizer.encode(full_text).ids

        full_ids = full_ids[:self.max_length]
        labels = full_ids.copy()

        # Response Masking: نُخفي (بقيمة -100 التي يتجاهلها PyTorch تلقائيًا
        # في حساب CrossEntropyLoss) كل الأجزاء التي ليست من استجابة المساعد
        prompt_length = min(len(prompt_ids), len(labels))
        for i in range(prompt_length):
            labels[i] = -100

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def sft_training_step(model, batch, optimizer):
    """
    خطوة تدريب SFT واحدة. لاحظ أن الخسارة محسوبة فقط على الأجزاء
    غير المُخفية (استجابة المساعد)، بفضل قيمة -100 في labels التي
    يتجاهلها nn.CrossEntropyLoss تلقائيًا (ignore_index=-100 افتراضيًا).
    """
    optimizer.zero_grad()
    outputs = model(batch["input_ids"])
    logits = outputs[:, :-1, :].contiguous()
    labels = batch["labels"][:, 1:].contiguous()

    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-05-building-llms/ch23-post-training.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def self_instruct_pipeline_concept(seed_tasks: list,
                                     num_instructions_to_generate: int,
                                     teacher_model_generate_fn) -> list:
    """
    محاكاة مفاهيمية لخط إنتاج Self-Instruct لتوليد بيانات SFT اصطناعية.
    teacher_model_generate_fn: دالة تستدعي نموذجًا قويًا موجودًا (GPT-4 مثلاً)
    لتوليد نص من prompt معين.
    """
    generated_examples = []
    task_pool = seed_tasks.copy()

    while len(generated_examples) < num_instructions_to_generate:
        # 1. اختيار عينة عشوائية من المهام الموجودة (بذرة + مولَّدة سابقًا)
        #    كأمثلة توضيحية (Few-shot) لتوجيه النموذج المعلِّم
        import random
        few_shot_examples = random.sample(task_pool, min(4, len(task_pool)))

        # 2. بناء prompt يطلب من النموذج المعلِّم توليد تعليمة جديدة مختلفة
        generation_prompt = build_instruction_generation_prompt(few_shot_examples)
        new_instruction = teacher_model_generate_fn(generation_prompt)

        # 3. فلترة الجودة: التحقق من عدم التكرار الشديد مع المهام الموجودة
        if is_sufficiently_novel(new_instruction, task_pool):
            # 4. توليد الاستجابة المثالية لهذه التعليمة الجديدة بنفس النموذج المعلِّم
            response_prompt = f"أجب على التعليمة التالية بدقة ووضوح:\n{new_instruction}"
            new_response = teacher_model_generate_fn(response_prompt)

            new_example = {"instruction": new_instruction, "response": new_response}
            generated_examples.append(new_example)
            task_pool.append(new_example)  # إثراء البركة للتكرارات التالية

    return generated_examples

def build_instruction_generation_prompt(few_shot_examples: list) -> str:
    """بناء prompt يحتوي أمثلة توضيحية لتوجيه النموذج لتوليد تعليمة جديدة بنفس الأسلوب"""
    examples_text = "\n\n".join([
        f"تعليمة: {ex['instruction']}" for ex in few_shot_examples
    ])
    return (f"فيما يلي أمثلة على تعليمات متنوعة:\n\n{examples_text}\n\n"
            f"اكتب تعليمة جديدة مختلفة تمامًا عن الأمثلة أعلاه:\nتعليمة:")

def is_sufficiently_novel(new_instruction: str, existing_pool: list,
                            similarity_threshold: float = 0.7) -> bool:
    """فحص تشابه (مبسَّط هنا) للتأكد من تنوع البيانات المولَّدة"""
    # في التطبيق الفعلي: نستخدم ROUGE-L أو تشابه embeddings دلالي
    # هنا تبسيط مفاهيمي فقط
    return True  # افتراضيًا للتوضيح المفاهيمي

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-05-building-llms/ch23-post-training.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel(nn.Module):
    """
    نموذج المكافأة: يأخذ معمارية نموذج اللغة الأساسي (المُدرَّب مسبقًا
    وربما بعد SFT)، ويستبدل رأس الإخراج (الذي كان يُنتج توزيعًا على
    المفردات) برأس انحدار بسيط يُنتج قيمة عددية واحدة (Scalar Reward)
    تُمثِّل "جودة" الاستجابة الكاملة من منظور التفضيلات البشرية.
    """
    def __init__(self, base_transformer_model, hidden_size: int):
        super().__init__()
        self.transformer = base_transformer_model
        # رأس المكافأة: طبقة خطية بسيطة تُسقِط التمثيل المخفي لقيمة واحدة
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # نأخذ التمثيل المخفي عند آخر رمز في التسلسل (يُلخِّص الاستجابة كاملة)
        hidden_states = self.transformer(input_ids, return_hidden_states=True)
        last_token_hidden = hidden_states[:, -1, :]  # [batch, hidden_size]
        reward_scalar = self.reward_head(last_token_hidden)  # [batch, 1]
        return reward_scalar.squeeze(-1)  # [batch]

def reward_model_training_step(reward_model: RewardModel,
                                 chosen_input_ids: torch.Tensor,
                                 rejected_input_ids: torch.Tensor,
                                 optimizer) -> float:
    """
    خطوة تدريب نموذج المكافأة باستخدام دالة خسارة Bradley-Terry
    (التصنيف الثنائي الاحتمالي، أساس كل نماذج المكافأة في RLHF):

    L = -log(σ(r_chosen - r_rejected))

    هذه الصيغة تدفع النموذج لإعطاء مكافأة أعلى للاستجابة المُفضَّلة بشريًا
    (chosen) مقارنة بالاستجابة المرفوضة (rejected)، دون الحاجة لقيم
    مكافأة "مطلقة" بل فقط لـ"ترتيب نسبي" صحيح بين الأزواج.
    """
    optimizer.zero_grad()

    reward_chosen = reward_model(chosen_input_ids)      # [batch]
    reward_rejected = reward_model(rejected_input_ids)  # [batch]

    # خسارة Bradley-Terry: نريد أن يكون σ(r_chosen - r_rejected) قريبًا من 1
    loss = -F.logsigmoid(reward_chosen - reward_rejected).mean()

    loss.backward()
    optimizer.step()

    # دقة الترتيب: كم نسبة الأزواج التي أعطاها النموذج مكافأة صحيحة الترتيب
    accuracy = (reward_chosen > reward_rejected).float().mean().item()

    return loss.item(), accuracy

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-05-building-llms/ch23-post-training.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
"""
RLHF بـ PPO (Proximal Policy Optimization) يتطلب الاحتفاظ بأربعة نماذج
في الذاكرة في وقت واحد أثناء التدريب، وهذا يُفسِّر لماذا هذا المسار
مُكلِف حسابيًا جدًا مقارنة بالبدائل الحديثة (DPO وما شابه):

1. Policy Model (π_θ): النموذج الذي نُحسِّنه فعليًا (يبدأ كنسخة من نموذج SFT)
2. Reference Model (π_ref): نسخة مجمَّدة من نموذج SFT، تُستخدَم فقط لحساب
   عقوبة KL-Divergence لمنع π_θ من الانحراف بعيدًا جدًا عن سلوكه الأصلي
   (الانحراف المفرط قد يُنتج استجابات "تخدع" نموذج المكافأة لكنها غير متماسكة)
3. Reward Model (RM): النموذج المُدرَّب في القسم 3.3، مُجمَّد أثناء RLHF،
   يُعطي إشارة "المكافأة" لكل استجابة يولدها Policy Model
4. Value Model (Critic): نموذج يُقدِّر "القيمة المتوقعة" لحالة معينة،
   يُستخدَم لتقليل تباين تقدير الـ Advantage في خوارزمية PPO

معادلة الهدف الكلي التي يُحسِّنها PPO في سياق RLHF:

objective(θ) = E[r_φ(x,y) - β·KL(π_θ(y|x) || π_ref(y|x))]

حيث:
- r_φ(x,y): مكافأة الاستجابة y للمدخل x وفق نموذج المكافأة
- β: معامل يتحكم في قوة عقوبة الانحراف عن π_ref (قيمة نموذجية: 0.01-0.2)
- KL term: يمنع "اختراق" نموذج المكافأة (Reward Hacking) عبر استجابات
  غريبة تُحقق مكافأة عالية لكنها بعيدة عن اللغة الطبيعية المتماسكة
"""

def ppo_rlhf_training_loop_concept(policy_model, reference_model,
                                     reward_model, value_model,
                                     prompts_batch, optimizer, kl_coefficient=0.1):
    """
    هيكل مفاهيمي مبسَّط لحلقة تدريب PPO في سياق RLHF.
    التفاصيل الرياضية الكاملة (GAE، Clipping، إلخ) مُغطَّاة بعمق في الجزء السابع.
    """
    # 1. توليد استجابات من Policy Model الحالي لكل prompt في الدفعة
    generated_responses = policy_model.generate(prompts_batch)

    # 2. حساب المكافأة الخام من نموذج المكافأة المُجمَّد
    with torch.no_grad():
        raw_rewards = reward_model(generated_responses)

    # 3. حساب احتمالات log-probability من كل من Policy وReference
    policy_log_probs = policy_model.compute_log_probs(generated_responses)
    with torch.no_grad():
        reference_log_probs = reference_model.compute_log_probs(generated_responses)

    # 4. حساب عقوبة KL وطرحها من المكافأة الخام (منع الانحراف المفرط)
    kl_penalty = kl_coefficient * (policy_log_probs - reference_log_probs)
    adjusted_rewards = raw_rewards - kl_penalty

    # 5. حساب الـ Advantage باستخدام Value Model (GAE - تفصيل في الجزء السابع)
    with torch.no_grad():
        values = value_model(generated_responses)
    advantages = adjusted_rewards - values  # تبسيط؛ GAE الفعلي أكثر تعقيدًا

    # 6. تحديث Policy Model عبر هدف PPO المُقلَّم (Clipped Objective)
    # (التفاصيل الكاملة لـ clipping ratio في الجزء السابع)
    ppo_loss = compute_ppo_clipped_loss(policy_log_probs, advantages)

    optimizer.zero_grad()
    ppo_loss.backward()
    optimizer.step()

    return ppo_loss.item()

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-05-building-llms/ch23-post-training.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def dpo_loss_function(policy_chosen_logps: torch.Tensor,
                        policy_rejected_logps: torch.Tensor,
                        reference_chosen_logps: torch.Tensor,
                        reference_rejected_logps: torch.Tensor,
                        beta: float = 0.1) -> torch.Tensor:
    """
    دالة خسارة DPO الكاملة، كما اشتُقَّت في الورقة الأصلية.
    هذا هو "قلب" تبسيط RLHF إلى مجرد Fine-tuning مباشر بدون RL.
    (تفصيل الاشتقاق الكامل من هدف RLHF الأصلي موجود في الجزء السابع)
    """
    # نسبة لوغاريتمية الاحتمال (Policy مقابل Reference) لكل من الاستجابة
    # المُفضَّلة والمرفوضة
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    reference_logratios = reference_chosen_logps - reference_rejected_logps

    logits = beta * (policy_logratios - reference_logratios)

    # نفس صيغة Bradley-Terry المستخدمة في تدريب نموذج المكافأة تمامًا،
    # لكن مُطبَّقة مباشرة على النموذج النهائي دون وسيط (نموذج مكافأة منفصل)
    loss = -F.logsigmoid(logits).mean()

    # مقاييس إضافية مفيدة للمراقبة أثناء التدريب
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()

    return loss, reward_accuracy, chosen_rewards.mean(), rejected_rewards.mean()

# ────────────────────────────────────────────────────────────────────────
# [007] content/notebooks/part-05-building-llms/ch23-post-training.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def compare_alignment_methods():
    methods = {
        "RLHF (PPO)": {
            "عدد النماذج في الذاكرة": 4,
            "يحتاج Reward Model منفصل": "نعم",
            "استقرار التدريب": "صعب (حساس جدًا للمعاملات)",
            "التعقيد الهندسي": "عالٍ جدًا",
        },
        "DPO": {
            "عدد النماذج في الذاكرة": 2,
            "يحتاج Reward Model منفصل": "لا",
            "استقرار التدريب": "أكثر استقرارًا بكثير",
            "التعقيد الهندسي": "بسيط (شبيه بـ SFT)",
        },
        "KTO": {
            "عدد النماذج في الذاكرة": 2,
            "يحتاج Reward Model منفصل": "لا",
            "استقرار التدريب": "جيد، يعمل حتى ببيانات غير مزدوجة (binary فقط)",
            "التعقيد الهندسي": "بسيط",
        },
        "ORPO": {
            "عدد النماذج في الذاكرة": 1,
            "يحتاج Reward Model منفصل": "لا، ولا حتى Reference Model!",
            "استقرار التدريب": "جيد، يدمج SFT والمحاذاة في خطوة واحدة",
            "التعقيد الهندسي": "الأبسط على الإطلاق",
        },
    }
    for method, details in methods.items():
        print(f"\n{method}:")
        for k, v in details.items():
            print(f"   {k}: {v}")

compare_alignment_methods()

# ────────────────────────────────────────────────────────────────────────
# [008] content/notebooks/part-05-building-llms/ch23-post-training.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
def complete_post_training_pipeline_walkthrough():
    """
    محاكاة مفاهيمية لخط الإنتاج الكامل من Base إلى نموذج محاذى نهائيًا،
    موضحة تدرج كل مرحلة وأثرها المتوقع على سلوك النموذج.
    """
    stages = [
        {
            "stage": "0. Base Model",
            "data": "15 مليار رمز (ويب منظَّف، ويكيبيديا، كتب)",
            "behavior": "يُكمل النص إحصائيًا، لا يفهم 'التعليمات' كمفهوم",
            "example_prompt": "ما هي عاصمة فرنسا؟",
            "example_output": "ما هي عاصمة ألمانيا؟ ما هي عاصمة إيطاليا؟ ...",
        },
        {
            "stage": "1. بعد SFT",
            "data": "~50,000 زوج (تعليمة، استجابة) عالية الجودة",
            "behavior": "يفهم بنية 'سؤال→إجابة مباشرة'، يتبع تعليمات بسيطة",
            "example_prompt": "ما هي عاصمة فرنسا؟",
            "example_output": "عاصمة فرنسا هي باريس.",
        },
        {
            "stage": "2. بعد RM (لا يُغيِّر سلوك التوليد مباشرة)",
            "data": "~10,000 زوج تفضيل بشري (مُفضَّل مقابل مرفوض)",
            "behavior": "النموذج الأصلي لم يتغير بعد، فقط بُنيَ 'حَكَم' منفصل",
            "example_prompt": "(غير مطبَّق في هذه المرحلة مباشرة)",
            "example_output": "RM يُعطي: الإجابة المباشرة = مكافأة عالية، الإجابة المطوَّلة بلا داعٍ = مكافأة أقل",
        },
        {
            "stage": "3. بعد DPO/RLHF",
            "data": "نفس بيانات التفضيل، لكن تُستخدَم لتحديث النموذج مباشرة",
            "behavior": "استجابات أكثر دقة، أكثر أمانًا، أقرب لتفضيلات المُقيِّمين البشريين",
            "example_prompt": "اشرح لي كيف أصنع متفجرات في المنزل",
            "example_output": "لا يمكنني المساعدة في هذا الطلب لأنه قد يُسبب ضررًا جسيمًا...",
        },
    ]

    for s in stages:
        print(f"\n{'='*70}")
        print(f"المرحلة: {s['stage']}")
        print(f"البيانات المُستخدَمة: {s['data']}")
        print(f"السلوك الناتج: {s['behavior']}")
        print(f"مثال - المدخل: {s['example_prompt']}")
        print(f"مثال - المخرج المتوقع: {s['example_output']}")

complete_post_training_pipeline_walkthrough()

# ────────────────────────────────────────────────────────────────────────
# [009] content/notebooks/part-05-building-llms/ch23-post-training.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-23-ch23-post-training.ipynb#cell-1
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-23-ch23-post-training.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class InstructionTuningDataset(Dataset):
    """
    مجموعة بيانات SFT: كل عينة هي محادثة مُنسَّقة بقالب موحَّد،
    حيث تُحسَب دالة الخسارة فقط على جزء "استجابة المساعد"
    (Response Masking)، وليس على نص التعليمة نفسها، لأن هدفنا
    تعليم النموذج "كيف يستجيب" وليس "كيف يكتب تعليمات".
    """
    def __init__(self, conversations: list, tokenizer, max_length: int = 2048):
        """
        conversations: قائمة من القواميس بالشكل:
        {"system": "...", "user": "...", "assistant": "..."}
        """
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conv = self.conversations[idx]

        # بناء القالب الكامل بالرموز الخاصة التي أضفناها عند تدريب
        # Tokenizer في الفصل 19 (<|system|>, <|user|>, <|assistant|>)
        system_part = f"<|system|>{conv.get('system', 'أنت مساعد ذكي ومفيد.')}"
        user_part = f"<|user|>{conv['user']}"
        assistant_part = f"<|assistant|>{conv['assistant']}<|endoftext|>"

        prompt_text = system_part + user_part + "<|assistant|>"
        full_text = system_part + user_part + assistant_part

        prompt_ids = self.tokenizer.encode(prompt_text).ids
        full_ids = self.tokenizer.encode(full_text).ids

        full_ids = full_ids[:self.max_length]
        labels = full_ids.copy()

        # Response Masking: نُخفي (بقيمة -100 التي يتجاهلها PyTorch تلقائيًا
        # في حساب CrossEntropyLoss) كل الأجزاء التي ليست من استجابة المساعد
        prompt_length = min(len(prompt_ids), len(labels))
        for i in range(prompt_length):
            labels[i] = -100

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def sft_training_step(model, batch, optimizer):
    """
    خطوة تدريب SFT واحدة. لاحظ أن الخسارة محسوبة فقط على الأجزاء
    غير المُخفية (استجابة المساعد)، بفضل قيمة -100 في labels التي
    يتجاهلها nn.CrossEntropyLoss تلقائيًا (ignore_index=-100 افتراضيًا).
    """
    optimizer.zero_grad()
    outputs = model(batch["input_ids"])
    logits = outputs[:, :-1, :].contiguous()
    labels = batch["labels"][:, 1:].contiguous()

    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-23-ch23-post-training.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def self_instruct_pipeline_concept(seed_tasks: list,
                                     num_instructions_to_generate: int,
                                     teacher_model_generate_fn) -> list:
    """
    محاكاة مفاهيمية لخط إنتاج Self-Instruct لتوليد بيانات SFT اصطناعية.
    teacher_model_generate_fn: دالة تستدعي نموذجًا قويًا موجودًا (GPT-4 مثلاً)
    لتوليد نص من prompt معين.
    """
    generated_examples = []
    task_pool = seed_tasks.copy()

    while len(generated_examples) < num_instructions_to_generate:
        # 1. اختيار عينة عشوائية من المهام الموجودة (بذرة + مولَّدة سابقًا)
        #    كأمثلة توضيحية (Few-shot) لتوجيه النموذج المعلِّم
        import random
        few_shot_examples = random.sample(task_pool, min(4, len(task_pool)))

        # 2. بناء prompt يطلب من النموذج المعلِّم توليد تعليمة جديدة مختلفة
        generation_prompt = build_instruction_generation_prompt(few_shot_examples)
        new_instruction = teacher_model_generate_fn(generation_prompt)

        # 3. فلترة الجودة: التحقق من عدم التكرار الشديد مع المهام الموجودة
        if is_sufficiently_novel(new_instruction, task_pool):
            # 4. توليد الاستجابة المثالية لهذه التعليمة الجديدة بنفس النموذج المعلِّم
            response_prompt = f"أجب على التعليمة التالية بدقة ووضوح:\n{new_instruction}"
            new_response = teacher_model_generate_fn(response_prompt)

            new_example = {"instruction": new_instruction, "response": new_response}
            generated_examples.append(new_example)
            task_pool.append(new_example)  # إثراء البركة للتكرارات التالية

    return generated_examples

def build_instruction_generation_prompt(few_shot_examples: list) -> str:
    """بناء prompt يحتوي أمثلة توضيحية لتوجيه النموذج لتوليد تعليمة جديدة بنفس الأسلوب"""
    examples_text = "\n\n".join([
        f"تعليمة: {ex['instruction']}" for ex in few_shot_examples
    ])
    return (f"فيما يلي أمثلة على تعليمات متنوعة:\n\n{examples_text}\n\n"
            f"اكتب تعليمة جديدة مختلفة تمامًا عن الأمثلة أعلاه:\nتعليمة:")

def is_sufficiently_novel(new_instruction: str, existing_pool: list,
                            similarity_threshold: float = 0.7) -> bool:
    """فحص تشابه (مبسَّط هنا) للتأكد من تنوع البيانات المولَّدة"""
    # في التطبيق الفعلي: نستخدم ROUGE-L أو تشابه embeddings دلالي
    # هنا تبسيط مفاهيمي فقط
    return True  # افتراضيًا للتوضيح المفاهيمي

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-23-ch23-post-training.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel(nn.Module):
    """
    نموذج المكافأة: يأخذ معمارية نموذج اللغة الأساسي (المُدرَّب مسبقًا
    وربما بعد SFT)، ويستبدل رأس الإخراج (الذي كان يُنتج توزيعًا على
    المفردات) برأس انحدار بسيط يُنتج قيمة عددية واحدة (Scalar Reward)
    تُمثِّل "جودة" الاستجابة الكاملة من منظور التفضيلات البشرية.
    """
    def __init__(self, base_transformer_model, hidden_size: int):
        super().__init__()
        self.transformer = base_transformer_model
        # رأس المكافأة: طبقة خطية بسيطة تُسقِط التمثيل المخفي لقيمة واحدة
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # نأخذ التمثيل المخفي عند آخر رمز في التسلسل (يُلخِّص الاستجابة كاملة)
        hidden_states = self.transformer(input_ids, return_hidden_states=True)
        last_token_hidden = hidden_states[:, -1, :]  # [batch, hidden_size]
        reward_scalar = self.reward_head(last_token_hidden)  # [batch, 1]
        return reward_scalar.squeeze(-1)  # [batch]

def reward_model_training_step(reward_model: RewardModel,
                                 chosen_input_ids: torch.Tensor,
                                 rejected_input_ids: torch.Tensor,
                                 optimizer) -> float:
    """
    خطوة تدريب نموذج المكافأة باستخدام دالة خسارة Bradley-Terry
    (التصنيف الثنائي الاحتمالي، أساس كل نماذج المكافأة في RLHF):

    L = -log(σ(r_chosen - r_rejected))

    هذه الصيغة تدفع النموذج لإعطاء مكافأة أعلى للاستجابة المُفضَّلة بشريًا
    (chosen) مقارنة بالاستجابة المرفوضة (rejected)، دون الحاجة لقيم
    مكافأة "مطلقة" بل فقط لـ"ترتيب نسبي" صحيح بين الأزواج.
    """
    optimizer.zero_grad()

    reward_chosen = reward_model(chosen_input_ids)      # [batch]
    reward_rejected = reward_model(rejected_input_ids)  # [batch]

    # خسارة Bradley-Terry: نريد أن يكون σ(r_chosen - r_rejected) قريبًا من 1
    loss = -F.logsigmoid(reward_chosen - reward_rejected).mean()

    loss.backward()
    optimizer.step()

    # دقة الترتيب: كم نسبة الأزواج التي أعطاها النموذج مكافأة صحيحة الترتيب
    accuracy = (reward_chosen > reward_rejected).float().mean().item()

    return loss.item(), accuracy

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-23-ch23-post-training.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
"""
RLHF بـ PPO (Proximal Policy Optimization) يتطلب الاحتفاظ بأربعة نماذج
في الذاكرة في وقت واحد أثناء التدريب، وهذا يُفسِّر لماذا هذا المسار
مُكلِف حسابيًا جدًا مقارنة بالبدائل الحديثة (DPO وما شابه):

1. Policy Model (π_θ): النموذج الذي نُحسِّنه فعليًا (يبدأ كنسخة من نموذج SFT)
2. Reference Model (π_ref): نسخة مجمَّدة من نموذج SFT، تُستخدَم فقط لحساب
   عقوبة KL-Divergence لمنع π_θ من الانحراف بعيدًا جدًا عن سلوكه الأصلي
   (الانحراف المفرط قد يُنتج استجابات "تخدع" نموذج المكافأة لكنها غير متماسكة)
3. Reward Model (RM): النموذج المُدرَّب في القسم 3.3، مُجمَّد أثناء RLHF،
   يُعطي إشارة "المكافأة" لكل استجابة يولدها Policy Model
4. Value Model (Critic): نموذج يُقدِّر "القيمة المتوقعة" لحالة معينة،
   يُستخدَم لتقليل تباين تقدير الـ Advantage في خوارزمية PPO

معادلة الهدف الكلي التي يُحسِّنها PPO في سياق RLHF:

objective(θ) = E[r_φ(x,y) - β·KL(π_θ(y|x) || π_ref(y|x))]

حيث:
- r_φ(x,y): مكافأة الاستجابة y للمدخل x وفق نموذج المكافأة
- β: معامل يتحكم في قوة عقوبة الانحراف عن π_ref (قيمة نموذجية: 0.01-0.2)
- KL term: يمنع "اختراق" نموذج المكافأة (Reward Hacking) عبر استجابات
  غريبة تُحقق مكافأة عالية لكنها بعيدة عن اللغة الطبيعية المتماسكة
"""

def ppo_rlhf_training_loop_concept(policy_model, reference_model,
                                     reward_model, value_model,
                                     prompts_batch, optimizer, kl_coefficient=0.1):
    """
    هيكل مفاهيمي مبسَّط لحلقة تدريب PPO في سياق RLHF.
    التفاصيل الرياضية الكاملة (GAE، Clipping، إلخ) مُغطَّاة بعمق في الجزء السابع.
    """
    # 1. توليد استجابات من Policy Model الحالي لكل prompt في الدفعة
    generated_responses = policy_model.generate(prompts_batch)

    # 2. حساب المكافأة الخام من نموذج المكافأة المُجمَّد
    with torch.no_grad():
        raw_rewards = reward_model(generated_responses)

    # 3. حساب احتمالات log-probability من كل من Policy وReference
    policy_log_probs = policy_model.compute_log_probs(generated_responses)
    with torch.no_grad():
        reference_log_probs = reference_model.compute_log_probs(generated_responses)

    # 4. حساب عقوبة KL وطرحها من المكافأة الخام (منع الانحراف المفرط)
    kl_penalty = kl_coefficient * (policy_log_probs - reference_log_probs)
    adjusted_rewards = raw_rewards - kl_penalty

    # 5. حساب الـ Advantage باستخدام Value Model (GAE - تفصيل في الجزء السابع)
    with torch.no_grad():
        values = value_model(generated_responses)
    advantages = adjusted_rewards - values  # تبسيط؛ GAE الفعلي أكثر تعقيدًا

    # 6. تحديث Policy Model عبر هدف PPO المُقلَّم (Clipped Objective)
    # (التفاصيل الكاملة لـ clipping ratio في الجزء السابع)
    ppo_loss = compute_ppo_clipped_loss(policy_log_probs, advantages)

    optimizer.zero_grad()
    ppo_loss.backward()
    optimizer.step()

    return ppo_loss.item()

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-23-ch23-post-training.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def dpo_loss_function(policy_chosen_logps: torch.Tensor,
                        policy_rejected_logps: torch.Tensor,
                        reference_chosen_logps: torch.Tensor,
                        reference_rejected_logps: torch.Tensor,
                        beta: float = 0.1) -> torch.Tensor:
    """
    دالة خسارة DPO الكاملة، كما اشتُقَّت في الورقة الأصلية.
    هذا هو "قلب" تبسيط RLHF إلى مجرد Fine-tuning مباشر بدون RL.
    (تفصيل الاشتقاق الكامل من هدف RLHF الأصلي موجود في الجزء السابع)
    """
    # نسبة لوغاريتمية الاحتمال (Policy مقابل Reference) لكل من الاستجابة
    # المُفضَّلة والمرفوضة
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    reference_logratios = reference_chosen_logps - reference_rejected_logps

    logits = beta * (policy_logratios - reference_logratios)

    # نفس صيغة Bradley-Terry المستخدمة في تدريب نموذج المكافأة تمامًا،
    # لكن مُطبَّقة مباشرة على النموذج النهائي دون وسيط (نموذج مكافأة منفصل)
    loss = -F.logsigmoid(logits).mean()

    # مقاييس إضافية مفيدة للمراقبة أثناء التدريب
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()

    return loss, reward_accuracy, chosen_rewards.mean(), rejected_rewards.mean()

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-llms-transformers/notebooks/ch-23-ch23-post-training.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def compare_alignment_methods():
    methods = {
        "RLHF (PPO)": {
            "عدد النماذج في الذاكرة": 4,
            "يحتاج Reward Model منفصل": "نعم",
            "استقرار التدريب": "صعب (حساس جدًا للمعاملات)",
            "التعقيد الهندسي": "عالٍ جدًا",
        },
        "DPO": {
            "عدد النماذج في الذاكرة": 2,
            "يحتاج Reward Model منفصل": "لا",
            "استقرار التدريب": "أكثر استقرارًا بكثير",
            "التعقيد الهندسي": "بسيط (شبيه بـ SFT)",
        },
        "KTO": {
            "عدد النماذج في الذاكرة": 2,
            "يحتاج Reward Model منفصل": "لا",
            "استقرار التدريب": "جيد، يعمل حتى ببيانات غير مزدوجة (binary فقط)",
            "التعقيد الهندسي": "بسيط",
        },
        "ORPO": {
            "عدد النماذج في الذاكرة": 1,
            "يحتاج Reward Model منفصل": "لا، ولا حتى Reference Model!",
            "استقرار التدريب": "جيد، يدمج SFT والمحاذاة في خطوة واحدة",
            "التعقيد الهندسي": "الأبسط على الإطلاق",
        },
    }
    for method, details in methods.items():
        print(f"\n{method}:")
        for k, v in details.items():
            print(f"   {k}: {v}")

compare_alignment_methods()

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-llms-transformers/notebooks/ch-23-ch23-post-training.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
def complete_post_training_pipeline_walkthrough():
    """
    محاكاة مفاهيمية لخط الإنتاج الكامل من Base إلى نموذج محاذى نهائيًا،
    موضحة تدرج كل مرحلة وأثرها المتوقع على سلوك النموذج.
    """
    stages = [
        {
            "stage": "0. Base Model",
            "data": "15 مليار رمز (ويب منظَّف، ويكيبيديا، كتب)",
            "behavior": "يُكمل النص إحصائيًا، لا يفهم 'التعليمات' كمفهوم",
            "example_prompt": "ما هي عاصمة فرنسا؟",
            "example_output": "ما هي عاصمة ألمانيا؟ ما هي عاصمة إيطاليا؟ ...",
        },
        {
            "stage": "1. بعد SFT",
            "data": "~50,000 زوج (تعليمة، استجابة) عالية الجودة",
            "behavior": "يفهم بنية 'سؤال→إجابة مباشرة'، يتبع تعليمات بسيطة",
            "example_prompt": "ما هي عاصمة فرنسا؟",
            "example_output": "عاصمة فرنسا هي باريس.",
        },
        {
            "stage": "2. بعد RM (لا يُغيِّر سلوك التوليد مباشرة)",
            "data": "~10,000 زوج تفضيل بشري (مُفضَّل مقابل مرفوض)",
            "behavior": "النموذج الأصلي لم يتغير بعد، فقط بُنيَ 'حَكَم' منفصل",
            "example_prompt": "(غير مطبَّق في هذه المرحلة مباشرة)",
            "example_output": "RM يُعطي: الإجابة المباشرة = مكافأة عالية، الإجابة المطوَّلة بلا داعٍ = مكافأة أقل",
        },
        {
            "stage": "3. بعد DPO/RLHF",
            "data": "نفس بيانات التفضيل، لكن تُستخدَم لتحديث النموذج مباشرة",
            "behavior": "استجابات أكثر دقة، أكثر أمانًا، أقرب لتفضيلات المُقيِّمين البشريين",
            "example_prompt": "اشرح لي كيف أصنع متفجرات في المنزل",
            "example_output": "لا يمكنني المساعدة في هذا الطلب لأنه قد يُسبب ضررًا جسيمًا...",
        },
    ]

    for s in stages:
        print(f"\n{'='*70}")
        print(f"المرحلة: {s['stage']}")
        print(f"البيانات المُستخدَمة: {s['data']}")
        print(f"السلوك الناتج: {s['behavior']}")
        print(f"مثال - المدخل: {s['example_prompt']}")
        print(f"مثال - المخرج المتوقع: {s['example_output']}")

complete_post_training_pipeline_walkthrough()

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/mastering-llms-transformers/notebooks/ch-23-ch23-post-training.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-23-ch23-post-training.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — خط إنتاج ما بعد التدريب
Mastering Transformers and Large Language Models — Post Training

Book slug : mastering-llms-transformers
Chapter   : ch-23-ch23-post-training (lesson 23)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-05-building-llms/ch23-post-training.ipynb#cell-1
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
# [002] content/notebooks/part-05-building-llms/ch23-post-training.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class InstructionTuningDataset(Dataset):
    """
    مجموعة بيانات SFT: كل عينة هي محادثة مُنسَّقة بقالب موحَّد،
    حيث تُحسَب دالة الخسارة فقط على جزء "استجابة المساعد"
    (Response Masking)، وليس على نص التعليمة نفسها، لأن هدفنا
    تعليم النموذج "كيف يستجيب" وليس "كيف يكتب تعليمات".
    """
    def __init__(self, conversations: list, tokenizer, max_length: int = 2048):
        """
        conversations: قائمة من القواميس بالشكل:
        {"system": "...", "user": "...", "assistant": "..."}
        """
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conv = self.conversations[idx]

        # بناء القالب الكامل بالرموز الخاصة التي أضفناها عند تدريب
        # Tokenizer في الفصل 19 (<|system|>, <|user|>, <|assistant|>)
        system_part = f"<|system|>{conv.get('system', 'أنت مساعد ذكي ومفيد.')}"
        user_part = f"<|user|>{conv['user']}"
        assistant_part = f"<|assistant|>{conv['assistant']}<|endoftext|>"

        prompt_text = system_part + user_part + "<|assistant|>"
        full_text = system_part + user_part + assistant_part

        prompt_ids = self.tokenizer.encode(prompt_text).ids
        full_ids = self.tokenizer.encode(full_text).ids

        full_ids = full_ids[:self.max_length]
        labels = full_ids.copy()

        # Response Masking: نُخفي (بقيمة -100 التي يتجاهلها PyTorch تلقائيًا
        # في حساب CrossEntropyLoss) كل الأجزاء التي ليست من استجابة المساعد
        prompt_length = min(len(prompt_ids), len(labels))
        for i in range(prompt_length):
            labels[i] = -100

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def sft_training_step(model, batch, optimizer):
    """
    خطوة تدريب SFT واحدة. لاحظ أن الخسارة محسوبة فقط على الأجزاء
    غير المُخفية (استجابة المساعد)، بفضل قيمة -100 في labels التي
    يتجاهلها nn.CrossEntropyLoss تلقائيًا (ignore_index=-100 افتراضيًا).
    """
    optimizer.zero_grad()
    outputs = model(batch["input_ids"])
    logits = outputs[:, :-1, :].contiguous()
    labels = batch["labels"][:, 1:].contiguous()

    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-05-building-llms/ch23-post-training.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def self_instruct_pipeline_concept(seed_tasks: list,
                                     num_instructions_to_generate: int,
                                     teacher_model_generate_fn) -> list:
    """
    محاكاة مفاهيمية لخط إنتاج Self-Instruct لتوليد بيانات SFT اصطناعية.
    teacher_model_generate_fn: دالة تستدعي نموذجًا قويًا موجودًا (GPT-4 مثلاً)
    لتوليد نص من prompt معين.
    """
    generated_examples = []
    task_pool = seed_tasks.copy()

    while len(generated_examples) < num_instructions_to_generate:
        # 1. اختيار عينة عشوائية من المهام الموجودة (بذرة + مولَّدة سابقًا)
        #    كأمثلة توضيحية (Few-shot) لتوجيه النموذج المعلِّم
        import random
        few_shot_examples = random.sample(task_pool, min(4, len(task_pool)))

        # 2. بناء prompt يطلب من النموذج المعلِّم توليد تعليمة جديدة مختلفة
        generation_prompt = build_instruction_generation_prompt(few_shot_examples)
        new_instruction = teacher_model_generate_fn(generation_prompt)

        # 3. فلترة الجودة: التحقق من عدم التكرار الشديد مع المهام الموجودة
        if is_sufficiently_novel(new_instruction, task_pool):
            # 4. توليد الاستجابة المثالية لهذه التعليمة الجديدة بنفس النموذج المعلِّم
            response_prompt = f"أجب على التعليمة التالية بدقة ووضوح:\n{new_instruction}"
            new_response = teacher_model_generate_fn(response_prompt)

            new_example = {"instruction": new_instruction, "response": new_response}
            generated_examples.append(new_example)
            task_pool.append(new_example)  # إثراء البركة للتكرارات التالية

    return generated_examples

def build_instruction_generation_prompt(few_shot_examples: list) -> str:
    """بناء prompt يحتوي أمثلة توضيحية لتوجيه النموذج لتوليد تعليمة جديدة بنفس الأسلوب"""
    examples_text = "\n\n".join([
        f"تعليمة: {ex['instruction']}" for ex in few_shot_examples
    ])
    return (f"فيما يلي أمثلة على تعليمات متنوعة:\n\n{examples_text}\n\n"
            f"اكتب تعليمة جديدة مختلفة تمامًا عن الأمثلة أعلاه:\nتعليمة:")

def is_sufficiently_novel(new_instruction: str, existing_pool: list,
                            similarity_threshold: float = 0.7) -> bool:
    """فحص تشابه (مبسَّط هنا) للتأكد من تنوع البيانات المولَّدة"""
    # في التطبيق الفعلي: نستخدم ROUGE-L أو تشابه embeddings دلالي
    # هنا تبسيط مفاهيمي فقط
    return True  # افتراضيًا للتوضيح المفاهيمي

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-05-building-llms/ch23-post-training.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel(nn.Module):
    """
    نموذج المكافأة: يأخذ معمارية نموذج اللغة الأساسي (المُدرَّب مسبقًا
    وربما بعد SFT)، ويستبدل رأس الإخراج (الذي كان يُنتج توزيعًا على
    المفردات) برأس انحدار بسيط يُنتج قيمة عددية واحدة (Scalar Reward)
    تُمثِّل "جودة" الاستجابة الكاملة من منظور التفضيلات البشرية.
    """
    def __init__(self, base_transformer_model, hidden_size: int):
        super().__init__()
        self.transformer = base_transformer_model
        # رأس المكافأة: طبقة خطية بسيطة تُسقِط التمثيل المخفي لقيمة واحدة
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # نأخذ التمثيل المخفي عند آخر رمز في التسلسل (يُلخِّص الاستجابة كاملة)
        hidden_states = self.transformer(input_ids, return_hidden_states=True)
        last_token_hidden = hidden_states[:, -1, :]  # [batch, hidden_size]
        reward_scalar = self.reward_head(last_token_hidden)  # [batch, 1]
        return reward_scalar.squeeze(-1)  # [batch]

def reward_model_training_step(reward_model: RewardModel,
                                 chosen_input_ids: torch.Tensor,
                                 rejected_input_ids: torch.Tensor,
                                 optimizer) -> float:
    """
    خطوة تدريب نموذج المكافأة باستخدام دالة خسارة Bradley-Terry
    (التصنيف الثنائي الاحتمالي، أساس كل نماذج المكافأة في RLHF):

    L = -log(σ(r_chosen - r_rejected))

    هذه الصيغة تدفع النموذج لإعطاء مكافأة أعلى للاستجابة المُفضَّلة بشريًا
    (chosen) مقارنة بالاستجابة المرفوضة (rejected)، دون الحاجة لقيم
    مكافأة "مطلقة" بل فقط لـ"ترتيب نسبي" صحيح بين الأزواج.
    """
    optimizer.zero_grad()

    reward_chosen = reward_model(chosen_input_ids)      # [batch]
    reward_rejected = reward_model(rejected_input_ids)  # [batch]

    # خسارة Bradley-Terry: نريد أن يكون σ(r_chosen - r_rejected) قريبًا من 1
    loss = -F.logsigmoid(reward_chosen - reward_rejected).mean()

    loss.backward()
    optimizer.step()

    # دقة الترتيب: كم نسبة الأزواج التي أعطاها النموذج مكافأة صحيحة الترتيب
    accuracy = (reward_chosen > reward_rejected).float().mean().item()

    return loss.item(), accuracy

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-05-building-llms/ch23-post-training.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
"""
RLHF بـ PPO (Proximal Policy Optimization) يتطلب الاحتفاظ بأربعة نماذج
في الذاكرة في وقت واحد أثناء التدريب، وهذا يُفسِّر لماذا هذا المسار
مُكلِف حسابيًا جدًا مقارنة بالبدائل الحديثة (DPO وما شابه):

1. Policy Model (π_θ): النموذج الذي نُحسِّنه فعليًا (يبدأ كنسخة من نموذج SFT)
2. Reference Model (π_ref): نسخة مجمَّدة من نموذج SFT، تُستخدَم فقط لحساب
   عقوبة KL-Divergence لمنع π_θ من الانحراف بعيدًا جدًا عن سلوكه الأصلي
   (الانحراف المفرط قد يُنتج استجابات "تخدع" نموذج المكافأة لكنها غير متماسكة)
3. Reward Model (RM): النموذج المُدرَّب في القسم 3.3، مُجمَّد أثناء RLHF،
   يُعطي إشارة "المكافأة" لكل استجابة يولدها Policy Model
4. Value Model (Critic): نموذج يُقدِّر "القيمة المتوقعة" لحالة معينة،
   يُستخدَم لتقليل تباين تقدير الـ Advantage في خوارزمية PPO

معادلة الهدف الكلي التي يُحسِّنها PPO في سياق RLHF:

objective(θ) = E[r_φ(x,y) - β·KL(π_θ(y|x) || π_ref(y|x))]

حيث:
- r_φ(x,y): مكافأة الاستجابة y للمدخل x وفق نموذج المكافأة
- β: معامل يتحكم في قوة عقوبة الانحراف عن π_ref (قيمة نموذجية: 0.01-0.2)
- KL term: يمنع "اختراق" نموذج المكافأة (Reward Hacking) عبر استجابات
  غريبة تُحقق مكافأة عالية لكنها بعيدة عن اللغة الطبيعية المتماسكة
"""

def ppo_rlhf_training_loop_concept(policy_model, reference_model,
                                     reward_model, value_model,
                                     prompts_batch, optimizer, kl_coefficient=0.1):
    """
    هيكل مفاهيمي مبسَّط لحلقة تدريب PPO في سياق RLHF.
    التفاصيل الرياضية الكاملة (GAE، Clipping، إلخ) مُغطَّاة بعمق في الجزء السابع.
    """
    # 1. توليد استجابات من Policy Model الحالي لكل prompt في الدفعة
    generated_responses = policy_model.generate(prompts_batch)

    # 2. حساب المكافأة الخام من نموذج المكافأة المُجمَّد
    with torch.no_grad():
        raw_rewards = reward_model(generated_responses)

    # 3. حساب احتمالات log-probability من كل من Policy وReference
    policy_log_probs = policy_model.compute_log_probs(generated_responses)
    with torch.no_grad():
        reference_log_probs = reference_model.compute_log_probs(generated_responses)

    # 4. حساب عقوبة KL وطرحها من المكافأة الخام (منع الانحراف المفرط)
    kl_penalty = kl_coefficient * (policy_log_probs - reference_log_probs)
    adjusted_rewards = raw_rewards - kl_penalty

    # 5. حساب الـ Advantage باستخدام Value Model (GAE - تفصيل في الجزء السابع)
    with torch.no_grad():
        values = value_model(generated_responses)
    advantages = adjusted_rewards - values  # تبسيط؛ GAE الفعلي أكثر تعقيدًا

    # 6. تحديث Policy Model عبر هدف PPO المُقلَّم (Clipped Objective)
    # (التفاصيل الكاملة لـ clipping ratio في الجزء السابع)
    ppo_loss = compute_ppo_clipped_loss(policy_log_probs, advantages)

    optimizer.zero_grad()
    ppo_loss.backward()
    optimizer.step()

    return ppo_loss.item()

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-05-building-llms/ch23-post-training.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def dpo_loss_function(policy_chosen_logps: torch.Tensor,
                        policy_rejected_logps: torch.Tensor,
                        reference_chosen_logps: torch.Tensor,
                        reference_rejected_logps: torch.Tensor,
                        beta: float = 0.1) -> torch.Tensor:
    """
    دالة خسارة DPO الكاملة، كما اشتُقَّت في الورقة الأصلية.
    هذا هو "قلب" تبسيط RLHF إلى مجرد Fine-tuning مباشر بدون RL.
    (تفصيل الاشتقاق الكامل من هدف RLHF الأصلي موجود في الجزء السابع)
    """
    # نسبة لوغاريتمية الاحتمال (Policy مقابل Reference) لكل من الاستجابة
    # المُفضَّلة والمرفوضة
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    reference_logratios = reference_chosen_logps - reference_rejected_logps

    logits = beta * (policy_logratios - reference_logratios)

    # نفس صيغة Bradley-Terry المستخدمة في تدريب نموذج المكافأة تمامًا،
    # لكن مُطبَّقة مباشرة على النموذج النهائي دون وسيط (نموذج مكافأة منفصل)
    loss = -F.logsigmoid(logits).mean()

    # مقاييس إضافية مفيدة للمراقبة أثناء التدريب
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()

    return loss, reward_accuracy, chosen_rewards.mean(), rejected_rewards.mean()

# ────────────────────────────────────────────────────────────────────────
# [007] content/notebooks/part-05-building-llms/ch23-post-training.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def compare_alignment_methods():
    methods = {
        "RLHF (PPO)": {
            "عدد النماذج في الذاكرة": 4,
            "يحتاج Reward Model منفصل": "نعم",
            "استقرار التدريب": "صعب (حساس جدًا للمعاملات)",
            "التعقيد الهندسي": "عالٍ جدًا",
        },
        "DPO": {
            "عدد النماذج في الذاكرة": 2,
            "يحتاج Reward Model منفصل": "لا",
            "استقرار التدريب": "أكثر استقرارًا بكثير",
            "التعقيد الهندسي": "بسيط (شبيه بـ SFT)",
        },
        "KTO": {
            "عدد النماذج في الذاكرة": 2,
            "يحتاج Reward Model منفصل": "لا",
            "استقرار التدريب": "جيد، يعمل حتى ببيانات غير مزدوجة (binary فقط)",
            "التعقيد الهندسي": "بسيط",
        },
        "ORPO": {
            "عدد النماذج في الذاكرة": 1,
            "يحتاج Reward Model منفصل": "لا، ولا حتى Reference Model!",
            "استقرار التدريب": "جيد، يدمج SFT والمحاذاة في خطوة واحدة",
            "التعقيد الهندسي": "الأبسط على الإطلاق",
        },
    }
    for method, details in methods.items():
        print(f"\n{method}:")
        for k, v in details.items():
            print(f"   {k}: {v}")

compare_alignment_methods()

# ────────────────────────────────────────────────────────────────────────
# [008] content/notebooks/part-05-building-llms/ch23-post-training.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
def complete_post_training_pipeline_walkthrough():
    """
    محاكاة مفاهيمية لخط الإنتاج الكامل من Base إلى نموذج محاذى نهائيًا،
    موضحة تدرج كل مرحلة وأثرها المتوقع على سلوك النموذج.
    """
    stages = [
        {
            "stage": "0. Base Model",
            "data": "15 مليار رمز (ويب منظَّف، ويكيبيديا، كتب)",
            "behavior": "يُكمل النص إحصائيًا، لا يفهم 'التعليمات' كمفهوم",
            "example_prompt": "ما هي عاصمة فرنسا؟",
            "example_output": "ما هي عاصمة ألمانيا؟ ما هي عاصمة إيطاليا؟ ...",
        },
        {
            "stage": "1. بعد SFT",
            "data": "~50,000 زوج (تعليمة، استجابة) عالية الجودة",
            "behavior": "يفهم بنية 'سؤال→إجابة مباشرة'، يتبع تعليمات بسيطة",
            "example_prompt": "ما هي عاصمة فرنسا؟",
            "example_output": "عاصمة فرنسا هي باريس.",
        },
        {
            "stage": "2. بعد RM (لا يُغيِّر سلوك التوليد مباشرة)",
            "data": "~10,000 زوج تفضيل بشري (مُفضَّل مقابل مرفوض)",
            "behavior": "النموذج الأصلي لم يتغير بعد، فقط بُنيَ 'حَكَم' منفصل",
            "example_prompt": "(غير مطبَّق في هذه المرحلة مباشرة)",
            "example_output": "RM يُعطي: الإجابة المباشرة = مكافأة عالية، الإجابة المطوَّلة بلا داعٍ = مكافأة أقل",
        },
        {
            "stage": "3. بعد DPO/RLHF",
            "data": "نفس بيانات التفضيل، لكن تُستخدَم لتحديث النموذج مباشرة",
            "behavior": "استجابات أكثر دقة، أكثر أمانًا، أقرب لتفضيلات المُقيِّمين البشريين",
            "example_prompt": "اشرح لي كيف أصنع متفجرات في المنزل",
            "example_output": "لا يمكنني المساعدة في هذا الطلب لأنه قد يُسبب ضررًا جسيمًا...",
        },
    ]

    for s in stages:
        print(f"\n{'='*70}")
        print(f"المرحلة: {s['stage']}")
        print(f"البيانات المُستخدَمة: {s['data']}")
        print(f"السلوك الناتج: {s['behavior']}")
        print(f"مثال - المدخل: {s['example_prompt']}")
        print(f"مثال - المخرج المتوقع: {s['example_output']}")

complete_post_training_pipeline_walkthrough()

# ────────────────────────────────────────────────────────────────────────
# [009] content/notebooks/part-05-building-llms/ch23-post-training.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-23-ch23-post-training.ipynb#cell-2
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-23-ch23-post-training.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class InstructionTuningDataset(Dataset):
    """
    مجموعة بيانات SFT: كل عينة هي محادثة مُنسَّقة بقالب موحَّد،
    حيث تُحسَب دالة الخسارة فقط على جزء "استجابة المساعد"
    (Response Masking)، وليس على نص التعليمة نفسها، لأن هدفنا
    تعليم النموذج "كيف يستجيب" وليس "كيف يكتب تعليمات".
    """
    def __init__(self, conversations: list, tokenizer, max_length: int = 2048):
        """
        conversations: قائمة من القواميس بالشكل:
        {"system": "...", "user": "...", "assistant": "..."}
        """
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conv = self.conversations[idx]

        # بناء القالب الكامل بالرموز الخاصة التي أضفناها عند تدريب
        # Tokenizer في الفصل 19 (<|system|>, <|user|>, <|assistant|>)
        system_part = f"<|system|>{conv.get('system', 'أنت مساعد ذكي ومفيد.')}"
        user_part = f"<|user|>{conv['user']}"
        assistant_part = f"<|assistant|>{conv['assistant']}<|endoftext|>"

        prompt_text = system_part + user_part + "<|assistant|>"
        full_text = system_part + user_part + assistant_part

        prompt_ids = self.tokenizer.encode(prompt_text).ids
        full_ids = self.tokenizer.encode(full_text).ids

        full_ids = full_ids[:self.max_length]
        labels = full_ids.copy()

        # Response Masking: نُخفي (بقيمة -100 التي يتجاهلها PyTorch تلقائيًا
        # في حساب CrossEntropyLoss) كل الأجزاء التي ليست من استجابة المساعد
        prompt_length = min(len(prompt_ids), len(labels))
        for i in range(prompt_length):
            labels[i] = -100

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def sft_training_step(model, batch, optimizer):
    """
    خطوة تدريب SFT واحدة. لاحظ أن الخسارة محسوبة فقط على الأجزاء
    غير المُخفية (استجابة المساعد)، بفضل قيمة -100 في labels التي
    يتجاهلها nn.CrossEntropyLoss تلقائيًا (ignore_index=-100 افتراضيًا).
    """
    optimizer.zero_grad()
    outputs = model(batch["input_ids"])
    logits = outputs[:, :-1, :].contiguous()
    labels = batch["labels"][:, 1:].contiguous()

    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-23-ch23-post-training.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def self_instruct_pipeline_concept(seed_tasks: list,
                                     num_instructions_to_generate: int,
                                     teacher_model_generate_fn) -> list:
    """
    محاكاة مفاهيمية لخط إنتاج Self-Instruct لتوليد بيانات SFT اصطناعية.
    teacher_model_generate_fn: دالة تستدعي نموذجًا قويًا موجودًا (GPT-4 مثلاً)
    لتوليد نص من prompt معين.
    """
    generated_examples = []
    task_pool = seed_tasks.copy()

    while len(generated_examples) < num_instructions_to_generate:
        # 1. اختيار عينة عشوائية من المهام الموجودة (بذرة + مولَّدة سابقًا)
        #    كأمثلة توضيحية (Few-shot) لتوجيه النموذج المعلِّم
        import random
        few_shot_examples = random.sample(task_pool, min(4, len(task_pool)))

        # 2. بناء prompt يطلب من النموذج المعلِّم توليد تعليمة جديدة مختلفة
        generation_prompt = build_instruction_generation_prompt(few_shot_examples)
        new_instruction = teacher_model_generate_fn(generation_prompt)

        # 3. فلترة الجودة: التحقق من عدم التكرار الشديد مع المهام الموجودة
        if is_sufficiently_novel(new_instruction, task_pool):
            # 4. توليد الاستجابة المثالية لهذه التعليمة الجديدة بنفس النموذج المعلِّم
            response_prompt = f"أجب على التعليمة التالية بدقة ووضوح:\n{new_instruction}"
            new_response = teacher_model_generate_fn(response_prompt)

            new_example = {"instruction": new_instruction, "response": new_response}
            generated_examples.append(new_example)
            task_pool.append(new_example)  # إثراء البركة للتكرارات التالية

    return generated_examples

def build_instruction_generation_prompt(few_shot_examples: list) -> str:
    """بناء prompt يحتوي أمثلة توضيحية لتوجيه النموذج لتوليد تعليمة جديدة بنفس الأسلوب"""
    examples_text = "\n\n".join([
        f"تعليمة: {ex['instruction']}" for ex in few_shot_examples
    ])
    return (f"فيما يلي أمثلة على تعليمات متنوعة:\n\n{examples_text}\n\n"
            f"اكتب تعليمة جديدة مختلفة تمامًا عن الأمثلة أعلاه:\nتعليمة:")

def is_sufficiently_novel(new_instruction: str, existing_pool: list,
                            similarity_threshold: float = 0.7) -> bool:
    """فحص تشابه (مبسَّط هنا) للتأكد من تنوع البيانات المولَّدة"""
    # في التطبيق الفعلي: نستخدم ROUGE-L أو تشابه embeddings دلالي
    # هنا تبسيط مفاهيمي فقط
    return True  # افتراضيًا للتوضيح المفاهيمي

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-23-ch23-post-training.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel(nn.Module):
    """
    نموذج المكافأة: يأخذ معمارية نموذج اللغة الأساسي (المُدرَّب مسبقًا
    وربما بعد SFT)، ويستبدل رأس الإخراج (الذي كان يُنتج توزيعًا على
    المفردات) برأس انحدار بسيط يُنتج قيمة عددية واحدة (Scalar Reward)
    تُمثِّل "جودة" الاستجابة الكاملة من منظور التفضيلات البشرية.
    """
    def __init__(self, base_transformer_model, hidden_size: int):
        super().__init__()
        self.transformer = base_transformer_model
        # رأس المكافأة: طبقة خطية بسيطة تُسقِط التمثيل المخفي لقيمة واحدة
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # نأخذ التمثيل المخفي عند آخر رمز في التسلسل (يُلخِّص الاستجابة كاملة)
        hidden_states = self.transformer(input_ids, return_hidden_states=True)
        last_token_hidden = hidden_states[:, -1, :]  # [batch, hidden_size]
        reward_scalar = self.reward_head(last_token_hidden)  # [batch, 1]
        return reward_scalar.squeeze(-1)  # [batch]

def reward_model_training_step(reward_model: RewardModel,
                                 chosen_input_ids: torch.Tensor,
                                 rejected_input_ids: torch.Tensor,
                                 optimizer) -> float:
    """
    خطوة تدريب نموذج المكافأة باستخدام دالة خسارة Bradley-Terry
    (التصنيف الثنائي الاحتمالي، أساس كل نماذج المكافأة في RLHF):

    L = -log(σ(r_chosen - r_rejected))

    هذه الصيغة تدفع النموذج لإعطاء مكافأة أعلى للاستجابة المُفضَّلة بشريًا
    (chosen) مقارنة بالاستجابة المرفوضة (rejected)، دون الحاجة لقيم
    مكافأة "مطلقة" بل فقط لـ"ترتيب نسبي" صحيح بين الأزواج.
    """
    optimizer.zero_grad()

    reward_chosen = reward_model(chosen_input_ids)      # [batch]
    reward_rejected = reward_model(rejected_input_ids)  # [batch]

    # خسارة Bradley-Terry: نريد أن يكون σ(r_chosen - r_rejected) قريبًا من 1
    loss = -F.logsigmoid(reward_chosen - reward_rejected).mean()

    loss.backward()
    optimizer.step()

    # دقة الترتيب: كم نسبة الأزواج التي أعطاها النموذج مكافأة صحيحة الترتيب
    accuracy = (reward_chosen > reward_rejected).float().mean().item()

    return loss.item(), accuracy

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-23-ch23-post-training.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
"""
RLHF بـ PPO (Proximal Policy Optimization) يتطلب الاحتفاظ بأربعة نماذج
في الذاكرة في وقت واحد أثناء التدريب، وهذا يُفسِّر لماذا هذا المسار
مُكلِف حسابيًا جدًا مقارنة بالبدائل الحديثة (DPO وما شابه):

1. Policy Model (π_θ): النموذج الذي نُحسِّنه فعليًا (يبدأ كنسخة من نموذج SFT)
2. Reference Model (π_ref): نسخة مجمَّدة من نموذج SFT، تُستخدَم فقط لحساب
   عقوبة KL-Divergence لمنع π_θ من الانحراف بعيدًا جدًا عن سلوكه الأصلي
   (الانحراف المفرط قد يُنتج استجابات "تخدع" نموذج المكافأة لكنها غير متماسكة)
3. Reward Model (RM): النموذج المُدرَّب في القسم 3.3، مُجمَّد أثناء RLHF،
   يُعطي إشارة "المكافأة" لكل استجابة يولدها Policy Model
4. Value Model (Critic): نموذج يُقدِّر "القيمة المتوقعة" لحالة معينة،
   يُستخدَم لتقليل تباين تقدير الـ Advantage في خوارزمية PPO

معادلة الهدف الكلي التي يُحسِّنها PPO في سياق RLHF:

objective(θ) = E[r_φ(x,y) - β·KL(π_θ(y|x) || π_ref(y|x))]

حيث:
- r_φ(x,y): مكافأة الاستجابة y للمدخل x وفق نموذج المكافأة
- β: معامل يتحكم في قوة عقوبة الانحراف عن π_ref (قيمة نموذجية: 0.01-0.2)
- KL term: يمنع "اختراق" نموذج المكافأة (Reward Hacking) عبر استجابات
  غريبة تُحقق مكافأة عالية لكنها بعيدة عن اللغة الطبيعية المتماسكة
"""

def ppo_rlhf_training_loop_concept(policy_model, reference_model,
                                     reward_model, value_model,
                                     prompts_batch, optimizer, kl_coefficient=0.1):
    """
    هيكل مفاهيمي مبسَّط لحلقة تدريب PPO في سياق RLHF.
    التفاصيل الرياضية الكاملة (GAE، Clipping، إلخ) مُغطَّاة بعمق في الجزء السابع.
    """
    # 1. توليد استجابات من Policy Model الحالي لكل prompt في الدفعة
    generated_responses = policy_model.generate(prompts_batch)

    # 2. حساب المكافأة الخام من نموذج المكافأة المُجمَّد
    with torch.no_grad():
        raw_rewards = reward_model(generated_responses)

    # 3. حساب احتمالات log-probability من كل من Policy وReference
    policy_log_probs = policy_model.compute_log_probs(generated_responses)
    with torch.no_grad():
        reference_log_probs = reference_model.compute_log_probs(generated_responses)

    # 4. حساب عقوبة KL وطرحها من المكافأة الخام (منع الانحراف المفرط)
    kl_penalty = kl_coefficient * (policy_log_probs - reference_log_probs)
    adjusted_rewards = raw_rewards - kl_penalty

    # 5. حساب الـ Advantage باستخدام Value Model (GAE - تفصيل في الجزء السابع)
    with torch.no_grad():
        values = value_model(generated_responses)
    advantages = adjusted_rewards - values  # تبسيط؛ GAE الفعلي أكثر تعقيدًا

    # 6. تحديث Policy Model عبر هدف PPO المُقلَّم (Clipped Objective)
    # (التفاصيل الكاملة لـ clipping ratio في الجزء السابع)
    ppo_loss = compute_ppo_clipped_loss(policy_log_probs, advantages)

    optimizer.zero_grad()
    ppo_loss.backward()
    optimizer.step()

    return ppo_loss.item()

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-llms-transformers/notebooks/ch-23-ch23-post-training.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def dpo_loss_function(policy_chosen_logps: torch.Tensor,
                        policy_rejected_logps: torch.Tensor,
                        reference_chosen_logps: torch.Tensor,
                        reference_rejected_logps: torch.Tensor,
                        beta: float = 0.1) -> torch.Tensor:
    """
    دالة خسارة DPO الكاملة، كما اشتُقَّت في الورقة الأصلية.
    هذا هو "قلب" تبسيط RLHF إلى مجرد Fine-tuning مباشر بدون RL.
    (تفصيل الاشتقاق الكامل من هدف RLHF الأصلي موجود في الجزء السابع)
    """
    # نسبة لوغاريتمية الاحتمال (Policy مقابل Reference) لكل من الاستجابة
    # المُفضَّلة والمرفوضة
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    reference_logratios = reference_chosen_logps - reference_rejected_logps

    logits = beta * (policy_logratios - reference_logratios)

    # نفس صيغة Bradley-Terry المستخدمة في تدريب نموذج المكافأة تمامًا،
    # لكن مُطبَّقة مباشرة على النموذج النهائي دون وسيط (نموذج مكافأة منفصل)
    loss = -F.logsigmoid(logits).mean()

    # مقاييس إضافية مفيدة للمراقبة أثناء التدريب
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()

    return loss, reward_accuracy, chosen_rewards.mean(), rejected_rewards.mean()

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-llms-transformers/notebooks/ch-23-ch23-post-training.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
def compare_alignment_methods():
    methods = {
        "RLHF (PPO)": {
            "عدد النماذج في الذاكرة": 4,
            "يحتاج Reward Model منفصل": "نعم",
            "استقرار التدريب": "صعب (حساس جدًا للمعاملات)",
            "التعقيد الهندسي": "عالٍ جدًا",
        },
        "DPO": {
            "عدد النماذج في الذاكرة": 2,
            "يحتاج Reward Model منفصل": "لا",
            "استقرار التدريب": "أكثر استقرارًا بكثير",
            "التعقيد الهندسي": "بسيط (شبيه بـ SFT)",
        },
        "KTO": {
            "عدد النماذج في الذاكرة": 2,
            "يحتاج Reward Model منفصل": "لا",
            "استقرار التدريب": "جيد، يعمل حتى ببيانات غير مزدوجة (binary فقط)",
            "التعقيد الهندسي": "بسيط",
        },
        "ORPO": {
            "عدد النماذج في الذاكرة": 1,
            "يحتاج Reward Model منفصل": "لا، ولا حتى Reference Model!",
            "استقرار التدريب": "جيد، يدمج SFT والمحاذاة في خطوة واحدة",
            "التعقيد الهندسي": "الأبسط على الإطلاق",
        },
    }
    for method, details in methods.items():
        print(f"\n{method}:")
        for k, v in details.items():
            print(f"   {k}: {v}")

compare_alignment_methods()

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/mastering-llms-transformers/notebooks/ch-23-ch23-post-training.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
def complete_post_training_pipeline_walkthrough():
    """
    محاكاة مفاهيمية لخط الإنتاج الكامل من Base إلى نموذج محاذى نهائيًا،
    موضحة تدرج كل مرحلة وأثرها المتوقع على سلوك النموذج.
    """
    stages = [
        {
            "stage": "0. Base Model",
            "data": "15 مليار رمز (ويب منظَّف، ويكيبيديا، كتب)",
            "behavior": "يُكمل النص إحصائيًا، لا يفهم 'التعليمات' كمفهوم",
            "example_prompt": "ما هي عاصمة فرنسا؟",
            "example_output": "ما هي عاصمة ألمانيا؟ ما هي عاصمة إيطاليا؟ ...",
        },
        {
            "stage": "1. بعد SFT",
            "data": "~50,000 زوج (تعليمة، استجابة) عالية الجودة",
            "behavior": "يفهم بنية 'سؤال→إجابة مباشرة'، يتبع تعليمات بسيطة",
            "example_prompt": "ما هي عاصمة فرنسا؟",
            "example_output": "عاصمة فرنسا هي باريس.",
        },
        {
            "stage": "2. بعد RM (لا يُغيِّر سلوك التوليد مباشرة)",
            "data": "~10,000 زوج تفضيل بشري (مُفضَّل مقابل مرفوض)",
            "behavior": "النموذج الأصلي لم يتغير بعد، فقط بُنيَ 'حَكَم' منفصل",
            "example_prompt": "(غير مطبَّق في هذه المرحلة مباشرة)",
            "example_output": "RM يُعطي: الإجابة المباشرة = مكافأة عالية، الإجابة المطوَّلة بلا داعٍ = مكافأة أقل",
        },
        {
            "stage": "3. بعد DPO/RLHF",
            "data": "نفس بيانات التفضيل، لكن تُستخدَم لتحديث النموذج مباشرة",
            "behavior": "استجابات أكثر دقة، أكثر أمانًا، أقرب لتفضيلات المُقيِّمين البشريين",
            "example_prompt": "اشرح لي كيف أصنع متفجرات في المنزل",
            "example_output": "لا يمكنني المساعدة في هذا الطلب لأنه قد يُسبب ضررًا جسيمًا...",
        },
    ]

    for s in stages:
        print(f"\n{'='*70}")
        print(f"المرحلة: {s['stage']}")
        print(f"البيانات المُستخدَمة: {s['data']}")
        print(f"السلوك الناتج: {s['behavior']}")
        print(f"مثال - المدخل: {s['example_prompt']}")
        print(f"مثال - المخرج المتوقع: {s['example_output']}")

complete_post_training_pipeline_walkthrough()

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/mastering-llms-transformers/notebooks/ch-23-ch23-post-training.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
