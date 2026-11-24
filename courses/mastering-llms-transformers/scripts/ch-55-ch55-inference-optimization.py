#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — تحسين الاستدلال المتقدم
Mastering Transformers and Large Language Models — Inference Optimization

Book slug : mastering-llms-transformers
Chapter   : ch-55-ch55-inference-optimization (lesson 55)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-55-ch55-inference-optimization.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — تحسين الاستدلال المتقدم
Mastering Transformers and Large Language Models — Inference Optimization

Book slug : mastering-llms-transformers
Chapter   : ch-55-ch55-inference-optimization (lesson 55)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-55-ch55-inference-optimization.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — تحسين الاستدلال المتقدم
Mastering Transformers and Large Language Models — Inference Optimization

Book slug : mastering-llms-transformers
Chapter   : ch-55-ch55-inference-optimization (lesson 55)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-11-llmops/ch55-inference-optimization.ipynb#cell-1
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
# [002] content/notebooks/part-11-llmops/ch55-inference-optimization.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
class PagedKVCacheConceptual:
    """
    توضيح مفاهيمي لفكرة PagedAttention:
    بدلاً من حجز ذاكرة متجاورة ضخمة لكل تسلسل، نُقسّم KV Cache
    إلى "صفحات" (blocks) ثابتة الحجم، تُخصَّص ديناميكيًا عند الحاجة فقط
    (تمامًا مثل Paging في أنظمة التشغيل)
    """
    def __init__(self, block_size=16, total_blocks=1000, head_dim=128, n_heads=8):
        self.block_size = block_size
        self.physical_blocks = torch.zeros(total_blocks, block_size, n_heads, head_dim * 2)
        self.free_blocks = list(range(total_blocks))
        self.block_tables = {}   # {sequence_id: [block_ids]} — خريطة منطقي -> فعلي

    def allocate_for_sequence(self, seq_id: str, num_tokens: int):
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        allocated = [self.free_blocks.pop() for _ in range(num_blocks_needed)]
        self.block_tables[seq_id] = allocated
        return allocated

    def free_sequence(self, seq_id: str):
        """تحرير فوري للذاكرة بمجرد انتهاء التسلسل — يُعاد استخدامها لطلبات جديدة فورًا"""
        self.free_blocks.extend(self.block_tables.pop(seq_id))

    def share_blocks_for_prefix(self, seq_id_a: str, seq_id_b: str, shared_prefix_len: int):
        """
        ميزة إضافية قوية: مشاركة الصفحات الفعلية لنفس البادئة
        (مثل system prompt مشترك) بين عدة طلبات دون نسخ! (Copy-on-Write)
        """
        shared_blocks_count = shared_prefix_len // self.block_size
        self.block_tables[seq_id_b][:shared_blocks_count] = \
            self.block_tables[seq_id_a][:shared_blocks_count]

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-11-llmops/ch55-inference-optimization.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class ContinuousBatchingScheduler:
    """
    توضيح مفاهيمي: بمجرد انتهاء أي تسلسل في الدفعة،
    يُستبدل فورًا بطلب جديد من قائمة الانتظار، دون انتظار بقية الدفعة
    """
    def __init__(self, max_batch_size=32):
        self.max_batch_size = max_batch_size
        self.active_sequences = {}   # {seq_id: state}
        self.waiting_queue = []

    def step(self, model):
        """خطوة واحدة من التوليد عبر كل التسلسلات النشطة معًا"""
        # ملء الدفعة من قائمة الانتظار إن وُجدت مساحة فارغة
        while len(self.active_sequences) < self.max_batch_size and self.waiting_queue:
            new_seq = self.waiting_queue.pop(0)
            self.active_sequences[new_seq["id"]] = new_seq

        if not self.active_sequences:
            return

        # تمريرة أمامية واحدة لكل التسلسلات النشطة دفعة واحدة (batched forward pass)
        batch_input = self._prepare_batch(self.active_sequences)
        next_tokens = model.forward_step(batch_input)   # token واحد جديد لكل تسلسل

        # تحديث الحالة وإزالة أي تسلسل اكتمل فورًا (تحرير مكانه للتالي في الطابور)
        for seq_id, token in next_tokens.items():
            self.active_sequences[seq_id]["tokens"].append(token)
            if token == EOS_TOKEN_ID or len(self.active_sequences[seq_id]["tokens"]) >= \
               self.active_sequences[seq_id]["max_tokens"]:
                completed = self.active_sequences.pop(seq_id)
                self._return_result(completed)

    def _prepare_batch(self, sequences): ...  # تفاصيل التجميع
    def _return_result(self, completed_seq): ...  # إرجاع النتيجة للمستخدم

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-11-llmops/ch55-inference-optimization.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def speculative_decoding_step(draft_model, target_model, tokenizer, context_ids,
                                num_speculative_tokens=4):
    """
    1. النموذج الصغير (draft) يُولّد K رموز مقترحة بسرعة (تسلسليًا لكنه سريع جدًا)
    2. النموذج الكبير (target) يتحقق من كل الرموز K دفعة واحدة (تمريرة أمامية واحدة فقط!)
    3. نقبل الرموز المتطابقة مع توزيع target، ونرفض ونُصحّح من أول اختلاف
    """
    # الخطوة 1: اقتراح سريع من النموذج الصغير
    draft_tokens = []
    draft_probs = []
    current_ids = context_ids.clone()

    for _ in range(num_speculative_tokens):
        logits = draft_model(current_ids).logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        draft_tokens.append(next_token)
        draft_probs.append(probs.gather(-1, next_token))
        current_ids = torch.cat([current_ids, next_token], dim=1)

    # الخطوة 2: تحقق دفعة واحدة من النموذج الكبير (تمريرة أمامية واحدة لكل الرموز المقترحة)
    full_sequence = current_ids
    target_logits = target_model(full_sequence).logits
    target_probs_all = F.softmax(target_logits, dim=-1)

    # الخطوة 3: قبول/رفض كل رمز مقترح (خوارزمية Rejection Sampling)
    accepted_tokens = []
    context_len = context_ids.shape[1]

    for i, (draft_token, draft_prob) in enumerate(zip(draft_tokens, draft_probs)):
        position = context_len + i - 1
        target_prob = target_probs_all[:, position, :].gather(-1, draft_token)

        acceptance_ratio = torch.min(torch.ones_like(target_prob), target_prob / draft_prob)
        if torch.rand_like(acceptance_ratio) < acceptance_ratio:
            accepted_tokens.append(draft_token)
        else:
            # رفض: نستبدل برمز جديد من توزيع "الفرق" المُصحَّح إحصائيًا
            corrected_dist = torch.clamp(
                target_probs_all[:, position, :] - draft_probs[i], min=0
            )
            corrected_dist = corrected_dist / corrected_dist.sum()
            correction_token = torch.multinomial(corrected_dist, num_samples=1)
            accepted_tokens.append(correction_token)
            break   # نتوقف عند أول رفض

    return torch.cat(accepted_tokens, dim=1) if accepted_tokens else draft_tokens[0]

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-11-llmops/ch55-inference-optimization.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-55-ch55-inference-optimization.ipynb#cell-1
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-55-ch55-inference-optimization.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
class PagedKVCacheConceptual:
    """
    توضيح مفاهيمي لفكرة PagedAttention:
    بدلاً من حجز ذاكرة متجاورة ضخمة لكل تسلسل، نُقسّم KV Cache
    إلى "صفحات" (blocks) ثابتة الحجم، تُخصَّص ديناميكيًا عند الحاجة فقط
    (تمامًا مثل Paging في أنظمة التشغيل)
    """
    def __init__(self, block_size=16, total_blocks=1000, head_dim=128, n_heads=8):
        self.block_size = block_size
        self.physical_blocks = torch.zeros(total_blocks, block_size, n_heads, head_dim * 2)
        self.free_blocks = list(range(total_blocks))
        self.block_tables = {}   # {sequence_id: [block_ids]} — خريطة منطقي -> فعلي

    def allocate_for_sequence(self, seq_id: str, num_tokens: int):
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        allocated = [self.free_blocks.pop() for _ in range(num_blocks_needed)]
        self.block_tables[seq_id] = allocated
        return allocated

    def free_sequence(self, seq_id: str):
        """تحرير فوري للذاكرة بمجرد انتهاء التسلسل — يُعاد استخدامها لطلبات جديدة فورًا"""
        self.free_blocks.extend(self.block_tables.pop(seq_id))

    def share_blocks_for_prefix(self, seq_id_a: str, seq_id_b: str, shared_prefix_len: int):
        """
        ميزة إضافية قوية: مشاركة الصفحات الفعلية لنفس البادئة
        (مثل system prompt مشترك) بين عدة طلبات دون نسخ! (Copy-on-Write)
        """
        shared_blocks_count = shared_prefix_len // self.block_size
        self.block_tables[seq_id_b][:shared_blocks_count] = \
            self.block_tables[seq_id_a][:shared_blocks_count]

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-55-ch55-inference-optimization.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class ContinuousBatchingScheduler:
    """
    توضيح مفاهيمي: بمجرد انتهاء أي تسلسل في الدفعة،
    يُستبدل فورًا بطلب جديد من قائمة الانتظار، دون انتظار بقية الدفعة
    """
    def __init__(self, max_batch_size=32):
        self.max_batch_size = max_batch_size
        self.active_sequences = {}   # {seq_id: state}
        self.waiting_queue = []

    def step(self, model):
        """خطوة واحدة من التوليد عبر كل التسلسلات النشطة معًا"""
        # ملء الدفعة من قائمة الانتظار إن وُجدت مساحة فارغة
        while len(self.active_sequences) < self.max_batch_size and self.waiting_queue:
            new_seq = self.waiting_queue.pop(0)
            self.active_sequences[new_seq["id"]] = new_seq

        if not self.active_sequences:
            return

        # تمريرة أمامية واحدة لكل التسلسلات النشطة دفعة واحدة (batched forward pass)
        batch_input = self._prepare_batch(self.active_sequences)
        next_tokens = model.forward_step(batch_input)   # token واحد جديد لكل تسلسل

        # تحديث الحالة وإزالة أي تسلسل اكتمل فورًا (تحرير مكانه للتالي في الطابور)
        for seq_id, token in next_tokens.items():
            self.active_sequences[seq_id]["tokens"].append(token)
            if token == EOS_TOKEN_ID or len(self.active_sequences[seq_id]["tokens"]) >= \
               self.active_sequences[seq_id]["max_tokens"]:
                completed = self.active_sequences.pop(seq_id)
                self._return_result(completed)

    def _prepare_batch(self, sequences): ...  # تفاصيل التجميع
    def _return_result(self, completed_seq): ...  # إرجاع النتيجة للمستخدم

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-55-ch55-inference-optimization.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def speculative_decoding_step(draft_model, target_model, tokenizer, context_ids,
                                num_speculative_tokens=4):
    """
    1. النموذج الصغير (draft) يُولّد K رموز مقترحة بسرعة (تسلسليًا لكنه سريع جدًا)
    2. النموذج الكبير (target) يتحقق من كل الرموز K دفعة واحدة (تمريرة أمامية واحدة فقط!)
    3. نقبل الرموز المتطابقة مع توزيع target، ونرفض ونُصحّح من أول اختلاف
    """
    # الخطوة 1: اقتراح سريع من النموذج الصغير
    draft_tokens = []
    draft_probs = []
    current_ids = context_ids.clone()

    for _ in range(num_speculative_tokens):
        logits = draft_model(current_ids).logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        draft_tokens.append(next_token)
        draft_probs.append(probs.gather(-1, next_token))
        current_ids = torch.cat([current_ids, next_token], dim=1)

    # الخطوة 2: تحقق دفعة واحدة من النموذج الكبير (تمريرة أمامية واحدة لكل الرموز المقترحة)
    full_sequence = current_ids
    target_logits = target_model(full_sequence).logits
    target_probs_all = F.softmax(target_logits, dim=-1)

    # الخطوة 3: قبول/رفض كل رمز مقترح (خوارزمية Rejection Sampling)
    accepted_tokens = []
    context_len = context_ids.shape[1]

    for i, (draft_token, draft_prob) in enumerate(zip(draft_tokens, draft_probs)):
        position = context_len + i - 1
        target_prob = target_probs_all[:, position, :].gather(-1, draft_token)

        acceptance_ratio = torch.min(torch.ones_like(target_prob), target_prob / draft_prob)
        if torch.rand_like(acceptance_ratio) < acceptance_ratio:
            accepted_tokens.append(draft_token)
        else:
            # رفض: نستبدل برمز جديد من توزيع "الفرق" المُصحَّح إحصائيًا
            corrected_dist = torch.clamp(
                target_probs_all[:, position, :] - draft_probs[i], min=0
            )
            corrected_dist = corrected_dist / corrected_dist.sum()
            correction_token = torch.multinomial(corrected_dist, num_samples=1)
            accepted_tokens.append(correction_token)
            break   # نتوقف عند أول رفض

    return torch.cat(accepted_tokens, dim=1) if accepted_tokens else draft_tokens[0]

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-55-ch55-inference-optimization.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-55-ch55-inference-optimization.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — تحسين الاستدلال المتقدم
Mastering Transformers and Large Language Models — Inference Optimization

Book slug : mastering-llms-transformers
Chapter   : ch-55-ch55-inference-optimization (lesson 55)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-11-llmops/ch55-inference-optimization.ipynb#cell-1
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
# [002] content/notebooks/part-11-llmops/ch55-inference-optimization.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
class PagedKVCacheConceptual:
    """
    توضيح مفاهيمي لفكرة PagedAttention:
    بدلاً من حجز ذاكرة متجاورة ضخمة لكل تسلسل، نُقسّم KV Cache
    إلى "صفحات" (blocks) ثابتة الحجم، تُخصَّص ديناميكيًا عند الحاجة فقط
    (تمامًا مثل Paging في أنظمة التشغيل)
    """
    def __init__(self, block_size=16, total_blocks=1000, head_dim=128, n_heads=8):
        self.block_size = block_size
        self.physical_blocks = torch.zeros(total_blocks, block_size, n_heads, head_dim * 2)
        self.free_blocks = list(range(total_blocks))
        self.block_tables = {}   # {sequence_id: [block_ids]} — خريطة منطقي -> فعلي

    def allocate_for_sequence(self, seq_id: str, num_tokens: int):
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        allocated = [self.free_blocks.pop() for _ in range(num_blocks_needed)]
        self.block_tables[seq_id] = allocated
        return allocated

    def free_sequence(self, seq_id: str):
        """تحرير فوري للذاكرة بمجرد انتهاء التسلسل — يُعاد استخدامها لطلبات جديدة فورًا"""
        self.free_blocks.extend(self.block_tables.pop(seq_id))

    def share_blocks_for_prefix(self, seq_id_a: str, seq_id_b: str, shared_prefix_len: int):
        """
        ميزة إضافية قوية: مشاركة الصفحات الفعلية لنفس البادئة
        (مثل system prompt مشترك) بين عدة طلبات دون نسخ! (Copy-on-Write)
        """
        shared_blocks_count = shared_prefix_len // self.block_size
        self.block_tables[seq_id_b][:shared_blocks_count] = \
            self.block_tables[seq_id_a][:shared_blocks_count]

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-11-llmops/ch55-inference-optimization.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class ContinuousBatchingScheduler:
    """
    توضيح مفاهيمي: بمجرد انتهاء أي تسلسل في الدفعة،
    يُستبدل فورًا بطلب جديد من قائمة الانتظار، دون انتظار بقية الدفعة
    """
    def __init__(self, max_batch_size=32):
        self.max_batch_size = max_batch_size
        self.active_sequences = {}   # {seq_id: state}
        self.waiting_queue = []

    def step(self, model):
        """خطوة واحدة من التوليد عبر كل التسلسلات النشطة معًا"""
        # ملء الدفعة من قائمة الانتظار إن وُجدت مساحة فارغة
        while len(self.active_sequences) < self.max_batch_size and self.waiting_queue:
            new_seq = self.waiting_queue.pop(0)
            self.active_sequences[new_seq["id"]] = new_seq

        if not self.active_sequences:
            return

        # تمريرة أمامية واحدة لكل التسلسلات النشطة دفعة واحدة (batched forward pass)
        batch_input = self._prepare_batch(self.active_sequences)
        next_tokens = model.forward_step(batch_input)   # token واحد جديد لكل تسلسل

        # تحديث الحالة وإزالة أي تسلسل اكتمل فورًا (تحرير مكانه للتالي في الطابور)
        for seq_id, token in next_tokens.items():
            self.active_sequences[seq_id]["tokens"].append(token)
            if token == EOS_TOKEN_ID or len(self.active_sequences[seq_id]["tokens"]) >= \
               self.active_sequences[seq_id]["max_tokens"]:
                completed = self.active_sequences.pop(seq_id)
                self._return_result(completed)

    def _prepare_batch(self, sequences): ...  # تفاصيل التجميع
    def _return_result(self, completed_seq): ...  # إرجاع النتيجة للمستخدم

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-11-llmops/ch55-inference-optimization.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def speculative_decoding_step(draft_model, target_model, tokenizer, context_ids,
                                num_speculative_tokens=4):
    """
    1. النموذج الصغير (draft) يُولّد K رموز مقترحة بسرعة (تسلسليًا لكنه سريع جدًا)
    2. النموذج الكبير (target) يتحقق من كل الرموز K دفعة واحدة (تمريرة أمامية واحدة فقط!)
    3. نقبل الرموز المتطابقة مع توزيع target، ونرفض ونُصحّح من أول اختلاف
    """
    # الخطوة 1: اقتراح سريع من النموذج الصغير
    draft_tokens = []
    draft_probs = []
    current_ids = context_ids.clone()

    for _ in range(num_speculative_tokens):
        logits = draft_model(current_ids).logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        draft_tokens.append(next_token)
        draft_probs.append(probs.gather(-1, next_token))
        current_ids = torch.cat([current_ids, next_token], dim=1)

    # الخطوة 2: تحقق دفعة واحدة من النموذج الكبير (تمريرة أمامية واحدة لكل الرموز المقترحة)
    full_sequence = current_ids
    target_logits = target_model(full_sequence).logits
    target_probs_all = F.softmax(target_logits, dim=-1)

    # الخطوة 3: قبول/رفض كل رمز مقترح (خوارزمية Rejection Sampling)
    accepted_tokens = []
    context_len = context_ids.shape[1]

    for i, (draft_token, draft_prob) in enumerate(zip(draft_tokens, draft_probs)):
        position = context_len + i - 1
        target_prob = target_probs_all[:, position, :].gather(-1, draft_token)

        acceptance_ratio = torch.min(torch.ones_like(target_prob), target_prob / draft_prob)
        if torch.rand_like(acceptance_ratio) < acceptance_ratio:
            accepted_tokens.append(draft_token)
        else:
            # رفض: نستبدل برمز جديد من توزيع "الفرق" المُصحَّح إحصائيًا
            corrected_dist = torch.clamp(
                target_probs_all[:, position, :] - draft_probs[i], min=0
            )
            corrected_dist = corrected_dist / corrected_dist.sum()
            correction_token = torch.multinomial(corrected_dist, num_samples=1)
            accepted_tokens.append(correction_token)
            break   # نتوقف عند أول رفض

    return torch.cat(accepted_tokens, dim=1) if accepted_tokens else draft_tokens[0]

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-11-llmops/ch55-inference-optimization.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-55-ch55-inference-optimization.ipynb#cell-2
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-55-ch55-inference-optimization.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class PagedKVCacheConceptual:
    """
    توضيح مفاهيمي لفكرة PagedAttention:
    بدلاً من حجز ذاكرة متجاورة ضخمة لكل تسلسل، نُقسّم KV Cache
    إلى "صفحات" (blocks) ثابتة الحجم، تُخصَّص ديناميكيًا عند الحاجة فقط
    (تمامًا مثل Paging في أنظمة التشغيل)
    """
    def __init__(self, block_size=16, total_blocks=1000, head_dim=128, n_heads=8):
        self.block_size = block_size
        self.physical_blocks = torch.zeros(total_blocks, block_size, n_heads, head_dim * 2)
        self.free_blocks = list(range(total_blocks))
        self.block_tables = {}   # {sequence_id: [block_ids]} — خريطة منطقي -> فعلي

    def allocate_for_sequence(self, seq_id: str, num_tokens: int):
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        allocated = [self.free_blocks.pop() for _ in range(num_blocks_needed)]
        self.block_tables[seq_id] = allocated
        return allocated

    def free_sequence(self, seq_id: str):
        """تحرير فوري للذاكرة بمجرد انتهاء التسلسل — يُعاد استخدامها لطلبات جديدة فورًا"""
        self.free_blocks.extend(self.block_tables.pop(seq_id))

    def share_blocks_for_prefix(self, seq_id_a: str, seq_id_b: str, shared_prefix_len: int):
        """
        ميزة إضافية قوية: مشاركة الصفحات الفعلية لنفس البادئة
        (مثل system prompt مشترك) بين عدة طلبات دون نسخ! (Copy-on-Write)
        """
        shared_blocks_count = shared_prefix_len // self.block_size
        self.block_tables[seq_id_b][:shared_blocks_count] = \
            self.block_tables[seq_id_a][:shared_blocks_count]

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-55-ch55-inference-optimization.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class ContinuousBatchingScheduler:
    """
    توضيح مفاهيمي: بمجرد انتهاء أي تسلسل في الدفعة،
    يُستبدل فورًا بطلب جديد من قائمة الانتظار، دون انتظار بقية الدفعة
    """
    def __init__(self, max_batch_size=32):
        self.max_batch_size = max_batch_size
        self.active_sequences = {}   # {seq_id: state}
        self.waiting_queue = []

    def step(self, model):
        """خطوة واحدة من التوليد عبر كل التسلسلات النشطة معًا"""
        # ملء الدفعة من قائمة الانتظار إن وُجدت مساحة فارغة
        while len(self.active_sequences) < self.max_batch_size and self.waiting_queue:
            new_seq = self.waiting_queue.pop(0)
            self.active_sequences[new_seq["id"]] = new_seq

        if not self.active_sequences:
            return

        # تمريرة أمامية واحدة لكل التسلسلات النشطة دفعة واحدة (batched forward pass)
        batch_input = self._prepare_batch(self.active_sequences)
        next_tokens = model.forward_step(batch_input)   # token واحد جديد لكل تسلسل

        # تحديث الحالة وإزالة أي تسلسل اكتمل فورًا (تحرير مكانه للتالي في الطابور)
        for seq_id, token in next_tokens.items():
            self.active_sequences[seq_id]["tokens"].append(token)
            if token == EOS_TOKEN_ID or len(self.active_sequences[seq_id]["tokens"]) >= \
               self.active_sequences[seq_id]["max_tokens"]:
                completed = self.active_sequences.pop(seq_id)
                self._return_result(completed)

    def _prepare_batch(self, sequences): ...  # تفاصيل التجميع
    def _return_result(self, completed_seq): ...  # إرجاع النتيجة للمستخدم

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-55-ch55-inference-optimization.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def speculative_decoding_step(draft_model, target_model, tokenizer, context_ids,
                                num_speculative_tokens=4):
    """
    1. النموذج الصغير (draft) يُولّد K رموز مقترحة بسرعة (تسلسليًا لكنه سريع جدًا)
    2. النموذج الكبير (target) يتحقق من كل الرموز K دفعة واحدة (تمريرة أمامية واحدة فقط!)
    3. نقبل الرموز المتطابقة مع توزيع target، ونرفض ونُصحّح من أول اختلاف
    """
    # الخطوة 1: اقتراح سريع من النموذج الصغير
    draft_tokens = []
    draft_probs = []
    current_ids = context_ids.clone()

    for _ in range(num_speculative_tokens):
        logits = draft_model(current_ids).logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        draft_tokens.append(next_token)
        draft_probs.append(probs.gather(-1, next_token))
        current_ids = torch.cat([current_ids, next_token], dim=1)

    # الخطوة 2: تحقق دفعة واحدة من النموذج الكبير (تمريرة أمامية واحدة لكل الرموز المقترحة)
    full_sequence = current_ids
    target_logits = target_model(full_sequence).logits
    target_probs_all = F.softmax(target_logits, dim=-1)

    # الخطوة 3: قبول/رفض كل رمز مقترح (خوارزمية Rejection Sampling)
    accepted_tokens = []
    context_len = context_ids.shape[1]

    for i, (draft_token, draft_prob) in enumerate(zip(draft_tokens, draft_probs)):
        position = context_len + i - 1
        target_prob = target_probs_all[:, position, :].gather(-1, draft_token)

        acceptance_ratio = torch.min(torch.ones_like(target_prob), target_prob / draft_prob)
        if torch.rand_like(acceptance_ratio) < acceptance_ratio:
            accepted_tokens.append(draft_token)
        else:
            # رفض: نستبدل برمز جديد من توزيع "الفرق" المُصحَّح إحصائيًا
            corrected_dist = torch.clamp(
                target_probs_all[:, position, :] - draft_probs[i], min=0
            )
            corrected_dist = corrected_dist / corrected_dist.sum()
            correction_token = torch.multinomial(corrected_dist, num_samples=1)
            accepted_tokens.append(correction_token)
            break   # نتوقف عند أول رفض

    return torch.cat(accepted_tokens, dim=1) if accepted_tokens else draft_tokens[0]

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-55-ch55-inference-optimization.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
