#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الشبكات المتكررة: RNN وLSTM وGRU
Mastering Transformers and Large Language Models — Recurrent Networks

Book slug : mastering-llms-transformers
Chapter   : ch-08-ch08-recurrent-networks (lesson 8)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-08-ch08-recurrent-networks.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الشبكات المتكررة: RNN وLSTM وGRU
Mastering Transformers and Large Language Models — Recurrent Networks

Book slug : mastering-llms-transformers
Chapter   : ch-08-ch08-recurrent-networks (lesson 8)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-08-ch08-recurrent-networks.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الشبكات المتكررة: RNN وLSTM وGRU
Mastering Transformers and Large Language Models — Recurrent Networks

Book slug : mastering-llms-transformers
Chapter   : ch-08-ch08-recurrent-networks (lesson 8)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-02-language-representation/ch08-recurrent-networks.ipynb#cell-1
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
    ("numpy", "numpy"),
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
# [002] content/notebooks/part-02-language-representation/ch08-recurrent-networks.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np

class SimpleRNNCell:
    """
    تطبيق خلية RNN واحدة من الصفر، لتوضيح الآلية الأساسية
    قبل الانتقال لتطبيق PyTorch الكامل.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        self.hidden_dim = hidden_dim
        scale = 1.0 / np.sqrt(hidden_dim)
        self.W_xh = np.random.randn(input_dim, hidden_dim) * scale
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * scale
        self.b_h = np.zeros(hidden_dim)

    def forward_step(self, x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        z = x_t @ self.W_xh + h_prev @ self.W_hh + self.b_h
        h_t = np.tanh(z)
        return h_t

    def forward_sequence(self, X: np.ndarray) -> list:
        """معالجة تسلسل كامل، خطوة زمنية تلو الأخرى (معالجة متسلسلة إلزامية!)."""
        seq_len = X.shape[0]
        h = np.zeros(self.hidden_dim)
        hidden_states = []
        for t in range(seq_len):
            h = self.forward_step(X[t], h)
            hidden_states.append(h.copy())
        return hidden_states

def demonstrate_vanishing_gradient_problem():
    """
    توضيح عملي لمشكلة تلاشي التدرجات في RNN: عبر قاعدة السلسلة،
    تدرج الخسارة بالنسبة لحالة مخفية بعيدة زمنيًا هو حاصل ضرب
    سلسلة طويلة من مشتقات tanh (أقصاها 1، وعادة أقل بكثير) مضروبة
    في W_hh تكراريًا. إذا كانت القيم الذاتية لـ W_hh أقل من 1،
    هذا الحاصل يتلاشى أُسّيًا مع طول التسلسل.
    """
    seq_lengths = [5, 20, 50, 100]
    # محاكاة مبسّطة: نفترض متوسط عامل انكماش لكل خطوة زمنية = 0.5
    # (قيمة واقعية جدًا لمشتقة tanh في أغلب نطاقات التشغيل العملية)
    decay_factor_per_step = 0.5

    print("تأثير طول التسلسل على مقدار التدرج المُتراكم (محاكاة):")
    for length in seq_lengths:
        gradient_magnitude = decay_factor_per_step ** length
        print(f"  طول التسلسل={length}: مقدار التدرج ≈ {gradient_magnitude:.2e}")

demonstrate_vanishing_gradient_problem()
# طول التسلسل=5: مقدار التدرج ≈ 3.13e-02
# طول التسلسل=20: مقدار التدرج ≈ 9.54e-07
# طول التسلسل=50: مقدار التدرج ≈ 8.88e-16
# طول التسلسل=100: مقدار التدرج ≈ 7.89e-31   <- تلاشى عمليًا تمامًا!

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-02-language-representation/ch08-recurrent-networks.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class LSTMCellFromScratch:
    """
    تطبيق كامل لخلية LSTM من الصفر (Hochreiter & Schmidhuber, 1997).
    الفكرة الثورية: فصل "حالة الخلية" (Cell State, C) عن "الحالة
    المخفية" (Hidden State, h). حالة الخلية تعمل كـ"حزام ناقل"
    (conveyor belt) يسمح للمعلومة بالتدفق عبر الزمن مع تدخل خطي
    بسيط فقط (جمع وضرب عنصري)، مما يحل مشكلة تلاشي التدرجات بشكل
    كبير جدًا مقارنة بـ RNN البسيطة.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        self.hidden_dim = hidden_dim
        concat_dim = input_dim + hidden_dim
        scale = 1.0 / np.sqrt(hidden_dim)

        # أربع بوابات، كل منها لها أوزانها الخاصة (تُطبَّق على [x_t, h_{t-1}] مُدمَجين)
        self.W_forget = np.random.randn(concat_dim, hidden_dim) * scale
        self.W_input = np.random.randn(concat_dim, hidden_dim) * scale
        self.W_candidate = np.random.randn(concat_dim, hidden_dim) * scale
        self.W_output = np.random.randn(concat_dim, hidden_dim) * scale

        self.b_forget = np.ones(hidden_dim)  # تهيئة forget bias بـ1 (حيلة مهمة!)
        self.b_input = np.zeros(hidden_dim)
        self.b_candidate = np.zeros(hidden_dim)
        self.b_output = np.zeros(hidden_dim)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward_step(self, x_t: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray):
        """
        أربع بوابات، كل منها بدور محدد بدقة:

        1. Forget Gate (f_t): "كم من الذاكرة القديمة (c_prev) نحتفظ بها؟"
        2. Input Gate (i_t): "كم من المعلومة الجديدة ندخلها للذاكرة؟"
        3. Candidate (c̃_t): "ما هي المعلومة الجديدة المُرشَّحة للإضافة؟"
        4. Output Gate (o_t): "كم من الذاكرة الحالية نُظهِر كحالة مخفية؟"
        """
        combined = np.concatenate([x_t, h_prev])

        forget_gate = self.sigmoid(combined @ self.W_forget + self.b_forget)
        input_gate = self.sigmoid(combined @ self.W_input + self.b_input)
        candidate = np.tanh(combined @ self.W_candidate + self.b_candidate)
        output_gate = self.sigmoid(combined @ self.W_output + self.b_output)

        # تحديث حالة الخلية: مزيج من "نسيان جزء من القديم" و"إضافة جزء من الجديد"
        # هذا التحديث **خطي بشكل أساسي** (جمع + ضرب عنصري، لا tanh متكرر)
        # وهذا بالضبط ما يسمح للتدرج بالتدفق عبر مسافات زمنية طويلة دون تلاشٍ شديد
        c_t = forget_gate * c_prev + input_gate * candidate

        # الحالة المخفية: جزء "مُرشَّح للإظهار" من حالة الخلية الجديدة
        h_t = output_gate * np.tanh(c_t)

        return h_t, c_t, {
            "forget_gate": forget_gate, "input_gate": input_gate,
            "candidate": candidate, "output_gate": output_gate,
        }

    def forward_sequence(self, X: np.ndarray):
        seq_len = X.shape[0]
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)
        hidden_states, cell_states = [], []

        for t in range(seq_len):
            h, c, gates = self.forward_step(X[t], h, c)
            hidden_states.append(h.copy())
            cell_states.append(c.copy())

        return hidden_states, cell_states

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-02-language-representation/ch08-recurrent-networks.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class GRUCellFromScratch:
    """
    تطبيق GRU (Cho et al., 2014) من الصفر: دمج بوابتي Forget وInput
    في بوابة واحدة (Update Gate)، وإزالة حالة الخلية المنفصلة تمامًا
    (الاعتماد فقط على الحالة المخفية h). نتيجة ذلك: معاملات أقل
    بنسبة ~25% من LSTM، وتدريب أسرع، مع أداء مقارب جدًا في أغلب المهام.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        self.hidden_dim = hidden_dim
        concat_dim = input_dim + hidden_dim
        scale = 1.0 / np.sqrt(hidden_dim)

        self.W_update = np.random.randn(concat_dim, hidden_dim) * scale
        self.W_reset = np.random.randn(concat_dim, hidden_dim) * scale
        self.W_candidate = np.random.randn(concat_dim, hidden_dim) * scale

        self.b_update = np.zeros(hidden_dim)
        self.b_reset = np.zeros(hidden_dim)
        self.b_candidate = np.zeros(hidden_dim)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward_step(self, x_t: np.ndarray, h_prev: np.ndarray):
        combined = np.concatenate([x_t, h_prev])

        # Update Gate: يُحدِّد "مزيج" الاحتفاظ بالقديم مقابل الجديد (يدمج دوري forget+input من LSTM)
        update_gate = self.sigmoid(combined @ self.W_update + self.b_update)

        # Reset Gate: يُحدِّد "كم من الماضي نتجاهل" عند حساب المُرشَّح الجديد
        reset_gate = self.sigmoid(combined @ self.W_reset + self.b_reset)

        combined_reset = np.concatenate([x_t, reset_gate * h_prev])
        candidate = np.tanh(combined_reset @ self.W_candidate + self.b_candidate)

        # التحديث النهائي: استيفاء خطي بين الحالة القديمة والمُرشَّح الجديد
        h_t = (1 - update_gate) * h_prev + update_gate * candidate
        return h_t

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-02-language-representation/ch08-recurrent-networks.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class LSTMTextClassifier(nn.Module):
    """
    مُصنِّف نصوص كامل باستخدام LSTM ثنائي الاتجاه (Bidirectional):
    يُعالِج التسلسل من البداية للنهاية ومن النهاية للبداية في آن واحد،
    مما يسمح لكل موضع بالوصول لمعلومات من كلا الاتجاهين — تحسين
    مهم جدًا قبل ظهور Attention، لكنه لا يزال أبطأ وأقل فعالية
    من Self-Attention الكامل (الفصل 10) في التقاط الاعتماديات
    البعيدة جدًا.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 hidden_dim: int = 256, num_classes: int = 2,
                 num_layers: int = 2, dropout: float = 0.3,
                 pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,     # معالجة في الاتجاهين
            dropout=dropout if num_layers > 1 else 0,
        )

        # ×2 لأن bidirectional يُركِّب (concatenate) الاتجاهين معًا
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor = None):
        embedded = self.embedding(input_ids)   # (batch, seq_len, embedding_dim)

        if lengths is not None:
            # Packing: تحسين كفاءة حسابية مهم جدًا عند وجود padding
            # (يتجاهل الحسابات على مواضع الحشو، يُسرّع التدريب بشكل ملحوظ)
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, (h_n, c_n) = self.lstm(packed)
        else:
            _, (h_n, c_n) = self.lstm(embedded)

        # h_n شكلها: (num_layers*2, batch, hidden_dim)
        # نأخذ الحالة المخفية الأخيرة من كلا الاتجاهين للطبقة الأخيرة فقط
        forward_last = h_n[-2, :, :]   # آخر حالة للاتجاه الأمامي
        backward_last = h_n[-1, :, :]  # آخر حالة للاتجاه الخلفي
        combined = torch.cat([forward_last, backward_last], dim=1)

        combined = self.dropout(combined)
        logits = self.fc(combined)
        return logits

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-02-language-representation/ch08-recurrent-networks.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def summarize_rnn_limitations():
    """
    ملخص القيود الجوهرية التي دفعت الباحثين لاختراع Attention
    والمحوّلات بالكامل (سنرى الحل الكامل في الفصل 10).
    """
    limitations = {
        "المعالجة المتسلسلة الإلزامية": (
            "لا يمكن حساب h_t دون معرفة h_{t-1} أولًا، مما يمنع "
            "التوازي الكامل عبر طول التسلسل أثناء التدريب — عنق "
            "زجاجة حاسوبي خطير جدًا لتدريب نماذج كبيرة على GPUs حديثة "
            "مُصمَّمة أساسًا للتوازي الضخم."
        ),
        "صعوبة الاعتماديات طويلة المدى": (
            "رغم تحسّن LSTM/GRU الكبير عن RNN البسيطة، لا تزال المعلومة "
            "تمر عبر 'عنق زجاجة' حالة مخفية ثابتة الحجم، ما يُصعِّب "
            "الاحتفاظ بتفاصيل دقيقة من بداية تسلسل طويل جدًا (آلاف الكلمات)."
        ),
        "صعوبة بناء 'خريطة اعتماديات' مباشرة وصريحة": (
            "لا توجد آلية مباشرة لأي كلمة للوصول مباشرة (بخطوة واحدة) "
            "لأي كلمة أخرى بعيدة في التسلسل — كل المعلومة يجب أن تمر "
            "عبر سلسلة من التحديثات التسلسلية الوسيطة."
        ),
    }

    for problem, explanation in limitations.items():
        print(f"❌ {problem}:\n   {explanation}\n")

summarize_rnn_limitations()

# ────────────────────────────────────────────────────────────────────────
# [007] content/notebooks/part-02-language-representation/ch08-recurrent-networks.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def compare_parameter_counts(input_dim=100, hidden_dim=256):
    # RNN بسيطة: W_xh + W_hh + b_h
    rnn_params = (input_dim * hidden_dim) + (hidden_dim * hidden_dim) + hidden_dim

    # LSTM: 4 بوابات، كل منها بنفس حجم أوزان RNN تقريبًا
    concat_dim = input_dim + hidden_dim
    lstm_params = 4 * (concat_dim * hidden_dim + hidden_dim)

    # GRU: 3 "بوابات" فقط (update, reset, candidate)
    gru_params = 3 * (concat_dim * hidden_dim + hidden_dim)

    print(f"RNN البسيطة:  {rnn_params:,} معامل")
    print(f"LSTM:         {lstm_params:,} معامل ({lstm_params/rnn_params:.1f}× من RNN)")
    print(f"GRU:          {gru_params:,} معامل ({gru_params/rnn_params:.1f}× من RNN، "
          f"{gru_params/lstm_params:.1%} من LSTM)")

compare_parameter_counts()
# RNN البسيطة:  91,392 معامل
# LSTM:         365,568 معامل (4.0× من RNN)
# GRU:          274,176 معامل (3.0× من RNN، 75.0% من LSTM)

# ────────────────────────────────────────────────────────────────────────
# [008] content/notebooks/part-02-language-representation/ch08-recurrent-networks.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-08-ch08-recurrent-networks.ipynb#cell-1
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
    ("numpy", "numpy"),
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-08-ch08-recurrent-networks.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np

class SimpleRNNCell:
    """
    تطبيق خلية RNN واحدة من الصفر، لتوضيح الآلية الأساسية
    قبل الانتقال لتطبيق PyTorch الكامل.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        self.hidden_dim = hidden_dim
        scale = 1.0 / np.sqrt(hidden_dim)
        self.W_xh = np.random.randn(input_dim, hidden_dim) * scale
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * scale
        self.b_h = np.zeros(hidden_dim)

    def forward_step(self, x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        z = x_t @ self.W_xh + h_prev @ self.W_hh + self.b_h
        h_t = np.tanh(z)
        return h_t

    def forward_sequence(self, X: np.ndarray) -> list:
        """معالجة تسلسل كامل، خطوة زمنية تلو الأخرى (معالجة متسلسلة إلزامية!)."""
        seq_len = X.shape[0]
        h = np.zeros(self.hidden_dim)
        hidden_states = []
        for t in range(seq_len):
            h = self.forward_step(X[t], h)
            hidden_states.append(h.copy())
        return hidden_states

def demonstrate_vanishing_gradient_problem():
    """
    توضيح عملي لمشكلة تلاشي التدرجات في RNN: عبر قاعدة السلسلة،
    تدرج الخسارة بالنسبة لحالة مخفية بعيدة زمنيًا هو حاصل ضرب
    سلسلة طويلة من مشتقات tanh (أقصاها 1، وعادة أقل بكثير) مضروبة
    في W_hh تكراريًا. إذا كانت القيم الذاتية لـ W_hh أقل من 1،
    هذا الحاصل يتلاشى أُسّيًا مع طول التسلسل.
    """
    seq_lengths = [5, 20, 50, 100]
    # محاكاة مبسّطة: نفترض متوسط عامل انكماش لكل خطوة زمنية = 0.5
    # (قيمة واقعية جدًا لمشتقة tanh في أغلب نطاقات التشغيل العملية)
    decay_factor_per_step = 0.5

    print("تأثير طول التسلسل على مقدار التدرج المُتراكم (محاكاة):")
    for length in seq_lengths:
        gradient_magnitude = decay_factor_per_step ** length
        print(f"  طول التسلسل={length}: مقدار التدرج ≈ {gradient_magnitude:.2e}")

demonstrate_vanishing_gradient_problem()
# طول التسلسل=5: مقدار التدرج ≈ 3.13e-02
# طول التسلسل=20: مقدار التدرج ≈ 9.54e-07
# طول التسلسل=50: مقدار التدرج ≈ 8.88e-16
# طول التسلسل=100: مقدار التدرج ≈ 7.89e-31   <- تلاشى عمليًا تمامًا!

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-08-ch08-recurrent-networks.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class LSTMCellFromScratch:
    """
    تطبيق كامل لخلية LSTM من الصفر (Hochreiter & Schmidhuber, 1997).
    الفكرة الثورية: فصل "حالة الخلية" (Cell State, C) عن "الحالة
    المخفية" (Hidden State, h). حالة الخلية تعمل كـ"حزام ناقل"
    (conveyor belt) يسمح للمعلومة بالتدفق عبر الزمن مع تدخل خطي
    بسيط فقط (جمع وضرب عنصري)، مما يحل مشكلة تلاشي التدرجات بشكل
    كبير جدًا مقارنة بـ RNN البسيطة.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        self.hidden_dim = hidden_dim
        concat_dim = input_dim + hidden_dim
        scale = 1.0 / np.sqrt(hidden_dim)

        # أربع بوابات، كل منها لها أوزانها الخاصة (تُطبَّق على [x_t, h_{t-1}] مُدمَجين)
        self.W_forget = np.random.randn(concat_dim, hidden_dim) * scale
        self.W_input = np.random.randn(concat_dim, hidden_dim) * scale
        self.W_candidate = np.random.randn(concat_dim, hidden_dim) * scale
        self.W_output = np.random.randn(concat_dim, hidden_dim) * scale

        self.b_forget = np.ones(hidden_dim)  # تهيئة forget bias بـ1 (حيلة مهمة!)
        self.b_input = np.zeros(hidden_dim)
        self.b_candidate = np.zeros(hidden_dim)
        self.b_output = np.zeros(hidden_dim)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward_step(self, x_t: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray):
        """
        أربع بوابات، كل منها بدور محدد بدقة:

        1. Forget Gate (f_t): "كم من الذاكرة القديمة (c_prev) نحتفظ بها؟"
        2. Input Gate (i_t): "كم من المعلومة الجديدة ندخلها للذاكرة؟"
        3. Candidate (c̃_t): "ما هي المعلومة الجديدة المُرشَّحة للإضافة؟"
        4. Output Gate (o_t): "كم من الذاكرة الحالية نُظهِر كحالة مخفية؟"
        """
        combined = np.concatenate([x_t, h_prev])

        forget_gate = self.sigmoid(combined @ self.W_forget + self.b_forget)
        input_gate = self.sigmoid(combined @ self.W_input + self.b_input)
        candidate = np.tanh(combined @ self.W_candidate + self.b_candidate)
        output_gate = self.sigmoid(combined @ self.W_output + self.b_output)

        # تحديث حالة الخلية: مزيج من "نسيان جزء من القديم" و"إضافة جزء من الجديد"
        # هذا التحديث **خطي بشكل أساسي** (جمع + ضرب عنصري، لا tanh متكرر)
        # وهذا بالضبط ما يسمح للتدرج بالتدفق عبر مسافات زمنية طويلة دون تلاشٍ شديد
        c_t = forget_gate * c_prev + input_gate * candidate

        # الحالة المخفية: جزء "مُرشَّح للإظهار" من حالة الخلية الجديدة
        h_t = output_gate * np.tanh(c_t)

        return h_t, c_t, {
            "forget_gate": forget_gate, "input_gate": input_gate,
            "candidate": candidate, "output_gate": output_gate,
        }

    def forward_sequence(self, X: np.ndarray):
        seq_len = X.shape[0]
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)
        hidden_states, cell_states = [], []

        for t in range(seq_len):
            h, c, gates = self.forward_step(X[t], h, c)
            hidden_states.append(h.copy())
            cell_states.append(c.copy())

        return hidden_states, cell_states

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-08-ch08-recurrent-networks.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class GRUCellFromScratch:
    """
    تطبيق GRU (Cho et al., 2014) من الصفر: دمج بوابتي Forget وInput
    في بوابة واحدة (Update Gate)، وإزالة حالة الخلية المنفصلة تمامًا
    (الاعتماد فقط على الحالة المخفية h). نتيجة ذلك: معاملات أقل
    بنسبة ~25% من LSTM، وتدريب أسرع، مع أداء مقارب جدًا في أغلب المهام.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        self.hidden_dim = hidden_dim
        concat_dim = input_dim + hidden_dim
        scale = 1.0 / np.sqrt(hidden_dim)

        self.W_update = np.random.randn(concat_dim, hidden_dim) * scale
        self.W_reset = np.random.randn(concat_dim, hidden_dim) * scale
        self.W_candidate = np.random.randn(concat_dim, hidden_dim) * scale

        self.b_update = np.zeros(hidden_dim)
        self.b_reset = np.zeros(hidden_dim)
        self.b_candidate = np.zeros(hidden_dim)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward_step(self, x_t: np.ndarray, h_prev: np.ndarray):
        combined = np.concatenate([x_t, h_prev])

        # Update Gate: يُحدِّد "مزيج" الاحتفاظ بالقديم مقابل الجديد (يدمج دوري forget+input من LSTM)
        update_gate = self.sigmoid(combined @ self.W_update + self.b_update)

        # Reset Gate: يُحدِّد "كم من الماضي نتجاهل" عند حساب المُرشَّح الجديد
        reset_gate = self.sigmoid(combined @ self.W_reset + self.b_reset)

        combined_reset = np.concatenate([x_t, reset_gate * h_prev])
        candidate = np.tanh(combined_reset @ self.W_candidate + self.b_candidate)

        # التحديث النهائي: استيفاء خطي بين الحالة القديمة والمُرشَّح الجديد
        h_t = (1 - update_gate) * h_prev + update_gate * candidate
        return h_t

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-08-ch08-recurrent-networks.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class LSTMTextClassifier(nn.Module):
    """
    مُصنِّف نصوص كامل باستخدام LSTM ثنائي الاتجاه (Bidirectional):
    يُعالِج التسلسل من البداية للنهاية ومن النهاية للبداية في آن واحد،
    مما يسمح لكل موضع بالوصول لمعلومات من كلا الاتجاهين — تحسين
    مهم جدًا قبل ظهور Attention، لكنه لا يزال أبطأ وأقل فعالية
    من Self-Attention الكامل (الفصل 10) في التقاط الاعتماديات
    البعيدة جدًا.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 hidden_dim: int = 256, num_classes: int = 2,
                 num_layers: int = 2, dropout: float = 0.3,
                 pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,     # معالجة في الاتجاهين
            dropout=dropout if num_layers > 1 else 0,
        )

        # ×2 لأن bidirectional يُركِّب (concatenate) الاتجاهين معًا
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor = None):
        embedded = self.embedding(input_ids)   # (batch, seq_len, embedding_dim)

        if lengths is not None:
            # Packing: تحسين كفاءة حسابية مهم جدًا عند وجود padding
            # (يتجاهل الحسابات على مواضع الحشو، يُسرّع التدريب بشكل ملحوظ)
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, (h_n, c_n) = self.lstm(packed)
        else:
            _, (h_n, c_n) = self.lstm(embedded)

        # h_n شكلها: (num_layers*2, batch, hidden_dim)
        # نأخذ الحالة المخفية الأخيرة من كلا الاتجاهين للطبقة الأخيرة فقط
        forward_last = h_n[-2, :, :]   # آخر حالة للاتجاه الأمامي
        backward_last = h_n[-1, :, :]  # آخر حالة للاتجاه الخلفي
        combined = torch.cat([forward_last, backward_last], dim=1)

        combined = self.dropout(combined)
        logits = self.fc(combined)
        return logits

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-08-ch08-recurrent-networks.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def summarize_rnn_limitations():
    """
    ملخص القيود الجوهرية التي دفعت الباحثين لاختراع Attention
    والمحوّلات بالكامل (سنرى الحل الكامل في الفصل 10).
    """
    limitations = {
        "المعالجة المتسلسلة الإلزامية": (
            "لا يمكن حساب h_t دون معرفة h_{t-1} أولًا، مما يمنع "
            "التوازي الكامل عبر طول التسلسل أثناء التدريب — عنق "
            "زجاجة حاسوبي خطير جدًا لتدريب نماذج كبيرة على GPUs حديثة "
            "مُصمَّمة أساسًا للتوازي الضخم."
        ),
        "صعوبة الاعتماديات طويلة المدى": (
            "رغم تحسّن LSTM/GRU الكبير عن RNN البسيطة، لا تزال المعلومة "
            "تمر عبر 'عنق زجاجة' حالة مخفية ثابتة الحجم، ما يُصعِّب "
            "الاحتفاظ بتفاصيل دقيقة من بداية تسلسل طويل جدًا (آلاف الكلمات)."
        ),
        "صعوبة بناء 'خريطة اعتماديات' مباشرة وصريحة": (
            "لا توجد آلية مباشرة لأي كلمة للوصول مباشرة (بخطوة واحدة) "
            "لأي كلمة أخرى بعيدة في التسلسل — كل المعلومة يجب أن تمر "
            "عبر سلسلة من التحديثات التسلسلية الوسيطة."
        ),
    }

    for problem, explanation in limitations.items():
        print(f"❌ {problem}:\n   {explanation}\n")

summarize_rnn_limitations()

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-llms-transformers/notebooks/ch-08-ch08-recurrent-networks.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def compare_parameter_counts(input_dim=100, hidden_dim=256):
    # RNN بسيطة: W_xh + W_hh + b_h
    rnn_params = (input_dim * hidden_dim) + (hidden_dim * hidden_dim) + hidden_dim

    # LSTM: 4 بوابات، كل منها بنفس حجم أوزان RNN تقريبًا
    concat_dim = input_dim + hidden_dim
    lstm_params = 4 * (concat_dim * hidden_dim + hidden_dim)

    # GRU: 3 "بوابات" فقط (update, reset, candidate)
    gru_params = 3 * (concat_dim * hidden_dim + hidden_dim)

    print(f"RNN البسيطة:  {rnn_params:,} معامل")
    print(f"LSTM:         {lstm_params:,} معامل ({lstm_params/rnn_params:.1f}× من RNN)")
    print(f"GRU:          {gru_params:,} معامل ({gru_params/rnn_params:.1f}× من RNN، "
          f"{gru_params/lstm_params:.1%} من LSTM)")

compare_parameter_counts()
# RNN البسيطة:  91,392 معامل
# LSTM:         365,568 معامل (4.0× من RNN)
# GRU:          274,176 معامل (3.0× من RNN، 75.0% من LSTM)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-llms-transformers/notebooks/ch-08-ch08-recurrent-networks.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-08-ch08-recurrent-networks.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الشبكات المتكررة: RNN وLSTM وGRU
Mastering Transformers and Large Language Models — Recurrent Networks

Book slug : mastering-llms-transformers
Chapter   : ch-08-ch08-recurrent-networks (lesson 8)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-02-language-representation/ch08-recurrent-networks.ipynb#cell-1
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
    ("numpy", "numpy"),
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
# [002] content/notebooks/part-02-language-representation/ch08-recurrent-networks.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np

class SimpleRNNCell:
    """
    تطبيق خلية RNN واحدة من الصفر، لتوضيح الآلية الأساسية
    قبل الانتقال لتطبيق PyTorch الكامل.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        self.hidden_dim = hidden_dim
        scale = 1.0 / np.sqrt(hidden_dim)
        self.W_xh = np.random.randn(input_dim, hidden_dim) * scale
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * scale
        self.b_h = np.zeros(hidden_dim)

    def forward_step(self, x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        z = x_t @ self.W_xh + h_prev @ self.W_hh + self.b_h
        h_t = np.tanh(z)
        return h_t

    def forward_sequence(self, X: np.ndarray) -> list:
        """معالجة تسلسل كامل، خطوة زمنية تلو الأخرى (معالجة متسلسلة إلزامية!)."""
        seq_len = X.shape[0]
        h = np.zeros(self.hidden_dim)
        hidden_states = []
        for t in range(seq_len):
            h = self.forward_step(X[t], h)
            hidden_states.append(h.copy())
        return hidden_states

def demonstrate_vanishing_gradient_problem():
    """
    توضيح عملي لمشكلة تلاشي التدرجات في RNN: عبر قاعدة السلسلة،
    تدرج الخسارة بالنسبة لحالة مخفية بعيدة زمنيًا هو حاصل ضرب
    سلسلة طويلة من مشتقات tanh (أقصاها 1، وعادة أقل بكثير) مضروبة
    في W_hh تكراريًا. إذا كانت القيم الذاتية لـ W_hh أقل من 1،
    هذا الحاصل يتلاشى أُسّيًا مع طول التسلسل.
    """
    seq_lengths = [5, 20, 50, 100]
    # محاكاة مبسّطة: نفترض متوسط عامل انكماش لكل خطوة زمنية = 0.5
    # (قيمة واقعية جدًا لمشتقة tanh في أغلب نطاقات التشغيل العملية)
    decay_factor_per_step = 0.5

    print("تأثير طول التسلسل على مقدار التدرج المُتراكم (محاكاة):")
    for length in seq_lengths:
        gradient_magnitude = decay_factor_per_step ** length
        print(f"  طول التسلسل={length}: مقدار التدرج ≈ {gradient_magnitude:.2e}")

demonstrate_vanishing_gradient_problem()
# طول التسلسل=5: مقدار التدرج ≈ 3.13e-02
# طول التسلسل=20: مقدار التدرج ≈ 9.54e-07
# طول التسلسل=50: مقدار التدرج ≈ 8.88e-16
# طول التسلسل=100: مقدار التدرج ≈ 7.89e-31   <- تلاشى عمليًا تمامًا!

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-02-language-representation/ch08-recurrent-networks.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class LSTMCellFromScratch:
    """
    تطبيق كامل لخلية LSTM من الصفر (Hochreiter & Schmidhuber, 1997).
    الفكرة الثورية: فصل "حالة الخلية" (Cell State, C) عن "الحالة
    المخفية" (Hidden State, h). حالة الخلية تعمل كـ"حزام ناقل"
    (conveyor belt) يسمح للمعلومة بالتدفق عبر الزمن مع تدخل خطي
    بسيط فقط (جمع وضرب عنصري)، مما يحل مشكلة تلاشي التدرجات بشكل
    كبير جدًا مقارنة بـ RNN البسيطة.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        self.hidden_dim = hidden_dim
        concat_dim = input_dim + hidden_dim
        scale = 1.0 / np.sqrt(hidden_dim)

        # أربع بوابات، كل منها لها أوزانها الخاصة (تُطبَّق على [x_t, h_{t-1}] مُدمَجين)
        self.W_forget = np.random.randn(concat_dim, hidden_dim) * scale
        self.W_input = np.random.randn(concat_dim, hidden_dim) * scale
        self.W_candidate = np.random.randn(concat_dim, hidden_dim) * scale
        self.W_output = np.random.randn(concat_dim, hidden_dim) * scale

        self.b_forget = np.ones(hidden_dim)  # تهيئة forget bias بـ1 (حيلة مهمة!)
        self.b_input = np.zeros(hidden_dim)
        self.b_candidate = np.zeros(hidden_dim)
        self.b_output = np.zeros(hidden_dim)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward_step(self, x_t: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray):
        """
        أربع بوابات، كل منها بدور محدد بدقة:

        1. Forget Gate (f_t): "كم من الذاكرة القديمة (c_prev) نحتفظ بها؟"
        2. Input Gate (i_t): "كم من المعلومة الجديدة ندخلها للذاكرة؟"
        3. Candidate (c̃_t): "ما هي المعلومة الجديدة المُرشَّحة للإضافة؟"
        4. Output Gate (o_t): "كم من الذاكرة الحالية نُظهِر كحالة مخفية؟"
        """
        combined = np.concatenate([x_t, h_prev])

        forget_gate = self.sigmoid(combined @ self.W_forget + self.b_forget)
        input_gate = self.sigmoid(combined @ self.W_input + self.b_input)
        candidate = np.tanh(combined @ self.W_candidate + self.b_candidate)
        output_gate = self.sigmoid(combined @ self.W_output + self.b_output)

        # تحديث حالة الخلية: مزيج من "نسيان جزء من القديم" و"إضافة جزء من الجديد"
        # هذا التحديث **خطي بشكل أساسي** (جمع + ضرب عنصري، لا tanh متكرر)
        # وهذا بالضبط ما يسمح للتدرج بالتدفق عبر مسافات زمنية طويلة دون تلاشٍ شديد
        c_t = forget_gate * c_prev + input_gate * candidate

        # الحالة المخفية: جزء "مُرشَّح للإظهار" من حالة الخلية الجديدة
        h_t = output_gate * np.tanh(c_t)

        return h_t, c_t, {
            "forget_gate": forget_gate, "input_gate": input_gate,
            "candidate": candidate, "output_gate": output_gate,
        }

    def forward_sequence(self, X: np.ndarray):
        seq_len = X.shape[0]
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)
        hidden_states, cell_states = [], []

        for t in range(seq_len):
            h, c, gates = self.forward_step(X[t], h, c)
            hidden_states.append(h.copy())
            cell_states.append(c.copy())

        return hidden_states, cell_states

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-02-language-representation/ch08-recurrent-networks.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class GRUCellFromScratch:
    """
    تطبيق GRU (Cho et al., 2014) من الصفر: دمج بوابتي Forget وInput
    في بوابة واحدة (Update Gate)، وإزالة حالة الخلية المنفصلة تمامًا
    (الاعتماد فقط على الحالة المخفية h). نتيجة ذلك: معاملات أقل
    بنسبة ~25% من LSTM، وتدريب أسرع، مع أداء مقارب جدًا في أغلب المهام.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        self.hidden_dim = hidden_dim
        concat_dim = input_dim + hidden_dim
        scale = 1.0 / np.sqrt(hidden_dim)

        self.W_update = np.random.randn(concat_dim, hidden_dim) * scale
        self.W_reset = np.random.randn(concat_dim, hidden_dim) * scale
        self.W_candidate = np.random.randn(concat_dim, hidden_dim) * scale

        self.b_update = np.zeros(hidden_dim)
        self.b_reset = np.zeros(hidden_dim)
        self.b_candidate = np.zeros(hidden_dim)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward_step(self, x_t: np.ndarray, h_prev: np.ndarray):
        combined = np.concatenate([x_t, h_prev])

        # Update Gate: يُحدِّد "مزيج" الاحتفاظ بالقديم مقابل الجديد (يدمج دوري forget+input من LSTM)
        update_gate = self.sigmoid(combined @ self.W_update + self.b_update)

        # Reset Gate: يُحدِّد "كم من الماضي نتجاهل" عند حساب المُرشَّح الجديد
        reset_gate = self.sigmoid(combined @ self.W_reset + self.b_reset)

        combined_reset = np.concatenate([x_t, reset_gate * h_prev])
        candidate = np.tanh(combined_reset @ self.W_candidate + self.b_candidate)

        # التحديث النهائي: استيفاء خطي بين الحالة القديمة والمُرشَّح الجديد
        h_t = (1 - update_gate) * h_prev + update_gate * candidate
        return h_t

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-02-language-representation/ch08-recurrent-networks.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class LSTMTextClassifier(nn.Module):
    """
    مُصنِّف نصوص كامل باستخدام LSTM ثنائي الاتجاه (Bidirectional):
    يُعالِج التسلسل من البداية للنهاية ومن النهاية للبداية في آن واحد،
    مما يسمح لكل موضع بالوصول لمعلومات من كلا الاتجاهين — تحسين
    مهم جدًا قبل ظهور Attention، لكنه لا يزال أبطأ وأقل فعالية
    من Self-Attention الكامل (الفصل 10) في التقاط الاعتماديات
    البعيدة جدًا.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 hidden_dim: int = 256, num_classes: int = 2,
                 num_layers: int = 2, dropout: float = 0.3,
                 pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,     # معالجة في الاتجاهين
            dropout=dropout if num_layers > 1 else 0,
        )

        # ×2 لأن bidirectional يُركِّب (concatenate) الاتجاهين معًا
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor = None):
        embedded = self.embedding(input_ids)   # (batch, seq_len, embedding_dim)

        if lengths is not None:
            # Packing: تحسين كفاءة حسابية مهم جدًا عند وجود padding
            # (يتجاهل الحسابات على مواضع الحشو، يُسرّع التدريب بشكل ملحوظ)
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, (h_n, c_n) = self.lstm(packed)
        else:
            _, (h_n, c_n) = self.lstm(embedded)

        # h_n شكلها: (num_layers*2, batch, hidden_dim)
        # نأخذ الحالة المخفية الأخيرة من كلا الاتجاهين للطبقة الأخيرة فقط
        forward_last = h_n[-2, :, :]   # آخر حالة للاتجاه الأمامي
        backward_last = h_n[-1, :, :]  # آخر حالة للاتجاه الخلفي
        combined = torch.cat([forward_last, backward_last], dim=1)

        combined = self.dropout(combined)
        logits = self.fc(combined)
        return logits

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-02-language-representation/ch08-recurrent-networks.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def summarize_rnn_limitations():
    """
    ملخص القيود الجوهرية التي دفعت الباحثين لاختراع Attention
    والمحوّلات بالكامل (سنرى الحل الكامل في الفصل 10).
    """
    limitations = {
        "المعالجة المتسلسلة الإلزامية": (
            "لا يمكن حساب h_t دون معرفة h_{t-1} أولًا، مما يمنع "
            "التوازي الكامل عبر طول التسلسل أثناء التدريب — عنق "
            "زجاجة حاسوبي خطير جدًا لتدريب نماذج كبيرة على GPUs حديثة "
            "مُصمَّمة أساسًا للتوازي الضخم."
        ),
        "صعوبة الاعتماديات طويلة المدى": (
            "رغم تحسّن LSTM/GRU الكبير عن RNN البسيطة، لا تزال المعلومة "
            "تمر عبر 'عنق زجاجة' حالة مخفية ثابتة الحجم، ما يُصعِّب "
            "الاحتفاظ بتفاصيل دقيقة من بداية تسلسل طويل جدًا (آلاف الكلمات)."
        ),
        "صعوبة بناء 'خريطة اعتماديات' مباشرة وصريحة": (
            "لا توجد آلية مباشرة لأي كلمة للوصول مباشرة (بخطوة واحدة) "
            "لأي كلمة أخرى بعيدة في التسلسل — كل المعلومة يجب أن تمر "
            "عبر سلسلة من التحديثات التسلسلية الوسيطة."
        ),
    }

    for problem, explanation in limitations.items():
        print(f"❌ {problem}:\n   {explanation}\n")

summarize_rnn_limitations()

# ────────────────────────────────────────────────────────────────────────
# [007] content/notebooks/part-02-language-representation/ch08-recurrent-networks.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def compare_parameter_counts(input_dim=100, hidden_dim=256):
    # RNN بسيطة: W_xh + W_hh + b_h
    rnn_params = (input_dim * hidden_dim) + (hidden_dim * hidden_dim) + hidden_dim

    # LSTM: 4 بوابات، كل منها بنفس حجم أوزان RNN تقريبًا
    concat_dim = input_dim + hidden_dim
    lstm_params = 4 * (concat_dim * hidden_dim + hidden_dim)

    # GRU: 3 "بوابات" فقط (update, reset, candidate)
    gru_params = 3 * (concat_dim * hidden_dim + hidden_dim)

    print(f"RNN البسيطة:  {rnn_params:,} معامل")
    print(f"LSTM:         {lstm_params:,} معامل ({lstm_params/rnn_params:.1f}× من RNN)")
    print(f"GRU:          {gru_params:,} معامل ({gru_params/rnn_params:.1f}× من RNN، "
          f"{gru_params/lstm_params:.1%} من LSTM)")

compare_parameter_counts()
# RNN البسيطة:  91,392 معامل
# LSTM:         365,568 معامل (4.0× من RNN)
# GRU:          274,176 معامل (3.0× من RNN، 75.0% من LSTM)

# ────────────────────────────────────────────────────────────────────────
# [008] content/notebooks/part-02-language-representation/ch08-recurrent-networks.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-08-ch08-recurrent-networks.ipynb#cell-2
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
    ("numpy", "numpy"),
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-08-ch08-recurrent-networks.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np

class SimpleRNNCell:
    """
    تطبيق خلية RNN واحدة من الصفر، لتوضيح الآلية الأساسية
    قبل الانتقال لتطبيق PyTorch الكامل.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        self.hidden_dim = hidden_dim
        scale = 1.0 / np.sqrt(hidden_dim)
        self.W_xh = np.random.randn(input_dim, hidden_dim) * scale
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * scale
        self.b_h = np.zeros(hidden_dim)

    def forward_step(self, x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        z = x_t @ self.W_xh + h_prev @ self.W_hh + self.b_h
        h_t = np.tanh(z)
        return h_t

    def forward_sequence(self, X: np.ndarray) -> list:
        """معالجة تسلسل كامل، خطوة زمنية تلو الأخرى (معالجة متسلسلة إلزامية!)."""
        seq_len = X.shape[0]
        h = np.zeros(self.hidden_dim)
        hidden_states = []
        for t in range(seq_len):
            h = self.forward_step(X[t], h)
            hidden_states.append(h.copy())
        return hidden_states

def demonstrate_vanishing_gradient_problem():
    """
    توضيح عملي لمشكلة تلاشي التدرجات في RNN: عبر قاعدة السلسلة،
    تدرج الخسارة بالنسبة لحالة مخفية بعيدة زمنيًا هو حاصل ضرب
    سلسلة طويلة من مشتقات tanh (أقصاها 1، وعادة أقل بكثير) مضروبة
    في W_hh تكراريًا. إذا كانت القيم الذاتية لـ W_hh أقل من 1،
    هذا الحاصل يتلاشى أُسّيًا مع طول التسلسل.
    """
    seq_lengths = [5, 20, 50, 100]
    # محاكاة مبسّطة: نفترض متوسط عامل انكماش لكل خطوة زمنية = 0.5
    # (قيمة واقعية جدًا لمشتقة tanh في أغلب نطاقات التشغيل العملية)
    decay_factor_per_step = 0.5

    print("تأثير طول التسلسل على مقدار التدرج المُتراكم (محاكاة):")
    for length in seq_lengths:
        gradient_magnitude = decay_factor_per_step ** length
        print(f"  طول التسلسل={length}: مقدار التدرج ≈ {gradient_magnitude:.2e}")

demonstrate_vanishing_gradient_problem()
# طول التسلسل=5: مقدار التدرج ≈ 3.13e-02
# طول التسلسل=20: مقدار التدرج ≈ 9.54e-07
# طول التسلسل=50: مقدار التدرج ≈ 8.88e-16
# طول التسلسل=100: مقدار التدرج ≈ 7.89e-31   <- تلاشى عمليًا تمامًا!

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-08-ch08-recurrent-networks.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class LSTMCellFromScratch:
    """
    تطبيق كامل لخلية LSTM من الصفر (Hochreiter & Schmidhuber, 1997).
    الفكرة الثورية: فصل "حالة الخلية" (Cell State, C) عن "الحالة
    المخفية" (Hidden State, h). حالة الخلية تعمل كـ"حزام ناقل"
    (conveyor belt) يسمح للمعلومة بالتدفق عبر الزمن مع تدخل خطي
    بسيط فقط (جمع وضرب عنصري)، مما يحل مشكلة تلاشي التدرجات بشكل
    كبير جدًا مقارنة بـ RNN البسيطة.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        self.hidden_dim = hidden_dim
        concat_dim = input_dim + hidden_dim
        scale = 1.0 / np.sqrt(hidden_dim)

        # أربع بوابات، كل منها لها أوزانها الخاصة (تُطبَّق على [x_t, h_{t-1}] مُدمَجين)
        self.W_forget = np.random.randn(concat_dim, hidden_dim) * scale
        self.W_input = np.random.randn(concat_dim, hidden_dim) * scale
        self.W_candidate = np.random.randn(concat_dim, hidden_dim) * scale
        self.W_output = np.random.randn(concat_dim, hidden_dim) * scale

        self.b_forget = np.ones(hidden_dim)  # تهيئة forget bias بـ1 (حيلة مهمة!)
        self.b_input = np.zeros(hidden_dim)
        self.b_candidate = np.zeros(hidden_dim)
        self.b_output = np.zeros(hidden_dim)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward_step(self, x_t: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray):
        """
        أربع بوابات، كل منها بدور محدد بدقة:

        1. Forget Gate (f_t): "كم من الذاكرة القديمة (c_prev) نحتفظ بها؟"
        2. Input Gate (i_t): "كم من المعلومة الجديدة ندخلها للذاكرة؟"
        3. Candidate (c̃_t): "ما هي المعلومة الجديدة المُرشَّحة للإضافة؟"
        4. Output Gate (o_t): "كم من الذاكرة الحالية نُظهِر كحالة مخفية؟"
        """
        combined = np.concatenate([x_t, h_prev])

        forget_gate = self.sigmoid(combined @ self.W_forget + self.b_forget)
        input_gate = self.sigmoid(combined @ self.W_input + self.b_input)
        candidate = np.tanh(combined @ self.W_candidate + self.b_candidate)
        output_gate = self.sigmoid(combined @ self.W_output + self.b_output)

        # تحديث حالة الخلية: مزيج من "نسيان جزء من القديم" و"إضافة جزء من الجديد"
        # هذا التحديث **خطي بشكل أساسي** (جمع + ضرب عنصري، لا tanh متكرر)
        # وهذا بالضبط ما يسمح للتدرج بالتدفق عبر مسافات زمنية طويلة دون تلاشٍ شديد
        c_t = forget_gate * c_prev + input_gate * candidate

        # الحالة المخفية: جزء "مُرشَّح للإظهار" من حالة الخلية الجديدة
        h_t = output_gate * np.tanh(c_t)

        return h_t, c_t, {
            "forget_gate": forget_gate, "input_gate": input_gate,
            "candidate": candidate, "output_gate": output_gate,
        }

    def forward_sequence(self, X: np.ndarray):
        seq_len = X.shape[0]
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)
        hidden_states, cell_states = [], []

        for t in range(seq_len):
            h, c, gates = self.forward_step(X[t], h, c)
            hidden_states.append(h.copy())
            cell_states.append(c.copy())

        return hidden_states, cell_states

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-08-ch08-recurrent-networks.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class GRUCellFromScratch:
    """
    تطبيق GRU (Cho et al., 2014) من الصفر: دمج بوابتي Forget وInput
    في بوابة واحدة (Update Gate)، وإزالة حالة الخلية المنفصلة تمامًا
    (الاعتماد فقط على الحالة المخفية h). نتيجة ذلك: معاملات أقل
    بنسبة ~25% من LSTM، وتدريب أسرع، مع أداء مقارب جدًا في أغلب المهام.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        self.hidden_dim = hidden_dim
        concat_dim = input_dim + hidden_dim
        scale = 1.0 / np.sqrt(hidden_dim)

        self.W_update = np.random.randn(concat_dim, hidden_dim) * scale
        self.W_reset = np.random.randn(concat_dim, hidden_dim) * scale
        self.W_candidate = np.random.randn(concat_dim, hidden_dim) * scale

        self.b_update = np.zeros(hidden_dim)
        self.b_reset = np.zeros(hidden_dim)
        self.b_candidate = np.zeros(hidden_dim)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward_step(self, x_t: np.ndarray, h_prev: np.ndarray):
        combined = np.concatenate([x_t, h_prev])

        # Update Gate: يُحدِّد "مزيج" الاحتفاظ بالقديم مقابل الجديد (يدمج دوري forget+input من LSTM)
        update_gate = self.sigmoid(combined @ self.W_update + self.b_update)

        # Reset Gate: يُحدِّد "كم من الماضي نتجاهل" عند حساب المُرشَّح الجديد
        reset_gate = self.sigmoid(combined @ self.W_reset + self.b_reset)

        combined_reset = np.concatenate([x_t, reset_gate * h_prev])
        candidate = np.tanh(combined_reset @ self.W_candidate + self.b_candidate)

        # التحديث النهائي: استيفاء خطي بين الحالة القديمة والمُرشَّح الجديد
        h_t = (1 - update_gate) * h_prev + update_gate * candidate
        return h_t

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-08-ch08-recurrent-networks.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class LSTMTextClassifier(nn.Module):
    """
    مُصنِّف نصوص كامل باستخدام LSTM ثنائي الاتجاه (Bidirectional):
    يُعالِج التسلسل من البداية للنهاية ومن النهاية للبداية في آن واحد،
    مما يسمح لكل موضع بالوصول لمعلومات من كلا الاتجاهين — تحسين
    مهم جدًا قبل ظهور Attention، لكنه لا يزال أبطأ وأقل فعالية
    من Self-Attention الكامل (الفصل 10) في التقاط الاعتماديات
    البعيدة جدًا.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 hidden_dim: int = 256, num_classes: int = 2,
                 num_layers: int = 2, dropout: float = 0.3,
                 pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,     # معالجة في الاتجاهين
            dropout=dropout if num_layers > 1 else 0,
        )

        # ×2 لأن bidirectional يُركِّب (concatenate) الاتجاهين معًا
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor = None):
        embedded = self.embedding(input_ids)   # (batch, seq_len, embedding_dim)

        if lengths is not None:
            # Packing: تحسين كفاءة حسابية مهم جدًا عند وجود padding
            # (يتجاهل الحسابات على مواضع الحشو، يُسرّع التدريب بشكل ملحوظ)
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, (h_n, c_n) = self.lstm(packed)
        else:
            _, (h_n, c_n) = self.lstm(embedded)

        # h_n شكلها: (num_layers*2, batch, hidden_dim)
        # نأخذ الحالة المخفية الأخيرة من كلا الاتجاهين للطبقة الأخيرة فقط
        forward_last = h_n[-2, :, :]   # آخر حالة للاتجاه الأمامي
        backward_last = h_n[-1, :, :]  # آخر حالة للاتجاه الخلفي
        combined = torch.cat([forward_last, backward_last], dim=1)

        combined = self.dropout(combined)
        logits = self.fc(combined)
        return logits

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-llms-transformers/notebooks/ch-08-ch08-recurrent-networks.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def summarize_rnn_limitations():
    """
    ملخص القيود الجوهرية التي دفعت الباحثين لاختراع Attention
    والمحوّلات بالكامل (سنرى الحل الكامل في الفصل 10).
    """
    limitations = {
        "المعالجة المتسلسلة الإلزامية": (
            "لا يمكن حساب h_t دون معرفة h_{t-1} أولًا، مما يمنع "
            "التوازي الكامل عبر طول التسلسل أثناء التدريب — عنق "
            "زجاجة حاسوبي خطير جدًا لتدريب نماذج كبيرة على GPUs حديثة "
            "مُصمَّمة أساسًا للتوازي الضخم."
        ),
        "صعوبة الاعتماديات طويلة المدى": (
            "رغم تحسّن LSTM/GRU الكبير عن RNN البسيطة، لا تزال المعلومة "
            "تمر عبر 'عنق زجاجة' حالة مخفية ثابتة الحجم، ما يُصعِّب "
            "الاحتفاظ بتفاصيل دقيقة من بداية تسلسل طويل جدًا (آلاف الكلمات)."
        ),
        "صعوبة بناء 'خريطة اعتماديات' مباشرة وصريحة": (
            "لا توجد آلية مباشرة لأي كلمة للوصول مباشرة (بخطوة واحدة) "
            "لأي كلمة أخرى بعيدة في التسلسل — كل المعلومة يجب أن تمر "
            "عبر سلسلة من التحديثات التسلسلية الوسيطة."
        ),
    }

    for problem, explanation in limitations.items():
        print(f"❌ {problem}:\n   {explanation}\n")

summarize_rnn_limitations()

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-llms-transformers/notebooks/ch-08-ch08-recurrent-networks.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
def compare_parameter_counts(input_dim=100, hidden_dim=256):
    # RNN بسيطة: W_xh + W_hh + b_h
    rnn_params = (input_dim * hidden_dim) + (hidden_dim * hidden_dim) + hidden_dim

    # LSTM: 4 بوابات، كل منها بنفس حجم أوزان RNN تقريبًا
    concat_dim = input_dim + hidden_dim
    lstm_params = 4 * (concat_dim * hidden_dim + hidden_dim)

    # GRU: 3 "بوابات" فقط (update, reset, candidate)
    gru_params = 3 * (concat_dim * hidden_dim + hidden_dim)

    print(f"RNN البسيطة:  {rnn_params:,} معامل")
    print(f"LSTM:         {lstm_params:,} معامل ({lstm_params/rnn_params:.1f}× من RNN)")
    print(f"GRU:          {gru_params:,} معامل ({gru_params/rnn_params:.1f}× من RNN، "
          f"{gru_params/lstm_params:.1%} من LSTM)")

compare_parameter_counts()
# RNN البسيطة:  91,392 معامل
# LSTM:         365,568 معامل (4.0× من RNN)
# GRU:          274,176 معامل (3.0× من RNN، 75.0% من LSTM)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/mastering-llms-transformers/notebooks/ch-08-ch08-recurrent-networks.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
