#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — أساسيات تقديم النماذج للاستدلال
Mastering Transformers and Large Language Models — Model Serving

Book slug : mastering-llms-transformers
Chapter   : ch-54-ch54-model-serving (lesson 54)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-54-ch54-model-serving.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — أساسيات تقديم النماذج للاستدلال
Mastering Transformers and Large Language Models — Model Serving

Book slug : mastering-llms-transformers
Chapter   : ch-54-ch54-model-serving (lesson 54)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-54-ch54-model-serving.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — أساسيات تقديم النماذج للاستدلال
Mastering Transformers and Large Language Models — Model Serving

Book slug : mastering-llms-transformers
Chapter   : ch-54-ch54-model-serving (lesson 54)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-11-llmops/ch54-model-serving.ipynb#cell-1
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
    ("requests", "requests"),
    ("torch", "torch"),
    ("vllm", "vllm")
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
# [002] content/notebooks/part-11-llmops/ch54-model-serving.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

class AttentionWithKVCache:
    """
    توضيح مفاهيمي لآلية KV Cache:
    نحتفظ بـ K وV المحسوبة سابقًا، ونضيف فقط K,V الخاصة بالـ token الجديد
    """
    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.o_proj = torch.nn.Linear(d_model, d_model)
        self.cache_k, self.cache_v = None, None   # الذاكرة المخبّأة

    def forward_with_cache(self, x_new_token: torch.Tensor) -> torch.Tensor:
        """x_new_token: [batch, 1, d_model] — رمز واحد جديد فقط!"""
        batch_size = x_new_token.shape[0]

        q = self.q_proj(x_new_token).view(batch_size, 1, self.n_heads, self.head_dim)
        k_new = self.k_proj(x_new_token).view(batch_size, 1, self.n_heads, self.head_dim)
        v_new = self.v_proj(x_new_token).view(batch_size, 1, self.n_heads, self.head_dim)

        if self.cache_k is None:
            self.cache_k, self.cache_v = k_new, v_new
        else:
            # الإضافة فقط - وليس إعادة الحساب الكامل!
            self.cache_k = torch.cat([self.cache_k, k_new], dim=1)
            self.cache_v = torch.cat([self.cache_v, v_new], dim=1)

        # الاستعلام الجديد ينتبه لكل التاريخ المخزّن (القديم + الجديد)
        attn_scores = torch.einsum("bqhd,bkhd->bhqk", q, self.cache_k) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.einsum("bhqk,bkhd->bqhd", attn_weights, self.cache_v)

        return self.o_proj(output.reshape(batch_size, 1, self.d_model))

    def reset_cache(self):
        self.cache_k, self.cache_v = None, None

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-11-llmops/ch54-model-serving.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def calculate_kv_cache_memory_gb(n_layers, n_kv_heads, head_dim, seq_len, batch_size,
                                    dtype_bytes=2) -> float:
    """
    حساب حجم KV Cache بالجيجابايت — ضروري لتخطيط سعة GPU
    العامل 2 لأننا نخزّن K و V منفصلين
    """
    total_bytes = 2 * n_layers * n_kv_heads * head_dim * seq_len * batch_size * dtype_bytes
    return total_bytes / (1024 ** 3)

# مثال: LLaMA-2-7B مع GQA (8 KV heads)، سياق 4096، batch=16
memory_gb = calculate_kv_cache_memory_gb(
    n_layers=32, n_kv_heads=8, head_dim=128, seq_len=4096, batch_size=16
)
print(f"حجم KV Cache المطلوب: {memory_gb:.2f} GB")  # يوضح لماذا GQA (الفصل 14) ضروري جدًا للإنتاج

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-11-llmops/ch54-model-serving.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ==================== استخدام vLLM (أشهر محرك استدلال مفتوح المصدر) ====================
from vllm import LLM, SamplingParams

def setup_vllm_server(model_path: str, tensor_parallel_size=1):
    """
    vLLM يُطبّق تلقائيًا: PagedAttention، Continuous Batching،
    وتحسينات ذاكرة متقدمة تزيد throughput بعشرات الأضعاف مقارنة بـ HuggingFace العادي
    """
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,   # توزيع عبر عدة GPUs
        gpu_memory_utilization=0.9,                   # استغلال 90% من ذاكرة GPU لـ KV cache
        max_model_len=8192,
        dtype="bfloat16",
        enable_prefix_caching=True,                   # تخزين مؤقت لبادئات مشتركة (system prompts)
    )
    return llm

def batch_inference_with_vllm(llm, prompts: list) -> list:
    sampling_params = SamplingParams(
        temperature=0.7, top_p=0.9, max_tokens=512,
        stop=["<|end|>"]
    )
    # vLLM يُجدول كل الطلبات معًا بكفاءة عبر Continuous Batching تلقائيًا
    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]

# ==================== بديل: Text Generation Inference (TGI) من Hugging Face ====================
"""
تشغيل TGI عبر Docker (النهج الإنتاجي الأكثر شيوعًا):

docker run --gpus all --shm-size 1g -p 8080:80 \
    -v $PWD/models:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-2-7b-chat-hf \
    --max-total-tokens 4096 \
    --max-batch-prefill-tokens 8192 \
    --quantize bitsandbytes-nf4
"""

import requests

def query_tgi_server(prompt: str, server_url="http://localhost:8080") -> str:
    response = requests.post(
        f"{server_url}/generate",
        json={
            "inputs": prompt,
            "parameters": {"max_new_tokens": 256, "temperature": 0.7, "top_p": 0.9}
        }
    )
    return response.json()["generated_text"]

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-11-llmops/ch54-model-serving.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-54-ch54-model-serving.ipynb#cell-1
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
    ("requests", "requests"),
    ("torch", "torch"),
    ("vllm", "vllm")
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-54-ch54-model-serving.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

class AttentionWithKVCache:
    """
    توضيح مفاهيمي لآلية KV Cache:
    نحتفظ بـ K وV المحسوبة سابقًا، ونضيف فقط K,V الخاصة بالـ token الجديد
    """
    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.o_proj = torch.nn.Linear(d_model, d_model)
        self.cache_k, self.cache_v = None, None   # الذاكرة المخبّأة

    def forward_with_cache(self, x_new_token: torch.Tensor) -> torch.Tensor:
        """x_new_token: [batch, 1, d_model] — رمز واحد جديد فقط!"""
        batch_size = x_new_token.shape[0]

        q = self.q_proj(x_new_token).view(batch_size, 1, self.n_heads, self.head_dim)
        k_new = self.k_proj(x_new_token).view(batch_size, 1, self.n_heads, self.head_dim)
        v_new = self.v_proj(x_new_token).view(batch_size, 1, self.n_heads, self.head_dim)

        if self.cache_k is None:
            self.cache_k, self.cache_v = k_new, v_new
        else:
            # الإضافة فقط - وليس إعادة الحساب الكامل!
            self.cache_k = torch.cat([self.cache_k, k_new], dim=1)
            self.cache_v = torch.cat([self.cache_v, v_new], dim=1)

        # الاستعلام الجديد ينتبه لكل التاريخ المخزّن (القديم + الجديد)
        attn_scores = torch.einsum("bqhd,bkhd->bhqk", q, self.cache_k) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.einsum("bhqk,bkhd->bqhd", attn_weights, self.cache_v)

        return self.o_proj(output.reshape(batch_size, 1, self.d_model))

    def reset_cache(self):
        self.cache_k, self.cache_v = None, None

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-54-ch54-model-serving.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def calculate_kv_cache_memory_gb(n_layers, n_kv_heads, head_dim, seq_len, batch_size,
                                    dtype_bytes=2) -> float:
    """
    حساب حجم KV Cache بالجيجابايت — ضروري لتخطيط سعة GPU
    العامل 2 لأننا نخزّن K و V منفصلين
    """
    total_bytes = 2 * n_layers * n_kv_heads * head_dim * seq_len * batch_size * dtype_bytes
    return total_bytes / (1024 ** 3)

# مثال: LLaMA-2-7B مع GQA (8 KV heads)، سياق 4096، batch=16
memory_gb = calculate_kv_cache_memory_gb(
    n_layers=32, n_kv_heads=8, head_dim=128, seq_len=4096, batch_size=16
)
print(f"حجم KV Cache المطلوب: {memory_gb:.2f} GB")  # يوضح لماذا GQA (الفصل 14) ضروري جدًا للإنتاج

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-54-ch54-model-serving.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ==================== استخدام vLLM (أشهر محرك استدلال مفتوح المصدر) ====================
from vllm import LLM, SamplingParams

def setup_vllm_server(model_path: str, tensor_parallel_size=1):
    """
    vLLM يُطبّق تلقائيًا: PagedAttention، Continuous Batching،
    وتحسينات ذاكرة متقدمة تزيد throughput بعشرات الأضعاف مقارنة بـ HuggingFace العادي
    """
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,   # توزيع عبر عدة GPUs
        gpu_memory_utilization=0.9,                   # استغلال 90% من ذاكرة GPU لـ KV cache
        max_model_len=8192,
        dtype="bfloat16",
        enable_prefix_caching=True,                   # تخزين مؤقت لبادئات مشتركة (system prompts)
    )
    return llm

def batch_inference_with_vllm(llm, prompts: list) -> list:
    sampling_params = SamplingParams(
        temperature=0.7, top_p=0.9, max_tokens=512,
        stop=["<|end|>"]
    )
    # vLLM يُجدول كل الطلبات معًا بكفاءة عبر Continuous Batching تلقائيًا
    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]

# ==================== بديل: Text Generation Inference (TGI) من Hugging Face ====================
"""
تشغيل TGI عبر Docker (النهج الإنتاجي الأكثر شيوعًا):

docker run --gpus all --shm-size 1g -p 8080:80 \
    -v $PWD/models:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-2-7b-chat-hf \
    --max-total-tokens 4096 \
    --max-batch-prefill-tokens 8192 \
    --quantize bitsandbytes-nf4
"""

import requests

def query_tgi_server(prompt: str, server_url="http://localhost:8080") -> str:
    response = requests.post(
        f"{server_url}/generate",
        json={
            "inputs": prompt,
            "parameters": {"max_new_tokens": 256, "temperature": 0.7, "top_p": 0.9}
        }
    )
    return response.json()["generated_text"]

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-54-ch54-model-serving.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-54-ch54-model-serving.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — أساسيات تقديم النماذج للاستدلال
Mastering Transformers and Large Language Models — Model Serving

Book slug : mastering-llms-transformers
Chapter   : ch-54-ch54-model-serving (lesson 54)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-11-llmops/ch54-model-serving.ipynb#cell-1
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
    ("requests", "requests"),
    ("torch", "torch"),
    ("vllm", "vllm")
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
# [002] content/notebooks/part-11-llmops/ch54-model-serving.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

class AttentionWithKVCache:
    """
    توضيح مفاهيمي لآلية KV Cache:
    نحتفظ بـ K وV المحسوبة سابقًا، ونضيف فقط K,V الخاصة بالـ token الجديد
    """
    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.o_proj = torch.nn.Linear(d_model, d_model)
        self.cache_k, self.cache_v = None, None   # الذاكرة المخبّأة

    def forward_with_cache(self, x_new_token: torch.Tensor) -> torch.Tensor:
        """x_new_token: [batch, 1, d_model] — رمز واحد جديد فقط!"""
        batch_size = x_new_token.shape[0]

        q = self.q_proj(x_new_token).view(batch_size, 1, self.n_heads, self.head_dim)
        k_new = self.k_proj(x_new_token).view(batch_size, 1, self.n_heads, self.head_dim)
        v_new = self.v_proj(x_new_token).view(batch_size, 1, self.n_heads, self.head_dim)

        if self.cache_k is None:
            self.cache_k, self.cache_v = k_new, v_new
        else:
            # الإضافة فقط - وليس إعادة الحساب الكامل!
            self.cache_k = torch.cat([self.cache_k, k_new], dim=1)
            self.cache_v = torch.cat([self.cache_v, v_new], dim=1)

        # الاستعلام الجديد ينتبه لكل التاريخ المخزّن (القديم + الجديد)
        attn_scores = torch.einsum("bqhd,bkhd->bhqk", q, self.cache_k) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.einsum("bhqk,bkhd->bqhd", attn_weights, self.cache_v)

        return self.o_proj(output.reshape(batch_size, 1, self.d_model))

    def reset_cache(self):
        self.cache_k, self.cache_v = None, None

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-11-llmops/ch54-model-serving.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def calculate_kv_cache_memory_gb(n_layers, n_kv_heads, head_dim, seq_len, batch_size,
                                    dtype_bytes=2) -> float:
    """
    حساب حجم KV Cache بالجيجابايت — ضروري لتخطيط سعة GPU
    العامل 2 لأننا نخزّن K و V منفصلين
    """
    total_bytes = 2 * n_layers * n_kv_heads * head_dim * seq_len * batch_size * dtype_bytes
    return total_bytes / (1024 ** 3)

# مثال: LLaMA-2-7B مع GQA (8 KV heads)، سياق 4096، batch=16
memory_gb = calculate_kv_cache_memory_gb(
    n_layers=32, n_kv_heads=8, head_dim=128, seq_len=4096, batch_size=16
)
print(f"حجم KV Cache المطلوب: {memory_gb:.2f} GB")  # يوضح لماذا GQA (الفصل 14) ضروري جدًا للإنتاج

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-11-llmops/ch54-model-serving.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ==================== استخدام vLLM (أشهر محرك استدلال مفتوح المصدر) ====================
from vllm import LLM, SamplingParams

def setup_vllm_server(model_path: str, tensor_parallel_size=1):
    """
    vLLM يُطبّق تلقائيًا: PagedAttention، Continuous Batching،
    وتحسينات ذاكرة متقدمة تزيد throughput بعشرات الأضعاف مقارنة بـ HuggingFace العادي
    """
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,   # توزيع عبر عدة GPUs
        gpu_memory_utilization=0.9,                   # استغلال 90% من ذاكرة GPU لـ KV cache
        max_model_len=8192,
        dtype="bfloat16",
        enable_prefix_caching=True,                   # تخزين مؤقت لبادئات مشتركة (system prompts)
    )
    return llm

def batch_inference_with_vllm(llm, prompts: list) -> list:
    sampling_params = SamplingParams(
        temperature=0.7, top_p=0.9, max_tokens=512,
        stop=["<|end|>"]
    )
    # vLLM يُجدول كل الطلبات معًا بكفاءة عبر Continuous Batching تلقائيًا
    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]

# ==================== بديل: Text Generation Inference (TGI) من Hugging Face ====================
"""
تشغيل TGI عبر Docker (النهج الإنتاجي الأكثر شيوعًا):

docker run --gpus all --shm-size 1g -p 8080:80 \
    -v $PWD/models:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-2-7b-chat-hf \
    --max-total-tokens 4096 \
    --max-batch-prefill-tokens 8192 \
    --quantize bitsandbytes-nf4
"""

import requests

def query_tgi_server(prompt: str, server_url="http://localhost:8080") -> str:
    response = requests.post(
        f"{server_url}/generate",
        json={
            "inputs": prompt,
            "parameters": {"max_new_tokens": 256, "temperature": 0.7, "top_p": 0.9}
        }
    )
    return response.json()["generated_text"]

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-11-llmops/ch54-model-serving.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-54-ch54-model-serving.ipynb#cell-2
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
    ("requests", "requests"),
    ("torch", "torch"),
    ("vllm", "vllm")
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-54-ch54-model-serving.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

class AttentionWithKVCache:
    """
    توضيح مفاهيمي لآلية KV Cache:
    نحتفظ بـ K وV المحسوبة سابقًا، ونضيف فقط K,V الخاصة بالـ token الجديد
    """
    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.o_proj = torch.nn.Linear(d_model, d_model)
        self.cache_k, self.cache_v = None, None   # الذاكرة المخبّأة

    def forward_with_cache(self, x_new_token: torch.Tensor) -> torch.Tensor:
        """x_new_token: [batch, 1, d_model] — رمز واحد جديد فقط!"""
        batch_size = x_new_token.shape[0]

        q = self.q_proj(x_new_token).view(batch_size, 1, self.n_heads, self.head_dim)
        k_new = self.k_proj(x_new_token).view(batch_size, 1, self.n_heads, self.head_dim)
        v_new = self.v_proj(x_new_token).view(batch_size, 1, self.n_heads, self.head_dim)

        if self.cache_k is None:
            self.cache_k, self.cache_v = k_new, v_new
        else:
            # الإضافة فقط - وليس إعادة الحساب الكامل!
            self.cache_k = torch.cat([self.cache_k, k_new], dim=1)
            self.cache_v = torch.cat([self.cache_v, v_new], dim=1)

        # الاستعلام الجديد ينتبه لكل التاريخ المخزّن (القديم + الجديد)
        attn_scores = torch.einsum("bqhd,bkhd->bhqk", q, self.cache_k) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.einsum("bhqk,bkhd->bqhd", attn_weights, self.cache_v)

        return self.o_proj(output.reshape(batch_size, 1, self.d_model))

    def reset_cache(self):
        self.cache_k, self.cache_v = None, None

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-54-ch54-model-serving.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def calculate_kv_cache_memory_gb(n_layers, n_kv_heads, head_dim, seq_len, batch_size,
                                    dtype_bytes=2) -> float:
    """
    حساب حجم KV Cache بالجيجابايت — ضروري لتخطيط سعة GPU
    العامل 2 لأننا نخزّن K و V منفصلين
    """
    total_bytes = 2 * n_layers * n_kv_heads * head_dim * seq_len * batch_size * dtype_bytes
    return total_bytes / (1024 ** 3)

# مثال: LLaMA-2-7B مع GQA (8 KV heads)، سياق 4096، batch=16
memory_gb = calculate_kv_cache_memory_gb(
    n_layers=32, n_kv_heads=8, head_dim=128, seq_len=4096, batch_size=16
)
print(f"حجم KV Cache المطلوب: {memory_gb:.2f} GB")  # يوضح لماذا GQA (الفصل 14) ضروري جدًا للإنتاج

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-54-ch54-model-serving.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ==================== استخدام vLLM (أشهر محرك استدلال مفتوح المصدر) ====================
from vllm import LLM, SamplingParams

def setup_vllm_server(model_path: str, tensor_parallel_size=1):
    """
    vLLM يُطبّق تلقائيًا: PagedAttention، Continuous Batching،
    وتحسينات ذاكرة متقدمة تزيد throughput بعشرات الأضعاف مقارنة بـ HuggingFace العادي
    """
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,   # توزيع عبر عدة GPUs
        gpu_memory_utilization=0.9,                   # استغلال 90% من ذاكرة GPU لـ KV cache
        max_model_len=8192,
        dtype="bfloat16",
        enable_prefix_caching=True,                   # تخزين مؤقت لبادئات مشتركة (system prompts)
    )
    return llm

def batch_inference_with_vllm(llm, prompts: list) -> list:
    sampling_params = SamplingParams(
        temperature=0.7, top_p=0.9, max_tokens=512,
        stop=["<|end|>"]
    )
    # vLLM يُجدول كل الطلبات معًا بكفاءة عبر Continuous Batching تلقائيًا
    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]

# ==================== بديل: Text Generation Inference (TGI) من Hugging Face ====================
"""
تشغيل TGI عبر Docker (النهج الإنتاجي الأكثر شيوعًا):

docker run --gpus all --shm-size 1g -p 8080:80 \
    -v $PWD/models:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-2-7b-chat-hf \
    --max-total-tokens 4096 \
    --max-batch-prefill-tokens 8192 \
    --quantize bitsandbytes-nf4
"""

import requests

def query_tgi_server(prompt: str, server_url="http://localhost:8080") -> str:
    response = requests.post(
        f"{server_url}/generate",
        json={
            "inputs": prompt,
            "parameters": {"max_new_tokens": 256, "temperature": 0.7, "top_p": 0.9}
        }
    )
    return response.json()["generated_text"]

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-54-ch54-model-serving.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
