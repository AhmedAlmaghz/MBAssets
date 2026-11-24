#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — خوادم الاستدلال — vLLM وTGI وTensorRT-LLM وllama.cpp
Large Language Model Engineering — vLLM

Book slug : llm-engineering-69-chapters
Chapter   : ch-60-chapter-60 (lesson 60)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-60-chapter-60.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — خوادم الاستدلال — vLLM وTGI وTensorRT-LLM وllama.cpp
Large Language Model Engineering — vLLM

Book slug : llm-engineering-69-chapters
Chapter   : ch-60-chapter-60 (lesson 60)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-60-chapter-60.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — خوادم الاستدلال — vLLM وTGI وTensorRT-LLM وllama.cpp
Large Language Model Engineering — vLLM

Book slug : llm-engineering-69-chapters
Chapter   : ch-60-chapter-60 (lesson 60)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_60.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 60: الفصل 60: خوادم الاستدلال — vLLM وTGI وTensorRT-LLM وllama.cpp."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_serving() -> None:
    arrivals = [0, 1, 2, 8, 9, 10, 20]
    batch_window, max_batch = (3, 3)
    batches, current = ([], [])
    for arrival in arrivals:
        if current and (arrival - current[0] > batch_window or len(current) == max_batch):
            batches.append(current)
            current = []
        current.append(arrival)
    if current:
        batches.append(current)
    print('arrival times:', arrivals)
    print('formed batches:', batches)
    print('throughput (items per batch):', [len(batch) for batch in batches])

if __name__ == "__main__":
    print("Chapter 60 lab: Simulate batching and server throughput")
    run_serving()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-11-deployment-llmops/chapter-60.md
# ────────────────────────────────────────────────────────────────────────
# vllm_server_setup.py
"""
نشر vLLM كخادم كامل متوافق مع OpenAI API — يمكن تشغيله بسطر واحد من CLI أيضًا:
    vllm serve meta-llama/Llama-3.1-8B-Instruct --quantization awq --max-model-len 4096
لكن هنا نوضح الإعداد البرمجي الكامل للتحكم الدقيق بكل معامل
"""

from vllm import LLM, SamplingParams
from vllm.entrypoints.openai.api_server import run_server
import argparse


def build_vllm_engine(model_path: str, quantization: str = "awq", tensor_parallel_size: int = 1):
    """
    إعداد محرك vLLM مع كل التحسينات التي تعلمناها في الفصل الماضي مُفعَّلة ضمنيًا:
    - PagedAttention: مُفعَّل افتراضيًا دائمًا (جوهر التصميم)
    - Continuous Batching: مُفعَّل افتراضيًا دائمًا
    - Quantization: نحدده صراحة هنا (awq / gptq / None)
    """
    llm = LLM(
        model=model_path,
        quantization=quantization,              # "awq", "gptq", أو None للنموذج الأصلي
        tensor_parallel_size=tensor_parallel_size,  # توزيع على عدة GPUs إذا لزم
        gpu_memory_utilization=0.90,             # نسبة ذاكرة GPU المحجوزة لـ KV-Cache
        max_model_len=4096,                      # أقصى طول سياق مدعوم
        enforce_eager=False,                     # False = يسمح بـ CUDA graphs لتسريع إضافي
    )
    return llm


def run_batch_inference_example(llm: LLM):
    """مثال على استدلال دفعي مباشر (بدون خادم HTTP) — مفيد لمعالجة offline ضخمة"""
    prompts = [
        "اشرح مفهوم PagedAttention في جملتين.",
        "ما الفرق بين vLLM وTGI؟",
        "اكتب دالة Python لحساب متوسط قائمة أرقام.",
    ]

    sampling_params = SamplingParams(
        temperature=0.7, top_p=0.9, max_tokens=200,
    )

    # هذا النداء الواحد يستفيد تلقائيًا من Continuous Batching لمعالجة الثلاثة بالتوازي
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        print(f"\nالسؤال: {output.prompt}")
        print(f"الإجابة: {output.outputs[0].text}")
        print(f"عدد التوكنات المُولَّدة: {len(output.outputs[0].token_ids)}")


if __name__ == "__main__":
    # للتشغيل كخادم HTTP كامل متوافق مع OpenAI، استخدم بدلًا من ذلك من سطر الأوامر:
    #   python -m vllm.entrypoints.openai.api_server \
    #       --model meta-llama/Llama-3.1-8B-Instruct \
    #       --quantization awq --tensor-parallel-size 1 --port 8000

    engine = build_vllm_engine("meta-llama/Llama-3.1-8B-Instruct", quantization="awq")
    run_batch_inference_example(engine)

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-11-deployment-llmops/chapter-60.md
# ────────────────────────────────────────────────────────────────────────
# client_vllm.py
"""بمجرد تشغيل خادم vLLM (عبر الأمر CLI أعلاه)، يصبح استدعاؤه مطابقًا تمامًا لاستدعاء OpenAI"""

from openai import OpenAI

# النقطة الأساسية: نفس عميل OpenAI، فقط نغيّر base_url للإشارة لخادمنا المحلي
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed-for-local-server",  # vLLM لا يتطلب مفتاحًا حقيقيًا افتراضيًا
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "اشرح لي PagedAttention بجملة واحدة بسيطة."}],
    temperature=0.5, max_tokens=150,
)

print(response.choices[0].message.content)
print(f"\nتوكنات مُستخدمة: {response.usage.total_tokens}")

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-11-deployment-llmops/chapter-60.md
# ────────────────────────────────────────────────────────────────────────
# client_tgi.py
"""TGI يوفر واجهته الخاصة (text-generation API) بالإضافة لواجهة OpenAI-compatible"""

from huggingface_hub import InferenceClient

client = InferenceClient(model="http://localhost:8080")

response = client.text_generation(
    prompt="اشرح لي الفرق بين TGI وvLLM بإيجاز.",
    max_new_tokens=150,
    temperature=0.5,
    stream=False,
)
print(response)

# ─── أو عبر واجهة OpenAI-compatible المدمجة في TGI الحديث ───
from openai import OpenAI

client_openai_style = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
response2 = client_openai_style.chat.completions.create(
    model="tgi",
    messages=[{"role": "user", "content": "اشرح PagedAttention بجملة واحدة."}],
    max_tokens=100,
)
print(response2.choices[0].message.content)

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-11-deployment-llmops/chapter-60.md
# ────────────────────────────────────────────────────────────────────────
# llama_cpp_setup.py
"""
llama.cpp عبر مكتبة llama-cpp-python — الخيار الأمثل للتشغيل المحلي
بدون الحاجة لـ GPU قوي، أو حتى بدون GPU إطلاقًا
"""

from llama_cpp import Llama

# تحميل نموذج بصيغة GGUF مُكمَّم مسبقًا (يمكن تحميله من Hugging Face Hub مباشرة)
llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
    filename="*q4_k_m.gguf",       # مستوى التكميم: Q4_K_M = توازن جيد بين الحجم والجودة
    n_gpu_layers=-1,                # -1 = حمّل كل الطبقات الممكنة على GPU إن توفر، 0 = CPU بالكامل
    n_ctx=4096,                     # طول السياق الأقصى
    n_threads=8,                    # عدد أنوية CPU المستخدمة (مهم جدًا عند التشغيل بدون GPU)
    verbose=False,
)

def chat_with_local_model(prompt: str) -> str:
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200, temperature=0.6,
    )
    return response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    result = chat_with_local_model("ما ميزة llama.cpp الأساسية مقارنة بـ vLLM؟")
    print(result)

    # llama.cpp يدعم أيضًا تشغيل خادم HTTP متوافق مع OpenAI مباشرة من CLI:
    #   python -m llama_cpp.server --model model.gguf --n_gpu_layers -1 --port 8000

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-11-deployment-llmops/chapter-60.md
# ────────────────────────────────────────────────────────────────────────
# benchmark_inference_servers.py
"""
نظام قياس موحّد يقيس Throughput وTTFT وTPOT عبر أي خادم متوافق مع OpenAI API
(يعمل مع vLLM, TGI, llama.cpp server بنفس الكود تمامًا — هذا بالضبط جمال المعيار الموحّد)
"""

import time
import asyncio
import statistics
from dataclasses import dataclass, field
from typing import List
from openai import AsyncOpenAI


@dataclass
class BenchmarkResult:
    server_name: str
    ttft_list: List[float] = field(default_factory=list)      # Time To First Token لكل طلب
    total_time_list: List[float] = field(default_factory=list)
    tokens_generated_list: List[int] = field(default_factory=list)

    def summary(self) -> dict:
        total_tokens = sum(self.tokens_generated_list)
        total_time = max(self.total_time_list) if self.total_time_list else 1
        return {
            "server": self.server_name,
            "avg_ttft_ms": round(statistics.mean(self.ttft_list) * 1000, 1) if self.ttft_list else 0,
            "p95_ttft_ms": round(statistics.quantiles(self.ttft_list, n=20)[18] * 1000, 1) if len(self.ttft_list) > 1 else 0,
            "throughput_tokens_per_sec": round(total_tokens / total_time, 1),
            "total_requests": len(self.total_time_list),
        }


async def single_streaming_request(client: AsyncOpenAI, model: str, prompt: str) -> dict:
    """نقيس TTFT عبر وضع streaming: الزمن من الإرسال حتى أول chunk فعلي"""
    start = time.time()
    first_token_time = None
    token_count = 0

    stream = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        max_tokens=150, temperature=0.5, stream=True,
    )

    async for chunk in stream:
        if chunk.choices[0].delta.content:
            if first_token_time is None:
                first_token_time = time.time()
            token_count += 1

    total_time = time.time() - start
    ttft = (first_token_time - start) if first_token_time else total_time

    return {"ttft": ttft, "total_time": total_time, "tokens": token_count}


async def run_concurrent_benchmark(base_url: str, model: str, server_name: str,
                                     n_concurrent: int = 10, prompts: List[str] = None) -> BenchmarkResult:
    """يُشغّل n_concurrent طلبًا متزامنًا — هذا بالضبط ما يكشف الفرق الحقيقي بين Continuous
    Batching (vLLM/TGI) وMicro-batching البسيط (llama.cpp الافتراضي)"""
    client = AsyncOpenAI(base_url=base_url, api_key="not-needed")
    prompts = prompts or ["اشرح مفهوم تقني بسيط بجملتين."] * n_concurrent

    result = BenchmarkResult(server_name=server_name)
    tasks = [single_streaming_request(client, model, p) for p in prompts]
    responses = await asyncio.gather(*tasks)

    for r in responses:
        result.ttft_list.append(r["ttft"])
        result.total_time_list.append(r["total_time"])
        result.tokens_generated_list.append(r["tokens"])

    return result


async def compare_all_servers():
    """شغّل هذا بعد تشغيل كل خادم على منفذ منفصل محليًا"""
    configs = [
        {"base_url": "http://localhost:8000/v1", "model": "meta-llama/Llama-3.1-8B-Instruct", "name": "vLLM"},
        {"base_url": "http://localhost:8080/v1", "model": "tgi", "name": "TGI"},
        {"base_url": "http://localhost:8001/v1", "model": "local-model", "name": "llama.cpp"},
    ]

    print(f"{'الخادم':<12} | {'متوسط TTFT (ms)':<18} | {'P95 TTFT (ms)':<15} | {'Throughput (tok/s)':<20}")
    print("-" * 75)

    for config in configs:
        try:
            result = await run_concurrent_benchmark(
                config["base_url"], config["model"], config["name"], n_concurrent=10
            )
            s = result.summary()
            print(f"{s['server']:<12} | {s['avg_ttft_ms']:<18} | {s['p95_ttft_ms']:<15} | {s['throughput_tokens_per_sec']:<20}")
        except Exception as e:
            print(f"{config['name']:<12} | فشل الاتصال: {e}")


if __name__ == "__main__":
    asyncio.run(compare_all_servers())

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-60-chapter-60.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 60: الفصل 60: خوادم الاستدلال — vLLM وTGI وTensorRT-LLM وllama.cpp."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_serving() -> None:
    arrivals = [0, 1, 2, 8, 9, 10, 20]
    batch_window, max_batch = (3, 3)
    batches, current = ([], [])
    for arrival in arrivals:
        if current and (arrival - current[0] > batch_window or len(current) == max_batch):
            batches.append(current)
            current = []
        current.append(arrival)
    if current:
        batches.append(current)
    print('arrival times:', arrivals)
    print('formed batches:', batches)
    print('throughput (items per batch):', [len(batch) for batch in batches])

if __name__ == "__main__":
    print("Chapter 60 lab: Simulate batching and server throughput")
    run_serving()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-60-chapter-60.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# vllm_server_setup.py
"""
نشر vLLM كخادم كامل متوافق مع OpenAI API — يمكن تشغيله بسطر واحد من CLI أيضًا:
    vllm serve meta-llama/Llama-3.1-8B-Instruct --quantization awq --max-model-len 4096
لكن هنا نوضح الإعداد البرمجي الكامل للتحكم الدقيق بكل معامل
"""

from vllm import LLM, SamplingParams
from vllm.entrypoints.openai.api_server import run_server
import argparse


def build_vllm_engine(model_path: str, quantization: str = "awq", tensor_parallel_size: int = 1):
    """
    إعداد محرك vLLM مع كل التحسينات التي تعلمناها في الفصل الماضي مُفعَّلة ضمنيًا:
    - PagedAttention: مُفعَّل افتراضيًا دائمًا (جوهر التصميم)
    - Continuous Batching: مُفعَّل افتراضيًا دائمًا
    - Quantization: نحدده صراحة هنا (awq / gptq / None)
    """
    llm = LLM(
        model=model_path,
        quantization=quantization,              # "awq", "gptq", أو None للنموذج الأصلي
        tensor_parallel_size=tensor_parallel_size,  # توزيع على عدة GPUs إذا لزم
        gpu_memory_utilization=0.90,             # نسبة ذاكرة GPU المحجوزة لـ KV-Cache
        max_model_len=4096,                      # أقصى طول سياق مدعوم
        enforce_eager=False,                     # False = يسمح بـ CUDA graphs لتسريع إضافي
    )
    return llm


def run_batch_inference_example(llm: LLM):
    """مثال على استدلال دفعي مباشر (بدون خادم HTTP) — مفيد لمعالجة offline ضخمة"""
    prompts = [
        "اشرح مفهوم PagedAttention في جملتين.",
        "ما الفرق بين vLLM وTGI؟",
        "اكتب دالة Python لحساب متوسط قائمة أرقام.",
    ]

    sampling_params = SamplingParams(
        temperature=0.7, top_p=0.9, max_tokens=200,
    )

    # هذا النداء الواحد يستفيد تلقائيًا من Continuous Batching لمعالجة الثلاثة بالتوازي
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        print(f"\nالسؤال: {output.prompt}")
        print(f"الإجابة: {output.outputs[0].text}")
        print(f"عدد التوكنات المُولَّدة: {len(output.outputs[0].token_ids)}")


if __name__ == "__main__":
    # للتشغيل كخادم HTTP كامل متوافق مع OpenAI، استخدم بدلًا من ذلك من سطر الأوامر:
    #   python -m vllm.entrypoints.openai.api_server \
    #       --model meta-llama/Llama-3.1-8B-Instruct \
    #       --quantization awq --tensor-parallel-size 1 --port 8000

    engine = build_vllm_engine("meta-llama/Llama-3.1-8B-Instruct", quantization="awq")
    run_batch_inference_example(engine)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-60-chapter-60.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# client_vllm.py
"""بمجرد تشغيل خادم vLLM (عبر الأمر CLI أعلاه)، يصبح استدعاؤه مطابقًا تمامًا لاستدعاء OpenAI"""

from openai import OpenAI

# النقطة الأساسية: نفس عميل OpenAI، فقط نغيّر base_url للإشارة لخادمنا المحلي
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed-for-local-server",  # vLLM لا يتطلب مفتاحًا حقيقيًا افتراضيًا
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "اشرح لي PagedAttention بجملة واحدة بسيطة."}],
    temperature=0.5, max_tokens=150,
)

print(response.choices[0].message.content)
print(f"\nتوكنات مُستخدمة: {response.usage.total_tokens}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-60-chapter-60.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# client_tgi.py
"""TGI يوفر واجهته الخاصة (text-generation API) بالإضافة لواجهة OpenAI-compatible"""

from huggingface_hub import InferenceClient

client = InferenceClient(model="http://localhost:8080")

response = client.text_generation(
    prompt="اشرح لي الفرق بين TGI وvLLM بإيجاز.",
    max_new_tokens=150,
    temperature=0.5,
    stream=False,
)
print(response)

# ─── أو عبر واجهة OpenAI-compatible المدمجة في TGI الحديث ───
from openai import OpenAI

client_openai_style = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
response2 = client_openai_style.chat.completions.create(
    model="tgi",
    messages=[{"role": "user", "content": "اشرح PagedAttention بجملة واحدة."}],
    max_tokens=100,
)
print(response2.choices[0].message.content)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-60-chapter-60.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# llama_cpp_setup.py
"""
llama.cpp عبر مكتبة llama-cpp-python — الخيار الأمثل للتشغيل المحلي
بدون الحاجة لـ GPU قوي، أو حتى بدون GPU إطلاقًا
"""

from llama_cpp import Llama

# تحميل نموذج بصيغة GGUF مُكمَّم مسبقًا (يمكن تحميله من Hugging Face Hub مباشرة)
llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
    filename="*q4_k_m.gguf",       # مستوى التكميم: Q4_K_M = توازن جيد بين الحجم والجودة
    n_gpu_layers=-1,                # -1 = حمّل كل الطبقات الممكنة على GPU إن توفر، 0 = CPU بالكامل
    n_ctx=4096,                     # طول السياق الأقصى
    n_threads=8,                    # عدد أنوية CPU المستخدمة (مهم جدًا عند التشغيل بدون GPU)
    verbose=False,
)

def chat_with_local_model(prompt: str) -> str:
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200, temperature=0.6,
    )
    return response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    result = chat_with_local_model("ما ميزة llama.cpp الأساسية مقارنة بـ vLLM؟")
    print(result)

    # llama.cpp يدعم أيضًا تشغيل خادم HTTP متوافق مع OpenAI مباشرة من CLI:
    #   python -m llama_cpp.server --model model.gguf --n_gpu_layers -1 --port 8000

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-60-chapter-60.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# benchmark_inference_servers.py
"""
نظام قياس موحّد يقيس Throughput وTTFT وTPOT عبر أي خادم متوافق مع OpenAI API
(يعمل مع vLLM, TGI, llama.cpp server بنفس الكود تمامًا — هذا بالضبط جمال المعيار الموحّد)
"""

import time
import asyncio
import statistics
from dataclasses import dataclass, field
from typing import List
from openai import AsyncOpenAI


@dataclass
class BenchmarkResult:
    server_name: str
    ttft_list: List[float] = field(default_factory=list)      # Time To First Token لكل طلب
    total_time_list: List[float] = field(default_factory=list)
    tokens_generated_list: List[int] = field(default_factory=list)

    def summary(self) -> dict:
        total_tokens = sum(self.tokens_generated_list)
        total_time = max(self.total_time_list) if self.total_time_list else 1
        return {
            "server": self.server_name,
            "avg_ttft_ms": round(statistics.mean(self.ttft_list) * 1000, 1) if self.ttft_list else 0,
            "p95_ttft_ms": round(statistics.quantiles(self.ttft_list, n=20)[18] * 1000, 1) if len(self.ttft_list) > 1 else 0,
            "throughput_tokens_per_sec": round(total_tokens / total_time, 1),
            "total_requests": len(self.total_time_list),
        }


async def single_streaming_request(client: AsyncOpenAI, model: str, prompt: str) -> dict:
    """نقيس TTFT عبر وضع streaming: الزمن من الإرسال حتى أول chunk فعلي"""
    start = time.time()
    first_token_time = None
    token_count = 0

    stream = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        max_tokens=150, temperature=0.5, stream=True,
    )

    async for chunk in stream:
        if chunk.choices[0].delta.content:
            if first_token_time is None:
                first_token_time = time.time()
            token_count += 1

    total_time = time.time() - start
    ttft = (first_token_time - start) if first_token_time else total_time

    return {"ttft": ttft, "total_time": total_time, "tokens": token_count}


async def run_concurrent_benchmark(base_url: str, model: str, server_name: str,
                                     n_concurrent: int = 10, prompts: List[str] = None) -> BenchmarkResult:
    """يُشغّل n_concurrent طلبًا متزامنًا — هذا بالضبط ما يكشف الفرق الحقيقي بين Continuous
    Batching (vLLM/TGI) وMicro-batching البسيط (llama.cpp الافتراضي)"""
    client = AsyncOpenAI(base_url=base_url, api_key="not-needed")
    prompts = prompts or ["اشرح مفهوم تقني بسيط بجملتين."] * n_concurrent

    result = BenchmarkResult(server_name=server_name)
    tasks = [single_streaming_request(client, model, p) for p in prompts]
    responses = await asyncio.gather(*tasks)

    for r in responses:
        result.ttft_list.append(r["ttft"])
        result.total_time_list.append(r["total_time"])
        result.tokens_generated_list.append(r["tokens"])

    return result


async def compare_all_servers():
    """شغّل هذا بعد تشغيل كل خادم على منفذ منفصل محليًا"""
    configs = [
        {"base_url": "http://localhost:8000/v1", "model": "meta-llama/Llama-3.1-8B-Instruct", "name": "vLLM"},
        {"base_url": "http://localhost:8080/v1", "model": "tgi", "name": "TGI"},
        {"base_url": "http://localhost:8001/v1", "model": "local-model", "name": "llama.cpp"},
    ]

    print(f"{'الخادم':<12} | {'متوسط TTFT (ms)':<18} | {'P95 TTFT (ms)':<15} | {'Throughput (tok/s)':<20}")
    print("-" * 75)

    for config in configs:
        try:
            result = await run_concurrent_benchmark(
                config["base_url"], config["model"], config["name"], n_concurrent=10
            )
            s = result.summary()
            print(f"{s['server']:<12} | {s['avg_ttft_ms']:<18} | {s['p95_ttft_ms']:<15} | {s['throughput_tokens_per_sec']:<20}")
        except Exception as e:
            print(f"{config['name']:<12} | فشل الاتصال: {e}")


if __name__ == "__main__":
    asyncio.run(compare_all_servers())

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-60-chapter-60.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — خوادم الاستدلال — vLLM وTGI وTensorRT-LLM وllama.cpp
Large Language Model Engineering — vLLM

Book slug : llm-engineering-69-chapters
Chapter   : ch-60-chapter-60 (lesson 60)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_60.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 60: الفصل 60: خوادم الاستدلال — vLLM وTGI وTensorRT-LLM وllama.cpp."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_serving() -> None:
    arrivals = [0, 1, 2, 8, 9, 10, 20]
    batch_window, max_batch = (3, 3)
    batches, current = ([], [])
    for arrival in arrivals:
        if current and (arrival - current[0] > batch_window or len(current) == max_batch):
            batches.append(current)
            current = []
        current.append(arrival)
    if current:
        batches.append(current)
    print('arrival times:', arrivals)
    print('formed batches:', batches)
    print('throughput (items per batch):', [len(batch) for batch in batches])

if __name__ == "__main__":
    print("Chapter 60 lab: Simulate batching and server throughput")
    run_serving()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-11-deployment-llmops/chapter-60.md
# ────────────────────────────────────────────────────────────────────────
# vllm_server_setup.py
"""
نشر vLLM كخادم كامل متوافق مع OpenAI API — يمكن تشغيله بسطر واحد من CLI أيضًا:
    vllm serve meta-llama/Llama-3.1-8B-Instruct --quantization awq --max-model-len 4096
لكن هنا نوضح الإعداد البرمجي الكامل للتحكم الدقيق بكل معامل
"""

from vllm import LLM, SamplingParams
from vllm.entrypoints.openai.api_server import run_server
import argparse


def build_vllm_engine(model_path: str, quantization: str = "awq", tensor_parallel_size: int = 1):
    """
    إعداد محرك vLLM مع كل التحسينات التي تعلمناها في الفصل الماضي مُفعَّلة ضمنيًا:
    - PagedAttention: مُفعَّل افتراضيًا دائمًا (جوهر التصميم)
    - Continuous Batching: مُفعَّل افتراضيًا دائمًا
    - Quantization: نحدده صراحة هنا (awq / gptq / None)
    """
    llm = LLM(
        model=model_path,
        quantization=quantization,              # "awq", "gptq", أو None للنموذج الأصلي
        tensor_parallel_size=tensor_parallel_size,  # توزيع على عدة GPUs إذا لزم
        gpu_memory_utilization=0.90,             # نسبة ذاكرة GPU المحجوزة لـ KV-Cache
        max_model_len=4096,                      # أقصى طول سياق مدعوم
        enforce_eager=False,                     # False = يسمح بـ CUDA graphs لتسريع إضافي
    )
    return llm


def run_batch_inference_example(llm: LLM):
    """مثال على استدلال دفعي مباشر (بدون خادم HTTP) — مفيد لمعالجة offline ضخمة"""
    prompts = [
        "اشرح مفهوم PagedAttention في جملتين.",
        "ما الفرق بين vLLM وTGI؟",
        "اكتب دالة Python لحساب متوسط قائمة أرقام.",
    ]

    sampling_params = SamplingParams(
        temperature=0.7, top_p=0.9, max_tokens=200,
    )

    # هذا النداء الواحد يستفيد تلقائيًا من Continuous Batching لمعالجة الثلاثة بالتوازي
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        print(f"\nالسؤال: {output.prompt}")
        print(f"الإجابة: {output.outputs[0].text}")
        print(f"عدد التوكنات المُولَّدة: {len(output.outputs[0].token_ids)}")


if __name__ == "__main__":
    # للتشغيل كخادم HTTP كامل متوافق مع OpenAI، استخدم بدلًا من ذلك من سطر الأوامر:
    #   python -m vllm.entrypoints.openai.api_server \
    #       --model meta-llama/Llama-3.1-8B-Instruct \
    #       --quantization awq --tensor-parallel-size 1 --port 8000

    engine = build_vllm_engine("meta-llama/Llama-3.1-8B-Instruct", quantization="awq")
    run_batch_inference_example(engine)

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-11-deployment-llmops/chapter-60.md
# ────────────────────────────────────────────────────────────────────────
# client_vllm.py
"""بمجرد تشغيل خادم vLLM (عبر الأمر CLI أعلاه)، يصبح استدعاؤه مطابقًا تمامًا لاستدعاء OpenAI"""

from openai import OpenAI

# النقطة الأساسية: نفس عميل OpenAI، فقط نغيّر base_url للإشارة لخادمنا المحلي
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed-for-local-server",  # vLLM لا يتطلب مفتاحًا حقيقيًا افتراضيًا
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "اشرح لي PagedAttention بجملة واحدة بسيطة."}],
    temperature=0.5, max_tokens=150,
)

print(response.choices[0].message.content)
print(f"\nتوكنات مُستخدمة: {response.usage.total_tokens}")

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-11-deployment-llmops/chapter-60.md
# ────────────────────────────────────────────────────────────────────────
# client_tgi.py
"""TGI يوفر واجهته الخاصة (text-generation API) بالإضافة لواجهة OpenAI-compatible"""

from huggingface_hub import InferenceClient

client = InferenceClient(model="http://localhost:8080")

response = client.text_generation(
    prompt="اشرح لي الفرق بين TGI وvLLM بإيجاز.",
    max_new_tokens=150,
    temperature=0.5,
    stream=False,
)
print(response)

# ─── أو عبر واجهة OpenAI-compatible المدمجة في TGI الحديث ───
from openai import OpenAI

client_openai_style = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
response2 = client_openai_style.chat.completions.create(
    model="tgi",
    messages=[{"role": "user", "content": "اشرح PagedAttention بجملة واحدة."}],
    max_tokens=100,
)
print(response2.choices[0].message.content)

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-11-deployment-llmops/chapter-60.md
# ────────────────────────────────────────────────────────────────────────
# llama_cpp_setup.py
"""
llama.cpp عبر مكتبة llama-cpp-python — الخيار الأمثل للتشغيل المحلي
بدون الحاجة لـ GPU قوي، أو حتى بدون GPU إطلاقًا
"""

from llama_cpp import Llama

# تحميل نموذج بصيغة GGUF مُكمَّم مسبقًا (يمكن تحميله من Hugging Face Hub مباشرة)
llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
    filename="*q4_k_m.gguf",       # مستوى التكميم: Q4_K_M = توازن جيد بين الحجم والجودة
    n_gpu_layers=-1,                # -1 = حمّل كل الطبقات الممكنة على GPU إن توفر، 0 = CPU بالكامل
    n_ctx=4096,                     # طول السياق الأقصى
    n_threads=8,                    # عدد أنوية CPU المستخدمة (مهم جدًا عند التشغيل بدون GPU)
    verbose=False,
)

def chat_with_local_model(prompt: str) -> str:
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200, temperature=0.6,
    )
    return response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    result = chat_with_local_model("ما ميزة llama.cpp الأساسية مقارنة بـ vLLM؟")
    print(result)

    # llama.cpp يدعم أيضًا تشغيل خادم HTTP متوافق مع OpenAI مباشرة من CLI:
    #   python -m llama_cpp.server --model model.gguf --n_gpu_layers -1 --port 8000

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-11-deployment-llmops/chapter-60.md
# ────────────────────────────────────────────────────────────────────────
# benchmark_inference_servers.py
"""
نظام قياس موحّد يقيس Throughput وTTFT وTPOT عبر أي خادم متوافق مع OpenAI API
(يعمل مع vLLM, TGI, llama.cpp server بنفس الكود تمامًا — هذا بالضبط جمال المعيار الموحّد)
"""

import time
import asyncio
import statistics
from dataclasses import dataclass, field
from typing import List
from openai import AsyncOpenAI


@dataclass
class BenchmarkResult:
    server_name: str
    ttft_list: List[float] = field(default_factory=list)      # Time To First Token لكل طلب
    total_time_list: List[float] = field(default_factory=list)
    tokens_generated_list: List[int] = field(default_factory=list)

    def summary(self) -> dict:
        total_tokens = sum(self.tokens_generated_list)
        total_time = max(self.total_time_list) if self.total_time_list else 1
        return {
            "server": self.server_name,
            "avg_ttft_ms": round(statistics.mean(self.ttft_list) * 1000, 1) if self.ttft_list else 0,
            "p95_ttft_ms": round(statistics.quantiles(self.ttft_list, n=20)[18] * 1000, 1) if len(self.ttft_list) > 1 else 0,
            "throughput_tokens_per_sec": round(total_tokens / total_time, 1),
            "total_requests": len(self.total_time_list),
        }


async def single_streaming_request(client: AsyncOpenAI, model: str, prompt: str) -> dict:
    """نقيس TTFT عبر وضع streaming: الزمن من الإرسال حتى أول chunk فعلي"""
    start = time.time()
    first_token_time = None
    token_count = 0

    stream = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        max_tokens=150, temperature=0.5, stream=True,
    )

    async for chunk in stream:
        if chunk.choices[0].delta.content:
            if first_token_time is None:
                first_token_time = time.time()
            token_count += 1

    total_time = time.time() - start
    ttft = (first_token_time - start) if first_token_time else total_time

    return {"ttft": ttft, "total_time": total_time, "tokens": token_count}


async def run_concurrent_benchmark(base_url: str, model: str, server_name: str,
                                     n_concurrent: int = 10, prompts: List[str] = None) -> BenchmarkResult:
    """يُشغّل n_concurrent طلبًا متزامنًا — هذا بالضبط ما يكشف الفرق الحقيقي بين Continuous
    Batching (vLLM/TGI) وMicro-batching البسيط (llama.cpp الافتراضي)"""
    client = AsyncOpenAI(base_url=base_url, api_key="not-needed")
    prompts = prompts or ["اشرح مفهوم تقني بسيط بجملتين."] * n_concurrent

    result = BenchmarkResult(server_name=server_name)
    tasks = [single_streaming_request(client, model, p) for p in prompts]
    responses = await asyncio.gather(*tasks)

    for r in responses:
        result.ttft_list.append(r["ttft"])
        result.total_time_list.append(r["total_time"])
        result.tokens_generated_list.append(r["tokens"])

    return result


async def compare_all_servers():
    """شغّل هذا بعد تشغيل كل خادم على منفذ منفصل محليًا"""
    configs = [
        {"base_url": "http://localhost:8000/v1", "model": "meta-llama/Llama-3.1-8B-Instruct", "name": "vLLM"},
        {"base_url": "http://localhost:8080/v1", "model": "tgi", "name": "TGI"},
        {"base_url": "http://localhost:8001/v1", "model": "local-model", "name": "llama.cpp"},
    ]

    print(f"{'الخادم':<12} | {'متوسط TTFT (ms)':<18} | {'P95 TTFT (ms)':<15} | {'Throughput (tok/s)':<20}")
    print("-" * 75)

    for config in configs:
        try:
            result = await run_concurrent_benchmark(
                config["base_url"], config["model"], config["name"], n_concurrent=10
            )
            s = result.summary()
            print(f"{s['server']:<12} | {s['avg_ttft_ms']:<18} | {s['p95_ttft_ms']:<15} | {s['throughput_tokens_per_sec']:<20}")
        except Exception as e:
            print(f"{config['name']:<12} | فشل الاتصال: {e}")


if __name__ == "__main__":
    asyncio.run(compare_all_servers())

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-60-chapter-60.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 60: الفصل 60: خوادم الاستدلال — vLLM وTGI وTensorRT-LLM وllama.cpp."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_serving() -> None:
    arrivals = [0, 1, 2, 8, 9, 10, 20]
    batch_window, max_batch = (3, 3)
    batches, current = ([], [])
    for arrival in arrivals:
        if current and (arrival - current[0] > batch_window or len(current) == max_batch):
            batches.append(current)
            current = []
        current.append(arrival)
    if current:
        batches.append(current)
    print('arrival times:', arrivals)
    print('formed batches:', batches)
    print('throughput (items per batch):', [len(batch) for batch in batches])

if __name__ == "__main__":
    print("Chapter 60 lab: Simulate batching and server throughput")
    run_serving()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-60-chapter-60.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# vllm_server_setup.py
"""
نشر vLLM كخادم كامل متوافق مع OpenAI API — يمكن تشغيله بسطر واحد من CLI أيضًا:
    vllm serve meta-llama/Llama-3.1-8B-Instruct --quantization awq --max-model-len 4096
لكن هنا نوضح الإعداد البرمجي الكامل للتحكم الدقيق بكل معامل
"""

from vllm import LLM, SamplingParams
from vllm.entrypoints.openai.api_server import run_server
import argparse


def build_vllm_engine(model_path: str, quantization: str = "awq", tensor_parallel_size: int = 1):
    """
    إعداد محرك vLLM مع كل التحسينات التي تعلمناها في الفصل الماضي مُفعَّلة ضمنيًا:
    - PagedAttention: مُفعَّل افتراضيًا دائمًا (جوهر التصميم)
    - Continuous Batching: مُفعَّل افتراضيًا دائمًا
    - Quantization: نحدده صراحة هنا (awq / gptq / None)
    """
    llm = LLM(
        model=model_path,
        quantization=quantization,              # "awq", "gptq", أو None للنموذج الأصلي
        tensor_parallel_size=tensor_parallel_size,  # توزيع على عدة GPUs إذا لزم
        gpu_memory_utilization=0.90,             # نسبة ذاكرة GPU المحجوزة لـ KV-Cache
        max_model_len=4096,                      # أقصى طول سياق مدعوم
        enforce_eager=False,                     # False = يسمح بـ CUDA graphs لتسريع إضافي
    )
    return llm


def run_batch_inference_example(llm: LLM):
    """مثال على استدلال دفعي مباشر (بدون خادم HTTP) — مفيد لمعالجة offline ضخمة"""
    prompts = [
        "اشرح مفهوم PagedAttention في جملتين.",
        "ما الفرق بين vLLM وTGI؟",
        "اكتب دالة Python لحساب متوسط قائمة أرقام.",
    ]

    sampling_params = SamplingParams(
        temperature=0.7, top_p=0.9, max_tokens=200,
    )

    # هذا النداء الواحد يستفيد تلقائيًا من Continuous Batching لمعالجة الثلاثة بالتوازي
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        print(f"\nالسؤال: {output.prompt}")
        print(f"الإجابة: {output.outputs[0].text}")
        print(f"عدد التوكنات المُولَّدة: {len(output.outputs[0].token_ids)}")


if __name__ == "__main__":
    # للتشغيل كخادم HTTP كامل متوافق مع OpenAI، استخدم بدلًا من ذلك من سطر الأوامر:
    #   python -m vllm.entrypoints.openai.api_server \
    #       --model meta-llama/Llama-3.1-8B-Instruct \
    #       --quantization awq --tensor-parallel-size 1 --port 8000

    engine = build_vllm_engine("meta-llama/Llama-3.1-8B-Instruct", quantization="awq")
    run_batch_inference_example(engine)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-60-chapter-60.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# client_vllm.py
"""بمجرد تشغيل خادم vLLM (عبر الأمر CLI أعلاه)، يصبح استدعاؤه مطابقًا تمامًا لاستدعاء OpenAI"""

from openai import OpenAI

# النقطة الأساسية: نفس عميل OpenAI، فقط نغيّر base_url للإشارة لخادمنا المحلي
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed-for-local-server",  # vLLM لا يتطلب مفتاحًا حقيقيًا افتراضيًا
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "اشرح لي PagedAttention بجملة واحدة بسيطة."}],
    temperature=0.5, max_tokens=150,
)

print(response.choices[0].message.content)
print(f"\nتوكنات مُستخدمة: {response.usage.total_tokens}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-60-chapter-60.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# client_tgi.py
"""TGI يوفر واجهته الخاصة (text-generation API) بالإضافة لواجهة OpenAI-compatible"""

from huggingface_hub import InferenceClient

client = InferenceClient(model="http://localhost:8080")

response = client.text_generation(
    prompt="اشرح لي الفرق بين TGI وvLLM بإيجاز.",
    max_new_tokens=150,
    temperature=0.5,
    stream=False,
)
print(response)

# ─── أو عبر واجهة OpenAI-compatible المدمجة في TGI الحديث ───
from openai import OpenAI

client_openai_style = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
response2 = client_openai_style.chat.completions.create(
    model="tgi",
    messages=[{"role": "user", "content": "اشرح PagedAttention بجملة واحدة."}],
    max_tokens=100,
)
print(response2.choices[0].message.content)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-60-chapter-60.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# llama_cpp_setup.py
"""
llama.cpp عبر مكتبة llama-cpp-python — الخيار الأمثل للتشغيل المحلي
بدون الحاجة لـ GPU قوي، أو حتى بدون GPU إطلاقًا
"""

from llama_cpp import Llama

# تحميل نموذج بصيغة GGUF مُكمَّم مسبقًا (يمكن تحميله من Hugging Face Hub مباشرة)
llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
    filename="*q4_k_m.gguf",       # مستوى التكميم: Q4_K_M = توازن جيد بين الحجم والجودة
    n_gpu_layers=-1,                # -1 = حمّل كل الطبقات الممكنة على GPU إن توفر، 0 = CPU بالكامل
    n_ctx=4096,                     # طول السياق الأقصى
    n_threads=8,                    # عدد أنوية CPU المستخدمة (مهم جدًا عند التشغيل بدون GPU)
    verbose=False,
)

def chat_with_local_model(prompt: str) -> str:
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200, temperature=0.6,
    )
    return response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    result = chat_with_local_model("ما ميزة llama.cpp الأساسية مقارنة بـ vLLM؟")
    print(result)

    # llama.cpp يدعم أيضًا تشغيل خادم HTTP متوافق مع OpenAI مباشرة من CLI:
    #   python -m llama_cpp.server --model model.gguf --n_gpu_layers -1 --port 8000

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-60-chapter-60.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# benchmark_inference_servers.py
"""
نظام قياس موحّد يقيس Throughput وTTFT وTPOT عبر أي خادم متوافق مع OpenAI API
(يعمل مع vLLM, TGI, llama.cpp server بنفس الكود تمامًا — هذا بالضبط جمال المعيار الموحّد)
"""

import time
import asyncio
import statistics
from dataclasses import dataclass, field
from typing import List
from openai import AsyncOpenAI


@dataclass
class BenchmarkResult:
    server_name: str
    ttft_list: List[float] = field(default_factory=list)      # Time To First Token لكل طلب
    total_time_list: List[float] = field(default_factory=list)
    tokens_generated_list: List[int] = field(default_factory=list)

    def summary(self) -> dict:
        total_tokens = sum(self.tokens_generated_list)
        total_time = max(self.total_time_list) if self.total_time_list else 1
        return {
            "server": self.server_name,
            "avg_ttft_ms": round(statistics.mean(self.ttft_list) * 1000, 1) if self.ttft_list else 0,
            "p95_ttft_ms": round(statistics.quantiles(self.ttft_list, n=20)[18] * 1000, 1) if len(self.ttft_list) > 1 else 0,
            "throughput_tokens_per_sec": round(total_tokens / total_time, 1),
            "total_requests": len(self.total_time_list),
        }


async def single_streaming_request(client: AsyncOpenAI, model: str, prompt: str) -> dict:
    """نقيس TTFT عبر وضع streaming: الزمن من الإرسال حتى أول chunk فعلي"""
    start = time.time()
    first_token_time = None
    token_count = 0

    stream = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        max_tokens=150, temperature=0.5, stream=True,
    )

    async for chunk in stream:
        if chunk.choices[0].delta.content:
            if first_token_time is None:
                first_token_time = time.time()
            token_count += 1

    total_time = time.time() - start
    ttft = (first_token_time - start) if first_token_time else total_time

    return {"ttft": ttft, "total_time": total_time, "tokens": token_count}


async def run_concurrent_benchmark(base_url: str, model: str, server_name: str,
                                     n_concurrent: int = 10, prompts: List[str] = None) -> BenchmarkResult:
    """يُشغّل n_concurrent طلبًا متزامنًا — هذا بالضبط ما يكشف الفرق الحقيقي بين Continuous
    Batching (vLLM/TGI) وMicro-batching البسيط (llama.cpp الافتراضي)"""
    client = AsyncOpenAI(base_url=base_url, api_key="not-needed")
    prompts = prompts or ["اشرح مفهوم تقني بسيط بجملتين."] * n_concurrent

    result = BenchmarkResult(server_name=server_name)
    tasks = [single_streaming_request(client, model, p) for p in prompts]
    responses = await asyncio.gather(*tasks)

    for r in responses:
        result.ttft_list.append(r["ttft"])
        result.total_time_list.append(r["total_time"])
        result.tokens_generated_list.append(r["tokens"])

    return result


async def compare_all_servers():
    """شغّل هذا بعد تشغيل كل خادم على منفذ منفصل محليًا"""
    configs = [
        {"base_url": "http://localhost:8000/v1", "model": "meta-llama/Llama-3.1-8B-Instruct", "name": "vLLM"},
        {"base_url": "http://localhost:8080/v1", "model": "tgi", "name": "TGI"},
        {"base_url": "http://localhost:8001/v1", "model": "local-model", "name": "llama.cpp"},
    ]

    print(f"{'الخادم':<12} | {'متوسط TTFT (ms)':<18} | {'P95 TTFT (ms)':<15} | {'Throughput (tok/s)':<20}")
    print("-" * 75)

    for config in configs:
        try:
            result = await run_concurrent_benchmark(
                config["base_url"], config["model"], config["name"], n_concurrent=10
            )
            s = result.summary()
            print(f"{s['server']:<12} | {s['avg_ttft_ms']:<18} | {s['p95_ttft_ms']:<15} | {s['throughput_tokens_per_sec']:<20}")
        except Exception as e:
            print(f"{config['name']:<12} | فشل الاتصال: {e}")


if __name__ == "__main__":
    asyncio.run(compare_all_servers())
