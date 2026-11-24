#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — البنية التحتية والتوسّع
Mastering Transformers and Large Language Models — Infrastructure Scaling

Book slug : mastering-llms-transformers
Chapter   : ch-56-ch56-infrastructure-scaling (lesson 56)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-56-ch56-infrastructure-scaling.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — البنية التحتية والتوسّع
Mastering Transformers and Large Language Models — Infrastructure Scaling

Book slug : mastering-llms-transformers
Chapter   : ch-56-ch56-infrastructure-scaling (lesson 56)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-56-ch56-infrastructure-scaling.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — البنية التحتية والتوسّع
Mastering Transformers and Large Language Models — Infrastructure Scaling

Book slug : mastering-llms-transformers
Chapter   : ch-56-ch56-infrastructure-scaling (lesson 56)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-11-llmops/ch56-infrastructure-scaling.ipynb#cell-1
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
# [002] content/notebooks/part-11-llmops/ch56-infrastructure-scaling.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
استراتيجيات التوازي للاستدلال (تختلف عن التدريب في الفصل 21 بتركيزها على الـ latency
بدلاً من throughput التدريب):
"""

# 1) Tensor Parallelism للاستدلال — لنماذج أكبر من ذاكرة GPU واحدة
TENSOR_PARALLEL_CONFIG = {
    "tensor_parallel_size": 4,   # تقسيم كل طبقة عبر 4 GPUs
    "note": "يقلّل latency لكل طلب (كل GPU يُنفّذ جزءًا من نفس الطبقة بالتوازي)،"
            " لكن يتطلب اتصال NVLink/InfiniBand سريع جدًا بين الـ GPUs"
}

# 2) Pipeline Parallelism للاستدلال — أقل شيوعًا للاستدلال التفاعلي
PIPELINE_PARALLEL_NOTE = """
أقل ملاءمة للاستدلال منخفض الـ latency لأنه يُدخل "فقاعات" انتظار بين مراحل الأنابيب
(كل GPU ينتظر دوره)، لكنه مفيد لـ throughput عالٍ في معالجة دفعية غير تفاعلية (offline batch inference)
"""

# 3) نموذج واحد لكل GPU (Replica-based Scaling) — الأبسط والأكثر شيوعًا للتوسّع الأفقي
REPLICA_SCALING_NOTE = """
إذا كان النموذج يتّسع في ذاكرة GPU واحدة، فإن الأسلوب الأبسط والأكثر فعالية
للتعامل مع زيادة الحمل هو ببساطة تشغيل عدة نسخ (replicas) مستقلة خلف Load Balancer،
بدلاً من أي شكل من التوازي المعقّد.
"""

def choose_parallelism_strategy(model_size_gb: float, gpu_memory_gb: float,
                                   latency_sensitive: bool) -> str:
    """دليل قرار مبسّط لاختيار استراتيجية التوزيع"""
    if model_size_gb <= gpu_memory_gb * 0.7:   # هامش أمان لـ KV cache
        return "Replica-based Scaling (عدة نسخ مستقلة خلف Load Balancer)"
    elif latency_sensitive:
        return "Tensor Parallelism (لتقليل زمن الاستجابة للطلب الواحد)"
    else:
        return "Pipeline Parallelism مقبول (إن كان throughput الأولوية وليس latency)"

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-11-llmops/ch56-infrastructure-scaling.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class LoadBalancerWithHealthAwareness:
    """
    موازن تحميل بسيط يأخذ بعين الاعتبار طول طابور الانتظار الفعلي لكل نسخة،
    وليس فقط Round-Robin الساذج
    """
    def __init__(self, backend_servers: list):
        self.servers = backend_servers   # [{"url": ..., "queue_length": 0}]

    def select_server(self) -> dict:
        """اختيار الخادم الأقل ازدحامًا (Least Connections / Least Queue)"""
        return min(self.servers, key=lambda s: s["queue_length"])

    async def route_request(self, request):
        server = self.select_server()
        server["queue_length"] += 1
        try:
            response = await self._forward_request(server["url"], request)
            return response
        finally:
            server["queue_length"] -= 1

    async def _forward_request(self, url, request): ...

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-11-llmops/ch56-infrastructure-scaling.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-56-ch56-infrastructure-scaling.ipynb#cell-1
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-56-ch56-infrastructure-scaling.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
استراتيجيات التوازي للاستدلال (تختلف عن التدريب في الفصل 21 بتركيزها على الـ latency
بدلاً من throughput التدريب):
"""

# 1) Tensor Parallelism للاستدلال — لنماذج أكبر من ذاكرة GPU واحدة
TENSOR_PARALLEL_CONFIG = {
    "tensor_parallel_size": 4,   # تقسيم كل طبقة عبر 4 GPUs
    "note": "يقلّل latency لكل طلب (كل GPU يُنفّذ جزءًا من نفس الطبقة بالتوازي)،"
            " لكن يتطلب اتصال NVLink/InfiniBand سريع جدًا بين الـ GPUs"
}

# 2) Pipeline Parallelism للاستدلال — أقل شيوعًا للاستدلال التفاعلي
PIPELINE_PARALLEL_NOTE = """
أقل ملاءمة للاستدلال منخفض الـ latency لأنه يُدخل "فقاعات" انتظار بين مراحل الأنابيب
(كل GPU ينتظر دوره)، لكنه مفيد لـ throughput عالٍ في معالجة دفعية غير تفاعلية (offline batch inference)
"""

# 3) نموذج واحد لكل GPU (Replica-based Scaling) — الأبسط والأكثر شيوعًا للتوسّع الأفقي
REPLICA_SCALING_NOTE = """
إذا كان النموذج يتّسع في ذاكرة GPU واحدة، فإن الأسلوب الأبسط والأكثر فعالية
للتعامل مع زيادة الحمل هو ببساطة تشغيل عدة نسخ (replicas) مستقلة خلف Load Balancer،
بدلاً من أي شكل من التوازي المعقّد.
"""

def choose_parallelism_strategy(model_size_gb: float, gpu_memory_gb: float,
                                   latency_sensitive: bool) -> str:
    """دليل قرار مبسّط لاختيار استراتيجية التوزيع"""
    if model_size_gb <= gpu_memory_gb * 0.7:   # هامش أمان لـ KV cache
        return "Replica-based Scaling (عدة نسخ مستقلة خلف Load Balancer)"
    elif latency_sensitive:
        return "Tensor Parallelism (لتقليل زمن الاستجابة للطلب الواحد)"
    else:
        return "Pipeline Parallelism مقبول (إن كان throughput الأولوية وليس latency)"

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-56-ch56-infrastructure-scaling.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class LoadBalancerWithHealthAwareness:
    """
    موازن تحميل بسيط يأخذ بعين الاعتبار طول طابور الانتظار الفعلي لكل نسخة،
    وليس فقط Round-Robin الساذج
    """
    def __init__(self, backend_servers: list):
        self.servers = backend_servers   # [{"url": ..., "queue_length": 0}]

    def select_server(self) -> dict:
        """اختيار الخادم الأقل ازدحامًا (Least Connections / Least Queue)"""
        return min(self.servers, key=lambda s: s["queue_length"])

    async def route_request(self, request):
        server = self.select_server()
        server["queue_length"] += 1
        try:
            response = await self._forward_request(server["url"], request)
            return response
        finally:
            server["queue_length"] -= 1

    async def _forward_request(self, url, request): ...

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-56-ch56-infrastructure-scaling.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-56-ch56-infrastructure-scaling.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — البنية التحتية والتوسّع
Mastering Transformers and Large Language Models — Infrastructure Scaling

Book slug : mastering-llms-transformers
Chapter   : ch-56-ch56-infrastructure-scaling (lesson 56)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-11-llmops/ch56-infrastructure-scaling.ipynb#cell-1
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
# [002] content/notebooks/part-11-llmops/ch56-infrastructure-scaling.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
استراتيجيات التوازي للاستدلال (تختلف عن التدريب في الفصل 21 بتركيزها على الـ latency
بدلاً من throughput التدريب):
"""

# 1) Tensor Parallelism للاستدلال — لنماذج أكبر من ذاكرة GPU واحدة
TENSOR_PARALLEL_CONFIG = {
    "tensor_parallel_size": 4,   # تقسيم كل طبقة عبر 4 GPUs
    "note": "يقلّل latency لكل طلب (كل GPU يُنفّذ جزءًا من نفس الطبقة بالتوازي)،"
            " لكن يتطلب اتصال NVLink/InfiniBand سريع جدًا بين الـ GPUs"
}

# 2) Pipeline Parallelism للاستدلال — أقل شيوعًا للاستدلال التفاعلي
PIPELINE_PARALLEL_NOTE = """
أقل ملاءمة للاستدلال منخفض الـ latency لأنه يُدخل "فقاعات" انتظار بين مراحل الأنابيب
(كل GPU ينتظر دوره)، لكنه مفيد لـ throughput عالٍ في معالجة دفعية غير تفاعلية (offline batch inference)
"""

# 3) نموذج واحد لكل GPU (Replica-based Scaling) — الأبسط والأكثر شيوعًا للتوسّع الأفقي
REPLICA_SCALING_NOTE = """
إذا كان النموذج يتّسع في ذاكرة GPU واحدة، فإن الأسلوب الأبسط والأكثر فعالية
للتعامل مع زيادة الحمل هو ببساطة تشغيل عدة نسخ (replicas) مستقلة خلف Load Balancer،
بدلاً من أي شكل من التوازي المعقّد.
"""

def choose_parallelism_strategy(model_size_gb: float, gpu_memory_gb: float,
                                   latency_sensitive: bool) -> str:
    """دليل قرار مبسّط لاختيار استراتيجية التوزيع"""
    if model_size_gb <= gpu_memory_gb * 0.7:   # هامش أمان لـ KV cache
        return "Replica-based Scaling (عدة نسخ مستقلة خلف Load Balancer)"
    elif latency_sensitive:
        return "Tensor Parallelism (لتقليل زمن الاستجابة للطلب الواحد)"
    else:
        return "Pipeline Parallelism مقبول (إن كان throughput الأولوية وليس latency)"

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-11-llmops/ch56-infrastructure-scaling.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class LoadBalancerWithHealthAwareness:
    """
    موازن تحميل بسيط يأخذ بعين الاعتبار طول طابور الانتظار الفعلي لكل نسخة،
    وليس فقط Round-Robin الساذج
    """
    def __init__(self, backend_servers: list):
        self.servers = backend_servers   # [{"url": ..., "queue_length": 0}]

    def select_server(self) -> dict:
        """اختيار الخادم الأقل ازدحامًا (Least Connections / Least Queue)"""
        return min(self.servers, key=lambda s: s["queue_length"])

    async def route_request(self, request):
        server = self.select_server()
        server["queue_length"] += 1
        try:
            response = await self._forward_request(server["url"], request)
            return response
        finally:
            server["queue_length"] -= 1

    async def _forward_request(self, url, request): ...

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-11-llmops/ch56-infrastructure-scaling.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-56-ch56-infrastructure-scaling.ipynb#cell-2
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-56-ch56-infrastructure-scaling.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
استراتيجيات التوازي للاستدلال (تختلف عن التدريب في الفصل 21 بتركيزها على الـ latency
بدلاً من throughput التدريب):
"""

# 1) Tensor Parallelism للاستدلال — لنماذج أكبر من ذاكرة GPU واحدة
TENSOR_PARALLEL_CONFIG = {
    "tensor_parallel_size": 4,   # تقسيم كل طبقة عبر 4 GPUs
    "note": "يقلّل latency لكل طلب (كل GPU يُنفّذ جزءًا من نفس الطبقة بالتوازي)،"
            " لكن يتطلب اتصال NVLink/InfiniBand سريع جدًا بين الـ GPUs"
}

# 2) Pipeline Parallelism للاستدلال — أقل شيوعًا للاستدلال التفاعلي
PIPELINE_PARALLEL_NOTE = """
أقل ملاءمة للاستدلال منخفض الـ latency لأنه يُدخل "فقاعات" انتظار بين مراحل الأنابيب
(كل GPU ينتظر دوره)، لكنه مفيد لـ throughput عالٍ في معالجة دفعية غير تفاعلية (offline batch inference)
"""

# 3) نموذج واحد لكل GPU (Replica-based Scaling) — الأبسط والأكثر شيوعًا للتوسّع الأفقي
REPLICA_SCALING_NOTE = """
إذا كان النموذج يتّسع في ذاكرة GPU واحدة، فإن الأسلوب الأبسط والأكثر فعالية
للتعامل مع زيادة الحمل هو ببساطة تشغيل عدة نسخ (replicas) مستقلة خلف Load Balancer،
بدلاً من أي شكل من التوازي المعقّد.
"""

def choose_parallelism_strategy(model_size_gb: float, gpu_memory_gb: float,
                                   latency_sensitive: bool) -> str:
    """دليل قرار مبسّط لاختيار استراتيجية التوزيع"""
    if model_size_gb <= gpu_memory_gb * 0.7:   # هامش أمان لـ KV cache
        return "Replica-based Scaling (عدة نسخ مستقلة خلف Load Balancer)"
    elif latency_sensitive:
        return "Tensor Parallelism (لتقليل زمن الاستجابة للطلب الواحد)"
    else:
        return "Pipeline Parallelism مقبول (إن كان throughput الأولوية وليس latency)"

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-56-ch56-infrastructure-scaling.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class LoadBalancerWithHealthAwareness:
    """
    موازن تحميل بسيط يأخذ بعين الاعتبار طول طابور الانتظار الفعلي لكل نسخة،
    وليس فقط Round-Robin الساذج
    """
    def __init__(self, backend_servers: list):
        self.servers = backend_servers   # [{"url": ..., "queue_length": 0}]

    def select_server(self) -> dict:
        """اختيار الخادم الأقل ازدحامًا (Least Connections / Least Queue)"""
        return min(self.servers, key=lambda s: s["queue_length"])

    async def route_request(self, request):
        server = self.select_server()
        server["queue_length"] += 1
        try:
            response = await self._forward_request(server["url"], request)
            return response
        finally:
            server["queue_length"] -= 1

    async def _forward_request(self, url, request): ...

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-56-ch56-infrastructure-scaling.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
