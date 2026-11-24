#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — المراقبة والتشغيل: Observability
Mastering Transformers and Large Language Models — Observability

Book slug : mastering-llms-transformers
Chapter   : ch-57-ch57-observability (lesson 57)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-57-ch57-observability.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — المراقبة والتشغيل: Observability
Mastering Transformers and Large Language Models — Observability

Book slug : mastering-llms-transformers
Chapter   : ch-57-ch57-observability (lesson 57)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-57-ch57-observability.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — المراقبة والتشغيل: Observability
Mastering Transformers and Large Language Models — Observability

Book slug : mastering-llms-transformers
Chapter   : ch-57-ch57-observability (lesson 57)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-11-llmops/ch57-observability.ipynb#cell-1
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
    ("opentelemetry", "opentelemetry-api opentelemetry-sdk"),
    ("prometheus_client", "prometheus-client")
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
# [002] content/notebooks/part-11-llmops/ch57-observability.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import time
import logging
import uuid
from dataclasses import dataclass, asdict
from prometheus_client import Counter, Histogram, Gauge

# ==================== 1. Metrics (مقاييس كمّية مُجمَّعة) ====================
REQUEST_COUNT = Counter('llm_requests_total', 'إجمالي الطلبات', ['model', 'status'])
REQUEST_LATENCY = Histogram('llm_request_duration_seconds', 'زمن معالجة الطلب',
                              buckets=[0.1, 0.5, 1, 2, 5, 10, 30])
TOKENS_GENERATED = Counter('llm_tokens_generated_total', 'إجمالي الرموز المُولَّدة')
TIME_TO_FIRST_TOKEN = Histogram('llm_ttft_seconds', 'الزمن حتى أول رمز (مهم جدًا لتجربة المستخدم)')
ACTIVE_REQUESTS = Gauge('llm_active_requests', 'عدد الطلبات النشطة حاليًا')
GPU_MEMORY_USAGE = Gauge('llm_gpu_memory_used_bytes', 'استهلاك ذاكرة GPU')

# ==================== 2. Structured Logging (سجلات مُهيكَلة قابلة للاستعلام) ====================
@dataclass
class InferenceLogEntry:
    request_id: str
    timestamp: float
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_latency_ms: float
    time_to_first_token_ms: float
    user_id: str
    status: str
    error_message: str = None

logger = logging.getLogger("llm_service")

def log_inference_structured(entry: InferenceLogEntry):
    """تسجيل بصيغة JSON مُهيكَلة — قابلة للاستعلام في Elasticsearch/Loki/CloudWatch"""
    logger.info(json.dumps(asdict(entry), ensure_ascii=False))

# ==================== 3. Distributed Tracing (تتبّع موزّع عبر خدمات متعددة) ====================
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer("llm-service")

class InstrumentedLLMService:
    """خدمة استدلال كاملة مع مراقبة شاملة مدمجة"""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_with_full_observability(self, prompt: str, user_id: str) -> dict:
        request_id = str(uuid.uuid4())
        start_time = time.time()
        ACTIVE_REQUESTS.inc()

        with tracer.start_as_current_span("llm_inference") as span:
            span.set_attribute("request_id", request_id)
            span.set_attribute("user_id", user_id)
            span.set_attribute("prompt_length", len(prompt))

            try:
                with tracer.start_as_current_span("tokenization"):
                    input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
                    prompt_tokens = input_ids.shape[1]

                with tracer.start_as_current_span("generation"):
                    first_token_time = None
                    output_tokens = []
                    for token in self._stream_generate(input_ids):
                        if first_token_time is None:
                            first_token_time = time.time()
                            ttft = first_token_time - start_time
                            TIME_TO_FIRST_TOKEN.observe(ttft)
                        output_tokens.append(token)

                total_latency = time.time() - start_time
                REQUEST_LATENCY.observe(total_latency)
                REQUEST_COUNT.labels(model="llama-3-8b", status="success").inc()
                TOKENS_GENERATED.inc(len(output_tokens))

                log_inference_structured(InferenceLogEntry(
                    request_id=request_id, timestamp=start_time, model_name="llama-3-8b",
                    prompt_tokens=prompt_tokens, completion_tokens=len(output_tokens),
                    total_latency_ms=total_latency * 1000,
                    time_to_first_token_ms=(first_token_time - start_time) * 1000,
                    user_id=user_id, status="success"
                ))
                span.set_status(Status(StatusCode.OK))
                return {"request_id": request_id, "response": self.tokenizer.decode(output_tokens)}

            except Exception as e:
                REQUEST_COUNT.labels(model="llama-3-8b", status="error").inc()
                log_inference_structured(InferenceLogEntry(
                    request_id=request_id, timestamp=start_time, model_name="llama-3-8b",
                    prompt_tokens=0, completion_tokens=0, total_latency_ms=(time.time()-start_time)*1000,
                    time_to_first_token_ms=0, user_id=user_id, status="error", error_message=str(e)
                ))
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            finally:
                ACTIVE_REQUESTS.dec()

    def _stream_generate(self, input_ids): ...  # توليد تدفقي token-by-token

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-11-llmops/ch57-observability.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
مثال قواعد تنبيه Prometheus/Alertmanager لخدمة LLM إنتاجية
"""
ALERTING_RULES_YAML = """
groups:
- name: llm_service_alerts
  rules:
  - alert: HighLatencyP99
    expr: histogram_quantile(0.99, llm_request_duration_seconds_bucket) > 10
    for: 5m
    annotations:
      summary: "زمن الاستجابة عند p99 يتجاوز 10 ثوانٍ لأكثر من 5 دقائق"

  - alert: HighErrorRate
    expr: rate(llm_requests_total{status="error"}[5m]) / rate(llm_requests_total[5m]) > 0.05
    for: 2m
    annotations:
      summary: "معدل الأخطاء يتجاوز 5% — تحقق فورًا"

  - alert: GPUMemoryNearCapacity
    expr: llm_gpu_memory_used_bytes / llm_gpu_memory_total_bytes > 0.95
    for: 1m
    annotations:
      summary: "ذاكرة GPU قريبة من الامتلاء الكامل — خطر OOM وشيك"

  - alert: QueueBacklogGrowing
    expr: llm_active_requests > 50
    for: 3m
    annotations:
      summary: "تراكم طابور الطلبات — قد تحتاج Autoscaling يدويًا أو فحص عنق الزجاجة"
"""

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-11-llmops/ch57-observability.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class OutputQualityMonitor:
    """
    مراقبة مستمرة لجودة المخرجات في الإنتاج، تكملة لمراقبة الأداء التقني
    (يبني على مفاهيم الفصل 46 — Continuous Evaluation، لكن بتركيز على الإنتاج الحي)
    """
    def __init__(self, safety_classifier, judge_client, sample_rate=0.02):
        self.safety_classifier = safety_classifier
        self.judge_client = judge_client
        self.sample_rate = sample_rate   # عيّنة من الطلبات فقط (لتقليل التكلفة)
        self.quality_history = []

    def monitor_response(self, prompt: str, response: str, request_id: str):
        # 1. فحص فوري وزهيد لكل طلب: اكتشاف أنماط مقلقة سريعة (طول غير طبيعي، تكرار)
        if self._detect_degenerate_output(response):
            self._raise_alert("DEGENERATE_OUTPUT", request_id)

        # 2. فحص أعمق (أغلى) لعيّنة فقط من الطلبات
        import random
        if random.random() < self.sample_rate:
            safety_result = self.safety_classifier(response)
            if safety_result["is_harmful"]:
                self._raise_alert("HARMFUL_CONTENT_DETECTED", request_id)

            quality_score = llm_judge_evaluate(self.judge_client, prompt, response)
            self.quality_history.append({"request_id": request_id, "score": quality_score["overall"]})

            # كشف انجراف الجودة (Quality Drift) عبر نافذة متحركة
            if len(self.quality_history) >= 100:
                recent_avg = sum(q["score"] for q in self.quality_history[-100:]) / 100
                if recent_avg < 6.0:   # عتبة جودة دنيا
                    self._raise_alert("QUALITY_DRIFT_DETECTED", request_id,
                                        details=f"متوسط الجودة الأخير: {recent_avg}")

    def _detect_degenerate_output(self, response: str) -> bool:
        """كشف سريع لمخرجات متدهورة شائعة: تكرار لا نهائي، فراغ، طول شاذ"""
        words = response.split()
        if len(words) == 0:
            return True
        if len(set(words[-20:])) < 5 and len(words) > 20:   # تكرار شديد في آخر الاستجابة
            return True
        return False

    def _raise_alert(self, alert_type: str, request_id: str, details: str = ""):
        logger.warning(f"⚠️ تنبيه جودة [{alert_type}] للطلب {request_id}: {details}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-11-llmops/ch57-observability.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-57-ch57-observability.ipynb#cell-1
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
    ("opentelemetry", "opentelemetry-api opentelemetry-sdk"),
    ("prometheus_client", "prometheus-client")
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-57-ch57-observability.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import time
import logging
import uuid
from dataclasses import dataclass, asdict
from prometheus_client import Counter, Histogram, Gauge

# ==================== 1. Metrics (مقاييس كمّية مُجمَّعة) ====================
REQUEST_COUNT = Counter('llm_requests_total', 'إجمالي الطلبات', ['model', 'status'])
REQUEST_LATENCY = Histogram('llm_request_duration_seconds', 'زمن معالجة الطلب',
                              buckets=[0.1, 0.5, 1, 2, 5, 10, 30])
TOKENS_GENERATED = Counter('llm_tokens_generated_total', 'إجمالي الرموز المُولَّدة')
TIME_TO_FIRST_TOKEN = Histogram('llm_ttft_seconds', 'الزمن حتى أول رمز (مهم جدًا لتجربة المستخدم)')
ACTIVE_REQUESTS = Gauge('llm_active_requests', 'عدد الطلبات النشطة حاليًا')
GPU_MEMORY_USAGE = Gauge('llm_gpu_memory_used_bytes', 'استهلاك ذاكرة GPU')

# ==================== 2. Structured Logging (سجلات مُهيكَلة قابلة للاستعلام) ====================
@dataclass
class InferenceLogEntry:
    request_id: str
    timestamp: float
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_latency_ms: float
    time_to_first_token_ms: float
    user_id: str
    status: str
    error_message: str = None

logger = logging.getLogger("llm_service")

def log_inference_structured(entry: InferenceLogEntry):
    """تسجيل بصيغة JSON مُهيكَلة — قابلة للاستعلام في Elasticsearch/Loki/CloudWatch"""
    logger.info(json.dumps(asdict(entry), ensure_ascii=False))

# ==================== 3. Distributed Tracing (تتبّع موزّع عبر خدمات متعددة) ====================
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer("llm-service")

class InstrumentedLLMService:
    """خدمة استدلال كاملة مع مراقبة شاملة مدمجة"""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_with_full_observability(self, prompt: str, user_id: str) -> dict:
        request_id = str(uuid.uuid4())
        start_time = time.time()
        ACTIVE_REQUESTS.inc()

        with tracer.start_as_current_span("llm_inference") as span:
            span.set_attribute("request_id", request_id)
            span.set_attribute("user_id", user_id)
            span.set_attribute("prompt_length", len(prompt))

            try:
                with tracer.start_as_current_span("tokenization"):
                    input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
                    prompt_tokens = input_ids.shape[1]

                with tracer.start_as_current_span("generation"):
                    first_token_time = None
                    output_tokens = []
                    for token in self._stream_generate(input_ids):
                        if first_token_time is None:
                            first_token_time = time.time()
                            ttft = first_token_time - start_time
                            TIME_TO_FIRST_TOKEN.observe(ttft)
                        output_tokens.append(token)

                total_latency = time.time() - start_time
                REQUEST_LATENCY.observe(total_latency)
                REQUEST_COUNT.labels(model="llama-3-8b", status="success").inc()
                TOKENS_GENERATED.inc(len(output_tokens))

                log_inference_structured(InferenceLogEntry(
                    request_id=request_id, timestamp=start_time, model_name="llama-3-8b",
                    prompt_tokens=prompt_tokens, completion_tokens=len(output_tokens),
                    total_latency_ms=total_latency * 1000,
                    time_to_first_token_ms=(first_token_time - start_time) * 1000,
                    user_id=user_id, status="success"
                ))
                span.set_status(Status(StatusCode.OK))
                return {"request_id": request_id, "response": self.tokenizer.decode(output_tokens)}

            except Exception as e:
                REQUEST_COUNT.labels(model="llama-3-8b", status="error").inc()
                log_inference_structured(InferenceLogEntry(
                    request_id=request_id, timestamp=start_time, model_name="llama-3-8b",
                    prompt_tokens=0, completion_tokens=0, total_latency_ms=(time.time()-start_time)*1000,
                    time_to_first_token_ms=0, user_id=user_id, status="error", error_message=str(e)
                ))
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            finally:
                ACTIVE_REQUESTS.dec()

    def _stream_generate(self, input_ids): ...  # توليد تدفقي token-by-token

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-57-ch57-observability.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
مثال قواعد تنبيه Prometheus/Alertmanager لخدمة LLM إنتاجية
"""
ALERTING_RULES_YAML = """
groups:
- name: llm_service_alerts
  rules:
  - alert: HighLatencyP99
    expr: histogram_quantile(0.99, llm_request_duration_seconds_bucket) > 10
    for: 5m
    annotations:
      summary: "زمن الاستجابة عند p99 يتجاوز 10 ثوانٍ لأكثر من 5 دقائق"

  - alert: HighErrorRate
    expr: rate(llm_requests_total{status="error"}[5m]) / rate(llm_requests_total[5m]) > 0.05
    for: 2m
    annotations:
      summary: "معدل الأخطاء يتجاوز 5% — تحقق فورًا"

  - alert: GPUMemoryNearCapacity
    expr: llm_gpu_memory_used_bytes / llm_gpu_memory_total_bytes > 0.95
    for: 1m
    annotations:
      summary: "ذاكرة GPU قريبة من الامتلاء الكامل — خطر OOM وشيك"

  - alert: QueueBacklogGrowing
    expr: llm_active_requests > 50
    for: 3m
    annotations:
      summary: "تراكم طابور الطلبات — قد تحتاج Autoscaling يدويًا أو فحص عنق الزجاجة"
"""

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-57-ch57-observability.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class OutputQualityMonitor:
    """
    مراقبة مستمرة لجودة المخرجات في الإنتاج، تكملة لمراقبة الأداء التقني
    (يبني على مفاهيم الفصل 46 — Continuous Evaluation، لكن بتركيز على الإنتاج الحي)
    """
    def __init__(self, safety_classifier, judge_client, sample_rate=0.02):
        self.safety_classifier = safety_classifier
        self.judge_client = judge_client
        self.sample_rate = sample_rate   # عيّنة من الطلبات فقط (لتقليل التكلفة)
        self.quality_history = []

    def monitor_response(self, prompt: str, response: str, request_id: str):
        # 1. فحص فوري وزهيد لكل طلب: اكتشاف أنماط مقلقة سريعة (طول غير طبيعي، تكرار)
        if self._detect_degenerate_output(response):
            self._raise_alert("DEGENERATE_OUTPUT", request_id)

        # 2. فحص أعمق (أغلى) لعيّنة فقط من الطلبات
        import random
        if random.random() < self.sample_rate:
            safety_result = self.safety_classifier(response)
            if safety_result["is_harmful"]:
                self._raise_alert("HARMFUL_CONTENT_DETECTED", request_id)

            quality_score = llm_judge_evaluate(self.judge_client, prompt, response)
            self.quality_history.append({"request_id": request_id, "score": quality_score["overall"]})

            # كشف انجراف الجودة (Quality Drift) عبر نافذة متحركة
            if len(self.quality_history) >= 100:
                recent_avg = sum(q["score"] for q in self.quality_history[-100:]) / 100
                if recent_avg < 6.0:   # عتبة جودة دنيا
                    self._raise_alert("QUALITY_DRIFT_DETECTED", request_id,
                                        details=f"متوسط الجودة الأخير: {recent_avg}")

    def _detect_degenerate_output(self, response: str) -> bool:
        """كشف سريع لمخرجات متدهورة شائعة: تكرار لا نهائي، فراغ، طول شاذ"""
        words = response.split()
        if len(words) == 0:
            return True
        if len(set(words[-20:])) < 5 and len(words) > 20:   # تكرار شديد في آخر الاستجابة
            return True
        return False

    def _raise_alert(self, alert_type: str, request_id: str, details: str = ""):
        logger.warning(f"⚠️ تنبيه جودة [{alert_type}] للطلب {request_id}: {details}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-57-ch57-observability.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-57-ch57-observability.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — المراقبة والتشغيل: Observability
Mastering Transformers and Large Language Models — Observability

Book slug : mastering-llms-transformers
Chapter   : ch-57-ch57-observability (lesson 57)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-11-llmops/ch57-observability.ipynb#cell-1
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
    ("opentelemetry", "opentelemetry-api opentelemetry-sdk"),
    ("prometheus_client", "prometheus-client")
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
# [002] content/notebooks/part-11-llmops/ch57-observability.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import time
import logging
import uuid
from dataclasses import dataclass, asdict
from prometheus_client import Counter, Histogram, Gauge

# ==================== 1. Metrics (مقاييس كمّية مُجمَّعة) ====================
REQUEST_COUNT = Counter('llm_requests_total', 'إجمالي الطلبات', ['model', 'status'])
REQUEST_LATENCY = Histogram('llm_request_duration_seconds', 'زمن معالجة الطلب',
                              buckets=[0.1, 0.5, 1, 2, 5, 10, 30])
TOKENS_GENERATED = Counter('llm_tokens_generated_total', 'إجمالي الرموز المُولَّدة')
TIME_TO_FIRST_TOKEN = Histogram('llm_ttft_seconds', 'الزمن حتى أول رمز (مهم جدًا لتجربة المستخدم)')
ACTIVE_REQUESTS = Gauge('llm_active_requests', 'عدد الطلبات النشطة حاليًا')
GPU_MEMORY_USAGE = Gauge('llm_gpu_memory_used_bytes', 'استهلاك ذاكرة GPU')

# ==================== 2. Structured Logging (سجلات مُهيكَلة قابلة للاستعلام) ====================
@dataclass
class InferenceLogEntry:
    request_id: str
    timestamp: float
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_latency_ms: float
    time_to_first_token_ms: float
    user_id: str
    status: str
    error_message: str = None

logger = logging.getLogger("llm_service")

def log_inference_structured(entry: InferenceLogEntry):
    """تسجيل بصيغة JSON مُهيكَلة — قابلة للاستعلام في Elasticsearch/Loki/CloudWatch"""
    logger.info(json.dumps(asdict(entry), ensure_ascii=False))

# ==================== 3. Distributed Tracing (تتبّع موزّع عبر خدمات متعددة) ====================
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer("llm-service")

class InstrumentedLLMService:
    """خدمة استدلال كاملة مع مراقبة شاملة مدمجة"""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_with_full_observability(self, prompt: str, user_id: str) -> dict:
        request_id = str(uuid.uuid4())
        start_time = time.time()
        ACTIVE_REQUESTS.inc()

        with tracer.start_as_current_span("llm_inference") as span:
            span.set_attribute("request_id", request_id)
            span.set_attribute("user_id", user_id)
            span.set_attribute("prompt_length", len(prompt))

            try:
                with tracer.start_as_current_span("tokenization"):
                    input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
                    prompt_tokens = input_ids.shape[1]

                with tracer.start_as_current_span("generation"):
                    first_token_time = None
                    output_tokens = []
                    for token in self._stream_generate(input_ids):
                        if first_token_time is None:
                            first_token_time = time.time()
                            ttft = first_token_time - start_time
                            TIME_TO_FIRST_TOKEN.observe(ttft)
                        output_tokens.append(token)

                total_latency = time.time() - start_time
                REQUEST_LATENCY.observe(total_latency)
                REQUEST_COUNT.labels(model="llama-3-8b", status="success").inc()
                TOKENS_GENERATED.inc(len(output_tokens))

                log_inference_structured(InferenceLogEntry(
                    request_id=request_id, timestamp=start_time, model_name="llama-3-8b",
                    prompt_tokens=prompt_tokens, completion_tokens=len(output_tokens),
                    total_latency_ms=total_latency * 1000,
                    time_to_first_token_ms=(first_token_time - start_time) * 1000,
                    user_id=user_id, status="success"
                ))
                span.set_status(Status(StatusCode.OK))
                return {"request_id": request_id, "response": self.tokenizer.decode(output_tokens)}

            except Exception as e:
                REQUEST_COUNT.labels(model="llama-3-8b", status="error").inc()
                log_inference_structured(InferenceLogEntry(
                    request_id=request_id, timestamp=start_time, model_name="llama-3-8b",
                    prompt_tokens=0, completion_tokens=0, total_latency_ms=(time.time()-start_time)*1000,
                    time_to_first_token_ms=0, user_id=user_id, status="error", error_message=str(e)
                ))
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            finally:
                ACTIVE_REQUESTS.dec()

    def _stream_generate(self, input_ids): ...  # توليد تدفقي token-by-token

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-11-llmops/ch57-observability.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
مثال قواعد تنبيه Prometheus/Alertmanager لخدمة LLM إنتاجية
"""
ALERTING_RULES_YAML = """
groups:
- name: llm_service_alerts
  rules:
  - alert: HighLatencyP99
    expr: histogram_quantile(0.99, llm_request_duration_seconds_bucket) > 10
    for: 5m
    annotations:
      summary: "زمن الاستجابة عند p99 يتجاوز 10 ثوانٍ لأكثر من 5 دقائق"

  - alert: HighErrorRate
    expr: rate(llm_requests_total{status="error"}[5m]) / rate(llm_requests_total[5m]) > 0.05
    for: 2m
    annotations:
      summary: "معدل الأخطاء يتجاوز 5% — تحقق فورًا"

  - alert: GPUMemoryNearCapacity
    expr: llm_gpu_memory_used_bytes / llm_gpu_memory_total_bytes > 0.95
    for: 1m
    annotations:
      summary: "ذاكرة GPU قريبة من الامتلاء الكامل — خطر OOM وشيك"

  - alert: QueueBacklogGrowing
    expr: llm_active_requests > 50
    for: 3m
    annotations:
      summary: "تراكم طابور الطلبات — قد تحتاج Autoscaling يدويًا أو فحص عنق الزجاجة"
"""

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-11-llmops/ch57-observability.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class OutputQualityMonitor:
    """
    مراقبة مستمرة لجودة المخرجات في الإنتاج، تكملة لمراقبة الأداء التقني
    (يبني على مفاهيم الفصل 46 — Continuous Evaluation، لكن بتركيز على الإنتاج الحي)
    """
    def __init__(self, safety_classifier, judge_client, sample_rate=0.02):
        self.safety_classifier = safety_classifier
        self.judge_client = judge_client
        self.sample_rate = sample_rate   # عيّنة من الطلبات فقط (لتقليل التكلفة)
        self.quality_history = []

    def monitor_response(self, prompt: str, response: str, request_id: str):
        # 1. فحص فوري وزهيد لكل طلب: اكتشاف أنماط مقلقة سريعة (طول غير طبيعي، تكرار)
        if self._detect_degenerate_output(response):
            self._raise_alert("DEGENERATE_OUTPUT", request_id)

        # 2. فحص أعمق (أغلى) لعيّنة فقط من الطلبات
        import random
        if random.random() < self.sample_rate:
            safety_result = self.safety_classifier(response)
            if safety_result["is_harmful"]:
                self._raise_alert("HARMFUL_CONTENT_DETECTED", request_id)

            quality_score = llm_judge_evaluate(self.judge_client, prompt, response)
            self.quality_history.append({"request_id": request_id, "score": quality_score["overall"]})

            # كشف انجراف الجودة (Quality Drift) عبر نافذة متحركة
            if len(self.quality_history) >= 100:
                recent_avg = sum(q["score"] for q in self.quality_history[-100:]) / 100
                if recent_avg < 6.0:   # عتبة جودة دنيا
                    self._raise_alert("QUALITY_DRIFT_DETECTED", request_id,
                                        details=f"متوسط الجودة الأخير: {recent_avg}")

    def _detect_degenerate_output(self, response: str) -> bool:
        """كشف سريع لمخرجات متدهورة شائعة: تكرار لا نهائي، فراغ، طول شاذ"""
        words = response.split()
        if len(words) == 0:
            return True
        if len(set(words[-20:])) < 5 and len(words) > 20:   # تكرار شديد في آخر الاستجابة
            return True
        return False

    def _raise_alert(self, alert_type: str, request_id: str, details: str = ""):
        logger.warning(f"⚠️ تنبيه جودة [{alert_type}] للطلب {request_id}: {details}")

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-11-llmops/ch57-observability.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-57-ch57-observability.ipynb#cell-2
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
    ("opentelemetry", "opentelemetry-api opentelemetry-sdk"),
    ("prometheus_client", "prometheus-client")
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-57-ch57-observability.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import time
import logging
import uuid
from dataclasses import dataclass, asdict
from prometheus_client import Counter, Histogram, Gauge

# ==================== 1. Metrics (مقاييس كمّية مُجمَّعة) ====================
REQUEST_COUNT = Counter('llm_requests_total', 'إجمالي الطلبات', ['model', 'status'])
REQUEST_LATENCY = Histogram('llm_request_duration_seconds', 'زمن معالجة الطلب',
                              buckets=[0.1, 0.5, 1, 2, 5, 10, 30])
TOKENS_GENERATED = Counter('llm_tokens_generated_total', 'إجمالي الرموز المُولَّدة')
TIME_TO_FIRST_TOKEN = Histogram('llm_ttft_seconds', 'الزمن حتى أول رمز (مهم جدًا لتجربة المستخدم)')
ACTIVE_REQUESTS = Gauge('llm_active_requests', 'عدد الطلبات النشطة حاليًا')
GPU_MEMORY_USAGE = Gauge('llm_gpu_memory_used_bytes', 'استهلاك ذاكرة GPU')

# ==================== 2. Structured Logging (سجلات مُهيكَلة قابلة للاستعلام) ====================
@dataclass
class InferenceLogEntry:
    request_id: str
    timestamp: float
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_latency_ms: float
    time_to_first_token_ms: float
    user_id: str
    status: str
    error_message: str = None

logger = logging.getLogger("llm_service")

def log_inference_structured(entry: InferenceLogEntry):
    """تسجيل بصيغة JSON مُهيكَلة — قابلة للاستعلام في Elasticsearch/Loki/CloudWatch"""
    logger.info(json.dumps(asdict(entry), ensure_ascii=False))

# ==================== 3. Distributed Tracing (تتبّع موزّع عبر خدمات متعددة) ====================
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer("llm-service")

class InstrumentedLLMService:
    """خدمة استدلال كاملة مع مراقبة شاملة مدمجة"""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_with_full_observability(self, prompt: str, user_id: str) -> dict:
        request_id = str(uuid.uuid4())
        start_time = time.time()
        ACTIVE_REQUESTS.inc()

        with tracer.start_as_current_span("llm_inference") as span:
            span.set_attribute("request_id", request_id)
            span.set_attribute("user_id", user_id)
            span.set_attribute("prompt_length", len(prompt))

            try:
                with tracer.start_as_current_span("tokenization"):
                    input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
                    prompt_tokens = input_ids.shape[1]

                with tracer.start_as_current_span("generation"):
                    first_token_time = None
                    output_tokens = []
                    for token in self._stream_generate(input_ids):
                        if first_token_time is None:
                            first_token_time = time.time()
                            ttft = first_token_time - start_time
                            TIME_TO_FIRST_TOKEN.observe(ttft)
                        output_tokens.append(token)

                total_latency = time.time() - start_time
                REQUEST_LATENCY.observe(total_latency)
                REQUEST_COUNT.labels(model="llama-3-8b", status="success").inc()
                TOKENS_GENERATED.inc(len(output_tokens))

                log_inference_structured(InferenceLogEntry(
                    request_id=request_id, timestamp=start_time, model_name="llama-3-8b",
                    prompt_tokens=prompt_tokens, completion_tokens=len(output_tokens),
                    total_latency_ms=total_latency * 1000,
                    time_to_first_token_ms=(first_token_time - start_time) * 1000,
                    user_id=user_id, status="success"
                ))
                span.set_status(Status(StatusCode.OK))
                return {"request_id": request_id, "response": self.tokenizer.decode(output_tokens)}

            except Exception as e:
                REQUEST_COUNT.labels(model="llama-3-8b", status="error").inc()
                log_inference_structured(InferenceLogEntry(
                    request_id=request_id, timestamp=start_time, model_name="llama-3-8b",
                    prompt_tokens=0, completion_tokens=0, total_latency_ms=(time.time()-start_time)*1000,
                    time_to_first_token_ms=0, user_id=user_id, status="error", error_message=str(e)
                ))
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            finally:
                ACTIVE_REQUESTS.dec()

    def _stream_generate(self, input_ids): ...  # توليد تدفقي token-by-token

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-57-ch57-observability.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
"""
مثال قواعد تنبيه Prometheus/Alertmanager لخدمة LLM إنتاجية
"""
ALERTING_RULES_YAML = """
groups:
- name: llm_service_alerts
  rules:
  - alert: HighLatencyP99
    expr: histogram_quantile(0.99, llm_request_duration_seconds_bucket) > 10
    for: 5m
    annotations:
      summary: "زمن الاستجابة عند p99 يتجاوز 10 ثوانٍ لأكثر من 5 دقائق"

  - alert: HighErrorRate
    expr: rate(llm_requests_total{status="error"}[5m]) / rate(llm_requests_total[5m]) > 0.05
    for: 2m
    annotations:
      summary: "معدل الأخطاء يتجاوز 5% — تحقق فورًا"

  - alert: GPUMemoryNearCapacity
    expr: llm_gpu_memory_used_bytes / llm_gpu_memory_total_bytes > 0.95
    for: 1m
    annotations:
      summary: "ذاكرة GPU قريبة من الامتلاء الكامل — خطر OOM وشيك"

  - alert: QueueBacklogGrowing
    expr: llm_active_requests > 50
    for: 3m
    annotations:
      summary: "تراكم طابور الطلبات — قد تحتاج Autoscaling يدويًا أو فحص عنق الزجاجة"
"""

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-57-ch57-observability.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class OutputQualityMonitor:
    """
    مراقبة مستمرة لجودة المخرجات في الإنتاج، تكملة لمراقبة الأداء التقني
    (يبني على مفاهيم الفصل 46 — Continuous Evaluation، لكن بتركيز على الإنتاج الحي)
    """
    def __init__(self, safety_classifier, judge_client, sample_rate=0.02):
        self.safety_classifier = safety_classifier
        self.judge_client = judge_client
        self.sample_rate = sample_rate   # عيّنة من الطلبات فقط (لتقليل التكلفة)
        self.quality_history = []

    def monitor_response(self, prompt: str, response: str, request_id: str):
        # 1. فحص فوري وزهيد لكل طلب: اكتشاف أنماط مقلقة سريعة (طول غير طبيعي، تكرار)
        if self._detect_degenerate_output(response):
            self._raise_alert("DEGENERATE_OUTPUT", request_id)

        # 2. فحص أعمق (أغلى) لعيّنة فقط من الطلبات
        import random
        if random.random() < self.sample_rate:
            safety_result = self.safety_classifier(response)
            if safety_result["is_harmful"]:
                self._raise_alert("HARMFUL_CONTENT_DETECTED", request_id)

            quality_score = llm_judge_evaluate(self.judge_client, prompt, response)
            self.quality_history.append({"request_id": request_id, "score": quality_score["overall"]})

            # كشف انجراف الجودة (Quality Drift) عبر نافذة متحركة
            if len(self.quality_history) >= 100:
                recent_avg = sum(q["score"] for q in self.quality_history[-100:]) / 100
                if recent_avg < 6.0:   # عتبة جودة دنيا
                    self._raise_alert("QUALITY_DRIFT_DETECTED", request_id,
                                        details=f"متوسط الجودة الأخير: {recent_avg}")

    def _detect_degenerate_output(self, response: str) -> bool:
        """كشف سريع لمخرجات متدهورة شائعة: تكرار لا نهائي، فراغ، طول شاذ"""
        words = response.split()
        if len(words) == 0:
            return True
        if len(set(words[-20:])) < 5 and len(words) > 20:   # تكرار شديد في آخر الاستجابة
            return True
        return False

    def _raise_alert(self, alert_type: str, request_id: str, details: str = ""):
        logger.warning(f"⚠️ تنبيه جودة [{alert_type}] للطلب {request_id}: {details}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-57-ch57-observability.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
