#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — إدارة التكلفة ونضج LLMOps
Mastering Transformers and Large Language Models — Cost Optimization

Book slug : mastering-llms-transformers
Chapter   : ch-59-ch59-cost-optimization (lesson 59)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-59-ch59-cost-optimization.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — إدارة التكلفة ونضج LLMOps
Mastering Transformers and Large Language Models — Cost Optimization

Book slug : mastering-llms-transformers
Chapter   : ch-59-ch59-cost-optimization (lesson 59)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-59-ch59-cost-optimization.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — إدارة التكلفة ونضج LLMOps
Mastering Transformers and Large Language Models — Cost Optimization

Book slug : mastering-llms-transformers
Chapter   : ch-59-ch59-cost-optimization (lesson 59)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-11-llmops/ch59-cost-optimization.ipynb#cell-1
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
# [002] content/notebooks/part-11-llmops/ch59-cost-optimization.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
class LLMCostCalculator:
    """حاسبة تكلفة شاملة لمقارنة خيارات النشر المختلفة"""

    def calculate_self_hosted_cost(self, gpu_type: str, num_gpus: int,
                                      hours_per_month: int, requests_per_month: int) -> dict:
        gpu_hourly_rates = {"A100-80GB": 2.5, "H100": 4.5, "A10G": 1.0}   # تقديرية، تختلف حسب المزوّد

        compute_cost = gpu_hourly_rates[gpu_type] * num_gpus * hours_per_month
        cost_per_request = compute_cost / requests_per_month if requests_per_month > 0 else 0

        return {
            "monthly_compute_cost": compute_cost,
            "cost_per_request": cost_per_request,
            "break_even_analysis": "مربح مقارنة بـ API إذا كان الحجم مرتفعًا بثبات"
        }

    def calculate_api_cost(self, provider_rate_per_1k_input: float,
                              provider_rate_per_1k_output: float,
                              avg_input_tokens: int, avg_output_tokens: int,
                              requests_per_month: int) -> dict:
        input_cost = (avg_input_tokens / 1000) * provider_rate_per_1k_input * requests_per_month
        output_cost = (avg_output_tokens / 1000) * provider_rate_per_1k_output * requests_per_month
        total = input_cost + output_cost

        return {
            "monthly_api_cost": total,
            "cost_per_request": total / requests_per_month if requests_per_month > 0 else 0
        }

    def recommend_deployment_strategy(self, requests_per_month: int,
                                         avg_tokens_per_request: int) -> str:
        """دليل قرار مبسّط: متى الاستضافة الذاتية أرخص من استخدام API؟"""
        self_hosted = self.calculate_self_hosted_cost("A100-80GB", 2, 730, requests_per_month)
        api_based = self.calculate_api_cost(0.01, 0.03, avg_tokens_per_request,
                                              avg_tokens_per_request, requests_per_month)

        if self_hosted["monthly_compute_cost"] < api_based["monthly_api_cost"]:
            return f"✅ الاستضافة الذاتية أرخص: ${self_hosted['monthly_compute_cost']:.0f} مقابل ${api_based['monthly_api_cost']:.0f}"
        return f"✅ استخدام API أرخص عند هذا الحجم: ${api_based['monthly_api_cost']:.0f} مقابل ${self_hosted['monthly_compute_cost']:.0f}"

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-11-llmops/ch59-cost-optimization.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class ResponseCache:
    """
    التخزين المؤقت للاستجابات — أبسط وأفعل تقنية لتقليل التكلفة
    خاصة للأسئلة المتكررة (FAQ-style) أو نفس السياق مع System Prompt ثابت
    """
    def __init__(self, embedder, similarity_threshold=0.95, max_cache_size=10000):
        self.embedder = embedder
        self.threshold = similarity_threshold
        self.cache = []   # [{"prompt_embedding": ..., "prompt": ..., "response": ...}]
        self.max_size = max_cache_size
        self.hit_count, self.miss_count = 0, 0

    def get_cached_response(self, prompt: str) -> str:
        """بحث دلالي (Semantic Caching) — وليس تطابقًا حرفيًا فقط"""
        if not self.cache:
            self.miss_count += 1
            return None

        query_embedding = self.embedder.encode([prompt])[0]
        similarities = [
            np.dot(query_embedding, entry["prompt_embedding"]) for entry in self.cache
        ]
        max_sim_idx = np.argmax(similarities)

        if similarities[max_sim_idx] >= self.threshold:
            self.hit_count += 1
            return self.cache[max_sim_idx]["response"]

        self.miss_count += 1
        return None

    def store(self, prompt: str, response: str):
        if len(self.cache) >= self.max_size:
            self.cache.pop(0)   # إزالة الأقدم (يمكن استبدالها بـ LRU حقيقي)
        self.cache.append({
            "prompt_embedding": self.embedder.encode([prompt])[0],
            "prompt": prompt, "response": response
        })

    def get_hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0

class ModelRouter:
    """
    توجيه ذكي بين نماذج مختلفة الحجم/التكلفة حسب تعقيد السؤال
    (نموذج صغير رخيص للأسئلة البسيطة، نموذج كبير فقط للمعقّدة)
    """
    def __init__(self, small_model, large_model, complexity_classifier):
        self.small_model = small_model
        self.large_model = large_model
        self.complexity_classifier = complexity_classifier

    def route_and_generate(self, prompt: str) -> dict:
        complexity_score = self.complexity_classifier(prompt)   # 0 (بسيط) - 1 (معقّد جدًا)

        if complexity_score < 0.3:
            response = self.small_model.generate(prompt)
            model_used = "small (رخيص)"
        else:
            response = self.large_model.generate(prompt)
            model_used = "large (مكلف لكن أدق)"

        return {"response": response, "model_used": model_used, "complexity_score": complexity_score}

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-11-llmops/ch59-cost-optimization.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
"""
خط CI/CD كامل لنموذج LLM — يدمج كل ما تعلمناه في هذا الجزء والجزء التاسع
"""

LLMOPS_PIPELINE_YAML = """
name: llm-deployment-pipeline

stages:
  - stage: data_validation
    steps:
      - run: python validate_training_data.py --check-contamination --check-pii

  - stage: training_or_finetuning
    steps:
      - run: python train.py --config configs/production.yaml
      - run: python save_checkpoint.py --to-model-registry

  - stage: offline_evaluation
    steps:
      - run: python run_evaluation_suite.py --benchmarks mmlu,gsm8k,arabic_mmlu
      - run: python run_safety_redteam.py
      - run: python run_llm_judge_comparison.py --baseline=current_prod_model
      - gate: "REJECTED_SMOKE_TEST_FAILURE أو REGRESSION_DETECTED يوقف الخط فورًا"

  - stage: staging_deployment
    steps:
      - run: deploy_to_staging.sh
      - run: python run_integration_tests.py

  - stage: canary_deployment
    steps:
      - run: deploy_canary.sh --traffic-percentage=5
      - monitor: "لمدة 24 ساعة: latency, error_rate, quality_scores"
      - gate: "أي انحراف إحصائي سلبي يُوقف الزيادة ويُفعّل rollback تلقائي"

  - stage: progressive_rollout
    steps:
      - run: increase_traffic.sh --to=25
      - run: increase_traffic.sh --to=50
      - run: increase_traffic.sh --to=100

  - stage: post_deployment_monitoring
    steps:
      - run: enable_continuous_quality_monitoring.py
      - run: enable_drift_detection.py
"""

class LLMOpsMaturityAssessment:
    """
    إطار تقييم ذاتي لنضج ممارسات LLMOps في المنظمة
    """
    MATURITY_LEVELS = {
        1: "يدوي بالكامل — نشر يدوي، لا مراقبة منهجية، لا نسخ احتياطي للنماذج",
        2: "آلي جزئيًا — CI/CD أساسي، مراقبة تقنية فقط (latency/errors)",
        3: "آلي شامل — تقييم آلي قبل النشر، canary deployment، مراقبة جودة أساسية",
        4: "استباقي — كشف انجراف مستمر، A/B testing منهجي، تحسين تكلفة آلي",
        5: "ذاتي التحسّن — حلقة تغذية راجعة من الإنتاج تُغذّي تحسينات تدريب مستقبلية تلقائيًا"
    }

    def assess(self, organization_practices: dict) -> dict:
        score = 1
        if organization_practices.get("has_cicd"): score = max(score, 2)
        if organization_practices.get("has_automated_evaluation"): score = max(score, 3)
        if organization_practices.get("has_canary_deployment"): score = max(score, 3)
        if organization_practices.get("has_drift_detection"): score = max(score, 4)
        if organization_practices.get("has_feedback_loop_to_training"): score = max(score, 5)

        return {"maturity_level": score, "description": self.MATURITY_LEVELS[score],
                "next_step": self.MATURITY_LEVELS.get(score + 1, "أنت في أعلى مستوى نضج")}

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-11-llmops/ch59-cost-optimization.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-59-ch59-cost-optimization.ipynb#cell-1
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-59-ch59-cost-optimization.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
class LLMCostCalculator:
    """حاسبة تكلفة شاملة لمقارنة خيارات النشر المختلفة"""

    def calculate_self_hosted_cost(self, gpu_type: str, num_gpus: int,
                                      hours_per_month: int, requests_per_month: int) -> dict:
        gpu_hourly_rates = {"A100-80GB": 2.5, "H100": 4.5, "A10G": 1.0}   # تقديرية، تختلف حسب المزوّد

        compute_cost = gpu_hourly_rates[gpu_type] * num_gpus * hours_per_month
        cost_per_request = compute_cost / requests_per_month if requests_per_month > 0 else 0

        return {
            "monthly_compute_cost": compute_cost,
            "cost_per_request": cost_per_request,
            "break_even_analysis": "مربح مقارنة بـ API إذا كان الحجم مرتفعًا بثبات"
        }

    def calculate_api_cost(self, provider_rate_per_1k_input: float,
                              provider_rate_per_1k_output: float,
                              avg_input_tokens: int, avg_output_tokens: int,
                              requests_per_month: int) -> dict:
        input_cost = (avg_input_tokens / 1000) * provider_rate_per_1k_input * requests_per_month
        output_cost = (avg_output_tokens / 1000) * provider_rate_per_1k_output * requests_per_month
        total = input_cost + output_cost

        return {
            "monthly_api_cost": total,
            "cost_per_request": total / requests_per_month if requests_per_month > 0 else 0
        }

    def recommend_deployment_strategy(self, requests_per_month: int,
                                         avg_tokens_per_request: int) -> str:
        """دليل قرار مبسّط: متى الاستضافة الذاتية أرخص من استخدام API؟"""
        self_hosted = self.calculate_self_hosted_cost("A100-80GB", 2, 730, requests_per_month)
        api_based = self.calculate_api_cost(0.01, 0.03, avg_tokens_per_request,
                                              avg_tokens_per_request, requests_per_month)

        if self_hosted["monthly_compute_cost"] < api_based["monthly_api_cost"]:
            return f"✅ الاستضافة الذاتية أرخص: ${self_hosted['monthly_compute_cost']:.0f} مقابل ${api_based['monthly_api_cost']:.0f}"
        return f"✅ استخدام API أرخص عند هذا الحجم: ${api_based['monthly_api_cost']:.0f} مقابل ${self_hosted['monthly_compute_cost']:.0f}"

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-59-ch59-cost-optimization.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class ResponseCache:
    """
    التخزين المؤقت للاستجابات — أبسط وأفعل تقنية لتقليل التكلفة
    خاصة للأسئلة المتكررة (FAQ-style) أو نفس السياق مع System Prompt ثابت
    """
    def __init__(self, embedder, similarity_threshold=0.95, max_cache_size=10000):
        self.embedder = embedder
        self.threshold = similarity_threshold
        self.cache = []   # [{"prompt_embedding": ..., "prompt": ..., "response": ...}]
        self.max_size = max_cache_size
        self.hit_count, self.miss_count = 0, 0

    def get_cached_response(self, prompt: str) -> str:
        """بحث دلالي (Semantic Caching) — وليس تطابقًا حرفيًا فقط"""
        if not self.cache:
            self.miss_count += 1
            return None

        query_embedding = self.embedder.encode([prompt])[0]
        similarities = [
            np.dot(query_embedding, entry["prompt_embedding"]) for entry in self.cache
        ]
        max_sim_idx = np.argmax(similarities)

        if similarities[max_sim_idx] >= self.threshold:
            self.hit_count += 1
            return self.cache[max_sim_idx]["response"]

        self.miss_count += 1
        return None

    def store(self, prompt: str, response: str):
        if len(self.cache) >= self.max_size:
            self.cache.pop(0)   # إزالة الأقدم (يمكن استبدالها بـ LRU حقيقي)
        self.cache.append({
            "prompt_embedding": self.embedder.encode([prompt])[0],
            "prompt": prompt, "response": response
        })

    def get_hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0

class ModelRouter:
    """
    توجيه ذكي بين نماذج مختلفة الحجم/التكلفة حسب تعقيد السؤال
    (نموذج صغير رخيص للأسئلة البسيطة، نموذج كبير فقط للمعقّدة)
    """
    def __init__(self, small_model, large_model, complexity_classifier):
        self.small_model = small_model
        self.large_model = large_model
        self.complexity_classifier = complexity_classifier

    def route_and_generate(self, prompt: str) -> dict:
        complexity_score = self.complexity_classifier(prompt)   # 0 (بسيط) - 1 (معقّد جدًا)

        if complexity_score < 0.3:
            response = self.small_model.generate(prompt)
            model_used = "small (رخيص)"
        else:
            response = self.large_model.generate(prompt)
            model_used = "large (مكلف لكن أدق)"

        return {"response": response, "model_used": model_used, "complexity_score": complexity_score}

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-59-ch59-cost-optimization.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
"""
خط CI/CD كامل لنموذج LLM — يدمج كل ما تعلمناه في هذا الجزء والجزء التاسع
"""

LLMOPS_PIPELINE_YAML = """
name: llm-deployment-pipeline

stages:
  - stage: data_validation
    steps:
      - run: python validate_training_data.py --check-contamination --check-pii

  - stage: training_or_finetuning
    steps:
      - run: python train.py --config configs/production.yaml
      - run: python save_checkpoint.py --to-model-registry

  - stage: offline_evaluation
    steps:
      - run: python run_evaluation_suite.py --benchmarks mmlu,gsm8k,arabic_mmlu
      - run: python run_safety_redteam.py
      - run: python run_llm_judge_comparison.py --baseline=current_prod_model
      - gate: "REJECTED_SMOKE_TEST_FAILURE أو REGRESSION_DETECTED يوقف الخط فورًا"

  - stage: staging_deployment
    steps:
      - run: deploy_to_staging.sh
      - run: python run_integration_tests.py

  - stage: canary_deployment
    steps:
      - run: deploy_canary.sh --traffic-percentage=5
      - monitor: "لمدة 24 ساعة: latency, error_rate, quality_scores"
      - gate: "أي انحراف إحصائي سلبي يُوقف الزيادة ويُفعّل rollback تلقائي"

  - stage: progressive_rollout
    steps:
      - run: increase_traffic.sh --to=25
      - run: increase_traffic.sh --to=50
      - run: increase_traffic.sh --to=100

  - stage: post_deployment_monitoring
    steps:
      - run: enable_continuous_quality_monitoring.py
      - run: enable_drift_detection.py
"""

class LLMOpsMaturityAssessment:
    """
    إطار تقييم ذاتي لنضج ممارسات LLMOps في المنظمة
    """
    MATURITY_LEVELS = {
        1: "يدوي بالكامل — نشر يدوي، لا مراقبة منهجية، لا نسخ احتياطي للنماذج",
        2: "آلي جزئيًا — CI/CD أساسي، مراقبة تقنية فقط (latency/errors)",
        3: "آلي شامل — تقييم آلي قبل النشر، canary deployment، مراقبة جودة أساسية",
        4: "استباقي — كشف انجراف مستمر، A/B testing منهجي، تحسين تكلفة آلي",
        5: "ذاتي التحسّن — حلقة تغذية راجعة من الإنتاج تُغذّي تحسينات تدريب مستقبلية تلقائيًا"
    }

    def assess(self, organization_practices: dict) -> dict:
        score = 1
        if organization_practices.get("has_cicd"): score = max(score, 2)
        if organization_practices.get("has_automated_evaluation"): score = max(score, 3)
        if organization_practices.get("has_canary_deployment"): score = max(score, 3)
        if organization_practices.get("has_drift_detection"): score = max(score, 4)
        if organization_practices.get("has_feedback_loop_to_training"): score = max(score, 5)

        return {"maturity_level": score, "description": self.MATURITY_LEVELS[score],
                "next_step": self.MATURITY_LEVELS.get(score + 1, "أنت في أعلى مستوى نضج")}

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-59-ch59-cost-optimization.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-59-ch59-cost-optimization.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — إدارة التكلفة ونضج LLMOps
Mastering Transformers and Large Language Models — Cost Optimization

Book slug : mastering-llms-transformers
Chapter   : ch-59-ch59-cost-optimization (lesson 59)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-11-llmops/ch59-cost-optimization.ipynb#cell-1
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
# [002] content/notebooks/part-11-llmops/ch59-cost-optimization.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
class LLMCostCalculator:
    """حاسبة تكلفة شاملة لمقارنة خيارات النشر المختلفة"""

    def calculate_self_hosted_cost(self, gpu_type: str, num_gpus: int,
                                      hours_per_month: int, requests_per_month: int) -> dict:
        gpu_hourly_rates = {"A100-80GB": 2.5, "H100": 4.5, "A10G": 1.0}   # تقديرية، تختلف حسب المزوّد

        compute_cost = gpu_hourly_rates[gpu_type] * num_gpus * hours_per_month
        cost_per_request = compute_cost / requests_per_month if requests_per_month > 0 else 0

        return {
            "monthly_compute_cost": compute_cost,
            "cost_per_request": cost_per_request,
            "break_even_analysis": "مربح مقارنة بـ API إذا كان الحجم مرتفعًا بثبات"
        }

    def calculate_api_cost(self, provider_rate_per_1k_input: float,
                              provider_rate_per_1k_output: float,
                              avg_input_tokens: int, avg_output_tokens: int,
                              requests_per_month: int) -> dict:
        input_cost = (avg_input_tokens / 1000) * provider_rate_per_1k_input * requests_per_month
        output_cost = (avg_output_tokens / 1000) * provider_rate_per_1k_output * requests_per_month
        total = input_cost + output_cost

        return {
            "monthly_api_cost": total,
            "cost_per_request": total / requests_per_month if requests_per_month > 0 else 0
        }

    def recommend_deployment_strategy(self, requests_per_month: int,
                                         avg_tokens_per_request: int) -> str:
        """دليل قرار مبسّط: متى الاستضافة الذاتية أرخص من استخدام API؟"""
        self_hosted = self.calculate_self_hosted_cost("A100-80GB", 2, 730, requests_per_month)
        api_based = self.calculate_api_cost(0.01, 0.03, avg_tokens_per_request,
                                              avg_tokens_per_request, requests_per_month)

        if self_hosted["monthly_compute_cost"] < api_based["monthly_api_cost"]:
            return f"✅ الاستضافة الذاتية أرخص: ${self_hosted['monthly_compute_cost']:.0f} مقابل ${api_based['monthly_api_cost']:.0f}"
        return f"✅ استخدام API أرخص عند هذا الحجم: ${api_based['monthly_api_cost']:.0f} مقابل ${self_hosted['monthly_compute_cost']:.0f}"

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-11-llmops/ch59-cost-optimization.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class ResponseCache:
    """
    التخزين المؤقت للاستجابات — أبسط وأفعل تقنية لتقليل التكلفة
    خاصة للأسئلة المتكررة (FAQ-style) أو نفس السياق مع System Prompt ثابت
    """
    def __init__(self, embedder, similarity_threshold=0.95, max_cache_size=10000):
        self.embedder = embedder
        self.threshold = similarity_threshold
        self.cache = []   # [{"prompt_embedding": ..., "prompt": ..., "response": ...}]
        self.max_size = max_cache_size
        self.hit_count, self.miss_count = 0, 0

    def get_cached_response(self, prompt: str) -> str:
        """بحث دلالي (Semantic Caching) — وليس تطابقًا حرفيًا فقط"""
        if not self.cache:
            self.miss_count += 1
            return None

        query_embedding = self.embedder.encode([prompt])[0]
        similarities = [
            np.dot(query_embedding, entry["prompt_embedding"]) for entry in self.cache
        ]
        max_sim_idx = np.argmax(similarities)

        if similarities[max_sim_idx] >= self.threshold:
            self.hit_count += 1
            return self.cache[max_sim_idx]["response"]

        self.miss_count += 1
        return None

    def store(self, prompt: str, response: str):
        if len(self.cache) >= self.max_size:
            self.cache.pop(0)   # إزالة الأقدم (يمكن استبدالها بـ LRU حقيقي)
        self.cache.append({
            "prompt_embedding": self.embedder.encode([prompt])[0],
            "prompt": prompt, "response": response
        })

    def get_hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0

class ModelRouter:
    """
    توجيه ذكي بين نماذج مختلفة الحجم/التكلفة حسب تعقيد السؤال
    (نموذج صغير رخيص للأسئلة البسيطة، نموذج كبير فقط للمعقّدة)
    """
    def __init__(self, small_model, large_model, complexity_classifier):
        self.small_model = small_model
        self.large_model = large_model
        self.complexity_classifier = complexity_classifier

    def route_and_generate(self, prompt: str) -> dict:
        complexity_score = self.complexity_classifier(prompt)   # 0 (بسيط) - 1 (معقّد جدًا)

        if complexity_score < 0.3:
            response = self.small_model.generate(prompt)
            model_used = "small (رخيص)"
        else:
            response = self.large_model.generate(prompt)
            model_used = "large (مكلف لكن أدق)"

        return {"response": response, "model_used": model_used, "complexity_score": complexity_score}

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-11-llmops/ch59-cost-optimization.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
"""
خط CI/CD كامل لنموذج LLM — يدمج كل ما تعلمناه في هذا الجزء والجزء التاسع
"""

LLMOPS_PIPELINE_YAML = """
name: llm-deployment-pipeline

stages:
  - stage: data_validation
    steps:
      - run: python validate_training_data.py --check-contamination --check-pii

  - stage: training_or_finetuning
    steps:
      - run: python train.py --config configs/production.yaml
      - run: python save_checkpoint.py --to-model-registry

  - stage: offline_evaluation
    steps:
      - run: python run_evaluation_suite.py --benchmarks mmlu,gsm8k,arabic_mmlu
      - run: python run_safety_redteam.py
      - run: python run_llm_judge_comparison.py --baseline=current_prod_model
      - gate: "REJECTED_SMOKE_TEST_FAILURE أو REGRESSION_DETECTED يوقف الخط فورًا"

  - stage: staging_deployment
    steps:
      - run: deploy_to_staging.sh
      - run: python run_integration_tests.py

  - stage: canary_deployment
    steps:
      - run: deploy_canary.sh --traffic-percentage=5
      - monitor: "لمدة 24 ساعة: latency, error_rate, quality_scores"
      - gate: "أي انحراف إحصائي سلبي يُوقف الزيادة ويُفعّل rollback تلقائي"

  - stage: progressive_rollout
    steps:
      - run: increase_traffic.sh --to=25
      - run: increase_traffic.sh --to=50
      - run: increase_traffic.sh --to=100

  - stage: post_deployment_monitoring
    steps:
      - run: enable_continuous_quality_monitoring.py
      - run: enable_drift_detection.py
"""

class LLMOpsMaturityAssessment:
    """
    إطار تقييم ذاتي لنضج ممارسات LLMOps في المنظمة
    """
    MATURITY_LEVELS = {
        1: "يدوي بالكامل — نشر يدوي، لا مراقبة منهجية، لا نسخ احتياطي للنماذج",
        2: "آلي جزئيًا — CI/CD أساسي، مراقبة تقنية فقط (latency/errors)",
        3: "آلي شامل — تقييم آلي قبل النشر، canary deployment، مراقبة جودة أساسية",
        4: "استباقي — كشف انجراف مستمر، A/B testing منهجي، تحسين تكلفة آلي",
        5: "ذاتي التحسّن — حلقة تغذية راجعة من الإنتاج تُغذّي تحسينات تدريب مستقبلية تلقائيًا"
    }

    def assess(self, organization_practices: dict) -> dict:
        score = 1
        if organization_practices.get("has_cicd"): score = max(score, 2)
        if organization_practices.get("has_automated_evaluation"): score = max(score, 3)
        if organization_practices.get("has_canary_deployment"): score = max(score, 3)
        if organization_practices.get("has_drift_detection"): score = max(score, 4)
        if organization_practices.get("has_feedback_loop_to_training"): score = max(score, 5)

        return {"maturity_level": score, "description": self.MATURITY_LEVELS[score],
                "next_step": self.MATURITY_LEVELS.get(score + 1, "أنت في أعلى مستوى نضج")}

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-11-llmops/ch59-cost-optimization.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-59-ch59-cost-optimization.ipynb#cell-2
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-59-ch59-cost-optimization.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class LLMCostCalculator:
    """حاسبة تكلفة شاملة لمقارنة خيارات النشر المختلفة"""

    def calculate_self_hosted_cost(self, gpu_type: str, num_gpus: int,
                                      hours_per_month: int, requests_per_month: int) -> dict:
        gpu_hourly_rates = {"A100-80GB": 2.5, "H100": 4.5, "A10G": 1.0}   # تقديرية، تختلف حسب المزوّد

        compute_cost = gpu_hourly_rates[gpu_type] * num_gpus * hours_per_month
        cost_per_request = compute_cost / requests_per_month if requests_per_month > 0 else 0

        return {
            "monthly_compute_cost": compute_cost,
            "cost_per_request": cost_per_request,
            "break_even_analysis": "مربح مقارنة بـ API إذا كان الحجم مرتفعًا بثبات"
        }

    def calculate_api_cost(self, provider_rate_per_1k_input: float,
                              provider_rate_per_1k_output: float,
                              avg_input_tokens: int, avg_output_tokens: int,
                              requests_per_month: int) -> dict:
        input_cost = (avg_input_tokens / 1000) * provider_rate_per_1k_input * requests_per_month
        output_cost = (avg_output_tokens / 1000) * provider_rate_per_1k_output * requests_per_month
        total = input_cost + output_cost

        return {
            "monthly_api_cost": total,
            "cost_per_request": total / requests_per_month if requests_per_month > 0 else 0
        }

    def recommend_deployment_strategy(self, requests_per_month: int,
                                         avg_tokens_per_request: int) -> str:
        """دليل قرار مبسّط: متى الاستضافة الذاتية أرخص من استخدام API؟"""
        self_hosted = self.calculate_self_hosted_cost("A100-80GB", 2, 730, requests_per_month)
        api_based = self.calculate_api_cost(0.01, 0.03, avg_tokens_per_request,
                                              avg_tokens_per_request, requests_per_month)

        if self_hosted["monthly_compute_cost"] < api_based["monthly_api_cost"]:
            return f"✅ الاستضافة الذاتية أرخص: ${self_hosted['monthly_compute_cost']:.0f} مقابل ${api_based['monthly_api_cost']:.0f}"
        return f"✅ استخدام API أرخص عند هذا الحجم: ${api_based['monthly_api_cost']:.0f} مقابل ${self_hosted['monthly_compute_cost']:.0f}"

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-59-ch59-cost-optimization.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class ResponseCache:
    """
    التخزين المؤقت للاستجابات — أبسط وأفعل تقنية لتقليل التكلفة
    خاصة للأسئلة المتكررة (FAQ-style) أو نفس السياق مع System Prompt ثابت
    """
    def __init__(self, embedder, similarity_threshold=0.95, max_cache_size=10000):
        self.embedder = embedder
        self.threshold = similarity_threshold
        self.cache = []   # [{"prompt_embedding": ..., "prompt": ..., "response": ...}]
        self.max_size = max_cache_size
        self.hit_count, self.miss_count = 0, 0

    def get_cached_response(self, prompt: str) -> str:
        """بحث دلالي (Semantic Caching) — وليس تطابقًا حرفيًا فقط"""
        if not self.cache:
            self.miss_count += 1
            return None

        query_embedding = self.embedder.encode([prompt])[0]
        similarities = [
            np.dot(query_embedding, entry["prompt_embedding"]) for entry in self.cache
        ]
        max_sim_idx = np.argmax(similarities)

        if similarities[max_sim_idx] >= self.threshold:
            self.hit_count += 1
            return self.cache[max_sim_idx]["response"]

        self.miss_count += 1
        return None

    def store(self, prompt: str, response: str):
        if len(self.cache) >= self.max_size:
            self.cache.pop(0)   # إزالة الأقدم (يمكن استبدالها بـ LRU حقيقي)
        self.cache.append({
            "prompt_embedding": self.embedder.encode([prompt])[0],
            "prompt": prompt, "response": response
        })

    def get_hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0

class ModelRouter:
    """
    توجيه ذكي بين نماذج مختلفة الحجم/التكلفة حسب تعقيد السؤال
    (نموذج صغير رخيص للأسئلة البسيطة، نموذج كبير فقط للمعقّدة)
    """
    def __init__(self, small_model, large_model, complexity_classifier):
        self.small_model = small_model
        self.large_model = large_model
        self.complexity_classifier = complexity_classifier

    def route_and_generate(self, prompt: str) -> dict:
        complexity_score = self.complexity_classifier(prompt)   # 0 (بسيط) - 1 (معقّد جدًا)

        if complexity_score < 0.3:
            response = self.small_model.generate(prompt)
            model_used = "small (رخيص)"
        else:
            response = self.large_model.generate(prompt)
            model_used = "large (مكلف لكن أدق)"

        return {"response": response, "model_used": model_used, "complexity_score": complexity_score}

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-59-ch59-cost-optimization.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
"""
خط CI/CD كامل لنموذج LLM — يدمج كل ما تعلمناه في هذا الجزء والجزء التاسع
"""

LLMOPS_PIPELINE_YAML = """
name: llm-deployment-pipeline

stages:
  - stage: data_validation
    steps:
      - run: python validate_training_data.py --check-contamination --check-pii

  - stage: training_or_finetuning
    steps:
      - run: python train.py --config configs/production.yaml
      - run: python save_checkpoint.py --to-model-registry

  - stage: offline_evaluation
    steps:
      - run: python run_evaluation_suite.py --benchmarks mmlu,gsm8k,arabic_mmlu
      - run: python run_safety_redteam.py
      - run: python run_llm_judge_comparison.py --baseline=current_prod_model
      - gate: "REJECTED_SMOKE_TEST_FAILURE أو REGRESSION_DETECTED يوقف الخط فورًا"

  - stage: staging_deployment
    steps:
      - run: deploy_to_staging.sh
      - run: python run_integration_tests.py

  - stage: canary_deployment
    steps:
      - run: deploy_canary.sh --traffic-percentage=5
      - monitor: "لمدة 24 ساعة: latency, error_rate, quality_scores"
      - gate: "أي انحراف إحصائي سلبي يُوقف الزيادة ويُفعّل rollback تلقائي"

  - stage: progressive_rollout
    steps:
      - run: increase_traffic.sh --to=25
      - run: increase_traffic.sh --to=50
      - run: increase_traffic.sh --to=100

  - stage: post_deployment_monitoring
    steps:
      - run: enable_continuous_quality_monitoring.py
      - run: enable_drift_detection.py
"""

class LLMOpsMaturityAssessment:
    """
    إطار تقييم ذاتي لنضج ممارسات LLMOps في المنظمة
    """
    MATURITY_LEVELS = {
        1: "يدوي بالكامل — نشر يدوي، لا مراقبة منهجية، لا نسخ احتياطي للنماذج",
        2: "آلي جزئيًا — CI/CD أساسي، مراقبة تقنية فقط (latency/errors)",
        3: "آلي شامل — تقييم آلي قبل النشر، canary deployment، مراقبة جودة أساسية",
        4: "استباقي — كشف انجراف مستمر، A/B testing منهجي، تحسين تكلفة آلي",
        5: "ذاتي التحسّن — حلقة تغذية راجعة من الإنتاج تُغذّي تحسينات تدريب مستقبلية تلقائيًا"
    }

    def assess(self, organization_practices: dict) -> dict:
        score = 1
        if organization_practices.get("has_cicd"): score = max(score, 2)
        if organization_practices.get("has_automated_evaluation"): score = max(score, 3)
        if organization_practices.get("has_canary_deployment"): score = max(score, 3)
        if organization_practices.get("has_drift_detection"): score = max(score, 4)
        if organization_practices.get("has_feedback_loop_to_training"): score = max(score, 5)

        return {"maturity_level": score, "description": self.MATURITY_LEVELS[score],
                "next_step": self.MATURITY_LEVELS.get(score + 1, "أنت في أعلى مستوى نضج")}

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-59-ch59-cost-optimization.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
