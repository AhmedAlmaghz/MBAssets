#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الأمان والحوكمة في الإنتاج
Mastering Transformers and Large Language Models — Security Governance

Book slug : mastering-llms-transformers
Chapter   : ch-58-ch58-security-governance (lesson 58)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-58-ch58-security-governance.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الأمان والحوكمة في الإنتاج
Mastering Transformers and Large Language Models — Security Governance

Book slug : mastering-llms-transformers
Chapter   : ch-58-ch58-security-governance (lesson 58)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-58-ch58-security-governance.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الأمان والحوكمة في الإنتاج
Mastering Transformers and Large Language Models — Security Governance

Book slug : mastering-llms-transformers
Chapter   : ch-58-ch58-security-governance (lesson 58)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-11-llmops/ch58-security-governance.ipynb#cell-1
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
# [002] content/notebooks/part-11-llmops/ch58-security-governance.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import hashlib
import time
from collections import defaultdict

class RateLimiter:
    """
    تحديد معدل الطلبات لمنع إساءة الاستخدام واستنزاف الموارد
    خوارزمية Token Bucket — مرنة وفعّالة
    """
    def __init__(self, max_tokens=100, refill_rate=10):   # 10 طلبات/ثانية لكل مستخدم
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self.buckets = defaultdict(lambda: {"tokens": max_tokens, "last_refill": time.time()})

    def allow_request(self, user_id: str, cost: int = 1) -> bool:
        bucket = self.buckets[user_id]
        now = time.time()
        elapsed = now - bucket["last_refill"]
        bucket["tokens"] = min(self.max_tokens, bucket["tokens"] + elapsed * self.refill_rate)
        bucket["last_refill"] = now

        if bucket["tokens"] >= cost:
            bucket["tokens"] -= cost
            return True
        return False

class APIKeyAuthenticator:
    """مصادقة ومفاتيح API مع صلاحيات متدرجة (Tiered Access)"""
    def __init__(self):
        self.api_keys = {}   # {hashed_key: {"tier": "free/pro/enterprise", "quota": int}}

    def generate_api_key(self, user_id: str, tier: str) -> str:
        raw_key = f"{user_id}_{time.time()}_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}"
        hashed = hashlib.sha256(raw_key.encode()).hexdigest()
        quotas = {"free": 100, "pro": 10000, "enterprise": 1000000}
        self.api_keys[hashed] = {"user_id": user_id, "tier": tier, "quota": quotas[tier], "used": 0}
        return raw_key

    def validate_and_consume(self, raw_key: str, cost: int = 1) -> dict:
        hashed = hashlib.sha256(raw_key.encode()).hexdigest()
        if hashed not in self.api_keys:
            return {"valid": False, "reason": "مفتاح API غير صالح"}

        key_info = self.api_keys[hashed]
        if key_info["used"] + cost > key_info["quota"]:
            return {"valid": False, "reason": "تم تجاوز الحصة المسموحة"}

        key_info["used"] += cost
        return {"valid": True, "remaining_quota": key_info["quota"] - key_info["used"]}

def sanitize_input_prompt(prompt: str, max_length=10000) -> dict:
    """تنظيف وفحص أولي للمدخلات قبل الوصول للنموذج"""
    if len(prompt) > max_length:
        return {"safe": False, "reason": f"المدخل يتجاوز الحد الأقصى ({max_length} حرف)"}

    # كشف أنماط هجوم معروفة (قائمة مبسّطة، في الإنتاج تُستخدم نماذج تصنيف متخصصة)
    suspicious_patterns = ["ignore previous instructions", "system prompt", "tجاهل التعليمات"]
    if any(pattern.lower() in prompt.lower() for pattern in suspicious_patterns):
        return {"safe": False, "reason": "نمط حقن تعليمات محتمل", "flagged": True}

    return {"safe": True}

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-11-llmops/ch58-security-governance.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class DataGovernanceLayer:
    """
    طبقة حوكمة تضمن الامتثال لمتطلبات الخصوصية (GDPR وما شابه)
    """
    def __init__(self, pii_detector, encryption_service):
        self.pii_detector = pii_detector
        self.encryption_service = encryption_service
        self.user_data_registry = {}   # لتتبّع أي بيانات مُخزَّنة لأي مستخدم (للحق في الحذف)

    def process_before_storage(self, user_id: str, interaction_data: dict) -> dict:
        """معالجة البيانات قبل أي تخزين دائم"""
        # 1. كشف وتنبيه عند وجود PII غير متوقع
        pii_found = self.pii_detector(interaction_data["user_message"])
        if pii_found["has_pii"]:
            interaction_data["user_message"] = self._redact_pii(
                interaction_data["user_message"], pii_found["entities"]
            )
            interaction_data["pii_redacted"] = True

        # 2. تشفير البيانات الحساسة في حالة السكون (Encryption at Rest)
        encrypted_data = self.encryption_service.encrypt(json.dumps(interaction_data))

        # 3. تسجيل في سجل تتبّع بيانات المستخدم (لأغراض الحق في الحذف لاحقًا)
        self.user_data_registry.setdefault(user_id, []).append({
            "timestamp": time.time(), "data_ref": encrypted_data
        })
        return {"stored": True, "pii_redacted": interaction_data.get("pii_redacted", False)}

    def handle_right_to_deletion(self, user_id: str) -> dict:
        """تنفيذ "الحق في النسيان" — حذف كل بيانات مستخدم معيّن عند الطلب"""
        if user_id in self.user_data_registry:
            records_count = len(self.user_data_registry[user_id])
            del self.user_data_registry[user_id]
            # في التطبيق الفعلي: حذف فعلي من قاعدة البيانات، الذاكرة طويلة المدى (الفصل 51)، إلخ
            return {"deleted": True, "records_removed": records_count}
        return {"deleted": False, "reason": "لا توجد بيانات مُسجَّلة لهذا المستخدم"}

    def _redact_pii(self, text: str, entities: list) -> str:
        for entity in entities:
            text = text.replace(entity["text"], f"[{entity['type']}_محجوب]")
        return text

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-11-llmops/ch58-security-governance.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class ModelExtractionDefense:
    """
    دفاعات ضد محاولات "سرقة" النموذج عبر الاستعلام المكثّف
    (Model Extraction/Stealing Attacks) — مهاجم يحاول بناء نموذج مُقلِّد
    عبر إرسال عدد هائل من الاستعلامات وتدريب نموذج بديل على المخرجات
    """
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.query_patterns = defaultdict(list)

    def detect_extraction_attempt(self, user_id: str, prompt: str) -> dict:
        self.query_patterns[user_id].append({"prompt": prompt, "time": time.time()})

        recent_queries = [q for q in self.query_patterns[user_id]
                           if time.time() - q["time"] < 3600]

        # مؤشرات شائعة لمحاولة استخراج نموذج:
        # 1. حجم استعلامات غير طبيعي في فترة قصيرة
        if len(recent_queries) > 5000:
            return {"suspicious": True, "reason": "حجم استعلامات غير طبيعي"}

        # 2. أنماط منهجية (استعلامات مصمَّمة لتغطية مساحة واسعة من المدخلات بشكل ممنهج)
        unique_prompts_ratio = len(set(q["prompt"][:50] for q in recent_queries)) / max(len(recent_queries), 1)
        if len(recent_queries) > 500 and unique_prompts_ratio > 0.95:
            return {"suspicious": True, "reason": "نمط استكشاف ممنهج للمساحة المدخلة"}

        return {"suspicious": False}

    def add_output_watermark(self, generated_text: str, watermark_key: str) -> str:
        """
        توضيح مفاهيمي لفكرة العلامة المائية للنصوص المُولَّدة (Text Watermarking)
        (نهج مستوحى من أبحاث مثل Kirchenbauer et al.)
        تُستخدم لاحقًا لإثبات أن نصًا معيّنًا صدر عن هذا النموذج تحديدًا
        """
        # في التطبيق الحقيقي: تحيّز اختيار الرموز بشكل خفي وقابل للكشف إحصائيًا لاحقًا
        # باستخدام دالة hash تعتمد على watermark_key والسياق السابق لكل رمز
        pass   # تطبيق كامل يتطلب تعديل دالة sampling نفسها أثناء التوليد

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-11-llmops/ch58-security-governance.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-58-ch58-security-governance.ipynb#cell-1
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-58-ch58-security-governance.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import hashlib
import time
from collections import defaultdict

class RateLimiter:
    """
    تحديد معدل الطلبات لمنع إساءة الاستخدام واستنزاف الموارد
    خوارزمية Token Bucket — مرنة وفعّالة
    """
    def __init__(self, max_tokens=100, refill_rate=10):   # 10 طلبات/ثانية لكل مستخدم
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self.buckets = defaultdict(lambda: {"tokens": max_tokens, "last_refill": time.time()})

    def allow_request(self, user_id: str, cost: int = 1) -> bool:
        bucket = self.buckets[user_id]
        now = time.time()
        elapsed = now - bucket["last_refill"]
        bucket["tokens"] = min(self.max_tokens, bucket["tokens"] + elapsed * self.refill_rate)
        bucket["last_refill"] = now

        if bucket["tokens"] >= cost:
            bucket["tokens"] -= cost
            return True
        return False

class APIKeyAuthenticator:
    """مصادقة ومفاتيح API مع صلاحيات متدرجة (Tiered Access)"""
    def __init__(self):
        self.api_keys = {}   # {hashed_key: {"tier": "free/pro/enterprise", "quota": int}}

    def generate_api_key(self, user_id: str, tier: str) -> str:
        raw_key = f"{user_id}_{time.time()}_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}"
        hashed = hashlib.sha256(raw_key.encode()).hexdigest()
        quotas = {"free": 100, "pro": 10000, "enterprise": 1000000}
        self.api_keys[hashed] = {"user_id": user_id, "tier": tier, "quota": quotas[tier], "used": 0}
        return raw_key

    def validate_and_consume(self, raw_key: str, cost: int = 1) -> dict:
        hashed = hashlib.sha256(raw_key.encode()).hexdigest()
        if hashed not in self.api_keys:
            return {"valid": False, "reason": "مفتاح API غير صالح"}

        key_info = self.api_keys[hashed]
        if key_info["used"] + cost > key_info["quota"]:
            return {"valid": False, "reason": "تم تجاوز الحصة المسموحة"}

        key_info["used"] += cost
        return {"valid": True, "remaining_quota": key_info["quota"] - key_info["used"]}

def sanitize_input_prompt(prompt: str, max_length=10000) -> dict:
    """تنظيف وفحص أولي للمدخلات قبل الوصول للنموذج"""
    if len(prompt) > max_length:
        return {"safe": False, "reason": f"المدخل يتجاوز الحد الأقصى ({max_length} حرف)"}

    # كشف أنماط هجوم معروفة (قائمة مبسّطة، في الإنتاج تُستخدم نماذج تصنيف متخصصة)
    suspicious_patterns = ["ignore previous instructions", "system prompt", "tجاهل التعليمات"]
    if any(pattern.lower() in prompt.lower() for pattern in suspicious_patterns):
        return {"safe": False, "reason": "نمط حقن تعليمات محتمل", "flagged": True}

    return {"safe": True}

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-58-ch58-security-governance.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class DataGovernanceLayer:
    """
    طبقة حوكمة تضمن الامتثال لمتطلبات الخصوصية (GDPR وما شابه)
    """
    def __init__(self, pii_detector, encryption_service):
        self.pii_detector = pii_detector
        self.encryption_service = encryption_service
        self.user_data_registry = {}   # لتتبّع أي بيانات مُخزَّنة لأي مستخدم (للحق في الحذف)

    def process_before_storage(self, user_id: str, interaction_data: dict) -> dict:
        """معالجة البيانات قبل أي تخزين دائم"""
        # 1. كشف وتنبيه عند وجود PII غير متوقع
        pii_found = self.pii_detector(interaction_data["user_message"])
        if pii_found["has_pii"]:
            interaction_data["user_message"] = self._redact_pii(
                interaction_data["user_message"], pii_found["entities"]
            )
            interaction_data["pii_redacted"] = True

        # 2. تشفير البيانات الحساسة في حالة السكون (Encryption at Rest)
        encrypted_data = self.encryption_service.encrypt(json.dumps(interaction_data))

        # 3. تسجيل في سجل تتبّع بيانات المستخدم (لأغراض الحق في الحذف لاحقًا)
        self.user_data_registry.setdefault(user_id, []).append({
            "timestamp": time.time(), "data_ref": encrypted_data
        })
        return {"stored": True, "pii_redacted": interaction_data.get("pii_redacted", False)}

    def handle_right_to_deletion(self, user_id: str) -> dict:
        """تنفيذ "الحق في النسيان" — حذف كل بيانات مستخدم معيّن عند الطلب"""
        if user_id in self.user_data_registry:
            records_count = len(self.user_data_registry[user_id])
            del self.user_data_registry[user_id]
            # في التطبيق الفعلي: حذف فعلي من قاعدة البيانات، الذاكرة طويلة المدى (الفصل 51)، إلخ
            return {"deleted": True, "records_removed": records_count}
        return {"deleted": False, "reason": "لا توجد بيانات مُسجَّلة لهذا المستخدم"}

    def _redact_pii(self, text: str, entities: list) -> str:
        for entity in entities:
            text = text.replace(entity["text"], f"[{entity['type']}_محجوب]")
        return text

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-58-ch58-security-governance.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class ModelExtractionDefense:
    """
    دفاعات ضد محاولات "سرقة" النموذج عبر الاستعلام المكثّف
    (Model Extraction/Stealing Attacks) — مهاجم يحاول بناء نموذج مُقلِّد
    عبر إرسال عدد هائل من الاستعلامات وتدريب نموذج بديل على المخرجات
    """
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.query_patterns = defaultdict(list)

    def detect_extraction_attempt(self, user_id: str, prompt: str) -> dict:
        self.query_patterns[user_id].append({"prompt": prompt, "time": time.time()})

        recent_queries = [q for q in self.query_patterns[user_id]
                           if time.time() - q["time"] < 3600]

        # مؤشرات شائعة لمحاولة استخراج نموذج:
        # 1. حجم استعلامات غير طبيعي في فترة قصيرة
        if len(recent_queries) > 5000:
            return {"suspicious": True, "reason": "حجم استعلامات غير طبيعي"}

        # 2. أنماط منهجية (استعلامات مصمَّمة لتغطية مساحة واسعة من المدخلات بشكل ممنهج)
        unique_prompts_ratio = len(set(q["prompt"][:50] for q in recent_queries)) / max(len(recent_queries), 1)
        if len(recent_queries) > 500 and unique_prompts_ratio > 0.95:
            return {"suspicious": True, "reason": "نمط استكشاف ممنهج للمساحة المدخلة"}

        return {"suspicious": False}

    def add_output_watermark(self, generated_text: str, watermark_key: str) -> str:
        """
        توضيح مفاهيمي لفكرة العلامة المائية للنصوص المُولَّدة (Text Watermarking)
        (نهج مستوحى من أبحاث مثل Kirchenbauer et al.)
        تُستخدم لاحقًا لإثبات أن نصًا معيّنًا صدر عن هذا النموذج تحديدًا
        """
        # في التطبيق الحقيقي: تحيّز اختيار الرموز بشكل خفي وقابل للكشف إحصائيًا لاحقًا
        # باستخدام دالة hash تعتمد على watermark_key والسياق السابق لكل رمز
        pass   # تطبيق كامل يتطلب تعديل دالة sampling نفسها أثناء التوليد

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-58-ch58-security-governance.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-58-ch58-security-governance.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الأمان والحوكمة في الإنتاج
Mastering Transformers and Large Language Models — Security Governance

Book slug : mastering-llms-transformers
Chapter   : ch-58-ch58-security-governance (lesson 58)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-11-llmops/ch58-security-governance.ipynb#cell-1
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
# [002] content/notebooks/part-11-llmops/ch58-security-governance.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import hashlib
import time
from collections import defaultdict

class RateLimiter:
    """
    تحديد معدل الطلبات لمنع إساءة الاستخدام واستنزاف الموارد
    خوارزمية Token Bucket — مرنة وفعّالة
    """
    def __init__(self, max_tokens=100, refill_rate=10):   # 10 طلبات/ثانية لكل مستخدم
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self.buckets = defaultdict(lambda: {"tokens": max_tokens, "last_refill": time.time()})

    def allow_request(self, user_id: str, cost: int = 1) -> bool:
        bucket = self.buckets[user_id]
        now = time.time()
        elapsed = now - bucket["last_refill"]
        bucket["tokens"] = min(self.max_tokens, bucket["tokens"] + elapsed * self.refill_rate)
        bucket["last_refill"] = now

        if bucket["tokens"] >= cost:
            bucket["tokens"] -= cost
            return True
        return False

class APIKeyAuthenticator:
    """مصادقة ومفاتيح API مع صلاحيات متدرجة (Tiered Access)"""
    def __init__(self):
        self.api_keys = {}   # {hashed_key: {"tier": "free/pro/enterprise", "quota": int}}

    def generate_api_key(self, user_id: str, tier: str) -> str:
        raw_key = f"{user_id}_{time.time()}_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}"
        hashed = hashlib.sha256(raw_key.encode()).hexdigest()
        quotas = {"free": 100, "pro": 10000, "enterprise": 1000000}
        self.api_keys[hashed] = {"user_id": user_id, "tier": tier, "quota": quotas[tier], "used": 0}
        return raw_key

    def validate_and_consume(self, raw_key: str, cost: int = 1) -> dict:
        hashed = hashlib.sha256(raw_key.encode()).hexdigest()
        if hashed not in self.api_keys:
            return {"valid": False, "reason": "مفتاح API غير صالح"}

        key_info = self.api_keys[hashed]
        if key_info["used"] + cost > key_info["quota"]:
            return {"valid": False, "reason": "تم تجاوز الحصة المسموحة"}

        key_info["used"] += cost
        return {"valid": True, "remaining_quota": key_info["quota"] - key_info["used"]}

def sanitize_input_prompt(prompt: str, max_length=10000) -> dict:
    """تنظيف وفحص أولي للمدخلات قبل الوصول للنموذج"""
    if len(prompt) > max_length:
        return {"safe": False, "reason": f"المدخل يتجاوز الحد الأقصى ({max_length} حرف)"}

    # كشف أنماط هجوم معروفة (قائمة مبسّطة، في الإنتاج تُستخدم نماذج تصنيف متخصصة)
    suspicious_patterns = ["ignore previous instructions", "system prompt", "tجاهل التعليمات"]
    if any(pattern.lower() in prompt.lower() for pattern in suspicious_patterns):
        return {"safe": False, "reason": "نمط حقن تعليمات محتمل", "flagged": True}

    return {"safe": True}

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-11-llmops/ch58-security-governance.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class DataGovernanceLayer:
    """
    طبقة حوكمة تضمن الامتثال لمتطلبات الخصوصية (GDPR وما شابه)
    """
    def __init__(self, pii_detector, encryption_service):
        self.pii_detector = pii_detector
        self.encryption_service = encryption_service
        self.user_data_registry = {}   # لتتبّع أي بيانات مُخزَّنة لأي مستخدم (للحق في الحذف)

    def process_before_storage(self, user_id: str, interaction_data: dict) -> dict:
        """معالجة البيانات قبل أي تخزين دائم"""
        # 1. كشف وتنبيه عند وجود PII غير متوقع
        pii_found = self.pii_detector(interaction_data["user_message"])
        if pii_found["has_pii"]:
            interaction_data["user_message"] = self._redact_pii(
                interaction_data["user_message"], pii_found["entities"]
            )
            interaction_data["pii_redacted"] = True

        # 2. تشفير البيانات الحساسة في حالة السكون (Encryption at Rest)
        encrypted_data = self.encryption_service.encrypt(json.dumps(interaction_data))

        # 3. تسجيل في سجل تتبّع بيانات المستخدم (لأغراض الحق في الحذف لاحقًا)
        self.user_data_registry.setdefault(user_id, []).append({
            "timestamp": time.time(), "data_ref": encrypted_data
        })
        return {"stored": True, "pii_redacted": interaction_data.get("pii_redacted", False)}

    def handle_right_to_deletion(self, user_id: str) -> dict:
        """تنفيذ "الحق في النسيان" — حذف كل بيانات مستخدم معيّن عند الطلب"""
        if user_id in self.user_data_registry:
            records_count = len(self.user_data_registry[user_id])
            del self.user_data_registry[user_id]
            # في التطبيق الفعلي: حذف فعلي من قاعدة البيانات، الذاكرة طويلة المدى (الفصل 51)، إلخ
            return {"deleted": True, "records_removed": records_count}
        return {"deleted": False, "reason": "لا توجد بيانات مُسجَّلة لهذا المستخدم"}

    def _redact_pii(self, text: str, entities: list) -> str:
        for entity in entities:
            text = text.replace(entity["text"], f"[{entity['type']}_محجوب]")
        return text

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-11-llmops/ch58-security-governance.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class ModelExtractionDefense:
    """
    دفاعات ضد محاولات "سرقة" النموذج عبر الاستعلام المكثّف
    (Model Extraction/Stealing Attacks) — مهاجم يحاول بناء نموذج مُقلِّد
    عبر إرسال عدد هائل من الاستعلامات وتدريب نموذج بديل على المخرجات
    """
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.query_patterns = defaultdict(list)

    def detect_extraction_attempt(self, user_id: str, prompt: str) -> dict:
        self.query_patterns[user_id].append({"prompt": prompt, "time": time.time()})

        recent_queries = [q for q in self.query_patterns[user_id]
                           if time.time() - q["time"] < 3600]

        # مؤشرات شائعة لمحاولة استخراج نموذج:
        # 1. حجم استعلامات غير طبيعي في فترة قصيرة
        if len(recent_queries) > 5000:
            return {"suspicious": True, "reason": "حجم استعلامات غير طبيعي"}

        # 2. أنماط منهجية (استعلامات مصمَّمة لتغطية مساحة واسعة من المدخلات بشكل ممنهج)
        unique_prompts_ratio = len(set(q["prompt"][:50] for q in recent_queries)) / max(len(recent_queries), 1)
        if len(recent_queries) > 500 and unique_prompts_ratio > 0.95:
            return {"suspicious": True, "reason": "نمط استكشاف ممنهج للمساحة المدخلة"}

        return {"suspicious": False}

    def add_output_watermark(self, generated_text: str, watermark_key: str) -> str:
        """
        توضيح مفاهيمي لفكرة العلامة المائية للنصوص المُولَّدة (Text Watermarking)
        (نهج مستوحى من أبحاث مثل Kirchenbauer et al.)
        تُستخدم لاحقًا لإثبات أن نصًا معيّنًا صدر عن هذا النموذج تحديدًا
        """
        # في التطبيق الحقيقي: تحيّز اختيار الرموز بشكل خفي وقابل للكشف إحصائيًا لاحقًا
        # باستخدام دالة hash تعتمد على watermark_key والسياق السابق لكل رمز
        pass   # تطبيق كامل يتطلب تعديل دالة sampling نفسها أثناء التوليد

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-11-llmops/ch58-security-governance.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-58-ch58-security-governance.ipynb#cell-2
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-58-ch58-security-governance.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import hashlib
import time
from collections import defaultdict

class RateLimiter:
    """
    تحديد معدل الطلبات لمنع إساءة الاستخدام واستنزاف الموارد
    خوارزمية Token Bucket — مرنة وفعّالة
    """
    def __init__(self, max_tokens=100, refill_rate=10):   # 10 طلبات/ثانية لكل مستخدم
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self.buckets = defaultdict(lambda: {"tokens": max_tokens, "last_refill": time.time()})

    def allow_request(self, user_id: str, cost: int = 1) -> bool:
        bucket = self.buckets[user_id]
        now = time.time()
        elapsed = now - bucket["last_refill"]
        bucket["tokens"] = min(self.max_tokens, bucket["tokens"] + elapsed * self.refill_rate)
        bucket["last_refill"] = now

        if bucket["tokens"] >= cost:
            bucket["tokens"] -= cost
            return True
        return False

class APIKeyAuthenticator:
    """مصادقة ومفاتيح API مع صلاحيات متدرجة (Tiered Access)"""
    def __init__(self):
        self.api_keys = {}   # {hashed_key: {"tier": "free/pro/enterprise", "quota": int}}

    def generate_api_key(self, user_id: str, tier: str) -> str:
        raw_key = f"{user_id}_{time.time()}_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}"
        hashed = hashlib.sha256(raw_key.encode()).hexdigest()
        quotas = {"free": 100, "pro": 10000, "enterprise": 1000000}
        self.api_keys[hashed] = {"user_id": user_id, "tier": tier, "quota": quotas[tier], "used": 0}
        return raw_key

    def validate_and_consume(self, raw_key: str, cost: int = 1) -> dict:
        hashed = hashlib.sha256(raw_key.encode()).hexdigest()
        if hashed not in self.api_keys:
            return {"valid": False, "reason": "مفتاح API غير صالح"}

        key_info = self.api_keys[hashed]
        if key_info["used"] + cost > key_info["quota"]:
            return {"valid": False, "reason": "تم تجاوز الحصة المسموحة"}

        key_info["used"] += cost
        return {"valid": True, "remaining_quota": key_info["quota"] - key_info["used"]}

def sanitize_input_prompt(prompt: str, max_length=10000) -> dict:
    """تنظيف وفحص أولي للمدخلات قبل الوصول للنموذج"""
    if len(prompt) > max_length:
        return {"safe": False, "reason": f"المدخل يتجاوز الحد الأقصى ({max_length} حرف)"}

    # كشف أنماط هجوم معروفة (قائمة مبسّطة، في الإنتاج تُستخدم نماذج تصنيف متخصصة)
    suspicious_patterns = ["ignore previous instructions", "system prompt", "tجاهل التعليمات"]
    if any(pattern.lower() in prompt.lower() for pattern in suspicious_patterns):
        return {"safe": False, "reason": "نمط حقن تعليمات محتمل", "flagged": True}

    return {"safe": True}

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-58-ch58-security-governance.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class DataGovernanceLayer:
    """
    طبقة حوكمة تضمن الامتثال لمتطلبات الخصوصية (GDPR وما شابه)
    """
    def __init__(self, pii_detector, encryption_service):
        self.pii_detector = pii_detector
        self.encryption_service = encryption_service
        self.user_data_registry = {}   # لتتبّع أي بيانات مُخزَّنة لأي مستخدم (للحق في الحذف)

    def process_before_storage(self, user_id: str, interaction_data: dict) -> dict:
        """معالجة البيانات قبل أي تخزين دائم"""
        # 1. كشف وتنبيه عند وجود PII غير متوقع
        pii_found = self.pii_detector(interaction_data["user_message"])
        if pii_found["has_pii"]:
            interaction_data["user_message"] = self._redact_pii(
                interaction_data["user_message"], pii_found["entities"]
            )
            interaction_data["pii_redacted"] = True

        # 2. تشفير البيانات الحساسة في حالة السكون (Encryption at Rest)
        encrypted_data = self.encryption_service.encrypt(json.dumps(interaction_data))

        # 3. تسجيل في سجل تتبّع بيانات المستخدم (لأغراض الحق في الحذف لاحقًا)
        self.user_data_registry.setdefault(user_id, []).append({
            "timestamp": time.time(), "data_ref": encrypted_data
        })
        return {"stored": True, "pii_redacted": interaction_data.get("pii_redacted", False)}

    def handle_right_to_deletion(self, user_id: str) -> dict:
        """تنفيذ "الحق في النسيان" — حذف كل بيانات مستخدم معيّن عند الطلب"""
        if user_id in self.user_data_registry:
            records_count = len(self.user_data_registry[user_id])
            del self.user_data_registry[user_id]
            # في التطبيق الفعلي: حذف فعلي من قاعدة البيانات، الذاكرة طويلة المدى (الفصل 51)، إلخ
            return {"deleted": True, "records_removed": records_count}
        return {"deleted": False, "reason": "لا توجد بيانات مُسجَّلة لهذا المستخدم"}

    def _redact_pii(self, text: str, entities: list) -> str:
        for entity in entities:
            text = text.replace(entity["text"], f"[{entity['type']}_محجوب]")
        return text

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-58-ch58-security-governance.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class ModelExtractionDefense:
    """
    دفاعات ضد محاولات "سرقة" النموذج عبر الاستعلام المكثّف
    (Model Extraction/Stealing Attacks) — مهاجم يحاول بناء نموذج مُقلِّد
    عبر إرسال عدد هائل من الاستعلامات وتدريب نموذج بديل على المخرجات
    """
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.query_patterns = defaultdict(list)

    def detect_extraction_attempt(self, user_id: str, prompt: str) -> dict:
        self.query_patterns[user_id].append({"prompt": prompt, "time": time.time()})

        recent_queries = [q for q in self.query_patterns[user_id]
                           if time.time() - q["time"] < 3600]

        # مؤشرات شائعة لمحاولة استخراج نموذج:
        # 1. حجم استعلامات غير طبيعي في فترة قصيرة
        if len(recent_queries) > 5000:
            return {"suspicious": True, "reason": "حجم استعلامات غير طبيعي"}

        # 2. أنماط منهجية (استعلامات مصمَّمة لتغطية مساحة واسعة من المدخلات بشكل ممنهج)
        unique_prompts_ratio = len(set(q["prompt"][:50] for q in recent_queries)) / max(len(recent_queries), 1)
        if len(recent_queries) > 500 and unique_prompts_ratio > 0.95:
            return {"suspicious": True, "reason": "نمط استكشاف ممنهج للمساحة المدخلة"}

        return {"suspicious": False}

    def add_output_watermark(self, generated_text: str, watermark_key: str) -> str:
        """
        توضيح مفاهيمي لفكرة العلامة المائية للنصوص المُولَّدة (Text Watermarking)
        (نهج مستوحى من أبحاث مثل Kirchenbauer et al.)
        تُستخدم لاحقًا لإثبات أن نصًا معيّنًا صدر عن هذا النموذج تحديدًا
        """
        # في التطبيق الحقيقي: تحيّز اختيار الرموز بشكل خفي وقابل للكشف إحصائيًا لاحقًا
        # باستخدام دالة hash تعتمد على watermark_key والسياق السابق لكل رمز
        pass   # تطبيق كامل يتطلب تعديل دالة sampling نفسها أثناء التوليد

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-58-ch58-security-governance.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
