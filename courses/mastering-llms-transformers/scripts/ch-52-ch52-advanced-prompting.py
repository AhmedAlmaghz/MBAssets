#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — هندسة الأوامر المتقدمة
Mastering Transformers and Large Language Models — Advanced Prompting

Book slug : mastering-llms-transformers
Chapter   : ch-52-ch52-advanced-prompting (lesson 52)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-52-ch52-advanced-prompting.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — هندسة الأوامر المتقدمة
Mastering Transformers and Large Language Models — Advanced Prompting

Book slug : mastering-llms-transformers
Chapter   : ch-52-ch52-advanced-prompting (lesson 52)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-52-ch52-advanced-prompting.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — هندسة الأوامر المتقدمة
Mastering Transformers and Large Language Models — Advanced Prompting

Book slug : mastering-llms-transformers
Chapter   : ch-52-ch52-advanced-prompting (lesson 52)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-10-applications-agents/ch52-advanced-prompting.ipynb#cell-1
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
    ("pydantic", "pydantic")
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
# [002] content/notebooks/part-10-applications-agents/ch52-advanced-prompting.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# 1) Zero-shot CoT — أبسط تقنية وأكثرها فعالية مقابل التكلفة
ZERO_SHOT_COT = "فكّر خطوة بخطوة."

# 2) Few-shot CoT — أمثلة توضح نمط الاستدلال المطلوب
FEW_SHOT_COT_EXAMPLE = """
سؤال: إذا كان لدى أحمد 5 تفاحات واشترى 3 أخرى ثم أكل تفاحتين، كم تفاحة لديه؟
الحل خطوة بخطوة:
1. البداية: 5 تفاحات
2. بعد الشراء: 5 + 3 = 8 تفاحات
3. بعد الأكل: 8 - 2 = 6 تفاحات
الإجابة: 6
"""

# 3) Self-Consistency — توليد عدة مسارات استدلال والتصويت بالأغلبية
def self_consistency_reasoning(llm, question: str, num_samples=5) -> str:
    """
    بدلاً من الاعتماد على مسار استدلال واحد (قد يحتوي خطأ عرضي)،
    نُولِّد عدة مسارات بـ temperature > 0 ونصوّت على الإجابة النهائية الأكثر تكرارًا
    """
    answers = []
    for _ in range(num_samples):
        response = llm.generate(f"{question}\n{ZERO_SHOT_COT}", temperature=0.7)
        final_answer = extract_final_answer(response)   # دالة استخراج مخصصة حسب التنسيق
        answers.append(final_answer)

    from collections import Counter
    most_common_answer, count = Counter(answers).most_common(1)[0]
    confidence = count / num_samples
    return {"answer": most_common_answer, "confidence": confidence, "all_answers": answers}

def extract_final_answer(response: str) -> str:
    if "الإجابة:" in response:
        return response.split("الإجابة:")[-1].strip().split("\n")[0]
    return response.strip()

# 4) Tree-of-Thoughts — استكشاف شجرة من مسارات الاستدلال مع تقييم وتراجع
class TreeOfThoughtsSolver:
    """
    بدلاً من مسار استدلال خطي واحد، نستكشف أشجارًا من "أفكار" جزئية
    مع تقييم كل فرع، والتراجع عن الأفرع الضعيفة (مشابه لـ BFS/DFS في البحث)
    """
    def __init__(self, llm, max_depth=3, branches_per_step=3):
        self.llm = llm
        self.max_depth = max_depth
        self.branches_per_step = branches_per_step

    def generate_next_thoughts(self, problem: str, current_path: list) -> list:
        prompt = f"""المسألة: {problem}
المسار الحالي من الأفكار: {current_path}

اقترح {self.branches_per_step} خطوات تفكير مختلفة ومحتملة للمتابعة من هنا."""
        response = self.llm.generate(prompt)
        return [line.strip() for line in response.split("\n") if line.strip()][:self.branches_per_step]

    def evaluate_path(self, problem: str, path: list) -> float:
        eval_prompt = f"""المسألة: {problem}
مسار التفكير: {path}

قيّم مدى واعدية هذا المسار للوصول لحل صحيح، من 0 إلى 1. أعطِ رقمًا فقط."""
        try:
            return float(self.llm.generate(eval_prompt).strip())
        except ValueError:
            return 0.5

    def solve(self, problem: str) -> dict:
        paths = [[]]   # نبدأ بمسار فارغ واحد

        for depth in range(self.max_depth):
            all_candidates = []
            for path in paths:
                next_thoughts = self.generate_next_thoughts(problem, path)
                for thought in next_thoughts:
                    new_path = path + [thought]
                    score = self.evaluate_path(problem, new_path)
                    all_candidates.append((new_path, score))

            # الاحتفاظ بأفضل المسارات فقط (Beam Search)
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            paths = [path for path, score in all_candidates[:self.branches_per_step]]

        best_path = paths[0]
        final_prompt = f"المسألة: {problem}\nمسار التفكير النهائي: {best_path}\nاستنتج الإجابة النهائية:"
        return {"final_answer": self.llm.generate(final_prompt), "best_path": best_path}

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-10-applications-agents/ch52-advanced-prompting.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from pydantic import BaseModel, Field
from typing import List

class ExtractedInvoice(BaseModel):
    """تعريف صارم للمخرج المطلوب — يُستخدم مع Function Calling/JSON mode"""
    invoice_number: str = Field(description="رقم الفاتورة")
    total_amount: float = Field(description="المبلغ الإجمالي")
    items: List[str] = Field(description="قائمة العناصر المذكورة في الفاتورة")
    date: str = Field(description="تاريخ الفاتورة بصيغة YYYY-MM-DD")

def structured_extraction_with_validation(llm, document_text: str, schema: BaseModel) -> dict:
    """
    استخدام JSON Schema + التحقق (validation) لضمان مخرجات موثوقة
    قابلة للمعالجة البرمجية المباشرة دون تحليل نص هش
    """
    schema_json = schema.model_json_schema()
    prompt = f"""استخرج المعلومات التالية من النص وفق المخطط المحدَّد بالضبط:
المخطط: {schema_json}

النص: {document_text}

أعطِ النتيجة بصيغة JSON صحيحة فقط، بدون أي نص إضافي."""

    raw_output = llm.generate(prompt, response_format="json")
    try:
        validated = schema.model_validate_json(raw_output)
        return {"success": True, "data": validated.model_dump()}
    except Exception as e:
        # إعادة محاولة بتصحيح الخطأ (Self-correction loop)
        correction_prompt = f"""المخرج السابق لم يطابق المخطط المطلوب. الخطأ: {str(e)}
المخرج الخاطئ: {raw_output}
صحّح المخرج ليطابق المخطط بالضبط:"""
        corrected_output = llm.generate(correction_prompt, response_format="json")
        validated = schema.model_validate_json(corrected_output)
        return {"success": True, "data": validated.model_dump(), "required_correction": True}

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-10-applications-agents/ch52-advanced-prompting.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
"""
توضيح مفاهيمي لفلسفة DSPy:
بدلاً من كتابة prompt ثابت، نُعرِّف "برنامجًا" من وحدات (modules) قابلة للتحسين،
ونسمح لإطار العمل بالبحث آليًا عن أفضل صياغة/أمثلة توضيحية بناءً على مقياس أداء محدَّد
"""

class OptimizablePromptModule:
    def __init__(self, llm, task_description: str, initial_instruction: str):
        self.llm = llm
        self.task_description = task_description
        self.instruction = initial_instruction
        self.few_shot_examples = []

    def predict(self, input_text: str) -> str:
        examples_text = "\n".join([
            f"مدخل: {ex['input']}\nمخرج: {ex['output']}" for ex in self.few_shot_examples
        ])
        prompt = f"{self.instruction}\n\n{examples_text}\n\nمدخل: {input_text}\nمخرج:"
        return self.llm.generate(prompt)

    def optimize(self, training_examples: list, metric_fn: callable, num_candidates=5):
        """
        بحث آلي مبسّط: نُجرِّب عدة صياغات تعليمات وأمثلة مختلفة،
        ونختار المجموعة التي تحقق أعلى درجة على مقياس الأداء المحدَّد
        """
        best_score, best_config = 0, None

        for _ in range(num_candidates):
            # توليد صياغة تعليمة بديلة واختيار عينة عشوائية من أمثلة توضيحية
            candidate_instruction = self.llm.generate(
                f"أعد صياغة هذه التعليمة بطريقة أوضح: {self.instruction}"
            )
            candidate_examples = random.sample(training_examples, min(3, len(training_examples)))

            self.instruction = candidate_instruction
            self.few_shot_examples = candidate_examples

            scores = [metric_fn(self.predict(ex["input"]), ex["output"]) for ex in training_examples[:20]]
            avg_score = sum(scores) / len(scores)

            if avg_score > best_score:
                best_score = avg_score
                best_config = {"instruction": candidate_instruction, "examples": candidate_examples}

        self.instruction = best_config["instruction"]
        self.few_shot_examples = best_config["examples"]
        return best_score

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-10-applications-agents/ch52-advanced-prompting.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-52-ch52-advanced-prompting.ipynb#cell-1
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
    ("pydantic", "pydantic")
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-52-ch52-advanced-prompting.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# 1) Zero-shot CoT — أبسط تقنية وأكثرها فعالية مقابل التكلفة
ZERO_SHOT_COT = "فكّر خطوة بخطوة."

# 2) Few-shot CoT — أمثلة توضح نمط الاستدلال المطلوب
FEW_SHOT_COT_EXAMPLE = """
سؤال: إذا كان لدى أحمد 5 تفاحات واشترى 3 أخرى ثم أكل تفاحتين، كم تفاحة لديه؟
الحل خطوة بخطوة:
1. البداية: 5 تفاحات
2. بعد الشراء: 5 + 3 = 8 تفاحات
3. بعد الأكل: 8 - 2 = 6 تفاحات
الإجابة: 6
"""

# 3) Self-Consistency — توليد عدة مسارات استدلال والتصويت بالأغلبية
def self_consistency_reasoning(llm, question: str, num_samples=5) -> str:
    """
    بدلاً من الاعتماد على مسار استدلال واحد (قد يحتوي خطأ عرضي)،
    نُولِّد عدة مسارات بـ temperature > 0 ونصوّت على الإجابة النهائية الأكثر تكرارًا
    """
    answers = []
    for _ in range(num_samples):
        response = llm.generate(f"{question}\n{ZERO_SHOT_COT}", temperature=0.7)
        final_answer = extract_final_answer(response)   # دالة استخراج مخصصة حسب التنسيق
        answers.append(final_answer)

    from collections import Counter
    most_common_answer, count = Counter(answers).most_common(1)[0]
    confidence = count / num_samples
    return {"answer": most_common_answer, "confidence": confidence, "all_answers": answers}

def extract_final_answer(response: str) -> str:
    if "الإجابة:" in response:
        return response.split("الإجابة:")[-1].strip().split("\n")[0]
    return response.strip()

# 4) Tree-of-Thoughts — استكشاف شجرة من مسارات الاستدلال مع تقييم وتراجع
class TreeOfThoughtsSolver:
    """
    بدلاً من مسار استدلال خطي واحد، نستكشف أشجارًا من "أفكار" جزئية
    مع تقييم كل فرع، والتراجع عن الأفرع الضعيفة (مشابه لـ BFS/DFS في البحث)
    """
    def __init__(self, llm, max_depth=3, branches_per_step=3):
        self.llm = llm
        self.max_depth = max_depth
        self.branches_per_step = branches_per_step

    def generate_next_thoughts(self, problem: str, current_path: list) -> list:
        prompt = f"""المسألة: {problem}
المسار الحالي من الأفكار: {current_path}

اقترح {self.branches_per_step} خطوات تفكير مختلفة ومحتملة للمتابعة من هنا."""
        response = self.llm.generate(prompt)
        return [line.strip() for line in response.split("\n") if line.strip()][:self.branches_per_step]

    def evaluate_path(self, problem: str, path: list) -> float:
        eval_prompt = f"""المسألة: {problem}
مسار التفكير: {path}

قيّم مدى واعدية هذا المسار للوصول لحل صحيح، من 0 إلى 1. أعطِ رقمًا فقط."""
        try:
            return float(self.llm.generate(eval_prompt).strip())
        except ValueError:
            return 0.5

    def solve(self, problem: str) -> dict:
        paths = [[]]   # نبدأ بمسار فارغ واحد

        for depth in range(self.max_depth):
            all_candidates = []
            for path in paths:
                next_thoughts = self.generate_next_thoughts(problem, path)
                for thought in next_thoughts:
                    new_path = path + [thought]
                    score = self.evaluate_path(problem, new_path)
                    all_candidates.append((new_path, score))

            # الاحتفاظ بأفضل المسارات فقط (Beam Search)
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            paths = [path for path, score in all_candidates[:self.branches_per_step]]

        best_path = paths[0]
        final_prompt = f"المسألة: {problem}\nمسار التفكير النهائي: {best_path}\nاستنتج الإجابة النهائية:"
        return {"final_answer": self.llm.generate(final_prompt), "best_path": best_path}

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-52-ch52-advanced-prompting.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from pydantic import BaseModel, Field
from typing import List

class ExtractedInvoice(BaseModel):
    """تعريف صارم للمخرج المطلوب — يُستخدم مع Function Calling/JSON mode"""
    invoice_number: str = Field(description="رقم الفاتورة")
    total_amount: float = Field(description="المبلغ الإجمالي")
    items: List[str] = Field(description="قائمة العناصر المذكورة في الفاتورة")
    date: str = Field(description="تاريخ الفاتورة بصيغة YYYY-MM-DD")

def structured_extraction_with_validation(llm, document_text: str, schema: BaseModel) -> dict:
    """
    استخدام JSON Schema + التحقق (validation) لضمان مخرجات موثوقة
    قابلة للمعالجة البرمجية المباشرة دون تحليل نص هش
    """
    schema_json = schema.model_json_schema()
    prompt = f"""استخرج المعلومات التالية من النص وفق المخطط المحدَّد بالضبط:
المخطط: {schema_json}

النص: {document_text}

أعطِ النتيجة بصيغة JSON صحيحة فقط، بدون أي نص إضافي."""

    raw_output = llm.generate(prompt, response_format="json")
    try:
        validated = schema.model_validate_json(raw_output)
        return {"success": True, "data": validated.model_dump()}
    except Exception as e:
        # إعادة محاولة بتصحيح الخطأ (Self-correction loop)
        correction_prompt = f"""المخرج السابق لم يطابق المخطط المطلوب. الخطأ: {str(e)}
المخرج الخاطئ: {raw_output}
صحّح المخرج ليطابق المخطط بالضبط:"""
        corrected_output = llm.generate(correction_prompt, response_format="json")
        validated = schema.model_validate_json(corrected_output)
        return {"success": True, "data": validated.model_dump(), "required_correction": True}

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-52-ch52-advanced-prompting.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
"""
توضيح مفاهيمي لفلسفة DSPy:
بدلاً من كتابة prompt ثابت، نُعرِّف "برنامجًا" من وحدات (modules) قابلة للتحسين،
ونسمح لإطار العمل بالبحث آليًا عن أفضل صياغة/أمثلة توضيحية بناءً على مقياس أداء محدَّد
"""

class OptimizablePromptModule:
    def __init__(self, llm, task_description: str, initial_instruction: str):
        self.llm = llm
        self.task_description = task_description
        self.instruction = initial_instruction
        self.few_shot_examples = []

    def predict(self, input_text: str) -> str:
        examples_text = "\n".join([
            f"مدخل: {ex['input']}\nمخرج: {ex['output']}" for ex in self.few_shot_examples
        ])
        prompt = f"{self.instruction}\n\n{examples_text}\n\nمدخل: {input_text}\nمخرج:"
        return self.llm.generate(prompt)

    def optimize(self, training_examples: list, metric_fn: callable, num_candidates=5):
        """
        بحث آلي مبسّط: نُجرِّب عدة صياغات تعليمات وأمثلة مختلفة،
        ونختار المجموعة التي تحقق أعلى درجة على مقياس الأداء المحدَّد
        """
        best_score, best_config = 0, None

        for _ in range(num_candidates):
            # توليد صياغة تعليمة بديلة واختيار عينة عشوائية من أمثلة توضيحية
            candidate_instruction = self.llm.generate(
                f"أعد صياغة هذه التعليمة بطريقة أوضح: {self.instruction}"
            )
            candidate_examples = random.sample(training_examples, min(3, len(training_examples)))

            self.instruction = candidate_instruction
            self.few_shot_examples = candidate_examples

            scores = [metric_fn(self.predict(ex["input"]), ex["output"]) for ex in training_examples[:20]]
            avg_score = sum(scores) / len(scores)

            if avg_score > best_score:
                best_score = avg_score
                best_config = {"instruction": candidate_instruction, "examples": candidate_examples}

        self.instruction = best_config["instruction"]
        self.few_shot_examples = best_config["examples"]
        return best_score

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-52-ch52-advanced-prompting.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-52-ch52-advanced-prompting.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — هندسة الأوامر المتقدمة
Mastering Transformers and Large Language Models — Advanced Prompting

Book slug : mastering-llms-transformers
Chapter   : ch-52-ch52-advanced-prompting (lesson 52)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-10-applications-agents/ch52-advanced-prompting.ipynb#cell-1
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
    ("pydantic", "pydantic")
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
# [002] content/notebooks/part-10-applications-agents/ch52-advanced-prompting.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# 1) Zero-shot CoT — أبسط تقنية وأكثرها فعالية مقابل التكلفة
ZERO_SHOT_COT = "فكّر خطوة بخطوة."

# 2) Few-shot CoT — أمثلة توضح نمط الاستدلال المطلوب
FEW_SHOT_COT_EXAMPLE = """
سؤال: إذا كان لدى أحمد 5 تفاحات واشترى 3 أخرى ثم أكل تفاحتين، كم تفاحة لديه؟
الحل خطوة بخطوة:
1. البداية: 5 تفاحات
2. بعد الشراء: 5 + 3 = 8 تفاحات
3. بعد الأكل: 8 - 2 = 6 تفاحات
الإجابة: 6
"""

# 3) Self-Consistency — توليد عدة مسارات استدلال والتصويت بالأغلبية
def self_consistency_reasoning(llm, question: str, num_samples=5) -> str:
    """
    بدلاً من الاعتماد على مسار استدلال واحد (قد يحتوي خطأ عرضي)،
    نُولِّد عدة مسارات بـ temperature > 0 ونصوّت على الإجابة النهائية الأكثر تكرارًا
    """
    answers = []
    for _ in range(num_samples):
        response = llm.generate(f"{question}\n{ZERO_SHOT_COT}", temperature=0.7)
        final_answer = extract_final_answer(response)   # دالة استخراج مخصصة حسب التنسيق
        answers.append(final_answer)

    from collections import Counter
    most_common_answer, count = Counter(answers).most_common(1)[0]
    confidence = count / num_samples
    return {"answer": most_common_answer, "confidence": confidence, "all_answers": answers}

def extract_final_answer(response: str) -> str:
    if "الإجابة:" in response:
        return response.split("الإجابة:")[-1].strip().split("\n")[0]
    return response.strip()

# 4) Tree-of-Thoughts — استكشاف شجرة من مسارات الاستدلال مع تقييم وتراجع
class TreeOfThoughtsSolver:
    """
    بدلاً من مسار استدلال خطي واحد، نستكشف أشجارًا من "أفكار" جزئية
    مع تقييم كل فرع، والتراجع عن الأفرع الضعيفة (مشابه لـ BFS/DFS في البحث)
    """
    def __init__(self, llm, max_depth=3, branches_per_step=3):
        self.llm = llm
        self.max_depth = max_depth
        self.branches_per_step = branches_per_step

    def generate_next_thoughts(self, problem: str, current_path: list) -> list:
        prompt = f"""المسألة: {problem}
المسار الحالي من الأفكار: {current_path}

اقترح {self.branches_per_step} خطوات تفكير مختلفة ومحتملة للمتابعة من هنا."""
        response = self.llm.generate(prompt)
        return [line.strip() for line in response.split("\n") if line.strip()][:self.branches_per_step]

    def evaluate_path(self, problem: str, path: list) -> float:
        eval_prompt = f"""المسألة: {problem}
مسار التفكير: {path}

قيّم مدى واعدية هذا المسار للوصول لحل صحيح، من 0 إلى 1. أعطِ رقمًا فقط."""
        try:
            return float(self.llm.generate(eval_prompt).strip())
        except ValueError:
            return 0.5

    def solve(self, problem: str) -> dict:
        paths = [[]]   # نبدأ بمسار فارغ واحد

        for depth in range(self.max_depth):
            all_candidates = []
            for path in paths:
                next_thoughts = self.generate_next_thoughts(problem, path)
                for thought in next_thoughts:
                    new_path = path + [thought]
                    score = self.evaluate_path(problem, new_path)
                    all_candidates.append((new_path, score))

            # الاحتفاظ بأفضل المسارات فقط (Beam Search)
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            paths = [path for path, score in all_candidates[:self.branches_per_step]]

        best_path = paths[0]
        final_prompt = f"المسألة: {problem}\nمسار التفكير النهائي: {best_path}\nاستنتج الإجابة النهائية:"
        return {"final_answer": self.llm.generate(final_prompt), "best_path": best_path}

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-10-applications-agents/ch52-advanced-prompting.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from pydantic import BaseModel, Field
from typing import List

class ExtractedInvoice(BaseModel):
    """تعريف صارم للمخرج المطلوب — يُستخدم مع Function Calling/JSON mode"""
    invoice_number: str = Field(description="رقم الفاتورة")
    total_amount: float = Field(description="المبلغ الإجمالي")
    items: List[str] = Field(description="قائمة العناصر المذكورة في الفاتورة")
    date: str = Field(description="تاريخ الفاتورة بصيغة YYYY-MM-DD")

def structured_extraction_with_validation(llm, document_text: str, schema: BaseModel) -> dict:
    """
    استخدام JSON Schema + التحقق (validation) لضمان مخرجات موثوقة
    قابلة للمعالجة البرمجية المباشرة دون تحليل نص هش
    """
    schema_json = schema.model_json_schema()
    prompt = f"""استخرج المعلومات التالية من النص وفق المخطط المحدَّد بالضبط:
المخطط: {schema_json}

النص: {document_text}

أعطِ النتيجة بصيغة JSON صحيحة فقط، بدون أي نص إضافي."""

    raw_output = llm.generate(prompt, response_format="json")
    try:
        validated = schema.model_validate_json(raw_output)
        return {"success": True, "data": validated.model_dump()}
    except Exception as e:
        # إعادة محاولة بتصحيح الخطأ (Self-correction loop)
        correction_prompt = f"""المخرج السابق لم يطابق المخطط المطلوب. الخطأ: {str(e)}
المخرج الخاطئ: {raw_output}
صحّح المخرج ليطابق المخطط بالضبط:"""
        corrected_output = llm.generate(correction_prompt, response_format="json")
        validated = schema.model_validate_json(corrected_output)
        return {"success": True, "data": validated.model_dump(), "required_correction": True}

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-10-applications-agents/ch52-advanced-prompting.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
"""
توضيح مفاهيمي لفلسفة DSPy:
بدلاً من كتابة prompt ثابت، نُعرِّف "برنامجًا" من وحدات (modules) قابلة للتحسين،
ونسمح لإطار العمل بالبحث آليًا عن أفضل صياغة/أمثلة توضيحية بناءً على مقياس أداء محدَّد
"""

class OptimizablePromptModule:
    def __init__(self, llm, task_description: str, initial_instruction: str):
        self.llm = llm
        self.task_description = task_description
        self.instruction = initial_instruction
        self.few_shot_examples = []

    def predict(self, input_text: str) -> str:
        examples_text = "\n".join([
            f"مدخل: {ex['input']}\nمخرج: {ex['output']}" for ex in self.few_shot_examples
        ])
        prompt = f"{self.instruction}\n\n{examples_text}\n\nمدخل: {input_text}\nمخرج:"
        return self.llm.generate(prompt)

    def optimize(self, training_examples: list, metric_fn: callable, num_candidates=5):
        """
        بحث آلي مبسّط: نُجرِّب عدة صياغات تعليمات وأمثلة مختلفة،
        ونختار المجموعة التي تحقق أعلى درجة على مقياس الأداء المحدَّد
        """
        best_score, best_config = 0, None

        for _ in range(num_candidates):
            # توليد صياغة تعليمة بديلة واختيار عينة عشوائية من أمثلة توضيحية
            candidate_instruction = self.llm.generate(
                f"أعد صياغة هذه التعليمة بطريقة أوضح: {self.instruction}"
            )
            candidate_examples = random.sample(training_examples, min(3, len(training_examples)))

            self.instruction = candidate_instruction
            self.few_shot_examples = candidate_examples

            scores = [metric_fn(self.predict(ex["input"]), ex["output"]) for ex in training_examples[:20]]
            avg_score = sum(scores) / len(scores)

            if avg_score > best_score:
                best_score = avg_score
                best_config = {"instruction": candidate_instruction, "examples": candidate_examples}

        self.instruction = best_config["instruction"]
        self.few_shot_examples = best_config["examples"]
        return best_score

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-10-applications-agents/ch52-advanced-prompting.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-52-ch52-advanced-prompting.ipynb#cell-2
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
    ("pydantic", "pydantic")
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-52-ch52-advanced-prompting.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# 1) Zero-shot CoT — أبسط تقنية وأكثرها فعالية مقابل التكلفة
ZERO_SHOT_COT = "فكّر خطوة بخطوة."

# 2) Few-shot CoT — أمثلة توضح نمط الاستدلال المطلوب
FEW_SHOT_COT_EXAMPLE = """
سؤال: إذا كان لدى أحمد 5 تفاحات واشترى 3 أخرى ثم أكل تفاحتين، كم تفاحة لديه؟
الحل خطوة بخطوة:
1. البداية: 5 تفاحات
2. بعد الشراء: 5 + 3 = 8 تفاحات
3. بعد الأكل: 8 - 2 = 6 تفاحات
الإجابة: 6
"""

# 3) Self-Consistency — توليد عدة مسارات استدلال والتصويت بالأغلبية
def self_consistency_reasoning(llm, question: str, num_samples=5) -> str:
    """
    بدلاً من الاعتماد على مسار استدلال واحد (قد يحتوي خطأ عرضي)،
    نُولِّد عدة مسارات بـ temperature > 0 ونصوّت على الإجابة النهائية الأكثر تكرارًا
    """
    answers = []
    for _ in range(num_samples):
        response = llm.generate(f"{question}\n{ZERO_SHOT_COT}", temperature=0.7)
        final_answer = extract_final_answer(response)   # دالة استخراج مخصصة حسب التنسيق
        answers.append(final_answer)

    from collections import Counter
    most_common_answer, count = Counter(answers).most_common(1)[0]
    confidence = count / num_samples
    return {"answer": most_common_answer, "confidence": confidence, "all_answers": answers}

def extract_final_answer(response: str) -> str:
    if "الإجابة:" in response:
        return response.split("الإجابة:")[-1].strip().split("\n")[0]
    return response.strip()

# 4) Tree-of-Thoughts — استكشاف شجرة من مسارات الاستدلال مع تقييم وتراجع
class TreeOfThoughtsSolver:
    """
    بدلاً من مسار استدلال خطي واحد، نستكشف أشجارًا من "أفكار" جزئية
    مع تقييم كل فرع، والتراجع عن الأفرع الضعيفة (مشابه لـ BFS/DFS في البحث)
    """
    def __init__(self, llm, max_depth=3, branches_per_step=3):
        self.llm = llm
        self.max_depth = max_depth
        self.branches_per_step = branches_per_step

    def generate_next_thoughts(self, problem: str, current_path: list) -> list:
        prompt = f"""المسألة: {problem}
المسار الحالي من الأفكار: {current_path}

اقترح {self.branches_per_step} خطوات تفكير مختلفة ومحتملة للمتابعة من هنا."""
        response = self.llm.generate(prompt)
        return [line.strip() for line in response.split("\n") if line.strip()][:self.branches_per_step]

    def evaluate_path(self, problem: str, path: list) -> float:
        eval_prompt = f"""المسألة: {problem}
مسار التفكير: {path}

قيّم مدى واعدية هذا المسار للوصول لحل صحيح، من 0 إلى 1. أعطِ رقمًا فقط."""
        try:
            return float(self.llm.generate(eval_prompt).strip())
        except ValueError:
            return 0.5

    def solve(self, problem: str) -> dict:
        paths = [[]]   # نبدأ بمسار فارغ واحد

        for depth in range(self.max_depth):
            all_candidates = []
            for path in paths:
                next_thoughts = self.generate_next_thoughts(problem, path)
                for thought in next_thoughts:
                    new_path = path + [thought]
                    score = self.evaluate_path(problem, new_path)
                    all_candidates.append((new_path, score))

            # الاحتفاظ بأفضل المسارات فقط (Beam Search)
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            paths = [path for path, score in all_candidates[:self.branches_per_step]]

        best_path = paths[0]
        final_prompt = f"المسألة: {problem}\nمسار التفكير النهائي: {best_path}\nاستنتج الإجابة النهائية:"
        return {"final_answer": self.llm.generate(final_prompt), "best_path": best_path}

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-52-ch52-advanced-prompting.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from pydantic import BaseModel, Field
from typing import List

class ExtractedInvoice(BaseModel):
    """تعريف صارم للمخرج المطلوب — يُستخدم مع Function Calling/JSON mode"""
    invoice_number: str = Field(description="رقم الفاتورة")
    total_amount: float = Field(description="المبلغ الإجمالي")
    items: List[str] = Field(description="قائمة العناصر المذكورة في الفاتورة")
    date: str = Field(description="تاريخ الفاتورة بصيغة YYYY-MM-DD")

def structured_extraction_with_validation(llm, document_text: str, schema: BaseModel) -> dict:
    """
    استخدام JSON Schema + التحقق (validation) لضمان مخرجات موثوقة
    قابلة للمعالجة البرمجية المباشرة دون تحليل نص هش
    """
    schema_json = schema.model_json_schema()
    prompt = f"""استخرج المعلومات التالية من النص وفق المخطط المحدَّد بالضبط:
المخطط: {schema_json}

النص: {document_text}

أعطِ النتيجة بصيغة JSON صحيحة فقط، بدون أي نص إضافي."""

    raw_output = llm.generate(prompt, response_format="json")
    try:
        validated = schema.model_validate_json(raw_output)
        return {"success": True, "data": validated.model_dump()}
    except Exception as e:
        # إعادة محاولة بتصحيح الخطأ (Self-correction loop)
        correction_prompt = f"""المخرج السابق لم يطابق المخطط المطلوب. الخطأ: {str(e)}
المخرج الخاطئ: {raw_output}
صحّح المخرج ليطابق المخطط بالضبط:"""
        corrected_output = llm.generate(correction_prompt, response_format="json")
        validated = schema.model_validate_json(corrected_output)
        return {"success": True, "data": validated.model_dump(), "required_correction": True}

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-52-ch52-advanced-prompting.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
"""
توضيح مفاهيمي لفلسفة DSPy:
بدلاً من كتابة prompt ثابت، نُعرِّف "برنامجًا" من وحدات (modules) قابلة للتحسين،
ونسمح لإطار العمل بالبحث آليًا عن أفضل صياغة/أمثلة توضيحية بناءً على مقياس أداء محدَّد
"""

class OptimizablePromptModule:
    def __init__(self, llm, task_description: str, initial_instruction: str):
        self.llm = llm
        self.task_description = task_description
        self.instruction = initial_instruction
        self.few_shot_examples = []

    def predict(self, input_text: str) -> str:
        examples_text = "\n".join([
            f"مدخل: {ex['input']}\nمخرج: {ex['output']}" for ex in self.few_shot_examples
        ])
        prompt = f"{self.instruction}\n\n{examples_text}\n\nمدخل: {input_text}\nمخرج:"
        return self.llm.generate(prompt)

    def optimize(self, training_examples: list, metric_fn: callable, num_candidates=5):
        """
        بحث آلي مبسّط: نُجرِّب عدة صياغات تعليمات وأمثلة مختلفة،
        ونختار المجموعة التي تحقق أعلى درجة على مقياس الأداء المحدَّد
        """
        best_score, best_config = 0, None

        for _ in range(num_candidates):
            # توليد صياغة تعليمة بديلة واختيار عينة عشوائية من أمثلة توضيحية
            candidate_instruction = self.llm.generate(
                f"أعد صياغة هذه التعليمة بطريقة أوضح: {self.instruction}"
            )
            candidate_examples = random.sample(training_examples, min(3, len(training_examples)))

            self.instruction = candidate_instruction
            self.few_shot_examples = candidate_examples

            scores = [metric_fn(self.predict(ex["input"]), ex["output"]) for ex in training_examples[:20]]
            avg_score = sum(scores) / len(scores)

            if avg_score > best_score:
                best_score = avg_score
                best_config = {"instruction": candidate_instruction, "examples": candidate_examples}

        self.instruction = best_config["instruction"]
        self.few_shot_examples = best_config["examples"]
        return best_score

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-52-ch52-advanced-prompting.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
