#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — خاتمة — منهجية متابعة الأبحاث ومواكبة التسارع التقني
Large Language Model Engineering — Chapter 69 — Frontier Trends

Book slug : llm-engineering-69-chapters
Chapter   : ch-69-chapter-69 (lesson 69)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-69-chapter-69.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — خاتمة — منهجية متابعة الأبحاث ومواكبة التسارع التقني
Large Language Model Engineering — Chapter 69 — Frontier Trends

Book slug : llm-engineering-69-chapters
Chapter   : ch-69-chapter-69 (lesson 69)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-69-chapter-69.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — خاتمة — منهجية متابعة الأبحاث ومواكبة التسارع التقني
Large Language Model Engineering — Chapter 69 — Frontier Trends

Book slug : llm-engineering-69-chapters
Chapter   : ch-69-chapter-69 (lesson 69)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_69.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 69: الفصل 69: خاتمة — منهجية متابعة الأبحاث ومواكبة التسارع التقني."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_research() -> None:
    papers = [{'title': 'Efficient Attention', 'year': 2024, 'relevance': 0.92}, {'title': 'Reliable Evaluation', 'year': 2025, 'relevance': 0.87}, {'title': 'Small Language Models', 'year': 2023, 'relevance': 0.89}]
    ranked = sorted(papers, key=lambda paper: (paper['relevance'], paper['year']), reverse=True)
    reading_log = [{'title': paper['title'], 'status': 'to-read'} for paper in ranked]
    print('ranked reading list:', reading_log)
    print('record exact citations and verify sources before citing.')

if __name__ == "__main__":
    print("Chapter 69 lab: Rank and track a reproducible research reading list")
    run_research()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-12-frontier-trends/chapter-69.md
# ────────────────────────────────────────────────────────────────────────
"""
claim_evaluation_checklist.py
أداة مساعدة منهجية (وليست أداة آلية تحليل نصوص) لتقييم ادعاءات
بحثية/تسويقية جديدة بسرعة، قبل الاستثمار في فهمها بعمق أو تبنّيها.
"""
from dataclasses import dataclass, field
from enum import Enum


class EvidenceLevel(Enum):
    ANECDOTAL = 1          # مثال واحد أو قصة فردية فقط
    BENCHMARK_CLAIMED = 2  # أرقام على معيار قياسي، لكن غير مُستنسخة (replicated)
    BENCHMARK_REPLICATED = 3  # نتائج أكدتها فرق مستقلة متعددة
    PRODUCTION_PROVEN = 4  # مُستخدمة فعليًا في أنظمة إنتاجية واسعة النطاق


@dataclass
class ClaimAssessment:
    claim_description: str
    source_type: str                 # "ورقة محكّمة" / "preprint" / "منشور مدونة شركة" / "تغريدة"
    evidence_level: EvidenceLevel
    has_open_code: bool
    has_been_replicated_independently: bool
    conflicts_of_interest: bool      # هل المصدر لديه مصلحة تجارية مباشرة في الادعاء؟
    comparison_is_fair: bool | None  # هل قورن بأحدث خط أساس (baseline) بإعدادات عادلة؟

    def compute_trust_score(self) -> float:
        """
        نقاط ثقة تقريبية من 0 إلى 10، كدليل استرشادي سريع
        وليس قياسًا علميًا دقيقًا.
        """
        score = self.evidence_level.value * 1.5  # حتى 6 نقاط من مستوى الدليل

        if self.has_open_code:
            score += 1.5
        if self.has_been_replicated_independently:
            score += 2.0
        if self.conflicts_of_interest:
            score -= 1.5
        if self.comparison_is_fair is False:
            score -= 2.0
        elif self.comparison_is_fair is None:
            score -= 0.5  # معلومة غير واضحة = حذر إضافي

        return max(0.0, min(10.0, score))

    def recommendation(self) -> str:
        score = self.compute_trust_score()
        if score >= 7.5:
            return "✅ جدير بالاستثمار في فهم عميق وربما تجربة فعلية"
        elif score >= 5.0:
            return "⚠️ واعد لكن انتظر تأكيدًا إضافيًا قبل الاعتماد عليه في الإنتاج"
        else:
            return "🔍 ضجيج محتمل — راقب من بعيد، لا تستثمر وقتًا كبيرًا الآن"


# ---------- مثال تطبيقي: تقييم ادعاء افتراضي ----------
claim_1 = ClaimAssessment(
    claim_description="نموذج جديد X يتفوق على GPT-4 بنصف التكلفة",
    source_type="منشور مدونة الشركة المُطوِّرة",
    evidence_level=EvidenceLevel.BENCHMARK_CLAIMED,
    has_open_code=False,
    has_been_replicated_independently=False,
    conflicts_of_interest=True,
    comparison_is_fair=None,
)

claim_2 = ClaimAssessment(
    claim_description="DeepSeek-R1 يحقق أداء استدلال منافس بتكلفة تدريب أقل بكثير",
    source_type="ورقة بحثية + كود وأوزان مفتوحة بالكامل",
    evidence_level=EvidenceLevel.BENCHMARK_REPLICATED,
    has_open_code=True,
    has_been_replicated_independently=True,
    conflicts_of_interest=False,
    comparison_is_fair=True,
)

for claim in [claim_1, claim_2]:
    print(f"\n📌 الادعاء: {claim.claim_description}")
    print(f"   نقاط الثقة: {claim.compute_trust_score():.1f}/10")
    print(f"   التوصية: {claim.recommendation()}")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-12-frontier-trends/chapter-69.md
# ────────────────────────────────────────────────────────────────────────
"""
learning_routine_tracker.py
أداة بسيطة لتتبع التزامك بروتين التعلّم المستدام عبر الزمن،
وتوليد تذكيرات وتقارير تقدّم.
"""
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum


class RoutineFrequency(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


@dataclass
class LearningActivity:
    activity_date: date
    frequency_type: RoutineFrequency
    description: str
    time_spent_minutes: int


@dataclass
class LearningTracker:
    activities: list[LearningActivity] = field(default_factory=list)

    def log_activity(
        self, frequency_type: RoutineFrequency, description: str, minutes: int
    ) -> None:
        self.activities.append(
            LearningActivity(date.today(), frequency_type, description, minutes)
        )

    def get_weekly_summary(self) -> dict[str, int]:
        one_week_ago = date.today() - timedelta(days=7)
        recent = [a for a in self.activities if a.activity_date >= one_week_ago]

        summary = {freq.value: 0 for freq in RoutineFrequency}
        for activity in recent:
            summary[activity.frequency_type.value] += activity.time_spent_minutes

        return summary

    def check_consistency_alert(self) -> list[str]:
        """تنبيهات بسيطة إذا انقطع الالتزام بأحد الأنماط الأساسية"""
        alerts = []
        one_week_ago = date.today() - timedelta(days=7)
        two_weeks_ago = date.today() - timedelta(days=14)

        weekly_deep_reads = [
            a for a in self.activities
            if a.frequency_type == RoutineFrequency.WEEKLY and a.activity_date >= one_week_ago
        ]
        if not weekly_deep_reads:
            alerts.append("⚠️ لم تقرأ أي ورقة بحثية بعمق خلال الأسبوع الماضي.")

        monthly_hands_on = [
            a for a in self.activities
            if a.frequency_type == RoutineFrequency.MONTHLY and a.activity_date >= two_weeks_ago
        ]
        if not monthly_hands_on and date.today().day > 20:
            alerts.append("⚠️ اقترب نهاية الشهر ولم تُجرِ تجربة عملية بعد.")

        return alerts


# مثال استخدام
tracker = LearningTracker()
tracker.log_activity(RoutineFrequency.DAILY, "فحص عناوين arXiv وHacker News", 15)
tracker.log_activity(
    RoutineFrequency.WEEKLY,
    "قراءة عميقة لورقة RULER حول تقييم السياق الطويل",
    120,
)
tracker.log_activity(
    RoutineFrequency.MONTHLY,
    "تجربة ضبط دقيق لـ Phi-3-mini على مهمة تصنيف شخصية",
    240,
)

print("📊 ملخص الأسبوع:")
for freq, minutes in tracker.get_weekly_summary().items():
    print(f"   {freq}: {minutes} دقيقة")

alerts = tracker.check_consistency_alert()
if alerts:
    print("\n🔔 تنبيهات:")
    for alert in alerts:
        print(f"   {alert}")
else:
    print("\n✅ الالتزام بالروتين جيد هذا الأسبوع.")

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-12-frontier-trends/chapter-69.md
# ────────────────────────────────────────────────────────────────────────
"""
personal_research_radar.py
نظام متكامل End-to-End: "رادار بحثي شخصي" يجلب أحدث الأوراق من arXiv
عبر الـ API الرسمي المجاني، يفلترها حسب اهتماماتك المحددة،
يطبّق تقييمًا أوليًا، وينتج تقريرًا أسبوعيًا جاهزًا للقراءة.

يستخدم مكتبة urllib القياسية (بدون الحاجة لمكتبات خارجية إضافية)
للاتصال بـ arXiv API الحقيقي والعامل فعليًا.
"""
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class ArxivPaper:
    title: str
    summary: str
    authors: list[str]
    published_date: datetime
    arxiv_id: str
    categories: list[str]
    pdf_url: str

    def relevance_score(self, interest_keywords: list[str]) -> float:
        """
        تقييم بسيط لمدى صلة الورقة باهتماماتك، بناءً على
        تطابق الكلمات المفتاحية في العنوان والملخص.
        (في نظام أكثر تقدمًا، يمكن استبدال هذا بتشابه embeddings دلالي).
        """
        combined_text = (self.title + " " + self.summary).lower()
        matches = sum(1 for kw in interest_keywords if kw.lower() in combined_text)
        return matches / max(len(interest_keywords), 1)


class ArxivResearchRadar:
    """واجهة بسيطة وفعلية للتفاعل مع arXiv API العام والمجاني"""

    BASE_URL = "http://export.arxiv.org/api/query"
    ATOM_NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}

    def fetch_recent_papers(
        self, search_query: str, max_results: int = 20
    ) -> list[ArxivPaper]:
        """
        جلب أحدث الأوراق البحثية من arXiv حسب استعلام بحث محدد.
        مثال على search_query: "cat:cs.CL AND abs:reasoning"
        """
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"

        with urllib.request.urlopen(url, timeout=15) as response:
            xml_data = response.read()

        return self._parse_arxiv_response(xml_data)

    def _parse_arxiv_response(self, xml_data: bytes) -> list[ArxivPaper]:
        root = ET.fromstring(xml_data)
        papers = []

        for entry in root.findall("atom:entry", self.ATOM_NAMESPACE):
            title = entry.find("atom:title", self.ATOM_NAMESPACE).text.strip()
            summary = entry.find("atom:summary", self.ATOM_NAMESPACE).text.strip()
            arxiv_id = entry.find("atom:id", self.ATOM_NAMESPACE).text.strip()

            published_str = entry.find("atom:published", self.ATOM_NAMESPACE).text
            published_date = datetime.strptime(published_str, "%Y-%m-%dT%H:%M:%SZ")

            authors = [
                author.find("atom:name", self.ATOM_NAMESPACE).text
                for author in entry.findall("atom:author", self.ATOM_NAMESPACE)
            ]

            categories = [
                cat.attrib.get("term", "")
                for cat in entry.findall(
                    "{http://arxiv.org/schemas/atom}category"
                )
            ] or [
                cat.attrib.get("term", "")
                for cat in entry.findall("atom:category", self.ATOM_NAMESPACE)
            ]

            pdf_url = arxiv_id.replace("abs", "pdf")

            papers.append(ArxivPaper(
                title=title, summary=summary, authors=authors,
                published_date=published_date, arxiv_id=arxiv_id,
                categories=categories, pdf_url=pdf_url,
            ))

        return papers


@dataclass
class WeeklyResearchReport:
    generated_at: datetime
    top_papers: list[tuple[ArxivPaper, float]] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"📬 تقرير الرادار البحثي الأسبوعي — {self.generated_at.strftime('%Y-%m-%d')}",
            "=" * 70,
        ]
        for rank, (paper, score) in enumerate(self.top_papers, start=1):
            lines.append(f"\n{rank}. {paper.title}")
            lines.append(f"   📅 تاريخ النشر: {paper.published_date.strftime('%Y-%m-%d')}")
            lines.append(f"   👥 المؤلفون: {', '.join(paper.authors[:3])}"
                          f"{' وآخرون' if len(paper.authors) > 3 else ''}")
            lines.append(f"   🎯 درجة الصلة باهتماماتك: {score:.0%}")
            lines.append(f"   📄 الملخص: {paper.summary[:200]}...")
            lines.append(f"   🔗 الرابط: {paper.pdf_url}")
        return "\n".join(lines)


def generate_personalized_weekly_report(
    interest_keywords: list[str],
    arxiv_category: str = "cs.CL",
    top_n: int = 5,
) -> WeeklyResearchReport:
    """
    الدالة الرئيسية: تجلب أوراقًا حديثة، تُقيّمها حسب اهتماماتك،
    وتُرجع تقريرًا جاهزًا بأهم top_n ورقة صلة.
    """
    radar = ArxivResearchRadar()

    search_query = f"cat:{arxiv_category}"
    papers = radar.fetch_recent_papers(search_query, max_results=50)

    scored_papers = [
        (paper, paper.relevance_score(interest_keywords)) for paper in papers
    ]
    scored_papers.sort(key=lambda x: x[1], reverse=True)

    report = WeeklyResearchReport(
        generated_at=datetime.now(),
        top_papers=scored_papers[:top_n],
    )
    return report


# ---------- تشغيل فعلي للنظام ----------
if __name__ == "__main__":
    my_interests = [
        "reasoning", "long context", "efficient inference",
        "agent", "small language model", "test-time compute",
    ]

    report = generate_personalized_weekly_report(
        interest_keywords=my_interests,
        arxiv_category="cs.CL",
        top_n=5,
    )

    print(report.render())

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-69-chapter-69.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 69: الفصل 69: خاتمة — منهجية متابعة الأبحاث ومواكبة التسارع التقني."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_research() -> None:
    papers = [{'title': 'Efficient Attention', 'year': 2024, 'relevance': 0.92}, {'title': 'Reliable Evaluation', 'year': 2025, 'relevance': 0.87}, {'title': 'Small Language Models', 'year': 2023, 'relevance': 0.89}]
    ranked = sorted(papers, key=lambda paper: (paper['relevance'], paper['year']), reverse=True)
    reading_log = [{'title': paper['title'], 'status': 'to-read'} for paper in ranked]
    print('ranked reading list:', reading_log)
    print('record exact citations and verify sources before citing.')

if __name__ == "__main__":
    print("Chapter 69 lab: Rank and track a reproducible research reading list")
    run_research()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-69-chapter-69.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
claim_evaluation_checklist.py
أداة مساعدة منهجية (وليست أداة آلية تحليل نصوص) لتقييم ادعاءات
بحثية/تسويقية جديدة بسرعة، قبل الاستثمار في فهمها بعمق أو تبنّيها.
"""
from dataclasses import dataclass, field
from enum import Enum


class EvidenceLevel(Enum):
    ANECDOTAL = 1          # مثال واحد أو قصة فردية فقط
    BENCHMARK_CLAIMED = 2  # أرقام على معيار قياسي، لكن غير مُستنسخة (replicated)
    BENCHMARK_REPLICATED = 3  # نتائج أكدتها فرق مستقلة متعددة
    PRODUCTION_PROVEN = 4  # مُستخدمة فعليًا في أنظمة إنتاجية واسعة النطاق


@dataclass
class ClaimAssessment:
    claim_description: str
    source_type: str                 # "ورقة محكّمة" / "preprint" / "منشور مدونة شركة" / "تغريدة"
    evidence_level: EvidenceLevel
    has_open_code: bool
    has_been_replicated_independently: bool
    conflicts_of_interest: bool      # هل المصدر لديه مصلحة تجارية مباشرة في الادعاء؟
    comparison_is_fair: bool | None  # هل قورن بأحدث خط أساس (baseline) بإعدادات عادلة؟

    def compute_trust_score(self) -> float:
        """
        نقاط ثقة تقريبية من 0 إلى 10، كدليل استرشادي سريع
        وليس قياسًا علميًا دقيقًا.
        """
        score = self.evidence_level.value * 1.5  # حتى 6 نقاط من مستوى الدليل

        if self.has_open_code:
            score += 1.5
        if self.has_been_replicated_independently:
            score += 2.0
        if self.conflicts_of_interest:
            score -= 1.5
        if self.comparison_is_fair is False:
            score -= 2.0
        elif self.comparison_is_fair is None:
            score -= 0.5  # معلومة غير واضحة = حذر إضافي

        return max(0.0, min(10.0, score))

    def recommendation(self) -> str:
        score = self.compute_trust_score()
        if score >= 7.5:
            return "✅ جدير بالاستثمار في فهم عميق وربما تجربة فعلية"
        elif score >= 5.0:
            return "⚠️ واعد لكن انتظر تأكيدًا إضافيًا قبل الاعتماد عليه في الإنتاج"
        else:
            return "🔍 ضجيج محتمل — راقب من بعيد، لا تستثمر وقتًا كبيرًا الآن"


# ---------- مثال تطبيقي: تقييم ادعاء افتراضي ----------
claim_1 = ClaimAssessment(
    claim_description="نموذج جديد X يتفوق على GPT-4 بنصف التكلفة",
    source_type="منشور مدونة الشركة المُطوِّرة",
    evidence_level=EvidenceLevel.BENCHMARK_CLAIMED,
    has_open_code=False,
    has_been_replicated_independently=False,
    conflicts_of_interest=True,
    comparison_is_fair=None,
)

claim_2 = ClaimAssessment(
    claim_description="DeepSeek-R1 يحقق أداء استدلال منافس بتكلفة تدريب أقل بكثير",
    source_type="ورقة بحثية + كود وأوزان مفتوحة بالكامل",
    evidence_level=EvidenceLevel.BENCHMARK_REPLICATED,
    has_open_code=True,
    has_been_replicated_independently=True,
    conflicts_of_interest=False,
    comparison_is_fair=True,
)

for claim in [claim_1, claim_2]:
    print(f"\n📌 الادعاء: {claim.claim_description}")
    print(f"   نقاط الثقة: {claim.compute_trust_score():.1f}/10")
    print(f"   التوصية: {claim.recommendation()}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-69-chapter-69.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
learning_routine_tracker.py
أداة بسيطة لتتبع التزامك بروتين التعلّم المستدام عبر الزمن،
وتوليد تذكيرات وتقارير تقدّم.
"""
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum


class RoutineFrequency(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


@dataclass
class LearningActivity:
    activity_date: date
    frequency_type: RoutineFrequency
    description: str
    time_spent_minutes: int


@dataclass
class LearningTracker:
    activities: list[LearningActivity] = field(default_factory=list)

    def log_activity(
        self, frequency_type: RoutineFrequency, description: str, minutes: int
    ) -> None:
        self.activities.append(
            LearningActivity(date.today(), frequency_type, description, minutes)
        )

    def get_weekly_summary(self) -> dict[str, int]:
        one_week_ago = date.today() - timedelta(days=7)
        recent = [a for a in self.activities if a.activity_date >= one_week_ago]

        summary = {freq.value: 0 for freq in RoutineFrequency}
        for activity in recent:
            summary[activity.frequency_type.value] += activity.time_spent_minutes

        return summary

    def check_consistency_alert(self) -> list[str]:
        """تنبيهات بسيطة إذا انقطع الالتزام بأحد الأنماط الأساسية"""
        alerts = []
        one_week_ago = date.today() - timedelta(days=7)
        two_weeks_ago = date.today() - timedelta(days=14)

        weekly_deep_reads = [
            a for a in self.activities
            if a.frequency_type == RoutineFrequency.WEEKLY and a.activity_date >= one_week_ago
        ]
        if not weekly_deep_reads:
            alerts.append("⚠️ لم تقرأ أي ورقة بحثية بعمق خلال الأسبوع الماضي.")

        monthly_hands_on = [
            a for a in self.activities
            if a.frequency_type == RoutineFrequency.MONTHLY and a.activity_date >= two_weeks_ago
        ]
        if not monthly_hands_on and date.today().day > 20:
            alerts.append("⚠️ اقترب نهاية الشهر ولم تُجرِ تجربة عملية بعد.")

        return alerts


# مثال استخدام
tracker = LearningTracker()
tracker.log_activity(RoutineFrequency.DAILY, "فحص عناوين arXiv وHacker News", 15)
tracker.log_activity(
    RoutineFrequency.WEEKLY,
    "قراءة عميقة لورقة RULER حول تقييم السياق الطويل",
    120,
)
tracker.log_activity(
    RoutineFrequency.MONTHLY,
    "تجربة ضبط دقيق لـ Phi-3-mini على مهمة تصنيف شخصية",
    240,
)

print("📊 ملخص الأسبوع:")
for freq, minutes in tracker.get_weekly_summary().items():
    print(f"   {freq}: {minutes} دقيقة")

alerts = tracker.check_consistency_alert()
if alerts:
    print("\n🔔 تنبيهات:")
    for alert in alerts:
        print(f"   {alert}")
else:
    print("\n✅ الالتزام بالروتين جيد هذا الأسبوع.")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-69-chapter-69.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
"""
personal_research_radar.py
نظام متكامل End-to-End: "رادار بحثي شخصي" يجلب أحدث الأوراق من arXiv
عبر الـ API الرسمي المجاني، يفلترها حسب اهتماماتك المحددة،
يطبّق تقييمًا أوليًا، وينتج تقريرًا أسبوعيًا جاهزًا للقراءة.

يستخدم مكتبة urllib القياسية (بدون الحاجة لمكتبات خارجية إضافية)
للاتصال بـ arXiv API الحقيقي والعامل فعليًا.
"""
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class ArxivPaper:
    title: str
    summary: str
    authors: list[str]
    published_date: datetime
    arxiv_id: str
    categories: list[str]
    pdf_url: str

    def relevance_score(self, interest_keywords: list[str]) -> float:
        """
        تقييم بسيط لمدى صلة الورقة باهتماماتك، بناءً على
        تطابق الكلمات المفتاحية في العنوان والملخص.
        (في نظام أكثر تقدمًا، يمكن استبدال هذا بتشابه embeddings دلالي).
        """
        combined_text = (self.title + " " + self.summary).lower()
        matches = sum(1 for kw in interest_keywords if kw.lower() in combined_text)
        return matches / max(len(interest_keywords), 1)


class ArxivResearchRadar:
    """واجهة بسيطة وفعلية للتفاعل مع arXiv API العام والمجاني"""

    BASE_URL = "http://export.arxiv.org/api/query"
    ATOM_NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}

    def fetch_recent_papers(
        self, search_query: str, max_results: int = 20
    ) -> list[ArxivPaper]:
        """
        جلب أحدث الأوراق البحثية من arXiv حسب استعلام بحث محدد.
        مثال على search_query: "cat:cs.CL AND abs:reasoning"
        """
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"

        with urllib.request.urlopen(url, timeout=15) as response:
            xml_data = response.read()

        return self._parse_arxiv_response(xml_data)

    def _parse_arxiv_response(self, xml_data: bytes) -> list[ArxivPaper]:
        root = ET.fromstring(xml_data)
        papers = []

        for entry in root.findall("atom:entry", self.ATOM_NAMESPACE):
            title = entry.find("atom:title", self.ATOM_NAMESPACE).text.strip()
            summary = entry.find("atom:summary", self.ATOM_NAMESPACE).text.strip()
            arxiv_id = entry.find("atom:id", self.ATOM_NAMESPACE).text.strip()

            published_str = entry.find("atom:published", self.ATOM_NAMESPACE).text
            published_date = datetime.strptime(published_str, "%Y-%m-%dT%H:%M:%SZ")

            authors = [
                author.find("atom:name", self.ATOM_NAMESPACE).text
                for author in entry.findall("atom:author", self.ATOM_NAMESPACE)
            ]

            categories = [
                cat.attrib.get("term", "")
                for cat in entry.findall(
                    "{http://arxiv.org/schemas/atom}category"
                )
            ] or [
                cat.attrib.get("term", "")
                for cat in entry.findall("atom:category", self.ATOM_NAMESPACE)
            ]

            pdf_url = arxiv_id.replace("abs", "pdf")

            papers.append(ArxivPaper(
                title=title, summary=summary, authors=authors,
                published_date=published_date, arxiv_id=arxiv_id,
                categories=categories, pdf_url=pdf_url,
            ))

        return papers


@dataclass
class WeeklyResearchReport:
    generated_at: datetime
    top_papers: list[tuple[ArxivPaper, float]] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"📬 تقرير الرادار البحثي الأسبوعي — {self.generated_at.strftime('%Y-%m-%d')}",
            "=" * 70,
        ]
        for rank, (paper, score) in enumerate(self.top_papers, start=1):
            lines.append(f"\n{rank}. {paper.title}")
            lines.append(f"   📅 تاريخ النشر: {paper.published_date.strftime('%Y-%m-%d')}")
            lines.append(f"   👥 المؤلفون: {', '.join(paper.authors[:3])}"
                          f"{' وآخرون' if len(paper.authors) > 3 else ''}")
            lines.append(f"   🎯 درجة الصلة باهتماماتك: {score:.0%}")
            lines.append(f"   📄 الملخص: {paper.summary[:200]}...")
            lines.append(f"   🔗 الرابط: {paper.pdf_url}")
        return "\n".join(lines)


def generate_personalized_weekly_report(
    interest_keywords: list[str],
    arxiv_category: str = "cs.CL",
    top_n: int = 5,
) -> WeeklyResearchReport:
    """
    الدالة الرئيسية: تجلب أوراقًا حديثة، تُقيّمها حسب اهتماماتك،
    وتُرجع تقريرًا جاهزًا بأهم top_n ورقة صلة.
    """
    radar = ArxivResearchRadar()

    search_query = f"cat:{arxiv_category}"
    papers = radar.fetch_recent_papers(search_query, max_results=50)

    scored_papers = [
        (paper, paper.relevance_score(interest_keywords)) for paper in papers
    ]
    scored_papers.sort(key=lambda x: x[1], reverse=True)

    report = WeeklyResearchReport(
        generated_at=datetime.now(),
        top_papers=scored_papers[:top_n],
    )
    return report


# ---------- تشغيل فعلي للنظام ----------
if __name__ == "__main__":
    my_interests = [
        "reasoning", "long context", "efficient inference",
        "agent", "small language model", "test-time compute",
    ]

    report = generate_personalized_weekly_report(
        interest_keywords=my_interests,
        arxiv_category="cs.CL",
        top_n=5,
    )

    print(report.render())

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-69-chapter-69.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — خاتمة — منهجية متابعة الأبحاث ومواكبة التسارع التقني
Large Language Model Engineering — Chapter 69 — Frontier Trends

Book slug : llm-engineering-69-chapters
Chapter   : ch-69-chapter-69 (lesson 69)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_69.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 69: الفصل 69: خاتمة — منهجية متابعة الأبحاث ومواكبة التسارع التقني."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_research() -> None:
    papers = [{'title': 'Efficient Attention', 'year': 2024, 'relevance': 0.92}, {'title': 'Reliable Evaluation', 'year': 2025, 'relevance': 0.87}, {'title': 'Small Language Models', 'year': 2023, 'relevance': 0.89}]
    ranked = sorted(papers, key=lambda paper: (paper['relevance'], paper['year']), reverse=True)
    reading_log = [{'title': paper['title'], 'status': 'to-read'} for paper in ranked]
    print('ranked reading list:', reading_log)
    print('record exact citations and verify sources before citing.')

if __name__ == "__main__":
    print("Chapter 69 lab: Rank and track a reproducible research reading list")
    run_research()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-12-frontier-trends/chapter-69.md
# ────────────────────────────────────────────────────────────────────────
"""
claim_evaluation_checklist.py
أداة مساعدة منهجية (وليست أداة آلية تحليل نصوص) لتقييم ادعاءات
بحثية/تسويقية جديدة بسرعة، قبل الاستثمار في فهمها بعمق أو تبنّيها.
"""
from dataclasses import dataclass, field
from enum import Enum


class EvidenceLevel(Enum):
    ANECDOTAL = 1          # مثال واحد أو قصة فردية فقط
    BENCHMARK_CLAIMED = 2  # أرقام على معيار قياسي، لكن غير مُستنسخة (replicated)
    BENCHMARK_REPLICATED = 3  # نتائج أكدتها فرق مستقلة متعددة
    PRODUCTION_PROVEN = 4  # مُستخدمة فعليًا في أنظمة إنتاجية واسعة النطاق


@dataclass
class ClaimAssessment:
    claim_description: str
    source_type: str                 # "ورقة محكّمة" / "preprint" / "منشور مدونة شركة" / "تغريدة"
    evidence_level: EvidenceLevel
    has_open_code: bool
    has_been_replicated_independently: bool
    conflicts_of_interest: bool      # هل المصدر لديه مصلحة تجارية مباشرة في الادعاء؟
    comparison_is_fair: bool | None  # هل قورن بأحدث خط أساس (baseline) بإعدادات عادلة؟

    def compute_trust_score(self) -> float:
        """
        نقاط ثقة تقريبية من 0 إلى 10، كدليل استرشادي سريع
        وليس قياسًا علميًا دقيقًا.
        """
        score = self.evidence_level.value * 1.5  # حتى 6 نقاط من مستوى الدليل

        if self.has_open_code:
            score += 1.5
        if self.has_been_replicated_independently:
            score += 2.0
        if self.conflicts_of_interest:
            score -= 1.5
        if self.comparison_is_fair is False:
            score -= 2.0
        elif self.comparison_is_fair is None:
            score -= 0.5  # معلومة غير واضحة = حذر إضافي

        return max(0.0, min(10.0, score))

    def recommendation(self) -> str:
        score = self.compute_trust_score()
        if score >= 7.5:
            return "✅ جدير بالاستثمار في فهم عميق وربما تجربة فعلية"
        elif score >= 5.0:
            return "⚠️ واعد لكن انتظر تأكيدًا إضافيًا قبل الاعتماد عليه في الإنتاج"
        else:
            return "🔍 ضجيج محتمل — راقب من بعيد، لا تستثمر وقتًا كبيرًا الآن"


# ---------- مثال تطبيقي: تقييم ادعاء افتراضي ----------
claim_1 = ClaimAssessment(
    claim_description="نموذج جديد X يتفوق على GPT-4 بنصف التكلفة",
    source_type="منشور مدونة الشركة المُطوِّرة",
    evidence_level=EvidenceLevel.BENCHMARK_CLAIMED,
    has_open_code=False,
    has_been_replicated_independently=False,
    conflicts_of_interest=True,
    comparison_is_fair=None,
)

claim_2 = ClaimAssessment(
    claim_description="DeepSeek-R1 يحقق أداء استدلال منافس بتكلفة تدريب أقل بكثير",
    source_type="ورقة بحثية + كود وأوزان مفتوحة بالكامل",
    evidence_level=EvidenceLevel.BENCHMARK_REPLICATED,
    has_open_code=True,
    has_been_replicated_independently=True,
    conflicts_of_interest=False,
    comparison_is_fair=True,
)

for claim in [claim_1, claim_2]:
    print(f"\n📌 الادعاء: {claim.claim_description}")
    print(f"   نقاط الثقة: {claim.compute_trust_score():.1f}/10")
    print(f"   التوصية: {claim.recommendation()}")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-12-frontier-trends/chapter-69.md
# ────────────────────────────────────────────────────────────────────────
"""
learning_routine_tracker.py
أداة بسيطة لتتبع التزامك بروتين التعلّم المستدام عبر الزمن،
وتوليد تذكيرات وتقارير تقدّم.
"""
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum


class RoutineFrequency(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


@dataclass
class LearningActivity:
    activity_date: date
    frequency_type: RoutineFrequency
    description: str
    time_spent_minutes: int


@dataclass
class LearningTracker:
    activities: list[LearningActivity] = field(default_factory=list)

    def log_activity(
        self, frequency_type: RoutineFrequency, description: str, minutes: int
    ) -> None:
        self.activities.append(
            LearningActivity(date.today(), frequency_type, description, minutes)
        )

    def get_weekly_summary(self) -> dict[str, int]:
        one_week_ago = date.today() - timedelta(days=7)
        recent = [a for a in self.activities if a.activity_date >= one_week_ago]

        summary = {freq.value: 0 for freq in RoutineFrequency}
        for activity in recent:
            summary[activity.frequency_type.value] += activity.time_spent_minutes

        return summary

    def check_consistency_alert(self) -> list[str]:
        """تنبيهات بسيطة إذا انقطع الالتزام بأحد الأنماط الأساسية"""
        alerts = []
        one_week_ago = date.today() - timedelta(days=7)
        two_weeks_ago = date.today() - timedelta(days=14)

        weekly_deep_reads = [
            a for a in self.activities
            if a.frequency_type == RoutineFrequency.WEEKLY and a.activity_date >= one_week_ago
        ]
        if not weekly_deep_reads:
            alerts.append("⚠️ لم تقرأ أي ورقة بحثية بعمق خلال الأسبوع الماضي.")

        monthly_hands_on = [
            a for a in self.activities
            if a.frequency_type == RoutineFrequency.MONTHLY and a.activity_date >= two_weeks_ago
        ]
        if not monthly_hands_on and date.today().day > 20:
            alerts.append("⚠️ اقترب نهاية الشهر ولم تُجرِ تجربة عملية بعد.")

        return alerts


# مثال استخدام
tracker = LearningTracker()
tracker.log_activity(RoutineFrequency.DAILY, "فحص عناوين arXiv وHacker News", 15)
tracker.log_activity(
    RoutineFrequency.WEEKLY,
    "قراءة عميقة لورقة RULER حول تقييم السياق الطويل",
    120,
)
tracker.log_activity(
    RoutineFrequency.MONTHLY,
    "تجربة ضبط دقيق لـ Phi-3-mini على مهمة تصنيف شخصية",
    240,
)

print("📊 ملخص الأسبوع:")
for freq, minutes in tracker.get_weekly_summary().items():
    print(f"   {freq}: {minutes} دقيقة")

alerts = tracker.check_consistency_alert()
if alerts:
    print("\n🔔 تنبيهات:")
    for alert in alerts:
        print(f"   {alert}")
else:
    print("\n✅ الالتزام بالروتين جيد هذا الأسبوع.")

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-12-frontier-trends/chapter-69.md
# ────────────────────────────────────────────────────────────────────────
"""
personal_research_radar.py
نظام متكامل End-to-End: "رادار بحثي شخصي" يجلب أحدث الأوراق من arXiv
عبر الـ API الرسمي المجاني، يفلترها حسب اهتماماتك المحددة،
يطبّق تقييمًا أوليًا، وينتج تقريرًا أسبوعيًا جاهزًا للقراءة.

يستخدم مكتبة urllib القياسية (بدون الحاجة لمكتبات خارجية إضافية)
للاتصال بـ arXiv API الحقيقي والعامل فعليًا.
"""
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class ArxivPaper:
    title: str
    summary: str
    authors: list[str]
    published_date: datetime
    arxiv_id: str
    categories: list[str]
    pdf_url: str

    def relevance_score(self, interest_keywords: list[str]) -> float:
        """
        تقييم بسيط لمدى صلة الورقة باهتماماتك، بناءً على
        تطابق الكلمات المفتاحية في العنوان والملخص.
        (في نظام أكثر تقدمًا، يمكن استبدال هذا بتشابه embeddings دلالي).
        """
        combined_text = (self.title + " " + self.summary).lower()
        matches = sum(1 for kw in interest_keywords if kw.lower() in combined_text)
        return matches / max(len(interest_keywords), 1)


class ArxivResearchRadar:
    """واجهة بسيطة وفعلية للتفاعل مع arXiv API العام والمجاني"""

    BASE_URL = "http://export.arxiv.org/api/query"
    ATOM_NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}

    def fetch_recent_papers(
        self, search_query: str, max_results: int = 20
    ) -> list[ArxivPaper]:
        """
        جلب أحدث الأوراق البحثية من arXiv حسب استعلام بحث محدد.
        مثال على search_query: "cat:cs.CL AND abs:reasoning"
        """
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"

        with urllib.request.urlopen(url, timeout=15) as response:
            xml_data = response.read()

        return self._parse_arxiv_response(xml_data)

    def _parse_arxiv_response(self, xml_data: bytes) -> list[ArxivPaper]:
        root = ET.fromstring(xml_data)
        papers = []

        for entry in root.findall("atom:entry", self.ATOM_NAMESPACE):
            title = entry.find("atom:title", self.ATOM_NAMESPACE).text.strip()
            summary = entry.find("atom:summary", self.ATOM_NAMESPACE).text.strip()
            arxiv_id = entry.find("atom:id", self.ATOM_NAMESPACE).text.strip()

            published_str = entry.find("atom:published", self.ATOM_NAMESPACE).text
            published_date = datetime.strptime(published_str, "%Y-%m-%dT%H:%M:%SZ")

            authors = [
                author.find("atom:name", self.ATOM_NAMESPACE).text
                for author in entry.findall("atom:author", self.ATOM_NAMESPACE)
            ]

            categories = [
                cat.attrib.get("term", "")
                for cat in entry.findall(
                    "{http://arxiv.org/schemas/atom}category"
                )
            ] or [
                cat.attrib.get("term", "")
                for cat in entry.findall("atom:category", self.ATOM_NAMESPACE)
            ]

            pdf_url = arxiv_id.replace("abs", "pdf")

            papers.append(ArxivPaper(
                title=title, summary=summary, authors=authors,
                published_date=published_date, arxiv_id=arxiv_id,
                categories=categories, pdf_url=pdf_url,
            ))

        return papers


@dataclass
class WeeklyResearchReport:
    generated_at: datetime
    top_papers: list[tuple[ArxivPaper, float]] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"📬 تقرير الرادار البحثي الأسبوعي — {self.generated_at.strftime('%Y-%m-%d')}",
            "=" * 70,
        ]
        for rank, (paper, score) in enumerate(self.top_papers, start=1):
            lines.append(f"\n{rank}. {paper.title}")
            lines.append(f"   📅 تاريخ النشر: {paper.published_date.strftime('%Y-%m-%d')}")
            lines.append(f"   👥 المؤلفون: {', '.join(paper.authors[:3])}"
                          f"{' وآخرون' if len(paper.authors) > 3 else ''}")
            lines.append(f"   🎯 درجة الصلة باهتماماتك: {score:.0%}")
            lines.append(f"   📄 الملخص: {paper.summary[:200]}...")
            lines.append(f"   🔗 الرابط: {paper.pdf_url}")
        return "\n".join(lines)


def generate_personalized_weekly_report(
    interest_keywords: list[str],
    arxiv_category: str = "cs.CL",
    top_n: int = 5,
) -> WeeklyResearchReport:
    """
    الدالة الرئيسية: تجلب أوراقًا حديثة، تُقيّمها حسب اهتماماتك،
    وتُرجع تقريرًا جاهزًا بأهم top_n ورقة صلة.
    """
    radar = ArxivResearchRadar()

    search_query = f"cat:{arxiv_category}"
    papers = radar.fetch_recent_papers(search_query, max_results=50)

    scored_papers = [
        (paper, paper.relevance_score(interest_keywords)) for paper in papers
    ]
    scored_papers.sort(key=lambda x: x[1], reverse=True)

    report = WeeklyResearchReport(
        generated_at=datetime.now(),
        top_papers=scored_papers[:top_n],
    )
    return report


# ---------- تشغيل فعلي للنظام ----------
if __name__ == "__main__":
    my_interests = [
        "reasoning", "long context", "efficient inference",
        "agent", "small language model", "test-time compute",
    ]

    report = generate_personalized_weekly_report(
        interest_keywords=my_interests,
        arxiv_category="cs.CL",
        top_n=5,
    )

    print(report.render())

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-69-chapter-69.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 69: الفصل 69: خاتمة — منهجية متابعة الأبحاث ومواكبة التسارع التقني."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_research() -> None:
    papers = [{'title': 'Efficient Attention', 'year': 2024, 'relevance': 0.92}, {'title': 'Reliable Evaluation', 'year': 2025, 'relevance': 0.87}, {'title': 'Small Language Models', 'year': 2023, 'relevance': 0.89}]
    ranked = sorted(papers, key=lambda paper: (paper['relevance'], paper['year']), reverse=True)
    reading_log = [{'title': paper['title'], 'status': 'to-read'} for paper in ranked]
    print('ranked reading list:', reading_log)
    print('record exact citations and verify sources before citing.')

if __name__ == "__main__":
    print("Chapter 69 lab: Rank and track a reproducible research reading list")
    run_research()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-69-chapter-69.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
claim_evaluation_checklist.py
أداة مساعدة منهجية (وليست أداة آلية تحليل نصوص) لتقييم ادعاءات
بحثية/تسويقية جديدة بسرعة، قبل الاستثمار في فهمها بعمق أو تبنّيها.
"""
from dataclasses import dataclass, field
from enum import Enum


class EvidenceLevel(Enum):
    ANECDOTAL = 1          # مثال واحد أو قصة فردية فقط
    BENCHMARK_CLAIMED = 2  # أرقام على معيار قياسي، لكن غير مُستنسخة (replicated)
    BENCHMARK_REPLICATED = 3  # نتائج أكدتها فرق مستقلة متعددة
    PRODUCTION_PROVEN = 4  # مُستخدمة فعليًا في أنظمة إنتاجية واسعة النطاق


@dataclass
class ClaimAssessment:
    claim_description: str
    source_type: str                 # "ورقة محكّمة" / "preprint" / "منشور مدونة شركة" / "تغريدة"
    evidence_level: EvidenceLevel
    has_open_code: bool
    has_been_replicated_independently: bool
    conflicts_of_interest: bool      # هل المصدر لديه مصلحة تجارية مباشرة في الادعاء؟
    comparison_is_fair: bool | None  # هل قورن بأحدث خط أساس (baseline) بإعدادات عادلة؟

    def compute_trust_score(self) -> float:
        """
        نقاط ثقة تقريبية من 0 إلى 10، كدليل استرشادي سريع
        وليس قياسًا علميًا دقيقًا.
        """
        score = self.evidence_level.value * 1.5  # حتى 6 نقاط من مستوى الدليل

        if self.has_open_code:
            score += 1.5
        if self.has_been_replicated_independently:
            score += 2.0
        if self.conflicts_of_interest:
            score -= 1.5
        if self.comparison_is_fair is False:
            score -= 2.0
        elif self.comparison_is_fair is None:
            score -= 0.5  # معلومة غير واضحة = حذر إضافي

        return max(0.0, min(10.0, score))

    def recommendation(self) -> str:
        score = self.compute_trust_score()
        if score >= 7.5:
            return "✅ جدير بالاستثمار في فهم عميق وربما تجربة فعلية"
        elif score >= 5.0:
            return "⚠️ واعد لكن انتظر تأكيدًا إضافيًا قبل الاعتماد عليه في الإنتاج"
        else:
            return "🔍 ضجيج محتمل — راقب من بعيد، لا تستثمر وقتًا كبيرًا الآن"


# ---------- مثال تطبيقي: تقييم ادعاء افتراضي ----------
claim_1 = ClaimAssessment(
    claim_description="نموذج جديد X يتفوق على GPT-4 بنصف التكلفة",
    source_type="منشور مدونة الشركة المُطوِّرة",
    evidence_level=EvidenceLevel.BENCHMARK_CLAIMED,
    has_open_code=False,
    has_been_replicated_independently=False,
    conflicts_of_interest=True,
    comparison_is_fair=None,
)

claim_2 = ClaimAssessment(
    claim_description="DeepSeek-R1 يحقق أداء استدلال منافس بتكلفة تدريب أقل بكثير",
    source_type="ورقة بحثية + كود وأوزان مفتوحة بالكامل",
    evidence_level=EvidenceLevel.BENCHMARK_REPLICATED,
    has_open_code=True,
    has_been_replicated_independently=True,
    conflicts_of_interest=False,
    comparison_is_fair=True,
)

for claim in [claim_1, claim_2]:
    print(f"\n📌 الادعاء: {claim.claim_description}")
    print(f"   نقاط الثقة: {claim.compute_trust_score():.1f}/10")
    print(f"   التوصية: {claim.recommendation()}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-69-chapter-69.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
"""
learning_routine_tracker.py
أداة بسيطة لتتبع التزامك بروتين التعلّم المستدام عبر الزمن،
وتوليد تذكيرات وتقارير تقدّم.
"""
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum


class RoutineFrequency(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


@dataclass
class LearningActivity:
    activity_date: date
    frequency_type: RoutineFrequency
    description: str
    time_spent_minutes: int


@dataclass
class LearningTracker:
    activities: list[LearningActivity] = field(default_factory=list)

    def log_activity(
        self, frequency_type: RoutineFrequency, description: str, minutes: int
    ) -> None:
        self.activities.append(
            LearningActivity(date.today(), frequency_type, description, minutes)
        )

    def get_weekly_summary(self) -> dict[str, int]:
        one_week_ago = date.today() - timedelta(days=7)
        recent = [a for a in self.activities if a.activity_date >= one_week_ago]

        summary = {freq.value: 0 for freq in RoutineFrequency}
        for activity in recent:
            summary[activity.frequency_type.value] += activity.time_spent_minutes

        return summary

    def check_consistency_alert(self) -> list[str]:
        """تنبيهات بسيطة إذا انقطع الالتزام بأحد الأنماط الأساسية"""
        alerts = []
        one_week_ago = date.today() - timedelta(days=7)
        two_weeks_ago = date.today() - timedelta(days=14)

        weekly_deep_reads = [
            a for a in self.activities
            if a.frequency_type == RoutineFrequency.WEEKLY and a.activity_date >= one_week_ago
        ]
        if not weekly_deep_reads:
            alerts.append("⚠️ لم تقرأ أي ورقة بحثية بعمق خلال الأسبوع الماضي.")

        monthly_hands_on = [
            a for a in self.activities
            if a.frequency_type == RoutineFrequency.MONTHLY and a.activity_date >= two_weeks_ago
        ]
        if not monthly_hands_on and date.today().day > 20:
            alerts.append("⚠️ اقترب نهاية الشهر ولم تُجرِ تجربة عملية بعد.")

        return alerts


# مثال استخدام
tracker = LearningTracker()
tracker.log_activity(RoutineFrequency.DAILY, "فحص عناوين arXiv وHacker News", 15)
tracker.log_activity(
    RoutineFrequency.WEEKLY,
    "قراءة عميقة لورقة RULER حول تقييم السياق الطويل",
    120,
)
tracker.log_activity(
    RoutineFrequency.MONTHLY,
    "تجربة ضبط دقيق لـ Phi-3-mini على مهمة تصنيف شخصية",
    240,
)

print("📊 ملخص الأسبوع:")
for freq, minutes in tracker.get_weekly_summary().items():
    print(f"   {freq}: {minutes} دقيقة")

alerts = tracker.check_consistency_alert()
if alerts:
    print("\n🔔 تنبيهات:")
    for alert in alerts:
        print(f"   {alert}")
else:
    print("\n✅ الالتزام بالروتين جيد هذا الأسبوع.")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-69-chapter-69.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
"""
personal_research_radar.py
نظام متكامل End-to-End: "رادار بحثي شخصي" يجلب أحدث الأوراق من arXiv
عبر الـ API الرسمي المجاني، يفلترها حسب اهتماماتك المحددة،
يطبّق تقييمًا أوليًا، وينتج تقريرًا أسبوعيًا جاهزًا للقراءة.

يستخدم مكتبة urllib القياسية (بدون الحاجة لمكتبات خارجية إضافية)
للاتصال بـ arXiv API الحقيقي والعامل فعليًا.
"""
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class ArxivPaper:
    title: str
    summary: str
    authors: list[str]
    published_date: datetime
    arxiv_id: str
    categories: list[str]
    pdf_url: str

    def relevance_score(self, interest_keywords: list[str]) -> float:
        """
        تقييم بسيط لمدى صلة الورقة باهتماماتك، بناءً على
        تطابق الكلمات المفتاحية في العنوان والملخص.
        (في نظام أكثر تقدمًا، يمكن استبدال هذا بتشابه embeddings دلالي).
        """
        combined_text = (self.title + " " + self.summary).lower()
        matches = sum(1 for kw in interest_keywords if kw.lower() in combined_text)
        return matches / max(len(interest_keywords), 1)


class ArxivResearchRadar:
    """واجهة بسيطة وفعلية للتفاعل مع arXiv API العام والمجاني"""

    BASE_URL = "http://export.arxiv.org/api/query"
    ATOM_NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}

    def fetch_recent_papers(
        self, search_query: str, max_results: int = 20
    ) -> list[ArxivPaper]:
        """
        جلب أحدث الأوراق البحثية من arXiv حسب استعلام بحث محدد.
        مثال على search_query: "cat:cs.CL AND abs:reasoning"
        """
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"

        with urllib.request.urlopen(url, timeout=15) as response:
            xml_data = response.read()

        return self._parse_arxiv_response(xml_data)

    def _parse_arxiv_response(self, xml_data: bytes) -> list[ArxivPaper]:
        root = ET.fromstring(xml_data)
        papers = []

        for entry in root.findall("atom:entry", self.ATOM_NAMESPACE):
            title = entry.find("atom:title", self.ATOM_NAMESPACE).text.strip()
            summary = entry.find("atom:summary", self.ATOM_NAMESPACE).text.strip()
            arxiv_id = entry.find("atom:id", self.ATOM_NAMESPACE).text.strip()

            published_str = entry.find("atom:published", self.ATOM_NAMESPACE).text
            published_date = datetime.strptime(published_str, "%Y-%m-%dT%H:%M:%SZ")

            authors = [
                author.find("atom:name", self.ATOM_NAMESPACE).text
                for author in entry.findall("atom:author", self.ATOM_NAMESPACE)
            ]

            categories = [
                cat.attrib.get("term", "")
                for cat in entry.findall(
                    "{http://arxiv.org/schemas/atom}category"
                )
            ] or [
                cat.attrib.get("term", "")
                for cat in entry.findall("atom:category", self.ATOM_NAMESPACE)
            ]

            pdf_url = arxiv_id.replace("abs", "pdf")

            papers.append(ArxivPaper(
                title=title, summary=summary, authors=authors,
                published_date=published_date, arxiv_id=arxiv_id,
                categories=categories, pdf_url=pdf_url,
            ))

        return papers


@dataclass
class WeeklyResearchReport:
    generated_at: datetime
    top_papers: list[tuple[ArxivPaper, float]] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"📬 تقرير الرادار البحثي الأسبوعي — {self.generated_at.strftime('%Y-%m-%d')}",
            "=" * 70,
        ]
        for rank, (paper, score) in enumerate(self.top_papers, start=1):
            lines.append(f"\n{rank}. {paper.title}")
            lines.append(f"   📅 تاريخ النشر: {paper.published_date.strftime('%Y-%m-%d')}")
            lines.append(f"   👥 المؤلفون: {', '.join(paper.authors[:3])}"
                          f"{' وآخرون' if len(paper.authors) > 3 else ''}")
            lines.append(f"   🎯 درجة الصلة باهتماماتك: {score:.0%}")
            lines.append(f"   📄 الملخص: {paper.summary[:200]}...")
            lines.append(f"   🔗 الرابط: {paper.pdf_url}")
        return "\n".join(lines)


def generate_personalized_weekly_report(
    interest_keywords: list[str],
    arxiv_category: str = "cs.CL",
    top_n: int = 5,
) -> WeeklyResearchReport:
    """
    الدالة الرئيسية: تجلب أوراقًا حديثة، تُقيّمها حسب اهتماماتك،
    وتُرجع تقريرًا جاهزًا بأهم top_n ورقة صلة.
    """
    radar = ArxivResearchRadar()

    search_query = f"cat:{arxiv_category}"
    papers = radar.fetch_recent_papers(search_query, max_results=50)

    scored_papers = [
        (paper, paper.relevance_score(interest_keywords)) for paper in papers
    ]
    scored_papers.sort(key=lambda x: x[1], reverse=True)

    report = WeeklyResearchReport(
        generated_at=datetime.now(),
        top_papers=scored_papers[:top_n],
    )
    return report


# ---------- تشغيل فعلي للنظام ----------
if __name__ == "__main__":
    my_interests = [
        "reasoning", "long context", "efficient inference",
        "agent", "small language model", "test-time compute",
    ]

    report = generate_personalized_weekly_report(
        interest_keywords=my_interests,
        arxiv_category="cs.CL",
        top_n=5,
    )

    print(report.render())
