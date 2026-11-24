#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
هندسة وكلاء الذكاء الاصطناعي — التطبيقات الثلاثة المرجعية الكاملة
AI Agent Engineering — Three Reference Production Applications

Book slug : agent-harness-engineering
Chapter   : app-projects (lesson projects)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/agent-harness-engineering/code/app-projects.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
هندسة وكلاء الذكاء الاصطناعي — التطبيقات الثلاثة المرجعية الكاملة
AI Agent Engineering — Three Reference Production Applications

Book slug : agent-harness-engineering
Chapter   : app-projects (lesson projects)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/agent-harness-engineering/code/app-projects.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
هندسة وكلاء الذكاء الاصطناعي — التطبيقات الثلاثة المرجعية الكاملة
AI Agent Engineering — Three Reference Production Applications

Book slug : agent-harness-engineering
Chapter   : app-projects (lesson projects)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] dist/md/app-projects.md
# ────────────────────────────────────────────────────────────────────────
import asyncio
import time
import uuid
from enum import StrEnum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


# =====================================================================
# 1. العقود المهيكلة لحوادث SRE وإجراءات الإصلاح (Immutable Schemas)
# =====================================================================

class SeverityLevel(StrEnum):
    SEV1_CRITICAL = "SEV1_CRITICAL"
    SEV2_MAJOR = "SEV2_MAJOR"
    SEV3_MINOR = "SEV3_MINOR"


class RemediationActionType(StrEnum):
    SCALE_DEPLOYMENT = "scale_deployment"
    ROLLBACK_RELEASE = "rollback_release"
    FLUSH_STALE_POOL = "flush_stale_connection_pool"
    DELETE_NAMESPACE = "delete_namespace"  # محظور أمنياً!


class IncidentAlert(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    incident_id: str
    service_name: str
    severity: SeverityLevel
    symptom_summary: str
    triggered_at_epoch: float = Field(default_factory=time.time)


class ClusterHealthSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    active_replicas: int = Field(ge=0)
    cpu_utilization_pct: float = Field(ge=0.0, le=100.0)
    db_pool_saturation_pct: float = Field(ge=0.0, le=100.0)
    p99_latency_ms: float = Field(ge=0.0)
    error_rate_pct: float = Field(ge=0.0, le=100.0)
    recent_log_signature: str


class ProposedRemediationPlan(BaseModel):
    """مخطط قرار وكيل SRE: يجبره على كتابة الفرضية الهندسية قبل اختيار الأداة."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    root_cause_hypothesis: str = Field(min_length=15)
    action_type: RemediationActionType
    target_service: str
    target_replicas: int | None = Field(default=None, ge=1, le=50)
    confidence_score: float = Field(ge=0.0, le=1.0)


class IncidentPostMortemReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    incident_id: str
    service_name: str
    resolved_autonomously: bool
    rollback_triggered: bool
    escalated_to_human_oncall: bool
    root_cause_hypothesis: str
    executed_actions: list[str]
    pre_remediation_health: ClusterHealthSnapshot
    post_remediation_health: ClusterHealthSnapshot
    total_mttr_ms: float = Field(ge=0.0, description="Mean Time To Recovery in ms")
    audit_timeline: list[str]


# =====================================================================
# 2. بوابة أمان العمليات السحابية (Blast-Radius Safety Gatekeeper)
# =====================================================================

class BlastRadiusViolationError(PermissionError):
    """يُرفع عندما يقترح الوكيل إجراءً يتجاوز نصف قطر الضرر المسموح به."""


class SREBlastRadiusGatekeeper:
    _FORBIDDEN_ACTIONS: frozenset[RemediationActionType] = frozenset({
        RemediationActionType.DELETE_NAMESPACE,
    })
    _MAX_ALLOWED_REPLICAS: int = 20

    @classmethod
    def validate_plan(
        cls,
        plan: ProposedRemediationPlan,
        current_health: ClusterHealthSnapshot,
    ) -> None:
        if plan.action_type in cls._FORBIDDEN_ACTIONS:
            raise BlastRadiusViolationError(
                f"إجراء تدميري محظور: '{plan.action_type}' ممنوع قطعياً في التشغيل التلقائي."
            )

        if plan.action_type == RemediationActionType.SCALE_DEPLOYMENT:
            if plan.target_replicas is None:
                raise BlastRadiusViolationError(
                    "يجب تحديد target_replicas عند طلب توسيع الخدمة."
                )
            # منع تقليص عدد الحاويات (Scale-Down) أثناء وجود حادثة نشطة!
            if plan.target_replicas < current_health.active_replicas:
                raise BlastRadiusViolationError(
                    f"مرفوض: لا يُسمح بتقليص النسخ من {current_health.active_replicas} "
                    f"إلى {plan.target_replicas} أثناء حادثة نشطة!"
                )
            if plan.target_replicas > cls._MAX_ALLOWED_REPLICAS:
                raise BlastRadiusViolationError(
                    f"تجاوز سقف السعة: طلب {plan.target_replicas} نسخة يتجاوز الحد الأقصى ({cls._MAX_ALLOWED_REPLICAS})."
                )


# =====================================================================
# 3. بيئة البنية التحتية المحاكاة وعقل الوكيل (Cluster Simulator & Engine)
# =====================================================================

class SimulatedKubernetesCluster:
    """بيئة عنقود سحابي قابلة للعكس تدعم اللقطات والتراجع (Snapshots & Rollback)."""

    def __init__(self) -> None:
        self._state = ClusterHealthSnapshot(
            active_replicas=4,
            cpu_utilization_pct=94.5,
            db_pool_saturation_pct=99.2,
            p99_latency_ms=2850.0,
            error_rate_pct=14.8,
            recent_log_signature="FATAL: psycopg.PoolTimeout: connection pool exhausted (active=100/100)",
        )

    def get_snapshot(self) -> ClusterHealthSnapshot:
        return self._state

    def restore_snapshot(self, snapshot: ClusterHealthSnapshot) -> None:
        self._state = snapshot

    async def gather_parallel_diagnostics(self) -> ClusterHealthSnapshot:
        async def _fetch_prom_metrics() -> tuple[float, float, float]:
            await asyncio.sleep(0.02)
            return (
                self._state.cpu_utilization_pct,
                self._state.p99_latency_ms,
                self._state.error_rate_pct,
            )

        async def _fetch_loki_logs() -> str:
            await asyncio.sleep(0.025)
            return self._state.recent_log_signature

        async def _fetch_k8s_state() -> tuple[int, float]:
            await asyncio.sleep(0.015)
            return (self._state.active_replicas, self._state.db_pool_saturation_pct)

        async with asyncio.TaskGroup() as tg:
            t_prom = tg.create_task(_fetch_prom_metrics())
            t_logs = tg.create_task(_fetch_loki_logs())
            t_k8s = tg.create_task(_fetch_k8s_state())

        cpu, lat, err = t_prom.result()
        logs = t_logs.result()
        reps, pool = t_k8s.result()

        return ClusterHealthSnapshot(
            active_replicas=reps,
            cpu_utilization_pct=cpu,
            db_pool_saturation_pct=pool,
            p99_latency_ms=lat,
            error_rate_pct=err,
            recent_log_signature=logs,
        )

    async def apply_action(self, plan: ProposedRemediationPlan) -> None:
        await asyncio.sleep(0.03)
        match plan.action_type:
            case RemediationActionType.SCALE_DEPLOYMENT:
                # فخ هندسي واقعي: زيادة عدد الحاويات عندما يكون مجمع قاعدة البيانات ممتلئاً بنسبة 99%
                # يزيد الطين بلة ويرفع معدل الأخطاء لأن الحاويات الجديدة تطلب اتصالات إضافية!
                new_reps = plan.target_replicas or self._state.active_replicas
                self._state = ClusterHealthSnapshot(
                    active_replicas=new_reps,
                    cpu_utilization_pct=70.0,
                    db_pool_saturation_pct=100.0,
                    p99_latency_ms=4200.0,  # تدهور أسوأ!
                    error_rate_pct=28.5,    # ارتفعت الأخطاء!
                    recent_log_signature="FATAL: max_connections reached on primary DB!",
                )
            case RemediationActionType.FLUSH_STALE_POOL:
                # الحل الصحيح: تفريغ الاتصالات المعلقة يعيد النظام لحالته الطبيعية فوراً
                self._state = ClusterHealthSnapshot(
                    active_replicas=self._state.active_replicas,
                    cpu_utilization_pct=42.0,
                    db_pool_saturation_pct=35.0,
                    p99_latency_ms=115.0,
                    error_rate_pct=0.05,
                    recent_log_signature="INFO: DB connection pool healthy (active=35/100)",
                )
            case _:
                pass


class AutonomousSRERemediationHarness:
    """
    إطار التحكم الذاتي لحوادث SRE:
    يشخّص بالتوازي، ويتحقق من بوابة الأمان، ويطبق الإصلاح بنمط Canary،
    ويتراجع تلقائياً (Auto-Rollback) إذا تدهور مؤشر مستوى الخدمة (SLO)!
    """

    def __init__(self, cluster: SimulatedKubernetesCluster) -> None:
        self._cluster = cluster

    @staticmethod
    def _is_slo_healthy(health: ClusterHealthSnapshot) -> bool:
        return health.error_rate_pct < 1.0 and health.p99_latency_ms < 400.0

    async def _propose_plan(
        self,
        alert: IncidentAlert,
        health: ClusterHealthSnapshot,
        attempt: int,
        previous_failure_note: str,
    ) -> ProposedRemediationPlan:
        await asyncio.sleep(0.01)
        if attempt == 1:
            # في المحاولة الأولى يقترح الوكيل توسيع عدد الحاويات (Scale Up) ظناً منه أنها مشكلة CPU
            return ProposedRemediationPlan(
                root_cause_hypothesis=(
                    "استهلاك المعالج 94.5%، سأقوم بمضاعفة عدد الحاويات من 4 إلى 8 لتخفيف الضغط."
                ),
                action_type=RemediationActionType.SCALE_DEPLOYMENT,
                target_service=alert.service_name,
                target_replicas=8,
                confidence_score=0.75,
            )

        # في المحاولة الثانية (بعد التراجع الآلي وقراءة ملاحظة فشل Canary):
        # يدرك الوكيل أن المشكلة هي تشبع مجمع اتصالات قاعدة البيانات وليس المعالج!
        return ProposedRemediationPlan(
            root_cause_hypothesis=(
                f"توسيع الحاويات أدى لتدهور الكمون لأن المشكلة الجذرية هي تشبع DB Pool "
                f"({health.db_pool_saturation_pct}%). يجب تفريغ الاتصالات الخاملة المعلقة."
            ),
            action_type=RemediationActionType.FLUSH_STALE_POOL,
            target_service=alert.service_name,
            confidence_score=0.96,
        )

    async def handle_incident(
        self, alert: IncidentAlert, max_attempts: int = 2
    ) -> IncidentPostMortemReport:
        t0 = time.perf_counter()
        timeline: list[str] = [
            f"[{alert.incident_id}] استلام إنذار {alert.severity} للخدمة '{alert.service_name}': {alert.symptom_summary}"
        ]

        # 1. التشخيص المتوازي عبر TaskGroup
        pre_health = await self._cluster.gather_parallel_diagnostics()
        timeline.append(
            f"اكتمل التشخيص المتوازي: P99={pre_health.p99_latency_ms}ms, "
            f"ErrorRate={pre_health.error_rate_pct}%, DB_Pool={pre_health.db_pool_saturation_pct}%"
        )

        executed_actions: list[str] = []
        rollback_triggered = False
        failure_feedback = ""
        last_hypothesis = ""

        for attempt in range(1, max_attempts + 1):
            current_snapshot = self._cluster.get_snapshot()
            plan = await self._propose_plan(
                alert, current_snapshot, attempt, failure_feedback
            )
            last_hypothesis = plan.root_cause_hypothesis
            timeline.append(
                f"[محاولة #{attempt}] الفرضية: {plan.root_cause_hypothesis} -> الإجراء: {plan.action_type}"
            )

            # 2. فحص بوابة نصف قطر الضرر (Blast-Radius Gate)
            try:
                SREBlastRadiusGatekeeper.validate_plan(plan, current_snapshot)
            except BlastRadiusViolationError as exc:
                timeline.append(f"🛑 حُجب الإجراء بواسطة بوابة الأمان: {exc}")
                failure_feedback = str(exc)
                continue

            # 3. تطبيق الإجراء مع الاحتفاظ بلقطة تراجع فورية (Canary Execution)
            await self._cluster.apply_action(plan)
            executed_actions.append(f"{plan.action_type.value} (attempt #{attempt})")

            # 4. فحص مؤشر مستوى الخدمة (SLO Health Verification)
            post_action_health = await self._cluster.gather_parallel_diagnostics()
            if self._is_slo_healthy(post_action_health):
                timeline.append(
                    f"✅ تعافى مؤشر الخدمة (SLO Recovered)! هبط الكمون إلى "
                    f"{post_action_health.p99_latency_ms}ms ومعدل الخطأ إلى {post_action_health.error_rate_pct}%."
                )
                mttr = round((time.perf_counter() - t0) * 1000, 2)
                return IncidentPostMortemReport(
                    incident_id=alert.incident_id,
                    service_name=alert.service_name,
                    resolved_autonomously=True,
                    rollback_triggered=rollback_triggered,
                    escalated_to_human_oncall=False,
                    root_cause_hypothesis=last_hypothesis,
                    executed_actions=executed_actions,
                    pre_remediation_health=pre_health,
                    post_remediation_health=post_action_health,
                    total_mttr_ms=mttr,
                    audit_timeline=timeline,
                )

            # 5. فشل مؤشر الخدمة بعد التعديل -> تراجع ذري فوري (Automatic Canary Rollback)!
            rollback_triggered = True
            self._cluster.restore_snapshot(current_snapshot)
            failure_feedback = (
                f"فشل Canary: بعد تنفيذ {plan.action_type} ارتفع الكمون إلى "
                f"{post_action_health.p99_latency_ms}ms ومعدل الأخطاء إلى {post_action_health.error_rate_pct}%."
            )
            timeline.append(
                f"⚠️ {failure_feedback} -> تم تنفيذ التراجع التلقائي (Auto-Rollback) لاستعادة اللقطة الآمنة!"
            )

        # إذا استُنفدت المحاولات، يتم التصعيد للمهندس المناوب
        final_health = self._cluster.get_snapshot()
        timeline.append("🚨 استُنفدت المحاولات الآمنة. تم تصعيد الحادثة للمهندس المناوب (PagerDuty).")
        mttr = round((time.perf_counter() - t0) * 1000, 2)
        return IncidentPostMortemReport(
            incident_id=alert.incident_id,
            service_name=alert.service_name,
            resolved_autonomously=False,
            rollback_triggered=rollback_triggered,
            escalated_to_human_oncall=True,
            root_cause_hypothesis=last_hypothesis,
            executed_actions=executed_actions,
            pre_remediation_health=pre_health,
            post_remediation_health=final_health,
            total_mttr_ms=mttr,
            audit_timeline=timeline,
        )


async def run_sre_application() -> None:
    cluster = SimulatedKubernetesCluster()
    harness = AutonomousSRERemediationHarness(cluster)

    alert = IncidentAlert(
        incident_id=f"INC-{uuid.uuid4().hex[:6].upper()}",
        service_name="checkout-payment-api",
        severity=SeverityLevel.SEV1_CRITICAL,
        symptom_summary="P99 latency > 2500ms and HTTP 503 spike on checkout-payment-api",
    )

    print("=" * 82)
    print("🚀 التطبيق الأول: وكيل SRE للاستجابة الذاتية للحوادث مع التراجع الآلي (Canary Rollback)")
    print("=" * 82)
    report = await harness.handle_incident(alert)

    for idx, entry in enumerate(report.audit_timeline, start=1):
        print(f"  {idx}. {entry}")

    print("-" * 82)
    print(
        f"📊 النتيجة النهائية: Resolved={report.resolved_autonomously} | "
        f"AutoRollbackUsed={report.rollback_triggered} | MTTR={report.total_mttr_ms}ms"
    )
    print(
        f"   • قبل الإصلاح : P99={report.pre_remediation_health.p99_latency_ms}ms, "
        f"Errors={report.pre_remediation_health.error_rate_pct}%"
    )
    print(
        f"   • بعد الإصلاح : P99={report.post_remediation_health.p99_latency_ms}ms, "
        f"Errors={report.post_remediation_health.error_rate_pct}%"
    )


if __name__ == "__main__":
    asyncio.run(run_sre_application())

# ────────────────────────────────────────────────────────────────────────
# [002] dist/md/app-projects.md
# ────────────────────────────────────────────────────────────────────────
import asyncio
import re
import sqlite3
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator


# =====================================================================
# 1. سياق الأمان وجدار حماية استعلامات SQL (Zero-Trust SQL Guard)
# =====================================================================

class TenantContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=3)
    analyst_id: str
    allow_raw_pii: bool = False


class ValidatedParameterizedSQL(BaseModel):
    """
    عقد صارم لاستعلام SQL:
    1. يجب أن يبدأ بـ SELECT.
    2. يمنع أي أوامر تعديل أو تعليقات أو استعلامات متعددة (;).
    3. يجب أن يحتوي إلزامياً على المعامل الآمن :injected_tenant_id لعزل البيانات.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    reasoning: str = Field(min_length=10)
    sql_query: str = Field(min_length=15, max_length=2000)

    @field_validator("sql_query")
    @classmethod
    def enforce_zero_trust_sql_rules(cls, v: str) -> str:
        cleaned = v.strip().rstrip(";")
        if ";" in cleaned or "--" in cleaned or "/*" in cleaned:
            raise ValueError(
                "يُمنع استخدام الفاصلة المنقوطة المتعددة أو تعليقات SQL."
            )

        upper_sql = cleaned.upper()
        if not upper_sql.startswith("SELECT"):
            raise ValueError("يُسمح فقط باستعلامات القراءة التي تبدأ بـ SELECT.")

        forbidden = {
            "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
            "CREATE", "PRAGMA", "ATTACH", "DETACH", "REPLACE",
        }
        tokens = set(re.findall(r"\b[A-Z_]+\b", upper_sql))
        violations = tokens.intersection(forbidden)
        if violations:
            raise ValueError(f"كلمات مفتاحية محظورة أمنياً في الاستعلام: {violations}")

        if ":injected_tenant_id" not in cleaned:
            raise ValueError(
                "خرق عزل المستأجرين (Row-Level Security Violation): يجب أن يتضمن الاستعلام "
                "شرط 'tenant_id = :injected_tenant_id' صراحةً!"
            )

        if "LIMIT" not in tokens:
            cleaned = f"{cleaned} LIMIT 50"

        return cleaned


class AnalyticsExecutionReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str
    executed_sql: str
    attempts_taken: int
    self_healed_errors: list[str]
    row_count: int
    sanitized_rows: list[dict[str, Any]]


# =====================================================================
# 2. محرك قاعدة البيانات المتعددة المستأجرين وحلقة التصحيح الذاتي
# =====================================================================

class EnterpriseAnalyticsDatabase:
    """قاعدة بيانات SQLite حقيقية في الذاكرة تضم بيانات لعدة شركات متنافسة."""

    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._seed_multi_tenant_data()

    def _seed_multi_tenant_data(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE subscription_invoices (
                    invoice_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    region TEXT NOT NULL,
                    customer_contact_email TEXT NOT NULL,
                    net_revenue_usd REAL NOT NULL,
                    status TEXT NOT NULL
                )
                """
            )
            rows = [
                # بيانات المستأجر الأول: ACME_CORP
                ("INV-101", "ACME_CORP", "MENA", "cfo@client-a.com", 45_000.0, "PAID"),
                ("INV-102", "ACME_CORP", "MENA", "finance@client-b.com", 32_500.0, "PAID"),
                ("INV-103", "ACME_CORP", "EU", "billing@eu-partner.de", 18_000.0, "PAID"),
                # بيانات المستأجر المنافس: GLOBEX_INC (يجب ألا تتسرب أبداً لـ ACME_CORP!)
                ("INV-999", "GLOBEX_INC", "MENA", "secret@globex.io", 999_999.0, "PAID"),
            ]
            self._conn.executemany(
                "INSERT INTO subscription_invoices VALUES (?, ?, ?, ?, ?, ?)", rows
            )

    def introspect_schema_ddl(self) -> str:
        """يستخرج المخطط الحقيقي للجداول والأعمدة لتغذية حلقة التصحيح الذاتي."""
        cur = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        return "\n".join(row["sql"] for row in cur.fetchall() if row["sql"])

    def execute_tenant_scoped_query(
        self, sql: str, tenant_id: str
    ) -> list[dict[str, Any]]:
        cur = self._conn.execute(sql, {"injected_tenant_id": tenant_id})
        return [dict(row) for row in cur.fetchall()]


class SelfHealingTextToSQLHarness:
    """
    يدير توليد SQL، ويطبق جدار الحماية، ويعالج أخطاء قاعدة البيانات ذاتياً
    عبر حقن مخطط DDL الفعلي، ويحجب البيانات الشخصية (PII Masking).
    """

    def __init__(
        self, db: EnterpriseAnalyticsDatabase, max_healing_retries: int = 3
    ) -> None:
        self._db = db
        self._max_retries = max_healing_retries

    @staticmethod
    def _mask_pii_in_row(row: dict[str, Any], allow_raw_pii: bool) -> dict[str, Any]:
        if allow_raw_pii:
            return row
        masked: dict[str, Any] = {}
        for k, val in row.items():
            if isinstance(val, str) and "@" in val:
                user_part, _, domain = val.partition("@")
                masked[k] = f"{user_part[:2]}***@{domain}"
            else:
                masked[k] = val
        return masked

    async def _simulate_llm_sql_generator(
        self,
        question: str,
        attempt: int,
        error_history: list[str],
        schema_hint: str,
    ) -> dict[str, str]:
        await asyncio.sleep(0.01)
        if attempt == 1:
            # المحاولة 1: النموذج ينسى وضع شرط :injected_tenant_id (سيصطاده Pydantic Firewall!)
            return {
                "reasoning": "سأقوم بجمع الفواتير حسب المنطقة.",
                "sql_query": "SELECT region, SUM(total_amount) FROM invoices GROUP BY region",
            }
        if attempt == 2:
            # المحاولة 2: النموذج يلتزم بشرط المستأجر، لكنه يهلوس باسم الجدول والعمود (total_amount بدلاً من net_revenue_usd)
            # ستصطاده قاعدة بيانات SQLite بـ OperationalError!
            return {
                "reasoning": "أضفت شرط عزل المستأجر، وسأستعلم عن مجموع الإيرادات.",
                "sql_query": (
                    "SELECT region, customer_contact_email, total_amount "
                    "FROM subscription_invoices "
                    "WHERE tenant_id = :injected_tenant_id"
                ),
            }
        # المحاولة 3: بعد أن حقن الـ Harness مخطط DDL الفعلي من SQLite، يكتب النموذج الاستعلام الصحيح تماماً!
        return {
            "reasoning": (
                "بناءً على مخطط الجدول المحقون، العمود الصحيح هو net_revenue_usd وليس total_amount."
            ),
            "sql_query": (
                "SELECT invoice_id, region, customer_contact_email, net_revenue_usd "
                "FROM subscription_invoices "
                "WHERE tenant_id = :injected_tenant_id AND status = 'PAID' "
                "ORDER BY net_revenue_usd DESC"
            ),
        }

    async def run_analytics_query(
        self, ctx: TenantContext, natural_language_question: str
    ) -> AnalyticsExecutionReport:
        healed_errors: list[str] = []
        schema_hint = ""

        for attempt in range(1, self._max_retries + 1):
            raw_llm_output = await self._simulate_llm_sql_generator(
                question=natural_language_question,
                attempt=attempt,
                error_history=healed_errors,
                schema_hint=schema_hint,
            )

            # البوابة 1: فحص Pydantic و Zero-Trust SQL Firewall
            try:
                validated = ValidatedParameterizedSQL.model_validate(raw_llm_output)
            except Exception as val_exc:
                err_msg = f"[Attempt #{attempt} Firewall Reject]: {val_exc}"
                healed_errors.append(err_msg)
                continue

            # البوابة 2: التنفيذ الفعلي داخل SQLite مع ربط المعاملات (Parameter Binding)
            try:
                raw_rows = self._db.execute_tenant_scoped_query(
                    sql=validated.sql_query,
                    tenant_id=ctx.tenant_id,
                )
                sanitized = [
                    self._mask_pii_in_row(r, ctx.allow_raw_pii) for r in raw_rows
                ]
                return AnalyticsExecutionReport(
                    tenant_id=ctx.tenant_id,
                    executed_sql=validated.sql_query,
                    attempts_taken=attempt,
                    self_healed_errors=healed_errors,
                    row_count=len(sanitized),
                    sanitized_rows=sanitized,
                )
            except sqlite3.OperationalError as db_exc:
                # استرجاع مخطط الجداول الفعلي تلقائياً لمساعدة النموذج على إصلاح خطئه
                schema_hint = self._db.introspect_schema_ddl()
                err_msg = (
                    f"[Attempt #{attempt} SQLite OperationalError]: {db_exc} "
                    f"-> Injected Actual Schema: {schema_hint.splitlines()[0]}..."
                )
                healed_errors.append(err_msg)

        raise RuntimeError(f"فشل إصلاح استعلام SQL بعد {self._max_retries} محاولات.")


async def run_text_to_sql_application() -> None:
    db = EnterpriseAnalyticsDatabase()
    harness = SelfHealingTextToSQLHarness(db=db, max_healing_retries=3)

    acme_ctx = TenantContext(
        tenant_id="ACME_CORP",
        analyst_id="bi_analyst_nadia",
        allow_raw_pii=False,
    )

    print("=" * 82)
    print("📊 التطبيق الثاني: وكيل Text-to-SQL ذاتي الإصلاح مع عزل المستأجرين وحجب PII")
    print("=" * 82)

    report = await harness.run_analytics_query(
        ctx=acme_ctx,
        natural_language_question="اعرض جميع الفواتير المدفوعة لشركتنا مرتبة حسب الإيراد الصافي.",
    )

    print(f"• المستأجر المعزول (Tenant) : {report.tenant_id}")
    print(f"• عدد محاولات التصحيح الذاتي: {report.attempts_taken}")
    print("• سجل الأخطاء التي أصلحها الـ Harness تلقائياً:")
    for err in report.self_healed_errors:
        print(f"   🛠️ {err}")

    print(f"\n• الاستعلام النهائي الآمن   : {report.executed_sql}")
    print(f"• النتائج المسترجعة ({report.row_count} صفوف — لاحظ عزل GLOBEX_INC وحجب البريد الإلكتروني):")
    for row in report.sanitized_rows:
        print(f"   -> {row}")


if __name__ == "__main__":
    asyncio.run(run_text_to_sql_application())

# ────────────────────────────────────────────────────────────────────────
# [003] dist/md/app-projects.md
# ────────────────────────────────────────────────────────────────────────
import asyncio
import hashlib
import hmac
import re
import uuid
from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field


# =====================================================================
# 1. نماذج البيانات السريرية ومطهر الخصوصية (HIPAA PHI Scrubber & Models)
# =====================================================================

class TriageAcuityLevel(StrEnum):
    ESI_1_RESUSCITATION = "ESI_1_RESUSCITATION"
    ESI_2_EMERGENT = "ESI_2_EMERGENT"
    ESI_3_URGENT = "ESI_3_URGENT"
    ESI_4_NON_URGENT = "ESI_4_NON_URGENT"


class RawPatientIntake(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    raw_clinical_note: str
    age: int = Field(ge=0, le=125)
    egfr_ml_min: float = Field(
        gt=0.0, description="معدل الترشيح الكبيبي لوظائف الكلى (الطبيعي > 60)"
    )
    current_medications: list[str]
    proposed_new_medication: str


class DeidentifiedClinicalCase(BaseModel):
    """سجل المريض بعد تجريده من كافة بيانات الهوية الشخصية (HIPAA Safe)."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    anonymous_case_id: str
    sanitized_clinical_note: str
    age: int
    egfr_ml_min: float
    renal_impairment_severe: bool
    current_medications: list[str]
    proposed_new_medication: str


class ClinicalSpecialistBallot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    specialist_role: str
    approve_proposed_medication: bool
    has_clinical_veto: bool = False
    contraindication_alerts: list[str] = Field(default_factory=list)
    recommended_safe_alternative: str | None = None
    clinical_rationale: str


class SignedPhysicianOrderTicket(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ticket_id: str
    anonymous_case_id: str
    acuity_level: TriageAcuityLevel
    proposed_medication_blocked: bool
    vetoed_by_specialists: list[str]
    final_recommended_regimen: str
    contraindication_warnings: list[str]
    requires_attending_physician_signoff: bool = True
    hmac_integrity_seal: str


class HIPAAPHIScrubber:
    """يجرد النصوص السريرية من الأسماء، وأرقام الهوية، وأرقام الهواتف."""

    _PATTERNS: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"المريض\s+[أ-ي]+\s+[أ-ي]+"), "المريض [PHI_NAME_REDACTED]"),
        (re.compile(r"\b\d{10}\b"), "[PHI_NATIONAL_ID_REDACTED]"),
        (re.compile(r"\+?\d[\d -]{8,14}\d"), "[PHI_PHONE_REDACTED]"),
    ]

    @classmethod
    def deidentify(cls, raw: RawPatientIntake) -> DeidentifiedClinicalCase:
        clean_note = raw.raw_clinical_note
        for pattern, repl in cls._PATTERNS:
            clean_note = pattern.sub(repl, clean_note)

        return DeidentifiedClinicalCase(
            anonymous_case_id=f"CLIN-{uuid.uuid4().hex[:8].upper()}",
            sanitized_clinical_note=clean_note,
            age=raw.age,
            egfr_ml_min=raw.egfr_ml_min,
            renal_impairment_severe=(raw.egfr_ml_min < 30.0),
            current_medications=[m.lower() for m in raw.current_medications],
            proposed_new_medication=raw.proposed_new_medication.lower(),
        )


# =====================================================================
# 2. محرك اليقظة الدوائية والوكلاء المتخصصين (Pharmacovigilance Swarm)
# =====================================================================

class ClinicalPharmacovigilanceHarness:
    """
    يدير سرب التقييم السريري المتوازي ويطبق حق النقض الدوائي المطلق (Absolute Veto)
    عند رصد أي تداخل دوائي قاتل أو موانع استعمال كلوية.
    """

    # قاعدة معرفة حتمية للتداخلات الدوائية الحرجة وموانع الاستعمال الكلوية
    _NSAID_DRUGS: frozenset[str] = frozenset({"ibuprofen", "naproxen", "ketorolac", "diclofenac"})
    _ANTICOAGULANTS: frozenset[str] = frozenset({"warfarin", "apixaban", "rivaroxaban"})

    def __init__(self, hmac_secret: str) -> None:
        self._secret = hmac_secret.encode("utf-8")

    async def _run_pharmacovigilance_specialist(
        self, case: DeidentifiedClinicalCase
    ) -> ClinicalSpecialistBallot:
        await asyncio.sleep(0.02)
        alerts: list[str] = []
        veto = False
        safe_alt: str | None = None

        proposed = case.proposed_new_medication
        current_set = set(case.current_medications)

        # فحص 1: مضادات الالتهاب غير الستيرويدية (NSAIDs) مع القصور الكلوي الحاد
        if proposed in self._NSAID_DRUGS and case.renal_impairment_severe:
            veto = True
            alerts.append(
                f"مانع استعمال كلوي قاطع (Renal Contraindication): الدواء '{proposed}' (NSAID) "
                f"يسبب فشلاً كلوياً حاداً لأن eGFR={case.egfr_ml_min} (< 30 mL/min)."
            )

        # فحص 2: التداخل الدوائي النزفي بين NSAIDs ومميعات الدم (Anticoagulants)
        active_anticoagulants = current_set.intersection(self._ANTICOAGULANTS)
        if proposed in self._NSAID_DRUGS and active_anticoagulants:
            veto = True
            alerts.append(
                f"تداخل دوائي خطير (Major Drug-Drug Interaction): الجمع بين '{proposed}' "
                f"ومميع الدم {list(active_anticoagulants)} يضاعف خطر النزيف الهضمي الحاد."
            )

        if veto:
            safe_alt = "acetaminophen (paracetamol) 500mg PO q8h PRN (Max 2g/day)"

        return ClinicalSpecialistBallot(
            specialist_role="clinical_pharmacologist_agent",
            approve_proposed_medication=(not veto),
            has_clinical_veto=veto,
            contraindication_alerts=alerts,
            recommended_safe_alternative=safe_alt,
            clinical_rationale=(
                "تم إجهاض الوصفة المقترحة بسبب موانع استعمال كلوية وتداخلات نزفية."
                if veto
                else "لا توجد تداخلات دوائية حرجة."
            ),
        )

    async def _run_internal_medicine_specialist(
        self, case: DeidentifiedClinicalCase
    ) -> ClinicalSpecialistBallot:
        await asyncio.sleep(0.025)
        return ClinicalSpecialistBallot(
            specialist_role="internal_medicine_triage_agent",
            approve_proposed_medication=True,
            has_clinical_veto=False,
            clinical_rationale=(
                "المريض يعاني من ألم مفصلي حاد ويحتاج لتسكين الألم مع مراقبة ضغط الدم."
            ),
        )

    async def _run_nephrology_guideline_specialist(
        self, case: DeidentifiedClinicalCase
    ) -> ClinicalSpecialistBallot:
        await asyncio.sleep(0.02)
        veto = case.renal_impairment_severe and (
            case.proposed_new_medication in self._NSAID_DRUGS
        )
        alerts = (
            ["KDIGO Guideline Violation: Avoid all NSAIDs in Stage 4/5 CKD (eGFR < 30)."]
            if veto
            else []
        )
        return ClinicalSpecialistBallot(
            specialist_role="nephrology_guideline_auditor",
            approve_proposed_medication=(not veto),
            has_clinical_veto=veto,
            contraindication_alerts=alerts,
            recommended_safe_alternative="acetaminophen (paracetamol) renal-adjusted dose",
            clinical_rationale="فحص التوافق مع البروتوكولات الإرشادية لأمراض الكلى (KDIGO).",
        )

    async def evaluate_clinical_case(
        self, raw_intake: RawPatientIntake
    ) -> tuple[DeidentifiedClinicalCase, SignedPhysicianOrderTicket]:
        # 1. تطهير بيانات الهوية الشخصية (HIPAA De-identification)
        clean_case = HIPAAPHIScrubber.deidentify(raw_intake)

        # 2. تشغيل سرب الأخصائيين بالتوازي المهيكل
        async with asyncio.TaskGroup() as tg:
            t_pharm = tg.create_task(
                self._run_pharmacovigilance_specialist(clean_case)
            )
            t_med = tg.create_task(
                self._run_internal_medicine_specialist(clean_case)
            )
            t_neph = tg.create_task(
                self._run_nephrology_guideline_specialist(clean_case)
            )

        ballots = [t_pharm.result(), t_med.result(), t_neph.result()]

        # 3. تطبيق بوابة الفيتو الطبي المطلق (Clinical Veto Gate)
        veto_roles = [b.specialist_role for b in ballots if b.has_clinical_veto]
        all_warnings = [
            alert for b in ballots for alert in b.contraindication_alerts
        ]

        if veto_roles:
            blocked = True
            alt = next(
                (
                    b.recommended_safe_alternative
                    for b in ballots
                    if b.recommended_safe_alternative
                ),
                "Consult Clinical Pharmacist Immediately",
            )
            final_regimen = (
                f"⛔ [BLOCKED: {clean_case.proposed_new_medication}] -> "
                f"✅ البديل الآمن المقترح: {alt}"
            )
            acuity = TriageAcuityLevel.ESI_2_EMERGENT
        else:
            blocked = False
            final_regimen = f"Approved: {clean_case.proposed_new_medication}"
            acuity = TriageAcuityLevel.ESI_3_URGENT

        ticket_id = f"PITL-{uuid.uuid4().hex[:8].upper()}"
        unsigned_payload = (
            f"{ticket_id}:{clean_case.anonymous_case_id}:{blocked}:{final_regimen}"
        )
        seal = hmac.new(
            self._secret, unsigned_payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        ticket = SignedPhysicianOrderTicket(
            ticket_id=ticket_id,
            anonymous_case_id=clean_case.anonymous_case_id,
            acuity_level=acuity,
            proposed_medication_blocked=blocked,
            vetoed_by_specialists=veto_roles,
            final_recommended_regimen=final_regimen,
            contraindication_warnings=all_warnings,
            hmac_integrity_seal=seal,
        )
        return clean_case, ticket


async def run_clinical_application() -> None:
    harness = ClinicalPharmacovigilanceHarness(
        hmac_secret="hospital-pitl-signing-key-2026"
    )

    raw_intake = RawPatientIntake(
        raw_clinical_note=(
            "حضر المريض خالد العتيبي (هوية رقم 1098765432، جوال +966501234567) "
            "يشكو من ألم حاد في الركبة اليمنى منذ يومين ويطلب مسكناً قوياً."
        ),
        age=68,
        egfr_ml_min=24.0,  # قصور كلوي حاد (أقل من 30)
        current_medications=["Warfarin", "Amlodipine"],  # يتناول مميع دم!
        proposed_new_medication="Ibuprofen",  # مسكن NSAID خطير جداً في هذه الحالة!
    )

    print("=" * 82)
    print("🏥 التطبيق الثالث: منظومة التيقظ الدوائي ودعم القرار السريري (HIPAA + Clinical Veto)")
    print("=" * 82)

    clean_case, ticket = await harness.evaluate_clinical_case(raw_intake)

    print(f"• معرف الحالة المعزول       : {clean_case.anonymous_case_id}")
    print(f"• الملاحظة بعد تطهير HIPAA  : {clean_case.sanitized_clinical_note}")
    print(f"• مستوى الفرز السريري       : {ticket.acuity_level}")
    print(f"• هل حُجب الدواء المقترح؟   : {ticket.proposed_medication_blocked}")
    print(f"• الأخصائيون أصحاب الفيتو   : {ticket.vetoed_by_specialists}")
    print("• تحذيرات التداخل الدوائي وموانع الاستعمال:")
    for warn in ticket.contraindication_warnings:
        print(f"   🚨 {warn}")
    print(f"\n• التوصية العلاجية للطبيب   : {ticket.final_recommended_regimen}")
    print(f"• ختم النزاهة الرقمي (HMAC) : {ticket.hmac_integrity_seal[:32]}...")


if __name__ == "__main__":
    asyncio.run(run_clinical_application())

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/agent-harness-engineering/notebooks/app-projects.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import asyncio
import time
import uuid
from enum import StrEnum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


# =====================================================================
# 1. العقود المهيكلة لحوادث SRE وإجراءات الإصلاح (Immutable Schemas)
# =====================================================================

class SeverityLevel(StrEnum):
    SEV1_CRITICAL = "SEV1_CRITICAL"
    SEV2_MAJOR = "SEV2_MAJOR"
    SEV3_MINOR = "SEV3_MINOR"


class RemediationActionType(StrEnum):
    SCALE_DEPLOYMENT = "scale_deployment"
    ROLLBACK_RELEASE = "rollback_release"
    FLUSH_STALE_POOL = "flush_stale_connection_pool"
    DELETE_NAMESPACE = "delete_namespace"  # محظور أمنياً!


class IncidentAlert(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    incident_id: str
    service_name: str
    severity: SeverityLevel
    symptom_summary: str
    triggered_at_epoch: float = Field(default_factory=time.time)


class ClusterHealthSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    active_replicas: int = Field(ge=0)
    cpu_utilization_pct: float = Field(ge=0.0, le=100.0)
    db_pool_saturation_pct: float = Field(ge=0.0, le=100.0)
    p99_latency_ms: float = Field(ge=0.0)
    error_rate_pct: float = Field(ge=0.0, le=100.0)
    recent_log_signature: str


class ProposedRemediationPlan(BaseModel):
    """مخطط قرار وكيل SRE: يجبره على كتابة الفرضية الهندسية قبل اختيار الأداة."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    root_cause_hypothesis: str = Field(min_length=15)
    action_type: RemediationActionType
    target_service: str
    target_replicas: int | None = Field(default=None, ge=1, le=50)
    confidence_score: float = Field(ge=0.0, le=1.0)


class IncidentPostMortemReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    incident_id: str
    service_name: str
    resolved_autonomously: bool
    rollback_triggered: bool
    escalated_to_human_oncall: bool
    root_cause_hypothesis: str
    executed_actions: list[str]
    pre_remediation_health: ClusterHealthSnapshot
    post_remediation_health: ClusterHealthSnapshot
    total_mttr_ms: float = Field(ge=0.0, description="Mean Time To Recovery in ms")
    audit_timeline: list[str]


# =====================================================================
# 2. بوابة أمان العمليات السحابية (Blast-Radius Safety Gatekeeper)
# =====================================================================

class BlastRadiusViolationError(PermissionError):
    """يُرفع عندما يقترح الوكيل إجراءً يتجاوز نصف قطر الضرر المسموح به."""


class SREBlastRadiusGatekeeper:
    _FORBIDDEN_ACTIONS: frozenset[RemediationActionType] = frozenset({
        RemediationActionType.DELETE_NAMESPACE,
    })
    _MAX_ALLOWED_REPLICAS: int = 20

    @classmethod
    def validate_plan(
        cls,
        plan: ProposedRemediationPlan,
        current_health: ClusterHealthSnapshot,
    ) -> None:
        if plan.action_type in cls._FORBIDDEN_ACTIONS:
            raise BlastRadiusViolationError(
                f"إجراء تدميري محظور: '{plan.action_type}' ممنوع قطعياً في التشغيل التلقائي."
            )

        if plan.action_type == RemediationActionType.SCALE_DEPLOYMENT:
            if plan.target_replicas is None:
                raise BlastRadiusViolationError(
                    "يجب تحديد target_replicas عند طلب توسيع الخدمة."
                )
            # منع تقليص عدد الحاويات (Scale-Down) أثناء وجود حادثة نشطة!
            if plan.target_replicas < current_health.active_replicas:
                raise BlastRadiusViolationError(
                    f"مرفوض: لا يُسمح بتقليص النسخ من {current_health.active_replicas} "
                    f"إلى {plan.target_replicas} أثناء حادثة نشطة!"
                )
            if plan.target_replicas > cls._MAX_ALLOWED_REPLICAS:
                raise BlastRadiusViolationError(
                    f"تجاوز سقف السعة: طلب {plan.target_replicas} نسخة يتجاوز الحد الأقصى ({cls._MAX_ALLOWED_REPLICAS})."
                )


# =====================================================================
# 3. بيئة البنية التحتية المحاكاة وعقل الوكيل (Cluster Simulator & Engine)
# =====================================================================

class SimulatedKubernetesCluster:
    """بيئة عنقود سحابي قابلة للعكس تدعم اللقطات والتراجع (Snapshots & Rollback)."""

    def __init__(self) -> None:
        self._state = ClusterHealthSnapshot(
            active_replicas=4,
            cpu_utilization_pct=94.5,
            db_pool_saturation_pct=99.2,
            p99_latency_ms=2850.0,
            error_rate_pct=14.8,
            recent_log_signature="FATAL: psycopg.PoolTimeout: connection pool exhausted (active=100/100)",
        )

    def get_snapshot(self) -> ClusterHealthSnapshot:
        return self._state

    def restore_snapshot(self, snapshot: ClusterHealthSnapshot) -> None:
        self._state = snapshot

    async def gather_parallel_diagnostics(self) -> ClusterHealthSnapshot:
        async def _fetch_prom_metrics() -> tuple[float, float, float]:
            await asyncio.sleep(0.02)
            return (
                self._state.cpu_utilization_pct,
                self._state.p99_latency_ms,
                self._state.error_rate_pct,
            )

        async def _fetch_loki_logs() -> str:
            await asyncio.sleep(0.025)
            return self._state.recent_log_signature

        async def _fetch_k8s_state() -> tuple[int, float]:
            await asyncio.sleep(0.015)
            return (self._state.active_replicas, self._state.db_pool_saturation_pct)

        async with asyncio.TaskGroup() as tg:
            t_prom = tg.create_task(_fetch_prom_metrics())
            t_logs = tg.create_task(_fetch_loki_logs())
            t_k8s = tg.create_task(_fetch_k8s_state())

        cpu, lat, err = t_prom.result()
        logs = t_logs.result()
        reps, pool = t_k8s.result()

        return ClusterHealthSnapshot(
            active_replicas=reps,
            cpu_utilization_pct=cpu,
            db_pool_saturation_pct=pool,
            p99_latency_ms=lat,
            error_rate_pct=err,
            recent_log_signature=logs,
        )

    async def apply_action(self, plan: ProposedRemediationPlan) -> None:
        await asyncio.sleep(0.03)
        match plan.action_type:
            case RemediationActionType.SCALE_DEPLOYMENT:
                # فخ هندسي واقعي: زيادة عدد الحاويات عندما يكون مجمع قاعدة البيانات ممتلئاً بنسبة 99%
                # يزيد الطين بلة ويرفع معدل الأخطاء لأن الحاويات الجديدة تطلب اتصالات إضافية!
                new_reps = plan.target_replicas or self._state.active_replicas
                self._state = ClusterHealthSnapshot(
                    active_replicas=new_reps,
                    cpu_utilization_pct=70.0,
                    db_pool_saturation_pct=100.0,
                    p99_latency_ms=4200.0,  # تدهور أسوأ!
                    error_rate_pct=28.5,    # ارتفعت الأخطاء!
                    recent_log_signature="FATAL: max_connections reached on primary DB!",
                )
            case RemediationActionType.FLUSH_STALE_POOL:
                # الحل الصحيح: تفريغ الاتصالات المعلقة يعيد النظام لحالته الطبيعية فوراً
                self._state = ClusterHealthSnapshot(
                    active_replicas=self._state.active_replicas,
                    cpu_utilization_pct=42.0,
                    db_pool_saturation_pct=35.0,
                    p99_latency_ms=115.0,
                    error_rate_pct=0.05,
                    recent_log_signature="INFO: DB connection pool healthy (active=35/100)",
                )
            case _:
                pass


class AutonomousSRERemediationHarness:
    """
    إطار التحكم الذاتي لحوادث SRE:
    يشخّص بالتوازي، ويتحقق من بوابة الأمان، ويطبق الإصلاح بنمط Canary،
    ويتراجع تلقائياً (Auto-Rollback) إذا تدهور مؤشر مستوى الخدمة (SLO)!
    """

    def __init__(self, cluster: SimulatedKubernetesCluster) -> None:
        self._cluster = cluster

    @staticmethod
    def _is_slo_healthy(health: ClusterHealthSnapshot) -> bool:
        return health.error_rate_pct < 1.0 and health.p99_latency_ms < 400.0

    async def _propose_plan(
        self,
        alert: IncidentAlert,
        health: ClusterHealthSnapshot,
        attempt: int,
        previous_failure_note: str,
    ) -> ProposedRemediationPlan:
        await asyncio.sleep(0.01)
        if attempt == 1:
            # في المحاولة الأولى يقترح الوكيل توسيع عدد الحاويات (Scale Up) ظناً منه أنها مشكلة CPU
            return ProposedRemediationPlan(
                root_cause_hypothesis=(
                    "استهلاك المعالج 94.5%، سأقوم بمضاعفة عدد الحاويات من 4 إلى 8 لتخفيف الضغط."
                ),
                action_type=RemediationActionType.SCALE_DEPLOYMENT,
                target_service=alert.service_name,
                target_replicas=8,
                confidence_score=0.75,
            )

        # في المحاولة الثانية (بعد التراجع الآلي وقراءة ملاحظة فشل Canary):
        # يدرك الوكيل أن المشكلة هي تشبع مجمع اتصالات قاعدة البيانات وليس المعالج!
        return ProposedRemediationPlan(
            root_cause_hypothesis=(
                f"توسيع الحاويات أدى لتدهور الكمون لأن المشكلة الجذرية هي تشبع DB Pool "
                f"({health.db_pool_saturation_pct}%). يجب تفريغ الاتصالات الخاملة المعلقة."
            ),
            action_type=RemediationActionType.FLUSH_STALE_POOL,
            target_service=alert.service_name,
            confidence_score=0.96,
        )

    async def handle_incident(
        self, alert: IncidentAlert, max_attempts: int = 2
    ) -> IncidentPostMortemReport:
        t0 = time.perf_counter()
        timeline: list[str] = [
            f"[{alert.incident_id}] استلام إنذار {alert.severity} للخدمة '{alert.service_name}': {alert.symptom_summary}"
        ]

        # 1. التشخيص المتوازي عبر TaskGroup
        pre_health = await self._cluster.gather_parallel_diagnostics()
        timeline.append(
            f"اكتمل التشخيص المتوازي: P99={pre_health.p99_latency_ms}ms, "
            f"ErrorRate={pre_health.error_rate_pct}%, DB_Pool={pre_health.db_pool_saturation_pct}%"
        )

        executed_actions: list[str] = []
        rollback_triggered = False
        failure_feedback = ""
        last_hypothesis = ""

        for attempt in range(1, max_attempts + 1):
            current_snapshot = self._cluster.get_snapshot()
            plan = await self._propose_plan(
                alert, current_snapshot, attempt, failure_feedback
            )
            last_hypothesis = plan.root_cause_hypothesis
            timeline.append(
                f"[محاولة #{attempt}] الفرضية: {plan.root_cause_hypothesis} -> الإجراء: {plan.action_type}"
            )

            # 2. فحص بوابة نصف قطر الضرر (Blast-Radius Gate)
            try:
                SREBlastRadiusGatekeeper.validate_plan(plan, current_snapshot)
            except BlastRadiusViolationError as exc:
                timeline.append(f"🛑 حُجب الإجراء بواسطة بوابة الأمان: {exc}")
                failure_feedback = str(exc)
                continue

            # 3. تطبيق الإجراء مع الاحتفاظ بلقطة تراجع فورية (Canary Execution)
            await self._cluster.apply_action(plan)
            executed_actions.append(f"{plan.action_type.value} (attempt #{attempt})")

            # 4. فحص مؤشر مستوى الخدمة (SLO Health Verification)
            post_action_health = await self._cluster.gather_parallel_diagnostics()
            if self._is_slo_healthy(post_action_health):
                timeline.append(
                    f"✅ تعافى مؤشر الخدمة (SLO Recovered)! هبط الكمون إلى "
                    f"{post_action_health.p99_latency_ms}ms ومعدل الخطأ إلى {post_action_health.error_rate_pct}%."
                )
                mttr = round((time.perf_counter() - t0) * 1000, 2)
                return IncidentPostMortemReport(
                    incident_id=alert.incident_id,
                    service_name=alert.service_name,
                    resolved_autonomously=True,
                    rollback_triggered=rollback_triggered,
                    escalated_to_human_oncall=False,
                    root_cause_hypothesis=last_hypothesis,
                    executed_actions=executed_actions,
                    pre_remediation_health=pre_health,
                    post_remediation_health=post_action_health,
                    total_mttr_ms=mttr,
                    audit_timeline=timeline,
                )

            # 5. فشل مؤشر الخدمة بعد التعديل -> تراجع ذري فوري (Automatic Canary Rollback)!
            rollback_triggered = True
            self._cluster.restore_snapshot(current_snapshot)
            failure_feedback = (
                f"فشل Canary: بعد تنفيذ {plan.action_type} ارتفع الكمون إلى "
                f"{post_action_health.p99_latency_ms}ms ومعدل الأخطاء إلى {post_action_health.error_rate_pct}%."
            )
            timeline.append(
                f"⚠️ {failure_feedback} -> تم تنفيذ التراجع التلقائي (Auto-Rollback) لاستعادة اللقطة الآمنة!"
            )

        # إذا استُنفدت المحاولات، يتم التصعيد للمهندس المناوب
        final_health = self._cluster.get_snapshot()
        timeline.append("🚨 استُنفدت المحاولات الآمنة. تم تصعيد الحادثة للمهندس المناوب (PagerDuty).")
        mttr = round((time.perf_counter() - t0) * 1000, 2)
        return IncidentPostMortemReport(
            incident_id=alert.incident_id,
            service_name=alert.service_name,
            resolved_autonomously=False,
            rollback_triggered=rollback_triggered,
            escalated_to_human_oncall=True,
            root_cause_hypothesis=last_hypothesis,
            executed_actions=executed_actions,
            pre_remediation_health=pre_health,
            post_remediation_health=final_health,
            total_mttr_ms=mttr,
            audit_timeline=timeline,
        )


async def run_sre_application() -> None:
    cluster = SimulatedKubernetesCluster()
    harness = AutonomousSRERemediationHarness(cluster)

    alert = IncidentAlert(
        incident_id=f"INC-{uuid.uuid4().hex[:6].upper()}",
        service_name="checkout-payment-api",
        severity=SeverityLevel.SEV1_CRITICAL,
        symptom_summary="P99 latency > 2500ms and HTTP 503 spike on checkout-payment-api",
    )

    print("=" * 82)
    print("🚀 التطبيق الأول: وكيل SRE للاستجابة الذاتية للحوادث مع التراجع الآلي (Canary Rollback)")
    print("=" * 82)
    report = await harness.handle_incident(alert)

    for idx, entry in enumerate(report.audit_timeline, start=1):
        print(f"  {idx}. {entry}")

    print("-" * 82)
    print(
        f"📊 النتيجة النهائية: Resolved={report.resolved_autonomously} | "
        f"AutoRollbackUsed={report.rollback_triggered} | MTTR={report.total_mttr_ms}ms"
    )
    print(
        f"   • قبل الإصلاح : P99={report.pre_remediation_health.p99_latency_ms}ms, "
        f"Errors={report.pre_remediation_health.error_rate_pct}%"
    )
    print(
        f"   • بعد الإصلاح : P99={report.post_remediation_health.p99_latency_ms}ms, "
        f"Errors={report.post_remediation_health.error_rate_pct}%"
    )


if __name__ == "__main__":
    asyncio.run(run_sre_application())

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/agent-harness-engineering/notebooks/app-projects.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import asyncio
import re
import sqlite3
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator


# =====================================================================
# 1. سياق الأمان وجدار حماية استعلامات SQL (Zero-Trust SQL Guard)
# =====================================================================

class TenantContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=3)
    analyst_id: str
    allow_raw_pii: bool = False


class ValidatedParameterizedSQL(BaseModel):
    """
    عقد صارم لاستعلام SQL:
    1. يجب أن يبدأ بـ SELECT.
    2. يمنع أي أوامر تعديل أو تعليقات أو استعلامات متعددة (;).
    3. يجب أن يحتوي إلزامياً على المعامل الآمن :injected_tenant_id لعزل البيانات.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    reasoning: str = Field(min_length=10)
    sql_query: str = Field(min_length=15, max_length=2000)

    @field_validator("sql_query")
    @classmethod
    def enforce_zero_trust_sql_rules(cls, v: str) -> str:
        cleaned = v.strip().rstrip(";")
        if ";" in cleaned or "--" in cleaned or "/*" in cleaned:
            raise ValueError(
                "يُمنع استخدام الفاصلة المنقوطة المتعددة أو تعليقات SQL."
            )

        upper_sql = cleaned.upper()
        if not upper_sql.startswith("SELECT"):
            raise ValueError("يُسمح فقط باستعلامات القراءة التي تبدأ بـ SELECT.")

        forbidden = {
            "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
            "CREATE", "PRAGMA", "ATTACH", "DETACH", "REPLACE",
        }
        tokens = set(re.findall(r"\b[A-Z_]+\b", upper_sql))
        violations = tokens.intersection(forbidden)
        if violations:
            raise ValueError(f"كلمات مفتاحية محظورة أمنياً في الاستعلام: {violations}")

        if ":injected_tenant_id" not in cleaned:
            raise ValueError(
                "خرق عزل المستأجرين (Row-Level Security Violation): يجب أن يتضمن الاستعلام "
                "شرط 'tenant_id = :injected_tenant_id' صراحةً!"
            )

        if "LIMIT" not in tokens:
            cleaned = f"{cleaned} LIMIT 50"

        return cleaned


class AnalyticsExecutionReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str
    executed_sql: str
    attempts_taken: int
    self_healed_errors: list[str]
    row_count: int
    sanitized_rows: list[dict[str, Any]]


# =====================================================================
# 2. محرك قاعدة البيانات المتعددة المستأجرين وحلقة التصحيح الذاتي
# =====================================================================

class EnterpriseAnalyticsDatabase:
    """قاعدة بيانات SQLite حقيقية في الذاكرة تضم بيانات لعدة شركات متنافسة."""

    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._seed_multi_tenant_data()

    def _seed_multi_tenant_data(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE subscription_invoices (
                    invoice_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    region TEXT NOT NULL,
                    customer_contact_email TEXT NOT NULL,
                    net_revenue_usd REAL NOT NULL,
                    status TEXT NOT NULL
                )
                """
            )
            rows = [
                # بيانات المستأجر الأول: ACME_CORP
                ("INV-101", "ACME_CORP", "MENA", "cfo@client-a.com", 45_000.0, "PAID"),
                ("INV-102", "ACME_CORP", "MENA", "finance@client-b.com", 32_500.0, "PAID"),
                ("INV-103", "ACME_CORP", "EU", "billing@eu-partner.de", 18_000.0, "PAID"),
                # بيانات المستأجر المنافس: GLOBEX_INC (يجب ألا تتسرب أبداً لـ ACME_CORP!)
                ("INV-999", "GLOBEX_INC", "MENA", "secret@globex.io", 999_999.0, "PAID"),
            ]
            self._conn.executemany(
                "INSERT INTO subscription_invoices VALUES (?, ?, ?, ?, ?, ?)", rows
            )

    def introspect_schema_ddl(self) -> str:
        """يستخرج المخطط الحقيقي للجداول والأعمدة لتغذية حلقة التصحيح الذاتي."""
        cur = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        return "\n".join(row["sql"] for row in cur.fetchall() if row["sql"])

    def execute_tenant_scoped_query(
        self, sql: str, tenant_id: str
    ) -> list[dict[str, Any]]:
        cur = self._conn.execute(sql, {"injected_tenant_id": tenant_id})
        return [dict(row) for row in cur.fetchall()]


class SelfHealingTextToSQLHarness:
    """
    يدير توليد SQL، ويطبق جدار الحماية، ويعالج أخطاء قاعدة البيانات ذاتياً
    عبر حقن مخطط DDL الفعلي، ويحجب البيانات الشخصية (PII Masking).
    """

    def __init__(
        self, db: EnterpriseAnalyticsDatabase, max_healing_retries: int = 3
    ) -> None:
        self._db = db
        self._max_retries = max_healing_retries

    @staticmethod
    def _mask_pii_in_row(row: dict[str, Any], allow_raw_pii: bool) -> dict[str, Any]:
        if allow_raw_pii:
            return row
        masked: dict[str, Any] = {}
        for k, val in row.items():
            if isinstance(val, str) and "@" in val:
                user_part, _, domain = val.partition("@")
                masked[k] = f"{user_part[:2]}***@{domain}"
            else:
                masked[k] = val
        return masked

    async def _simulate_llm_sql_generator(
        self,
        question: str,
        attempt: int,
        error_history: list[str],
        schema_hint: str,
    ) -> dict[str, str]:
        await asyncio.sleep(0.01)
        if attempt == 1:
            # المحاولة 1: النموذج ينسى وضع شرط :injected_tenant_id (سيصطاده Pydantic Firewall!)
            return {
                "reasoning": "سأقوم بجمع الفواتير حسب المنطقة.",
                "sql_query": "SELECT region, SUM(total_amount) FROM invoices GROUP BY region",
            }
        if attempt == 2:
            # المحاولة 2: النموذج يلتزم بشرط المستأجر، لكنه يهلوس باسم الجدول والعمود (total_amount بدلاً من net_revenue_usd)
            # ستصطاده قاعدة بيانات SQLite بـ OperationalError!
            return {
                "reasoning": "أضفت شرط عزل المستأجر، وسأستعلم عن مجموع الإيرادات.",
                "sql_query": (
                    "SELECT region, customer_contact_email, total_amount "
                    "FROM subscription_invoices "
                    "WHERE tenant_id = :injected_tenant_id"
                ),
            }
        # المحاولة 3: بعد أن حقن الـ Harness مخطط DDL الفعلي من SQLite، يكتب النموذج الاستعلام الصحيح تماماً!
        return {
            "reasoning": (
                "بناءً على مخطط الجدول المحقون، العمود الصحيح هو net_revenue_usd وليس total_amount."
            ),
            "sql_query": (
                "SELECT invoice_id, region, customer_contact_email, net_revenue_usd "
                "FROM subscription_invoices "
                "WHERE tenant_id = :injected_tenant_id AND status = 'PAID' "
                "ORDER BY net_revenue_usd DESC"
            ),
        }

    async def run_analytics_query(
        self, ctx: TenantContext, natural_language_question: str
    ) -> AnalyticsExecutionReport:
        healed_errors: list[str] = []
        schema_hint = ""

        for attempt in range(1, self._max_retries + 1):
            raw_llm_output = await self._simulate_llm_sql_generator(
                question=natural_language_question,
                attempt=attempt,
                error_history=healed_errors,
                schema_hint=schema_hint,
            )

            # البوابة 1: فحص Pydantic و Zero-Trust SQL Firewall
            try:
                validated = ValidatedParameterizedSQL.model_validate(raw_llm_output)
            except Exception as val_exc:
                err_msg = f"[Attempt #{attempt} Firewall Reject]: {val_exc}"
                healed_errors.append(err_msg)
                continue

            # البوابة 2: التنفيذ الفعلي داخل SQLite مع ربط المعاملات (Parameter Binding)
            try:
                raw_rows = self._db.execute_tenant_scoped_query(
                    sql=validated.sql_query,
                    tenant_id=ctx.tenant_id,
                )
                sanitized = [
                    self._mask_pii_in_row(r, ctx.allow_raw_pii) for r in raw_rows
                ]
                return AnalyticsExecutionReport(
                    tenant_id=ctx.tenant_id,
                    executed_sql=validated.sql_query,
                    attempts_taken=attempt,
                    self_healed_errors=healed_errors,
                    row_count=len(sanitized),
                    sanitized_rows=sanitized,
                )
            except sqlite3.OperationalError as db_exc:
                # استرجاع مخطط الجداول الفعلي تلقائياً لمساعدة النموذج على إصلاح خطئه
                schema_hint = self._db.introspect_schema_ddl()
                err_msg = (
                    f"[Attempt #{attempt} SQLite OperationalError]: {db_exc} "
                    f"-> Injected Actual Schema: {schema_hint.splitlines()[0]}..."
                )
                healed_errors.append(err_msg)

        raise RuntimeError(f"فشل إصلاح استعلام SQL بعد {self._max_retries} محاولات.")


async def run_text_to_sql_application() -> None:
    db = EnterpriseAnalyticsDatabase()
    harness = SelfHealingTextToSQLHarness(db=db, max_healing_retries=3)

    acme_ctx = TenantContext(
        tenant_id="ACME_CORP",
        analyst_id="bi_analyst_nadia",
        allow_raw_pii=False,
    )

    print("=" * 82)
    print("📊 التطبيق الثاني: وكيل Text-to-SQL ذاتي الإصلاح مع عزل المستأجرين وحجب PII")
    print("=" * 82)

    report = await harness.run_analytics_query(
        ctx=acme_ctx,
        natural_language_question="اعرض جميع الفواتير المدفوعة لشركتنا مرتبة حسب الإيراد الصافي.",
    )

    print(f"• المستأجر المعزول (Tenant) : {report.tenant_id}")
    print(f"• عدد محاولات التصحيح الذاتي: {report.attempts_taken}")
    print("• سجل الأخطاء التي أصلحها الـ Harness تلقائياً:")
    for err in report.self_healed_errors:
        print(f"   🛠️ {err}")

    print(f"\n• الاستعلام النهائي الآمن   : {report.executed_sql}")
    print(f"• النتائج المسترجعة ({report.row_count} صفوف — لاحظ عزل GLOBEX_INC وحجب البريد الإلكتروني):")
    for row in report.sanitized_rows:
        print(f"   -> {row}")


if __name__ == "__main__":
    asyncio.run(run_text_to_sql_application())

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/agent-harness-engineering/notebooks/app-projects.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import asyncio
import hashlib
import hmac
import re
import uuid
from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field


# =====================================================================
# 1. نماذج البيانات السريرية ومطهر الخصوصية (HIPAA PHI Scrubber & Models)
# =====================================================================

class TriageAcuityLevel(StrEnum):
    ESI_1_RESUSCITATION = "ESI_1_RESUSCITATION"
    ESI_2_EMERGENT = "ESI_2_EMERGENT"
    ESI_3_URGENT = "ESI_3_URGENT"
    ESI_4_NON_URGENT = "ESI_4_NON_URGENT"


class RawPatientIntake(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    raw_clinical_note: str
    age: int = Field(ge=0, le=125)
    egfr_ml_min: float = Field(
        gt=0.0, description="معدل الترشيح الكبيبي لوظائف الكلى (الطبيعي > 60)"
    )
    current_medications: list[str]
    proposed_new_medication: str


class DeidentifiedClinicalCase(BaseModel):
    """سجل المريض بعد تجريده من كافة بيانات الهوية الشخصية (HIPAA Safe)."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    anonymous_case_id: str
    sanitized_clinical_note: str
    age: int
    egfr_ml_min: float
    renal_impairment_severe: bool
    current_medications: list[str]
    proposed_new_medication: str


class ClinicalSpecialistBallot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    specialist_role: str
    approve_proposed_medication: bool
    has_clinical_veto: bool = False
    contraindication_alerts: list[str] = Field(default_factory=list)
    recommended_safe_alternative: str | None = None
    clinical_rationale: str


class SignedPhysicianOrderTicket(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ticket_id: str
    anonymous_case_id: str
    acuity_level: TriageAcuityLevel
    proposed_medication_blocked: bool
    vetoed_by_specialists: list[str]
    final_recommended_regimen: str
    contraindication_warnings: list[str]
    requires_attending_physician_signoff: bool = True
    hmac_integrity_seal: str


class HIPAAPHIScrubber:
    """يجرد النصوص السريرية من الأسماء، وأرقام الهوية، وأرقام الهواتف."""

    _PATTERNS: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"المريض\s+[أ-ي]+\s+[أ-ي]+"), "المريض [PHI_NAME_REDACTED]"),
        (re.compile(r"\b\d{10}\b"), "[PHI_NATIONAL_ID_REDACTED]"),
        (re.compile(r"\+?\d[\d -]{8,14}\d"), "[PHI_PHONE_REDACTED]"),
    ]

    @classmethod
    def deidentify(cls, raw: RawPatientIntake) -> DeidentifiedClinicalCase:
        clean_note = raw.raw_clinical_note
        for pattern, repl in cls._PATTERNS:
            clean_note = pattern.sub(repl, clean_note)

        return DeidentifiedClinicalCase(
            anonymous_case_id=f"CLIN-{uuid.uuid4().hex[:8].upper()}",
            sanitized_clinical_note=clean_note,
            age=raw.age,
            egfr_ml_min=raw.egfr_ml_min,
            renal_impairment_severe=(raw.egfr_ml_min < 30.0),
            current_medications=[m.lower() for m in raw.current_medications],
            proposed_new_medication=raw.proposed_new_medication.lower(),
        )


# =====================================================================
# 2. محرك اليقظة الدوائية والوكلاء المتخصصين (Pharmacovigilance Swarm)
# =====================================================================

class ClinicalPharmacovigilanceHarness:
    """
    يدير سرب التقييم السريري المتوازي ويطبق حق النقض الدوائي المطلق (Absolute Veto)
    عند رصد أي تداخل دوائي قاتل أو موانع استعمال كلوية.
    """

    # قاعدة معرفة حتمية للتداخلات الدوائية الحرجة وموانع الاستعمال الكلوية
    _NSAID_DRUGS: frozenset[str] = frozenset({"ibuprofen", "naproxen", "ketorolac", "diclofenac"})
    _ANTICOAGULANTS: frozenset[str] = frozenset({"warfarin", "apixaban", "rivaroxaban"})

    def __init__(self, hmac_secret: str) -> None:
        self._secret = hmac_secret.encode("utf-8")

    async def _run_pharmacovigilance_specialist(
        self, case: DeidentifiedClinicalCase
    ) -> ClinicalSpecialistBallot:
        await asyncio.sleep(0.02)
        alerts: list[str] = []
        veto = False
        safe_alt: str | None = None

        proposed = case.proposed_new_medication
        current_set = set(case.current_medications)

        # فحص 1: مضادات الالتهاب غير الستيرويدية (NSAIDs) مع القصور الكلوي الحاد
        if proposed in self._NSAID_DRUGS and case.renal_impairment_severe:
            veto = True
            alerts.append(
                f"مانع استعمال كلوي قاطع (Renal Contraindication): الدواء '{proposed}' (NSAID) "
                f"يسبب فشلاً كلوياً حاداً لأن eGFR={case.egfr_ml_min} (< 30 mL/min)."
            )

        # فحص 2: التداخل الدوائي النزفي بين NSAIDs ومميعات الدم (Anticoagulants)
        active_anticoagulants = current_set.intersection(self._ANTICOAGULANTS)
        if proposed in self._NSAID_DRUGS and active_anticoagulants:
            veto = True
            alerts.append(
                f"تداخل دوائي خطير (Major Drug-Drug Interaction): الجمع بين '{proposed}' "
                f"ومميع الدم {list(active_anticoagulants)} يضاعف خطر النزيف الهضمي الحاد."
            )

        if veto:
            safe_alt = "acetaminophen (paracetamol) 500mg PO q8h PRN (Max 2g/day)"

        return ClinicalSpecialistBallot(
            specialist_role="clinical_pharmacologist_agent",
            approve_proposed_medication=(not veto),
            has_clinical_veto=veto,
            contraindication_alerts=alerts,
            recommended_safe_alternative=safe_alt,
            clinical_rationale=(
                "تم إجهاض الوصفة المقترحة بسبب موانع استعمال كلوية وتداخلات نزفية."
                if veto
                else "لا توجد تداخلات دوائية حرجة."
            ),
        )

    async def _run_internal_medicine_specialist(
        self, case: DeidentifiedClinicalCase
    ) -> ClinicalSpecialistBallot:
        await asyncio.sleep(0.025)
        return ClinicalSpecialistBallot(
            specialist_role="internal_medicine_triage_agent",
            approve_proposed_medication=True,
            has_clinical_veto=False,
            clinical_rationale=(
                "المريض يعاني من ألم مفصلي حاد ويحتاج لتسكين الألم مع مراقبة ضغط الدم."
            ),
        )

    async def _run_nephrology_guideline_specialist(
        self, case: DeidentifiedClinicalCase
    ) -> ClinicalSpecialistBallot:
        await asyncio.sleep(0.02)
        veto = case.renal_impairment_severe and (
            case.proposed_new_medication in self._NSAID_DRUGS
        )
        alerts = (
            ["KDIGO Guideline Violation: Avoid all NSAIDs in Stage 4/5 CKD (eGFR < 30)."]
            if veto
            else []
        )
        return ClinicalSpecialistBallot(
            specialist_role="nephrology_guideline_auditor",
            approve_proposed_medication=(not veto),
            has_clinical_veto=veto,
            contraindication_alerts=alerts,
            recommended_safe_alternative="acetaminophen (paracetamol) renal-adjusted dose",
            clinical_rationale="فحص التوافق مع البروتوكولات الإرشادية لأمراض الكلى (KDIGO).",
        )

    async def evaluate_clinical_case(
        self, raw_intake: RawPatientIntake
    ) -> tuple[DeidentifiedClinicalCase, SignedPhysicianOrderTicket]:
        # 1. تطهير بيانات الهوية الشخصية (HIPAA De-identification)
        clean_case = HIPAAPHIScrubber.deidentify(raw_intake)

        # 2. تشغيل سرب الأخصائيين بالتوازي المهيكل
        async with asyncio.TaskGroup() as tg:
            t_pharm = tg.create_task(
                self._run_pharmacovigilance_specialist(clean_case)
            )
            t_med = tg.create_task(
                self._run_internal_medicine_specialist(clean_case)
            )
            t_neph = tg.create_task(
                self._run_nephrology_guideline_specialist(clean_case)
            )

        ballots = [t_pharm.result(), t_med.result(), t_neph.result()]

        # 3. تطبيق بوابة الفيتو الطبي المطلق (Clinical Veto Gate)
        veto_roles = [b.specialist_role for b in ballots if b.has_clinical_veto]
        all_warnings = [
            alert for b in ballots for alert in b.contraindication_alerts
        ]

        if veto_roles:
            blocked = True
            alt = next(
                (
                    b.recommended_safe_alternative
                    for b in ballots
                    if b.recommended_safe_alternative
                ),
                "Consult Clinical Pharmacist Immediately",
            )
            final_regimen = (
                f"⛔ [BLOCKED: {clean_case.proposed_new_medication}] -> "
                f"✅ البديل الآمن المقترح: {alt}"
            )
            acuity = TriageAcuityLevel.ESI_2_EMERGENT
        else:
            blocked = False
            final_regimen = f"Approved: {clean_case.proposed_new_medication}"
            acuity = TriageAcuityLevel.ESI_3_URGENT

        ticket_id = f"PITL-{uuid.uuid4().hex[:8].upper()}"
        unsigned_payload = (
            f"{ticket_id}:{clean_case.anonymous_case_id}:{blocked}:{final_regimen}"
        )
        seal = hmac.new(
            self._secret, unsigned_payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        ticket = SignedPhysicianOrderTicket(
            ticket_id=ticket_id,
            anonymous_case_id=clean_case.anonymous_case_id,
            acuity_level=acuity,
            proposed_medication_blocked=blocked,
            vetoed_by_specialists=veto_roles,
            final_recommended_regimen=final_regimen,
            contraindication_warnings=all_warnings,
            hmac_integrity_seal=seal,
        )
        return clean_case, ticket


async def run_clinical_application() -> None:
    harness = ClinicalPharmacovigilanceHarness(
        hmac_secret="hospital-pitl-signing-key-2026"
    )

    raw_intake = RawPatientIntake(
        raw_clinical_note=(
            "حضر المريض خالد العتيبي (هوية رقم 1098765432، جوال +966501234567) "
            "يشكو من ألم حاد في الركبة اليمنى منذ يومين ويطلب مسكناً قوياً."
        ),
        age=68,
        egfr_ml_min=24.0,  # قصور كلوي حاد (أقل من 30)
        current_medications=["Warfarin", "Amlodipine"],  # يتناول مميع دم!
        proposed_new_medication="Ibuprofen",  # مسكن NSAID خطير جداً في هذه الحالة!
    )

    print("=" * 82)
    print("🏥 التطبيق الثالث: منظومة التيقظ الدوائي ودعم القرار السريري (HIPAA + Clinical Veto)")
    print("=" * 82)

    clean_case, ticket = await harness.evaluate_clinical_case(raw_intake)

    print(f"• معرف الحالة المعزول       : {clean_case.anonymous_case_id}")
    print(f"• الملاحظة بعد تطهير HIPAA  : {clean_case.sanitized_clinical_note}")
    print(f"• مستوى الفرز السريري       : {ticket.acuity_level}")
    print(f"• هل حُجب الدواء المقترح؟   : {ticket.proposed_medication_blocked}")
    print(f"• الأخصائيون أصحاب الفيتو   : {ticket.vetoed_by_specialists}")
    print("• تحذيرات التداخل الدوائي وموانع الاستعمال:")
    for warn in ticket.contraindication_warnings:
        print(f"   🚨 {warn}")
    print(f"\n• التوصية العلاجية للطبيب   : {ticket.final_recommended_regimen}")
    print(f"• ختم النزاهة الرقمي (HMAC) : {ticket.hmac_integrity_seal[:32]}...")


if __name__ == "__main__":
    asyncio.run(run_clinical_application())

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/agent-harness-engineering/notebooks/app-projects.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
هندسة وكلاء الذكاء الاصطناعي — التطبيقات الثلاثة المرجعية الكاملة
AI Agent Engineering — Three Reference Production Applications

Book slug : agent-harness-engineering
Chapter   : app-projects (lesson projects)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] dist/md/app-projects.md
# ────────────────────────────────────────────────────────────────────────
import asyncio
import time
import uuid
from enum import StrEnum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


# =====================================================================
# 1. العقود المهيكلة لحوادث SRE وإجراءات الإصلاح (Immutable Schemas)
# =====================================================================

class SeverityLevel(StrEnum):
    SEV1_CRITICAL = "SEV1_CRITICAL"
    SEV2_MAJOR = "SEV2_MAJOR"
    SEV3_MINOR = "SEV3_MINOR"


class RemediationActionType(StrEnum):
    SCALE_DEPLOYMENT = "scale_deployment"
    ROLLBACK_RELEASE = "rollback_release"
    FLUSH_STALE_POOL = "flush_stale_connection_pool"
    DELETE_NAMESPACE = "delete_namespace"  # محظور أمنياً!


class IncidentAlert(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    incident_id: str
    service_name: str
    severity: SeverityLevel
    symptom_summary: str
    triggered_at_epoch: float = Field(default_factory=time.time)


class ClusterHealthSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    active_replicas: int = Field(ge=0)
    cpu_utilization_pct: float = Field(ge=0.0, le=100.0)
    db_pool_saturation_pct: float = Field(ge=0.0, le=100.0)
    p99_latency_ms: float = Field(ge=0.0)
    error_rate_pct: float = Field(ge=0.0, le=100.0)
    recent_log_signature: str


class ProposedRemediationPlan(BaseModel):
    """مخطط قرار وكيل SRE: يجبره على كتابة الفرضية الهندسية قبل اختيار الأداة."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    root_cause_hypothesis: str = Field(min_length=15)
    action_type: RemediationActionType
    target_service: str
    target_replicas: int | None = Field(default=None, ge=1, le=50)
    confidence_score: float = Field(ge=0.0, le=1.0)


class IncidentPostMortemReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    incident_id: str
    service_name: str
    resolved_autonomously: bool
    rollback_triggered: bool
    escalated_to_human_oncall: bool
    root_cause_hypothesis: str
    executed_actions: list[str]
    pre_remediation_health: ClusterHealthSnapshot
    post_remediation_health: ClusterHealthSnapshot
    total_mttr_ms: float = Field(ge=0.0, description="Mean Time To Recovery in ms")
    audit_timeline: list[str]


# =====================================================================
# 2. بوابة أمان العمليات السحابية (Blast-Radius Safety Gatekeeper)
# =====================================================================

class BlastRadiusViolationError(PermissionError):
    """يُرفع عندما يقترح الوكيل إجراءً يتجاوز نصف قطر الضرر المسموح به."""


class SREBlastRadiusGatekeeper:
    _FORBIDDEN_ACTIONS: frozenset[RemediationActionType] = frozenset({
        RemediationActionType.DELETE_NAMESPACE,
    })
    _MAX_ALLOWED_REPLICAS: int = 20

    @classmethod
    def validate_plan(
        cls,
        plan: ProposedRemediationPlan,
        current_health: ClusterHealthSnapshot,
    ) -> None:
        if plan.action_type in cls._FORBIDDEN_ACTIONS:
            raise BlastRadiusViolationError(
                f"إجراء تدميري محظور: '{plan.action_type}' ممنوع قطعياً في التشغيل التلقائي."
            )

        if plan.action_type == RemediationActionType.SCALE_DEPLOYMENT:
            if plan.target_replicas is None:
                raise BlastRadiusViolationError(
                    "يجب تحديد target_replicas عند طلب توسيع الخدمة."
                )
            # منع تقليص عدد الحاويات (Scale-Down) أثناء وجود حادثة نشطة!
            if plan.target_replicas < current_health.active_replicas:
                raise BlastRadiusViolationError(
                    f"مرفوض: لا يُسمح بتقليص النسخ من {current_health.active_replicas} "
                    f"إلى {plan.target_replicas} أثناء حادثة نشطة!"
                )
            if plan.target_replicas > cls._MAX_ALLOWED_REPLICAS:
                raise BlastRadiusViolationError(
                    f"تجاوز سقف السعة: طلب {plan.target_replicas} نسخة يتجاوز الحد الأقصى ({cls._MAX_ALLOWED_REPLICAS})."
                )


# =====================================================================
# 3. بيئة البنية التحتية المحاكاة وعقل الوكيل (Cluster Simulator & Engine)
# =====================================================================

class SimulatedKubernetesCluster:
    """بيئة عنقود سحابي قابلة للعكس تدعم اللقطات والتراجع (Snapshots & Rollback)."""

    def __init__(self) -> None:
        self._state = ClusterHealthSnapshot(
            active_replicas=4,
            cpu_utilization_pct=94.5,
            db_pool_saturation_pct=99.2,
            p99_latency_ms=2850.0,
            error_rate_pct=14.8,
            recent_log_signature="FATAL: psycopg.PoolTimeout: connection pool exhausted (active=100/100)",
        )

    def get_snapshot(self) -> ClusterHealthSnapshot:
        return self._state

    def restore_snapshot(self, snapshot: ClusterHealthSnapshot) -> None:
        self._state = snapshot

    async def gather_parallel_diagnostics(self) -> ClusterHealthSnapshot:
        async def _fetch_prom_metrics() -> tuple[float, float, float]:
            await asyncio.sleep(0.02)
            return (
                self._state.cpu_utilization_pct,
                self._state.p99_latency_ms,
                self._state.error_rate_pct,
            )

        async def _fetch_loki_logs() -> str:
            await asyncio.sleep(0.025)
            return self._state.recent_log_signature

        async def _fetch_k8s_state() -> tuple[int, float]:
            await asyncio.sleep(0.015)
            return (self._state.active_replicas, self._state.db_pool_saturation_pct)

        async with asyncio.TaskGroup() as tg:
            t_prom = tg.create_task(_fetch_prom_metrics())
            t_logs = tg.create_task(_fetch_loki_logs())
            t_k8s = tg.create_task(_fetch_k8s_state())

        cpu, lat, err = t_prom.result()
        logs = t_logs.result()
        reps, pool = t_k8s.result()

        return ClusterHealthSnapshot(
            active_replicas=reps,
            cpu_utilization_pct=cpu,
            db_pool_saturation_pct=pool,
            p99_latency_ms=lat,
            error_rate_pct=err,
            recent_log_signature=logs,
        )

    async def apply_action(self, plan: ProposedRemediationPlan) -> None:
        await asyncio.sleep(0.03)
        match plan.action_type:
            case RemediationActionType.SCALE_DEPLOYMENT:
                # فخ هندسي واقعي: زيادة عدد الحاويات عندما يكون مجمع قاعدة البيانات ممتلئاً بنسبة 99%
                # يزيد الطين بلة ويرفع معدل الأخطاء لأن الحاويات الجديدة تطلب اتصالات إضافية!
                new_reps = plan.target_replicas or self._state.active_replicas
                self._state = ClusterHealthSnapshot(
                    active_replicas=new_reps,
                    cpu_utilization_pct=70.0,
                    db_pool_saturation_pct=100.0,
                    p99_latency_ms=4200.0,  # تدهور أسوأ!
                    error_rate_pct=28.5,    # ارتفعت الأخطاء!
                    recent_log_signature="FATAL: max_connections reached on primary DB!",
                )
            case RemediationActionType.FLUSH_STALE_POOL:
                # الحل الصحيح: تفريغ الاتصالات المعلقة يعيد النظام لحالته الطبيعية فوراً
                self._state = ClusterHealthSnapshot(
                    active_replicas=self._state.active_replicas,
                    cpu_utilization_pct=42.0,
                    db_pool_saturation_pct=35.0,
                    p99_latency_ms=115.0,
                    error_rate_pct=0.05,
                    recent_log_signature="INFO: DB connection pool healthy (active=35/100)",
                )
            case _:
                pass


class AutonomousSRERemediationHarness:
    """
    إطار التحكم الذاتي لحوادث SRE:
    يشخّص بالتوازي، ويتحقق من بوابة الأمان، ويطبق الإصلاح بنمط Canary،
    ويتراجع تلقائياً (Auto-Rollback) إذا تدهور مؤشر مستوى الخدمة (SLO)!
    """

    def __init__(self, cluster: SimulatedKubernetesCluster) -> None:
        self._cluster = cluster

    @staticmethod
    def _is_slo_healthy(health: ClusterHealthSnapshot) -> bool:
        return health.error_rate_pct < 1.0 and health.p99_latency_ms < 400.0

    async def _propose_plan(
        self,
        alert: IncidentAlert,
        health: ClusterHealthSnapshot,
        attempt: int,
        previous_failure_note: str,
    ) -> ProposedRemediationPlan:
        await asyncio.sleep(0.01)
        if attempt == 1:
            # في المحاولة الأولى يقترح الوكيل توسيع عدد الحاويات (Scale Up) ظناً منه أنها مشكلة CPU
            return ProposedRemediationPlan(
                root_cause_hypothesis=(
                    "استهلاك المعالج 94.5%، سأقوم بمضاعفة عدد الحاويات من 4 إلى 8 لتخفيف الضغط."
                ),
                action_type=RemediationActionType.SCALE_DEPLOYMENT,
                target_service=alert.service_name,
                target_replicas=8,
                confidence_score=0.75,
            )

        # في المحاولة الثانية (بعد التراجع الآلي وقراءة ملاحظة فشل Canary):
        # يدرك الوكيل أن المشكلة هي تشبع مجمع اتصالات قاعدة البيانات وليس المعالج!
        return ProposedRemediationPlan(
            root_cause_hypothesis=(
                f"توسيع الحاويات أدى لتدهور الكمون لأن المشكلة الجذرية هي تشبع DB Pool "
                f"({health.db_pool_saturation_pct}%). يجب تفريغ الاتصالات الخاملة المعلقة."
            ),
            action_type=RemediationActionType.FLUSH_STALE_POOL,
            target_service=alert.service_name,
            confidence_score=0.96,
        )

    async def handle_incident(
        self, alert: IncidentAlert, max_attempts: int = 2
    ) -> IncidentPostMortemReport:
        t0 = time.perf_counter()
        timeline: list[str] = [
            f"[{alert.incident_id}] استلام إنذار {alert.severity} للخدمة '{alert.service_name}': {alert.symptom_summary}"
        ]

        # 1. التشخيص المتوازي عبر TaskGroup
        pre_health = await self._cluster.gather_parallel_diagnostics()
        timeline.append(
            f"اكتمل التشخيص المتوازي: P99={pre_health.p99_latency_ms}ms, "
            f"ErrorRate={pre_health.error_rate_pct}%, DB_Pool={pre_health.db_pool_saturation_pct}%"
        )

        executed_actions: list[str] = []
        rollback_triggered = False
        failure_feedback = ""
        last_hypothesis = ""

        for attempt in range(1, max_attempts + 1):
            current_snapshot = self._cluster.get_snapshot()
            plan = await self._propose_plan(
                alert, current_snapshot, attempt, failure_feedback
            )
            last_hypothesis = plan.root_cause_hypothesis
            timeline.append(
                f"[محاولة #{attempt}] الفرضية: {plan.root_cause_hypothesis} -> الإجراء: {plan.action_type}"
            )

            # 2. فحص بوابة نصف قطر الضرر (Blast-Radius Gate)
            try:
                SREBlastRadiusGatekeeper.validate_plan(plan, current_snapshot)
            except BlastRadiusViolationError as exc:
                timeline.append(f"🛑 حُجب الإجراء بواسطة بوابة الأمان: {exc}")
                failure_feedback = str(exc)
                continue

            # 3. تطبيق الإجراء مع الاحتفاظ بلقطة تراجع فورية (Canary Execution)
            await self._cluster.apply_action(plan)
            executed_actions.append(f"{plan.action_type.value} (attempt #{attempt})")

            # 4. فحص مؤشر مستوى الخدمة (SLO Health Verification)
            post_action_health = await self._cluster.gather_parallel_diagnostics()
            if self._is_slo_healthy(post_action_health):
                timeline.append(
                    f"✅ تعافى مؤشر الخدمة (SLO Recovered)! هبط الكمون إلى "
                    f"{post_action_health.p99_latency_ms}ms ومعدل الخطأ إلى {post_action_health.error_rate_pct}%."
                )
                mttr = round((time.perf_counter() - t0) * 1000, 2)
                return IncidentPostMortemReport(
                    incident_id=alert.incident_id,
                    service_name=alert.service_name,
                    resolved_autonomously=True,
                    rollback_triggered=rollback_triggered,
                    escalated_to_human_oncall=False,
                    root_cause_hypothesis=last_hypothesis,
                    executed_actions=executed_actions,
                    pre_remediation_health=pre_health,
                    post_remediation_health=post_action_health,
                    total_mttr_ms=mttr,
                    audit_timeline=timeline,
                )

            # 5. فشل مؤشر الخدمة بعد التعديل -> تراجع ذري فوري (Automatic Canary Rollback)!
            rollback_triggered = True
            self._cluster.restore_snapshot(current_snapshot)
            failure_feedback = (
                f"فشل Canary: بعد تنفيذ {plan.action_type} ارتفع الكمون إلى "
                f"{post_action_health.p99_latency_ms}ms ومعدل الأخطاء إلى {post_action_health.error_rate_pct}%."
            )
            timeline.append(
                f"⚠️ {failure_feedback} -> تم تنفيذ التراجع التلقائي (Auto-Rollback) لاستعادة اللقطة الآمنة!"
            )

        # إذا استُنفدت المحاولات، يتم التصعيد للمهندس المناوب
        final_health = self._cluster.get_snapshot()
        timeline.append("🚨 استُنفدت المحاولات الآمنة. تم تصعيد الحادثة للمهندس المناوب (PagerDuty).")
        mttr = round((time.perf_counter() - t0) * 1000, 2)
        return IncidentPostMortemReport(
            incident_id=alert.incident_id,
            service_name=alert.service_name,
            resolved_autonomously=False,
            rollback_triggered=rollback_triggered,
            escalated_to_human_oncall=True,
            root_cause_hypothesis=last_hypothesis,
            executed_actions=executed_actions,
            pre_remediation_health=pre_health,
            post_remediation_health=final_health,
            total_mttr_ms=mttr,
            audit_timeline=timeline,
        )


async def run_sre_application() -> None:
    cluster = SimulatedKubernetesCluster()
    harness = AutonomousSRERemediationHarness(cluster)

    alert = IncidentAlert(
        incident_id=f"INC-{uuid.uuid4().hex[:6].upper()}",
        service_name="checkout-payment-api",
        severity=SeverityLevel.SEV1_CRITICAL,
        symptom_summary="P99 latency > 2500ms and HTTP 503 spike on checkout-payment-api",
    )

    print("=" * 82)
    print("🚀 التطبيق الأول: وكيل SRE للاستجابة الذاتية للحوادث مع التراجع الآلي (Canary Rollback)")
    print("=" * 82)
    report = await harness.handle_incident(alert)

    for idx, entry in enumerate(report.audit_timeline, start=1):
        print(f"  {idx}. {entry}")

    print("-" * 82)
    print(
        f"📊 النتيجة النهائية: Resolved={report.resolved_autonomously} | "
        f"AutoRollbackUsed={report.rollback_triggered} | MTTR={report.total_mttr_ms}ms"
    )
    print(
        f"   • قبل الإصلاح : P99={report.pre_remediation_health.p99_latency_ms}ms, "
        f"Errors={report.pre_remediation_health.error_rate_pct}%"
    )
    print(
        f"   • بعد الإصلاح : P99={report.post_remediation_health.p99_latency_ms}ms, "
        f"Errors={report.post_remediation_health.error_rate_pct}%"
    )


if __name__ == "__main__":
    asyncio.run(run_sre_application())

# ────────────────────────────────────────────────────────────────────────
# [002] dist/md/app-projects.md
# ────────────────────────────────────────────────────────────────────────
import asyncio
import re
import sqlite3
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator


# =====================================================================
# 1. سياق الأمان وجدار حماية استعلامات SQL (Zero-Trust SQL Guard)
# =====================================================================

class TenantContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=3)
    analyst_id: str
    allow_raw_pii: bool = False


class ValidatedParameterizedSQL(BaseModel):
    """
    عقد صارم لاستعلام SQL:
    1. يجب أن يبدأ بـ SELECT.
    2. يمنع أي أوامر تعديل أو تعليقات أو استعلامات متعددة (;).
    3. يجب أن يحتوي إلزامياً على المعامل الآمن :injected_tenant_id لعزل البيانات.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    reasoning: str = Field(min_length=10)
    sql_query: str = Field(min_length=15, max_length=2000)

    @field_validator("sql_query")
    @classmethod
    def enforce_zero_trust_sql_rules(cls, v: str) -> str:
        cleaned = v.strip().rstrip(";")
        if ";" in cleaned or "--" in cleaned or "/*" in cleaned:
            raise ValueError(
                "يُمنع استخدام الفاصلة المنقوطة المتعددة أو تعليقات SQL."
            )

        upper_sql = cleaned.upper()
        if not upper_sql.startswith("SELECT"):
            raise ValueError("يُسمح فقط باستعلامات القراءة التي تبدأ بـ SELECT.")

        forbidden = {
            "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
            "CREATE", "PRAGMA", "ATTACH", "DETACH", "REPLACE",
        }
        tokens = set(re.findall(r"\b[A-Z_]+\b", upper_sql))
        violations = tokens.intersection(forbidden)
        if violations:
            raise ValueError(f"كلمات مفتاحية محظورة أمنياً في الاستعلام: {violations}")

        if ":injected_tenant_id" not in cleaned:
            raise ValueError(
                "خرق عزل المستأجرين (Row-Level Security Violation): يجب أن يتضمن الاستعلام "
                "شرط 'tenant_id = :injected_tenant_id' صراحةً!"
            )

        if "LIMIT" not in tokens:
            cleaned = f"{cleaned} LIMIT 50"

        return cleaned


class AnalyticsExecutionReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str
    executed_sql: str
    attempts_taken: int
    self_healed_errors: list[str]
    row_count: int
    sanitized_rows: list[dict[str, Any]]


# =====================================================================
# 2. محرك قاعدة البيانات المتعددة المستأجرين وحلقة التصحيح الذاتي
# =====================================================================

class EnterpriseAnalyticsDatabase:
    """قاعدة بيانات SQLite حقيقية في الذاكرة تضم بيانات لعدة شركات متنافسة."""

    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._seed_multi_tenant_data()

    def _seed_multi_tenant_data(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE subscription_invoices (
                    invoice_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    region TEXT NOT NULL,
                    customer_contact_email TEXT NOT NULL,
                    net_revenue_usd REAL NOT NULL,
                    status TEXT NOT NULL
                )
                """
            )
            rows = [
                # بيانات المستأجر الأول: ACME_CORP
                ("INV-101", "ACME_CORP", "MENA", "cfo@client-a.com", 45_000.0, "PAID"),
                ("INV-102", "ACME_CORP", "MENA", "finance@client-b.com", 32_500.0, "PAID"),
                ("INV-103", "ACME_CORP", "EU", "billing@eu-partner.de", 18_000.0, "PAID"),
                # بيانات المستأجر المنافس: GLOBEX_INC (يجب ألا تتسرب أبداً لـ ACME_CORP!)
                ("INV-999", "GLOBEX_INC", "MENA", "secret@globex.io", 999_999.0, "PAID"),
            ]
            self._conn.executemany(
                "INSERT INTO subscription_invoices VALUES (?, ?, ?, ?, ?, ?)", rows
            )

    def introspect_schema_ddl(self) -> str:
        """يستخرج المخطط الحقيقي للجداول والأعمدة لتغذية حلقة التصحيح الذاتي."""
        cur = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        return "\n".join(row["sql"] for row in cur.fetchall() if row["sql"])

    def execute_tenant_scoped_query(
        self, sql: str, tenant_id: str
    ) -> list[dict[str, Any]]:
        cur = self._conn.execute(sql, {"injected_tenant_id": tenant_id})
        return [dict(row) for row in cur.fetchall()]


class SelfHealingTextToSQLHarness:
    """
    يدير توليد SQL، ويطبق جدار الحماية، ويعالج أخطاء قاعدة البيانات ذاتياً
    عبر حقن مخطط DDL الفعلي، ويحجب البيانات الشخصية (PII Masking).
    """

    def __init__(
        self, db: EnterpriseAnalyticsDatabase, max_healing_retries: int = 3
    ) -> None:
        self._db = db
        self._max_retries = max_healing_retries

    @staticmethod
    def _mask_pii_in_row(row: dict[str, Any], allow_raw_pii: bool) -> dict[str, Any]:
        if allow_raw_pii:
            return row
        masked: dict[str, Any] = {}
        for k, val in row.items():
            if isinstance(val, str) and "@" in val:
                user_part, _, domain = val.partition("@")
                masked[k] = f"{user_part[:2]}***@{domain}"
            else:
                masked[k] = val
        return masked

    async def _simulate_llm_sql_generator(
        self,
        question: str,
        attempt: int,
        error_history: list[str],
        schema_hint: str,
    ) -> dict[str, str]:
        await asyncio.sleep(0.01)
        if attempt == 1:
            # المحاولة 1: النموذج ينسى وضع شرط :injected_tenant_id (سيصطاده Pydantic Firewall!)
            return {
                "reasoning": "سأقوم بجمع الفواتير حسب المنطقة.",
                "sql_query": "SELECT region, SUM(total_amount) FROM invoices GROUP BY region",
            }
        if attempt == 2:
            # المحاولة 2: النموذج يلتزم بشرط المستأجر، لكنه يهلوس باسم الجدول والعمود (total_amount بدلاً من net_revenue_usd)
            # ستصطاده قاعدة بيانات SQLite بـ OperationalError!
            return {
                "reasoning": "أضفت شرط عزل المستأجر، وسأستعلم عن مجموع الإيرادات.",
                "sql_query": (
                    "SELECT region, customer_contact_email, total_amount "
                    "FROM subscription_invoices "
                    "WHERE tenant_id = :injected_tenant_id"
                ),
            }
        # المحاولة 3: بعد أن حقن الـ Harness مخطط DDL الفعلي من SQLite، يكتب النموذج الاستعلام الصحيح تماماً!
        return {
            "reasoning": (
                "بناءً على مخطط الجدول المحقون، العمود الصحيح هو net_revenue_usd وليس total_amount."
            ),
            "sql_query": (
                "SELECT invoice_id, region, customer_contact_email, net_revenue_usd "
                "FROM subscription_invoices "
                "WHERE tenant_id = :injected_tenant_id AND status = 'PAID' "
                "ORDER BY net_revenue_usd DESC"
            ),
        }

    async def run_analytics_query(
        self, ctx: TenantContext, natural_language_question: str
    ) -> AnalyticsExecutionReport:
        healed_errors: list[str] = []
        schema_hint = ""

        for attempt in range(1, self._max_retries + 1):
            raw_llm_output = await self._simulate_llm_sql_generator(
                question=natural_language_question,
                attempt=attempt,
                error_history=healed_errors,
                schema_hint=schema_hint,
            )

            # البوابة 1: فحص Pydantic و Zero-Trust SQL Firewall
            try:
                validated = ValidatedParameterizedSQL.model_validate(raw_llm_output)
            except Exception as val_exc:
                err_msg = f"[Attempt #{attempt} Firewall Reject]: {val_exc}"
                healed_errors.append(err_msg)
                continue

            # البوابة 2: التنفيذ الفعلي داخل SQLite مع ربط المعاملات (Parameter Binding)
            try:
                raw_rows = self._db.execute_tenant_scoped_query(
                    sql=validated.sql_query,
                    tenant_id=ctx.tenant_id,
                )
                sanitized = [
                    self._mask_pii_in_row(r, ctx.allow_raw_pii) for r in raw_rows
                ]
                return AnalyticsExecutionReport(
                    tenant_id=ctx.tenant_id,
                    executed_sql=validated.sql_query,
                    attempts_taken=attempt,
                    self_healed_errors=healed_errors,
                    row_count=len(sanitized),
                    sanitized_rows=sanitized,
                )
            except sqlite3.OperationalError as db_exc:
                # استرجاع مخطط الجداول الفعلي تلقائياً لمساعدة النموذج على إصلاح خطئه
                schema_hint = self._db.introspect_schema_ddl()
                err_msg = (
                    f"[Attempt #{attempt} SQLite OperationalError]: {db_exc} "
                    f"-> Injected Actual Schema: {schema_hint.splitlines()[0]}..."
                )
                healed_errors.append(err_msg)

        raise RuntimeError(f"فشل إصلاح استعلام SQL بعد {self._max_retries} محاولات.")


async def run_text_to_sql_application() -> None:
    db = EnterpriseAnalyticsDatabase()
    harness = SelfHealingTextToSQLHarness(db=db, max_healing_retries=3)

    acme_ctx = TenantContext(
        tenant_id="ACME_CORP",
        analyst_id="bi_analyst_nadia",
        allow_raw_pii=False,
    )

    print("=" * 82)
    print("📊 التطبيق الثاني: وكيل Text-to-SQL ذاتي الإصلاح مع عزل المستأجرين وحجب PII")
    print("=" * 82)

    report = await harness.run_analytics_query(
        ctx=acme_ctx,
        natural_language_question="اعرض جميع الفواتير المدفوعة لشركتنا مرتبة حسب الإيراد الصافي.",
    )

    print(f"• المستأجر المعزول (Tenant) : {report.tenant_id}")
    print(f"• عدد محاولات التصحيح الذاتي: {report.attempts_taken}")
    print("• سجل الأخطاء التي أصلحها الـ Harness تلقائياً:")
    for err in report.self_healed_errors:
        print(f"   🛠️ {err}")

    print(f"\n• الاستعلام النهائي الآمن   : {report.executed_sql}")
    print(f"• النتائج المسترجعة ({report.row_count} صفوف — لاحظ عزل GLOBEX_INC وحجب البريد الإلكتروني):")
    for row in report.sanitized_rows:
        print(f"   -> {row}")


if __name__ == "__main__":
    asyncio.run(run_text_to_sql_application())

# ────────────────────────────────────────────────────────────────────────
# [003] dist/md/app-projects.md
# ────────────────────────────────────────────────────────────────────────
import asyncio
import hashlib
import hmac
import re
import uuid
from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field


# =====================================================================
# 1. نماذج البيانات السريرية ومطهر الخصوصية (HIPAA PHI Scrubber & Models)
# =====================================================================

class TriageAcuityLevel(StrEnum):
    ESI_1_RESUSCITATION = "ESI_1_RESUSCITATION"
    ESI_2_EMERGENT = "ESI_2_EMERGENT"
    ESI_3_URGENT = "ESI_3_URGENT"
    ESI_4_NON_URGENT = "ESI_4_NON_URGENT"


class RawPatientIntake(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    raw_clinical_note: str
    age: int = Field(ge=0, le=125)
    egfr_ml_min: float = Field(
        gt=0.0, description="معدل الترشيح الكبيبي لوظائف الكلى (الطبيعي > 60)"
    )
    current_medications: list[str]
    proposed_new_medication: str


class DeidentifiedClinicalCase(BaseModel):
    """سجل المريض بعد تجريده من كافة بيانات الهوية الشخصية (HIPAA Safe)."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    anonymous_case_id: str
    sanitized_clinical_note: str
    age: int
    egfr_ml_min: float
    renal_impairment_severe: bool
    current_medications: list[str]
    proposed_new_medication: str


class ClinicalSpecialistBallot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    specialist_role: str
    approve_proposed_medication: bool
    has_clinical_veto: bool = False
    contraindication_alerts: list[str] = Field(default_factory=list)
    recommended_safe_alternative: str | None = None
    clinical_rationale: str


class SignedPhysicianOrderTicket(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ticket_id: str
    anonymous_case_id: str
    acuity_level: TriageAcuityLevel
    proposed_medication_blocked: bool
    vetoed_by_specialists: list[str]
    final_recommended_regimen: str
    contraindication_warnings: list[str]
    requires_attending_physician_signoff: bool = True
    hmac_integrity_seal: str


class HIPAAPHIScrubber:
    """يجرد النصوص السريرية من الأسماء، وأرقام الهوية، وأرقام الهواتف."""

    _PATTERNS: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"المريض\s+[أ-ي]+\s+[أ-ي]+"), "المريض [PHI_NAME_REDACTED]"),
        (re.compile(r"\b\d{10}\b"), "[PHI_NATIONAL_ID_REDACTED]"),
        (re.compile(r"\+?\d[\d -]{8,14}\d"), "[PHI_PHONE_REDACTED]"),
    ]

    @classmethod
    def deidentify(cls, raw: RawPatientIntake) -> DeidentifiedClinicalCase:
        clean_note = raw.raw_clinical_note
        for pattern, repl in cls._PATTERNS:
            clean_note = pattern.sub(repl, clean_note)

        return DeidentifiedClinicalCase(
            anonymous_case_id=f"CLIN-{uuid.uuid4().hex[:8].upper()}",
            sanitized_clinical_note=clean_note,
            age=raw.age,
            egfr_ml_min=raw.egfr_ml_min,
            renal_impairment_severe=(raw.egfr_ml_min < 30.0),
            current_medications=[m.lower() for m in raw.current_medications],
            proposed_new_medication=raw.proposed_new_medication.lower(),
        )


# =====================================================================
# 2. محرك اليقظة الدوائية والوكلاء المتخصصين (Pharmacovigilance Swarm)
# =====================================================================

class ClinicalPharmacovigilanceHarness:
    """
    يدير سرب التقييم السريري المتوازي ويطبق حق النقض الدوائي المطلق (Absolute Veto)
    عند رصد أي تداخل دوائي قاتل أو موانع استعمال كلوية.
    """

    # قاعدة معرفة حتمية للتداخلات الدوائية الحرجة وموانع الاستعمال الكلوية
    _NSAID_DRUGS: frozenset[str] = frozenset({"ibuprofen", "naproxen", "ketorolac", "diclofenac"})
    _ANTICOAGULANTS: frozenset[str] = frozenset({"warfarin", "apixaban", "rivaroxaban"})

    def __init__(self, hmac_secret: str) -> None:
        self._secret = hmac_secret.encode("utf-8")

    async def _run_pharmacovigilance_specialist(
        self, case: DeidentifiedClinicalCase
    ) -> ClinicalSpecialistBallot:
        await asyncio.sleep(0.02)
        alerts: list[str] = []
        veto = False
        safe_alt: str | None = None

        proposed = case.proposed_new_medication
        current_set = set(case.current_medications)

        # فحص 1: مضادات الالتهاب غير الستيرويدية (NSAIDs) مع القصور الكلوي الحاد
        if proposed in self._NSAID_DRUGS and case.renal_impairment_severe:
            veto = True
            alerts.append(
                f"مانع استعمال كلوي قاطع (Renal Contraindication): الدواء '{proposed}' (NSAID) "
                f"يسبب فشلاً كلوياً حاداً لأن eGFR={case.egfr_ml_min} (< 30 mL/min)."
            )

        # فحص 2: التداخل الدوائي النزفي بين NSAIDs ومميعات الدم (Anticoagulants)
        active_anticoagulants = current_set.intersection(self._ANTICOAGULANTS)
        if proposed in self._NSAID_DRUGS and active_anticoagulants:
            veto = True
            alerts.append(
                f"تداخل دوائي خطير (Major Drug-Drug Interaction): الجمع بين '{proposed}' "
                f"ومميع الدم {list(active_anticoagulants)} يضاعف خطر النزيف الهضمي الحاد."
            )

        if veto:
            safe_alt = "acetaminophen (paracetamol) 500mg PO q8h PRN (Max 2g/day)"

        return ClinicalSpecialistBallot(
            specialist_role="clinical_pharmacologist_agent",
            approve_proposed_medication=(not veto),
            has_clinical_veto=veto,
            contraindication_alerts=alerts,
            recommended_safe_alternative=safe_alt,
            clinical_rationale=(
                "تم إجهاض الوصفة المقترحة بسبب موانع استعمال كلوية وتداخلات نزفية."
                if veto
                else "لا توجد تداخلات دوائية حرجة."
            ),
        )

    async def _run_internal_medicine_specialist(
        self, case: DeidentifiedClinicalCase
    ) -> ClinicalSpecialistBallot:
        await asyncio.sleep(0.025)
        return ClinicalSpecialistBallot(
            specialist_role="internal_medicine_triage_agent",
            approve_proposed_medication=True,
            has_clinical_veto=False,
            clinical_rationale=(
                "المريض يعاني من ألم مفصلي حاد ويحتاج لتسكين الألم مع مراقبة ضغط الدم."
            ),
        )

    async def _run_nephrology_guideline_specialist(
        self, case: DeidentifiedClinicalCase
    ) -> ClinicalSpecialistBallot:
        await asyncio.sleep(0.02)
        veto = case.renal_impairment_severe and (
            case.proposed_new_medication in self._NSAID_DRUGS
        )
        alerts = (
            ["KDIGO Guideline Violation: Avoid all NSAIDs in Stage 4/5 CKD (eGFR < 30)."]
            if veto
            else []
        )
        return ClinicalSpecialistBallot(
            specialist_role="nephrology_guideline_auditor",
            approve_proposed_medication=(not veto),
            has_clinical_veto=veto,
            contraindication_alerts=alerts,
            recommended_safe_alternative="acetaminophen (paracetamol) renal-adjusted dose",
            clinical_rationale="فحص التوافق مع البروتوكولات الإرشادية لأمراض الكلى (KDIGO).",
        )

    async def evaluate_clinical_case(
        self, raw_intake: RawPatientIntake
    ) -> tuple[DeidentifiedClinicalCase, SignedPhysicianOrderTicket]:
        # 1. تطهير بيانات الهوية الشخصية (HIPAA De-identification)
        clean_case = HIPAAPHIScrubber.deidentify(raw_intake)

        # 2. تشغيل سرب الأخصائيين بالتوازي المهيكل
        async with asyncio.TaskGroup() as tg:
            t_pharm = tg.create_task(
                self._run_pharmacovigilance_specialist(clean_case)
            )
            t_med = tg.create_task(
                self._run_internal_medicine_specialist(clean_case)
            )
            t_neph = tg.create_task(
                self._run_nephrology_guideline_specialist(clean_case)
            )

        ballots = [t_pharm.result(), t_med.result(), t_neph.result()]

        # 3. تطبيق بوابة الفيتو الطبي المطلق (Clinical Veto Gate)
        veto_roles = [b.specialist_role for b in ballots if b.has_clinical_veto]
        all_warnings = [
            alert for b in ballots for alert in b.contraindication_alerts
        ]

        if veto_roles:
            blocked = True
            alt = next(
                (
                    b.recommended_safe_alternative
                    for b in ballots
                    if b.recommended_safe_alternative
                ),
                "Consult Clinical Pharmacist Immediately",
            )
            final_regimen = (
                f"⛔ [BLOCKED: {clean_case.proposed_new_medication}] -> "
                f"✅ البديل الآمن المقترح: {alt}"
            )
            acuity = TriageAcuityLevel.ESI_2_EMERGENT
        else:
            blocked = False
            final_regimen = f"Approved: {clean_case.proposed_new_medication}"
            acuity = TriageAcuityLevel.ESI_3_URGENT

        ticket_id = f"PITL-{uuid.uuid4().hex[:8].upper()}"
        unsigned_payload = (
            f"{ticket_id}:{clean_case.anonymous_case_id}:{blocked}:{final_regimen}"
        )
        seal = hmac.new(
            self._secret, unsigned_payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        ticket = SignedPhysicianOrderTicket(
            ticket_id=ticket_id,
            anonymous_case_id=clean_case.anonymous_case_id,
            acuity_level=acuity,
            proposed_medication_blocked=blocked,
            vetoed_by_specialists=veto_roles,
            final_recommended_regimen=final_regimen,
            contraindication_warnings=all_warnings,
            hmac_integrity_seal=seal,
        )
        return clean_case, ticket


async def run_clinical_application() -> None:
    harness = ClinicalPharmacovigilanceHarness(
        hmac_secret="hospital-pitl-signing-key-2026"
    )

    raw_intake = RawPatientIntake(
        raw_clinical_note=(
            "حضر المريض خالد العتيبي (هوية رقم 1098765432، جوال +966501234567) "
            "يشكو من ألم حاد في الركبة اليمنى منذ يومين ويطلب مسكناً قوياً."
        ),
        age=68,
        egfr_ml_min=24.0,  # قصور كلوي حاد (أقل من 30)
        current_medications=["Warfarin", "Amlodipine"],  # يتناول مميع دم!
        proposed_new_medication="Ibuprofen",  # مسكن NSAID خطير جداً في هذه الحالة!
    )

    print("=" * 82)
    print("🏥 التطبيق الثالث: منظومة التيقظ الدوائي ودعم القرار السريري (HIPAA + Clinical Veto)")
    print("=" * 82)

    clean_case, ticket = await harness.evaluate_clinical_case(raw_intake)

    print(f"• معرف الحالة المعزول       : {clean_case.anonymous_case_id}")
    print(f"• الملاحظة بعد تطهير HIPAA  : {clean_case.sanitized_clinical_note}")
    print(f"• مستوى الفرز السريري       : {ticket.acuity_level}")
    print(f"• هل حُجب الدواء المقترح؟   : {ticket.proposed_medication_blocked}")
    print(f"• الأخصائيون أصحاب الفيتو   : {ticket.vetoed_by_specialists}")
    print("• تحذيرات التداخل الدوائي وموانع الاستعمال:")
    for warn in ticket.contraindication_warnings:
        print(f"   🚨 {warn}")
    print(f"\n• التوصية العلاجية للطبيب   : {ticket.final_recommended_regimen}")
    print(f"• ختم النزاهة الرقمي (HMAC) : {ticket.hmac_integrity_seal[:32]}...")


if __name__ == "__main__":
    asyncio.run(run_clinical_application())

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/agent-harness-engineering/notebooks/app-projects.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import asyncio
import time
import uuid
from enum import StrEnum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


# =====================================================================
# 1. العقود المهيكلة لحوادث SRE وإجراءات الإصلاح (Immutable Schemas)
# =====================================================================

class SeverityLevel(StrEnum):
    SEV1_CRITICAL = "SEV1_CRITICAL"
    SEV2_MAJOR = "SEV2_MAJOR"
    SEV3_MINOR = "SEV3_MINOR"


class RemediationActionType(StrEnum):
    SCALE_DEPLOYMENT = "scale_deployment"
    ROLLBACK_RELEASE = "rollback_release"
    FLUSH_STALE_POOL = "flush_stale_connection_pool"
    DELETE_NAMESPACE = "delete_namespace"  # محظور أمنياً!


class IncidentAlert(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    incident_id: str
    service_name: str
    severity: SeverityLevel
    symptom_summary: str
    triggered_at_epoch: float = Field(default_factory=time.time)


class ClusterHealthSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    active_replicas: int = Field(ge=0)
    cpu_utilization_pct: float = Field(ge=0.0, le=100.0)
    db_pool_saturation_pct: float = Field(ge=0.0, le=100.0)
    p99_latency_ms: float = Field(ge=0.0)
    error_rate_pct: float = Field(ge=0.0, le=100.0)
    recent_log_signature: str


class ProposedRemediationPlan(BaseModel):
    """مخطط قرار وكيل SRE: يجبره على كتابة الفرضية الهندسية قبل اختيار الأداة."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    root_cause_hypothesis: str = Field(min_length=15)
    action_type: RemediationActionType
    target_service: str
    target_replicas: int | None = Field(default=None, ge=1, le=50)
    confidence_score: float = Field(ge=0.0, le=1.0)


class IncidentPostMortemReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    incident_id: str
    service_name: str
    resolved_autonomously: bool
    rollback_triggered: bool
    escalated_to_human_oncall: bool
    root_cause_hypothesis: str
    executed_actions: list[str]
    pre_remediation_health: ClusterHealthSnapshot
    post_remediation_health: ClusterHealthSnapshot
    total_mttr_ms: float = Field(ge=0.0, description="Mean Time To Recovery in ms")
    audit_timeline: list[str]


# =====================================================================
# 2. بوابة أمان العمليات السحابية (Blast-Radius Safety Gatekeeper)
# =====================================================================

class BlastRadiusViolationError(PermissionError):
    """يُرفع عندما يقترح الوكيل إجراءً يتجاوز نصف قطر الضرر المسموح به."""


class SREBlastRadiusGatekeeper:
    _FORBIDDEN_ACTIONS: frozenset[RemediationActionType] = frozenset({
        RemediationActionType.DELETE_NAMESPACE,
    })
    _MAX_ALLOWED_REPLICAS: int = 20

    @classmethod
    def validate_plan(
        cls,
        plan: ProposedRemediationPlan,
        current_health: ClusterHealthSnapshot,
    ) -> None:
        if plan.action_type in cls._FORBIDDEN_ACTIONS:
            raise BlastRadiusViolationError(
                f"إجراء تدميري محظور: '{plan.action_type}' ممنوع قطعياً في التشغيل التلقائي."
            )

        if plan.action_type == RemediationActionType.SCALE_DEPLOYMENT:
            if plan.target_replicas is None:
                raise BlastRadiusViolationError(
                    "يجب تحديد target_replicas عند طلب توسيع الخدمة."
                )
            # منع تقليص عدد الحاويات (Scale-Down) أثناء وجود حادثة نشطة!
            if plan.target_replicas < current_health.active_replicas:
                raise BlastRadiusViolationError(
                    f"مرفوض: لا يُسمح بتقليص النسخ من {current_health.active_replicas} "
                    f"إلى {plan.target_replicas} أثناء حادثة نشطة!"
                )
            if plan.target_replicas > cls._MAX_ALLOWED_REPLICAS:
                raise BlastRadiusViolationError(
                    f"تجاوز سقف السعة: طلب {plan.target_replicas} نسخة يتجاوز الحد الأقصى ({cls._MAX_ALLOWED_REPLICAS})."
                )


# =====================================================================
# 3. بيئة البنية التحتية المحاكاة وعقل الوكيل (Cluster Simulator & Engine)
# =====================================================================

class SimulatedKubernetesCluster:
    """بيئة عنقود سحابي قابلة للعكس تدعم اللقطات والتراجع (Snapshots & Rollback)."""

    def __init__(self) -> None:
        self._state = ClusterHealthSnapshot(
            active_replicas=4,
            cpu_utilization_pct=94.5,
            db_pool_saturation_pct=99.2,
            p99_latency_ms=2850.0,
            error_rate_pct=14.8,
            recent_log_signature="FATAL: psycopg.PoolTimeout: connection pool exhausted (active=100/100)",
        )

    def get_snapshot(self) -> ClusterHealthSnapshot:
        return self._state

    def restore_snapshot(self, snapshot: ClusterHealthSnapshot) -> None:
        self._state = snapshot

    async def gather_parallel_diagnostics(self) -> ClusterHealthSnapshot:
        async def _fetch_prom_metrics() -> tuple[float, float, float]:
            await asyncio.sleep(0.02)
            return (
                self._state.cpu_utilization_pct,
                self._state.p99_latency_ms,
                self._state.error_rate_pct,
            )

        async def _fetch_loki_logs() -> str:
            await asyncio.sleep(0.025)
            return self._state.recent_log_signature

        async def _fetch_k8s_state() -> tuple[int, float]:
            await asyncio.sleep(0.015)
            return (self._state.active_replicas, self._state.db_pool_saturation_pct)

        async with asyncio.TaskGroup() as tg:
            t_prom = tg.create_task(_fetch_prom_metrics())
            t_logs = tg.create_task(_fetch_loki_logs())
            t_k8s = tg.create_task(_fetch_k8s_state())

        cpu, lat, err = t_prom.result()
        logs = t_logs.result()
        reps, pool = t_k8s.result()

        return ClusterHealthSnapshot(
            active_replicas=reps,
            cpu_utilization_pct=cpu,
            db_pool_saturation_pct=pool,
            p99_latency_ms=lat,
            error_rate_pct=err,
            recent_log_signature=logs,
        )

    async def apply_action(self, plan: ProposedRemediationPlan) -> None:
        await asyncio.sleep(0.03)
        match plan.action_type:
            case RemediationActionType.SCALE_DEPLOYMENT:
                # فخ هندسي واقعي: زيادة عدد الحاويات عندما يكون مجمع قاعدة البيانات ممتلئاً بنسبة 99%
                # يزيد الطين بلة ويرفع معدل الأخطاء لأن الحاويات الجديدة تطلب اتصالات إضافية!
                new_reps = plan.target_replicas or self._state.active_replicas
                self._state = ClusterHealthSnapshot(
                    active_replicas=new_reps,
                    cpu_utilization_pct=70.0,
                    db_pool_saturation_pct=100.0,
                    p99_latency_ms=4200.0,  # تدهور أسوأ!
                    error_rate_pct=28.5,    # ارتفعت الأخطاء!
                    recent_log_signature="FATAL: max_connections reached on primary DB!",
                )
            case RemediationActionType.FLUSH_STALE_POOL:
                # الحل الصحيح: تفريغ الاتصالات المعلقة يعيد النظام لحالته الطبيعية فوراً
                self._state = ClusterHealthSnapshot(
                    active_replicas=self._state.active_replicas,
                    cpu_utilization_pct=42.0,
                    db_pool_saturation_pct=35.0,
                    p99_latency_ms=115.0,
                    error_rate_pct=0.05,
                    recent_log_signature="INFO: DB connection pool healthy (active=35/100)",
                )
            case _:
                pass


class AutonomousSRERemediationHarness:
    """
    إطار التحكم الذاتي لحوادث SRE:
    يشخّص بالتوازي، ويتحقق من بوابة الأمان، ويطبق الإصلاح بنمط Canary،
    ويتراجع تلقائياً (Auto-Rollback) إذا تدهور مؤشر مستوى الخدمة (SLO)!
    """

    def __init__(self, cluster: SimulatedKubernetesCluster) -> None:
        self._cluster = cluster

    @staticmethod
    def _is_slo_healthy(health: ClusterHealthSnapshot) -> bool:
        return health.error_rate_pct < 1.0 and health.p99_latency_ms < 400.0

    async def _propose_plan(
        self,
        alert: IncidentAlert,
        health: ClusterHealthSnapshot,
        attempt: int,
        previous_failure_note: str,
    ) -> ProposedRemediationPlan:
        await asyncio.sleep(0.01)
        if attempt == 1:
            # في المحاولة الأولى يقترح الوكيل توسيع عدد الحاويات (Scale Up) ظناً منه أنها مشكلة CPU
            return ProposedRemediationPlan(
                root_cause_hypothesis=(
                    "استهلاك المعالج 94.5%، سأقوم بمضاعفة عدد الحاويات من 4 إلى 8 لتخفيف الضغط."
                ),
                action_type=RemediationActionType.SCALE_DEPLOYMENT,
                target_service=alert.service_name,
                target_replicas=8,
                confidence_score=0.75,
            )

        # في المحاولة الثانية (بعد التراجع الآلي وقراءة ملاحظة فشل Canary):
        # يدرك الوكيل أن المشكلة هي تشبع مجمع اتصالات قاعدة البيانات وليس المعالج!
        return ProposedRemediationPlan(
            root_cause_hypothesis=(
                f"توسيع الحاويات أدى لتدهور الكمون لأن المشكلة الجذرية هي تشبع DB Pool "
                f"({health.db_pool_saturation_pct}%). يجب تفريغ الاتصالات الخاملة المعلقة."
            ),
            action_type=RemediationActionType.FLUSH_STALE_POOL,
            target_service=alert.service_name,
            confidence_score=0.96,
        )

    async def handle_incident(
        self, alert: IncidentAlert, max_attempts: int = 2
    ) -> IncidentPostMortemReport:
        t0 = time.perf_counter()
        timeline: list[str] = [
            f"[{alert.incident_id}] استلام إنذار {alert.severity} للخدمة '{alert.service_name}': {alert.symptom_summary}"
        ]

        # 1. التشخيص المتوازي عبر TaskGroup
        pre_health = await self._cluster.gather_parallel_diagnostics()
        timeline.append(
            f"اكتمل التشخيص المتوازي: P99={pre_health.p99_latency_ms}ms, "
            f"ErrorRate={pre_health.error_rate_pct}%, DB_Pool={pre_health.db_pool_saturation_pct}%"
        )

        executed_actions: list[str] = []
        rollback_triggered = False
        failure_feedback = ""
        last_hypothesis = ""

        for attempt in range(1, max_attempts + 1):
            current_snapshot = self._cluster.get_snapshot()
            plan = await self._propose_plan(
                alert, current_snapshot, attempt, failure_feedback
            )
            last_hypothesis = plan.root_cause_hypothesis
            timeline.append(
                f"[محاولة #{attempt}] الفرضية: {plan.root_cause_hypothesis} -> الإجراء: {plan.action_type}"
            )

            # 2. فحص بوابة نصف قطر الضرر (Blast-Radius Gate)
            try:
                SREBlastRadiusGatekeeper.validate_plan(plan, current_snapshot)
            except BlastRadiusViolationError as exc:
                timeline.append(f"🛑 حُجب الإجراء بواسطة بوابة الأمان: {exc}")
                failure_feedback = str(exc)
                continue

            # 3. تطبيق الإجراء مع الاحتفاظ بلقطة تراجع فورية (Canary Execution)
            await self._cluster.apply_action(plan)
            executed_actions.append(f"{plan.action_type.value} (attempt #{attempt})")

            # 4. فحص مؤشر مستوى الخدمة (SLO Health Verification)
            post_action_health = await self._cluster.gather_parallel_diagnostics()
            if self._is_slo_healthy(post_action_health):
                timeline.append(
                    f"✅ تعافى مؤشر الخدمة (SLO Recovered)! هبط الكمون إلى "
                    f"{post_action_health.p99_latency_ms}ms ومعدل الخطأ إلى {post_action_health.error_rate_pct}%."
                )
                mttr = round((time.perf_counter() - t0) * 1000, 2)
                return IncidentPostMortemReport(
                    incident_id=alert.incident_id,
                    service_name=alert.service_name,
                    resolved_autonomously=True,
                    rollback_triggered=rollback_triggered,
                    escalated_to_human_oncall=False,
                    root_cause_hypothesis=last_hypothesis,
                    executed_actions=executed_actions,
                    pre_remediation_health=pre_health,
                    post_remediation_health=post_action_health,
                    total_mttr_ms=mttr,
                    audit_timeline=timeline,
                )

            # 5. فشل مؤشر الخدمة بعد التعديل -> تراجع ذري فوري (Automatic Canary Rollback)!
            rollback_triggered = True
            self._cluster.restore_snapshot(current_snapshot)
            failure_feedback = (
                f"فشل Canary: بعد تنفيذ {plan.action_type} ارتفع الكمون إلى "
                f"{post_action_health.p99_latency_ms}ms ومعدل الأخطاء إلى {post_action_health.error_rate_pct}%."
            )
            timeline.append(
                f"⚠️ {failure_feedback} -> تم تنفيذ التراجع التلقائي (Auto-Rollback) لاستعادة اللقطة الآمنة!"
            )

        # إذا استُنفدت المحاولات، يتم التصعيد للمهندس المناوب
        final_health = self._cluster.get_snapshot()
        timeline.append("🚨 استُنفدت المحاولات الآمنة. تم تصعيد الحادثة للمهندس المناوب (PagerDuty).")
        mttr = round((time.perf_counter() - t0) * 1000, 2)
        return IncidentPostMortemReport(
            incident_id=alert.incident_id,
            service_name=alert.service_name,
            resolved_autonomously=False,
            rollback_triggered=rollback_triggered,
            escalated_to_human_oncall=True,
            root_cause_hypothesis=last_hypothesis,
            executed_actions=executed_actions,
            pre_remediation_health=pre_health,
            post_remediation_health=final_health,
            total_mttr_ms=mttr,
            audit_timeline=timeline,
        )


async def run_sre_application() -> None:
    cluster = SimulatedKubernetesCluster()
    harness = AutonomousSRERemediationHarness(cluster)

    alert = IncidentAlert(
        incident_id=f"INC-{uuid.uuid4().hex[:6].upper()}",
        service_name="checkout-payment-api",
        severity=SeverityLevel.SEV1_CRITICAL,
        symptom_summary="P99 latency > 2500ms and HTTP 503 spike on checkout-payment-api",
    )

    print("=" * 82)
    print("🚀 التطبيق الأول: وكيل SRE للاستجابة الذاتية للحوادث مع التراجع الآلي (Canary Rollback)")
    print("=" * 82)
    report = await harness.handle_incident(alert)

    for idx, entry in enumerate(report.audit_timeline, start=1):
        print(f"  {idx}. {entry}")

    print("-" * 82)
    print(
        f"📊 النتيجة النهائية: Resolved={report.resolved_autonomously} | "
        f"AutoRollbackUsed={report.rollback_triggered} | MTTR={report.total_mttr_ms}ms"
    )
    print(
        f"   • قبل الإصلاح : P99={report.pre_remediation_health.p99_latency_ms}ms, "
        f"Errors={report.pre_remediation_health.error_rate_pct}%"
    )
    print(
        f"   • بعد الإصلاح : P99={report.post_remediation_health.p99_latency_ms}ms, "
        f"Errors={report.post_remediation_health.error_rate_pct}%"
    )


if __name__ == "__main__":
    asyncio.run(run_sre_application())

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/agent-harness-engineering/notebooks/app-projects.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import asyncio
import re
import sqlite3
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator


# =====================================================================
# 1. سياق الأمان وجدار حماية استعلامات SQL (Zero-Trust SQL Guard)
# =====================================================================

class TenantContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=3)
    analyst_id: str
    allow_raw_pii: bool = False


class ValidatedParameterizedSQL(BaseModel):
    """
    عقد صارم لاستعلام SQL:
    1. يجب أن يبدأ بـ SELECT.
    2. يمنع أي أوامر تعديل أو تعليقات أو استعلامات متعددة (;).
    3. يجب أن يحتوي إلزامياً على المعامل الآمن :injected_tenant_id لعزل البيانات.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    reasoning: str = Field(min_length=10)
    sql_query: str = Field(min_length=15, max_length=2000)

    @field_validator("sql_query")
    @classmethod
    def enforce_zero_trust_sql_rules(cls, v: str) -> str:
        cleaned = v.strip().rstrip(";")
        if ";" in cleaned or "--" in cleaned or "/*" in cleaned:
            raise ValueError(
                "يُمنع استخدام الفاصلة المنقوطة المتعددة أو تعليقات SQL."
            )

        upper_sql = cleaned.upper()
        if not upper_sql.startswith("SELECT"):
            raise ValueError("يُسمح فقط باستعلامات القراءة التي تبدأ بـ SELECT.")

        forbidden = {
            "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
            "CREATE", "PRAGMA", "ATTACH", "DETACH", "REPLACE",
        }
        tokens = set(re.findall(r"\b[A-Z_]+\b", upper_sql))
        violations = tokens.intersection(forbidden)
        if violations:
            raise ValueError(f"كلمات مفتاحية محظورة أمنياً في الاستعلام: {violations}")

        if ":injected_tenant_id" not in cleaned:
            raise ValueError(
                "خرق عزل المستأجرين (Row-Level Security Violation): يجب أن يتضمن الاستعلام "
                "شرط 'tenant_id = :injected_tenant_id' صراحةً!"
            )

        if "LIMIT" not in tokens:
            cleaned = f"{cleaned} LIMIT 50"

        return cleaned


class AnalyticsExecutionReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str
    executed_sql: str
    attempts_taken: int
    self_healed_errors: list[str]
    row_count: int
    sanitized_rows: list[dict[str, Any]]


# =====================================================================
# 2. محرك قاعدة البيانات المتعددة المستأجرين وحلقة التصحيح الذاتي
# =====================================================================

class EnterpriseAnalyticsDatabase:
    """قاعدة بيانات SQLite حقيقية في الذاكرة تضم بيانات لعدة شركات متنافسة."""

    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._seed_multi_tenant_data()

    def _seed_multi_tenant_data(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE subscription_invoices (
                    invoice_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    region TEXT NOT NULL,
                    customer_contact_email TEXT NOT NULL,
                    net_revenue_usd REAL NOT NULL,
                    status TEXT NOT NULL
                )
                """
            )
            rows = [
                # بيانات المستأجر الأول: ACME_CORP
                ("INV-101", "ACME_CORP", "MENA", "cfo@client-a.com", 45_000.0, "PAID"),
                ("INV-102", "ACME_CORP", "MENA", "finance@client-b.com", 32_500.0, "PAID"),
                ("INV-103", "ACME_CORP", "EU", "billing@eu-partner.de", 18_000.0, "PAID"),
                # بيانات المستأجر المنافس: GLOBEX_INC (يجب ألا تتسرب أبداً لـ ACME_CORP!)
                ("INV-999", "GLOBEX_INC", "MENA", "secret@globex.io", 999_999.0, "PAID"),
            ]
            self._conn.executemany(
                "INSERT INTO subscription_invoices VALUES (?, ?, ?, ?, ?, ?)", rows
            )

    def introspect_schema_ddl(self) -> str:
        """يستخرج المخطط الحقيقي للجداول والأعمدة لتغذية حلقة التصحيح الذاتي."""
        cur = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        return "\n".join(row["sql"] for row in cur.fetchall() if row["sql"])

    def execute_tenant_scoped_query(
        self, sql: str, tenant_id: str
    ) -> list[dict[str, Any]]:
        cur = self._conn.execute(sql, {"injected_tenant_id": tenant_id})
        return [dict(row) for row in cur.fetchall()]


class SelfHealingTextToSQLHarness:
    """
    يدير توليد SQL، ويطبق جدار الحماية، ويعالج أخطاء قاعدة البيانات ذاتياً
    عبر حقن مخطط DDL الفعلي، ويحجب البيانات الشخصية (PII Masking).
    """

    def __init__(
        self, db: EnterpriseAnalyticsDatabase, max_healing_retries: int = 3
    ) -> None:
        self._db = db
        self._max_retries = max_healing_retries

    @staticmethod
    def _mask_pii_in_row(row: dict[str, Any], allow_raw_pii: bool) -> dict[str, Any]:
        if allow_raw_pii:
            return row
        masked: dict[str, Any] = {}
        for k, val in row.items():
            if isinstance(val, str) and "@" in val:
                user_part, _, domain = val.partition("@")
                masked[k] = f"{user_part[:2]}***@{domain}"
            else:
                masked[k] = val
        return masked

    async def _simulate_llm_sql_generator(
        self,
        question: str,
        attempt: int,
        error_history: list[str],
        schema_hint: str,
    ) -> dict[str, str]:
        await asyncio.sleep(0.01)
        if attempt == 1:
            # المحاولة 1: النموذج ينسى وضع شرط :injected_tenant_id (سيصطاده Pydantic Firewall!)
            return {
                "reasoning": "سأقوم بجمع الفواتير حسب المنطقة.",
                "sql_query": "SELECT region, SUM(total_amount) FROM invoices GROUP BY region",
            }
        if attempt == 2:
            # المحاولة 2: النموذج يلتزم بشرط المستأجر، لكنه يهلوس باسم الجدول والعمود (total_amount بدلاً من net_revenue_usd)
            # ستصطاده قاعدة بيانات SQLite بـ OperationalError!
            return {
                "reasoning": "أضفت شرط عزل المستأجر، وسأستعلم عن مجموع الإيرادات.",
                "sql_query": (
                    "SELECT region, customer_contact_email, total_amount "
                    "FROM subscription_invoices "
                    "WHERE tenant_id = :injected_tenant_id"
                ),
            }
        # المحاولة 3: بعد أن حقن الـ Harness مخطط DDL الفعلي من SQLite، يكتب النموذج الاستعلام الصحيح تماماً!
        return {
            "reasoning": (
                "بناءً على مخطط الجدول المحقون، العمود الصحيح هو net_revenue_usd وليس total_amount."
            ),
            "sql_query": (
                "SELECT invoice_id, region, customer_contact_email, net_revenue_usd "
                "FROM subscription_invoices "
                "WHERE tenant_id = :injected_tenant_id AND status = 'PAID' "
                "ORDER BY net_revenue_usd DESC"
            ),
        }

    async def run_analytics_query(
        self, ctx: TenantContext, natural_language_question: str
    ) -> AnalyticsExecutionReport:
        healed_errors: list[str] = []
        schema_hint = ""

        for attempt in range(1, self._max_retries + 1):
            raw_llm_output = await self._simulate_llm_sql_generator(
                question=natural_language_question,
                attempt=attempt,
                error_history=healed_errors,
                schema_hint=schema_hint,
            )

            # البوابة 1: فحص Pydantic و Zero-Trust SQL Firewall
            try:
                validated = ValidatedParameterizedSQL.model_validate(raw_llm_output)
            except Exception as val_exc:
                err_msg = f"[Attempt #{attempt} Firewall Reject]: {val_exc}"
                healed_errors.append(err_msg)
                continue

            # البوابة 2: التنفيذ الفعلي داخل SQLite مع ربط المعاملات (Parameter Binding)
            try:
                raw_rows = self._db.execute_tenant_scoped_query(
                    sql=validated.sql_query,
                    tenant_id=ctx.tenant_id,
                )
                sanitized = [
                    self._mask_pii_in_row(r, ctx.allow_raw_pii) for r in raw_rows
                ]
                return AnalyticsExecutionReport(
                    tenant_id=ctx.tenant_id,
                    executed_sql=validated.sql_query,
                    attempts_taken=attempt,
                    self_healed_errors=healed_errors,
                    row_count=len(sanitized),
                    sanitized_rows=sanitized,
                )
            except sqlite3.OperationalError as db_exc:
                # استرجاع مخطط الجداول الفعلي تلقائياً لمساعدة النموذج على إصلاح خطئه
                schema_hint = self._db.introspect_schema_ddl()
                err_msg = (
                    f"[Attempt #{attempt} SQLite OperationalError]: {db_exc} "
                    f"-> Injected Actual Schema: {schema_hint.splitlines()[0]}..."
                )
                healed_errors.append(err_msg)

        raise RuntimeError(f"فشل إصلاح استعلام SQL بعد {self._max_retries} محاولات.")


async def run_text_to_sql_application() -> None:
    db = EnterpriseAnalyticsDatabase()
    harness = SelfHealingTextToSQLHarness(db=db, max_healing_retries=3)

    acme_ctx = TenantContext(
        tenant_id="ACME_CORP",
        analyst_id="bi_analyst_nadia",
        allow_raw_pii=False,
    )

    print("=" * 82)
    print("📊 التطبيق الثاني: وكيل Text-to-SQL ذاتي الإصلاح مع عزل المستأجرين وحجب PII")
    print("=" * 82)

    report = await harness.run_analytics_query(
        ctx=acme_ctx,
        natural_language_question="اعرض جميع الفواتير المدفوعة لشركتنا مرتبة حسب الإيراد الصافي.",
    )

    print(f"• المستأجر المعزول (Tenant) : {report.tenant_id}")
    print(f"• عدد محاولات التصحيح الذاتي: {report.attempts_taken}")
    print("• سجل الأخطاء التي أصلحها الـ Harness تلقائياً:")
    for err in report.self_healed_errors:
        print(f"   🛠️ {err}")

    print(f"\n• الاستعلام النهائي الآمن   : {report.executed_sql}")
    print(f"• النتائج المسترجعة ({report.row_count} صفوف — لاحظ عزل GLOBEX_INC وحجب البريد الإلكتروني):")
    for row in report.sanitized_rows:
        print(f"   -> {row}")


if __name__ == "__main__":
    asyncio.run(run_text_to_sql_application())

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/agent-harness-engineering/notebooks/app-projects.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import asyncio
import hashlib
import hmac
import re
import uuid
from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field


# =====================================================================
# 1. نماذج البيانات السريرية ومطهر الخصوصية (HIPAA PHI Scrubber & Models)
# =====================================================================

class TriageAcuityLevel(StrEnum):
    ESI_1_RESUSCITATION = "ESI_1_RESUSCITATION"
    ESI_2_EMERGENT = "ESI_2_EMERGENT"
    ESI_3_URGENT = "ESI_3_URGENT"
    ESI_4_NON_URGENT = "ESI_4_NON_URGENT"


class RawPatientIntake(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    raw_clinical_note: str
    age: int = Field(ge=0, le=125)
    egfr_ml_min: float = Field(
        gt=0.0, description="معدل الترشيح الكبيبي لوظائف الكلى (الطبيعي > 60)"
    )
    current_medications: list[str]
    proposed_new_medication: str


class DeidentifiedClinicalCase(BaseModel):
    """سجل المريض بعد تجريده من كافة بيانات الهوية الشخصية (HIPAA Safe)."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    anonymous_case_id: str
    sanitized_clinical_note: str
    age: int
    egfr_ml_min: float
    renal_impairment_severe: bool
    current_medications: list[str]
    proposed_new_medication: str


class ClinicalSpecialistBallot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    specialist_role: str
    approve_proposed_medication: bool
    has_clinical_veto: bool = False
    contraindication_alerts: list[str] = Field(default_factory=list)
    recommended_safe_alternative: str | None = None
    clinical_rationale: str


class SignedPhysicianOrderTicket(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ticket_id: str
    anonymous_case_id: str
    acuity_level: TriageAcuityLevel
    proposed_medication_blocked: bool
    vetoed_by_specialists: list[str]
    final_recommended_regimen: str
    contraindication_warnings: list[str]
    requires_attending_physician_signoff: bool = True
    hmac_integrity_seal: str


class HIPAAPHIScrubber:
    """يجرد النصوص السريرية من الأسماء، وأرقام الهوية، وأرقام الهواتف."""

    _PATTERNS: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"المريض\s+[أ-ي]+\s+[أ-ي]+"), "المريض [PHI_NAME_REDACTED]"),
        (re.compile(r"\b\d{10}\b"), "[PHI_NATIONAL_ID_REDACTED]"),
        (re.compile(r"\+?\d[\d -]{8,14}\d"), "[PHI_PHONE_REDACTED]"),
    ]

    @classmethod
    def deidentify(cls, raw: RawPatientIntake) -> DeidentifiedClinicalCase:
        clean_note = raw.raw_clinical_note
        for pattern, repl in cls._PATTERNS:
            clean_note = pattern.sub(repl, clean_note)

        return DeidentifiedClinicalCase(
            anonymous_case_id=f"CLIN-{uuid.uuid4().hex[:8].upper()}",
            sanitized_clinical_note=clean_note,
            age=raw.age,
            egfr_ml_min=raw.egfr_ml_min,
            renal_impairment_severe=(raw.egfr_ml_min < 30.0),
            current_medications=[m.lower() for m in raw.current_medications],
            proposed_new_medication=raw.proposed_new_medication.lower(),
        )


# =====================================================================
# 2. محرك اليقظة الدوائية والوكلاء المتخصصين (Pharmacovigilance Swarm)
# =====================================================================

class ClinicalPharmacovigilanceHarness:
    """
    يدير سرب التقييم السريري المتوازي ويطبق حق النقض الدوائي المطلق (Absolute Veto)
    عند رصد أي تداخل دوائي قاتل أو موانع استعمال كلوية.
    """

    # قاعدة معرفة حتمية للتداخلات الدوائية الحرجة وموانع الاستعمال الكلوية
    _NSAID_DRUGS: frozenset[str] = frozenset({"ibuprofen", "naproxen", "ketorolac", "diclofenac"})
    _ANTICOAGULANTS: frozenset[str] = frozenset({"warfarin", "apixaban", "rivaroxaban"})

    def __init__(self, hmac_secret: str) -> None:
        self._secret = hmac_secret.encode("utf-8")

    async def _run_pharmacovigilance_specialist(
        self, case: DeidentifiedClinicalCase
    ) -> ClinicalSpecialistBallot:
        await asyncio.sleep(0.02)
        alerts: list[str] = []
        veto = False
        safe_alt: str | None = None

        proposed = case.proposed_new_medication
        current_set = set(case.current_medications)

        # فحص 1: مضادات الالتهاب غير الستيرويدية (NSAIDs) مع القصور الكلوي الحاد
        if proposed in self._NSAID_DRUGS and case.renal_impairment_severe:
            veto = True
            alerts.append(
                f"مانع استعمال كلوي قاطع (Renal Contraindication): الدواء '{proposed}' (NSAID) "
                f"يسبب فشلاً كلوياً حاداً لأن eGFR={case.egfr_ml_min} (< 30 mL/min)."
            )

        # فحص 2: التداخل الدوائي النزفي بين NSAIDs ومميعات الدم (Anticoagulants)
        active_anticoagulants = current_set.intersection(self._ANTICOAGULANTS)
        if proposed in self._NSAID_DRUGS and active_anticoagulants:
            veto = True
            alerts.append(
                f"تداخل دوائي خطير (Major Drug-Drug Interaction): الجمع بين '{proposed}' "
                f"ومميع الدم {list(active_anticoagulants)} يضاعف خطر النزيف الهضمي الحاد."
            )

        if veto:
            safe_alt = "acetaminophen (paracetamol) 500mg PO q8h PRN (Max 2g/day)"

        return ClinicalSpecialistBallot(
            specialist_role="clinical_pharmacologist_agent",
            approve_proposed_medication=(not veto),
            has_clinical_veto=veto,
            contraindication_alerts=alerts,
            recommended_safe_alternative=safe_alt,
            clinical_rationale=(
                "تم إجهاض الوصفة المقترحة بسبب موانع استعمال كلوية وتداخلات نزفية."
                if veto
                else "لا توجد تداخلات دوائية حرجة."
            ),
        )

    async def _run_internal_medicine_specialist(
        self, case: DeidentifiedClinicalCase
    ) -> ClinicalSpecialistBallot:
        await asyncio.sleep(0.025)
        return ClinicalSpecialistBallot(
            specialist_role="internal_medicine_triage_agent",
            approve_proposed_medication=True,
            has_clinical_veto=False,
            clinical_rationale=(
                "المريض يعاني من ألم مفصلي حاد ويحتاج لتسكين الألم مع مراقبة ضغط الدم."
            ),
        )

    async def _run_nephrology_guideline_specialist(
        self, case: DeidentifiedClinicalCase
    ) -> ClinicalSpecialistBallot:
        await asyncio.sleep(0.02)
        veto = case.renal_impairment_severe and (
            case.proposed_new_medication in self._NSAID_DRUGS
        )
        alerts = (
            ["KDIGO Guideline Violation: Avoid all NSAIDs in Stage 4/5 CKD (eGFR < 30)."]
            if veto
            else []
        )
        return ClinicalSpecialistBallot(
            specialist_role="nephrology_guideline_auditor",
            approve_proposed_medication=(not veto),
            has_clinical_veto=veto,
            contraindication_alerts=alerts,
            recommended_safe_alternative="acetaminophen (paracetamol) renal-adjusted dose",
            clinical_rationale="فحص التوافق مع البروتوكولات الإرشادية لأمراض الكلى (KDIGO).",
        )

    async def evaluate_clinical_case(
        self, raw_intake: RawPatientIntake
    ) -> tuple[DeidentifiedClinicalCase, SignedPhysicianOrderTicket]:
        # 1. تطهير بيانات الهوية الشخصية (HIPAA De-identification)
        clean_case = HIPAAPHIScrubber.deidentify(raw_intake)

        # 2. تشغيل سرب الأخصائيين بالتوازي المهيكل
        async with asyncio.TaskGroup() as tg:
            t_pharm = tg.create_task(
                self._run_pharmacovigilance_specialist(clean_case)
            )
            t_med = tg.create_task(
                self._run_internal_medicine_specialist(clean_case)
            )
            t_neph = tg.create_task(
                self._run_nephrology_guideline_specialist(clean_case)
            )

        ballots = [t_pharm.result(), t_med.result(), t_neph.result()]

        # 3. تطبيق بوابة الفيتو الطبي المطلق (Clinical Veto Gate)
        veto_roles = [b.specialist_role for b in ballots if b.has_clinical_veto]
        all_warnings = [
            alert for b in ballots for alert in b.contraindication_alerts
        ]

        if veto_roles:
            blocked = True
            alt = next(
                (
                    b.recommended_safe_alternative
                    for b in ballots
                    if b.recommended_safe_alternative
                ),
                "Consult Clinical Pharmacist Immediately",
            )
            final_regimen = (
                f"⛔ [BLOCKED: {clean_case.proposed_new_medication}] -> "
                f"✅ البديل الآمن المقترح: {alt}"
            )
            acuity = TriageAcuityLevel.ESI_2_EMERGENT
        else:
            blocked = False
            final_regimen = f"Approved: {clean_case.proposed_new_medication}"
            acuity = TriageAcuityLevel.ESI_3_URGENT

        ticket_id = f"PITL-{uuid.uuid4().hex[:8].upper()}"
        unsigned_payload = (
            f"{ticket_id}:{clean_case.anonymous_case_id}:{blocked}:{final_regimen}"
        )
        seal = hmac.new(
            self._secret, unsigned_payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        ticket = SignedPhysicianOrderTicket(
            ticket_id=ticket_id,
            anonymous_case_id=clean_case.anonymous_case_id,
            acuity_level=acuity,
            proposed_medication_blocked=blocked,
            vetoed_by_specialists=veto_roles,
            final_recommended_regimen=final_regimen,
            contraindication_warnings=all_warnings,
            hmac_integrity_seal=seal,
        )
        return clean_case, ticket


async def run_clinical_application() -> None:
    harness = ClinicalPharmacovigilanceHarness(
        hmac_secret="hospital-pitl-signing-key-2026"
    )

    raw_intake = RawPatientIntake(
        raw_clinical_note=(
            "حضر المريض خالد العتيبي (هوية رقم 1098765432، جوال +966501234567) "
            "يشكو من ألم حاد في الركبة اليمنى منذ يومين ويطلب مسكناً قوياً."
        ),
        age=68,
        egfr_ml_min=24.0,  # قصور كلوي حاد (أقل من 30)
        current_medications=["Warfarin", "Amlodipine"],  # يتناول مميع دم!
        proposed_new_medication="Ibuprofen",  # مسكن NSAID خطير جداً في هذه الحالة!
    )

    print("=" * 82)
    print("🏥 التطبيق الثالث: منظومة التيقظ الدوائي ودعم القرار السريري (HIPAA + Clinical Veto)")
    print("=" * 82)

    clean_case, ticket = await harness.evaluate_clinical_case(raw_intake)

    print(f"• معرف الحالة المعزول       : {clean_case.anonymous_case_id}")
    print(f"• الملاحظة بعد تطهير HIPAA  : {clean_case.sanitized_clinical_note}")
    print(f"• مستوى الفرز السريري       : {ticket.acuity_level}")
    print(f"• هل حُجب الدواء المقترح؟   : {ticket.proposed_medication_blocked}")
    print(f"• الأخصائيون أصحاب الفيتو   : {ticket.vetoed_by_specialists}")
    print("• تحذيرات التداخل الدوائي وموانع الاستعمال:")
    for warn in ticket.contraindication_warnings:
        print(f"   🚨 {warn}")
    print(f"\n• التوصية العلاجية للطبيب   : {ticket.final_recommended_regimen}")
    print(f"• ختم النزاهة الرقمي (HMAC) : {ticket.hmac_integrity_seal[:32]}...")


if __name__ == "__main__":
    asyncio.run(run_clinical_application())
