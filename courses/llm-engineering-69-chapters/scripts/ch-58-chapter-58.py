#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — مشروع تطبيقي — Agent متكامل يستخدم RAG + أدوات خارجية
Large Language Model Engineering — Agent

Book slug : llm-engineering-69-chapters
Chapter   : ch-58-chapter-58 (lesson 58)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-58-chapter-58.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — مشروع تطبيقي — Agent متكامل يستخدم RAG + أدوات خارجية
Large Language Model Engineering — Agent

Book slug : llm-engineering-69-chapters
Chapter   : ch-58-chapter-58 (lesson 58)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-58-chapter-58.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — مشروع تطبيقي — Agent متكامل يستخدم RAG + أدوات خارجية
Large Language Model Engineering — Agent

Book slug : llm-engineering-69-chapters
Chapter   : ch-58-chapter-58 (lesson 58)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_58.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 58: الفصل 58: مشروع تطبيقي — Agent متكامل يستخدم RAG + أدوات خارجية."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_agent_rag() -> None:
    knowledge = {'attention': 'Attention combines values using context-dependent weights.', 'tokenizer': 'A tokenizer converts text into discrete identifiers.'}
    query = 'attention'
    evidence = knowledge.get(query)
    state = {'step': 'retrieve', 'evidence': evidence, 'tool_calls': 1}
    answer = f'{evidence} [source:{query}]' if evidence else 'I do not have evidence.'
    print('workflow state:', state)
    print('grounded answer:', answer)

if __name__ == "__main__":
    print("Chapter 58 lab: Run a local retrieval-and-tool workflow")
    run_agent_rag()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-10-applications-agents/chapter-58.md
# ────────────────────────────────────────────────────────────────────────
import os
import glob
from dataclasses import dataclass
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


class CompanyKnowledgeBase:
    """فهرسة واسترجاع وثائق الشركة الداخلية — مبني على أساس الفصل 53"""

    def __init__(self, docs_dir: str, chunk_size: int = 300, chunk_overlap: int = 50):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.chunks: List[Chunk] = []
        self.index = None
        self._build_index(docs_dir, chunk_size, chunk_overlap)

    def _chunk_text(self, text: str, source: str, chunk_size: int, overlap: int) -> List[Chunk]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end])
            chunks.append(Chunk(text=chunk_text, source=source, chunk_id=len(self.chunks) + len(chunks)))
            start += chunk_size - overlap
        return chunks

    def _build_index(self, docs_dir: str, chunk_size: int, overlap: int):
        for filepath in glob.glob(os.path.join(docs_dir, "*.md")):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            source_name = os.path.basename(filepath)
            self.chunks.extend(self._chunk_text(text, source_name, chunk_size, overlap))

        if not self.chunks:
            raise ValueError(f"لم يتم العثور على أي وثائق في {docs_dir}")

        embeddings = self.embedder.encode(
            [c.text for c in self.chunks], normalize_embeddings=True, show_progress_bar=False
        )
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(np.array(embeddings).astype("float32"))
        print(f"✅ تم فهرسة {len(self.chunks)} قطعة نصية من {docs_dir}")

    def search(self, query: str, k: int = 3, min_score: float = 0.3) -> List[dict]:
        query_emb = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(query_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < min_score:
                continue
            chunk = self.chunks[idx]
            results.append({"text": chunk.text, "source": chunk.source, "score": float(score)})
        return results

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-10-applications-agents/chapter-58.md
# ────────────────────────────────────────────────────────────────────────
import json
import math
from datetime import datetime
from typing import Dict, Any


def calculator_tool(expression: str) -> str:
    """حاسبة آمنة — تقيّم تعبيرات رياضية فقط، لا تنفيذ كود عشوائي"""
    allowed_chars = set("0123456789+-*/(). ")
    if not all(c in allowed_chars for c in expression):
        return "خطأ: التعبير يحتوي على رموز غير مسموحة."
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return f"النتيجة: {result}"
    except Exception as e:
        return f"خطأ في الحساب: {str(e)}"


def get_current_datetime_tool(_: str = "") -> str:
    """يُرجع التاريخ والوقت الحاليين — مهم لأسئلة مرتبطة بالمواعيد النهائية"""
    now = datetime.now()
    return f"التاريخ والوقت الحاليان: {now.strftime('%Y-%m-%d %H:%M')} ({now.strftime('%A')})"


def mock_currency_exchange_tool(query: str) -> str:
    """محاكاة API خارجي لأسعار الصرف (في الإنتاج: استبدله بنداء API حقيقي كـ exchangerate-api.com)"""
    mock_rates = {"USD": 3.75, "EUR": 4.10, "GBP": 4.75}
    for currency, rate in mock_rates.items():
        if currency in query.upper():
            return f"سعر صرف {currency} الحالي: {rate} ريال سعودي (بيانات تجريبية)"
    return "لم أتمكن من التعرف على العملة المطلوبة. العملات المدعومة: USD, EUR, GBP"


TOOLS_REGISTRY: Dict[str, Any] = {
    "calculator": calculator_tool,
    "get_current_datetime": get_current_datetime_tool,
    "currency_exchange": mock_currency_exchange_tool,
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "يحسب تعبيرات رياضية (جمع، طرح، ضرب، قسمة، أقواس)",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "التعبير الرياضي، مثل '150*3+20'"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "يُرجع التاريخ والوقت الحاليين، مفيد لحساب المواعيد النهائية والفترات الزمنية",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "currency_exchange",
            "description": "يُرجع سعر صرف عملة أجنبية مقابل الريال السعودي",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "اسم العملة، مثل 'USD' أو 'دولار'"}},
                "required": ["query"],
            },
        },
    },
]

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-10-applications-agents/chapter-58.md
# ────────────────────────────────────────────────────────────────────────
import time
from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class UserSession:
    """ذاكرة مبسّطة لكل مستخدم — مبنية على مبادئ الفصل 57 لكن مُصغَّرة لهذا المشروع"""
    user_id: str
    conversation_buffer: List[Dict[str, str]] = field(default_factory=list)
    known_facts: List[str] = field(default_factory=list)
    last_active: float = field(default_factory=time.time)


class MemoryStore:
    """مخزن ذاكرة متعدد المستخدمين — عزل كامل بالـ user_id (كما نصحنا في الفصل 57)"""

    def __init__(self):
        self._sessions: Dict[str, UserSession] = {}

    def get_or_create(self, user_id: str) -> UserSession:
        if user_id not in self._sessions:
            self._sessions[user_id] = UserSession(user_id=user_id)
        return self._sessions[user_id]

    def add_turn(self, user_id: str, user_msg: str, assistant_msg: str):
        session = self.get_or_create(user_id)
        session.conversation_buffer.append({"role": "user", "content": user_msg})
        session.conversation_buffer.append({"role": "assistant", "content": assistant_msg})
        session.conversation_buffer = session.conversation_buffer[-10:]  # آخر 5 تبادلات فقط
        session.last_active = time.time()

    def add_fact(self, user_id: str, fact: str):
        session = self.get_or_create(user_id)
        if fact not in session.known_facts:
            session.known_facts.append(fact)

    def get_context(self, user_id: str) -> str:
        session = self.get_or_create(user_id)
        facts_text = "\n".join(f"- {f}" for f in session.known_facts) or "لا توجد معلومات مسجّلة بعد"
        history_text = "\n".join(f"{m['role']}: {m['content']}" for m in session.conversation_buffer) or "محادثة جديدة"
        return f"[حقائق معروفة عن المستخدم]:\n{facts_text}\n\n[آخر المحادثة]:\n{history_text}"

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-10-applications-agents/chapter-58.md
# ────────────────────────────────────────────────────────────────────────
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BLOCKED_PATTERNS = ["كلمة مرور", "password", "رقم بطاقة", "credit card"]


def check_input_safety(user_input: str) -> tuple[bool, str]:
    """فحص أولي سريع بدون LLM — يمنع طلبات واضحة الخطورة فورًا بتكلفة صفر"""
    lower_input = user_input.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern in lower_input:
            return False, "عذرًا، لا يمكنني معالجة طلبات تتعلق بمعلومات حساسة كهذه."
    return True, ""


def check_output_groundedness(answer: str, retrieved_context: str) -> tuple[bool, str]:
    """فحص: هل الإجابة المبنية على RAG مدعومة فعلًا بالسياق المسترجَع؟ (مانع هلوسة أخير)"""
    if not retrieved_context.strip():
        return True, answer  # لا يوجد سياق RAG ليُفحص أصلًا (إجابة مباشرة أو أداة)

    prompt = f"""هل الإجابة التالية مدعومة فعليًا بالسياق المرفق، أم تحتوي على ادعاءات غير موجودة فيه؟
أجب بكلمة واحدة فقط: "مدعومة" أو "غير_مدعومة".

[السياق]:
{retrieved_context}

[الإجابة]:
{answer}"""

    verdict = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    ).choices[0].message.content.strip()

    if "غير_مدعومة" in verdict:
        safe_answer = "عذرًا، لا أملك معلومات كافية موثوقة في وثائق الشركة للإجابة على هذا بدقة. أنصح بالتواصل مع قسم الموارد البشرية مباشرة."
        return False, safe_answer
    return True, answer

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-10-applications-agents/chapter-58.md
# ────────────────────────────────────────────────────────────────────────
import json
from typing import TypedDict, List, Optional, Literal
from openai import OpenAI
from langgraph.graph import StateGraph, END
import os

from rag_module import CompanyKnowledgeBase
from tools_module import TOOLS_REGISTRY, TOOLS_SCHEMA
from memory_module import MemoryStore
from guardrails_module import check_input_safety, check_output_groundedness

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class AgentState(TypedDict):
    user_id: str
    question: str
    route: Optional[Literal["rag", "tool", "direct"]]
    retrieved_context: str
    sources: List[str]
    tool_results: List[str]
    final_answer: str
    is_safe: bool
    cost_tokens: int


# ───────────────────── العقدة 1: التوجيه (Router) ─────────────────────
def route_question(state: AgentState) -> dict:
    system_prompt = """صنّف سؤال المستخدم إلى واحدة من ثلاث فئات بدقة:
- "rag": أسئلة عن سياسات الشركة الداخلية (إجازات، مصاريف، عمل عن بُعد)
- "tool": أسئلة تحتاج حسابًا رياضيًا، تاريخًا/وقتًا حاليًا، أو أسعار صرف عملات
- "direct": أسئلة عامة لا تحتاج أيًا مما سبق (تحيات، أسئلة عامة بسيطة)

أعد JSON فقط: {"route": "rag|tool|direct"}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": state["question"]}],
        temperature=0.0, response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    tokens = response.usage.total_tokens

    return {"route": result.get("route", "direct"), "cost_tokens": state.get("cost_tokens", 0) + tokens}


def routing_decision(state: AgentState) -> str:
    """دالة التوجيه الشرطي التي يستخدمها LangGraph لاختيار العقدة التالية"""
    return state["route"]


# ───────────────────── العقدة 2أ: استرجاع RAG ─────────────────────
def make_rag_node(kb: CompanyKnowledgeBase):
    def rag_retrieval(state: AgentState) -> dict:
        results = kb.search(state["question"], k=3)
        context = "\n\n".join(f"[من {r['source']}]: {r['text']}" for r in results)
        sources = list(set(r["source"] for r in results))
        return {"retrieved_context": context, "sources": sources}
    return rag_retrieval


# ───────────────────── العقدة 2ب: استدعاء الأدوات ─────────────────────
def tool_calling(state: AgentState) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": state["question"]}],
        tools=TOOLS_SCHEMA, tool_choice="auto", temperature=0.0,
    )
    message = response.choices[0].message
    tokens = response.usage.total_tokens
    tool_results = []

    if message.tool_calls:
        for call in message.tool_calls:
            func_name = call.function.name
            func_args = json.loads(call.function.arguments)
            result = TOOLS_REGISTRY[func_name](**func_args)
            tool_results.append(f"[أداة {func_name}]: {result}")

    return {"tool_results": tool_results, "cost_tokens": state.get("cost_tokens", 0) + tokens}


# ───────────────────── العقدة 3: التوليف النهائي ─────────────────────
def make_synthesis_node(memory: MemoryStore):
    def synthesize_answer(state: AgentState) -> dict:
        user_context = memory.get_context(state["user_id"])

        extra_info = ""
        if state.get("retrieved_context"):
            extra_info += f"\n[معلومات من وثائق الشركة]:\n{state['retrieved_context']}"
        if state.get("tool_results"):
            extra_info += "\n[نتائج أدوات]:\n" + "\n".join(state["tool_results"])

        system_prompt = f"""أنت مساعد موظفي TechCorp. أجب بدقة ووضوح واختصار.
إذا استخدمت معلومات من وثائق الشركة، اذكر ذلك صراحة. لا تختلق معلومات غير موجودة في السياق المرفق.

{user_context}
{extra_info}"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": state["question"]}],
            temperature=0.4,
        )
        answer = response.choices[0].message.content
        tokens = response.usage.total_tokens

        return {"final_answer": answer, "cost_tokens": state.get("cost_tokens", 0) + tokens}
    return synthesize_answer


# ───────────────────── العقدة 4: فحص الحماية النهائي ─────────────────────
def guardrail_node(state: AgentState) -> dict:
    is_grounded, safe_answer = check_output_groundedness(
        state["final_answer"], state.get("retrieved_context", "")
    )
    return {"final_answer": safe_answer, "is_safe": is_grounded}


# ───────────────────── بناء الرسم البياني الكامل ─────────────────────
def build_agent_graph(kb: CompanyKnowledgeBase, memory: MemoryStore):
    graph = StateGraph(AgentState)

    graph.add_node("router", route_question)
    graph.add_node("rag_retrieval", make_rag_node(kb))
    graph.add_node("tool_calling", tool_calling)
    graph.add_node("synthesis", make_synthesis_node(memory))
    graph.add_node("guardrail", guardrail_node)

    graph.set_entry_point("router")

    # التوجيه الشرطي: من router نذهب إلى واحد من ثلاثة مسارات
    graph.add_conditional_edges(
        "router", routing_decision,
        {"rag": "rag_retrieval", "tool": "tool_calling", "direct": "synthesis"},
    )

    graph.add_edge("rag_retrieval", "synthesis")
    graph.add_edge("tool_calling", "synthesis")
    graph.add_edge("synthesis", "guardrail")
    graph.add_edge("guardrail", END)

    return graph.compile()

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-10-applications-agents/chapter-58.md
# ────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import logging

from rag_module import CompanyKnowledgeBase
from memory_module import MemoryStore
from graph import build_agent_graph
from guardrails_module import check_input_safety

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("techcorp_assistant")

app = FastAPI(title="TechCorp AI Assistant", version="1.0.0")

# تحميل مكوّن واحد فقط عند بدء التشغيل (وليس في كل طلب)
knowledge_base = CompanyKnowledgeBase(docs_dir="knowledge_base/")
memory_store = MemoryStore()
agent = build_agent_graph(knowledge_base, memory_store)


class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    route_used: str
    tokens_used: int
    latency_ms: float
    is_grounded: bool


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    start_time = time.time()

    is_safe, rejection_msg = check_input_safety(request.message)
    if not is_safe:
        return ChatResponse(answer=rejection_msg, sources=[], route_used="blocked",
                             tokens_used=0, latency_ms=0, is_grounded=True)

    try:
        result = agent.invoke({
            "user_id": request.user_id, "question": request.message,
            "route": None, "retrieved_context": "", "sources": [],
            "tool_results": [], "final_answer": "", "is_safe": True, "cost_tokens": 0,
        })
    except Exception as e:
        logger.error(f"خطأ في معالجة الطلب: {e}")
        raise HTTPException(status_code=500, detail="حدث خطأ داخلي، حاول مرة أخرى.")

    memory_store.add_turn(request.user_id, request.message, result["final_answer"])
    latency = (time.time() - start_time) * 1000

    logger.info(f"user={request.user_id} route={result['route']} "
                f"tokens={result['cost_tokens']} latency={latency:.0f}ms")

    return ChatResponse(
        answer=result["final_answer"], sources=result.get("sources", []),
        route_used=result["route"], tokens_used=result["cost_tokens"],
        latency_ms=round(latency, 1), is_grounded=result.get("is_safe", True),
    )


@app.get("/health")
def health_check():
    return {"status": "healthy", "knowledge_base_chunks": len(knowledge_base.chunks)}

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-10-applications-agents/chapter-58.md
# ────────────────────────────────────────────────────────────────────────
# test_integration.py
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_rag_route():
    """يجب أن يُوجَّه سؤال سياسة الإجازات إلى RAG ويعيد مصدرًا صحيحًا"""
    response = client.post("/chat", json={"user_id": "test_user_1", "message": "كم يوم إجازة سنوية يحق لي؟"})
    data = response.json()
    assert response.status_code == 200
    assert data["route_used"] == "rag"
    assert "vacation_policy.md" in data["sources"]
    assert "25" in data["answer"]


def test_tool_route():
    """يجب أن يُوجَّه سؤال حسابي إلى الأداة ويحسب بدقة"""
    response = client.post("/chat", json={"user_id": "test_user_2", "message": "كم ناتج 450 ضرب 3 زائد 100؟"})
    data = response.json()
    assert data["route_used"] == "tool"
    assert "1450" in data["answer"]


def test_memory_persistence():
    """يجب أن يتذكر النظام معلومة مذكورة في رسالة سابقة لنفس المستخدم"""
    client.post("/chat", json={"user_id": "test_user_3", "message": "اسمي سارة وأعمل في قسم الهندسة."})
    response = client.post("/chat", json={"user_id": "test_user_3", "message": "في أي قسم أعمل؟"})
    assert "هندسة" in response.json()["answer"]


def test_input_guardrail():
    """يجب رفض طلبات تحتوي على معلومات حساسة فورًا دون الوصول للـ LLM"""
    response = client.post("/chat", json={"user_id": "test_user_4", "message": "ما هي كلمة مرور النظام الداخلي؟"})
    data = response.json()
    assert data["route_used"] == "blocked"
    assert data["tokens_used"] == 0


if __name__ == "__main__":
    test_rag_route()
    test_tool_route()
    test_memory_persistence()
    test_input_guardrail()
    print("✅ جميع اختبارات التكامل نجحت!")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-58-chapter-58.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 58: الفصل 58: مشروع تطبيقي — Agent متكامل يستخدم RAG + أدوات خارجية."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_agent_rag() -> None:
    knowledge = {'attention': 'Attention combines values using context-dependent weights.', 'tokenizer': 'A tokenizer converts text into discrete identifiers.'}
    query = 'attention'
    evidence = knowledge.get(query)
    state = {'step': 'retrieve', 'evidence': evidence, 'tool_calls': 1}
    answer = f'{evidence} [source:{query}]' if evidence else 'I do not have evidence.'
    print('workflow state:', state)
    print('grounded answer:', answer)

if __name__ == "__main__":
    print("Chapter 58 lab: Run a local retrieval-and-tool workflow")
    run_agent_rag()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-58-chapter-58.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import os
import glob
from dataclasses import dataclass
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


class CompanyKnowledgeBase:
    """فهرسة واسترجاع وثائق الشركة الداخلية — مبني على أساس الفصل 53"""

    def __init__(self, docs_dir: str, chunk_size: int = 300, chunk_overlap: int = 50):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.chunks: List[Chunk] = []
        self.index = None
        self._build_index(docs_dir, chunk_size, chunk_overlap)

    def _chunk_text(self, text: str, source: str, chunk_size: int, overlap: int) -> List[Chunk]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end])
            chunks.append(Chunk(text=chunk_text, source=source, chunk_id=len(self.chunks) + len(chunks)))
            start += chunk_size - overlap
        return chunks

    def _build_index(self, docs_dir: str, chunk_size: int, overlap: int):
        for filepath in glob.glob(os.path.join(docs_dir, "*.md")):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            source_name = os.path.basename(filepath)
            self.chunks.extend(self._chunk_text(text, source_name, chunk_size, overlap))

        if not self.chunks:
            raise ValueError(f"لم يتم العثور على أي وثائق في {docs_dir}")

        embeddings = self.embedder.encode(
            [c.text for c in self.chunks], normalize_embeddings=True, show_progress_bar=False
        )
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(np.array(embeddings).astype("float32"))
        print(f"✅ تم فهرسة {len(self.chunks)} قطعة نصية من {docs_dir}")

    def search(self, query: str, k: int = 3, min_score: float = 0.3) -> List[dict]:
        query_emb = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(query_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < min_score:
                continue
            chunk = self.chunks[idx]
            results.append({"text": chunk.text, "source": chunk.source, "score": float(score)})
        return results

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-58-chapter-58.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import json
import math
from datetime import datetime
from typing import Dict, Any


def calculator_tool(expression: str) -> str:
    """حاسبة آمنة — تقيّم تعبيرات رياضية فقط، لا تنفيذ كود عشوائي"""
    allowed_chars = set("0123456789+-*/(). ")
    if not all(c in allowed_chars for c in expression):
        return "خطأ: التعبير يحتوي على رموز غير مسموحة."
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return f"النتيجة: {result}"
    except Exception as e:
        return f"خطأ في الحساب: {str(e)}"


def get_current_datetime_tool(_: str = "") -> str:
    """يُرجع التاريخ والوقت الحاليين — مهم لأسئلة مرتبطة بالمواعيد النهائية"""
    now = datetime.now()
    return f"التاريخ والوقت الحاليان: {now.strftime('%Y-%m-%d %H:%M')} ({now.strftime('%A')})"


def mock_currency_exchange_tool(query: str) -> str:
    """محاكاة API خارجي لأسعار الصرف (في الإنتاج: استبدله بنداء API حقيقي كـ exchangerate-api.com)"""
    mock_rates = {"USD": 3.75, "EUR": 4.10, "GBP": 4.75}
    for currency, rate in mock_rates.items():
        if currency in query.upper():
            return f"سعر صرف {currency} الحالي: {rate} ريال سعودي (بيانات تجريبية)"
    return "لم أتمكن من التعرف على العملة المطلوبة. العملات المدعومة: USD, EUR, GBP"


TOOLS_REGISTRY: Dict[str, Any] = {
    "calculator": calculator_tool,
    "get_current_datetime": get_current_datetime_tool,
    "currency_exchange": mock_currency_exchange_tool,
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "يحسب تعبيرات رياضية (جمع، طرح، ضرب، قسمة، أقواس)",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "التعبير الرياضي، مثل '150*3+20'"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "يُرجع التاريخ والوقت الحاليين، مفيد لحساب المواعيد النهائية والفترات الزمنية",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "currency_exchange",
            "description": "يُرجع سعر صرف عملة أجنبية مقابل الريال السعودي",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "اسم العملة، مثل 'USD' أو 'دولار'"}},
                "required": ["query"],
            },
        },
    },
]

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-58-chapter-58.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import time
from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class UserSession:
    """ذاكرة مبسّطة لكل مستخدم — مبنية على مبادئ الفصل 57 لكن مُصغَّرة لهذا المشروع"""
    user_id: str
    conversation_buffer: List[Dict[str, str]] = field(default_factory=list)
    known_facts: List[str] = field(default_factory=list)
    last_active: float = field(default_factory=time.time)


class MemoryStore:
    """مخزن ذاكرة متعدد المستخدمين — عزل كامل بالـ user_id (كما نصحنا في الفصل 57)"""

    def __init__(self):
        self._sessions: Dict[str, UserSession] = {}

    def get_or_create(self, user_id: str) -> UserSession:
        if user_id not in self._sessions:
            self._sessions[user_id] = UserSession(user_id=user_id)
        return self._sessions[user_id]

    def add_turn(self, user_id: str, user_msg: str, assistant_msg: str):
        session = self.get_or_create(user_id)
        session.conversation_buffer.append({"role": "user", "content": user_msg})
        session.conversation_buffer.append({"role": "assistant", "content": assistant_msg})
        session.conversation_buffer = session.conversation_buffer[-10:]  # آخر 5 تبادلات فقط
        session.last_active = time.time()

    def add_fact(self, user_id: str, fact: str):
        session = self.get_or_create(user_id)
        if fact not in session.known_facts:
            session.known_facts.append(fact)

    def get_context(self, user_id: str) -> str:
        session = self.get_or_create(user_id)
        facts_text = "\n".join(f"- {f}" for f in session.known_facts) or "لا توجد معلومات مسجّلة بعد"
        history_text = "\n".join(f"{m['role']}: {m['content']}" for m in session.conversation_buffer) or "محادثة جديدة"
        return f"[حقائق معروفة عن المستخدم]:\n{facts_text}\n\n[آخر المحادثة]:\n{history_text}"

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-58-chapter-58.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BLOCKED_PATTERNS = ["كلمة مرور", "password", "رقم بطاقة", "credit card"]


def check_input_safety(user_input: str) -> tuple[bool, str]:
    """فحص أولي سريع بدون LLM — يمنع طلبات واضحة الخطورة فورًا بتكلفة صفر"""
    lower_input = user_input.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern in lower_input:
            return False, "عذرًا، لا يمكنني معالجة طلبات تتعلق بمعلومات حساسة كهذه."
    return True, ""


def check_output_groundedness(answer: str, retrieved_context: str) -> tuple[bool, str]:
    """فحص: هل الإجابة المبنية على RAG مدعومة فعلًا بالسياق المسترجَع؟ (مانع هلوسة أخير)"""
    if not retrieved_context.strip():
        return True, answer  # لا يوجد سياق RAG ليُفحص أصلًا (إجابة مباشرة أو أداة)

    prompt = f"""هل الإجابة التالية مدعومة فعليًا بالسياق المرفق، أم تحتوي على ادعاءات غير موجودة فيه؟
أجب بكلمة واحدة فقط: "مدعومة" أو "غير_مدعومة".

[السياق]:
{retrieved_context}

[الإجابة]:
{answer}"""

    verdict = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    ).choices[0].message.content.strip()

    if "غير_مدعومة" in verdict:
        safe_answer = "عذرًا، لا أملك معلومات كافية موثوقة في وثائق الشركة للإجابة على هذا بدقة. أنصح بالتواصل مع قسم الموارد البشرية مباشرة."
        return False, safe_answer
    return True, answer

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-58-chapter-58.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import json
from typing import TypedDict, List, Optional, Literal
from openai import OpenAI
from langgraph.graph import StateGraph, END
import os

from rag_module import CompanyKnowledgeBase
from tools_module import TOOLS_REGISTRY, TOOLS_SCHEMA
from memory_module import MemoryStore
from guardrails_module import check_input_safety, check_output_groundedness

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class AgentState(TypedDict):
    user_id: str
    question: str
    route: Optional[Literal["rag", "tool", "direct"]]
    retrieved_context: str
    sources: List[str]
    tool_results: List[str]
    final_answer: str
    is_safe: bool
    cost_tokens: int


# ───────────────────── العقدة 1: التوجيه (Router) ─────────────────────
def route_question(state: AgentState) -> dict:
    system_prompt = """صنّف سؤال المستخدم إلى واحدة من ثلاث فئات بدقة:
- "rag": أسئلة عن سياسات الشركة الداخلية (إجازات، مصاريف، عمل عن بُعد)
- "tool": أسئلة تحتاج حسابًا رياضيًا، تاريخًا/وقتًا حاليًا، أو أسعار صرف عملات
- "direct": أسئلة عامة لا تحتاج أيًا مما سبق (تحيات، أسئلة عامة بسيطة)

أعد JSON فقط: {"route": "rag|tool|direct"}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": state["question"]}],
        temperature=0.0, response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    tokens = response.usage.total_tokens

    return {"route": result.get("route", "direct"), "cost_tokens": state.get("cost_tokens", 0) + tokens}


def routing_decision(state: AgentState) -> str:
    """دالة التوجيه الشرطي التي يستخدمها LangGraph لاختيار العقدة التالية"""
    return state["route"]


# ───────────────────── العقدة 2أ: استرجاع RAG ─────────────────────
def make_rag_node(kb: CompanyKnowledgeBase):
    def rag_retrieval(state: AgentState) -> dict:
        results = kb.search(state["question"], k=3)
        context = "\n\n".join(f"[من {r['source']}]: {r['text']}" for r in results)
        sources = list(set(r["source"] for r in results))
        return {"retrieved_context": context, "sources": sources}
    return rag_retrieval


# ───────────────────── العقدة 2ب: استدعاء الأدوات ─────────────────────
def tool_calling(state: AgentState) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": state["question"]}],
        tools=TOOLS_SCHEMA, tool_choice="auto", temperature=0.0,
    )
    message = response.choices[0].message
    tokens = response.usage.total_tokens
    tool_results = []

    if message.tool_calls:
        for call in message.tool_calls:
            func_name = call.function.name
            func_args = json.loads(call.function.arguments)
            result = TOOLS_REGISTRY[func_name](**func_args)
            tool_results.append(f"[أداة {func_name}]: {result}")

    return {"tool_results": tool_results, "cost_tokens": state.get("cost_tokens", 0) + tokens}


# ───────────────────── العقدة 3: التوليف النهائي ─────────────────────
def make_synthesis_node(memory: MemoryStore):
    def synthesize_answer(state: AgentState) -> dict:
        user_context = memory.get_context(state["user_id"])

        extra_info = ""
        if state.get("retrieved_context"):
            extra_info += f"\n[معلومات من وثائق الشركة]:\n{state['retrieved_context']}"
        if state.get("tool_results"):
            extra_info += "\n[نتائج أدوات]:\n" + "\n".join(state["tool_results"])

        system_prompt = f"""أنت مساعد موظفي TechCorp. أجب بدقة ووضوح واختصار.
إذا استخدمت معلومات من وثائق الشركة، اذكر ذلك صراحة. لا تختلق معلومات غير موجودة في السياق المرفق.

{user_context}
{extra_info}"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": state["question"]}],
            temperature=0.4,
        )
        answer = response.choices[0].message.content
        tokens = response.usage.total_tokens

        return {"final_answer": answer, "cost_tokens": state.get("cost_tokens", 0) + tokens}
    return synthesize_answer


# ───────────────────── العقدة 4: فحص الحماية النهائي ─────────────────────
def guardrail_node(state: AgentState) -> dict:
    is_grounded, safe_answer = check_output_groundedness(
        state["final_answer"], state.get("retrieved_context", "")
    )
    return {"final_answer": safe_answer, "is_safe": is_grounded}


# ───────────────────── بناء الرسم البياني الكامل ─────────────────────
def build_agent_graph(kb: CompanyKnowledgeBase, memory: MemoryStore):
    graph = StateGraph(AgentState)

    graph.add_node("router", route_question)
    graph.add_node("rag_retrieval", make_rag_node(kb))
    graph.add_node("tool_calling", tool_calling)
    graph.add_node("synthesis", make_synthesis_node(memory))
    graph.add_node("guardrail", guardrail_node)

    graph.set_entry_point("router")

    # التوجيه الشرطي: من router نذهب إلى واحد من ثلاثة مسارات
    graph.add_conditional_edges(
        "router", routing_decision,
        {"rag": "rag_retrieval", "tool": "tool_calling", "direct": "synthesis"},
    )

    graph.add_edge("rag_retrieval", "synthesis")
    graph.add_edge("tool_calling", "synthesis")
    graph.add_edge("synthesis", "guardrail")
    graph.add_edge("guardrail", END)

    return graph.compile()

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-58-chapter-58.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import logging

from rag_module import CompanyKnowledgeBase
from memory_module import MemoryStore
from graph import build_agent_graph
from guardrails_module import check_input_safety

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("techcorp_assistant")

app = FastAPI(title="TechCorp AI Assistant", version="1.0.0")

# تحميل مكوّن واحد فقط عند بدء التشغيل (وليس في كل طلب)
knowledge_base = CompanyKnowledgeBase(docs_dir="knowledge_base/")
memory_store = MemoryStore()
agent = build_agent_graph(knowledge_base, memory_store)


class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    route_used: str
    tokens_used: int
    latency_ms: float
    is_grounded: bool


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    start_time = time.time()

    is_safe, rejection_msg = check_input_safety(request.message)
    if not is_safe:
        return ChatResponse(answer=rejection_msg, sources=[], route_used="blocked",
                             tokens_used=0, latency_ms=0, is_grounded=True)

    try:
        result = agent.invoke({
            "user_id": request.user_id, "question": request.message,
            "route": None, "retrieved_context": "", "sources": [],
            "tool_results": [], "final_answer": "", "is_safe": True, "cost_tokens": 0,
        })
    except Exception as e:
        logger.error(f"خطأ في معالجة الطلب: {e}")
        raise HTTPException(status_code=500, detail="حدث خطأ داخلي، حاول مرة أخرى.")

    memory_store.add_turn(request.user_id, request.message, result["final_answer"])
    latency = (time.time() - start_time) * 1000

    logger.info(f"user={request.user_id} route={result['route']} "
                f"tokens={result['cost_tokens']} latency={latency:.0f}ms")

    return ChatResponse(
        answer=result["final_answer"], sources=result.get("sources", []),
        route_used=result["route"], tokens_used=result["cost_tokens"],
        latency_ms=round(latency, 1), is_grounded=result.get("is_safe", True),
    )


@app.get("/health")
def health_check():
    return {"status": "healthy", "knowledge_base_chunks": len(knowledge_base.chunks)}

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-58-chapter-58.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# test_integration.py
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_rag_route():
    """يجب أن يُوجَّه سؤال سياسة الإجازات إلى RAG ويعيد مصدرًا صحيحًا"""
    response = client.post("/chat", json={"user_id": "test_user_1", "message": "كم يوم إجازة سنوية يحق لي؟"})
    data = response.json()
    assert response.status_code == 200
    assert data["route_used"] == "rag"
    assert "vacation_policy.md" in data["sources"]
    assert "25" in data["answer"]


def test_tool_route():
    """يجب أن يُوجَّه سؤال حسابي إلى الأداة ويحسب بدقة"""
    response = client.post("/chat", json={"user_id": "test_user_2", "message": "كم ناتج 450 ضرب 3 زائد 100؟"})
    data = response.json()
    assert data["route_used"] == "tool"
    assert "1450" in data["answer"]


def test_memory_persistence():
    """يجب أن يتذكر النظام معلومة مذكورة في رسالة سابقة لنفس المستخدم"""
    client.post("/chat", json={"user_id": "test_user_3", "message": "اسمي سارة وأعمل في قسم الهندسة."})
    response = client.post("/chat", json={"user_id": "test_user_3", "message": "في أي قسم أعمل؟"})
    assert "هندسة" in response.json()["answer"]


def test_input_guardrail():
    """يجب رفض طلبات تحتوي على معلومات حساسة فورًا دون الوصول للـ LLM"""
    response = client.post("/chat", json={"user_id": "test_user_4", "message": "ما هي كلمة مرور النظام الداخلي؟"})
    data = response.json()
    assert data["route_used"] == "blocked"
    assert data["tokens_used"] == 0


if __name__ == "__main__":
    test_rag_route()
    test_tool_route()
    test_memory_persistence()
    test_input_guardrail()
    print("✅ جميع اختبارات التكامل نجحت!")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-58-chapter-58.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — مشروع تطبيقي — Agent متكامل يستخدم RAG + أدوات خارجية
Large Language Model Engineering — Agent

Book slug : llm-engineering-69-chapters
Chapter   : ch-58-chapter-58 (lesson 58)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_58.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 58: الفصل 58: مشروع تطبيقي — Agent متكامل يستخدم RAG + أدوات خارجية."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_agent_rag() -> None:
    knowledge = {'attention': 'Attention combines values using context-dependent weights.', 'tokenizer': 'A tokenizer converts text into discrete identifiers.'}
    query = 'attention'
    evidence = knowledge.get(query)
    state = {'step': 'retrieve', 'evidence': evidence, 'tool_calls': 1}
    answer = f'{evidence} [source:{query}]' if evidence else 'I do not have evidence.'
    print('workflow state:', state)
    print('grounded answer:', answer)

if __name__ == "__main__":
    print("Chapter 58 lab: Run a local retrieval-and-tool workflow")
    run_agent_rag()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-10-applications-agents/chapter-58.md
# ────────────────────────────────────────────────────────────────────────
import os
import glob
from dataclasses import dataclass
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


class CompanyKnowledgeBase:
    """فهرسة واسترجاع وثائق الشركة الداخلية — مبني على أساس الفصل 53"""

    def __init__(self, docs_dir: str, chunk_size: int = 300, chunk_overlap: int = 50):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.chunks: List[Chunk] = []
        self.index = None
        self._build_index(docs_dir, chunk_size, chunk_overlap)

    def _chunk_text(self, text: str, source: str, chunk_size: int, overlap: int) -> List[Chunk]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end])
            chunks.append(Chunk(text=chunk_text, source=source, chunk_id=len(self.chunks) + len(chunks)))
            start += chunk_size - overlap
        return chunks

    def _build_index(self, docs_dir: str, chunk_size: int, overlap: int):
        for filepath in glob.glob(os.path.join(docs_dir, "*.md")):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            source_name = os.path.basename(filepath)
            self.chunks.extend(self._chunk_text(text, source_name, chunk_size, overlap))

        if not self.chunks:
            raise ValueError(f"لم يتم العثور على أي وثائق في {docs_dir}")

        embeddings = self.embedder.encode(
            [c.text for c in self.chunks], normalize_embeddings=True, show_progress_bar=False
        )
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(np.array(embeddings).astype("float32"))
        print(f"✅ تم فهرسة {len(self.chunks)} قطعة نصية من {docs_dir}")

    def search(self, query: str, k: int = 3, min_score: float = 0.3) -> List[dict]:
        query_emb = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(query_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < min_score:
                continue
            chunk = self.chunks[idx]
            results.append({"text": chunk.text, "source": chunk.source, "score": float(score)})
        return results

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-10-applications-agents/chapter-58.md
# ────────────────────────────────────────────────────────────────────────
import json
import math
from datetime import datetime
from typing import Dict, Any


def calculator_tool(expression: str) -> str:
    """حاسبة آمنة — تقيّم تعبيرات رياضية فقط، لا تنفيذ كود عشوائي"""
    allowed_chars = set("0123456789+-*/(). ")
    if not all(c in allowed_chars for c in expression):
        return "خطأ: التعبير يحتوي على رموز غير مسموحة."
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return f"النتيجة: {result}"
    except Exception as e:
        return f"خطأ في الحساب: {str(e)}"


def get_current_datetime_tool(_: str = "") -> str:
    """يُرجع التاريخ والوقت الحاليين — مهم لأسئلة مرتبطة بالمواعيد النهائية"""
    now = datetime.now()
    return f"التاريخ والوقت الحاليان: {now.strftime('%Y-%m-%d %H:%M')} ({now.strftime('%A')})"


def mock_currency_exchange_tool(query: str) -> str:
    """محاكاة API خارجي لأسعار الصرف (في الإنتاج: استبدله بنداء API حقيقي كـ exchangerate-api.com)"""
    mock_rates = {"USD": 3.75, "EUR": 4.10, "GBP": 4.75}
    for currency, rate in mock_rates.items():
        if currency in query.upper():
            return f"سعر صرف {currency} الحالي: {rate} ريال سعودي (بيانات تجريبية)"
    return "لم أتمكن من التعرف على العملة المطلوبة. العملات المدعومة: USD, EUR, GBP"


TOOLS_REGISTRY: Dict[str, Any] = {
    "calculator": calculator_tool,
    "get_current_datetime": get_current_datetime_tool,
    "currency_exchange": mock_currency_exchange_tool,
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "يحسب تعبيرات رياضية (جمع، طرح، ضرب، قسمة، أقواس)",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "التعبير الرياضي، مثل '150*3+20'"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "يُرجع التاريخ والوقت الحاليين، مفيد لحساب المواعيد النهائية والفترات الزمنية",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "currency_exchange",
            "description": "يُرجع سعر صرف عملة أجنبية مقابل الريال السعودي",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "اسم العملة، مثل 'USD' أو 'دولار'"}},
                "required": ["query"],
            },
        },
    },
]

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-10-applications-agents/chapter-58.md
# ────────────────────────────────────────────────────────────────────────
import time
from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class UserSession:
    """ذاكرة مبسّطة لكل مستخدم — مبنية على مبادئ الفصل 57 لكن مُصغَّرة لهذا المشروع"""
    user_id: str
    conversation_buffer: List[Dict[str, str]] = field(default_factory=list)
    known_facts: List[str] = field(default_factory=list)
    last_active: float = field(default_factory=time.time)


class MemoryStore:
    """مخزن ذاكرة متعدد المستخدمين — عزل كامل بالـ user_id (كما نصحنا في الفصل 57)"""

    def __init__(self):
        self._sessions: Dict[str, UserSession] = {}

    def get_or_create(self, user_id: str) -> UserSession:
        if user_id not in self._sessions:
            self._sessions[user_id] = UserSession(user_id=user_id)
        return self._sessions[user_id]

    def add_turn(self, user_id: str, user_msg: str, assistant_msg: str):
        session = self.get_or_create(user_id)
        session.conversation_buffer.append({"role": "user", "content": user_msg})
        session.conversation_buffer.append({"role": "assistant", "content": assistant_msg})
        session.conversation_buffer = session.conversation_buffer[-10:]  # آخر 5 تبادلات فقط
        session.last_active = time.time()

    def add_fact(self, user_id: str, fact: str):
        session = self.get_or_create(user_id)
        if fact not in session.known_facts:
            session.known_facts.append(fact)

    def get_context(self, user_id: str) -> str:
        session = self.get_or_create(user_id)
        facts_text = "\n".join(f"- {f}" for f in session.known_facts) or "لا توجد معلومات مسجّلة بعد"
        history_text = "\n".join(f"{m['role']}: {m['content']}" for m in session.conversation_buffer) or "محادثة جديدة"
        return f"[حقائق معروفة عن المستخدم]:\n{facts_text}\n\n[آخر المحادثة]:\n{history_text}"

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-10-applications-agents/chapter-58.md
# ────────────────────────────────────────────────────────────────────────
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BLOCKED_PATTERNS = ["كلمة مرور", "password", "رقم بطاقة", "credit card"]


def check_input_safety(user_input: str) -> tuple[bool, str]:
    """فحص أولي سريع بدون LLM — يمنع طلبات واضحة الخطورة فورًا بتكلفة صفر"""
    lower_input = user_input.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern in lower_input:
            return False, "عذرًا، لا يمكنني معالجة طلبات تتعلق بمعلومات حساسة كهذه."
    return True, ""


def check_output_groundedness(answer: str, retrieved_context: str) -> tuple[bool, str]:
    """فحص: هل الإجابة المبنية على RAG مدعومة فعلًا بالسياق المسترجَع؟ (مانع هلوسة أخير)"""
    if not retrieved_context.strip():
        return True, answer  # لا يوجد سياق RAG ليُفحص أصلًا (إجابة مباشرة أو أداة)

    prompt = f"""هل الإجابة التالية مدعومة فعليًا بالسياق المرفق، أم تحتوي على ادعاءات غير موجودة فيه؟
أجب بكلمة واحدة فقط: "مدعومة" أو "غير_مدعومة".

[السياق]:
{retrieved_context}

[الإجابة]:
{answer}"""

    verdict = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    ).choices[0].message.content.strip()

    if "غير_مدعومة" in verdict:
        safe_answer = "عذرًا، لا أملك معلومات كافية موثوقة في وثائق الشركة للإجابة على هذا بدقة. أنصح بالتواصل مع قسم الموارد البشرية مباشرة."
        return False, safe_answer
    return True, answer

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-10-applications-agents/chapter-58.md
# ────────────────────────────────────────────────────────────────────────
import json
from typing import TypedDict, List, Optional, Literal
from openai import OpenAI
from langgraph.graph import StateGraph, END
import os

from rag_module import CompanyKnowledgeBase
from tools_module import TOOLS_REGISTRY, TOOLS_SCHEMA
from memory_module import MemoryStore
from guardrails_module import check_input_safety, check_output_groundedness

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class AgentState(TypedDict):
    user_id: str
    question: str
    route: Optional[Literal["rag", "tool", "direct"]]
    retrieved_context: str
    sources: List[str]
    tool_results: List[str]
    final_answer: str
    is_safe: bool
    cost_tokens: int


# ───────────────────── العقدة 1: التوجيه (Router) ─────────────────────
def route_question(state: AgentState) -> dict:
    system_prompt = """صنّف سؤال المستخدم إلى واحدة من ثلاث فئات بدقة:
- "rag": أسئلة عن سياسات الشركة الداخلية (إجازات، مصاريف، عمل عن بُعد)
- "tool": أسئلة تحتاج حسابًا رياضيًا، تاريخًا/وقتًا حاليًا، أو أسعار صرف عملات
- "direct": أسئلة عامة لا تحتاج أيًا مما سبق (تحيات، أسئلة عامة بسيطة)

أعد JSON فقط: {"route": "rag|tool|direct"}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": state["question"]}],
        temperature=0.0, response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    tokens = response.usage.total_tokens

    return {"route": result.get("route", "direct"), "cost_tokens": state.get("cost_tokens", 0) + tokens}


def routing_decision(state: AgentState) -> str:
    """دالة التوجيه الشرطي التي يستخدمها LangGraph لاختيار العقدة التالية"""
    return state["route"]


# ───────────────────── العقدة 2أ: استرجاع RAG ─────────────────────
def make_rag_node(kb: CompanyKnowledgeBase):
    def rag_retrieval(state: AgentState) -> dict:
        results = kb.search(state["question"], k=3)
        context = "\n\n".join(f"[من {r['source']}]: {r['text']}" for r in results)
        sources = list(set(r["source"] for r in results))
        return {"retrieved_context": context, "sources": sources}
    return rag_retrieval


# ───────────────────── العقدة 2ب: استدعاء الأدوات ─────────────────────
def tool_calling(state: AgentState) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": state["question"]}],
        tools=TOOLS_SCHEMA, tool_choice="auto", temperature=0.0,
    )
    message = response.choices[0].message
    tokens = response.usage.total_tokens
    tool_results = []

    if message.tool_calls:
        for call in message.tool_calls:
            func_name = call.function.name
            func_args = json.loads(call.function.arguments)
            result = TOOLS_REGISTRY[func_name](**func_args)
            tool_results.append(f"[أداة {func_name}]: {result}")

    return {"tool_results": tool_results, "cost_tokens": state.get("cost_tokens", 0) + tokens}


# ───────────────────── العقدة 3: التوليف النهائي ─────────────────────
def make_synthesis_node(memory: MemoryStore):
    def synthesize_answer(state: AgentState) -> dict:
        user_context = memory.get_context(state["user_id"])

        extra_info = ""
        if state.get("retrieved_context"):
            extra_info += f"\n[معلومات من وثائق الشركة]:\n{state['retrieved_context']}"
        if state.get("tool_results"):
            extra_info += "\n[نتائج أدوات]:\n" + "\n".join(state["tool_results"])

        system_prompt = f"""أنت مساعد موظفي TechCorp. أجب بدقة ووضوح واختصار.
إذا استخدمت معلومات من وثائق الشركة، اذكر ذلك صراحة. لا تختلق معلومات غير موجودة في السياق المرفق.

{user_context}
{extra_info}"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": state["question"]}],
            temperature=0.4,
        )
        answer = response.choices[0].message.content
        tokens = response.usage.total_tokens

        return {"final_answer": answer, "cost_tokens": state.get("cost_tokens", 0) + tokens}
    return synthesize_answer


# ───────────────────── العقدة 4: فحص الحماية النهائي ─────────────────────
def guardrail_node(state: AgentState) -> dict:
    is_grounded, safe_answer = check_output_groundedness(
        state["final_answer"], state.get("retrieved_context", "")
    )
    return {"final_answer": safe_answer, "is_safe": is_grounded}


# ───────────────────── بناء الرسم البياني الكامل ─────────────────────
def build_agent_graph(kb: CompanyKnowledgeBase, memory: MemoryStore):
    graph = StateGraph(AgentState)

    graph.add_node("router", route_question)
    graph.add_node("rag_retrieval", make_rag_node(kb))
    graph.add_node("tool_calling", tool_calling)
    graph.add_node("synthesis", make_synthesis_node(memory))
    graph.add_node("guardrail", guardrail_node)

    graph.set_entry_point("router")

    # التوجيه الشرطي: من router نذهب إلى واحد من ثلاثة مسارات
    graph.add_conditional_edges(
        "router", routing_decision,
        {"rag": "rag_retrieval", "tool": "tool_calling", "direct": "synthesis"},
    )

    graph.add_edge("rag_retrieval", "synthesis")
    graph.add_edge("tool_calling", "synthesis")
    graph.add_edge("synthesis", "guardrail")
    graph.add_edge("guardrail", END)

    return graph.compile()

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-10-applications-agents/chapter-58.md
# ────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import logging

from rag_module import CompanyKnowledgeBase
from memory_module import MemoryStore
from graph import build_agent_graph
from guardrails_module import check_input_safety

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("techcorp_assistant")

app = FastAPI(title="TechCorp AI Assistant", version="1.0.0")

# تحميل مكوّن واحد فقط عند بدء التشغيل (وليس في كل طلب)
knowledge_base = CompanyKnowledgeBase(docs_dir="knowledge_base/")
memory_store = MemoryStore()
agent = build_agent_graph(knowledge_base, memory_store)


class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    route_used: str
    tokens_used: int
    latency_ms: float
    is_grounded: bool


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    start_time = time.time()

    is_safe, rejection_msg = check_input_safety(request.message)
    if not is_safe:
        return ChatResponse(answer=rejection_msg, sources=[], route_used="blocked",
                             tokens_used=0, latency_ms=0, is_grounded=True)

    try:
        result = agent.invoke({
            "user_id": request.user_id, "question": request.message,
            "route": None, "retrieved_context": "", "sources": [],
            "tool_results": [], "final_answer": "", "is_safe": True, "cost_tokens": 0,
        })
    except Exception as e:
        logger.error(f"خطأ في معالجة الطلب: {e}")
        raise HTTPException(status_code=500, detail="حدث خطأ داخلي، حاول مرة أخرى.")

    memory_store.add_turn(request.user_id, request.message, result["final_answer"])
    latency = (time.time() - start_time) * 1000

    logger.info(f"user={request.user_id} route={result['route']} "
                f"tokens={result['cost_tokens']} latency={latency:.0f}ms")

    return ChatResponse(
        answer=result["final_answer"], sources=result.get("sources", []),
        route_used=result["route"], tokens_used=result["cost_tokens"],
        latency_ms=round(latency, 1), is_grounded=result.get("is_safe", True),
    )


@app.get("/health")
def health_check():
    return {"status": "healthy", "knowledge_base_chunks": len(knowledge_base.chunks)}

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-10-applications-agents/chapter-58.md
# ────────────────────────────────────────────────────────────────────────
# test_integration.py
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_rag_route():
    """يجب أن يُوجَّه سؤال سياسة الإجازات إلى RAG ويعيد مصدرًا صحيحًا"""
    response = client.post("/chat", json={"user_id": "test_user_1", "message": "كم يوم إجازة سنوية يحق لي؟"})
    data = response.json()
    assert response.status_code == 200
    assert data["route_used"] == "rag"
    assert "vacation_policy.md" in data["sources"]
    assert "25" in data["answer"]


def test_tool_route():
    """يجب أن يُوجَّه سؤال حسابي إلى الأداة ويحسب بدقة"""
    response = client.post("/chat", json={"user_id": "test_user_2", "message": "كم ناتج 450 ضرب 3 زائد 100؟"})
    data = response.json()
    assert data["route_used"] == "tool"
    assert "1450" in data["answer"]


def test_memory_persistence():
    """يجب أن يتذكر النظام معلومة مذكورة في رسالة سابقة لنفس المستخدم"""
    client.post("/chat", json={"user_id": "test_user_3", "message": "اسمي سارة وأعمل في قسم الهندسة."})
    response = client.post("/chat", json={"user_id": "test_user_3", "message": "في أي قسم أعمل؟"})
    assert "هندسة" in response.json()["answer"]


def test_input_guardrail():
    """يجب رفض طلبات تحتوي على معلومات حساسة فورًا دون الوصول للـ LLM"""
    response = client.post("/chat", json={"user_id": "test_user_4", "message": "ما هي كلمة مرور النظام الداخلي؟"})
    data = response.json()
    assert data["route_used"] == "blocked"
    assert data["tokens_used"] == 0


if __name__ == "__main__":
    test_rag_route()
    test_tool_route()
    test_memory_persistence()
    test_input_guardrail()
    print("✅ جميع اختبارات التكامل نجحت!")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-58-chapter-58.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 58: الفصل 58: مشروع تطبيقي — Agent متكامل يستخدم RAG + أدوات خارجية."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_agent_rag() -> None:
    knowledge = {'attention': 'Attention combines values using context-dependent weights.', 'tokenizer': 'A tokenizer converts text into discrete identifiers.'}
    query = 'attention'
    evidence = knowledge.get(query)
    state = {'step': 'retrieve', 'evidence': evidence, 'tool_calls': 1}
    answer = f'{evidence} [source:{query}]' if evidence else 'I do not have evidence.'
    print('workflow state:', state)
    print('grounded answer:', answer)

if __name__ == "__main__":
    print("Chapter 58 lab: Run a local retrieval-and-tool workflow")
    run_agent_rag()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-58-chapter-58.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import os
import glob
from dataclasses import dataclass
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


class CompanyKnowledgeBase:
    """فهرسة واسترجاع وثائق الشركة الداخلية — مبني على أساس الفصل 53"""

    def __init__(self, docs_dir: str, chunk_size: int = 300, chunk_overlap: int = 50):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.chunks: List[Chunk] = []
        self.index = None
        self._build_index(docs_dir, chunk_size, chunk_overlap)

    def _chunk_text(self, text: str, source: str, chunk_size: int, overlap: int) -> List[Chunk]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end])
            chunks.append(Chunk(text=chunk_text, source=source, chunk_id=len(self.chunks) + len(chunks)))
            start += chunk_size - overlap
        return chunks

    def _build_index(self, docs_dir: str, chunk_size: int, overlap: int):
        for filepath in glob.glob(os.path.join(docs_dir, "*.md")):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            source_name = os.path.basename(filepath)
            self.chunks.extend(self._chunk_text(text, source_name, chunk_size, overlap))

        if not self.chunks:
            raise ValueError(f"لم يتم العثور على أي وثائق في {docs_dir}")

        embeddings = self.embedder.encode(
            [c.text for c in self.chunks], normalize_embeddings=True, show_progress_bar=False
        )
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(np.array(embeddings).astype("float32"))
        print(f"✅ تم فهرسة {len(self.chunks)} قطعة نصية من {docs_dir}")

    def search(self, query: str, k: int = 3, min_score: float = 0.3) -> List[dict]:
        query_emb = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(query_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < min_score:
                continue
            chunk = self.chunks[idx]
            results.append({"text": chunk.text, "source": chunk.source, "score": float(score)})
        return results

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-58-chapter-58.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import json
import math
from datetime import datetime
from typing import Dict, Any


def calculator_tool(expression: str) -> str:
    """حاسبة آمنة — تقيّم تعبيرات رياضية فقط، لا تنفيذ كود عشوائي"""
    allowed_chars = set("0123456789+-*/(). ")
    if not all(c in allowed_chars for c in expression):
        return "خطأ: التعبير يحتوي على رموز غير مسموحة."
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return f"النتيجة: {result}"
    except Exception as e:
        return f"خطأ في الحساب: {str(e)}"


def get_current_datetime_tool(_: str = "") -> str:
    """يُرجع التاريخ والوقت الحاليين — مهم لأسئلة مرتبطة بالمواعيد النهائية"""
    now = datetime.now()
    return f"التاريخ والوقت الحاليان: {now.strftime('%Y-%m-%d %H:%M')} ({now.strftime('%A')})"


def mock_currency_exchange_tool(query: str) -> str:
    """محاكاة API خارجي لأسعار الصرف (في الإنتاج: استبدله بنداء API حقيقي كـ exchangerate-api.com)"""
    mock_rates = {"USD": 3.75, "EUR": 4.10, "GBP": 4.75}
    for currency, rate in mock_rates.items():
        if currency in query.upper():
            return f"سعر صرف {currency} الحالي: {rate} ريال سعودي (بيانات تجريبية)"
    return "لم أتمكن من التعرف على العملة المطلوبة. العملات المدعومة: USD, EUR, GBP"


TOOLS_REGISTRY: Dict[str, Any] = {
    "calculator": calculator_tool,
    "get_current_datetime": get_current_datetime_tool,
    "currency_exchange": mock_currency_exchange_tool,
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "يحسب تعبيرات رياضية (جمع، طرح، ضرب، قسمة، أقواس)",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "التعبير الرياضي، مثل '150*3+20'"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "يُرجع التاريخ والوقت الحاليين، مفيد لحساب المواعيد النهائية والفترات الزمنية",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "currency_exchange",
            "description": "يُرجع سعر صرف عملة أجنبية مقابل الريال السعودي",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "اسم العملة، مثل 'USD' أو 'دولار'"}},
                "required": ["query"],
            },
        },
    },
]

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-58-chapter-58.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import time
from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class UserSession:
    """ذاكرة مبسّطة لكل مستخدم — مبنية على مبادئ الفصل 57 لكن مُصغَّرة لهذا المشروع"""
    user_id: str
    conversation_buffer: List[Dict[str, str]] = field(default_factory=list)
    known_facts: List[str] = field(default_factory=list)
    last_active: float = field(default_factory=time.time)


class MemoryStore:
    """مخزن ذاكرة متعدد المستخدمين — عزل كامل بالـ user_id (كما نصحنا في الفصل 57)"""

    def __init__(self):
        self._sessions: Dict[str, UserSession] = {}

    def get_or_create(self, user_id: str) -> UserSession:
        if user_id not in self._sessions:
            self._sessions[user_id] = UserSession(user_id=user_id)
        return self._sessions[user_id]

    def add_turn(self, user_id: str, user_msg: str, assistant_msg: str):
        session = self.get_or_create(user_id)
        session.conversation_buffer.append({"role": "user", "content": user_msg})
        session.conversation_buffer.append({"role": "assistant", "content": assistant_msg})
        session.conversation_buffer = session.conversation_buffer[-10:]  # آخر 5 تبادلات فقط
        session.last_active = time.time()

    def add_fact(self, user_id: str, fact: str):
        session = self.get_or_create(user_id)
        if fact not in session.known_facts:
            session.known_facts.append(fact)

    def get_context(self, user_id: str) -> str:
        session = self.get_or_create(user_id)
        facts_text = "\n".join(f"- {f}" for f in session.known_facts) or "لا توجد معلومات مسجّلة بعد"
        history_text = "\n".join(f"{m['role']}: {m['content']}" for m in session.conversation_buffer) or "محادثة جديدة"
        return f"[حقائق معروفة عن المستخدم]:\n{facts_text}\n\n[آخر المحادثة]:\n{history_text}"

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-58-chapter-58.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BLOCKED_PATTERNS = ["كلمة مرور", "password", "رقم بطاقة", "credit card"]


def check_input_safety(user_input: str) -> tuple[bool, str]:
    """فحص أولي سريع بدون LLM — يمنع طلبات واضحة الخطورة فورًا بتكلفة صفر"""
    lower_input = user_input.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern in lower_input:
            return False, "عذرًا، لا يمكنني معالجة طلبات تتعلق بمعلومات حساسة كهذه."
    return True, ""


def check_output_groundedness(answer: str, retrieved_context: str) -> tuple[bool, str]:
    """فحص: هل الإجابة المبنية على RAG مدعومة فعلًا بالسياق المسترجَع؟ (مانع هلوسة أخير)"""
    if not retrieved_context.strip():
        return True, answer  # لا يوجد سياق RAG ليُفحص أصلًا (إجابة مباشرة أو أداة)

    prompt = f"""هل الإجابة التالية مدعومة فعليًا بالسياق المرفق، أم تحتوي على ادعاءات غير موجودة فيه؟
أجب بكلمة واحدة فقط: "مدعومة" أو "غير_مدعومة".

[السياق]:
{retrieved_context}

[الإجابة]:
{answer}"""

    verdict = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    ).choices[0].message.content.strip()

    if "غير_مدعومة" in verdict:
        safe_answer = "عذرًا، لا أملك معلومات كافية موثوقة في وثائق الشركة للإجابة على هذا بدقة. أنصح بالتواصل مع قسم الموارد البشرية مباشرة."
        return False, safe_answer
    return True, answer

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-58-chapter-58.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import json
from typing import TypedDict, List, Optional, Literal
from openai import OpenAI
from langgraph.graph import StateGraph, END
import os

from rag_module import CompanyKnowledgeBase
from tools_module import TOOLS_REGISTRY, TOOLS_SCHEMA
from memory_module import MemoryStore
from guardrails_module import check_input_safety, check_output_groundedness

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class AgentState(TypedDict):
    user_id: str
    question: str
    route: Optional[Literal["rag", "tool", "direct"]]
    retrieved_context: str
    sources: List[str]
    tool_results: List[str]
    final_answer: str
    is_safe: bool
    cost_tokens: int


# ───────────────────── العقدة 1: التوجيه (Router) ─────────────────────
def route_question(state: AgentState) -> dict:
    system_prompt = """صنّف سؤال المستخدم إلى واحدة من ثلاث فئات بدقة:
- "rag": أسئلة عن سياسات الشركة الداخلية (إجازات، مصاريف، عمل عن بُعد)
- "tool": أسئلة تحتاج حسابًا رياضيًا، تاريخًا/وقتًا حاليًا، أو أسعار صرف عملات
- "direct": أسئلة عامة لا تحتاج أيًا مما سبق (تحيات، أسئلة عامة بسيطة)

أعد JSON فقط: {"route": "rag|tool|direct"}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": state["question"]}],
        temperature=0.0, response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    tokens = response.usage.total_tokens

    return {"route": result.get("route", "direct"), "cost_tokens": state.get("cost_tokens", 0) + tokens}


def routing_decision(state: AgentState) -> str:
    """دالة التوجيه الشرطي التي يستخدمها LangGraph لاختيار العقدة التالية"""
    return state["route"]


# ───────────────────── العقدة 2أ: استرجاع RAG ─────────────────────
def make_rag_node(kb: CompanyKnowledgeBase):
    def rag_retrieval(state: AgentState) -> dict:
        results = kb.search(state["question"], k=3)
        context = "\n\n".join(f"[من {r['source']}]: {r['text']}" for r in results)
        sources = list(set(r["source"] for r in results))
        return {"retrieved_context": context, "sources": sources}
    return rag_retrieval


# ───────────────────── العقدة 2ب: استدعاء الأدوات ─────────────────────
def tool_calling(state: AgentState) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": state["question"]}],
        tools=TOOLS_SCHEMA, tool_choice="auto", temperature=0.0,
    )
    message = response.choices[0].message
    tokens = response.usage.total_tokens
    tool_results = []

    if message.tool_calls:
        for call in message.tool_calls:
            func_name = call.function.name
            func_args = json.loads(call.function.arguments)
            result = TOOLS_REGISTRY[func_name](**func_args)
            tool_results.append(f"[أداة {func_name}]: {result}")

    return {"tool_results": tool_results, "cost_tokens": state.get("cost_tokens", 0) + tokens}


# ───────────────────── العقدة 3: التوليف النهائي ─────────────────────
def make_synthesis_node(memory: MemoryStore):
    def synthesize_answer(state: AgentState) -> dict:
        user_context = memory.get_context(state["user_id"])

        extra_info = ""
        if state.get("retrieved_context"):
            extra_info += f"\n[معلومات من وثائق الشركة]:\n{state['retrieved_context']}"
        if state.get("tool_results"):
            extra_info += "\n[نتائج أدوات]:\n" + "\n".join(state["tool_results"])

        system_prompt = f"""أنت مساعد موظفي TechCorp. أجب بدقة ووضوح واختصار.
إذا استخدمت معلومات من وثائق الشركة، اذكر ذلك صراحة. لا تختلق معلومات غير موجودة في السياق المرفق.

{user_context}
{extra_info}"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": state["question"]}],
            temperature=0.4,
        )
        answer = response.choices[0].message.content
        tokens = response.usage.total_tokens

        return {"final_answer": answer, "cost_tokens": state.get("cost_tokens", 0) + tokens}
    return synthesize_answer


# ───────────────────── العقدة 4: فحص الحماية النهائي ─────────────────────
def guardrail_node(state: AgentState) -> dict:
    is_grounded, safe_answer = check_output_groundedness(
        state["final_answer"], state.get("retrieved_context", "")
    )
    return {"final_answer": safe_answer, "is_safe": is_grounded}


# ───────────────────── بناء الرسم البياني الكامل ─────────────────────
def build_agent_graph(kb: CompanyKnowledgeBase, memory: MemoryStore):
    graph = StateGraph(AgentState)

    graph.add_node("router", route_question)
    graph.add_node("rag_retrieval", make_rag_node(kb))
    graph.add_node("tool_calling", tool_calling)
    graph.add_node("synthesis", make_synthesis_node(memory))
    graph.add_node("guardrail", guardrail_node)

    graph.set_entry_point("router")

    # التوجيه الشرطي: من router نذهب إلى واحد من ثلاثة مسارات
    graph.add_conditional_edges(
        "router", routing_decision,
        {"rag": "rag_retrieval", "tool": "tool_calling", "direct": "synthesis"},
    )

    graph.add_edge("rag_retrieval", "synthesis")
    graph.add_edge("tool_calling", "synthesis")
    graph.add_edge("synthesis", "guardrail")
    graph.add_edge("guardrail", END)

    return graph.compile()

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-58-chapter-58.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import logging

from rag_module import CompanyKnowledgeBase
from memory_module import MemoryStore
from graph import build_agent_graph
from guardrails_module import check_input_safety

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("techcorp_assistant")

app = FastAPI(title="TechCorp AI Assistant", version="1.0.0")

# تحميل مكوّن واحد فقط عند بدء التشغيل (وليس في كل طلب)
knowledge_base = CompanyKnowledgeBase(docs_dir="knowledge_base/")
memory_store = MemoryStore()
agent = build_agent_graph(knowledge_base, memory_store)


class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    route_used: str
    tokens_used: int
    latency_ms: float
    is_grounded: bool


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    start_time = time.time()

    is_safe, rejection_msg = check_input_safety(request.message)
    if not is_safe:
        return ChatResponse(answer=rejection_msg, sources=[], route_used="blocked",
                             tokens_used=0, latency_ms=0, is_grounded=True)

    try:
        result = agent.invoke({
            "user_id": request.user_id, "question": request.message,
            "route": None, "retrieved_context": "", "sources": [],
            "tool_results": [], "final_answer": "", "is_safe": True, "cost_tokens": 0,
        })
    except Exception as e:
        logger.error(f"خطأ في معالجة الطلب: {e}")
        raise HTTPException(status_code=500, detail="حدث خطأ داخلي، حاول مرة أخرى.")

    memory_store.add_turn(request.user_id, request.message, result["final_answer"])
    latency = (time.time() - start_time) * 1000

    logger.info(f"user={request.user_id} route={result['route']} "
                f"tokens={result['cost_tokens']} latency={latency:.0f}ms")

    return ChatResponse(
        answer=result["final_answer"], sources=result.get("sources", []),
        route_used=result["route"], tokens_used=result["cost_tokens"],
        latency_ms=round(latency, 1), is_grounded=result.get("is_safe", True),
    )


@app.get("/health")
def health_check():
    return {"status": "healthy", "knowledge_base_chunks": len(knowledge_base.chunks)}

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-58-chapter-58.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# test_integration.py
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_rag_route():
    """يجب أن يُوجَّه سؤال سياسة الإجازات إلى RAG ويعيد مصدرًا صحيحًا"""
    response = client.post("/chat", json={"user_id": "test_user_1", "message": "كم يوم إجازة سنوية يحق لي؟"})
    data = response.json()
    assert response.status_code == 200
    assert data["route_used"] == "rag"
    assert "vacation_policy.md" in data["sources"]
    assert "25" in data["answer"]


def test_tool_route():
    """يجب أن يُوجَّه سؤال حسابي إلى الأداة ويحسب بدقة"""
    response = client.post("/chat", json={"user_id": "test_user_2", "message": "كم ناتج 450 ضرب 3 زائد 100؟"})
    data = response.json()
    assert data["route_used"] == "tool"
    assert "1450" in data["answer"]


def test_memory_persistence():
    """يجب أن يتذكر النظام معلومة مذكورة في رسالة سابقة لنفس المستخدم"""
    client.post("/chat", json={"user_id": "test_user_3", "message": "اسمي سارة وأعمل في قسم الهندسة."})
    response = client.post("/chat", json={"user_id": "test_user_3", "message": "في أي قسم أعمل؟"})
    assert "هندسة" in response.json()["answer"]


def test_input_guardrail():
    """يجب رفض طلبات تحتوي على معلومات حساسة فورًا دون الوصول للـ LLM"""
    response = client.post("/chat", json={"user_id": "test_user_4", "message": "ما هي كلمة مرور النظام الداخلي؟"})
    data = response.json()
    assert data["route_used"] == "blocked"
    assert data["tokens_used"] == 0


if __name__ == "__main__":
    test_rag_route()
    test_tool_route()
    test_memory_persistence()
    test_input_guardrail()
    print("✅ جميع اختبارات التكامل نجحت!")
