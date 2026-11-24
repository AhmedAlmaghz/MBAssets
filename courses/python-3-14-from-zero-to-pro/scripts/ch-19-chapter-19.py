#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
Python 3.14 — From Zero to Pro — API

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-19-chapter-19 (lesson 19)
Code files: 38 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/python-3-14-from-zero-to-pro/code/ch-19-chapter-19.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
Python 3.14 — From Zero to Pro — API

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-19-chapter-19 (lesson 19)
Code files: 37 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] dist/notebooks/chapter-19.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# تثبيت الحزم الناقصة فقط (سريع إن كانت مثبّتة أصلًا)
import importlib.util as _iu
_need = [m for m in ["fastapi", "httpx", "pydantic"] if _iu.find_spec(m) is None]
if _need:
    # %pip install -q fastapi httpx pydantic

# ────────────────────────────────────────────────────────────────────────
# [002] dist/notebooks/chapter-19.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "مرحبًا بك في واجهة API!"}

# ────────────────────────────────────────────────────────────────────────
# [003] dist/notebooks/chapter-19.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI

app = FastAPI()

@app.get("/tasks/{task_id}")
def get_task(task_id: int) -> dict[str, int]:
    return {"task_id": task_id}

# ────────────────────────────────────────────────────────────────────────
# [004] dist/notebooks/chapter-19.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
@app.get("/tasks")
def list_tasks(skip: int = 0, limit: int = 10) -> dict[str, int]:
    return {"skip": skip, "limit": limit}

# ────────────────────────────────────────────────────────────────────────
# [005] dist/notebooks/chapter-19.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
@app.get("/search")
def search_tasks(keyword: str | None = None) -> dict[str, str | None]:
    if keyword is None:
        return {"message": "لم يُحدد كلمة بحث"}
    return {"message": f"البحث عن: {keyword}"}

# ────────────────────────────────────────────────────────────────────────
# [006] dist/notebooks/chapter-19.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class TaskCreate(BaseModel):
    title: str
    priority: str = "متوسطة"
    done: bool = False

@app.post("/tasks")
def create_task(task: TaskCreate) -> TaskCreate:
    return task

# ────────────────────────────────────────────────────────────────────────
# [007] dist/notebooks/chapter-19.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from pydantic import BaseModel, Field

class TaskCreate(BaseModel):
    title: str = Field(min_length=1, max_length=100)
    priority: str = Field(default="متوسطة", pattern="^(منخفضة|متوسطة|عالية)$")
    estimated_hours: float = Field(default=1.0, gt=0, le=100)

# ────────────────────────────────────────────────────────────────────────
# [008] dist/notebooks/chapter-19.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
class TaskOut(BaseModel):
    id: int
    title: str
    priority: str
    done: bool

@app.post("/tasks", response_model=TaskOut)
def create_task(task: TaskCreate) -> TaskOut:
    new_id = 1  # لاحقًا سيأتي من قاعدة بيانات حقيقية
    return TaskOut(id=new_id, title=task.title, priority=task.priority, done=task.done)

# ────────────────────────────────────────────────────────────────────────
# [009] dist/notebooks/chapter-19.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException

app = FastAPI()

tasks_db: dict[int, str] = {1: "تعلم Python", 2: "تعلم FastAPI"}

@app.get("/tasks/{task_id}")
def get_task(task_id: int) -> dict[str, str]:
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="المهمة غير موجودة")
    return {"title": tasks_db[task_id]}

# ────────────────────────────────────────────────────────────────────────
# [010] dist/notebooks/chapter-19.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import asyncio

@app.get("/slow-operation")
async def slow_operation() -> dict[str, str]:
    await asyncio.sleep(2)  # محاكاة عملية بطيئة (استدعاء API خارجي، قاعدة بيانات، إلخ)
    return {"message": "اكتملت العملية البطيئة"}

# ────────────────────────────────────────────────────────────────────────
# [011] dist/notebooks/chapter-19.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# src/task_api/routers/tasks.py
from fastapi import APIRouter

router = APIRouter(prefix="/tasks", tags=["Tasks"])

@router.get("/")
def list_tasks() -> list[str]:
    return ["مهمة 1", "مهمة 2"]

# ────────────────────────────────────────────────────────────────────────
# [012] dist/notebooks/chapter-19.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI

app = FastAPI()

@app.get("/greet/{name}")
def greet(name: str) -> dict[str, str]:
    return {"message": f"مرحبًا، {name}!"}

@app.get("/calculate")
def calculate(a: int, b: int) -> dict[str, int]:
    return {"result": a + b}

# ────────────────────────────────────────────────────────────────────────
# [013] dist/notebooks/chapter-19.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

app = FastAPI()

class ItemCreate(BaseModel):
    name: str = Field(min_length=1)
    price: float = Field(gt=0)

@app.get("/items/{item_id}")
def get_item(item_id: int, q: str | None = None) -> dict:
    ...

@app.post("/items", status_code=status.HTTP_201_CREATED)
def create_item(item: ItemCreate) -> ItemCreate:
    ...

@app.patch("/items/{item_id}")
def update_item(item_id: int, item: ItemCreate) -> ItemCreate:
    ...

@app.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_item(item_id: int) -> None:
    ...

# تشغيل:
# uv run uvicorn main:app --reload
# التوثيق: http://127.0.0.1:8000/docs

# ────────────────────────────────────────────────────────────────────────
# [014] code/chapters/ch19/19-001-read_root.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.4: أول واجهة API: «مرحبًا أيها العالم»

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-001-read_root.py
"""

# main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "مرحبًا بك في واجهة API!"}

# ────────────────────────────────────────────────────────────────────────
# [015] code/chapters/ch19/19-002-get_task.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.5: المعاملات: Path Parameters و Query Parameters

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-002-get_task.py
"""

from fastapi import FastAPI

app = FastAPI()

@app.get("/tasks/{task_id}")
def get_task(task_id: int) -> dict[str, int]:
    return {"task_id": task_id}

# ────────────────────────────────────────────────────────────────────────
# [016] code/chapters/ch19/19-003-list_tasks.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.5: المعاملات: Path Parameters و Query Parameters

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-003-list_tasks.py
"""

@app.get("/tasks")
def list_tasks(skip: int = 0, limit: int = 10) -> dict[str, int]:
    return {"skip": skip, "limit": limit}

# ────────────────────────────────────────────────────────────────────────
# [017] code/chapters/ch19/19-004-search_tasks.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.5: المعاملات: Path Parameters و Query Parameters

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-004-search_tasks.py
"""

@app.get("/search")
def search_tasks(keyword: str | None = None) -> dict[str, str | None]:
    if keyword is None:
        return {"message": "لم يُحدد كلمة بحث"}
    return {"message": f"البحث عن: {keyword}"}

# ────────────────────────────────────────────────────────────────────────
# [018] code/chapters/ch19/19-005-create_task.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.6: Pydantic: نمذجة البيانات الواردة (Request Body)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-005-create_task.py
"""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class TaskCreate(BaseModel):
    title: str
    priority: str = "متوسطة"
    done: bool = False

@app.post("/tasks")
def create_task(task: TaskCreate) -> TaskCreate:
    return task

# ────────────────────────────────────────────────────────────────────────
# [019] code/chapters/ch19/19-006-task_create.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.6: Pydantic: نمذجة البيانات الواردة (Request Body)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-006-task_create.py
"""

from pydantic import BaseModel, Field

class TaskCreate(BaseModel):
    title: str = Field(min_length=1, max_length=100)
    priority: str = Field(default="متوسطة", pattern="^(منخفضة|متوسطة|عالية)$")
    estimated_hours: float = Field(default=1.0, gt=0, le=100)

# ────────────────────────────────────────────────────────────────────────
# [020] code/chapters/ch19/19-007-create_task.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.6: Pydantic: نمذجة البيانات الواردة (Request Body)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-007-create_task.py
"""

class TaskOut(BaseModel):
    id: int
    title: str
    priority: str
    done: bool

@app.post("/tasks", response_model=TaskOut)
def create_task(task: TaskCreate) -> TaskOut:
    new_id = 1  # لاحقًا سيأتي من قاعدة بيانات حقيقية
    return TaskOut(id=new_id, title=task.title, priority=task.priority, done=task.done)

# ────────────────────────────────────────────────────────────────────────
# [021] code/chapters/ch19/19-008-get_task.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.7: معالجة الأخطاء: `HTTPException`

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-008-get_task.py
"""

from fastapi import FastAPI, HTTPException

app = FastAPI()

tasks_db: dict[int, str] = {1: "تعلم Python", 2: "تعلم FastAPI"}

@app.get("/tasks/{task_id}")
def get_task(task_id: int) -> dict[str, str]:
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="المهمة غير موجودة")
    return {"title": tasks_db[task_id]}

# ────────────────────────────────────────────────────────────────────────
# [022] code/chapters/ch19/19-009-slow_operation.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.8: دعم async بشكل كامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-009-slow_operation.py
"""

import asyncio

@app.get("/slow-operation")
async def slow_operation() -> dict[str, str]:
    await asyncio.sleep(2)  # محاكاة عملية بطيئة (استدعاء API خارجي، قاعدة بيانات، إلخ)
    return {"message": "اكتملت العملية البطيئة"}

# ────────────────────────────────────────────────────────────────────────
# [023] code/chapters/ch19/19-010-list_tasks.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.9: تنظيم مشروع أكبر: `APIRouter`

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-010-list_tasks.py
"""

# src/task_api/routers/tasks.py
from fastapi import APIRouter

router = APIRouter(prefix="/tasks", tags=["Tasks"])

@router.get("/")
def list_tasks() -> list[str]:
    return ["مهمة 1", "مهمة 2"]

# ────────────────────────────────────────────────────────────────────────
# [024] code/chapters/ch19/19-011-app.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.9: تنظيم مشروع أكبر: `APIRouter`

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-011-app.py
"""

# src/task_api/main.py
from fastapi import FastAPI
from task_api.routers import tasks

app = FastAPI()
app.include_router(tasks.router)

# ────────────────────────────────────────────────────────────────────────
# [025] code/chapters/ch19/19-012-task_create.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.10: مشروع تطبيقي متكامل: واجهة API لإدارة المهام

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-012-task_create.py
"""

"""نماذج البيانات (Pydantic Models)."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

type Priority = Literal["منخفضة", "متوسطة", "عالية"]

class TaskCreate(BaseModel):
    """البيانات المطلوبة لإنشاء مهمة جديدة."""
    title: str = Field(min_length=1, max_length=200)
    priority: Priority = "متوسطة"

class TaskUpdate(BaseModel):
    """البيانات القابلة للتحديث في مهمة موجودة (كلها اختيارية)."""
    title: str | None = Field(default=None, min_length=1, max_length=200)
    priority: Priority | None = None
    done: bool | None = None

class Task(BaseModel):
    """نموذج المهمة الكامل كما يُخزَّن ويُرجَع للعميل."""
    id: int
    title: str
    priority: Priority
    done: bool = False

# ────────────────────────────────────────────────────────────────────────
# [026] code/chapters/ch19/19-013-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.10: مشروع تطبيقي متكامل: واجهة API لإدارة المهام

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-013-__init__.py
"""

"""طبقة تخزين بسيطة تعتمد على ملف JSON (كما تعلمنا في الفصل 16)."""
from __future__ import annotations

import json
from pathlib import Path

from task_api.models import Task

class TaskStorage:
    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self._tasks: dict[int, Task] = {}
        self._next_id: int = 1
        self._load()

    def _load(self) -> None:
        if not self.file_path.exists():
            return
        try:
            raw_data = json.loads(self.file_path.read_text(encoding="utf-8"))
            self._tasks = {item["id"]: Task(**item) for item in raw_data}
            if self._tasks:
                self._next_id = max(self._tasks.keys()) + 1
        except (json.JSONDecodeError, KeyError):
            self._tasks = {}

    def _save(self) -> None:
        data = [task.model_dump() for task in self._tasks.values()]
        self.file_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_all(self) -> list[Task]:
        return list(self._tasks.values())

    def get_by_id(self, task_id: int) -> Task | None:
        return self._tasks.get(task_id)

    def create(self, title: str, priority: str) -> Task:
        task = Task(id=self._next_id, title=title, priority=priority, done=False)  # type: ignore[arg-type]
        self._tasks[task.id] = task
        self._next_id += 1
        self._save()
        return task

    def update(self, task_id: int, **changes: object) -> Task | None:
        task = self._tasks.get(task_id)
        if task is None:
            return None
        updated_task = task.model_copy(update={k: v for k, v in changes.items() if v is not None})
        self._tasks[task_id] = updated_task
        self._save()
        return updated_task

    def delete(self, task_id: int) -> bool:
        if task_id not in self._tasks:
            return False
        del self._tasks[task_id]
        self._save()
        return True

# نسخة واحدة مشتركة للتطبيق بالكامل (Singleton بسيط)
storage = TaskStorage(Path("tasks.json"))

# ────────────────────────────────────────────────────────────────────────
# [027] code/chapters/ch19/19-014-list_tasks.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.10: مشروع تطبيقي متكامل: واجهة API لإدارة المهام

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-014-list_tasks.py
"""

"""نقاط النهاية الخاصة بإدارة المهام."""
from fastapi import APIRouter, HTTPException, status

from task_api.models import Task, TaskCreate, TaskUpdate
from task_api.storage import storage

router = APIRouter(prefix="/tasks", tags=["Tasks"])

@router.get("/", response_model=list[Task])
def list_tasks(done: bool | None = None) -> list[Task]:
    """يُرجع كل المهام، مع إمكانية التصفية حسب حالة الإنجاز."""
    all_tasks = storage.get_all()
    if done is None:
        return all_tasks
    return [task for task in all_tasks if task.done == done]

@router.get("/{task_id}", response_model=Task)
def get_task(task_id: int) -> Task:
    """يُرجع مهمة واحدة بمعرّفها."""
    task = storage.get_by_id(task_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"لا توجد مهمة بالمعرّف {task_id}",
        )
    return task

@router.post("/", response_model=Task, status_code=status.HTTP_201_CREATED)
def create_task(task_data: TaskCreate) -> Task:
    """ينشئ مهمة جديدة."""
    return storage.create(title=task_data.title, priority=task_data.priority)

@router.patch("/{task_id}", response_model=Task)
def update_task(task_id: int, task_data: TaskUpdate) -> Task:
    """يُحدّث حقولًا معيّنة في مهمة موجودة."""
    updated = storage.update(
        task_id,
        title=task_data.title,
        priority=task_data.priority,
        done=task_data.done,
    )
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"لا توجد مهمة بالمعرّف {task_id}",
        )
    return updated

@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_task(task_id: int) -> None:
    """يحذف مهمة موجودة."""
    deleted = storage.delete(task_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"لا توجد مهمة بالمعرّف {task_id}",
        )

# ────────────────────────────────────────────────────────────────────────
# [028] code/chapters/ch19/19-015-read_root.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.10: مشروع تطبيقي متكامل: واجهة API لإدارة المهام

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-015-read_root.py
"""

"""نقطة الدخول الرئيسية لتطبيق FastAPI."""
from fastapi import FastAPI

from task_api.routers import tasks

app = FastAPI(
    title="واجهة API لإدارة المهام",
    description="مشروع تطبيقي من كتاب تعلّم Python",
    version="1.0.0",
)

app.include_router(tasks.router)

@app.get("/", tags=["Root"])
def read_root() -> dict[str, str]:
    return {"message": "مرحبًا بك في واجهة API لإدارة المهام! زُر /docs للتوثيق التفاعلي"}

# ────────────────────────────────────────────────────────────────────────
# [029] code/chapters/ch19/19-016-ex19_11.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.11: أفضل الممارسات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-016-ex19_11.py
"""

status.HTTP_200_OK          # نجاح عام
status.HTTP_201_CREATED     # بعد إنشاء مورد جديد (POST)
status.HTTP_204_NO_CONTENT  # نجاح بدون محتوى مُرجَع (DELETE)
status.HTTP_404_NOT_FOUND   # المورد غير موجود
status.HTTP_422_UNPROCESSABLE_ENTITY  # بيانات غير صالحة (يتم تلقائيًا من Pydantic)

# ────────────────────────────────────────────────────────────────────────
# [030] code/chapters/ch19/19-017-greet.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.12: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-017-greet.py
"""

from fastapi import FastAPI

app = FastAPI()

@app.get("/greet/{name}")
def greet(name: str) -> dict[str, str]:
    return {"message": f"مرحبًا، {name}!"}

@app.get("/calculate")
def calculate(a: int, b: int) -> dict[str, int]:
    return {"result": a + b}

# ────────────────────────────────────────────────────────────────────────
# [031] code/chapters/ch19/19-018-get_item.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.14: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-018-get_item.py
"""

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

app = FastAPI()

class ItemCreate(BaseModel):
    name: str = Field(min_length=1)
    price: float = Field(gt=0)

@app.get("/items/{item_id}")
def get_item(item_id: int, q: str | None = None) -> dict:
    ...

@app.post("/items", status_code=status.HTTP_201_CREATED)
def create_item(item: ItemCreate) -> ItemCreate:
    ...

@app.patch("/items/{item_id}")
def update_item(item_id: int, item: ItemCreate) -> ItemCreate:
    ...

@app.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_item(item_id: int) -> None:
    ...

# تشغيل:
# uv run uvicorn main:app --reload
# التوثيق: http://127.0.0.1:8000/docs

# ────────────────────────────────────────────────────────────────────────
# [032] dist/markdown/chapter-19.md
# ────────────────────────────────────────────────────────────────────────
# src/task_api/main.py
from fastapi import FastAPI
from task_api.routers import tasks

app = FastAPI()
app.include_router(tasks.router)

# ────────────────────────────────────────────────────────────────────────
# [033] dist/markdown/chapter-19.md
# ────────────────────────────────────────────────────────────────────────
"""نماذج البيانات (Pydantic Models)."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

type Priority = Literal["منخفضة", "متوسطة", "عالية"]

class TaskCreate(BaseModel):
    """البيانات المطلوبة لإنشاء مهمة جديدة."""
    title: str = Field(min_length=1, max_length=200)
    priority: Priority = "متوسطة"

class TaskUpdate(BaseModel):
    """البيانات القابلة للتحديث في مهمة موجودة (كلها اختيارية)."""
    title: str | None = Field(default=None, min_length=1, max_length=200)
    priority: Priority | None = None
    done: bool | None = None

class Task(BaseModel):
    """نموذج المهمة الكامل كما يُخزَّن ويُرجَع للعميل."""
    id: int
    title: str
    priority: Priority
    done: bool = False

# ────────────────────────────────────────────────────────────────────────
# [034] dist/markdown/chapter-19.md
# ────────────────────────────────────────────────────────────────────────
"""طبقة تخزين بسيطة تعتمد على ملف JSON (كما تعلمنا في الفصل 16)."""
from __future__ import annotations

import json
from pathlib import Path

from task_api.models import Task

class TaskStorage:
    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self._tasks: dict[int, Task] = {}
        self._next_id: int = 1
        self._load()

    def _load(self) -> None:
        if not self.file_path.exists():
            return
        try:
            raw_data = json.loads(self.file_path.read_text(encoding="utf-8"))
            self._tasks = {item["id"]: Task(**item) for item in raw_data}
            if self._tasks:
                self._next_id = max(self._tasks.keys()) + 1
        except (json.JSONDecodeError, KeyError):
            self._tasks = {}

    def _save(self) -> None:
        data = [task.model_dump() for task in self._tasks.values()]
        self.file_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_all(self) -> list[Task]:
        return list(self._tasks.values())

    def get_by_id(self, task_id: int) -> Task | None:
        return self._tasks.get(task_id)

    def create(self, title: str, priority: str) -> Task:
        task = Task(id=self._next_id, title=title, priority=priority, done=False)  # type: ignore[arg-type]
        self._tasks[task.id] = task
        self._next_id += 1
        self._save()
        return task

    def update(self, task_id: int, **changes: object) -> Task | None:
        task = self._tasks.get(task_id)
        if task is None:
            return None
        updated_task = task.model_copy(update={k: v for k, v in changes.items() if v is not None})
        self._tasks[task_id] = updated_task
        self._save()
        return updated_task

    def delete(self, task_id: int) -> bool:
        if task_id not in self._tasks:
            return False
        del self._tasks[task_id]
        self._save()
        return True

# نسخة واحدة مشتركة للتطبيق بالكامل (Singleton بسيط)
storage = TaskStorage(Path("tasks.json"))

# ────────────────────────────────────────────────────────────────────────
# [035] dist/markdown/chapter-19.md
# ────────────────────────────────────────────────────────────────────────
"""نقاط النهاية الخاصة بإدارة المهام."""
from fastapi import APIRouter, HTTPException, status

from task_api.models import Task, TaskCreate, TaskUpdate
from task_api.storage import storage

router = APIRouter(prefix="/tasks", tags=["Tasks"])

@router.get("/", response_model=list[Task])
def list_tasks(done: bool | None = None) -> list[Task]:
    """يُرجع كل المهام، مع إمكانية التصفية حسب حالة الإنجاز."""
    all_tasks = storage.get_all()
    if done is None:
        return all_tasks
    return [task for task in all_tasks if task.done == done]

@router.get("/{task_id}", response_model=Task)
def get_task(task_id: int) -> Task:
    """يُرجع مهمة واحدة بمعرّفها."""
    task = storage.get_by_id(task_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"لا توجد مهمة بالمعرّف {task_id}",
        )
    return task

@router.post("/", response_model=Task, status_code=status.HTTP_201_CREATED)
def create_task(task_data: TaskCreate) -> Task:
    """ينشئ مهمة جديدة."""
    return storage.create(title=task_data.title, priority=task_data.priority)

@router.patch("/{task_id}", response_model=Task)
def update_task(task_id: int, task_data: TaskUpdate) -> Task:
    """يُحدّث حقولًا معيّنة في مهمة موجودة."""
    updated = storage.update(
        task_id,
        title=task_data.title,
        priority=task_data.priority,
        done=task_data.done,
    )
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"لا توجد مهمة بالمعرّف {task_id}",
        )
    return updated

@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_task(task_id: int) -> None:
    """يحذف مهمة موجودة."""
    deleted = storage.delete(task_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"لا توجد مهمة بالمعرّف {task_id}",
        )

# ────────────────────────────────────────────────────────────────────────
# [036] dist/markdown/chapter-19.md
# ────────────────────────────────────────────────────────────────────────
"""نقطة الدخول الرئيسية لتطبيق FastAPI."""
from fastapi import FastAPI

from task_api.routers import tasks

app = FastAPI(
    title="واجهة API لإدارة المهام",
    description="مشروع تطبيقي من كتاب تعلّم Python",
    version="1.0.0",
)

app.include_router(tasks.router)

@app.get("/", tags=["Root"])
def read_root() -> dict[str, str]:
    return {"message": "مرحبًا بك في واجهة API لإدارة المهام! زُر /docs للتوثيق التفاعلي"}

# ────────────────────────────────────────────────────────────────────────
# [037] dist/markdown/chapter-19.md
# ────────────────────────────────────────────────────────────────────────
status.HTTP_200_OK          # نجاح عام
status.HTTP_201_CREATED     # بعد إنشاء مورد جديد (POST)
status.HTTP_204_NO_CONTENT  # نجاح بدون محتوى مُرجَع (DELETE)
status.HTTP_404_NOT_FOUND   # المورد غير موجود
status.HTTP_422_UNPROCESSABLE_ENTITY  # بيانات غير صالحة (يتم تلقائيًا من Pydantic)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# تثبيت الحزم الناقصة فقط (سريع إن كانت مثبّتة أصلًا)
import importlib.util as _iu
_need = [m for m in ["fastapi", "httpx", "pydantic"] if _iu.find_spec(m) is None]
if _need:
    # %pip install -q fastapi httpx pydantic

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "مرحبًا بك في واجهة API!"}

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI

app = FastAPI()

@app.get("/tasks/{task_id}")
def get_task(task_id: int) -> dict[str, int]:
    return {"task_id": task_id}

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
@app.get("/tasks")
def list_tasks(skip: int = 0, limit: int = 10) -> dict[str, int]:
    return {"skip": skip, "limit": limit}

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
@app.get("/search")
def search_tasks(keyword: str | None = None) -> dict[str, str | None]:
    if keyword is None:
        return {"message": "لم يُحدد كلمة بحث"}
    return {"message": f"البحث عن: {keyword}"}

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class TaskCreate(BaseModel):
    title: str
    priority: str = "متوسطة"
    done: bool = False

@app.post("/tasks")
def create_task(task: TaskCreate) -> TaskCreate:
    return task

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from pydantic import BaseModel, Field

class TaskCreate(BaseModel):
    title: str = Field(min_length=1, max_length=100)
    priority: str = Field(default="متوسطة", pattern="^(منخفضة|متوسطة|عالية)$")
    estimated_hours: float = Field(default=1.0, gt=0, le=100)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
class TaskOut(BaseModel):
    id: int
    title: str
    priority: str
    done: bool

@app.post("/tasks", response_model=TaskOut)
def create_task(task: TaskCreate) -> TaskOut:
    new_id = 1  # لاحقًا سيأتي من قاعدة بيانات حقيقية
    return TaskOut(id=new_id, title=task.title, priority=task.priority, done=task.done)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException

app = FastAPI()

tasks_db: dict[int, str] = {1: "تعلم Python", 2: "تعلم FastAPI"}

@app.get("/tasks/{task_id}")
def get_task(task_id: int) -> dict[str, str]:
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="المهمة غير موجودة")
    return {"title": tasks_db[task_id]}

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import asyncio

@app.get("/slow-operation")
async def slow_operation() -> dict[str, str]:
    await asyncio.sleep(2)  # محاكاة عملية بطيئة (استدعاء API خارجي، قاعدة بيانات، إلخ)
    return {"message": "اكتملت العملية البطيئة"}

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# src/task_api/routers/tasks.py
from fastapi import APIRouter

router = APIRouter(prefix="/tasks", tags=["Tasks"])

@router.get("/")
def list_tasks() -> list[str]:
    return ["مهمة 1", "مهمة 2"]

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI

app = FastAPI()

@app.get("/greet/{name}")
def greet(name: str) -> dict[str, str]:
    return {"message": f"مرحبًا، {name}!"}

@app.get("/calculate")
def calculate(a: int, b: int) -> dict[str, int]:
    return {"result": a + b}

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

app = FastAPI()

class ItemCreate(BaseModel):
    name: str = Field(min_length=1)
    price: float = Field(gt=0)

@app.get("/items/{item_id}")
def get_item(item_id: int, q: str | None = None) -> dict:
    ...

@app.post("/items", status_code=status.HTTP_201_CREATED)
def create_item(item: ItemCreate) -> ItemCreate:
    ...

@app.patch("/items/{item_id}")
def update_item(item_id: int, item: ItemCreate) -> ItemCreate:
    ...

@app.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_item(item_id: int) -> None:
    ...

# تشغيل:
# uv run uvicorn main:app --reload
# التوثيق: http://127.0.0.1:8000/docs

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.4: أول واجهة API: «مرحبًا أيها العالم»

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-001-read_root.py
"""

# main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "مرحبًا بك في واجهة API!"}

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.5: المعاملات: Path Parameters و Query Parameters

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-002-get_task.py
"""

from fastapi import FastAPI

app = FastAPI()

@app.get("/tasks/{task_id}")
def get_task(task_id: int) -> dict[str, int]:
    return {"task_id": task_id}

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.5: المعاملات: Path Parameters و Query Parameters

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-003-list_tasks.py
"""

@app.get("/tasks")
def list_tasks(skip: int = 0, limit: int = 10) -> dict[str, int]:
    return {"skip": skip, "limit": limit}

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.5: المعاملات: Path Parameters و Query Parameters

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-004-search_tasks.py
"""

@app.get("/search")
def search_tasks(keyword: str | None = None) -> dict[str, str | None]:
    if keyword is None:
        return {"message": "لم يُحدد كلمة بحث"}
    return {"message": f"البحث عن: {keyword}"}

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.6: Pydantic: نمذجة البيانات الواردة (Request Body)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-005-create_task.py
"""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class TaskCreate(BaseModel):
    title: str
    priority: str = "متوسطة"
    done: bool = False

@app.post("/tasks")
def create_task(task: TaskCreate) -> TaskCreate:
    return task

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.6: Pydantic: نمذجة البيانات الواردة (Request Body)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-006-task_create.py
"""

from pydantic import BaseModel, Field

class TaskCreate(BaseModel):
    title: str = Field(min_length=1, max_length=100)
    priority: str = Field(default="متوسطة", pattern="^(منخفضة|متوسطة|عالية)$")
    estimated_hours: float = Field(default=1.0, gt=0, le=100)

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.6: Pydantic: نمذجة البيانات الواردة (Request Body)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-007-create_task.py
"""

class TaskOut(BaseModel):
    id: int
    title: str
    priority: str
    done: bool

@app.post("/tasks", response_model=TaskOut)
def create_task(task: TaskCreate) -> TaskOut:
    new_id = 1  # لاحقًا سيأتي من قاعدة بيانات حقيقية
    return TaskOut(id=new_id, title=task.title, priority=task.priority, done=task.done)

# ────────────────────────────────────────────────────────────────────────
# [022] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.7: معالجة الأخطاء: `HTTPException`

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-008-get_task.py
"""

from fastapi import FastAPI, HTTPException

app = FastAPI()

tasks_db: dict[int, str] = {1: "تعلم Python", 2: "تعلم FastAPI"}

@app.get("/tasks/{task_id}")
def get_task(task_id: int) -> dict[str, str]:
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="المهمة غير موجودة")
    return {"title": tasks_db[task_id]}

# ────────────────────────────────────────────────────────────────────────
# [023] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.8: دعم async بشكل كامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-009-slow_operation.py
"""

import asyncio

@app.get("/slow-operation")
async def slow_operation() -> dict[str, str]:
    await asyncio.sleep(2)  # محاكاة عملية بطيئة (استدعاء API خارجي، قاعدة بيانات، إلخ)
    return {"message": "اكتملت العملية البطيئة"}

# ────────────────────────────────────────────────────────────────────────
# [024] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.9: تنظيم مشروع أكبر: `APIRouter`

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-010-list_tasks.py
"""

# src/task_api/routers/tasks.py
from fastapi import APIRouter

router = APIRouter(prefix="/tasks", tags=["Tasks"])

@router.get("/")
def list_tasks() -> list[str]:
    return ["مهمة 1", "مهمة 2"]

# ────────────────────────────────────────────────────────────────────────
# [025] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.9: تنظيم مشروع أكبر: `APIRouter`

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-011-app.py
"""

# src/task_api/main.py
from fastapi import FastAPI
from task_api.routers import tasks

app = FastAPI()
app.include_router(tasks.router)

# ────────────────────────────────────────────────────────────────────────
# [026] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.10: مشروع تطبيقي متكامل: واجهة API لإدارة المهام

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-012-task_create.py
"""

"""نماذج البيانات (Pydantic Models)."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

type Priority = Literal["منخفضة", "متوسطة", "عالية"]

class TaskCreate(BaseModel):
    """البيانات المطلوبة لإنشاء مهمة جديدة."""
    title: str = Field(min_length=1, max_length=200)
    priority: Priority = "متوسطة"

class TaskUpdate(BaseModel):
    """البيانات القابلة للتحديث في مهمة موجودة (كلها اختيارية)."""
    title: str | None = Field(default=None, min_length=1, max_length=200)
    priority: Priority | None = None
    done: bool | None = None

class Task(BaseModel):
    """نموذج المهمة الكامل كما يُخزَّن ويُرجَع للعميل."""
    id: int
    title: str
    priority: Priority
    done: bool = False

# ────────────────────────────────────────────────────────────────────────
# [027] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-26
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.10: مشروع تطبيقي متكامل: واجهة API لإدارة المهام

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-013-__init__.py
"""

"""طبقة تخزين بسيطة تعتمد على ملف JSON (كما تعلمنا في الفصل 16)."""
from __future__ import annotations

import json
from pathlib import Path

from task_api.models import Task

class TaskStorage:
    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self._tasks: dict[int, Task] = {}
        self._next_id: int = 1
        self._load()

    def _load(self) -> None:
        if not self.file_path.exists():
            return
        try:
            raw_data = json.loads(self.file_path.read_text(encoding="utf-8"))
            self._tasks = {item["id"]: Task(**item) for item in raw_data}
            if self._tasks:
                self._next_id = max(self._tasks.keys()) + 1
        except (json.JSONDecodeError, KeyError):
            self._tasks = {}

    def _save(self) -> None:
        data = [task.model_dump() for task in self._tasks.values()]
        self.file_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_all(self) -> list[Task]:
        return list(self._tasks.values())

    def get_by_id(self, task_id: int) -> Task | None:
        return self._tasks.get(task_id)

    def create(self, title: str, priority: str) -> Task:
        task = Task(id=self._next_id, title=title, priority=priority, done=False)  # type: ignore[arg-type]
        self._tasks[task.id] = task
        self._next_id += 1
        self._save()
        return task

    def update(self, task_id: int, **changes: object) -> Task | None:
        task = self._tasks.get(task_id)
        if task is None:
            return None
        updated_task = task.model_copy(update={k: v for k, v in changes.items() if v is not None})
        self._tasks[task_id] = updated_task
        self._save()
        return updated_task

    def delete(self, task_id: int) -> bool:
        if task_id not in self._tasks:
            return False
        del self._tasks[task_id]
        self._save()
        return True

# نسخة واحدة مشتركة للتطبيق بالكامل (Singleton بسيط)
storage = TaskStorage(Path("tasks.json"))

# ────────────────────────────────────────────────────────────────────────
# [028] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-27
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.10: مشروع تطبيقي متكامل: واجهة API لإدارة المهام

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-014-list_tasks.py
"""

"""نقاط النهاية الخاصة بإدارة المهام."""
from fastapi import APIRouter, HTTPException, status

from task_api.models import Task, TaskCreate, TaskUpdate
from task_api.storage import storage

router = APIRouter(prefix="/tasks", tags=["Tasks"])

@router.get("/", response_model=list[Task])
def list_tasks(done: bool | None = None) -> list[Task]:
    """يُرجع كل المهام، مع إمكانية التصفية حسب حالة الإنجاز."""
    all_tasks = storage.get_all()
    if done is None:
        return all_tasks
    return [task for task in all_tasks if task.done == done]

@router.get("/{task_id}", response_model=Task)
def get_task(task_id: int) -> Task:
    """يُرجع مهمة واحدة بمعرّفها."""
    task = storage.get_by_id(task_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"لا توجد مهمة بالمعرّف {task_id}",
        )
    return task

@router.post("/", response_model=Task, status_code=status.HTTP_201_CREATED)
def create_task(task_data: TaskCreate) -> Task:
    """ينشئ مهمة جديدة."""
    return storage.create(title=task_data.title, priority=task_data.priority)

@router.patch("/{task_id}", response_model=Task)
def update_task(task_id: int, task_data: TaskUpdate) -> Task:
    """يُحدّث حقولًا معيّنة في مهمة موجودة."""
    updated = storage.update(
        task_id,
        title=task_data.title,
        priority=task_data.priority,
        done=task_data.done,
    )
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"لا توجد مهمة بالمعرّف {task_id}",
        )
    return updated

@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_task(task_id: int) -> None:
    """يحذف مهمة موجودة."""
    deleted = storage.delete(task_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"لا توجد مهمة بالمعرّف {task_id}",
        )

# ────────────────────────────────────────────────────────────────────────
# [029] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-28
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.10: مشروع تطبيقي متكامل: واجهة API لإدارة المهام

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-015-read_root.py
"""

"""نقطة الدخول الرئيسية لتطبيق FastAPI."""
from fastapi import FastAPI

from task_api.routers import tasks

app = FastAPI(
    title="واجهة API لإدارة المهام",
    description="مشروع تطبيقي من كتاب تعلّم Python",
    version="1.0.0",
)

app.include_router(tasks.router)

@app.get("/", tags=["Root"])
def read_root() -> dict[str, str]:
    return {"message": "مرحبًا بك في واجهة API لإدارة المهام! زُر /docs للتوثيق التفاعلي"}

# ────────────────────────────────────────────────────────────────────────
# [030] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-29
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.11: أفضل الممارسات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-016-ex19_11.py
"""

status.HTTP_200_OK          # نجاح عام
status.HTTP_201_CREATED     # بعد إنشاء مورد جديد (POST)
status.HTTP_204_NO_CONTENT  # نجاح بدون محتوى مُرجَع (DELETE)
status.HTTP_404_NOT_FOUND   # المورد غير موجود
status.HTTP_422_UNPROCESSABLE_ENTITY  # بيانات غير صالحة (يتم تلقائيًا من Pydantic)

# ────────────────────────────────────────────────────────────────────────
# [031] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-30
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.12: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-017-greet.py
"""

from fastapi import FastAPI

app = FastAPI()

@app.get("/greet/{name}")
def greet(name: str) -> dict[str, str]:
    return {"message": f"مرحبًا، {name}!"}

@app.get("/calculate")
def calculate(a: int, b: int) -> dict[str, int]:
    return {"result": a + b}

# ────────────────────────────────────────────────────────────────────────
# [032] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-31
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.14: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-018-get_item.py
"""

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

app = FastAPI()

class ItemCreate(BaseModel):
    name: str = Field(min_length=1)
    price: float = Field(gt=0)

@app.get("/items/{item_id}")
def get_item(item_id: int, q: str | None = None) -> dict:
    ...

@app.post("/items", status_code=status.HTTP_201_CREATED)
def create_item(item: ItemCreate) -> ItemCreate:
    ...

@app.patch("/items/{item_id}")
def update_item(item_id: int, item: ItemCreate) -> ItemCreate:
    ...

@app.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_item(item_id: int) -> None:
    ...

# تشغيل:
# uv run uvicorn main:app --reload
# التوثيق: http://127.0.0.1:8000/docs

# ────────────────────────────────────────────────────────────────────────
# [033] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-32
# ────────────────────────────────────────────────────────────────────────
# src/task_api/main.py
from fastapi import FastAPI
from task_api.routers import tasks

app = FastAPI()
app.include_router(tasks.router)

# ────────────────────────────────────────────────────────────────────────
# [034] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-33
# ────────────────────────────────────────────────────────────────────────
"""نماذج البيانات (Pydantic Models)."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

type Priority = Literal["منخفضة", "متوسطة", "عالية"]

class TaskCreate(BaseModel):
    """البيانات المطلوبة لإنشاء مهمة جديدة."""
    title: str = Field(min_length=1, max_length=200)
    priority: Priority = "متوسطة"

class TaskUpdate(BaseModel):
    """البيانات القابلة للتحديث في مهمة موجودة (كلها اختيارية)."""
    title: str | None = Field(default=None, min_length=1, max_length=200)
    priority: Priority | None = None
    done: bool | None = None

class Task(BaseModel):
    """نموذج المهمة الكامل كما يُخزَّن ويُرجَع للعميل."""
    id: int
    title: str
    priority: Priority
    done: bool = False

# ────────────────────────────────────────────────────────────────────────
# [035] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-34
# ────────────────────────────────────────────────────────────────────────
"""طبقة تخزين بسيطة تعتمد على ملف JSON (كما تعلمنا في الفصل 16)."""
from __future__ import annotations

import json
from pathlib import Path

from task_api.models import Task

class TaskStorage:
    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self._tasks: dict[int, Task] = {}
        self._next_id: int = 1
        self._load()

    def _load(self) -> None:
        if not self.file_path.exists():
            return
        try:
            raw_data = json.loads(self.file_path.read_text(encoding="utf-8"))
            self._tasks = {item["id"]: Task(**item) for item in raw_data}
            if self._tasks:
                self._next_id = max(self._tasks.keys()) + 1
        except (json.JSONDecodeError, KeyError):
            self._tasks = {}

    def _save(self) -> None:
        data = [task.model_dump() for task in self._tasks.values()]
        self.file_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_all(self) -> list[Task]:
        return list(self._tasks.values())

    def get_by_id(self, task_id: int) -> Task | None:
        return self._tasks.get(task_id)

    def create(self, title: str, priority: str) -> Task:
        task = Task(id=self._next_id, title=title, priority=priority, done=False)  # type: ignore[arg-type]
        self._tasks[task.id] = task
        self._next_id += 1
        self._save()
        return task

    def update(self, task_id: int, **changes: object) -> Task | None:
        task = self._tasks.get(task_id)
        if task is None:
            return None
        updated_task = task.model_copy(update={k: v for k, v in changes.items() if v is not None})
        self._tasks[task_id] = updated_task
        self._save()
        return updated_task

    def delete(self, task_id: int) -> bool:
        if task_id not in self._tasks:
            return False
        del self._tasks[task_id]
        self._save()
        return True

# نسخة واحدة مشتركة للتطبيق بالكامل (Singleton بسيط)
storage = TaskStorage(Path("tasks.json"))

# ────────────────────────────────────────────────────────────────────────
# [036] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-35
# ────────────────────────────────────────────────────────────────────────
"""نقاط النهاية الخاصة بإدارة المهام."""
from fastapi import APIRouter, HTTPException, status

from task_api.models import Task, TaskCreate, TaskUpdate
from task_api.storage import storage

router = APIRouter(prefix="/tasks", tags=["Tasks"])

@router.get("/", response_model=list[Task])
def list_tasks(done: bool | None = None) -> list[Task]:
    """يُرجع كل المهام، مع إمكانية التصفية حسب حالة الإنجاز."""
    all_tasks = storage.get_all()
    if done is None:
        return all_tasks
    return [task for task in all_tasks if task.done == done]

@router.get("/{task_id}", response_model=Task)
def get_task(task_id: int) -> Task:
    """يُرجع مهمة واحدة بمعرّفها."""
    task = storage.get_by_id(task_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"لا توجد مهمة بالمعرّف {task_id}",
        )
    return task

@router.post("/", response_model=Task, status_code=status.HTTP_201_CREATED)
def create_task(task_data: TaskCreate) -> Task:
    """ينشئ مهمة جديدة."""
    return storage.create(title=task_data.title, priority=task_data.priority)

@router.patch("/{task_id}", response_model=Task)
def update_task(task_id: int, task_data: TaskUpdate) -> Task:
    """يُحدّث حقولًا معيّنة في مهمة موجودة."""
    updated = storage.update(
        task_id,
        title=task_data.title,
        priority=task_data.priority,
        done=task_data.done,
    )
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"لا توجد مهمة بالمعرّف {task_id}",
        )
    return updated

@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_task(task_id: int) -> None:
    """يحذف مهمة موجودة."""
    deleted = storage.delete(task_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"لا توجد مهمة بالمعرّف {task_id}",
        )

# ────────────────────────────────────────────────────────────────────────
# [037] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-36
# ────────────────────────────────────────────────────────────────────────
"""نقطة الدخول الرئيسية لتطبيق FastAPI."""
from fastapi import FastAPI

from task_api.routers import tasks

app = FastAPI(
    title="واجهة API لإدارة المهام",
    description="مشروع تطبيقي من كتاب تعلّم Python",
    version="1.0.0",
)

app.include_router(tasks.router)

@app.get("/", tags=["Root"])
def read_root() -> dict[str, str]:
    return {"message": "مرحبًا بك في واجهة API لإدارة المهام! زُر /docs للتوثيق التفاعلي"}

# ────────────────────────────────────────────────────────────────────────
# [038] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-19-chapter-19.ipynb#cell-37
# ────────────────────────────────────────────────────────────────────────
status.HTTP_200_OK          # نجاح عام
status.HTTP_201_CREATED     # بعد إنشاء مورد جديد (POST)
status.HTTP_204_NO_CONTENT  # نجاح بدون محتوى مُرجَع (DELETE)
status.HTTP_404_NOT_FOUND   # المورد غير موجود
status.HTTP_422_UNPROCESSABLE_ENTITY  # بيانات غير صالحة (يتم تلقائيًا من Pydantic)
