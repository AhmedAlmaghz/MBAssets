#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — الملفات وتنسيقات البيانات
Python 3.14 — From Zero to Pro — Chapter 16 — Modern Python

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-16-chapter-16 (lesson 16)
Code files: 76 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/python-3-14-from-zero-to-pro/code/ch-16-chapter-16.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — الملفات وتنسيقات البيانات
Python 3.14 — From Zero to Pro — Chapter 16 — Modern Python

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-16-chapter-16 (lesson 16)
Code files: 75 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] dist/notebooks/chapter-16.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# مجلد عمل معزول حتى لا تتناثر الملفات في جذر الدفتر
WORK = pathlib.Path("book_runtime")
WORK.mkdir(exist_ok=True)
os.chdir(WORK)
print("مجلد العمل:", pathlib.Path.cwd().name)

# ملفات البداية التي يفترضها الفصل (تُكتب دائمًا لضمان نتيجة ثابتة)
pathlib.Path('notes.txt').write_text('ملاحظاتي اليومية\n- تعلّم open() و with\n- تذكّر إغلاق الملف\n', encoding='utf-8')

# ────────────────────────────────────────────────────────────────────────
# [002] dist/notebooks/chapter-16.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
file = open("notes.txt", "r", encoding="utf-8")
content = file.read()
print(content)
file.close()  # يجب أن نتذكر إغلاق الملف يدويًا!

# ────────────────────────────────────────────────────────────────────────
# [003] dist/notebooks/chapter-16.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
with open("notes.txt", "r", encoding="utf-8") as file:
    content: str = file.read()
    print(content)
# الملف يُغلق تلقائيًا هنا، حتى لو حدث خطأ بالداخل!

# ────────────────────────────────────────────────────────────────────────
# [004] dist/notebooks/chapter-16.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
with open("diary.txt", "w", encoding="utf-8") as file:
    file.write("اليوم الأول: بدأت تعلم Python.\n")
    file.write("كان يومًا ممتعًا جدًا!\n")

# ────────────────────────────────────────────────────────────────────────
# [005] dist/notebooks/chapter-16.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
with open("diary.txt", "a", encoding="utf-8") as file:
    file.write("اليوم الثاني: تعلمت عن الملفات!\n")

# ────────────────────────────────────────────────────────────────────────
# [006] dist/notebooks/chapter-16.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# الطريقة 1: قراءة الملف بالكامل كنص واحد
with open("diary.txt", "r", encoding="utf-8") as file:
    content: str = file.read()
    print(content)

# الطريقة 2: قراءة سطر واحد فقط
with open("diary.txt", "r", encoding="utf-8") as file:
    first_line: str = file.readline()
    print(first_line)  # "اليوم الأول: بدأت تعلم Python.\n"

# الطريقة 3: قراءة كل الأسطر كقائمة
with open("diary.txt", "r", encoding="utf-8") as file:
    lines: list[str] = file.readlines()
    print(lines)
    # ['اليوم الأول: ...\n', 'اليوم الثاني: ...\n', ...]

# الطريقة 4 (الأفضل للملفات الكبيرة): التكرار المباشر سطرًا بسطر
with open("diary.txt", "r", encoding="utf-8") as file:
    for line in file:
        print(line.strip())  # .strip() لإزالة \n الزائدة

# ────────────────────────────────────────────────────────────────────────
# [007] dist/notebooks/chapter-16.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
try:
    with open("missing_file.txt", "r", encoding="utf-8") as file:
        content = file.read()
except FileNotFoundError:
    print("الملف غير موجود!")
except PermissionError:
    print("لا تملك صلاحية الوصول لهذا الملف!")

# ────────────────────────────────────────────────────────────────────────
# [008] dist/notebooks/chapter-16.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from pathlib import Path

# مسار نسبي
notes_path = Path("notes.txt")

# مسار بمجلدات متعددة - لاحظ استخدام / كعامل دمج!
project_path = Path("data") / "2024" / "report.txt"
print(project_path)
# على لينكس/ماك: data/2024/report.txt
# على Windows: data\2024\report.txt  (يتكيف تلقائيًا!)

# ────────────────────────────────────────────────────────────────────────
# [009] dist/notebooks/chapter-16.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
from pathlib import Path

p = Path("data/2024/report.txt")

print(p.name)        # "report.txt" - اسم الملف كاملاً
print(p.stem)        # "report" - الاسم بدون الامتداد
print(p.suffix)      # ".txt" - الامتداد فقط
print(p.parent)      # "data/2024" - المجلد الأب
print(p.parts)       # ('data', '2024', 'report.txt') - كل الأجزاء

print(p.exists())    # هل المسار موجود فعليًا؟ True/False
print(p.is_file())   # هل هو ملف؟
print(p.is_dir())    # هل هو مجلد؟
print(p.absolute())  # المسار الكامل المطلق

# ────────────────────────────────────────────────────────────────────────
# [010] dist/notebooks/chapter-16.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
from pathlib import Path

data_dir = Path("data/2024")
data_dir.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────
# [011] dist/notebooks/chapter-16.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
from pathlib import Path

notes = Path("notes.txt")

# كتابة (تُنشئ الملف تلقائيًا إن لم يكن موجودًا، وتفتح وتغلق تلقائيًا)
notes.write_text("مرحبًا من pathlib!\n", encoding="utf-8")

# قراءة
content: str = notes.read_text(encoding="utf-8")
print(content)

# ────────────────────────────────────────────────────────────────────────
# [012] dist/notebooks/chapter-16.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
from pathlib import Path

project_dir = Path(".")  # المجلد الحالي

# كل الملفات والمجلدات مباشرة (دون تعمق)
for item in project_dir.iterdir():
    print(item)

# البحث عن كل ملفات .py في المجلد الحالي فقط
for py_file in project_dir.glob("*.py"):
    print(py_file)

# البحث التكراري في كل المجلدات الفرعية أيضًا
for py_file in project_dir.glob("**/*.py"):
    print(py_file)

# بديل أحدث وأوضح لنفس الغرض (Python 3.12+)
for py_file in project_dir.rglob("*.py"):
    print(py_file)

# ────────────────────────────────────────────────────────────────────────
# [013] dist/notebooks/chapter-16.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
from pathlib import Path
import shutil

def organize_downloads(downloads_path: Path) -> None:
    """يُنظّم الملفات في مجلد حسب امتدادها، كل نوع في مجلد فرعي خاص به."""
    extension_folders: dict[str, str] = {
        ".pdf": "مستندات",
        ".jpg": "صور",
        ".png": "صور",
        ".mp4": "فيديوهات",
        ".zip": "أرشيف",
    }

    for file_path in downloads_path.iterdir():
        if file_path.is_file():
            folder_name = extension_folders.get(file_path.suffix.lower())
            if folder_name is not None:
                target_dir = downloads_path / folder_name
                target_dir.mkdir(exist_ok=True)
                shutil.move(str(file_path), str(target_dir / file_path.name))
                print(f"نُقل {file_path.name} إلى {folder_name}/")

organize_downloads(Path.home() / "Downloads")

# ────────────────────────────────────────────────────────────────────────
# [014] dist/notebooks/chapter-16.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
import json

# ────────────────────────────────────────────────────────────────────────
# [015] dist/notebooks/chapter-16.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
import json

user_data: dict[str, object] = {
    "name": "أحمد",
    "age": 30,
    "is_active": True,
    "hobbies": ["قراءة", "برمجة"],
    "spouse": None,
}

# dumps (مع s) تُرجع نصًا (string) يمكن طباعته أو إرساله
json_string: str = json.dumps(user_data, ensure_ascii=False, indent=2)
print(json_string)

# ────────────────────────────────────────────────────────────────────────
# [016] dist/notebooks/chapter-16.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
import json
from pathlib import Path

user_data = {"name": "سارة", "age": 25, "city": "دبي"}

with open("user.json", "w", encoding="utf-8") as file:
    json.dump(user_data, file, ensure_ascii=False, indent=2)

# ────────────────────────────────────────────────────────────────────────
# [017] dist/notebooks/chapter-16.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
import json

json_text = '{"name": "خالد", "age": 28}'
data: dict[str, object] = json.loads(json_text)

print(data["name"])  # "خالد"
print(type(data))    # <class 'dict'>

# ────────────────────────────────────────────────────────────────────────
# [018] dist/notebooks/chapter-16.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
import json

with open("user.json", "r", encoding="utf-8") as file:
    data: dict[str, object] = json.load(file)

print(data["name"])  # "سارة"

# ────────────────────────────────────────────────────────────────────────
# [019] dist/notebooks/chapter-16.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
import json
from pathlib import Path

# كتابة
Path("user.json").write_text(
    json.dumps(user_data, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

# قراءة
data = json.loads(Path("user.json").read_text(encoding="utf-8"))

# ────────────────────────────────────────────────────────────────────────
# [020] dist/notebooks/chapter-16.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
import json

corrupted_json = '{"name": "أحمد", "age": }'  # صيغة خاطئة

try:
    data = json.loads(corrupted_json)
except json.JSONDecodeError as e:
    print(f"خطأ في تنسيق JSON: {e}")

# ────────────────────────────────────────────────────────────────────────
# [021] dist/notebooks/chapter-16.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass, asdict
import json

@dataclass
class Product:
    name: str
    price: float

product = Product("لابتوب", 1200.0)

product_dict: dict[str, object] = asdict(product)
json_string = json.dumps(product_dict, ensure_ascii=False, indent=2)
print(json_string)
# {
#   "name": "لابتوب",
#   "price": 1200.0
# }

# ────────────────────────────────────────────────────────────────────────
# [022] dist/notebooks/chapter-16.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
data = json.loads(json_string)
restored_product = Product(**data)  # فك تغليف القاموس كمعاملات
print(restored_product)  # Product(name='لابتوب', price=1200.0)

# ────────────────────────────────────────────────────────────────────────
# [023] dist/notebooks/chapter-16.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass, asdict
from pathlib import Path
import json

@dataclass
class Task:
    title: str
    done: bool = False

def save_tasks(tasks: list[Task], file_path: Path) -> None:
    data = [asdict(task) for task in tasks]
    file_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def load_tasks(file_path: Path) -> list[Task]:
    if not file_path.exists():
        return []
    raw_data = json.loads(file_path.read_text(encoding="utf-8"))
    return [Task(**item) for item in raw_data]

tasks_file = Path("tasks.json")

tasks = [
    Task("تعلم JSON"),
    Task("تعلم CSV", done=True),
]
save_tasks(tasks, tasks_file)

loaded_tasks = load_tasks(tasks_file)
for task in loaded_tasks:
    status = "✅" if task.done else "⬜"
    print(f"{status} {task.title}")

# ────────────────────────────────────────────────────────────────────────
# [024] dist/notebooks/chapter-16.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
import csv

rows: list[list[str]] = [
    ["name", "age", "city"],
    ["أحمد", "30", "القاهرة"],
    ["سارة", "25", "دبي"],
]

with open("people.csv", "w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(rows)

# ────────────────────────────────────────────────────────────────────────
# [025] dist/notebooks/chapter-16.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
import csv

with open("people.csv", "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)  # كل صف عبارة عن list[str]
# ['name', 'age', 'city']
# ['أحمد', '30', 'القاهرة']
# ['سارة', '25', 'دبي']

# ────────────────────────────────────────────────────────────────────────
# [026] dist/notebooks/chapter-16.ipynb#cell-26
# ────────────────────────────────────────────────────────────────────────
import csv

# القراءة بـ DictReader
with open("people.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(row)
        # {'name': 'أحمد', 'age': '30', 'city': 'القاهرة'}
        print(f"{row['name']} يعيش في {row['city']}")

# ────────────────────────────────────────────────────────────────────────
# [027] dist/notebooks/chapter-16.ipynb#cell-27
# ────────────────────────────────────────────────────────────────────────
import csv

# الكتابة بـ DictWriter
people: list[dict[str, str]] = [
    {"name": "منى", "age": "22", "city": "عمّان"},
    {"name": "طارق", "age": "35", "city": "بيروت"},
]

with open("people2.csv", "w", encoding="utf-8", newline="") as file:
    fieldnames = ["name", "age", "city"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()   # يكتب صف العناوين تلقائيًا
    writer.writerows(people)

# ────────────────────────────────────────────────────────────────────────
# [028] dist/notebooks/chapter-16.ipynb#cell-28
# ────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass, asdict, fields
import csv
from pathlib import Path

@dataclass
class Employee:
    name: str
    department: str
    salary: float

def export_to_csv(employees: list[Employee], file_path: Path) -> None:
    if not employees:
        return
    fieldnames = [f.name for f in fields(Employee)]
    with open(file_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for emp in employees:
            writer.writerow(asdict(emp))

employees = [
    Employee("ياسمين", "تسويق", 6000.0),
    Employee("عمر", "هندسة", 8500.0),
]

export_to_csv(employees, Path("employees.csv"))

# ────────────────────────────────────────────────────────────────────────
# [029] dist/notebooks/chapter-16.ipynb#cell-29
# ────────────────────────────────────────────────────────────────────────
from pathlib import Path

config_path = Path("config.json")

if config_path.exists():
    config = json.loads(config_path.read_text(encoding="utf-8"))
else:
    config = {}  # قيمة افتراضية آمنة

# ────────────────────────────────────────────────────────────────────────
# [030] dist/notebooks/chapter-16.ipynb#cell-30
# ────────────────────────────────────────────────────────────────────────
# ✅ بنية جيدة: دوال واضحة ومسؤولة عن مهمة واحدة
def load_data(path: Path) -> list[dict[str, object]]:
    ...

def save_data(data: list[dict[str, object]], path: Path) -> None:
    ...

def process_data(data: list[dict[str, object]]) -> list[dict[str, object]]:
     # المنطق الفعلي منفصل تمامًا عن I/O

# ────────────────────────────────────────────────────────────────────────
# [031] dist/notebooks/chapter-16.ipynb#cell-31
# ────────────────────────────────────────────────────────────────────────
"""
مدير جهات الاتصال - مشروع تطبيقي شامل
يغطي: الملفات النصية، pathlib، JSON، CSV، ومعالجة الأخطاء
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Literal
import json
import csv

type ContactGroup = Literal["عائلة", "عمل", "أصدقاء", "أخرى"]

# ---------------------------------------------------------
# نموذج البيانات
# ---------------------------------------------------------
@dataclass
class Contact:
    name: str
    phone: str
    email: str = ""
    group: ContactGroup = "أخرى"

    def display(self) -> str:
        return f"📇 {self.name} | {self.phone} | {self.email} | {self.group}"

# ---------------------------------------------------------
# استثناءات مخصصة
# ---------------------------------------------------------
class ContactNotFoundError(Exception):
    """يُرفع عند عدم العثور على جهة اتصال بالاسم المطلوب."""

# ---------------------------------------------------------
# مدير جهات الاتصال
# ---------------------------------------------------------
class ContactManager:
    def __init__(self, storage_path: Path) -> None:
        self.storage_path = storage_path
        self._contacts: list[Contact] = []
        self._load()

    # ----- التحميل والحفظ -----
    def _load(self) -> None:
        """يحمّل جهات الاتصال من ملف JSON إن وُجد، وإلا يبدأ بقائمة فارغة."""
        if not self.storage_path.exists():
            self._contacts = []
            return

        try:
            raw_text = self.storage_path.read_text(encoding="utf-8")
            raw_data = json.loads(raw_text)
            self._contacts = [Contact(**item) for item in raw_data]
        except json.JSONDecodeError:
            print("⚠️ تحذير: ملف البيانات تالف. سيتم البدء بقائمة فارغة.")
            self._contacts = []
        except TypeError:
            print("⚠️ تحذير: بنية البيانات غير متوافقة. سيتم البدء بقائمة فارغة.")
            self._contacts = []

    def save(self) -> None:
        """يحفظ كل جهات الاتصال الحالية إلى ملف JSON."""
        data = [asdict(contact) for contact in self._contacts]
        self.storage_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ----- العمليات الأساسية -----
    def add_contact(
        self,
        name: str,
        phone: str,
        email: str = "",
        group: ContactGroup = "أخرى",
    ) -> Contact:
        contact = Contact(name=name, phone=phone, email=email, group=group)
        self._contacts.append(contact)
        self.save()
        return contact

    def find_by_name(self, name: str) -> Contact:
        for contact in self._contacts:
            if contact.name == name:
                return contact
        raise ContactNotFoundError(f"لا توجد جهة اتصال باسم '{name}'")

    def delete_contact(self, name: str) -> None:
        contact = self.find_by_name(name)  # يرفع ContactNotFoundError تلقائيًا إن لم يوجد
        self._contacts.remove(contact)
        self.save()

    def search(self, keyword: str) -> list[Contact]:
        keyword_lower = keyword.lower()
        return [
            c for c in self._contacts
            if keyword_lower in c.name.lower() or keyword_lower in c.phone
        ]

    def get_by_group(self, group: ContactGroup) -> list[Contact]:
        return [c for c in self._contacts if c.group == group]

    def all_contacts(self) -> list[Contact]:
        return self._contacts.copy()

    def count(self) -> int:
        return len(self._contacts)

    # ----- التصدير -----
    def export_to_csv(self, csv_path: Path) -> None:
        if not self._contacts:
            print("لا توجد جهات اتصال للتصدير.")
            return

        fieldnames = [f.name for f in fields(Contact)]
        with open(csv_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for contact in self._contacts:
                writer.writerow(asdict(contact))
        print(f"✅ تم تصدير {len(self._contacts)} جهة اتصال إلى {csv_path}")

    def import_from_csv(self, csv_path: Path) -> int:
        if not csv_path.exists():
            raise FileNotFoundError(f"ملف الاستيراد غير موجود: {csv_path}")

        imported_count = 0
        with open(csv_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                self._contacts.append(Contact(**row))  # type: ignore[arg-type]
                imported_count += 1
        self.save()
        return imported_count

# ---------------------------------------------------------
# برنامج تجريبي
# ---------------------------------------------------------
def main() -> None:
    manager = ContactManager(Path("contacts.json"))

    manager.add_contact("أحمد علي", "0501234567", "ahmed@example.com", "عمل")
    manager.add_contact("سارة محمد", "0559876543", group="عائلة")
    manager.add_contact("خالد يوسف", "0507654321", "khaled@example.com", "أصدقاء")

    print(f"إجمالي جهات الاتصال: {manager.count()}")
    print()

    print("كل جهات الاتصال:")
    for contact in manager.all_contacts():
        print(contact.display())
    print()

    print("البحث عن 'سارة':")
    for contact in manager.search("سارة"):
        print(contact.display())
    print()

    try:
        manager.find_by_name("غير موجود")
    except ContactNotFoundError as e:
        print(f"خطأ متوقع: {e}")
    print()

    manager.export_to_csv(Path("contacts_export.csv"))

if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [032] dist/notebooks/chapter-16.ipynb#cell-32
# ────────────────────────────────────────────────────────────────────────
# تمرين 4
import csv
from pathlib import Path

def calculate_total_sales(csv_path: Path) -> float:
    if not csv_path.exists():
        print("ملف المبيعات غير موجود!")
        return 0.0

    total = 0.0
    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            total += float(row["amount"])
    return total

print(calculate_total_sales(Path("sales.csv")))

# تمرين 5
from pathlib import Path
from datetime import datetime

class Logger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, "a", encoding="utf-8") as file:
            file.write(f"[{timestamp}] {message}\n")

logger = Logger(Path("app.log"))
logger.log("بدأ تشغيل التطبيق")
logger.log("تم تسجيل دخول مستخدم جديد")

# تمرين 6
import json
from pathlib import Path

def save_config(config: dict[str, object], path: Path) -> None:
    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def load_config(path: Path) -> dict[str, object]:
    default_config: dict[str, object] = {"theme": "light", "language": "ar"}
    if not path.exists():
        return default_config
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default_config

config = load_config(Path("config.json"))
print(config)
save_config({"theme": "dark", "language": "en"}, Path("config.json"))

# ────────────────────────────────────────────────────────────────────────
# [033] dist/notebooks/chapter-16.ipynb#cell-33
# ────────────────────────────────────────────────────────────────────────
from pathlib import Path
from datetime import datetime
import shutil

def backup_file(source: Path, backup_dir: Path) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"الملف الأصلي غير موجود: {source}")

    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{source.stem}_{timestamp}{source.suffix}"
    backup_path = backup_dir / backup_name

    shutil.copy2(source, backup_path)
    return backup_path

def list_backups(backup_dir: Path, original_name: str) -> list[Path]:
    if not backup_dir.exists():
        return []

    stem = Path(original_name).stem
    suffix = Path(original_name).suffix

    matches = list(backup_dir.glob(f"{stem}_*{suffix}"))
    return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)

def restore_latest(backup_dir: Path, original_name: str, destination: Path) -> bool:
    backups = list_backups(backup_dir, original_name)
    if not backups:
        return False

    latest_backup = backups[0]
    shutil.copy2(latest_backup, destination)
    return True

# تجربة
source_file = Path("notes.txt")
source_file.write_text("محتوى تجريبي أول", encoding="utf-8")

backup_dir = Path("backups")

backup_file(source_file, backup_dir)

import time
time.sleep(1)  # للتأكد من اختلاف الطابع الزمني بين النسختين

source_file.write_text("محتوى تجريبي ثانٍ - مُحدّث", encoding="utf-8")
backup_file(source_file, backup_dir)

backups = list_backups(backup_dir, "notes.txt")
print(f"عدد النسخ الاحتياطية: {len(backups)}")
for b in backups:
    print(f"  {b.name}")

success = restore_latest(backup_dir, "notes.txt", Path("restored_notes.txt"))
print(f"تمت الاستعادة بنجاح: {success}")
print(Path("restored_notes.txt").read_text(encoding="utf-8"))

# ────────────────────────────────────────────────────────────────────────
# [034] code/chapters/ch16/16-001-file.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.2: الملفات النصية: الأساسيات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-001-file.py
"""

file = open("notes.txt", "r", encoding="utf-8")
content = file.read()
print(content)
file.close()  # يجب أن نتذكر إغلاق الملف يدويًا!

# ────────────────────────────────────────────────────────────────────────
# [035] code/chapters/ch16/16-002-ex16_2.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.2: الملفات النصية: الأساسيات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-002-ex16_2.py
"""

with open("notes.txt", "r", encoding="utf-8") as file:
    content: str = file.read()
    print(content)
# الملف يُغلق تلقائيًا هنا، حتى لو حدث خطأ بالداخل!

# ────────────────────────────────────────────────────────────────────────
# [036] code/chapters/ch16/16-003-ex16_2.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.2: الملفات النصية: الأساسيات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-003-ex16_2.py
"""

open(file_path, mode, encoding="utf-8")

# ────────────────────────────────────────────────────────────────────────
# [037] code/chapters/ch16/16-004-ex16_2.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.2: الملفات النصية: الأساسيات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-004-ex16_2.py
"""

with open("diary.txt", "w", encoding="utf-8") as file:
    file.write("اليوم الأول: بدأت تعلم Python.\n")
    file.write("كان يومًا ممتعًا جدًا!\n")

# ────────────────────────────────────────────────────────────────────────
# [038] code/chapters/ch16/16-005-ex16_2.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.2: الملفات النصية: الأساسيات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-005-ex16_2.py
"""

with open("diary.txt", "a", encoding="utf-8") as file:
    file.write("اليوم الثاني: تعلمت عن الملفات!\n")

# ────────────────────────────────────────────────────────────────────────
# [039] code/chapters/ch16/16-006-ex16_2.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.2: الملفات النصية: الأساسيات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-006-ex16_2.py
"""

# الطريقة 1: قراءة الملف بالكامل كنص واحد
with open("diary.txt", "r", encoding="utf-8") as file:
    content: str = file.read()
    print(content)

# الطريقة 2: قراءة سطر واحد فقط
with open("diary.txt", "r", encoding="utf-8") as file:
    first_line: str = file.readline()
    print(first_line)  # "اليوم الأول: بدأت تعلم Python.\n"

# الطريقة 3: قراءة كل الأسطر كقائمة
with open("diary.txt", "r", encoding="utf-8") as file:
    lines: list[str] = file.readlines()
    print(lines)
    # ['اليوم الأول: ...\n', 'اليوم الثاني: ...\n', ...]

# الطريقة 4 (الأفضل للملفات الكبيرة): التكرار المباشر سطرًا بسطر
with open("diary.txt", "r", encoding="utf-8") as file:
    for line in file:
        print(line.strip())  # .strip() لإزالة \n الزائدة

# ────────────────────────────────────────────────────────────────────────
# [040] code/chapters/ch16/16-007-content.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.2: الملفات النصية: الأساسيات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-007-content.py
"""

# ❌ خطير: الاعتماد على الترميز الافتراضي لنظام التشغيل
with open("notes.txt", "r") as file:
    content = file.read()

# ✅ آمن: تحديد الترميز صراحة
with open("notes.txt", "r", encoding="utf-8") as file:
    content = file.read()

# ────────────────────────────────────────────────────────────────────────
# [041] code/chapters/ch16/16-008-content.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.2: الملفات النصية: الأساسيات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-008-content.py
"""

try:
    with open("missing_file.txt", "r", encoding="utf-8") as file:
        content = file.read()
except FileNotFoundError:
    print("الملف غير موجود!")
except PermissionError:
    print("لا تملك صلاحية الوصول لهذا الملف!")

# ────────────────────────────────────────────────────────────────────────
# [042] code/chapters/ch16/16-009-notes_path.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.3: `pathlib`: الأسلوب الحديث للتعامل مع المسارات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-009-notes_path.py
"""

from pathlib import Path

# مسار نسبي
notes_path = Path("notes.txt")

# مسار بمجلدات متعددة - لاحظ استخدام / كعامل دمج!
project_path = Path("data") / "2024" / "report.txt"
print(project_path)
# على لينكس/ماك: data/2024/report.txt
# على Windows: data\2024\report.txt  (يتكيف تلقائيًا!)

# ────────────────────────────────────────────────────────────────────────
# [043] code/chapters/ch16/16-010-ex16_3.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.3: `pathlib`: الأسلوب الحديث للتعامل مع المسارات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-010-ex16_3.py
"""

from pathlib import Path

p = Path("data/2024/report.txt")

print(p.name)        # "report.txt" - اسم الملف كاملاً
print(p.stem)        # "report" - الاسم بدون الامتداد
print(p.suffix)      # ".txt" - الامتداد فقط
print(p.parent)      # "data/2024" - المجلد الأب
print(p.parts)       # ('data', '2024', 'report.txt') - كل الأجزاء

print(p.exists())    # هل المسار موجود فعليًا؟ True/False
print(p.is_file())   # هل هو ملف؟
print(p.is_dir())    # هل هو مجلد؟
print(p.absolute())  # المسار الكامل المطلق

# ────────────────────────────────────────────────────────────────────────
# [044] code/chapters/ch16/16-011-data_dir.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.3: `pathlib`: الأسلوب الحديث للتعامل مع المسارات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-011-data_dir.py
"""

from pathlib import Path

data_dir = Path("data/2024")
data_dir.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────
# [045] code/chapters/ch16/16-012-notes.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.3: `pathlib`: الأسلوب الحديث للتعامل مع المسارات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-012-notes.py
"""

from pathlib import Path

notes = Path("notes.txt")

# كتابة (تُنشئ الملف تلقائيًا إن لم يكن موجودًا، وتفتح وتغلق تلقائيًا)
notes.write_text("مرحبًا من pathlib!\n", encoding="utf-8")

# قراءة
content: str = notes.read_text(encoding="utf-8")
print(content)

# ────────────────────────────────────────────────────────────────────────
# [046] code/chapters/ch16/16-013-project_dir.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.3: `pathlib`: الأسلوب الحديث للتعامل مع المسارات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-013-project_dir.py
"""

from pathlib import Path

project_dir = Path(".")  # المجلد الحالي

# كل الملفات والمجلدات مباشرة (دون تعمق)
for item in project_dir.iterdir():
    print(item)

# البحث عن كل ملفات .py في المجلد الحالي فقط
for py_file in project_dir.glob("*.py"):
    print(py_file)

# البحث التكراري في كل المجلدات الفرعية أيضًا
for py_file in project_dir.glob("**/*.py"):
    print(py_file)

# بديل أحدث وأوضح لنفس الغرض (Python 3.12+)
for py_file in project_dir.rglob("*.py"):
    print(py_file)

# ────────────────────────────────────────────────────────────────────────
# [047] code/chapters/ch16/16-014-organize_downloads.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.3: `pathlib`: الأسلوب الحديث للتعامل مع المسارات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-014-organize_downloads.py
"""

from pathlib import Path
import shutil

def organize_downloads(downloads_path: Path) -> None:
    """يُنظّم الملفات في مجلد حسب امتدادها، كل نوع في مجلد فرعي خاص به."""
    extension_folders: dict[str, str] = {
        ".pdf": "مستندات",
        ".jpg": "صور",
        ".png": "صور",
        ".mp4": "فيديوهات",
        ".zip": "أرشيف",
    }

    for file_path in downloads_path.iterdir():
        if file_path.is_file():
            folder_name = extension_folders.get(file_path.suffix.lower())
            if folder_name is not None:
                target_dir = downloads_path / folder_name
                target_dir.mkdir(exist_ok=True)
                shutil.move(str(file_path), str(target_dir / file_path.name))
                print(f"نُقل {file_path.name} إلى {folder_name}/")

organize_downloads(Path.home() / "Downloads")

# ────────────────────────────────────────────────────────────────────────
# [048] code/chapters/ch16/16-015-ex16_4.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.4: JSON: لغة تبادل البيانات العالمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-015-ex16_4.py
"""

import json

# ────────────────────────────────────────────────────────────────────────
# [049] code/chapters/ch16/16-016-ex16_4.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.4: JSON: لغة تبادل البيانات العالمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-016-ex16_4.py
"""

import json

user_data: dict[str, object] = {
    "name": "أحمد",
    "age": 30,
    "is_active": True,
    "hobbies": ["قراءة", "برمجة"],
    "spouse": None,
}

# dumps (مع s) تُرجع نصًا (string) يمكن طباعته أو إرساله
json_string: str = json.dumps(user_data, ensure_ascii=False, indent=2)
print(json_string)

# ────────────────────────────────────────────────────────────────────────
# [050] code/chapters/ch16/16-017-user_data.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.4: JSON: لغة تبادل البيانات العالمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-017-user_data.py
"""

import json
from pathlib import Path

user_data = {"name": "سارة", "age": 25, "city": "دبي"}

with open("user.json", "w", encoding="utf-8") as file:
    json.dump(user_data, file, ensure_ascii=False, indent=2)

# ────────────────────────────────────────────────────────────────────────
# [051] code/chapters/ch16/16-018-json_text.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.4: JSON: لغة تبادل البيانات العالمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-018-json_text.py
"""

import json

json_text = '{"name": "خالد", "age": 28}'
data: dict[str, object] = json.loads(json_text)

print(data["name"])  # "خالد"
print(type(data))    # <class 'dict'>

# ────────────────────────────────────────────────────────────────────────
# [052] code/chapters/ch16/16-019-ex16_4.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.4: JSON: لغة تبادل البيانات العالمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-019-ex16_4.py
"""

import json

with open("user.json", "r", encoding="utf-8") as file:
    data: dict[str, object] = json.load(file)

print(data["name"])  # "سارة"

# ────────────────────────────────────────────────────────────────────────
# [053] code/chapters/ch16/16-020-encoding.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.4: JSON: لغة تبادل البيانات العالمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-020-encoding.py
"""

import json
from pathlib import Path

# كتابة
Path("user.json").write_text(
    json.dumps(user_data, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

# قراءة
data = json.loads(Path("user.json").read_text(encoding="utf-8"))

# ────────────────────────────────────────────────────────────────────────
# [054] code/chapters/ch16/16-021-corrupted_json.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.4: JSON: لغة تبادل البيانات العالمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-021-corrupted_json.py
"""

import json

corrupted_json = '{"name": "أحمد", "age": }'  # صيغة خاطئة

try:
    data = json.loads(corrupted_json)
except json.JSONDecodeError as e:
    print(f"خطأ في تنسيق JSON: {e}")

# ────────────────────────────────────────────────────────────────────────
# [055] code/chapters/ch16/16-022-product.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.4: JSON: لغة تبادل البيانات العالمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-022-product.py
"""

from dataclasses import dataclass
import json

@dataclass
class Product:
    name: str
    price: float

product = Product("لابتوب", 1200.0)
json.dumps(product)  # ❌ TypeError: Object of type Product is not JSON serializable

# ────────────────────────────────────────────────────────────────────────
# [056] code/chapters/ch16/16-023-product.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.4: JSON: لغة تبادل البيانات العالمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-023-product.py
"""

from dataclasses import dataclass, asdict
import json

@dataclass
class Product:
    name: str
    price: float

product = Product("لابتوب", 1200.0)

product_dict: dict[str, object] = asdict(product)
json_string = json.dumps(product_dict, ensure_ascii=False, indent=2)
print(json_string)
# {
#   "name": "لابتوب",
#   "price": 1200.0
# }

# ────────────────────────────────────────────────────────────────────────
# [057] code/chapters/ch16/16-024-data.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.4: JSON: لغة تبادل البيانات العالمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-024-data.py
"""

data = json.loads(json_string)
restored_product = Product(**data)  # فك تغليف القاموس كمعاملات
print(restored_product)  # Product(name='لابتوب', price=1200.0)

# ────────────────────────────────────────────────────────────────────────
# [058] code/chapters/ch16/16-025-save_tasks.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.4: JSON: لغة تبادل البيانات العالمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-025-save_tasks.py
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import json

@dataclass
class Task:
    title: str
    done: bool = False

def save_tasks(tasks: list[Task], file_path: Path) -> None:
    data = [asdict(task) for task in tasks]
    file_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def load_tasks(file_path: Path) -> list[Task]:
    if not file_path.exists():
        return []
    raw_data = json.loads(file_path.read_text(encoding="utf-8"))
    return [Task(**item) for item in raw_data]

tasks_file = Path("tasks.json")

tasks = [
    Task("تعلم JSON"),
    Task("تعلم CSV", done=True),
]
save_tasks(tasks, tasks_file)

loaded_tasks = load_tasks(tasks_file)
for task in loaded_tasks:
    status = "✅" if task.done else "⬜"
    print(f"{status} {task.title}")

# ────────────────────────────────────────────────────────────────────────
# [059] code/chapters/ch16/16-026-writer.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.5: CSV: البيانات الجدولية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-026-writer.py
"""

import csv

rows: list[list[str]] = [
    ["name", "age", "city"],
    ["أحمد", "30", "القاهرة"],
    ["سارة", "25", "دبي"],
]

with open("people.csv", "w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(rows)

# ────────────────────────────────────────────────────────────────────────
# [060] code/chapters/ch16/16-027-reader.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.5: CSV: البيانات الجدولية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-027-reader.py
"""

import csv

with open("people.csv", "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)  # كل صف عبارة عن list[str]
# ['name', 'age', 'city']
# ['أحمد', '30', 'القاهرة']
# ['سارة', '25', 'دبي']

# ────────────────────────────────────────────────────────────────────────
# [061] code/chapters/ch16/16-028-reader.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.5: CSV: البيانات الجدولية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-028-reader.py
"""

import csv

# القراءة بـ DictReader
with open("people.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(row)
        # {'name': 'أحمد', 'age': '30', 'city': 'القاهرة'}
        print(f"{row['name']} يعيش في {row['city']}")

# ────────────────────────────────────────────────────────────────────────
# [062] code/chapters/ch16/16-029-fieldnames.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.5: CSV: البيانات الجدولية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-029-fieldnames.py
"""

import csv

# الكتابة بـ DictWriter
people: list[dict[str, str]] = [
    {"name": "منى", "age": "22", "city": "عمّان"},
    {"name": "طارق", "age": "35", "city": "بيروت"},
]

with open("people2.csv", "w", encoding="utf-8", newline="") as file:
    fieldnames = ["name", "age", "city"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()   # يكتب صف العناوين تلقائيًا
    writer.writerows(people)

# ────────────────────────────────────────────────────────────────────────
# [063] code/chapters/ch16/16-030-export_to_csv.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.5: CSV: البيانات الجدولية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-030-export_to_csv.py
"""

from dataclasses import dataclass, asdict, fields
import csv
from pathlib import Path

@dataclass
class Employee:
    name: str
    department: str
    salary: float

def export_to_csv(employees: list[Employee], file_path: Path) -> None:
    if not employees:
        return
    fieldnames = [f.name for f in fields(Employee)]
    with open(file_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for emp in employees:
            writer.writerow(asdict(emp))

employees = [
    Employee("ياسمين", "تسويق", 6000.0),
    Employee("عمر", "هندسة", 8500.0),
]

export_to_csv(employees, Path("employees.csv"))

# ────────────────────────────────────────────────────────────────────────
# [064] code/chapters/ch16/16-031-file_path.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.6: أفضل الممارسات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-031-file_path.py
"""

# ❌ تجنّب
file_path = "data" + "/" + "2024" + "/" + "report.txt"

# ✅ استخدم
from pathlib import Path
file_path = Path("data") / "2024" / "report.txt"

# ────────────────────────────────────────────────────────────────────────
# [065] code/chapters/ch16/16-032-config_path.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.6: أفضل الممارسات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-032-config_path.py
"""

from pathlib import Path

config_path = Path("config.json")

if config_path.exists():
    config = json.loads(config_path.read_text(encoding="utf-8"))
else:
    config = {}  # قيمة افتراضية آمنة

# ────────────────────────────────────────────────────────────────────────
# [066] code/chapters/ch16/16-033-load_data.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.6: أفضل الممارسات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-033-load_data.py
"""

# ✅ بنية جيدة: دوال واضحة ومسؤولة عن مهمة واحدة
def load_data(path: Path) -> list[dict[str, object]]:
    ...

def save_data(data: list[dict[str, object]], path: Path) -> None:
    ...

def process_data(data: list[dict[str, object]]) -> list[dict[str, object]]:
     # المنطق الفعلي منفصل تمامًا عن I/O

# ────────────────────────────────────────────────────────────────────────
# [067] code/chapters/ch16/16-034-display.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.7: مشروع تطبيقي: مدير جهات اتصال متكامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-034-display.py
"""

"""
مدير جهات الاتصال - مشروع تطبيقي شامل
يغطي: الملفات النصية، pathlib، JSON، CSV، ومعالجة الأخطاء
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Literal
import json
import csv

type ContactGroup = Literal["عائلة", "عمل", "أصدقاء", "أخرى"]

# ---------------------------------------------------------
# نموذج البيانات
# ---------------------------------------------------------
@dataclass
class Contact:
    name: str
    phone: str
    email: str = ""
    group: ContactGroup = "أخرى"

    def display(self) -> str:
        return f"📇 {self.name} | {self.phone} | {self.email} | {self.group}"

# ---------------------------------------------------------
# استثناءات مخصصة
# ---------------------------------------------------------
class ContactNotFoundError(Exception):
    """يُرفع عند عدم العثور على جهة اتصال بالاسم المطلوب."""

# ---------------------------------------------------------
# مدير جهات الاتصال
# ---------------------------------------------------------
class ContactManager:
    def __init__(self, storage_path: Path) -> None:
        self.storage_path = storage_path
        self._contacts: list[Contact] = []
        self._load()

    # ----- التحميل والحفظ -----
    def _load(self) -> None:
        """يحمّل جهات الاتصال من ملف JSON إن وُجد، وإلا يبدأ بقائمة فارغة."""
        if not self.storage_path.exists():
            self._contacts = []
            return

        try:
            raw_text = self.storage_path.read_text(encoding="utf-8")
            raw_data = json.loads(raw_text)
            self._contacts = [Contact(**item) for item in raw_data]
        except json.JSONDecodeError:
            print("⚠️ تحذير: ملف البيانات تالف. سيتم البدء بقائمة فارغة.")
            self._contacts = []
        except TypeError:
            print("⚠️ تحذير: بنية البيانات غير متوافقة. سيتم البدء بقائمة فارغة.")
            self._contacts = []

    def save(self) -> None:
        """يحفظ كل جهات الاتصال الحالية إلى ملف JSON."""
        data = [asdict(contact) for contact in self._contacts]
        self.storage_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ----- العمليات الأساسية -----
    def add_contact(
        self,
        name: str,
        phone: str,
        email: str = "",
        group: ContactGroup = "أخرى",
    ) -> Contact:
        contact = Contact(name=name, phone=phone, email=email, group=group)
        self._contacts.append(contact)
        self.save()
        return contact

    def find_by_name(self, name: str) -> Contact:
        for contact in self._contacts:
            if contact.name == name:
                return contact
        raise ContactNotFoundError(f"لا توجد جهة اتصال باسم '{name}'")

    def delete_contact(self, name: str) -> None:
        contact = self.find_by_name(name)  # يرفع ContactNotFoundError تلقائيًا إن لم يوجد
        self._contacts.remove(contact)
        self.save()

    def search(self, keyword: str) -> list[Contact]:
        keyword_lower = keyword.lower()
        return [
            c for c in self._contacts
            if keyword_lower in c.name.lower() or keyword_lower in c.phone
        ]

    def get_by_group(self, group: ContactGroup) -> list[Contact]:
        return [c for c in self._contacts if c.group == group]

    def all_contacts(self) -> list[Contact]:
        return self._contacts.copy()

    def count(self) -> int:
        return len(self._contacts)

    # ----- التصدير -----
    def export_to_csv(self, csv_path: Path) -> None:
        if not self._contacts:
            print("لا توجد جهات اتصال للتصدير.")
            return

        fieldnames = [f.name for f in fields(Contact)]
        with open(csv_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for contact in self._contacts:
                writer.writerow(asdict(contact))
        print(f"✅ تم تصدير {len(self._contacts)} جهة اتصال إلى {csv_path}")

    def import_from_csv(self, csv_path: Path) -> int:
        if not csv_path.exists():
            raise FileNotFoundError(f"ملف الاستيراد غير موجود: {csv_path}")

        imported_count = 0
        with open(csv_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                self._contacts.append(Contact(**row))  # type: ignore[arg-type]
                imported_count += 1
        self.save()
        return imported_count

# ---------------------------------------------------------
# برنامج تجريبي
# ---------------------------------------------------------
def main() -> None:
    manager = ContactManager(Path("contacts.json"))

    manager.add_contact("أحمد علي", "0501234567", "ahmed@example.com", "عمل")
    manager.add_contact("سارة محمد", "0559876543", group="عائلة")
    manager.add_contact("خالد يوسف", "0507654321", "khaled@example.com", "أصدقاء")

    print(f"إجمالي جهات الاتصال: {manager.count()}")
    print()

    print("كل جهات الاتصال:")
    for contact in manager.all_contacts():
        print(contact.display())
    print()

    print("البحث عن 'سارة':")
    for contact in manager.search("سارة"):
        print(contact.display())
    print()

    try:
        manager.find_by_name("غير موجود")
    except ContactNotFoundError as e:
        print(f"خطأ متوقع: {e}")
    print()

    manager.export_to_csv(Path("contacts_export.csv"))

if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [068] code/chapters/ch16/16-035-count_lines.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-035-count_lines.py
"""

# تمرين 1
names: list[str] = [input(f"أدخل الاسم {i + 1}: ") for i in range(3)]

with open("names.txt", "w", encoding="utf-8") as file:
    for name in names:
        file.write(name + "\n")

# تمرين 2
from pathlib import Path

def count_lines(file_path: Path) -> int:
    with open(file_path, "r", encoding="utf-8") as file:
        return sum(1 for _ in file)

print(count_lines(Path("names.txt")))

# تمرين 3
import json
from pathlib import Path

person: dict[str, object] = {"name": "ليلى", "age": 27, "city": "تونس"}

Path("person.json").write_text(
    json.dumps(person, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

loaded = json.loads(Path("person.json").read_text(encoding="utf-8"))
print(loaded)

# ────────────────────────────────────────────────────────────────────────
# [069] code/chapters/ch16/16-036-calculate_total_sales.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-036-calculate_total_sales.py
"""

# تمرين 4
import csv
from pathlib import Path

def calculate_total_sales(csv_path: Path) -> float:
    if not csv_path.exists():
        print("ملف المبيعات غير موجود!")
        return 0.0

    total = 0.0
    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            total += float(row["amount"])
    return total

print(calculate_total_sales(Path("sales.csv")))

# تمرين 5
from pathlib import Path
from datetime import datetime

class Logger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, "a", encoding="utf-8") as file:
            file.write(f"[{timestamp}] {message}\n")

logger = Logger(Path("app.log"))
logger.log("بدأ تشغيل التطبيق")
logger.log("تم تسجيل دخول مستخدم جديد")

# تمرين 6
import json
from pathlib import Path

def save_config(config: dict[str, object], path: Path) -> None:
    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def load_config(path: Path) -> dict[str, object]:
    default_config: dict[str, object] = {"theme": "light", "language": "ar"}
    if not path.exists():
        return default_config
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default_config

config = load_config(Path("config.json"))
print(config)
save_config({"theme": "dark", "language": "en"}, Path("config.json"))

# ────────────────────────────────────────────────────────────────────────
# [070] code/chapters/ch16/16-037-backup_file.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-037-backup_file.py
"""

from pathlib import Path
from datetime import datetime
import shutil

def backup_file(source: Path, backup_dir: Path) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"الملف الأصلي غير موجود: {source}")

    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{source.stem}_{timestamp}{source.suffix}"
    backup_path = backup_dir / backup_name

    shutil.copy2(source, backup_path)
    return backup_path

def list_backups(backup_dir: Path, original_name: str) -> list[Path]:
    if not backup_dir.exists():
        return []

    stem = Path(original_name).stem
    suffix = Path(original_name).suffix

    matches = list(backup_dir.glob(f"{stem}_*{suffix}"))
    return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)

def restore_latest(backup_dir: Path, original_name: str, destination: Path) -> bool:
    backups = list_backups(backup_dir, original_name)
    if not backups:
        return False

    latest_backup = backups[0]
    shutil.copy2(latest_backup, destination)
    return True

# تجربة
source_file = Path("notes.txt")
source_file.write_text("محتوى تجريبي أول", encoding="utf-8")

backup_dir = Path("backups")

backup_file(source_file, backup_dir)

import time
time.sleep(1)  # للتأكد من اختلاف الطابع الزمني بين النسختين

source_file.write_text("محتوى تجريبي ثانٍ - مُحدّث", encoding="utf-8")
backup_file(source_file, backup_dir)

backups = list_backups(backup_dir, "notes.txt")
print(f"عدد النسخ الاحتياطية: {len(backups)}")
for b in backups:
    print(f"  {b.name}")

success = restore_latest(backup_dir, "notes.txt", Path("restored_notes.txt"))
print(f"تمت الاستعادة بنجاح: {success}")
print(Path("restored_notes.txt").read_text(encoding="utf-8"))

# ────────────────────────────────────────────────────────────────────────
# [071] code/chapters/ch16/16-038-content.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.10: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-038-content.py
"""

# ===== فتح الملفات النصية =====
with open("file.txt", "r", encoding="utf-8") as f:
    content = f.read()           # الملف كاملاً كنص
    # أو
    lines = f.readlines()        # قائمة بكل الأسطر
    # أو (الأفضل للملفات الكبيرة)
    for line in f:
        print(line.strip())

with open("file.txt", "w", encoding="utf-8") as f:
    f.write("نص جديد\n")         # يمحو المحتوى القديم!

with open("file.txt", "a", encoding="utf-8") as f:
    f.write("إضافة\n")           # يُضيف دون حذف

# ===== pathlib =====
from pathlib import Path

p = Path("data") / "file.txt"
p.exists()          # هل موجود؟
p.is_file()         # هل ملف؟
p.mkdir(parents=True, exist_ok=True)
p.write_text("محتوى", encoding="utf-8")
content = p.read_text(encoding="utf-8")
p.name, p.stem, p.suffix, p.parent

for f in Path(".").glob("*.py"):
    print(f)

# ===== JSON =====
import json

json.dumps(data, ensure_ascii=False, indent=2)   # Python → نص JSON
json.loads(json_text)                             # نص JSON → Python
json.dump(data, file, ensure_ascii=False, indent=2)  # Python → ملف
json.load(file)                                   # ملف → Python

# مع Dataclasses
from dataclasses import asdict
json.dumps(asdict(my_object), ensure_ascii=False)

# ===== CSV =====
import csv

with open("f.csv", "w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["a", "b"])
    writer.writeheader()
    writer.writerows(data)

with open("f.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(row)  # dict[str, str]

# ────────────────────────────────────────────────────────────────────────
# [072] dist/markdown/chapter-16.md
# ────────────────────────────────────────────────────────────────────────
open(file_path, mode, encoding="utf-8")

# ────────────────────────────────────────────────────────────────────────
# [073] dist/markdown/chapter-16.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطير: الاعتماد على الترميز الافتراضي لنظام التشغيل
with open("notes.txt", "r") as file:
    content = file.read()

# ✅ آمن: تحديد الترميز صراحة
with open("notes.txt", "r", encoding="utf-8") as file:
    content = file.read()

# ────────────────────────────────────────────────────────────────────────
# [074] dist/markdown/chapter-16.md
# ────────────────────────────────────────────────────────────────────────
> # أسلوب LBYL (تحقق قبل التنفيذ)
> import os
> if os.path.exists("file.txt"):
>     with open("file.txt", "r", encoding="utf-8") as f:
>         content = f.read()
> else:
>     print(«الملف غير موجود»)
>
> # أسلوب EAFP (الأسلوب الـ Pythonic الموصى به)
> try:
>     with open("file.txt", "r", encoding="utf-8") as f:
>         content = f.read()
> except FileNotFoundError:
>     print(«الملف غير موجود»)
> ```
> الأسلوب الثاني أفضل لأنه **آمن من سباقات التوقيت (Race Conditions)**: قد يُحذف الملف تمامًا في اللحظة الفاصلة بين `os.path.exists()` واستدعاء `open()`!

---

## 16.3 `pathlib`: الأسلوب الحديث للتعامل مع المسارات

في الإصدارات القديمة من Python، كان التعامل مع مسارات الملفات يتم عبر وحدة `os.path`، باستخدام نصوص عادية ودمجها يدويًا. هذا الأسلوب كان عرضة لمشاكل كثيرة (خاصة اختلاف فاصل المسارات بين Windows `\` ولينكس/ماك `/`).

منذ Python 3.4، أصبح لدينا `pathlib`: وحدة حديثة تُمثّل المسارات ككائنات حقيقية (Objects)، بدلًا من مجرد نصوص. هذا هو **الأسلوب الموصى به حصريًا** في هذا الكتاب.

### إنشاء كائن `Path`

# ────────────────────────────────────────────────────────────────────────
# [075] dist/markdown/chapter-16.md
# ────────────────────────────────────────────────────────────────────────
> 🔄 **مقارنة: `os.path` القديمة مقابل `pathlib` الحديثة**
> ```python
> # الأسلوب القديم (os.path) - لا تستخدمه في كود جديد
> import os
> path = os.path.join("data", "2024", "report.txt")
> if os.path.exists(path):
>     print(os.path.getsize(path))
>
> # الأسلوب الحديث (pathlib) - الموصى به
> from pathlib import Path
> path = Path("data") / "2024" / "report.txt"
> if path.exists():
>     print(path.stat().st_size)
> ```
> `pathlib` أكثر قابلية للقراءة، وتعمل بنفس الطريقة على كل أنظمة التشغيل، وتجمع كل الوظائف ذات الصلة بالمسار في كائن واحد متماسك (Object-Oriented)، بدلًا من دوال متفرقة.

### خصائص ودوال مفيدة في `Path`

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# مجلد عمل معزول حتى لا تتناثر الملفات في جذر الدفتر
WORK = pathlib.Path("book_runtime")
WORK.mkdir(exist_ok=True)
os.chdir(WORK)
print("مجلد العمل:", pathlib.Path.cwd().name)

# ملفات البداية التي يفترضها الفصل (تُكتب دائمًا لضمان نتيجة ثابتة)
pathlib.Path('notes.txt').write_text('ملاحظاتي اليومية\n- تعلّم open() و with\n- تذكّر إغلاق الملف\n', encoding='utf-8')

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
file = open("notes.txt", "r", encoding="utf-8")
content = file.read()
print(content)
file.close()  # يجب أن نتذكر إغلاق الملف يدويًا!

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
with open("notes.txt", "r", encoding="utf-8") as file:
    content: str = file.read()
    print(content)
# الملف يُغلق تلقائيًا هنا، حتى لو حدث خطأ بالداخل!

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
with open("diary.txt", "w", encoding="utf-8") as file:
    file.write("اليوم الأول: بدأت تعلم Python.\n")
    file.write("كان يومًا ممتعًا جدًا!\n")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
with open("diary.txt", "a", encoding="utf-8") as file:
    file.write("اليوم الثاني: تعلمت عن الملفات!\n")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# الطريقة 1: قراءة الملف بالكامل كنص واحد
with open("diary.txt", "r", encoding="utf-8") as file:
    content: str = file.read()
    print(content)

# الطريقة 2: قراءة سطر واحد فقط
with open("diary.txt", "r", encoding="utf-8") as file:
    first_line: str = file.readline()
    print(first_line)  # "اليوم الأول: بدأت تعلم Python.\n"

# الطريقة 3: قراءة كل الأسطر كقائمة
with open("diary.txt", "r", encoding="utf-8") as file:
    lines: list[str] = file.readlines()
    print(lines)
    # ['اليوم الأول: ...\n', 'اليوم الثاني: ...\n', ...]

# الطريقة 4 (الأفضل للملفات الكبيرة): التكرار المباشر سطرًا بسطر
with open("diary.txt", "r", encoding="utf-8") as file:
    for line in file:
        print(line.strip())  # .strip() لإزالة \n الزائدة

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
try:
    with open("missing_file.txt", "r", encoding="utf-8") as file:
        content = file.read()
except FileNotFoundError:
    print("الملف غير موجود!")
except PermissionError:
    print("لا تملك صلاحية الوصول لهذا الملف!")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from pathlib import Path

# مسار نسبي
notes_path = Path("notes.txt")

# مسار بمجلدات متعددة - لاحظ استخدام / كعامل دمج!
project_path = Path("data") / "2024" / "report.txt"
print(project_path)
# على لينكس/ماك: data/2024/report.txt
# على Windows: data\2024\report.txt  (يتكيف تلقائيًا!)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
from pathlib import Path

p = Path("data/2024/report.txt")

print(p.name)        # "report.txt" - اسم الملف كاملاً
print(p.stem)        # "report" - الاسم بدون الامتداد
print(p.suffix)      # ".txt" - الامتداد فقط
print(p.parent)      # "data/2024" - المجلد الأب
print(p.parts)       # ('data', '2024', 'report.txt') - كل الأجزاء

print(p.exists())    # هل المسار موجود فعليًا؟ True/False
print(p.is_file())   # هل هو ملف؟
print(p.is_dir())    # هل هو مجلد؟
print(p.absolute())  # المسار الكامل المطلق

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
from pathlib import Path

data_dir = Path("data/2024")
data_dir.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
from pathlib import Path

notes = Path("notes.txt")

# كتابة (تُنشئ الملف تلقائيًا إن لم يكن موجودًا، وتفتح وتغلق تلقائيًا)
notes.write_text("مرحبًا من pathlib!\n", encoding="utf-8")

# قراءة
content: str = notes.read_text(encoding="utf-8")
print(content)

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
from pathlib import Path

project_dir = Path(".")  # المجلد الحالي

# كل الملفات والمجلدات مباشرة (دون تعمق)
for item in project_dir.iterdir():
    print(item)

# البحث عن كل ملفات .py في المجلد الحالي فقط
for py_file in project_dir.glob("*.py"):
    print(py_file)

# البحث التكراري في كل المجلدات الفرعية أيضًا
for py_file in project_dir.glob("**/*.py"):
    print(py_file)

# بديل أحدث وأوضح لنفس الغرض (Python 3.12+)
for py_file in project_dir.rglob("*.py"):
    print(py_file)

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
from pathlib import Path
import shutil

def organize_downloads(downloads_path: Path) -> None:
    """يُنظّم الملفات في مجلد حسب امتدادها، كل نوع في مجلد فرعي خاص به."""
    extension_folders: dict[str, str] = {
        ".pdf": "مستندات",
        ".jpg": "صور",
        ".png": "صور",
        ".mp4": "فيديوهات",
        ".zip": "أرشيف",
    }

    for file_path in downloads_path.iterdir():
        if file_path.is_file():
            folder_name = extension_folders.get(file_path.suffix.lower())
            if folder_name is not None:
                target_dir = downloads_path / folder_name
                target_dir.mkdir(exist_ok=True)
                shutil.move(str(file_path), str(target_dir / file_path.name))
                print(f"نُقل {file_path.name} إلى {folder_name}/")

organize_downloads(Path.home() / "Downloads")

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
import json

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
import json

user_data: dict[str, object] = {
    "name": "أحمد",
    "age": 30,
    "is_active": True,
    "hobbies": ["قراءة", "برمجة"],
    "spouse": None,
}

# dumps (مع s) تُرجع نصًا (string) يمكن طباعته أو إرساله
json_string: str = json.dumps(user_data, ensure_ascii=False, indent=2)
print(json_string)

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
import json
from pathlib import Path

user_data = {"name": "سارة", "age": 25, "city": "دبي"}

with open("user.json", "w", encoding="utf-8") as file:
    json.dump(user_data, file, ensure_ascii=False, indent=2)

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
import json

json_text = '{"name": "خالد", "age": 28}'
data: dict[str, object] = json.loads(json_text)

print(data["name"])  # "خالد"
print(type(data))    # <class 'dict'>

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
import json

with open("user.json", "r", encoding="utf-8") as file:
    data: dict[str, object] = json.load(file)

print(data["name"])  # "سارة"

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
import json
from pathlib import Path

# كتابة
Path("user.json").write_text(
    json.dumps(user_data, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

# قراءة
data = json.loads(Path("user.json").read_text(encoding="utf-8"))

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
import json

corrupted_json = '{"name": "أحمد", "age": }'  # صيغة خاطئة

try:
    data = json.loads(corrupted_json)
except json.JSONDecodeError as e:
    print(f"خطأ في تنسيق JSON: {e}")

# ────────────────────────────────────────────────────────────────────────
# [022] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass, asdict
import json

@dataclass
class Product:
    name: str
    price: float

product = Product("لابتوب", 1200.0)

product_dict: dict[str, object] = asdict(product)
json_string = json.dumps(product_dict, ensure_ascii=False, indent=2)
print(json_string)
# {
#   "name": "لابتوب",
#   "price": 1200.0
# }

# ────────────────────────────────────────────────────────────────────────
# [023] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
data = json.loads(json_string)
restored_product = Product(**data)  # فك تغليف القاموس كمعاملات
print(restored_product)  # Product(name='لابتوب', price=1200.0)

# ────────────────────────────────────────────────────────────────────────
# [024] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass, asdict
from pathlib import Path
import json

@dataclass
class Task:
    title: str
    done: bool = False

def save_tasks(tasks: list[Task], file_path: Path) -> None:
    data = [asdict(task) for task in tasks]
    file_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def load_tasks(file_path: Path) -> list[Task]:
    if not file_path.exists():
        return []
    raw_data = json.loads(file_path.read_text(encoding="utf-8"))
    return [Task(**item) for item in raw_data]

tasks_file = Path("tasks.json")

tasks = [
    Task("تعلم JSON"),
    Task("تعلم CSV", done=True),
]
save_tasks(tasks, tasks_file)

loaded_tasks = load_tasks(tasks_file)
for task in loaded_tasks:
    status = "✅" if task.done else "⬜"
    print(f"{status} {task.title}")

# ────────────────────────────────────────────────────────────────────────
# [025] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
import csv

rows: list[list[str]] = [
    ["name", "age", "city"],
    ["أحمد", "30", "القاهرة"],
    ["سارة", "25", "دبي"],
]

with open("people.csv", "w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(rows)

# ────────────────────────────────────────────────────────────────────────
# [026] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
import csv

with open("people.csv", "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)  # كل صف عبارة عن list[str]
# ['name', 'age', 'city']
# ['أحمد', '30', 'القاهرة']
# ['سارة', '25', 'دبي']

# ────────────────────────────────────────────────────────────────────────
# [027] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-26
# ────────────────────────────────────────────────────────────────────────
import csv

# القراءة بـ DictReader
with open("people.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(row)
        # {'name': 'أحمد', 'age': '30', 'city': 'القاهرة'}
        print(f"{row['name']} يعيش في {row['city']}")

# ────────────────────────────────────────────────────────────────────────
# [028] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-27
# ────────────────────────────────────────────────────────────────────────
import csv

# الكتابة بـ DictWriter
people: list[dict[str, str]] = [
    {"name": "منى", "age": "22", "city": "عمّان"},
    {"name": "طارق", "age": "35", "city": "بيروت"},
]

with open("people2.csv", "w", encoding="utf-8", newline="") as file:
    fieldnames = ["name", "age", "city"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()   # يكتب صف العناوين تلقائيًا
    writer.writerows(people)

# ────────────────────────────────────────────────────────────────────────
# [029] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-28
# ────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass, asdict, fields
import csv
from pathlib import Path

@dataclass
class Employee:
    name: str
    department: str
    salary: float

def export_to_csv(employees: list[Employee], file_path: Path) -> None:
    if not employees:
        return
    fieldnames = [f.name for f in fields(Employee)]
    with open(file_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for emp in employees:
            writer.writerow(asdict(emp))

employees = [
    Employee("ياسمين", "تسويق", 6000.0),
    Employee("عمر", "هندسة", 8500.0),
]

export_to_csv(employees, Path("employees.csv"))

# ────────────────────────────────────────────────────────────────────────
# [030] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-29
# ────────────────────────────────────────────────────────────────────────
from pathlib import Path

config_path = Path("config.json")

if config_path.exists():
    config = json.loads(config_path.read_text(encoding="utf-8"))
else:
    config = {}  # قيمة افتراضية آمنة

# ────────────────────────────────────────────────────────────────────────
# [031] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-30
# ────────────────────────────────────────────────────────────────────────
# ✅ بنية جيدة: دوال واضحة ومسؤولة عن مهمة واحدة
def load_data(path: Path) -> list[dict[str, object]]:
    ...

def save_data(data: list[dict[str, object]], path: Path) -> None:
    ...

def process_data(data: list[dict[str, object]]) -> list[dict[str, object]]:
     # المنطق الفعلي منفصل تمامًا عن I/O

# ────────────────────────────────────────────────────────────────────────
# [032] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-31
# ────────────────────────────────────────────────────────────────────────
"""
مدير جهات الاتصال - مشروع تطبيقي شامل
يغطي: الملفات النصية، pathlib، JSON، CSV، ومعالجة الأخطاء
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Literal
import json
import csv

type ContactGroup = Literal["عائلة", "عمل", "أصدقاء", "أخرى"]

# ---------------------------------------------------------
# نموذج البيانات
# ---------------------------------------------------------
@dataclass
class Contact:
    name: str
    phone: str
    email: str = ""
    group: ContactGroup = "أخرى"

    def display(self) -> str:
        return f"📇 {self.name} | {self.phone} | {self.email} | {self.group}"

# ---------------------------------------------------------
# استثناءات مخصصة
# ---------------------------------------------------------
class ContactNotFoundError(Exception):
    """يُرفع عند عدم العثور على جهة اتصال بالاسم المطلوب."""

# ---------------------------------------------------------
# مدير جهات الاتصال
# ---------------------------------------------------------
class ContactManager:
    def __init__(self, storage_path: Path) -> None:
        self.storage_path = storage_path
        self._contacts: list[Contact] = []
        self._load()

    # ----- التحميل والحفظ -----
    def _load(self) -> None:
        """يحمّل جهات الاتصال من ملف JSON إن وُجد، وإلا يبدأ بقائمة فارغة."""
        if not self.storage_path.exists():
            self._contacts = []
            return

        try:
            raw_text = self.storage_path.read_text(encoding="utf-8")
            raw_data = json.loads(raw_text)
            self._contacts = [Contact(**item) for item in raw_data]
        except json.JSONDecodeError:
            print("⚠️ تحذير: ملف البيانات تالف. سيتم البدء بقائمة فارغة.")
            self._contacts = []
        except TypeError:
            print("⚠️ تحذير: بنية البيانات غير متوافقة. سيتم البدء بقائمة فارغة.")
            self._contacts = []

    def save(self) -> None:
        """يحفظ كل جهات الاتصال الحالية إلى ملف JSON."""
        data = [asdict(contact) for contact in self._contacts]
        self.storage_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ----- العمليات الأساسية -----
    def add_contact(
        self,
        name: str,
        phone: str,
        email: str = "",
        group: ContactGroup = "أخرى",
    ) -> Contact:
        contact = Contact(name=name, phone=phone, email=email, group=group)
        self._contacts.append(contact)
        self.save()
        return contact

    def find_by_name(self, name: str) -> Contact:
        for contact in self._contacts:
            if contact.name == name:
                return contact
        raise ContactNotFoundError(f"لا توجد جهة اتصال باسم '{name}'")

    def delete_contact(self, name: str) -> None:
        contact = self.find_by_name(name)  # يرفع ContactNotFoundError تلقائيًا إن لم يوجد
        self._contacts.remove(contact)
        self.save()

    def search(self, keyword: str) -> list[Contact]:
        keyword_lower = keyword.lower()
        return [
            c for c in self._contacts
            if keyword_lower in c.name.lower() or keyword_lower in c.phone
        ]

    def get_by_group(self, group: ContactGroup) -> list[Contact]:
        return [c for c in self._contacts if c.group == group]

    def all_contacts(self) -> list[Contact]:
        return self._contacts.copy()

    def count(self) -> int:
        return len(self._contacts)

    # ----- التصدير -----
    def export_to_csv(self, csv_path: Path) -> None:
        if not self._contacts:
            print("لا توجد جهات اتصال للتصدير.")
            return

        fieldnames = [f.name for f in fields(Contact)]
        with open(csv_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for contact in self._contacts:
                writer.writerow(asdict(contact))
        print(f"✅ تم تصدير {len(self._contacts)} جهة اتصال إلى {csv_path}")

    def import_from_csv(self, csv_path: Path) -> int:
        if not csv_path.exists():
            raise FileNotFoundError(f"ملف الاستيراد غير موجود: {csv_path}")

        imported_count = 0
        with open(csv_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                self._contacts.append(Contact(**row))  # type: ignore[arg-type]
                imported_count += 1
        self.save()
        return imported_count

# ---------------------------------------------------------
# برنامج تجريبي
# ---------------------------------------------------------
def main() -> None:
    manager = ContactManager(Path("contacts.json"))

    manager.add_contact("أحمد علي", "0501234567", "ahmed@example.com", "عمل")
    manager.add_contact("سارة محمد", "0559876543", group="عائلة")
    manager.add_contact("خالد يوسف", "0507654321", "khaled@example.com", "أصدقاء")

    print(f"إجمالي جهات الاتصال: {manager.count()}")
    print()

    print("كل جهات الاتصال:")
    for contact in manager.all_contacts():
        print(contact.display())
    print()

    print("البحث عن 'سارة':")
    for contact in manager.search("سارة"):
        print(contact.display())
    print()

    try:
        manager.find_by_name("غير موجود")
    except ContactNotFoundError as e:
        print(f"خطأ متوقع: {e}")
    print()

    manager.export_to_csv(Path("contacts_export.csv"))

if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [033] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-32
# ────────────────────────────────────────────────────────────────────────
# تمرين 4
import csv
from pathlib import Path

def calculate_total_sales(csv_path: Path) -> float:
    if not csv_path.exists():
        print("ملف المبيعات غير موجود!")
        return 0.0

    total = 0.0
    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            total += float(row["amount"])
    return total

print(calculate_total_sales(Path("sales.csv")))

# تمرين 5
from pathlib import Path
from datetime import datetime

class Logger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, "a", encoding="utf-8") as file:
            file.write(f"[{timestamp}] {message}\n")

logger = Logger(Path("app.log"))
logger.log("بدأ تشغيل التطبيق")
logger.log("تم تسجيل دخول مستخدم جديد")

# تمرين 6
import json
from pathlib import Path

def save_config(config: dict[str, object], path: Path) -> None:
    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def load_config(path: Path) -> dict[str, object]:
    default_config: dict[str, object] = {"theme": "light", "language": "ar"}
    if not path.exists():
        return default_config
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default_config

config = load_config(Path("config.json"))
print(config)
save_config({"theme": "dark", "language": "en"}, Path("config.json"))

# ────────────────────────────────────────────────────────────────────────
# [034] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-33
# ────────────────────────────────────────────────────────────────────────
from pathlib import Path
from datetime import datetime
import shutil

def backup_file(source: Path, backup_dir: Path) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"الملف الأصلي غير موجود: {source}")

    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{source.stem}_{timestamp}{source.suffix}"
    backup_path = backup_dir / backup_name

    shutil.copy2(source, backup_path)
    return backup_path

def list_backups(backup_dir: Path, original_name: str) -> list[Path]:
    if not backup_dir.exists():
        return []

    stem = Path(original_name).stem
    suffix = Path(original_name).suffix

    matches = list(backup_dir.glob(f"{stem}_*{suffix}"))
    return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)

def restore_latest(backup_dir: Path, original_name: str, destination: Path) -> bool:
    backups = list_backups(backup_dir, original_name)
    if not backups:
        return False

    latest_backup = backups[0]
    shutil.copy2(latest_backup, destination)
    return True

# تجربة
source_file = Path("notes.txt")
source_file.write_text("محتوى تجريبي أول", encoding="utf-8")

backup_dir = Path("backups")

backup_file(source_file, backup_dir)

import time
time.sleep(1)  # للتأكد من اختلاف الطابع الزمني بين النسختين

source_file.write_text("محتوى تجريبي ثانٍ - مُحدّث", encoding="utf-8")
backup_file(source_file, backup_dir)

backups = list_backups(backup_dir, "notes.txt")
print(f"عدد النسخ الاحتياطية: {len(backups)}")
for b in backups:
    print(f"  {b.name}")

success = restore_latest(backup_dir, "notes.txt", Path("restored_notes.txt"))
print(f"تمت الاستعادة بنجاح: {success}")
print(Path("restored_notes.txt").read_text(encoding="utf-8"))

# ────────────────────────────────────────────────────────────────────────
# [035] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-34
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.2: الملفات النصية: الأساسيات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-001-file.py
"""

file = open("notes.txt", "r", encoding="utf-8")
content = file.read()
print(content)
file.close()  # يجب أن نتذكر إغلاق الملف يدويًا!

# ────────────────────────────────────────────────────────────────────────
# [036] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-35
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.2: الملفات النصية: الأساسيات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-002-ex16_2.py
"""

with open("notes.txt", "r", encoding="utf-8") as file:
    content: str = file.read()
    print(content)
# الملف يُغلق تلقائيًا هنا، حتى لو حدث خطأ بالداخل!

# ────────────────────────────────────────────────────────────────────────
# [037] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-36
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.2: الملفات النصية: الأساسيات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-003-ex16_2.py
"""

open(file_path, mode, encoding="utf-8")

# ────────────────────────────────────────────────────────────────────────
# [038] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-37
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.2: الملفات النصية: الأساسيات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-004-ex16_2.py
"""

with open("diary.txt", "w", encoding="utf-8") as file:
    file.write("اليوم الأول: بدأت تعلم Python.\n")
    file.write("كان يومًا ممتعًا جدًا!\n")

# ────────────────────────────────────────────────────────────────────────
# [039] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-38
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.2: الملفات النصية: الأساسيات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-005-ex16_2.py
"""

with open("diary.txt", "a", encoding="utf-8") as file:
    file.write("اليوم الثاني: تعلمت عن الملفات!\n")

# ────────────────────────────────────────────────────────────────────────
# [040] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-39
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.2: الملفات النصية: الأساسيات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-006-ex16_2.py
"""

# الطريقة 1: قراءة الملف بالكامل كنص واحد
with open("diary.txt", "r", encoding="utf-8") as file:
    content: str = file.read()
    print(content)

# الطريقة 2: قراءة سطر واحد فقط
with open("diary.txt", "r", encoding="utf-8") as file:
    first_line: str = file.readline()
    print(first_line)  # "اليوم الأول: بدأت تعلم Python.\n"

# الطريقة 3: قراءة كل الأسطر كقائمة
with open("diary.txt", "r", encoding="utf-8") as file:
    lines: list[str] = file.readlines()
    print(lines)
    # ['اليوم الأول: ...\n', 'اليوم الثاني: ...\n', ...]

# الطريقة 4 (الأفضل للملفات الكبيرة): التكرار المباشر سطرًا بسطر
with open("diary.txt", "r", encoding="utf-8") as file:
    for line in file:
        print(line.strip())  # .strip() لإزالة \n الزائدة

# ────────────────────────────────────────────────────────────────────────
# [041] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-40
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.2: الملفات النصية: الأساسيات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-007-content.py
"""

# ❌ خطير: الاعتماد على الترميز الافتراضي لنظام التشغيل
with open("notes.txt", "r") as file:
    content = file.read()

# ✅ آمن: تحديد الترميز صراحة
with open("notes.txt", "r", encoding="utf-8") as file:
    content = file.read()

# ────────────────────────────────────────────────────────────────────────
# [042] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-41
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.2: الملفات النصية: الأساسيات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-008-content.py
"""

try:
    with open("missing_file.txt", "r", encoding="utf-8") as file:
        content = file.read()
except FileNotFoundError:
    print("الملف غير موجود!")
except PermissionError:
    print("لا تملك صلاحية الوصول لهذا الملف!")

# ────────────────────────────────────────────────────────────────────────
# [043] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-42
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.3: `pathlib`: الأسلوب الحديث للتعامل مع المسارات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-009-notes_path.py
"""

from pathlib import Path

# مسار نسبي
notes_path = Path("notes.txt")

# مسار بمجلدات متعددة - لاحظ استخدام / كعامل دمج!
project_path = Path("data") / "2024" / "report.txt"
print(project_path)
# على لينكس/ماك: data/2024/report.txt
# على Windows: data\2024\report.txt  (يتكيف تلقائيًا!)

# ────────────────────────────────────────────────────────────────────────
# [044] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-43
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.3: `pathlib`: الأسلوب الحديث للتعامل مع المسارات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-010-ex16_3.py
"""

from pathlib import Path

p = Path("data/2024/report.txt")

print(p.name)        # "report.txt" - اسم الملف كاملاً
print(p.stem)        # "report" - الاسم بدون الامتداد
print(p.suffix)      # ".txt" - الامتداد فقط
print(p.parent)      # "data/2024" - المجلد الأب
print(p.parts)       # ('data', '2024', 'report.txt') - كل الأجزاء

print(p.exists())    # هل المسار موجود فعليًا؟ True/False
print(p.is_file())   # هل هو ملف؟
print(p.is_dir())    # هل هو مجلد؟
print(p.absolute())  # المسار الكامل المطلق

# ────────────────────────────────────────────────────────────────────────
# [045] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-44
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.3: `pathlib`: الأسلوب الحديث للتعامل مع المسارات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-011-data_dir.py
"""

from pathlib import Path

data_dir = Path("data/2024")
data_dir.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────
# [046] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-45
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.3: `pathlib`: الأسلوب الحديث للتعامل مع المسارات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-012-notes.py
"""

from pathlib import Path

notes = Path("notes.txt")

# كتابة (تُنشئ الملف تلقائيًا إن لم يكن موجودًا، وتفتح وتغلق تلقائيًا)
notes.write_text("مرحبًا من pathlib!\n", encoding="utf-8")

# قراءة
content: str = notes.read_text(encoding="utf-8")
print(content)

# ────────────────────────────────────────────────────────────────────────
# [047] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-46
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.3: `pathlib`: الأسلوب الحديث للتعامل مع المسارات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-013-project_dir.py
"""

from pathlib import Path

project_dir = Path(".")  # المجلد الحالي

# كل الملفات والمجلدات مباشرة (دون تعمق)
for item in project_dir.iterdir():
    print(item)

# البحث عن كل ملفات .py في المجلد الحالي فقط
for py_file in project_dir.glob("*.py"):
    print(py_file)

# البحث التكراري في كل المجلدات الفرعية أيضًا
for py_file in project_dir.glob("**/*.py"):
    print(py_file)

# بديل أحدث وأوضح لنفس الغرض (Python 3.12+)
for py_file in project_dir.rglob("*.py"):
    print(py_file)

# ────────────────────────────────────────────────────────────────────────
# [048] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-47
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.3: `pathlib`: الأسلوب الحديث للتعامل مع المسارات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-014-organize_downloads.py
"""

from pathlib import Path
import shutil

def organize_downloads(downloads_path: Path) -> None:
    """يُنظّم الملفات في مجلد حسب امتدادها، كل نوع في مجلد فرعي خاص به."""
    extension_folders: dict[str, str] = {
        ".pdf": "مستندات",
        ".jpg": "صور",
        ".png": "صور",
        ".mp4": "فيديوهات",
        ".zip": "أرشيف",
    }

    for file_path in downloads_path.iterdir():
        if file_path.is_file():
            folder_name = extension_folders.get(file_path.suffix.lower())
            if folder_name is not None:
                target_dir = downloads_path / folder_name
                target_dir.mkdir(exist_ok=True)
                shutil.move(str(file_path), str(target_dir / file_path.name))
                print(f"نُقل {file_path.name} إلى {folder_name}/")

organize_downloads(Path.home() / "Downloads")

# ────────────────────────────────────────────────────────────────────────
# [049] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-48
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.4: JSON: لغة تبادل البيانات العالمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-015-ex16_4.py
"""

import json

# ────────────────────────────────────────────────────────────────────────
# [050] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-49
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.4: JSON: لغة تبادل البيانات العالمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-016-ex16_4.py
"""

import json

user_data: dict[str, object] = {
    "name": "أحمد",
    "age": 30,
    "is_active": True,
    "hobbies": ["قراءة", "برمجة"],
    "spouse": None,
}

# dumps (مع s) تُرجع نصًا (string) يمكن طباعته أو إرساله
json_string: str = json.dumps(user_data, ensure_ascii=False, indent=2)
print(json_string)

# ────────────────────────────────────────────────────────────────────────
# [051] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-50
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.4: JSON: لغة تبادل البيانات العالمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-017-user_data.py
"""

import json
from pathlib import Path

user_data = {"name": "سارة", "age": 25, "city": "دبي"}

with open("user.json", "w", encoding="utf-8") as file:
    json.dump(user_data, file, ensure_ascii=False, indent=2)

# ────────────────────────────────────────────────────────────────────────
# [052] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-51
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.4: JSON: لغة تبادل البيانات العالمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-018-json_text.py
"""

import json

json_text = '{"name": "خالد", "age": 28}'
data: dict[str, object] = json.loads(json_text)

print(data["name"])  # "خالد"
print(type(data))    # <class 'dict'>

# ────────────────────────────────────────────────────────────────────────
# [053] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-52
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.4: JSON: لغة تبادل البيانات العالمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-019-ex16_4.py
"""

import json

with open("user.json", "r", encoding="utf-8") as file:
    data: dict[str, object] = json.load(file)

print(data["name"])  # "سارة"

# ────────────────────────────────────────────────────────────────────────
# [054] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-53
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.4: JSON: لغة تبادل البيانات العالمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-020-encoding.py
"""

import json
from pathlib import Path

# كتابة
Path("user.json").write_text(
    json.dumps(user_data, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

# قراءة
data = json.loads(Path("user.json").read_text(encoding="utf-8"))

# ────────────────────────────────────────────────────────────────────────
# [055] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-54
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.4: JSON: لغة تبادل البيانات العالمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-021-corrupted_json.py
"""

import json

corrupted_json = '{"name": "أحمد", "age": }'  # صيغة خاطئة

try:
    data = json.loads(corrupted_json)
except json.JSONDecodeError as e:
    print(f"خطأ في تنسيق JSON: {e}")

# ────────────────────────────────────────────────────────────────────────
# [056] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-55
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.4: JSON: لغة تبادل البيانات العالمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-022-product.py
"""

from dataclasses import dataclass
import json

@dataclass
class Product:
    name: str
    price: float

product = Product("لابتوب", 1200.0)
json.dumps(product)  # ❌ TypeError: Object of type Product is not JSON serializable

# ────────────────────────────────────────────────────────────────────────
# [057] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-56
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.4: JSON: لغة تبادل البيانات العالمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-023-product.py
"""

from dataclasses import dataclass, asdict
import json

@dataclass
class Product:
    name: str
    price: float

product = Product("لابتوب", 1200.0)

product_dict: dict[str, object] = asdict(product)
json_string = json.dumps(product_dict, ensure_ascii=False, indent=2)
print(json_string)
# {
#   "name": "لابتوب",
#   "price": 1200.0
# }

# ────────────────────────────────────────────────────────────────────────
# [058] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-57
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.4: JSON: لغة تبادل البيانات العالمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-024-data.py
"""

data = json.loads(json_string)
restored_product = Product(**data)  # فك تغليف القاموس كمعاملات
print(restored_product)  # Product(name='لابتوب', price=1200.0)

# ────────────────────────────────────────────────────────────────────────
# [059] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-58
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.4: JSON: لغة تبادل البيانات العالمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-025-save_tasks.py
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import json

@dataclass
class Task:
    title: str
    done: bool = False

def save_tasks(tasks: list[Task], file_path: Path) -> None:
    data = [asdict(task) for task in tasks]
    file_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def load_tasks(file_path: Path) -> list[Task]:
    if not file_path.exists():
        return []
    raw_data = json.loads(file_path.read_text(encoding="utf-8"))
    return [Task(**item) for item in raw_data]

tasks_file = Path("tasks.json")

tasks = [
    Task("تعلم JSON"),
    Task("تعلم CSV", done=True),
]
save_tasks(tasks, tasks_file)

loaded_tasks = load_tasks(tasks_file)
for task in loaded_tasks:
    status = "✅" if task.done else "⬜"
    print(f"{status} {task.title}")

# ────────────────────────────────────────────────────────────────────────
# [060] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-59
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.5: CSV: البيانات الجدولية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-026-writer.py
"""

import csv

rows: list[list[str]] = [
    ["name", "age", "city"],
    ["أحمد", "30", "القاهرة"],
    ["سارة", "25", "دبي"],
]

with open("people.csv", "w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(rows)

# ────────────────────────────────────────────────────────────────────────
# [061] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-60
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.5: CSV: البيانات الجدولية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-027-reader.py
"""

import csv

with open("people.csv", "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)  # كل صف عبارة عن list[str]
# ['name', 'age', 'city']
# ['أحمد', '30', 'القاهرة']
# ['سارة', '25', 'دبي']

# ────────────────────────────────────────────────────────────────────────
# [062] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-61
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.5: CSV: البيانات الجدولية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-028-reader.py
"""

import csv

# القراءة بـ DictReader
with open("people.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(row)
        # {'name': 'أحمد', 'age': '30', 'city': 'القاهرة'}
        print(f"{row['name']} يعيش في {row['city']}")

# ────────────────────────────────────────────────────────────────────────
# [063] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-62
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.5: CSV: البيانات الجدولية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-029-fieldnames.py
"""

import csv

# الكتابة بـ DictWriter
people: list[dict[str, str]] = [
    {"name": "منى", "age": "22", "city": "عمّان"},
    {"name": "طارق", "age": "35", "city": "بيروت"},
]

with open("people2.csv", "w", encoding="utf-8", newline="") as file:
    fieldnames = ["name", "age", "city"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()   # يكتب صف العناوين تلقائيًا
    writer.writerows(people)

# ────────────────────────────────────────────────────────────────────────
# [064] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-63
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.5: CSV: البيانات الجدولية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-030-export_to_csv.py
"""

from dataclasses import dataclass, asdict, fields
import csv
from pathlib import Path

@dataclass
class Employee:
    name: str
    department: str
    salary: float

def export_to_csv(employees: list[Employee], file_path: Path) -> None:
    if not employees:
        return
    fieldnames = [f.name for f in fields(Employee)]
    with open(file_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for emp in employees:
            writer.writerow(asdict(emp))

employees = [
    Employee("ياسمين", "تسويق", 6000.0),
    Employee("عمر", "هندسة", 8500.0),
]

export_to_csv(employees, Path("employees.csv"))

# ────────────────────────────────────────────────────────────────────────
# [065] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-64
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.6: أفضل الممارسات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-031-file_path.py
"""

# ❌ تجنّب
file_path = "data" + "/" + "2024" + "/" + "report.txt"

# ✅ استخدم
from pathlib import Path
file_path = Path("data") / "2024" / "report.txt"

# ────────────────────────────────────────────────────────────────────────
# [066] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-65
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.6: أفضل الممارسات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-032-config_path.py
"""

from pathlib import Path

config_path = Path("config.json")

if config_path.exists():
    config = json.loads(config_path.read_text(encoding="utf-8"))
else:
    config = {}  # قيمة افتراضية آمنة

# ────────────────────────────────────────────────────────────────────────
# [067] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-66
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.6: أفضل الممارسات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-033-load_data.py
"""

# ✅ بنية جيدة: دوال واضحة ومسؤولة عن مهمة واحدة
def load_data(path: Path) -> list[dict[str, object]]:
    ...

def save_data(data: list[dict[str, object]], path: Path) -> None:
    ...

def process_data(data: list[dict[str, object]]) -> list[dict[str, object]]:
     # المنطق الفعلي منفصل تمامًا عن I/O

# ────────────────────────────────────────────────────────────────────────
# [068] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-67
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.7: مشروع تطبيقي: مدير جهات اتصال متكامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-034-display.py
"""

"""
مدير جهات الاتصال - مشروع تطبيقي شامل
يغطي: الملفات النصية، pathlib، JSON، CSV، ومعالجة الأخطاء
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Literal
import json
import csv

type ContactGroup = Literal["عائلة", "عمل", "أصدقاء", "أخرى"]

# ---------------------------------------------------------
# نموذج البيانات
# ---------------------------------------------------------
@dataclass
class Contact:
    name: str
    phone: str
    email: str = ""
    group: ContactGroup = "أخرى"

    def display(self) -> str:
        return f"📇 {self.name} | {self.phone} | {self.email} | {self.group}"

# ---------------------------------------------------------
# استثناءات مخصصة
# ---------------------------------------------------------
class ContactNotFoundError(Exception):
    """يُرفع عند عدم العثور على جهة اتصال بالاسم المطلوب."""

# ---------------------------------------------------------
# مدير جهات الاتصال
# ---------------------------------------------------------
class ContactManager:
    def __init__(self, storage_path: Path) -> None:
        self.storage_path = storage_path
        self._contacts: list[Contact] = []
        self._load()

    # ----- التحميل والحفظ -----
    def _load(self) -> None:
        """يحمّل جهات الاتصال من ملف JSON إن وُجد، وإلا يبدأ بقائمة فارغة."""
        if not self.storage_path.exists():
            self._contacts = []
            return

        try:
            raw_text = self.storage_path.read_text(encoding="utf-8")
            raw_data = json.loads(raw_text)
            self._contacts = [Contact(**item) for item in raw_data]
        except json.JSONDecodeError:
            print("⚠️ تحذير: ملف البيانات تالف. سيتم البدء بقائمة فارغة.")
            self._contacts = []
        except TypeError:
            print("⚠️ تحذير: بنية البيانات غير متوافقة. سيتم البدء بقائمة فارغة.")
            self._contacts = []

    def save(self) -> None:
        """يحفظ كل جهات الاتصال الحالية إلى ملف JSON."""
        data = [asdict(contact) for contact in self._contacts]
        self.storage_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ----- العمليات الأساسية -----
    def add_contact(
        self,
        name: str,
        phone: str,
        email: str = "",
        group: ContactGroup = "أخرى",
    ) -> Contact:
        contact = Contact(name=name, phone=phone, email=email, group=group)
        self._contacts.append(contact)
        self.save()
        return contact

    def find_by_name(self, name: str) -> Contact:
        for contact in self._contacts:
            if contact.name == name:
                return contact
        raise ContactNotFoundError(f"لا توجد جهة اتصال باسم '{name}'")

    def delete_contact(self, name: str) -> None:
        contact = self.find_by_name(name)  # يرفع ContactNotFoundError تلقائيًا إن لم يوجد
        self._contacts.remove(contact)
        self.save()

    def search(self, keyword: str) -> list[Contact]:
        keyword_lower = keyword.lower()
        return [
            c for c in self._contacts
            if keyword_lower in c.name.lower() or keyword_lower in c.phone
        ]

    def get_by_group(self, group: ContactGroup) -> list[Contact]:
        return [c for c in self._contacts if c.group == group]

    def all_contacts(self) -> list[Contact]:
        return self._contacts.copy()

    def count(self) -> int:
        return len(self._contacts)

    # ----- التصدير -----
    def export_to_csv(self, csv_path: Path) -> None:
        if not self._contacts:
            print("لا توجد جهات اتصال للتصدير.")
            return

        fieldnames = [f.name for f in fields(Contact)]
        with open(csv_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for contact in self._contacts:
                writer.writerow(asdict(contact))
        print(f"✅ تم تصدير {len(self._contacts)} جهة اتصال إلى {csv_path}")

    def import_from_csv(self, csv_path: Path) -> int:
        if not csv_path.exists():
            raise FileNotFoundError(f"ملف الاستيراد غير موجود: {csv_path}")

        imported_count = 0
        with open(csv_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                self._contacts.append(Contact(**row))  # type: ignore[arg-type]
                imported_count += 1
        self.save()
        return imported_count

# ---------------------------------------------------------
# برنامج تجريبي
# ---------------------------------------------------------
def main() -> None:
    manager = ContactManager(Path("contacts.json"))

    manager.add_contact("أحمد علي", "0501234567", "ahmed@example.com", "عمل")
    manager.add_contact("سارة محمد", "0559876543", group="عائلة")
    manager.add_contact("خالد يوسف", "0507654321", "khaled@example.com", "أصدقاء")

    print(f"إجمالي جهات الاتصال: {manager.count()}")
    print()

    print("كل جهات الاتصال:")
    for contact in manager.all_contacts():
        print(contact.display())
    print()

    print("البحث عن 'سارة':")
    for contact in manager.search("سارة"):
        print(contact.display())
    print()

    try:
        manager.find_by_name("غير موجود")
    except ContactNotFoundError as e:
        print(f"خطأ متوقع: {e}")
    print()

    manager.export_to_csv(Path("contacts_export.csv"))

if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [069] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-68
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-035-count_lines.py
"""

# تمرين 1
names: list[str] = [input(f"أدخل الاسم {i + 1}: ") for i in range(3)]

with open("names.txt", "w", encoding="utf-8") as file:
    for name in names:
        file.write(name + "\n")

# تمرين 2
from pathlib import Path

def count_lines(file_path: Path) -> int:
    with open(file_path, "r", encoding="utf-8") as file:
        return sum(1 for _ in file)

print(count_lines(Path("names.txt")))

# تمرين 3
import json
from pathlib import Path

person: dict[str, object] = {"name": "ليلى", "age": 27, "city": "تونس"}

Path("person.json").write_text(
    json.dumps(person, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

loaded = json.loads(Path("person.json").read_text(encoding="utf-8"))
print(loaded)

# ────────────────────────────────────────────────────────────────────────
# [070] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-69
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-036-calculate_total_sales.py
"""

# تمرين 4
import csv
from pathlib import Path

def calculate_total_sales(csv_path: Path) -> float:
    if not csv_path.exists():
        print("ملف المبيعات غير موجود!")
        return 0.0

    total = 0.0
    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            total += float(row["amount"])
    return total

print(calculate_total_sales(Path("sales.csv")))

# تمرين 5
from pathlib import Path
from datetime import datetime

class Logger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, "a", encoding="utf-8") as file:
            file.write(f"[{timestamp}] {message}\n")

logger = Logger(Path("app.log"))
logger.log("بدأ تشغيل التطبيق")
logger.log("تم تسجيل دخول مستخدم جديد")

# تمرين 6
import json
from pathlib import Path

def save_config(config: dict[str, object], path: Path) -> None:
    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def load_config(path: Path) -> dict[str, object]:
    default_config: dict[str, object] = {"theme": "light", "language": "ar"}
    if not path.exists():
        return default_config
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default_config

config = load_config(Path("config.json"))
print(config)
save_config({"theme": "dark", "language": "en"}, Path("config.json"))

# ────────────────────────────────────────────────────────────────────────
# [071] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-70
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-037-backup_file.py
"""

from pathlib import Path
from datetime import datetime
import shutil

def backup_file(source: Path, backup_dir: Path) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"الملف الأصلي غير موجود: {source}")

    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{source.stem}_{timestamp}{source.suffix}"
    backup_path = backup_dir / backup_name

    shutil.copy2(source, backup_path)
    return backup_path

def list_backups(backup_dir: Path, original_name: str) -> list[Path]:
    if not backup_dir.exists():
        return []

    stem = Path(original_name).stem
    suffix = Path(original_name).suffix

    matches = list(backup_dir.glob(f"{stem}_*{suffix}"))
    return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)

def restore_latest(backup_dir: Path, original_name: str, destination: Path) -> bool:
    backups = list_backups(backup_dir, original_name)
    if not backups:
        return False

    latest_backup = backups[0]
    shutil.copy2(latest_backup, destination)
    return True

# تجربة
source_file = Path("notes.txt")
source_file.write_text("محتوى تجريبي أول", encoding="utf-8")

backup_dir = Path("backups")

backup_file(source_file, backup_dir)

import time
time.sleep(1)  # للتأكد من اختلاف الطابع الزمني بين النسختين

source_file.write_text("محتوى تجريبي ثانٍ - مُحدّث", encoding="utf-8")
backup_file(source_file, backup_dir)

backups = list_backups(backup_dir, "notes.txt")
print(f"عدد النسخ الاحتياطية: {len(backups)}")
for b in backups:
    print(f"  {b.name}")

success = restore_latest(backup_dir, "notes.txt", Path("restored_notes.txt"))
print(f"تمت الاستعادة بنجاح: {success}")
print(Path("restored_notes.txt").read_text(encoding="utf-8"))

# ────────────────────────────────────────────────────────────────────────
# [072] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-71
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.10: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-038-content.py
"""

# ===== فتح الملفات النصية =====
with open("file.txt", "r", encoding="utf-8") as f:
    content = f.read()           # الملف كاملاً كنص
    # أو
    lines = f.readlines()        # قائمة بكل الأسطر
    # أو (الأفضل للملفات الكبيرة)
    for line in f:
        print(line.strip())

with open("file.txt", "w", encoding="utf-8") as f:
    f.write("نص جديد\n")         # يمحو المحتوى القديم!

with open("file.txt", "a", encoding="utf-8") as f:
    f.write("إضافة\n")           # يُضيف دون حذف

# ===== pathlib =====
from pathlib import Path

p = Path("data") / "file.txt"
p.exists()          # هل موجود؟
p.is_file()         # هل ملف؟
p.mkdir(parents=True, exist_ok=True)
p.write_text("محتوى", encoding="utf-8")
content = p.read_text(encoding="utf-8")
p.name, p.stem, p.suffix, p.parent

for f in Path(".").glob("*.py"):
    print(f)

# ===== JSON =====
import json

json.dumps(data, ensure_ascii=False, indent=2)   # Python → نص JSON
json.loads(json_text)                             # نص JSON → Python
json.dump(data, file, ensure_ascii=False, indent=2)  # Python → ملف
json.load(file)                                   # ملف → Python

# مع Dataclasses
from dataclasses import asdict
json.dumps(asdict(my_object), ensure_ascii=False)

# ===== CSV =====
import csv

with open("f.csv", "w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["a", "b"])
    writer.writeheader()
    writer.writerows(data)

with open("f.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(row)  # dict[str, str]

# ────────────────────────────────────────────────────────────────────────
# [073] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-72
# ────────────────────────────────────────────────────────────────────────
open(file_path, mode, encoding="utf-8")

# ────────────────────────────────────────────────────────────────────────
# [074] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-73
# ────────────────────────────────────────────────────────────────────────
# ❌ خطير: الاعتماد على الترميز الافتراضي لنظام التشغيل
with open("notes.txt", "r") as file:
    content = file.read()

# ✅ آمن: تحديد الترميز صراحة
with open("notes.txt", "r", encoding="utf-8") as file:
    content = file.read()

# ────────────────────────────────────────────────────────────────────────
# [075] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-74
# ────────────────────────────────────────────────────────────────────────
> # أسلوب LBYL (تحقق قبل التنفيذ)
> import os
> if os.path.exists("file.txt"):
>     with open("file.txt", "r", encoding="utf-8") as f:
>         content = f.read()
> else:
>     print(«الملف غير موجود»)
>
> # أسلوب EAFP (الأسلوب الـ Pythonic الموصى به)
> try:
>     with open("file.txt", "r", encoding="utf-8") as f:
>         content = f.read()
> except FileNotFoundError:
>     print(«الملف غير موجود»)
> ```
> الأسلوب الثاني أفضل لأنه **آمن من سباقات التوقيت (Race Conditions)**: قد يُحذف الملف تمامًا في اللحظة الفاصلة بين `os.path.exists()` واستدعاء `open()`!

---

## 16.3 `pathlib`: الأسلوب الحديث للتعامل مع المسارات

في الإصدارات القديمة من Python، كان التعامل مع مسارات الملفات يتم عبر وحدة `os.path`، باستخدام نصوص عادية ودمجها يدويًا. هذا الأسلوب كان عرضة لمشاكل كثيرة (خاصة اختلاف فاصل المسارات بين Windows `\` ولينكس/ماك `/`).

منذ Python 3.4، أصبح لدينا `pathlib`: وحدة حديثة تُمثّل المسارات ككائنات حقيقية (Objects)، بدلًا من مجرد نصوص. هذا هو **الأسلوب الموصى به حصريًا** في هذا الكتاب.

### إنشاء كائن `Path`

# ────────────────────────────────────────────────────────────────────────
# [076] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-16-chapter-16.ipynb#cell-75
# ────────────────────────────────────────────────────────────────────────
> 🔄 **مقارنة: `os.path` القديمة مقابل `pathlib` الحديثة**
> ```python
> # الأسلوب القديم (os.path) - لا تستخدمه في كود جديد
> import os
> path = os.path.join("data", "2024", "report.txt")
> if os.path.exists(path):
>     print(os.path.getsize(path))
>
> # الأسلوب الحديث (pathlib) - الموصى به
> from pathlib import Path
> path = Path("data") / "2024" / "report.txt"
> if path.exists():
>     print(path.stat().st_size)
> ```
> `pathlib` أكثر قابلية للقراءة، وتعمل بنفس الطريقة على كل أنظمة التشغيل، وتجمع كل الوظائف ذات الصلة بالمسار في كائن واحد متماسك (Object-Oriented)، بدلًا من دوال متفرقة.

### خصائص ودوال مفيدة في `Path`
