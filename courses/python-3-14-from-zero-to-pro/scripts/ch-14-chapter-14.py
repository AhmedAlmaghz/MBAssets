#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — البرمجة الكائنية
Python 3.14 — From Zero to Pro — Chapter 14 — Functions and Data Structures

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-14-chapter-14 (lesson 14)
Code files: 61 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/python-3-14-from-zero-to-pro/code/ch-14-chapter-14.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — البرمجة الكائنية
Python 3.14 — From Zero to Pro — Chapter 14 — Functions and Data Structures

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-14-chapter-14 (lesson 14)
Code files: 60 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] dist/notebooks/chapter-14.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# ────────────────────────────────────────────────────────────────────────
# [002] dist/notebooks/chapter-14.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
class Product:
    def __init__(self, name: str, price: float):
        self.name = name
        self.price = price

    def display_info(self) -> str:
        return f"{self.name}: {self.price} ر.س"

class Book(Product):  # Book ترث من Product
    def __init__(self, name: str, price: float, author: str):
        super().__init__(name, price)  # استدعاء __init__ من الفئة الأساسية
        self.author = author

    def display_info(self) -> str:
        return f"{super().display_info()} - تأليف: {self.author}"

book = Book("بايثون للمحترفين", 120, "محمد أحمد")
print(book.display_info())
# بايثون للمحترفين: 120 ر.س - تأليف: محمد أحمد

# ────────────────────────────────────────────────────────────────────────
# [003] dist/notebooks/chapter-14.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class Animal:  # Parent / Base / Super Class
    def __init__(self, name: str):
        self.name = name

    def make_sound(self) -> str:
        return "صوت ما..."

    def __str__(self) -> str:
        return f"{self.name} يصدر صوت: {self.make_sound()}"

class Dog(Animal):  # Child / Derived / Sub Class
    def make_sound(self) -> str:  # Override - استبدال السلوك
        return "واق واق!"

class Cat(Animal):
    def make_sound(self) -> str:
        return "مياو!"

dog = Dog("ريكس")
cat = Cat("لونا")

print(dog)  # ريكس يصدر صوت: واق واق!
print(cat)  # لونا يصدر صوت: مياو!

# ────────────────────────────────────────────────────────────────────────
# [004] dist/notebooks/chapter-14.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class Employee:
    def __init__(self, name: str, salary: float):
        self.name = name
        self.salary = salary

    def display(self) -> str:
        return f"{self.name}: {self.salary} ر.س"

class Manager(Employee):
    def __init__(self, name: str, salary: float, team_size: int):
        super().__init__(name, salary)  # استدعاء __init__ الأساسي أولاً
        self.team_size = team_size

    def display(self) -> str:
        base_info = super().display()  # استدعاء display الأساسي، ثم إضافة عليه
        return f"{base_info} (يدير فريق من {self.team_size} أشخاص)"

manager = Manager("نورة", 15000, 8)
print(manager.display())
# نورة: 15000 ر.س (يدير فريق من 8 أشخاص)

# ────────────────────────────────────────────────────────────────────────
# [005] dist/notebooks/chapter-14.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    class Manager(Employee):
        def __init__(self, name, salary, team_size):
            # ⚠️ نسيان super().__init__(name, salary)!
            self.team_size = team_size

    manager = Manager("نورة", 15000, 8)
    print(manager.name)
    # 💥 AttributeError: 'Manager' object has no attribute 'name'
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [006] dist/notebooks/chapter-14.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
print(isinstance(dog, Dog))     # True
print(isinstance(dog, Animal))  # True - Dog هي أيضاً Animal!
print(isinstance(dog, Cat))     # False

print(issubclass(Dog, Animal))  # True
print(issubclass(Animal, Dog))  # False - العلاقة أحادية الاتجاه

# ────────────────────────────────────────────────────────────────────────
# [007] dist/notebooks/chapter-14.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
class Vehicle:
    def __init__(self, brand: str):
        self.brand = brand

    def describe(self) -> str:
        return f"مركبة من {self.brand}"

class Car(Vehicle):
    def __init__(self, brand: str, doors: int):
        super().__init__(brand)
        self.doors = doors

    def describe(self) -> str:
        return f"{super().describe()}, عدد الأبواب: {self.doors}"

class SportsCar(Car):
    def __init__(self, brand: str, doors: int, top_speed: int):
        super().__init__(brand, doors)
        self.top_speed = top_speed

    def describe(self) -> str:
        return f"{super().describe()}, السرعة القصوى: {self.top_speed} كم/س"

ferrari = SportsCar("Ferrari", 2, 340)
print(ferrari.describe())
# مركبة من Ferrari, عدد الأبواب: 2, السرعة القصوى: 340 كم/س

print(isinstance(ferrari, Vehicle))   # True
print(isinstance(ferrari, Car))       # True
print(isinstance(ferrari, SportsCar)) # True

# ────────────────────────────────────────────────────────────────────────
# [008] dist/notebooks/chapter-14.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
class Swimmer:
    def swim(self) -> str:
        return "يسبح 🏊"

class Flyer:
    def fly(self) -> str:
        return "يطير 🦅"

class Duck(Swimmer, Flyer):  # وراثة من فئتين معاً
    def __init__(self, name: str):
        self.name = name

duck = Duck("دونالد")
print(duck.swim())  # يسبح 🏊
print(duck.fly())   # يطير 🦅

# ────────────────────────────────────────────────────────────────────────
# [009] dist/notebooks/chapter-14.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
print(Duck.__mro__)
# (<class 'Duck'>, <class 'Swimmer'>, <class 'Flyer'>, <class 'object'>)

# ────────────────────────────────────────────────────────────────────────
# [010] dist/notebooks/chapter-14.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
class Shape:
    def area(self) -> float:
        raise NotImplementedError("يجب تخصيص هذه الدالة في الفئة الفرعية")

class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius

    def area(self) -> float:
        return 3.14159 * self.radius ** 2

class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

class Triangle(Shape):
    def __init__(self, base: float, height: float):
        self.base = base
        self.height = height

    def area(self) -> float:
        return 0.5 * self.base * self.height

# هنا يكمن سحر تعدد الأشكال:
shapes: list[Shape] = [Circle(5), Rectangle(4, 6), Triangle(3, 8)]

for shape in shapes:
    print(f"{type(shape).__name__}: المساحة = {shape.area():.2f}")

# ────────────────────────────────────────────────────────────────────────
# [011] dist/notebooks/chapter-14.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
class Duck:
    def make_sound(self) -> str:
        return "Quack!"

class Robot:
    def make_sound(self) -> str:
        return "Beep Boop!"

def animate(entity) -> None:
    """لا تهتم بنوع entity، فقط تحتاج أن تملك make_sound()"""
    print(entity.make_sound())

animate(Duck())    # Quack!
animate(Robot())   # Beep Boop! -- Robot ليست حيواناً إطلاقاً لكنها تعمل!

# ────────────────────────────────────────────────────────────────────────
# [012] dist/notebooks/chapter-14.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    from abc import ABC, abstractmethod

    class Shape(ABC):  # ترث من ABC (Abstract Base Class)
        @abstractmethod
        def area(self) -> float:
            """يجب أن تُخصَّص في كل فئة فرعية."""
            pass

        @abstractmethod
        def perimeter(self) -> float:
            """يجب أن تُخصَّص في كل فئة فرعية."""
            pass

        def describe(self) -> str:
            """method عادية - موروثة كما هي، لا حاجة لتخصيصها."""
            return f"{type(self).__name__}: مساحة={self.area():.2f}, محيط={self.perimeter():.2f}"

    class Circle(Shape):
        def __init__(self, radius: float):
            self.radius = radius

        def area(self) -> float:
            return 3.14159 * self.radius ** 2

        def perimeter(self) -> float:
            return 2 * 3.14159 * self.radius

    # shape = Shape()
    # 💥 TypeError: Can't instantiate abstract class Shape with abstract methods area, perimeter

    circle = Circle(5)
    print(circle.describe())  # Circle: مساحة=78.54, محيط=31.42

    class Square(Shape):
        def __init__(self, side: float):
            self.side = side
        # ⚠️ لو نسينا تعريف area() أو perimeter() هنا:

    # square = Square(5)
    # 💥 TypeError: Can't instantiate abstract class Square with abstract method area, perimeter
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [013] dist/notebooks/chapter-14.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
# الطريقة التقليدية - كثير من الكود المتكرر لفئة بسيطة تخزن بيانات فقط
class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"Point(x={self.x}, y={self.y})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y

# ────────────────────────────────────────────────────────────────────────
# [014] dist/notebooks/chapter-14.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

p1 = Point(3, 4)
p2 = Point(3, 4)

print(p1)          # Point(x=3, y=4)  -- __repr__ تلقائياً!
print(p1 == p2)     # True  -- __eq__ تلقائياً!

# ────────────────────────────────────────────────────────────────────────
# [015] dist/notebooks/chapter-14.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass, field

@dataclass
class Student:
    name: str
    age: int = 18  # قيمة افتراضية بسيطة
    grades: list[float] = field(default_factory=list)  # للقيم القابلة للتعديل!

s1 = Student("أحمد")
s1.grades.append(90)
print(s1)  # Student(name='أحمد', age=18, grades=[90])

s2 = Student("سارة")
print(s2)  # Student(name='سارة', age=18, grades=[])  -- مستقلة تماماً!

# ────────────────────────────────────────────────────────────────────────
# [016] dist/notebooks/chapter-14.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    @dataclass
    class Student:
        grades: list = []  # 💥 ValueError: mutable default <class 'list'> is not allowed
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [017] dist/notebooks/chapter-14.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass

@dataclass
class Rectangle:
    width: float
    height: float

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)

rect = Rectangle(5, 3)
print(rect)             # Rectangle(width=5, height=3)
print(rect.area())      # 15

# ────────────────────────────────────────────────────────────────────────
# [018] dist/notebooks/chapter-14.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    from dataclasses import dataclass

    @dataclass(frozen=True)  # يجعل الكائن immutable - كالـ tuple!
    class ImmutablePoint:
        x: float
        y: float

    p = ImmutablePoint(3, 4)
    # p.x = 10
    # 💥 FrozenInstanceError: cannot assign to field 'x'

    @dataclass(order=True)  # يُضيف __lt__, __le__, __gt__, __ge__ تلقائياً
    class Version:
        major: int
        minor: int

    v1 = Version(1, 5)
    v2 = Version(2, 0)
    print(v1 < v2)  # True -- تقارن الحقول بالترتيب المُعرَّف
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [019] dist/notebooks/chapter-14.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
class Multiplier:
    def __init__(self, factor: float):
        self.factor = factor

    def __call__(self, value: float) -> float:
        return value * self.factor

double = Multiplier(2)
print(double(5))   # 10 -- الكائن نفسه يُستدعى كدالة!
print(double(10))  # 20

# ────────────────────────────────────────────────────────────────────────
# [020] dist/notebooks/chapter-14.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
class Playlist:
    def __init__(self, songs: list[str]):
        self.songs = songs

    def __getitem__(self, index: int) -> str:
        return self.songs[index]

    def __len__(self) -> int:
        return len(self.songs)

playlist = Playlist(["أغنية 1", "أغنية 2", "أغنية 3"])
print(playlist[0])  # أغنية 1  -- الفهرسة تعمل الآن!

for song in playlist:  # __getitem__ تجعل الكائن قابلاً للتكرار تلقائياً أيضاً!
    print(song)

# ────────────────────────────────────────────────────────────────────────
# [021] dist/notebooks/chapter-14.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
class Timer:
    """context manager يقيس زمن تنفيذ كتلة كود."""

    def __enter__(self):
        import time
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        import time
        self.end = time.perf_counter()
        print(f"⏱️ استغرق التنفيذ: {self.end - self.start:.4f} ثانية")

with Timer():
    total = sum(i ** 2 for i in range(1_000_000))
    print(f"النتيجة: {total}")

# ────────────────────────────────────────────────────────────────────────
# [022] dist/notebooks/chapter-14.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto

# ============ الطبقة 1: نماذج البيانات (Dataclasses) ============

class ItemStatus(Enum):
    """حالة العنصر في المكتبة."""
    AVAILABLE = auto()
    BORROWED = auto()
    RESERVED = auto()

@dataclass
class Member:
    """يمثل عضواً في المكتبة."""
    member_id: int
    name: str
    borrowed_items: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"عضو #{self.member_id}: {self.name}"

# ============ الطبقة 2: الفئة المجردة للعناصر القابلة للاستعارة ============

class LibraryItem(ABC):
    """فئة مجردة أساسية لأي عنصر في المكتبة."""

    def __init__(self, item_id: str, title: str, loan_days: int):
        self.item_id = item_id
        self.title = title
        self.loan_days = loan_days  # عدد الأيام المسموحة للاستعارة
        self.status = ItemStatus.AVAILABLE
        self.borrowed_by: Member | None = None
        self.due_date: datetime | None = None

    @abstractmethod
    def get_category(self) -> str:
        """كل فئة فرعية يجب أن تحدد فئتها."""
        pass

    @abstractmethod
    def display_details(self) -> str:
        """كل فئة فرعية تعرض تفاصيلها الخاصة."""
        pass

    def borrow(self, member: Member) -> bool:
        """استعارة العنصر - سلوك مشترك لكل الأنواع."""
        if self.status != ItemStatus.AVAILABLE:
            print(f"❌ '{self.title}' غير متاح حالياً (الحالة: {self.status.name}).")
            return False

        self.status = ItemStatus.BORROWED
        self.borrowed_by = member
        self.due_date = datetime.now() + timedelta(days=self.loan_days)
        member.borrowed_items.append(self.item_id)

        print(f"✅ استعار {member.name} '{self.title}'. "
              f"الإرجاع المتوقع: {self.due_date.strftime('%Y-%m-%d')}")
        return True

    def return_item(self) -> None:
        """إرجاع العنصر."""
        if self.status != ItemStatus.BORROWED:
            print(f"⚠️ '{self.title}' ليس مُستعاراً أصلاً.")
            return

        if self.borrowed_by:
            self.borrowed_by.borrowed_items.remove(self.item_id)
            print(f"✅ تم إرجاع '{self.title}' من {self.borrowed_by.name}.")

        self.status = ItemStatus.AVAILABLE
        self.borrowed_by = None
        self.due_date = None

    def is_overdue(self) -> bool:
        """التحقق من تأخر الإرجاع - سلوك مشترك."""
        if self.status != ItemStatus.BORROWED or self.due_date is None:
            return False
        return datetime.now() > self.due_date

    def __str__(self) -> str:
        status_display = self.status.name
        return f"[{self.get_category()}] {self.title} ({self.item_id}) - {status_display}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, LibraryItem):
            return False
        return self.item_id == other.item_id

    def __hash__(self) -> int:
        return hash(self.item_id)

# ============ الطبقة 3: الفئات الفرعية (تطبيق تعدد الأشكال) ============

class Book(LibraryItem):
    def __init__(self, item_id: str, title: str, author: str, pages: int):
        super().__init__(item_id, title, loan_days=14)  # الكتب تُستعار 14 يوماً
        self.author = author
        self.pages = pages

    def get_category(self) -> str:
        return "كتاب"

    def display_details(self) -> str:
        return f"📖 {self.title} - تأليف: {self.author} ({self.pages} صفحة)"

class DVD(LibraryItem):
    def __init__(self, item_id: str, title: str, duration_minutes: int):
        super().__init__(item_id, title, loan_days=5)  # الأفلام تُستعار 5 أيام فقط
        self.duration_minutes = duration_minutes

    def get_category(self) -> str:
        return "دي في دي"

    def display_details(self) -> str:
        return f"🎬 {self.title} - المدة: {self.duration_minutes} دقيقة"

class Magazine(LibraryItem):
    def __init__(self, item_id: str, title: str, issue_number: int):
        super().__init__(item_id, title, loan_days=3)  # المجلات 3 أيام فقط
        self.issue_number = issue_number

    def get_category(self) -> str:
        return "مجلة"

    def display_details(self) -> str:
        return f"📰 {self.title} - العدد: {self.issue_number}"

# ============ الطبقة 4: استثناءات مخصصة ============

class LibraryError(Exception):
    """الفئة الأساسية لكل أخطاء نظام المكتبة."""
    pass

class ItemNotFoundError(LibraryError):
    def __init__(self, item_id: str):
        super().__init__(f"العنصر '{item_id}' غير موجود في المكتبة.")

class MemberNotFoundError(LibraryError):
    def __init__(self, member_id: int):
        super().__init__(f"العضو رقم {member_id} غير موجود.")

# ============ الطبقة 5: النظام الرئيسي ============

class Library:
    """يُدير كل عمليات المكتبة: الأعضاء، العناصر، الاستعارة، الإرجاع."""

    def __init__(self, name: str):
        self.name = name
        self.items: dict[str, LibraryItem] = {}
        self.members: dict[int, Member] = {}

    # --- إدارة العناصر ---
    def add_item(self, item: LibraryItem) -> None:
        self.items[item.item_id] = item
        print(f"📚 أُضيف '{item.title}' إلى مكتبة '{self.name}'.")

    def find_item(self, item_id: str) -> LibraryItem:
        try:
            return self.items[item_id]
        except KeyError:
            raise ItemNotFoundError(item_id)

    # --- إدارة الأعضاء ---
    def register_member(self, member: Member) -> None:
        self.members[member.member_id] = member
        print(f"👤 تم تسجيل {member.name} كعضو جديد.")

    def find_member(self, member_id: int) -> Member:
        try:
            return self.members[member_id]
        except KeyError:
            raise MemberNotFoundError(member_id)

    # --- العمليات الأساسية ---
    def borrow_item(self, item_id: str, member_id: int) -> None:
        item = self.find_item(item_id)
        member = self.find_member(member_id)
        item.borrow(member)

    def return_item(self, item_id: str) -> None:
        item = self.find_item(item_id)
        item.return_item()

    # --- تقارير (تطبيق تعدد الأشكال بوضوح) ---
    def list_all_items(self) -> None:
        print(f"\n📋 كل عناصر مكتبة '{self.name}':")
        print("-" * 60)
        for item in self.items.values():
            print(item.display_details())  # كل نوع يعرض تفاصيله الخاصة!
        print("-" * 60)

    def list_available_items(self) -> list[LibraryItem]:
        return [item for item in self.items.values()
                if item.status == ItemStatus.AVAILABLE]

    def list_overdue_items(self) -> list[LibraryItem]:
        return [item for item in self.items.values() if item.is_overdue()]

    def items_by_category(self) -> dict[str, int]:
        """عدد العناصر حسب الفئة - باستخدام polymorphism (get_category)."""
        counts: dict[str, int] = {}
        for item in self.items.values():
            category = item.get_category()
            counts[category] = counts.get(category, 0) + 1
        return counts

# ============ التشغيل التجريبي ============

if __name__ == "__main__":
    library = Library("مكتبة الأمل العامة")

    # إضافة عناصر متنوعة (تعدد الأشكال في العمل)
    library.add_item(Book("B001", "بايثون للمحترفين", "محمد أحمد", 350))
    library.add_item(Book("B002", "Clean Code", "Robert Martin", 400))
    library.add_item(DVD("D001", "رحلة إلى الفضاء", 120))
    library.add_item(Magazine("M001", "التقنية اليوم", 45))

    # تسجيل أعضاء
    library.register_member(Member(1, "أحمد"))
    library.register_member(Member(2, "سارة"))

    print()
    library.list_all_items()

    print("\n--- عمليات الاستعارة ---")
    library.borrow_item("B001", 1)
    library.borrow_item("D001", 2)

    try:
        library.borrow_item("B001", 2)  # محاولة استعارة كتاب مُستعار بالفعل
    except LibraryError as e:
        print(f"❌ {e}")

    try:
        library.borrow_item("X999", 1)  # عنصر غير موجود
    except ItemNotFoundError as e:
        print(f"❌ {e}")

    print("\n--- حالة المكتبة ---")
    library.list_all_items()

    available = library.list_available_items()
    print(f"\n✅ العناصر المتاحة حالياً: {len(available)}")
    for item in available:
        print(f"  - {item}")

    print(f"\n📊 توزيع العناصر حسب الفئة: {library.items_by_category()}")

    print("\n--- إرجاع عنصر ---")
    library.return_item("B001")

    library.list_all_items()

# ────────────────────────────────────────────────────────────────────────
# [023] dist/notebooks/chapter-14.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
# تمرين 1
class Shape:
    def area(self) -> float:
        return 0

class Square(Shape):
    def __init__(self, side: float):
        self.side = side

    def area(self) -> float:
        return self.side ** 2

sq = Square(4)
print(sq.area())  # 16

# تمرين 2
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

p1 = Point(1, 2)
p2 = Point(1, 2)
print(p1)          # Point(x=1, y=2)
print(p1 == p2)     # True

# ────────────────────────────────────────────────────────────────────────
# [024] dist/notebooks/chapter-14.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    # تمرين 3
    from abc import ABC, abstractmethod

    class PaymentMethod(ABC):
        @abstractmethod
        def process(self, amount: float) -> None:
            pass

    class CreditCard(PaymentMethod):
        def process(self, amount: float) -> None:
            print(f"💳 تمت معالجة {amount} ر.س عبر البطاقة الائتمانية.")

    class PayPal(PaymentMethod):
        def process(self, amount: float) -> None:
            print(f"🅿️ تمت معالجة {amount} ر.س عبر PayPal.")

    methods: list[PaymentMethod] = [CreditCard(), PayPal()]
    for method in methods:
        method.process(100)

    # تمرين 4
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Product:
        name: str
        price: float
        quantity: int = 1

        def total(self) -> float:
            return self.price * self.quantity

    p = Product("لابتوب", 3500, 2)
    print(p.total())  # 7000
    # p.price = 4000  # 💥 FrozenInstanceError
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [025] dist/notebooks/chapter-14.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
from abc import ABC, abstractmethod

class Employee(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def calculate_salary(self) -> float:
        pass

    def __str__(self) -> str:
        return f"{self.name}: {self.calculate_salary():.2f} ر.س"

class FullTimeEmployee(Employee):
    def __init__(self, name: str, monthly_salary: float):
        super().__init__(name)
        self.monthly_salary = monthly_salary

    def calculate_salary(self) -> float:
        return self.monthly_salary

class FreelanceEmployee(Employee):
    def __init__(self, name: str, hourly_rate: float, hours_worked: float):
        super().__init__(name)
        self.hourly_rate = hourly_rate
        self.hours_worked = hours_worked

    def calculate_salary(self) -> float:
        return self.hourly_rate * self.hours_worked

class Payroll:
    def __init__(self):
        self.employees: list[Employee] = []

    def add_employee(self, employee: Employee) -> None:
        self.employees.append(employee)

    def total_payroll(self) -> float:
        return sum(emp.calculate_salary() for emp in self.employees)

    def print_report(self) -> None:
        print("📋 تقرير الرواتب:")
        for emp in self.employees:
            print(f"  {emp}")
        print(f"الإجمالي: {self.total_payroll():.2f} ر.س")

payroll = Payroll()
payroll.add_employee(FullTimeEmployee("أحمد", 8000))
payroll.add_employee(FreelanceEmployee("سارة", 50, 120))
payroll.print_report()

# 📋 تقرير الرواتب:
#   أحمد: 8000.00 ر.س
#   سارة: 6000.00 ر.س
# الإجمالي: 14000.00 ر.س

# ────────────────────────────────────────────────────────────────────────
# [026] code/chapters/ch14/14-001-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.1: لماذا الوراثة أحد أهم اختراعات البرمجة الكائنية؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-001-__init__.py
"""

class Product:
    def __init__(self, name: str, price: float):
        self.name = name
        self.price = price

    def display_info(self) -> str:
        return f"{self.name}: {self.price} ر.س"

class Book(Product):  # Book ترث من Product
    def __init__(self, name: str, price: float, author: str):
        super().__init__(name, price)  # استدعاء __init__ من الفئة الأساسية
        self.author = author

    def display_info(self) -> str:
        return f"{super().display_info()} - تأليف: {self.author}"

book = Book("بايثون للمحترفين", 120, "محمد أحمد")
print(book.display_info())
# بايثون للمحترفين: 120 ر.س - تأليف: محمد أحمد

# ────────────────────────────────────────────────────────────────────────
# [027] code/chapters/ch14/14-002-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.2: الوراثة (Inheritance) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-002-__init__.py
"""

class Animal:  # Parent / Base / Super Class
    def __init__(self, name: str):
        self.name = name

    def make_sound(self) -> str:
        return "صوت ما..."

    def __str__(self) -> str:
        return f"{self.name} يصدر صوت: {self.make_sound()}"

class Dog(Animal):  # Child / Derived / Sub Class
    def make_sound(self) -> str:  # Override - استبدال السلوك
        return "واق واق!"

class Cat(Animal):
    def make_sound(self) -> str:
        return "مياو!"

dog = Dog("ريكس")
cat = Cat("لونا")

print(dog)  # ريكس يصدر صوت: واق واق!
print(cat)  # لونا يصدر صوت: مياو!

# ────────────────────────────────────────────────────────────────────────
# [028] code/chapters/ch14/14-003-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.2: الوراثة (Inheritance) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-003-__init__.py
"""

class Employee:
    def __init__(self, name: str, salary: float):
        self.name = name
        self.salary = salary

    def display(self) -> str:
        return f"{self.name}: {self.salary} ر.س"

class Manager(Employee):
    def __init__(self, name: str, salary: float, team_size: int):
        super().__init__(name, salary)  # استدعاء __init__ الأساسي أولاً
        self.team_size = team_size

    def display(self) -> str:
        base_info = super().display()  # استدعاء display الأساسي، ثم إضافة عليه
        return f"{base_info} (يدير فريق من {self.team_size} أشخاص)"

manager = Manager("نورة", 15000, 8)
print(manager.display())
# نورة: 15000 ر.س (يدير فريق من 8 أشخاص)

# ────────────────────────────────────────────────────────────────────────
# [029] code/chapters/ch14/14-004-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.2: الوراثة (Inheritance) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-004-__init__.py
"""

class Manager(Employee):
    def __init__(self, name, salary, team_size):
        # ⚠️ نسيان super().__init__(name, salary)!
        self.team_size = team_size

manager = Manager("نورة", 15000, 8)
print(manager.name)
# 💥 AttributeError: 'Manager' object has no attribute 'name'

# ────────────────────────────────────────────────────────────────────────
# [030] code/chapters/ch14/14-005-ex14_2.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.2: الوراثة (Inheritance) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-005-ex14_2.py
"""

print(isinstance(dog, Dog))     # True
print(isinstance(dog, Animal))  # True - Dog هي أيضاً Animal!
print(isinstance(dog, Cat))     # False

print(issubclass(Dog, Animal))  # True
print(issubclass(Animal, Dog))  # False - العلاقة أحادية الاتجاه

# ────────────────────────────────────────────────────────────────────────
# [031] code/chapters/ch14/14-006-ex14_2.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.2: الوراثة (Inheritance) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-006-ex14_2.py
"""

# ❌ لا يأخذ الوراثة في الحسبان
if type(dog) == Animal:  # False حتى لو dog هو كلب (نوع فرعي من Animal)!
    pass

# ✅ يأخذ الوراثة في الحسبان
if isinstance(dog, Animal):  # True
    pass

# ────────────────────────────────────────────────────────────────────────
# [032] code/chapters/ch14/14-007-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.2: الوراثة (Inheritance) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-007-__init__.py
"""

class Vehicle:
    def __init__(self, brand: str):
        self.brand = brand

    def describe(self) -> str:
        return f"مركبة من {self.brand}"

class Car(Vehicle):
    def __init__(self, brand: str, doors: int):
        super().__init__(brand)
        self.doors = doors

    def describe(self) -> str:
        return f"{super().describe()}, عدد الأبواب: {self.doors}"

class SportsCar(Car):
    def __init__(self, brand: str, doors: int, top_speed: int):
        super().__init__(brand, doors)
        self.top_speed = top_speed

    def describe(self) -> str:
        return f"{super().describe()}, السرعة القصوى: {self.top_speed} كم/س"

ferrari = SportsCar("Ferrari", 2, 340)
print(ferrari.describe())
# مركبة من Ferrari, عدد الأبواب: 2, السرعة القصوى: 340 كم/س

print(isinstance(ferrari, Vehicle))   # True
print(isinstance(ferrari, Car))       # True
print(isinstance(ferrari, SportsCar)) # True

# ────────────────────────────────────────────────────────────────────────
# [033] code/chapters/ch14/14-008-swim.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.2: الوراثة (Inheritance) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-008-swim.py
"""

class Swimmer:
    def swim(self) -> str:
        return "يسبح 🏊"

class Flyer:
    def fly(self) -> str:
        return "يطير 🦅"

class Duck(Swimmer, Flyer):  # وراثة من فئتين معاً
    def __init__(self, name: str):
        self.name = name

duck = Duck("دونالد")
print(duck.swim())  # يسبح 🏊
print(duck.fly())   # يطير 🦅

# ────────────────────────────────────────────────────────────────────────
# [034] code/chapters/ch14/14-009-ex14_2.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.2: الوراثة (Inheritance) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-009-ex14_2.py
"""

print(Duck.__mro__)
# (<class 'Duck'>, <class 'Swimmer'>, <class 'Flyer'>, <class 'object'>)

# ────────────────────────────────────────────────────────────────────────
# [035] code/chapters/ch14/14-010-area.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.3: تعدد الأشكال (Polymorphism)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-010-area.py
"""

class Shape:
    def area(self) -> float:
        raise NotImplementedError("يجب تخصيص هذه الدالة في الفئة الفرعية")

class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius

    def area(self) -> float:
        return 3.14159 * self.radius ** 2

class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

class Triangle(Shape):
    def __init__(self, base: float, height: float):
        self.base = base
        self.height = height

    def area(self) -> float:
        return 0.5 * self.base * self.height

# هنا يكمن سحر تعدد الأشكال:
shapes: list[Shape] = [Circle(5), Rectangle(4, 6), Triangle(3, 8)]

for shape in shapes:
    print(f"{type(shape).__name__}: المساحة = {shape.area():.2f}")

# ────────────────────────────────────────────────────────────────────────
# [036] code/chapters/ch14/14-011-make_sound.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.3: تعدد الأشكال (Polymorphism)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-011-make_sound.py
"""

class Duck:
    def make_sound(self) -> str:
        return "Quack!"

class Robot:
    def make_sound(self) -> str:
        return "Beep Boop!"

def animate(entity) -> None:
    """لا تهتم بنوع entity، فقط تحتاج أن تملك make_sound()"""
    print(entity.make_sound())

animate(Duck())    # Quack!
animate(Robot())   # Beep Boop! -- Robot ليست حيواناً إطلاقاً لكنها تعمل!

# ────────────────────────────────────────────────────────────────────────
# [037] code/chapters/ch14/14-012-area.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.4: الفئات المجردة (Abstract Base Classes)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-012-area.py
"""

from abc import ABC, abstractmethod

class Shape(ABC):  # ترث من ABC (Abstract Base Class)
    @abstractmethod
    def area(self) -> float:
        """يجب أن تُخصَّص في كل فئة فرعية."""
        pass

    @abstractmethod
    def perimeter(self) -> float:
        """يجب أن تُخصَّص في كل فئة فرعية."""
        pass

    def describe(self) -> str:
        """method عادية - موروثة كما هي، لا حاجة لتخصيصها."""
        return f"{type(self).__name__}: مساحة={self.area():.2f}, محيط={self.perimeter():.2f}"

class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius

    def area(self) -> float:
        return 3.14159 * self.radius ** 2

    def perimeter(self) -> float:
        return 2 * 3.14159 * self.radius

# shape = Shape()
# 💥 TypeError: Can't instantiate abstract class Shape with abstract methods area, perimeter

circle = Circle(5)
print(circle.describe())  # Circle: مساحة=78.54, محيط=31.42

class Square(Shape):
    def __init__(self, side: float):
        self.side = side
    # ⚠️ لو نسينا تعريف area() أو perimeter() هنا:

# square = Square(5)
# 💥 TypeError: Can't instantiate abstract class Square with abstract method area, perimeter

# ────────────────────────────────────────────────────────────────────────
# [038] code/chapters/ch14/14-013-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.5: Dataclasses: اختصار كتابة الفئات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-013-__init__.py
"""

# الطريقة التقليدية - كثير من الكود المتكرر لفئة بسيطة تخزن بيانات فقط
class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"Point(x={self.x}, y={self.y})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y

# ────────────────────────────────────────────────────────────────────────
# [039] code/chapters/ch14/14-014-point.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.5: Dataclasses: اختصار كتابة الفئات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-014-point.py
"""

from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

p1 = Point(3, 4)
p2 = Point(3, 4)

print(p1)          # Point(x=3, y=4)  -- __repr__ تلقائياً!
print(p1 == p2)     # True  -- __eq__ تلقائياً!

# ────────────────────────────────────────────────────────────────────────
# [040] code/chapters/ch14/14-015-student.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.5: Dataclasses: اختصار كتابة الفئات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-015-student.py
"""

from dataclasses import dataclass, field

@dataclass
class Student:
    name: str
    age: int = 18  # قيمة افتراضية بسيطة
    grades: list[float] = field(default_factory=list)  # للقيم القابلة للتعديل!

s1 = Student("أحمد")
s1.grades.append(90)
print(s1)  # Student(name='أحمد', age=18, grades=[90])

s2 = Student("سارة")
print(s2)  # Student(name='سارة', age=18, grades=[])  -- مستقلة تماماً!

# ────────────────────────────────────────────────────────────────────────
# [041] code/chapters/ch14/14-016-student.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.5: Dataclasses: اختصار كتابة الفئات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-016-student.py
"""

@dataclass
class Student:
    grades: list = []  # 💥 ValueError: mutable default <class 'list'> is not allowed

# ────────────────────────────────────────────────────────────────────────
# [042] code/chapters/ch14/14-017-area.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.5: Dataclasses: اختصار كتابة الفئات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-017-area.py
"""

from dataclasses import dataclass

@dataclass
class Rectangle:
    width: float
    height: float

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)

rect = Rectangle(5, 3)
print(rect)             # Rectangle(width=5, height=3)
print(rect.area())      # 15

# ────────────────────────────────────────────────────────────────────────
# [043] code/chapters/ch14/14-018-immutable_point.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.5: Dataclasses: اختصار كتابة الفئات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-018-immutable_point.py
"""

from dataclasses import dataclass

@dataclass(frozen=True)  # يجعل الكائن immutable - كالـ tuple!
class ImmutablePoint:
    x: float
    y: float

p = ImmutablePoint(3, 4)
# p.x = 10
# 💥 FrozenInstanceError: cannot assign to field 'x'

@dataclass(order=True)  # يُضيف __lt__, __le__, __gt__, __ge__ تلقائياً
class Version:
    major: int
    minor: int

v1 = Version(1, 5)
v2 = Version(2, 0)
print(v1 < v2)  # True -- تقارن الحقول بالترتيب المُعرَّف

# ────────────────────────────────────────────────────────────────────────
# [044] code/chapters/ch14/14-019-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.6: مزيد من Magic Methods: `__call__`, `__getitem__`, Context Managers

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-019-__init__.py
"""

class Multiplier:
    def __init__(self, factor: float):
        self.factor = factor

    def __call__(self, value: float) -> float:
        return value * self.factor

double = Multiplier(2)
print(double(5))   # 10 -- الكائن نفسه يُستدعى كدالة!
print(double(10))  # 20

# ────────────────────────────────────────────────────────────────────────
# [045] code/chapters/ch14/14-020-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.6: مزيد من Magic Methods: `__call__`, `__getitem__`, Context Managers

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-020-__init__.py
"""

class Playlist:
    def __init__(self, songs: list[str]):
        self.songs = songs

    def __getitem__(self, index: int) -> str:
        return self.songs[index]

    def __len__(self) -> int:
        return len(self.songs)

playlist = Playlist(["أغنية 1", "أغنية 2", "أغنية 3"])
print(playlist[0])  # أغنية 1  -- الفهرسة تعمل الآن!

for song in playlist:  # __getitem__ تجعل الكائن قابلاً للتكرار تلقائياً أيضاً!
    print(song)

# ────────────────────────────────────────────────────────────────────────
# [046] code/chapters/ch14/14-021-__enter__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.6: مزيد من Magic Methods: `__call__`, `__getitem__`, Context Managers

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-021-__enter__.py
"""

class Timer:
    """context manager يقيس زمن تنفيذ كتلة كود."""

    def __enter__(self):
        import time
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        import time
        self.end = time.perf_counter()
        print(f"⏱️ استغرق التنفيذ: {self.end - self.start:.4f} ثانية")

with Timer():
    total = sum(i ** 2 for i in range(1_000_000))
    print(f"النتيجة: {total}")

# ────────────────────────────────────────────────────────────────────────
# [047] code/chapters/ch14/14-022-__str__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.7: مشروع تطبيقي متكامل: نظام حجز مكتبة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-022-__str__.py
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto

# ============ الطبقة 1: نماذج البيانات (Dataclasses) ============

class ItemStatus(Enum):
    """حالة العنصر في المكتبة."""
    AVAILABLE = auto()
    BORROWED = auto()
    RESERVED = auto()

@dataclass
class Member:
    """يمثل عضواً في المكتبة."""
    member_id: int
    name: str
    borrowed_items: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"عضو #{self.member_id}: {self.name}"

# ============ الطبقة 2: الفئة المجردة للعناصر القابلة للاستعارة ============

class LibraryItem(ABC):
    """فئة مجردة أساسية لأي عنصر في المكتبة."""

    def __init__(self, item_id: str, title: str, loan_days: int):
        self.item_id = item_id
        self.title = title
        self.loan_days = loan_days  # عدد الأيام المسموحة للاستعارة
        self.status = ItemStatus.AVAILABLE
        self.borrowed_by: Member | None = None
        self.due_date: datetime | None = None

    @abstractmethod
    def get_category(self) -> str:
        """كل فئة فرعية يجب أن تحدد فئتها."""
        pass

    @abstractmethod
    def display_details(self) -> str:
        """كل فئة فرعية تعرض تفاصيلها الخاصة."""
        pass

    def borrow(self, member: Member) -> bool:
        """استعارة العنصر - سلوك مشترك لكل الأنواع."""
        if self.status != ItemStatus.AVAILABLE:
            print(f"❌ '{self.title}' غير متاح حالياً (الحالة: {self.status.name}).")
            return False

        self.status = ItemStatus.BORROWED
        self.borrowed_by = member
        self.due_date = datetime.now() + timedelta(days=self.loan_days)
        member.borrowed_items.append(self.item_id)

        print(f"✅ استعار {member.name} '{self.title}'. "
              f"الإرجاع المتوقع: {self.due_date.strftime('%Y-%m-%d')}")
        return True

    def return_item(self) -> None:
        """إرجاع العنصر."""
        if self.status != ItemStatus.BORROWED:
            print(f"⚠️ '{self.title}' ليس مُستعاراً أصلاً.")
            return

        if self.borrowed_by:
            self.borrowed_by.borrowed_items.remove(self.item_id)
            print(f"✅ تم إرجاع '{self.title}' من {self.borrowed_by.name}.")

        self.status = ItemStatus.AVAILABLE
        self.borrowed_by = None
        self.due_date = None

    def is_overdue(self) -> bool:
        """التحقق من تأخر الإرجاع - سلوك مشترك."""
        if self.status != ItemStatus.BORROWED or self.due_date is None:
            return False
        return datetime.now() > self.due_date

    def __str__(self) -> str:
        status_display = self.status.name
        return f"[{self.get_category()}] {self.title} ({self.item_id}) - {status_display}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, LibraryItem):
            return False
        return self.item_id == other.item_id

    def __hash__(self) -> int:
        return hash(self.item_id)

# ============ الطبقة 3: الفئات الفرعية (تطبيق تعدد الأشكال) ============

class Book(LibraryItem):
    def __init__(self, item_id: str, title: str, author: str, pages: int):
        super().__init__(item_id, title, loan_days=14)  # الكتب تُستعار 14 يوماً
        self.author = author
        self.pages = pages

    def get_category(self) -> str:
        return "كتاب"

    def display_details(self) -> str:
        return f"📖 {self.title} - تأليف: {self.author} ({self.pages} صفحة)"

class DVD(LibraryItem):
    def __init__(self, item_id: str, title: str, duration_minutes: int):
        super().__init__(item_id, title, loan_days=5)  # الأفلام تُستعار 5 أيام فقط
        self.duration_minutes = duration_minutes

    def get_category(self) -> str:
        return "دي في دي"

    def display_details(self) -> str:
        return f"🎬 {self.title} - المدة: {self.duration_minutes} دقيقة"

class Magazine(LibraryItem):
    def __init__(self, item_id: str, title: str, issue_number: int):
        super().__init__(item_id, title, loan_days=3)  # المجلات 3 أيام فقط
        self.issue_number = issue_number

    def get_category(self) -> str:
        return "مجلة"

    def display_details(self) -> str:
        return f"📰 {self.title} - العدد: {self.issue_number}"

# ============ الطبقة 4: استثناءات مخصصة ============

class LibraryError(Exception):
    """الفئة الأساسية لكل أخطاء نظام المكتبة."""
    pass

class ItemNotFoundError(LibraryError):
    def __init__(self, item_id: str):
        super().__init__(f"العنصر '{item_id}' غير موجود في المكتبة.")

class MemberNotFoundError(LibraryError):
    def __init__(self, member_id: int):
        super().__init__(f"العضو رقم {member_id} غير موجود.")

# ============ الطبقة 5: النظام الرئيسي ============

class Library:
    """يُدير كل عمليات المكتبة: الأعضاء، العناصر، الاستعارة، الإرجاع."""

    def __init__(self, name: str):
        self.name = name
        self.items: dict[str, LibraryItem] = {}
        self.members: dict[int, Member] = {}

    # --- إدارة العناصر ---
    def add_item(self, item: LibraryItem) -> None:
        self.items[item.item_id] = item
        print(f"📚 أُضيف '{item.title}' إلى مكتبة '{self.name}'.")

    def find_item(self, item_id: str) -> LibraryItem:
        try:
            return self.items[item_id]
        except KeyError:
            raise ItemNotFoundError(item_id)

    # --- إدارة الأعضاء ---
    def register_member(self, member: Member) -> None:
        self.members[member.member_id] = member
        print(f"👤 تم تسجيل {member.name} كعضو جديد.")

    def find_member(self, member_id: int) -> Member:
        try:
            return self.members[member_id]
        except KeyError:
            raise MemberNotFoundError(member_id)

    # --- العمليات الأساسية ---
    def borrow_item(self, item_id: str, member_id: int) -> None:
        item = self.find_item(item_id)
        member = self.find_member(member_id)
        item.borrow(member)

    def return_item(self, item_id: str) -> None:
        item = self.find_item(item_id)
        item.return_item()

    # --- تقارير (تطبيق تعدد الأشكال بوضوح) ---
    def list_all_items(self) -> None:
        print(f"\n📋 كل عناصر مكتبة '{self.name}':")
        print("-" * 60)
        for item in self.items.values():
            print(item.display_details())  # كل نوع يعرض تفاصيله الخاصة!
        print("-" * 60)

    def list_available_items(self) -> list[LibraryItem]:
        return [item for item in self.items.values()
                if item.status == ItemStatus.AVAILABLE]

    def list_overdue_items(self) -> list[LibraryItem]:
        return [item for item in self.items.values() if item.is_overdue()]

    def items_by_category(self) -> dict[str, int]:
        """عدد العناصر حسب الفئة - باستخدام polymorphism (get_category)."""
        counts: dict[str, int] = {}
        for item in self.items.values():
            category = item.get_category()
            counts[category] = counts.get(category, 0) + 1
        return counts

# ============ التشغيل التجريبي ============

if __name__ == "__main__":
    library = Library("مكتبة الأمل العامة")

    # إضافة عناصر متنوعة (تعدد الأشكال في العمل)
    library.add_item(Book("B001", "بايثون للمحترفين", "محمد أحمد", 350))
    library.add_item(Book("B002", "Clean Code", "Robert Martin", 400))
    library.add_item(DVD("D001", "رحلة إلى الفضاء", 120))
    library.add_item(Magazine("M001", "التقنية اليوم", 45))

    # تسجيل أعضاء
    library.register_member(Member(1, "أحمد"))
    library.register_member(Member(2, "سارة"))

    print()
    library.list_all_items()

    print("\n--- عمليات الاستعارة ---")
    library.borrow_item("B001", 1)
    library.borrow_item("D001", 2)

    try:
        library.borrow_item("B001", 2)  # محاولة استعارة كتاب مُستعار بالفعل
    except LibraryError as e:
        print(f"❌ {e}")

    try:
        library.borrow_item("X999", 1)  # عنصر غير موجود
    except ItemNotFoundError as e:
        print(f"❌ {e}")

    print("\n--- حالة المكتبة ---")
    library.list_all_items()

    available = library.list_available_items()
    print(f"\n✅ العناصر المتاحة حالياً: {len(available)}")
    for item in available:
        print(f"  - {item}")

    print(f"\n📊 توزيع العناصر حسب الفئة: {library.items_by_category()}")

    print("\n--- إرجاع عنصر ---")
    library.return_item("B001")

    library.list_all_items()

# ────────────────────────────────────────────────────────────────────────
# [048] code/chapters/ch14/14-023-calculate_fine.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.7: مشروع تطبيقي متكامل: نظام حجز مكتبة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-023-calculate_fine.py
"""

class LibraryItem(ABC):
    # ... (الكود السابق كما هو) ...
    FINE_PER_DAY = 2.0  # ريال لكل يوم تأخير

    def calculate_fine(self) -> float:
        """حساب الغرامة إذا كان العنصر متأخراً."""
        if not self.is_overdue() or self.due_date is None:
            return 0.0
        days_late = (datetime.now() - self.due_date).days
        return days_late * self.FINE_PER_DAY

# استخدام:
item = library.find_item("B001")
fine = item.calculate_fine()
if fine > 0:
    print(f"⚠️ غرامة تأخير: {fine} ر.س")

# ────────────────────────────────────────────────────────────────────────
# [049] code/chapters/ch14/14-024-area.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-024-area.py
"""

# تمرين 1
class Shape:
    def area(self) -> float:
        return 0

class Square(Shape):
    def __init__(self, side: float):
        self.side = side

    def area(self) -> float:
        return self.side ** 2

sq = Square(4)
print(sq.area())  # 16

# تمرين 2
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

p1 = Point(1, 2)
p2 = Point(1, 2)
print(p1)          # Point(x=1, y=2)
print(p1 == p2)     # True

# ────────────────────────────────────────────────────────────────────────
# [050] code/chapters/ch14/14-025-process.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-025-process.py
"""

# تمرين 3
from abc import ABC, abstractmethod

class PaymentMethod(ABC):
    @abstractmethod
    def process(self, amount: float) -> None:
        pass

class CreditCard(PaymentMethod):
    def process(self, amount: float) -> None:
        print(f"💳 تمت معالجة {amount} ر.س عبر البطاقة الائتمانية.")

class PayPal(PaymentMethod):
    def process(self, amount: float) -> None:
        print(f"🅿️ تمت معالجة {amount} ر.س عبر PayPal.")

methods: list[PaymentMethod] = [CreditCard(), PayPal()]
for method in methods:
    method.process(100)

# تمرين 4
from dataclasses import dataclass

@dataclass(frozen=True)
class Product:
    name: str
    price: float
    quantity: int = 1

    def total(self) -> float:
        return self.price * self.quantity

p = Product("لابتوب", 3500, 2)
print(p.total())  # 7000
# p.price = 4000  # 💥 FrozenInstanceError

# ────────────────────────────────────────────────────────────────────────
# [051] code/chapters/ch14/14-026-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-026-__init__.py
"""

from abc import ABC, abstractmethod

class Employee(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def calculate_salary(self) -> float:
        pass

    def __str__(self) -> str:
        return f"{self.name}: {self.calculate_salary():.2f} ر.س"

class FullTimeEmployee(Employee):
    def __init__(self, name: str, monthly_salary: float):
        super().__init__(name)
        self.monthly_salary = monthly_salary

    def calculate_salary(self) -> float:
        return self.monthly_salary

class FreelanceEmployee(Employee):
    def __init__(self, name: str, hourly_rate: float, hours_worked: float):
        super().__init__(name)
        self.hourly_rate = hourly_rate
        self.hours_worked = hours_worked

    def calculate_salary(self) -> float:
        return self.hourly_rate * self.hours_worked

class Payroll:
    def __init__(self):
        self.employees: list[Employee] = []

    def add_employee(self, employee: Employee) -> None:
        self.employees.append(employee)

    def total_payroll(self) -> float:
        return sum(emp.calculate_salary() for emp in self.employees)

    def print_report(self) -> None:
        print("📋 تقرير الرواتب:")
        for emp in self.employees:
            print(f"  {emp}")
        print(f"الإجمالي: {self.total_payroll():.2f} ر.س")

payroll = Payroll()
payroll.add_employee(FullTimeEmployee("أحمد", 8000))
payroll.add_employee(FreelanceEmployee("سارة", 50, 120))
payroll.print_report()

# 📋 تقرير الرواتب:
#   أحمد: 8000.00 ر.س
#   سارة: 6000.00 ر.س
# الإجمالي: 14000.00 ر.س

# ────────────────────────────────────────────────────────────────────────
# [052] code/chapters/ch14/14-027-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.10: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-027-__init__.py
"""

# ============ الوراثة ============
class Parent:
    def __init__(self, x):
        self.x = x
    def greet(self):
        return f"Parent: {self.x}"

class Child(Parent):
    def __init__(self, x, y):
        super().__init__(x)    # استدعاء __init__ الأساسي
        self.y = y
    def greet(self):
        return f"{super().greet()}, Child: {self.y}"  # تمديد السلوك

# ============ تعدد الأشكال ============
for obj in [ClassA(), ClassB(), ClassC()]:
    obj.common_method()  # كل فئة تُنفّذ بطريقتها

# ============ الفئات المجردة ============
from abc import ABC, abstractmethod

class Base(ABC):
    @abstractmethod
    def required_method(self): ...

# Base()  # 💥 TypeError - لا يمكن إنشاء كائن مباشرة

# ============ Dataclasses ============
from dataclasses import dataclass, field

@dataclass
class Point:
    x: float
    y: float
    tags: list = field(default_factory=list)  # ⚠️ ليس []

@dataclass(frozen=True)   # immutable
@dataclass(order=True)    # يدعم < <= > >=

# ============ Magic Methods إضافية ============
def __call__(self, *args): ...      # obj(args)
def __getitem__(self, i): ...       # obj[i]
def __enter__(self): ...            # with obj as x:
def __exit__(self, *args): ...

# ============ isinstance ============
isinstance(obj, SomeClass)          # ✅ يأخذ الوراثة بعين الاعتبار
type(obj) == SomeClass              # ❌ تطابق دقيق فقط

# ────────────────────────────────────────────────────────────────────────
# [053] dist/markdown/chapter-14.md
# ────────────────────────────────────────────────────────────────────────
class Manager(Employee):
    def __init__(self, name, salary, team_size):
        # ⚠️ نسيان super().__init__(name, salary)!
        self.team_size = team_size

manager = Manager("نورة", 15000, 8)
print(manager.name)
# 💥 AttributeError: 'Manager' object has no attribute 'name'

# ────────────────────────────────────────────────────────────────────────
# [054] dist/markdown/chapter-14.md
# ────────────────────────────────────────────────────────────────────────
# ❌ لا يأخذ الوراثة في الحسبان
if type(dog) == Animal:  # False حتى لو dog هو كلب (نوع فرعي من Animal)!
    pass

# ✅ يأخذ الوراثة في الحسبان
if isinstance(dog, Animal):  # True
    pass

# ────────────────────────────────────────────────────────────────────────
# [055] dist/markdown/chapter-14.md
# ────────────────────────────────────────────────────────────────────────
from abc import ABC, abstractmethod

class Shape(ABC):  # ترث من ABC (Abstract Base Class)
    @abstractmethod
    def area(self) -> float:
        """يجب أن تُخصَّص في كل فئة فرعية."""
        pass

    @abstractmethod
    def perimeter(self) -> float:
        """يجب أن تُخصَّص في كل فئة فرعية."""
        pass

    def describe(self) -> str:
        """method عادية - موروثة كما هي، لا حاجة لتخصيصها."""
        return f"{type(self).__name__}: مساحة={self.area():.2f}, محيط={self.perimeter():.2f}"

class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius

    def area(self) -> float:
        return 3.14159 * self.radius ** 2

    def perimeter(self) -> float:
        return 2 * 3.14159 * self.radius

# shape = Shape()
# 💥 TypeError: Can't instantiate abstract class Shape with abstract methods area, perimeter

circle = Circle(5)
print(circle.describe())  # Circle: مساحة=78.54, محيط=31.42

class Square(Shape):
    def __init__(self, side: float):
        self.side = side
    # ⚠️ لو نسينا تعريف area() أو perimeter() هنا:

# square = Square(5)
# 💥 TypeError: Can't instantiate abstract class Square with abstract method area, perimeter

# ────────────────────────────────────────────────────────────────────────
# [056] dist/markdown/chapter-14.md
# ────────────────────────────────────────────────────────────────────────
@dataclass
class Student:
    grades: list = []  # 💥 ValueError: mutable default <class 'list'> is not allowed

# ────────────────────────────────────────────────────────────────────────
# [057] dist/markdown/chapter-14.md
# ────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass

@dataclass(frozen=True)  # يجعل الكائن immutable - كالـ tuple!
class ImmutablePoint:
    x: float
    y: float

p = ImmutablePoint(3, 4)
# p.x = 10
# 💥 FrozenInstanceError: cannot assign to field 'x'

@dataclass(order=True)  # يُضيف __lt__, __le__, __gt__, __ge__ تلقائياً
class Version:
    major: int
    minor: int

v1 = Version(1, 5)
v2 = Version(2, 0)
print(v1 < v2)  # True -- تقارن الحقول بالترتيب المُعرَّف

# ────────────────────────────────────────────────────────────────────────
# [058] dist/markdown/chapter-14.md
# ────────────────────────────────────────────────────────────────────────
class LibraryItem(ABC):
    # ... (الكود السابق كما هو) ...
    FINE_PER_DAY = 2.0  # ريال لكل يوم تأخير

    def calculate_fine(self) -> float:
        """حساب الغرامة إذا كان العنصر متأخراً."""
        if not self.is_overdue() or self.due_date is None:
            return 0.0
        days_late = (datetime.now() - self.due_date).days
        return days_late * self.FINE_PER_DAY

# استخدام:
item = library.find_item("B001")
fine = item.calculate_fine()
if fine > 0:
    print(f"⚠️ غرامة تأخير: {fine} ر.س")

# ────────────────────────────────────────────────────────────────────────
# [059] dist/markdown/chapter-14.md
# ────────────────────────────────────────────────────────────────────────
# تمرين 3
from abc import ABC, abstractmethod

class PaymentMethod(ABC):
    @abstractmethod
    def process(self, amount: float) -> None:
        pass

class CreditCard(PaymentMethod):
    def process(self, amount: float) -> None:
        print(f"💳 تمت معالجة {amount} ر.س عبر البطاقة الائتمانية.")

class PayPal(PaymentMethod):
    def process(self, amount: float) -> None:
        print(f"🅿️ تمت معالجة {amount} ر.س عبر PayPal.")

methods: list[PaymentMethod] = [CreditCard(), PayPal()]
for method in methods:
    method.process(100)

# تمرين 4
from dataclasses import dataclass

@dataclass(frozen=True)
class Product:
    name: str
    price: float
    quantity: int = 1

    def total(self) -> float:
        return self.price * self.quantity

p = Product("لابتوب", 3500, 2)
print(p.total())  # 7000
# p.price = 4000  # 💥 FrozenInstanceError

# ────────────────────────────────────────────────────────────────────────
# [060] dist/markdown/chapter-14.md
# ────────────────────────────────────────────────────────────────────────
# ============ الوراثة ============
class Parent:
    def __init__(self, x):
        self.x = x
    def greet(self):
        return f"Parent: {self.x}"

class Child(Parent):
    def __init__(self, x, y):
        super().__init__(x)    # استدعاء __init__ الأساسي
        self.y = y
    def greet(self):
        return f"{super().greet()}, Child: {self.y}"  # تمديد السلوك

# ============ تعدد الأشكال ============
for obj in [ClassA(), ClassB(), ClassC()]:
    obj.common_method()  # كل فئة تُنفّذ بطريقتها

# ============ الفئات المجردة ============
from abc import ABC, abstractmethod

class Base(ABC):
    @abstractmethod
    def required_method(self): ...

# Base()  # 💥 TypeError - لا يمكن إنشاء كائن مباشرة

# ============ Dataclasses ============
from dataclasses import dataclass, field

@dataclass
class Point:
    x: float
    y: float
    tags: list = field(default_factory=list)  # ⚠️ ليس []

@dataclass(frozen=True)   # immutable
@dataclass(order=True)    # يدعم < <= > >=

# ============ Magic Methods إضافية ============
def __call__(self, *args): ...      # obj(args)
def __getitem__(self, i): ...       # obj[i]
def __enter__(self): ...            # with obj as x:
def __exit__(self, *args): ...

# ============ isinstance ============
isinstance(obj, SomeClass)          # ✅ يأخذ الوراثة بعين الاعتبار
type(obj) == SomeClass              # ❌ تطابق دقيق فقط

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
class Product:
    def __init__(self, name: str, price: float):
        self.name = name
        self.price = price

    def display_info(self) -> str:
        return f"{self.name}: {self.price} ر.س"

class Book(Product):  # Book ترث من Product
    def __init__(self, name: str, price: float, author: str):
        super().__init__(name, price)  # استدعاء __init__ من الفئة الأساسية
        self.author = author

    def display_info(self) -> str:
        return f"{super().display_info()} - تأليف: {self.author}"

book = Book("بايثون للمحترفين", 120, "محمد أحمد")
print(book.display_info())
# بايثون للمحترفين: 120 ر.س - تأليف: محمد أحمد

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class Animal:  # Parent / Base / Super Class
    def __init__(self, name: str):
        self.name = name

    def make_sound(self) -> str:
        return "صوت ما..."

    def __str__(self) -> str:
        return f"{self.name} يصدر صوت: {self.make_sound()}"

class Dog(Animal):  # Child / Derived / Sub Class
    def make_sound(self) -> str:  # Override - استبدال السلوك
        return "واق واق!"

class Cat(Animal):
    def make_sound(self) -> str:
        return "مياو!"

dog = Dog("ريكس")
cat = Cat("لونا")

print(dog)  # ريكس يصدر صوت: واق واق!
print(cat)  # لونا يصدر صوت: مياو!

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class Employee:
    def __init__(self, name: str, salary: float):
        self.name = name
        self.salary = salary

    def display(self) -> str:
        return f"{self.name}: {self.salary} ر.س"

class Manager(Employee):
    def __init__(self, name: str, salary: float, team_size: int):
        super().__init__(name, salary)  # استدعاء __init__ الأساسي أولاً
        self.team_size = team_size

    def display(self) -> str:
        base_info = super().display()  # استدعاء display الأساسي، ثم إضافة عليه
        return f"{base_info} (يدير فريق من {self.team_size} أشخاص)"

manager = Manager("نورة", 15000, 8)
print(manager.display())
# نورة: 15000 ر.س (يدير فريق من 8 أشخاص)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    class Manager(Employee):
        def __init__(self, name, salary, team_size):
            # ⚠️ نسيان super().__init__(name, salary)!
            self.team_size = team_size

    manager = Manager("نورة", 15000, 8)
    print(manager.name)
    # 💥 AttributeError: 'Manager' object has no attribute 'name'
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
print(isinstance(dog, Dog))     # True
print(isinstance(dog, Animal))  # True - Dog هي أيضاً Animal!
print(isinstance(dog, Cat))     # False

print(issubclass(Dog, Animal))  # True
print(issubclass(Animal, Dog))  # False - العلاقة أحادية الاتجاه

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
class Vehicle:
    def __init__(self, brand: str):
        self.brand = brand

    def describe(self) -> str:
        return f"مركبة من {self.brand}"

class Car(Vehicle):
    def __init__(self, brand: str, doors: int):
        super().__init__(brand)
        self.doors = doors

    def describe(self) -> str:
        return f"{super().describe()}, عدد الأبواب: {self.doors}"

class SportsCar(Car):
    def __init__(self, brand: str, doors: int, top_speed: int):
        super().__init__(brand, doors)
        self.top_speed = top_speed

    def describe(self) -> str:
        return f"{super().describe()}, السرعة القصوى: {self.top_speed} كم/س"

ferrari = SportsCar("Ferrari", 2, 340)
print(ferrari.describe())
# مركبة من Ferrari, عدد الأبواب: 2, السرعة القصوى: 340 كم/س

print(isinstance(ferrari, Vehicle))   # True
print(isinstance(ferrari, Car))       # True
print(isinstance(ferrari, SportsCar)) # True

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
class Swimmer:
    def swim(self) -> str:
        return "يسبح 🏊"

class Flyer:
    def fly(self) -> str:
        return "يطير 🦅"

class Duck(Swimmer, Flyer):  # وراثة من فئتين معاً
    def __init__(self, name: str):
        self.name = name

duck = Duck("دونالد")
print(duck.swim())  # يسبح 🏊
print(duck.fly())   # يطير 🦅

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
print(Duck.__mro__)
# (<class 'Duck'>, <class 'Swimmer'>, <class 'Flyer'>, <class 'object'>)

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
class Shape:
    def area(self) -> float:
        raise NotImplementedError("يجب تخصيص هذه الدالة في الفئة الفرعية")

class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius

    def area(self) -> float:
        return 3.14159 * self.radius ** 2

class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

class Triangle(Shape):
    def __init__(self, base: float, height: float):
        self.base = base
        self.height = height

    def area(self) -> float:
        return 0.5 * self.base * self.height

# هنا يكمن سحر تعدد الأشكال:
shapes: list[Shape] = [Circle(5), Rectangle(4, 6), Triangle(3, 8)]

for shape in shapes:
    print(f"{type(shape).__name__}: المساحة = {shape.area():.2f}")

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
class Duck:
    def make_sound(self) -> str:
        return "Quack!"

class Robot:
    def make_sound(self) -> str:
        return "Beep Boop!"

def animate(entity) -> None:
    """لا تهتم بنوع entity، فقط تحتاج أن تملك make_sound()"""
    print(entity.make_sound())

animate(Duck())    # Quack!
animate(Robot())   # Beep Boop! -- Robot ليست حيواناً إطلاقاً لكنها تعمل!

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    from abc import ABC, abstractmethod

    class Shape(ABC):  # ترث من ABC (Abstract Base Class)
        @abstractmethod
        def area(self) -> float:
            """يجب أن تُخصَّص في كل فئة فرعية."""
            pass

        @abstractmethod
        def perimeter(self) -> float:
            """يجب أن تُخصَّص في كل فئة فرعية."""
            pass

        def describe(self) -> str:
            """method عادية - موروثة كما هي، لا حاجة لتخصيصها."""
            return f"{type(self).__name__}: مساحة={self.area():.2f}, محيط={self.perimeter():.2f}"

    class Circle(Shape):
        def __init__(self, radius: float):
            self.radius = radius

        def area(self) -> float:
            return 3.14159 * self.radius ** 2

        def perimeter(self) -> float:
            return 2 * 3.14159 * self.radius

    # shape = Shape()
    # 💥 TypeError: Can't instantiate abstract class Shape with abstract methods area, perimeter

    circle = Circle(5)
    print(circle.describe())  # Circle: مساحة=78.54, محيط=31.42

    class Square(Shape):
        def __init__(self, side: float):
            self.side = side
        # ⚠️ لو نسينا تعريف area() أو perimeter() هنا:

    # square = Square(5)
    # 💥 TypeError: Can't instantiate abstract class Square with abstract method area, perimeter
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
# الطريقة التقليدية - كثير من الكود المتكرر لفئة بسيطة تخزن بيانات فقط
class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"Point(x={self.x}, y={self.y})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

p1 = Point(3, 4)
p2 = Point(3, 4)

print(p1)          # Point(x=3, y=4)  -- __repr__ تلقائياً!
print(p1 == p2)     # True  -- __eq__ تلقائياً!

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass, field

@dataclass
class Student:
    name: str
    age: int = 18  # قيمة افتراضية بسيطة
    grades: list[float] = field(default_factory=list)  # للقيم القابلة للتعديل!

s1 = Student("أحمد")
s1.grades.append(90)
print(s1)  # Student(name='أحمد', age=18, grades=[90])

s2 = Student("سارة")
print(s2)  # Student(name='سارة', age=18, grades=[])  -- مستقلة تماماً!

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    @dataclass
    class Student:
        grades: list = []  # 💥 ValueError: mutable default <class 'list'> is not allowed
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass

@dataclass
class Rectangle:
    width: float
    height: float

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)

rect = Rectangle(5, 3)
print(rect)             # Rectangle(width=5, height=3)
print(rect.area())      # 15

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    from dataclasses import dataclass

    @dataclass(frozen=True)  # يجعل الكائن immutable - كالـ tuple!
    class ImmutablePoint:
        x: float
        y: float

    p = ImmutablePoint(3, 4)
    # p.x = 10
    # 💥 FrozenInstanceError: cannot assign to field 'x'

    @dataclass(order=True)  # يُضيف __lt__, __le__, __gt__, __ge__ تلقائياً
    class Version:
        major: int
        minor: int

    v1 = Version(1, 5)
    v2 = Version(2, 0)
    print(v1 < v2)  # True -- تقارن الحقول بالترتيب المُعرَّف
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
class Multiplier:
    def __init__(self, factor: float):
        self.factor = factor

    def __call__(self, value: float) -> float:
        return value * self.factor

double = Multiplier(2)
print(double(5))   # 10 -- الكائن نفسه يُستدعى كدالة!
print(double(10))  # 20

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
class Playlist:
    def __init__(self, songs: list[str]):
        self.songs = songs

    def __getitem__(self, index: int) -> str:
        return self.songs[index]

    def __len__(self) -> int:
        return len(self.songs)

playlist = Playlist(["أغنية 1", "أغنية 2", "أغنية 3"])
print(playlist[0])  # أغنية 1  -- الفهرسة تعمل الآن!

for song in playlist:  # __getitem__ تجعل الكائن قابلاً للتكرار تلقائياً أيضاً!
    print(song)

# ────────────────────────────────────────────────────────────────────────
# [022] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
class Timer:
    """context manager يقيس زمن تنفيذ كتلة كود."""

    def __enter__(self):
        import time
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        import time
        self.end = time.perf_counter()
        print(f"⏱️ استغرق التنفيذ: {self.end - self.start:.4f} ثانية")

with Timer():
    total = sum(i ** 2 for i in range(1_000_000))
    print(f"النتيجة: {total}")

# ────────────────────────────────────────────────────────────────────────
# [023] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto

# ============ الطبقة 1: نماذج البيانات (Dataclasses) ============

class ItemStatus(Enum):
    """حالة العنصر في المكتبة."""
    AVAILABLE = auto()
    BORROWED = auto()
    RESERVED = auto()

@dataclass
class Member:
    """يمثل عضواً في المكتبة."""
    member_id: int
    name: str
    borrowed_items: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"عضو #{self.member_id}: {self.name}"

# ============ الطبقة 2: الفئة المجردة للعناصر القابلة للاستعارة ============

class LibraryItem(ABC):
    """فئة مجردة أساسية لأي عنصر في المكتبة."""

    def __init__(self, item_id: str, title: str, loan_days: int):
        self.item_id = item_id
        self.title = title
        self.loan_days = loan_days  # عدد الأيام المسموحة للاستعارة
        self.status = ItemStatus.AVAILABLE
        self.borrowed_by: Member | None = None
        self.due_date: datetime | None = None

    @abstractmethod
    def get_category(self) -> str:
        """كل فئة فرعية يجب أن تحدد فئتها."""
        pass

    @abstractmethod
    def display_details(self) -> str:
        """كل فئة فرعية تعرض تفاصيلها الخاصة."""
        pass

    def borrow(self, member: Member) -> bool:
        """استعارة العنصر - سلوك مشترك لكل الأنواع."""
        if self.status != ItemStatus.AVAILABLE:
            print(f"❌ '{self.title}' غير متاح حالياً (الحالة: {self.status.name}).")
            return False

        self.status = ItemStatus.BORROWED
        self.borrowed_by = member
        self.due_date = datetime.now() + timedelta(days=self.loan_days)
        member.borrowed_items.append(self.item_id)

        print(f"✅ استعار {member.name} '{self.title}'. "
              f"الإرجاع المتوقع: {self.due_date.strftime('%Y-%m-%d')}")
        return True

    def return_item(self) -> None:
        """إرجاع العنصر."""
        if self.status != ItemStatus.BORROWED:
            print(f"⚠️ '{self.title}' ليس مُستعاراً أصلاً.")
            return

        if self.borrowed_by:
            self.borrowed_by.borrowed_items.remove(self.item_id)
            print(f"✅ تم إرجاع '{self.title}' من {self.borrowed_by.name}.")

        self.status = ItemStatus.AVAILABLE
        self.borrowed_by = None
        self.due_date = None

    def is_overdue(self) -> bool:
        """التحقق من تأخر الإرجاع - سلوك مشترك."""
        if self.status != ItemStatus.BORROWED or self.due_date is None:
            return False
        return datetime.now() > self.due_date

    def __str__(self) -> str:
        status_display = self.status.name
        return f"[{self.get_category()}] {self.title} ({self.item_id}) - {status_display}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, LibraryItem):
            return False
        return self.item_id == other.item_id

    def __hash__(self) -> int:
        return hash(self.item_id)

# ============ الطبقة 3: الفئات الفرعية (تطبيق تعدد الأشكال) ============

class Book(LibraryItem):
    def __init__(self, item_id: str, title: str, author: str, pages: int):
        super().__init__(item_id, title, loan_days=14)  # الكتب تُستعار 14 يوماً
        self.author = author
        self.pages = pages

    def get_category(self) -> str:
        return "كتاب"

    def display_details(self) -> str:
        return f"📖 {self.title} - تأليف: {self.author} ({self.pages} صفحة)"

class DVD(LibraryItem):
    def __init__(self, item_id: str, title: str, duration_minutes: int):
        super().__init__(item_id, title, loan_days=5)  # الأفلام تُستعار 5 أيام فقط
        self.duration_minutes = duration_minutes

    def get_category(self) -> str:
        return "دي في دي"

    def display_details(self) -> str:
        return f"🎬 {self.title} - المدة: {self.duration_minutes} دقيقة"

class Magazine(LibraryItem):
    def __init__(self, item_id: str, title: str, issue_number: int):
        super().__init__(item_id, title, loan_days=3)  # المجلات 3 أيام فقط
        self.issue_number = issue_number

    def get_category(self) -> str:
        return "مجلة"

    def display_details(self) -> str:
        return f"📰 {self.title} - العدد: {self.issue_number}"

# ============ الطبقة 4: استثناءات مخصصة ============

class LibraryError(Exception):
    """الفئة الأساسية لكل أخطاء نظام المكتبة."""
    pass

class ItemNotFoundError(LibraryError):
    def __init__(self, item_id: str):
        super().__init__(f"العنصر '{item_id}' غير موجود في المكتبة.")

class MemberNotFoundError(LibraryError):
    def __init__(self, member_id: int):
        super().__init__(f"العضو رقم {member_id} غير موجود.")

# ============ الطبقة 5: النظام الرئيسي ============

class Library:
    """يُدير كل عمليات المكتبة: الأعضاء، العناصر، الاستعارة، الإرجاع."""

    def __init__(self, name: str):
        self.name = name
        self.items: dict[str, LibraryItem] = {}
        self.members: dict[int, Member] = {}

    # --- إدارة العناصر ---
    def add_item(self, item: LibraryItem) -> None:
        self.items[item.item_id] = item
        print(f"📚 أُضيف '{item.title}' إلى مكتبة '{self.name}'.")

    def find_item(self, item_id: str) -> LibraryItem:
        try:
            return self.items[item_id]
        except KeyError:
            raise ItemNotFoundError(item_id)

    # --- إدارة الأعضاء ---
    def register_member(self, member: Member) -> None:
        self.members[member.member_id] = member
        print(f"👤 تم تسجيل {member.name} كعضو جديد.")

    def find_member(self, member_id: int) -> Member:
        try:
            return self.members[member_id]
        except KeyError:
            raise MemberNotFoundError(member_id)

    # --- العمليات الأساسية ---
    def borrow_item(self, item_id: str, member_id: int) -> None:
        item = self.find_item(item_id)
        member = self.find_member(member_id)
        item.borrow(member)

    def return_item(self, item_id: str) -> None:
        item = self.find_item(item_id)
        item.return_item()

    # --- تقارير (تطبيق تعدد الأشكال بوضوح) ---
    def list_all_items(self) -> None:
        print(f"\n📋 كل عناصر مكتبة '{self.name}':")
        print("-" * 60)
        for item in self.items.values():
            print(item.display_details())  # كل نوع يعرض تفاصيله الخاصة!
        print("-" * 60)

    def list_available_items(self) -> list[LibraryItem]:
        return [item for item in self.items.values()
                if item.status == ItemStatus.AVAILABLE]

    def list_overdue_items(self) -> list[LibraryItem]:
        return [item for item in self.items.values() if item.is_overdue()]

    def items_by_category(self) -> dict[str, int]:
        """عدد العناصر حسب الفئة - باستخدام polymorphism (get_category)."""
        counts: dict[str, int] = {}
        for item in self.items.values():
            category = item.get_category()
            counts[category] = counts.get(category, 0) + 1
        return counts

# ============ التشغيل التجريبي ============

if __name__ == "__main__":
    library = Library("مكتبة الأمل العامة")

    # إضافة عناصر متنوعة (تعدد الأشكال في العمل)
    library.add_item(Book("B001", "بايثون للمحترفين", "محمد أحمد", 350))
    library.add_item(Book("B002", "Clean Code", "Robert Martin", 400))
    library.add_item(DVD("D001", "رحلة إلى الفضاء", 120))
    library.add_item(Magazine("M001", "التقنية اليوم", 45))

    # تسجيل أعضاء
    library.register_member(Member(1, "أحمد"))
    library.register_member(Member(2, "سارة"))

    print()
    library.list_all_items()

    print("\n--- عمليات الاستعارة ---")
    library.borrow_item("B001", 1)
    library.borrow_item("D001", 2)

    try:
        library.borrow_item("B001", 2)  # محاولة استعارة كتاب مُستعار بالفعل
    except LibraryError as e:
        print(f"❌ {e}")

    try:
        library.borrow_item("X999", 1)  # عنصر غير موجود
    except ItemNotFoundError as e:
        print(f"❌ {e}")

    print("\n--- حالة المكتبة ---")
    library.list_all_items()

    available = library.list_available_items()
    print(f"\n✅ العناصر المتاحة حالياً: {len(available)}")
    for item in available:
        print(f"  - {item}")

    print(f"\n📊 توزيع العناصر حسب الفئة: {library.items_by_category()}")

    print("\n--- إرجاع عنصر ---")
    library.return_item("B001")

    library.list_all_items()

# ────────────────────────────────────────────────────────────────────────
# [024] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
# تمرين 1
class Shape:
    def area(self) -> float:
        return 0

class Square(Shape):
    def __init__(self, side: float):
        self.side = side

    def area(self) -> float:
        return self.side ** 2

sq = Square(4)
print(sq.area())  # 16

# تمرين 2
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

p1 = Point(1, 2)
p2 = Point(1, 2)
print(p1)          # Point(x=1, y=2)
print(p1 == p2)     # True

# ────────────────────────────────────────────────────────────────────────
# [025] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    # تمرين 3
    from abc import ABC, abstractmethod

    class PaymentMethod(ABC):
        @abstractmethod
        def process(self, amount: float) -> None:
            pass

    class CreditCard(PaymentMethod):
        def process(self, amount: float) -> None:
            print(f"💳 تمت معالجة {amount} ر.س عبر البطاقة الائتمانية.")

    class PayPal(PaymentMethod):
        def process(self, amount: float) -> None:
            print(f"🅿️ تمت معالجة {amount} ر.س عبر PayPal.")

    methods: list[PaymentMethod] = [CreditCard(), PayPal()]
    for method in methods:
        method.process(100)

    # تمرين 4
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Product:
        name: str
        price: float
        quantity: int = 1

        def total(self) -> float:
            return self.price * self.quantity

    p = Product("لابتوب", 3500, 2)
    print(p.total())  # 7000
    # p.price = 4000  # 💥 FrozenInstanceError
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [026] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
from abc import ABC, abstractmethod

class Employee(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def calculate_salary(self) -> float:
        pass

    def __str__(self) -> str:
        return f"{self.name}: {self.calculate_salary():.2f} ر.س"

class FullTimeEmployee(Employee):
    def __init__(self, name: str, monthly_salary: float):
        super().__init__(name)
        self.monthly_salary = monthly_salary

    def calculate_salary(self) -> float:
        return self.monthly_salary

class FreelanceEmployee(Employee):
    def __init__(self, name: str, hourly_rate: float, hours_worked: float):
        super().__init__(name)
        self.hourly_rate = hourly_rate
        self.hours_worked = hours_worked

    def calculate_salary(self) -> float:
        return self.hourly_rate * self.hours_worked

class Payroll:
    def __init__(self):
        self.employees: list[Employee] = []

    def add_employee(self, employee: Employee) -> None:
        self.employees.append(employee)

    def total_payroll(self) -> float:
        return sum(emp.calculate_salary() for emp in self.employees)

    def print_report(self) -> None:
        print("📋 تقرير الرواتب:")
        for emp in self.employees:
            print(f"  {emp}")
        print(f"الإجمالي: {self.total_payroll():.2f} ر.س")

payroll = Payroll()
payroll.add_employee(FullTimeEmployee("أحمد", 8000))
payroll.add_employee(FreelanceEmployee("سارة", 50, 120))
payroll.print_report()

# 📋 تقرير الرواتب:
#   أحمد: 8000.00 ر.س
#   سارة: 6000.00 ر.س
# الإجمالي: 14000.00 ر.س

# ────────────────────────────────────────────────────────────────────────
# [027] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-26
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.1: لماذا الوراثة أحد أهم اختراعات البرمجة الكائنية؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-001-__init__.py
"""

class Product:
    def __init__(self, name: str, price: float):
        self.name = name
        self.price = price

    def display_info(self) -> str:
        return f"{self.name}: {self.price} ر.س"

class Book(Product):  # Book ترث من Product
    def __init__(self, name: str, price: float, author: str):
        super().__init__(name, price)  # استدعاء __init__ من الفئة الأساسية
        self.author = author

    def display_info(self) -> str:
        return f"{super().display_info()} - تأليف: {self.author}"

book = Book("بايثون للمحترفين", 120, "محمد أحمد")
print(book.display_info())
# بايثون للمحترفين: 120 ر.س - تأليف: محمد أحمد

# ────────────────────────────────────────────────────────────────────────
# [028] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-27
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.2: الوراثة (Inheritance) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-002-__init__.py
"""

class Animal:  # Parent / Base / Super Class
    def __init__(self, name: str):
        self.name = name

    def make_sound(self) -> str:
        return "صوت ما..."

    def __str__(self) -> str:
        return f"{self.name} يصدر صوت: {self.make_sound()}"

class Dog(Animal):  # Child / Derived / Sub Class
    def make_sound(self) -> str:  # Override - استبدال السلوك
        return "واق واق!"

class Cat(Animal):
    def make_sound(self) -> str:
        return "مياو!"

dog = Dog("ريكس")
cat = Cat("لونا")

print(dog)  # ريكس يصدر صوت: واق واق!
print(cat)  # لونا يصدر صوت: مياو!

# ────────────────────────────────────────────────────────────────────────
# [029] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-28
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.2: الوراثة (Inheritance) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-003-__init__.py
"""

class Employee:
    def __init__(self, name: str, salary: float):
        self.name = name
        self.salary = salary

    def display(self) -> str:
        return f"{self.name}: {self.salary} ر.س"

class Manager(Employee):
    def __init__(self, name: str, salary: float, team_size: int):
        super().__init__(name, salary)  # استدعاء __init__ الأساسي أولاً
        self.team_size = team_size

    def display(self) -> str:
        base_info = super().display()  # استدعاء display الأساسي، ثم إضافة عليه
        return f"{base_info} (يدير فريق من {self.team_size} أشخاص)"

manager = Manager("نورة", 15000, 8)
print(manager.display())
# نورة: 15000 ر.س (يدير فريق من 8 أشخاص)

# ────────────────────────────────────────────────────────────────────────
# [030] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-29
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.2: الوراثة (Inheritance) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-004-__init__.py
"""

class Manager(Employee):
    def __init__(self, name, salary, team_size):
        # ⚠️ نسيان super().__init__(name, salary)!
        self.team_size = team_size

manager = Manager("نورة", 15000, 8)
print(manager.name)
# 💥 AttributeError: 'Manager' object has no attribute 'name'

# ────────────────────────────────────────────────────────────────────────
# [031] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-30
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.2: الوراثة (Inheritance) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-005-ex14_2.py
"""

print(isinstance(dog, Dog))     # True
print(isinstance(dog, Animal))  # True - Dog هي أيضاً Animal!
print(isinstance(dog, Cat))     # False

print(issubclass(Dog, Animal))  # True
print(issubclass(Animal, Dog))  # False - العلاقة أحادية الاتجاه

# ────────────────────────────────────────────────────────────────────────
# [032] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-31
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.2: الوراثة (Inheritance) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-006-ex14_2.py
"""

# ❌ لا يأخذ الوراثة في الحسبان
if type(dog) == Animal:  # False حتى لو dog هو كلب (نوع فرعي من Animal)!
    pass

# ✅ يأخذ الوراثة في الحسبان
if isinstance(dog, Animal):  # True
    pass

# ────────────────────────────────────────────────────────────────────────
# [033] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-32
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.2: الوراثة (Inheritance) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-007-__init__.py
"""

class Vehicle:
    def __init__(self, brand: str):
        self.brand = brand

    def describe(self) -> str:
        return f"مركبة من {self.brand}"

class Car(Vehicle):
    def __init__(self, brand: str, doors: int):
        super().__init__(brand)
        self.doors = doors

    def describe(self) -> str:
        return f"{super().describe()}, عدد الأبواب: {self.doors}"

class SportsCar(Car):
    def __init__(self, brand: str, doors: int, top_speed: int):
        super().__init__(brand, doors)
        self.top_speed = top_speed

    def describe(self) -> str:
        return f"{super().describe()}, السرعة القصوى: {self.top_speed} كم/س"

ferrari = SportsCar("Ferrari", 2, 340)
print(ferrari.describe())
# مركبة من Ferrari, عدد الأبواب: 2, السرعة القصوى: 340 كم/س

print(isinstance(ferrari, Vehicle))   # True
print(isinstance(ferrari, Car))       # True
print(isinstance(ferrari, SportsCar)) # True

# ────────────────────────────────────────────────────────────────────────
# [034] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-33
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.2: الوراثة (Inheritance) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-008-swim.py
"""

class Swimmer:
    def swim(self) -> str:
        return "يسبح 🏊"

class Flyer:
    def fly(self) -> str:
        return "يطير 🦅"

class Duck(Swimmer, Flyer):  # وراثة من فئتين معاً
    def __init__(self, name: str):
        self.name = name

duck = Duck("دونالد")
print(duck.swim())  # يسبح 🏊
print(duck.fly())   # يطير 🦅

# ────────────────────────────────────────────────────────────────────────
# [035] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-34
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.2: الوراثة (Inheritance) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-009-ex14_2.py
"""

print(Duck.__mro__)
# (<class 'Duck'>, <class 'Swimmer'>, <class 'Flyer'>, <class 'object'>)

# ────────────────────────────────────────────────────────────────────────
# [036] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-35
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.3: تعدد الأشكال (Polymorphism)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-010-area.py
"""

class Shape:
    def area(self) -> float:
        raise NotImplementedError("يجب تخصيص هذه الدالة في الفئة الفرعية")

class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius

    def area(self) -> float:
        return 3.14159 * self.radius ** 2

class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

class Triangle(Shape):
    def __init__(self, base: float, height: float):
        self.base = base
        self.height = height

    def area(self) -> float:
        return 0.5 * self.base * self.height

# هنا يكمن سحر تعدد الأشكال:
shapes: list[Shape] = [Circle(5), Rectangle(4, 6), Triangle(3, 8)]

for shape in shapes:
    print(f"{type(shape).__name__}: المساحة = {shape.area():.2f}")

# ────────────────────────────────────────────────────────────────────────
# [037] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-36
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.3: تعدد الأشكال (Polymorphism)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-011-make_sound.py
"""

class Duck:
    def make_sound(self) -> str:
        return "Quack!"

class Robot:
    def make_sound(self) -> str:
        return "Beep Boop!"

def animate(entity) -> None:
    """لا تهتم بنوع entity، فقط تحتاج أن تملك make_sound()"""
    print(entity.make_sound())

animate(Duck())    # Quack!
animate(Robot())   # Beep Boop! -- Robot ليست حيواناً إطلاقاً لكنها تعمل!

# ────────────────────────────────────────────────────────────────────────
# [038] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-37
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.4: الفئات المجردة (Abstract Base Classes)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-012-area.py
"""

from abc import ABC, abstractmethod

class Shape(ABC):  # ترث من ABC (Abstract Base Class)
    @abstractmethod
    def area(self) -> float:
        """يجب أن تُخصَّص في كل فئة فرعية."""
        pass

    @abstractmethod
    def perimeter(self) -> float:
        """يجب أن تُخصَّص في كل فئة فرعية."""
        pass

    def describe(self) -> str:
        """method عادية - موروثة كما هي، لا حاجة لتخصيصها."""
        return f"{type(self).__name__}: مساحة={self.area():.2f}, محيط={self.perimeter():.2f}"

class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius

    def area(self) -> float:
        return 3.14159 * self.radius ** 2

    def perimeter(self) -> float:
        return 2 * 3.14159 * self.radius

# shape = Shape()
# 💥 TypeError: Can't instantiate abstract class Shape with abstract methods area, perimeter

circle = Circle(5)
print(circle.describe())  # Circle: مساحة=78.54, محيط=31.42

class Square(Shape):
    def __init__(self, side: float):
        self.side = side
    # ⚠️ لو نسينا تعريف area() أو perimeter() هنا:

# square = Square(5)
# 💥 TypeError: Can't instantiate abstract class Square with abstract method area, perimeter

# ────────────────────────────────────────────────────────────────────────
# [039] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-38
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.5: Dataclasses: اختصار كتابة الفئات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-013-__init__.py
"""

# الطريقة التقليدية - كثير من الكود المتكرر لفئة بسيطة تخزن بيانات فقط
class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"Point(x={self.x}, y={self.y})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y

# ────────────────────────────────────────────────────────────────────────
# [040] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-39
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.5: Dataclasses: اختصار كتابة الفئات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-014-point.py
"""

from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

p1 = Point(3, 4)
p2 = Point(3, 4)

print(p1)          # Point(x=3, y=4)  -- __repr__ تلقائياً!
print(p1 == p2)     # True  -- __eq__ تلقائياً!

# ────────────────────────────────────────────────────────────────────────
# [041] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-40
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.5: Dataclasses: اختصار كتابة الفئات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-015-student.py
"""

from dataclasses import dataclass, field

@dataclass
class Student:
    name: str
    age: int = 18  # قيمة افتراضية بسيطة
    grades: list[float] = field(default_factory=list)  # للقيم القابلة للتعديل!

s1 = Student("أحمد")
s1.grades.append(90)
print(s1)  # Student(name='أحمد', age=18, grades=[90])

s2 = Student("سارة")
print(s2)  # Student(name='سارة', age=18, grades=[])  -- مستقلة تماماً!

# ────────────────────────────────────────────────────────────────────────
# [042] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-41
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.5: Dataclasses: اختصار كتابة الفئات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-016-student.py
"""

@dataclass
class Student:
    grades: list = []  # 💥 ValueError: mutable default <class 'list'> is not allowed

# ────────────────────────────────────────────────────────────────────────
# [043] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-42
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.5: Dataclasses: اختصار كتابة الفئات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-017-area.py
"""

from dataclasses import dataclass

@dataclass
class Rectangle:
    width: float
    height: float

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)

rect = Rectangle(5, 3)
print(rect)             # Rectangle(width=5, height=3)
print(rect.area())      # 15

# ────────────────────────────────────────────────────────────────────────
# [044] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-43
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.5: Dataclasses: اختصار كتابة الفئات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-018-immutable_point.py
"""

from dataclasses import dataclass

@dataclass(frozen=True)  # يجعل الكائن immutable - كالـ tuple!
class ImmutablePoint:
    x: float
    y: float

p = ImmutablePoint(3, 4)
# p.x = 10
# 💥 FrozenInstanceError: cannot assign to field 'x'

@dataclass(order=True)  # يُضيف __lt__, __le__, __gt__, __ge__ تلقائياً
class Version:
    major: int
    minor: int

v1 = Version(1, 5)
v2 = Version(2, 0)
print(v1 < v2)  # True -- تقارن الحقول بالترتيب المُعرَّف

# ────────────────────────────────────────────────────────────────────────
# [045] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-44
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.6: مزيد من Magic Methods: `__call__`, `__getitem__`, Context Managers

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-019-__init__.py
"""

class Multiplier:
    def __init__(self, factor: float):
        self.factor = factor

    def __call__(self, value: float) -> float:
        return value * self.factor

double = Multiplier(2)
print(double(5))   # 10 -- الكائن نفسه يُستدعى كدالة!
print(double(10))  # 20

# ────────────────────────────────────────────────────────────────────────
# [046] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-45
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.6: مزيد من Magic Methods: `__call__`, `__getitem__`, Context Managers

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-020-__init__.py
"""

class Playlist:
    def __init__(self, songs: list[str]):
        self.songs = songs

    def __getitem__(self, index: int) -> str:
        return self.songs[index]

    def __len__(self) -> int:
        return len(self.songs)

playlist = Playlist(["أغنية 1", "أغنية 2", "أغنية 3"])
print(playlist[0])  # أغنية 1  -- الفهرسة تعمل الآن!

for song in playlist:  # __getitem__ تجعل الكائن قابلاً للتكرار تلقائياً أيضاً!
    print(song)

# ────────────────────────────────────────────────────────────────────────
# [047] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-46
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.6: مزيد من Magic Methods: `__call__`, `__getitem__`, Context Managers

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-021-__enter__.py
"""

class Timer:
    """context manager يقيس زمن تنفيذ كتلة كود."""

    def __enter__(self):
        import time
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        import time
        self.end = time.perf_counter()
        print(f"⏱️ استغرق التنفيذ: {self.end - self.start:.4f} ثانية")

with Timer():
    total = sum(i ** 2 for i in range(1_000_000))
    print(f"النتيجة: {total}")

# ────────────────────────────────────────────────────────────────────────
# [048] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-47
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.7: مشروع تطبيقي متكامل: نظام حجز مكتبة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-022-__str__.py
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto

# ============ الطبقة 1: نماذج البيانات (Dataclasses) ============

class ItemStatus(Enum):
    """حالة العنصر في المكتبة."""
    AVAILABLE = auto()
    BORROWED = auto()
    RESERVED = auto()

@dataclass
class Member:
    """يمثل عضواً في المكتبة."""
    member_id: int
    name: str
    borrowed_items: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"عضو #{self.member_id}: {self.name}"

# ============ الطبقة 2: الفئة المجردة للعناصر القابلة للاستعارة ============

class LibraryItem(ABC):
    """فئة مجردة أساسية لأي عنصر في المكتبة."""

    def __init__(self, item_id: str, title: str, loan_days: int):
        self.item_id = item_id
        self.title = title
        self.loan_days = loan_days  # عدد الأيام المسموحة للاستعارة
        self.status = ItemStatus.AVAILABLE
        self.borrowed_by: Member | None = None
        self.due_date: datetime | None = None

    @abstractmethod
    def get_category(self) -> str:
        """كل فئة فرعية يجب أن تحدد فئتها."""
        pass

    @abstractmethod
    def display_details(self) -> str:
        """كل فئة فرعية تعرض تفاصيلها الخاصة."""
        pass

    def borrow(self, member: Member) -> bool:
        """استعارة العنصر - سلوك مشترك لكل الأنواع."""
        if self.status != ItemStatus.AVAILABLE:
            print(f"❌ '{self.title}' غير متاح حالياً (الحالة: {self.status.name}).")
            return False

        self.status = ItemStatus.BORROWED
        self.borrowed_by = member
        self.due_date = datetime.now() + timedelta(days=self.loan_days)
        member.borrowed_items.append(self.item_id)

        print(f"✅ استعار {member.name} '{self.title}'. "
              f"الإرجاع المتوقع: {self.due_date.strftime('%Y-%m-%d')}")
        return True

    def return_item(self) -> None:
        """إرجاع العنصر."""
        if self.status != ItemStatus.BORROWED:
            print(f"⚠️ '{self.title}' ليس مُستعاراً أصلاً.")
            return

        if self.borrowed_by:
            self.borrowed_by.borrowed_items.remove(self.item_id)
            print(f"✅ تم إرجاع '{self.title}' من {self.borrowed_by.name}.")

        self.status = ItemStatus.AVAILABLE
        self.borrowed_by = None
        self.due_date = None

    def is_overdue(self) -> bool:
        """التحقق من تأخر الإرجاع - سلوك مشترك."""
        if self.status != ItemStatus.BORROWED or self.due_date is None:
            return False
        return datetime.now() > self.due_date

    def __str__(self) -> str:
        status_display = self.status.name
        return f"[{self.get_category()}] {self.title} ({self.item_id}) - {status_display}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, LibraryItem):
            return False
        return self.item_id == other.item_id

    def __hash__(self) -> int:
        return hash(self.item_id)

# ============ الطبقة 3: الفئات الفرعية (تطبيق تعدد الأشكال) ============

class Book(LibraryItem):
    def __init__(self, item_id: str, title: str, author: str, pages: int):
        super().__init__(item_id, title, loan_days=14)  # الكتب تُستعار 14 يوماً
        self.author = author
        self.pages = pages

    def get_category(self) -> str:
        return "كتاب"

    def display_details(self) -> str:
        return f"📖 {self.title} - تأليف: {self.author} ({self.pages} صفحة)"

class DVD(LibraryItem):
    def __init__(self, item_id: str, title: str, duration_minutes: int):
        super().__init__(item_id, title, loan_days=5)  # الأفلام تُستعار 5 أيام فقط
        self.duration_minutes = duration_minutes

    def get_category(self) -> str:
        return "دي في دي"

    def display_details(self) -> str:
        return f"🎬 {self.title} - المدة: {self.duration_minutes} دقيقة"

class Magazine(LibraryItem):
    def __init__(self, item_id: str, title: str, issue_number: int):
        super().__init__(item_id, title, loan_days=3)  # المجلات 3 أيام فقط
        self.issue_number = issue_number

    def get_category(self) -> str:
        return "مجلة"

    def display_details(self) -> str:
        return f"📰 {self.title} - العدد: {self.issue_number}"

# ============ الطبقة 4: استثناءات مخصصة ============

class LibraryError(Exception):
    """الفئة الأساسية لكل أخطاء نظام المكتبة."""
    pass

class ItemNotFoundError(LibraryError):
    def __init__(self, item_id: str):
        super().__init__(f"العنصر '{item_id}' غير موجود في المكتبة.")

class MemberNotFoundError(LibraryError):
    def __init__(self, member_id: int):
        super().__init__(f"العضو رقم {member_id} غير موجود.")

# ============ الطبقة 5: النظام الرئيسي ============

class Library:
    """يُدير كل عمليات المكتبة: الأعضاء، العناصر، الاستعارة، الإرجاع."""

    def __init__(self, name: str):
        self.name = name
        self.items: dict[str, LibraryItem] = {}
        self.members: dict[int, Member] = {}

    # --- إدارة العناصر ---
    def add_item(self, item: LibraryItem) -> None:
        self.items[item.item_id] = item
        print(f"📚 أُضيف '{item.title}' إلى مكتبة '{self.name}'.")

    def find_item(self, item_id: str) -> LibraryItem:
        try:
            return self.items[item_id]
        except KeyError:
            raise ItemNotFoundError(item_id)

    # --- إدارة الأعضاء ---
    def register_member(self, member: Member) -> None:
        self.members[member.member_id] = member
        print(f"👤 تم تسجيل {member.name} كعضو جديد.")

    def find_member(self, member_id: int) -> Member:
        try:
            return self.members[member_id]
        except KeyError:
            raise MemberNotFoundError(member_id)

    # --- العمليات الأساسية ---
    def borrow_item(self, item_id: str, member_id: int) -> None:
        item = self.find_item(item_id)
        member = self.find_member(member_id)
        item.borrow(member)

    def return_item(self, item_id: str) -> None:
        item = self.find_item(item_id)
        item.return_item()

    # --- تقارير (تطبيق تعدد الأشكال بوضوح) ---
    def list_all_items(self) -> None:
        print(f"\n📋 كل عناصر مكتبة '{self.name}':")
        print("-" * 60)
        for item in self.items.values():
            print(item.display_details())  # كل نوع يعرض تفاصيله الخاصة!
        print("-" * 60)

    def list_available_items(self) -> list[LibraryItem]:
        return [item for item in self.items.values()
                if item.status == ItemStatus.AVAILABLE]

    def list_overdue_items(self) -> list[LibraryItem]:
        return [item for item in self.items.values() if item.is_overdue()]

    def items_by_category(self) -> dict[str, int]:
        """عدد العناصر حسب الفئة - باستخدام polymorphism (get_category)."""
        counts: dict[str, int] = {}
        for item in self.items.values():
            category = item.get_category()
            counts[category] = counts.get(category, 0) + 1
        return counts

# ============ التشغيل التجريبي ============

if __name__ == "__main__":
    library = Library("مكتبة الأمل العامة")

    # إضافة عناصر متنوعة (تعدد الأشكال في العمل)
    library.add_item(Book("B001", "بايثون للمحترفين", "محمد أحمد", 350))
    library.add_item(Book("B002", "Clean Code", "Robert Martin", 400))
    library.add_item(DVD("D001", "رحلة إلى الفضاء", 120))
    library.add_item(Magazine("M001", "التقنية اليوم", 45))

    # تسجيل أعضاء
    library.register_member(Member(1, "أحمد"))
    library.register_member(Member(2, "سارة"))

    print()
    library.list_all_items()

    print("\n--- عمليات الاستعارة ---")
    library.borrow_item("B001", 1)
    library.borrow_item("D001", 2)

    try:
        library.borrow_item("B001", 2)  # محاولة استعارة كتاب مُستعار بالفعل
    except LibraryError as e:
        print(f"❌ {e}")

    try:
        library.borrow_item("X999", 1)  # عنصر غير موجود
    except ItemNotFoundError as e:
        print(f"❌ {e}")

    print("\n--- حالة المكتبة ---")
    library.list_all_items()

    available = library.list_available_items()
    print(f"\n✅ العناصر المتاحة حالياً: {len(available)}")
    for item in available:
        print(f"  - {item}")

    print(f"\n📊 توزيع العناصر حسب الفئة: {library.items_by_category()}")

    print("\n--- إرجاع عنصر ---")
    library.return_item("B001")

    library.list_all_items()

# ────────────────────────────────────────────────────────────────────────
# [049] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-48
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.7: مشروع تطبيقي متكامل: نظام حجز مكتبة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-023-calculate_fine.py
"""

class LibraryItem(ABC):
    # ... (الكود السابق كما هو) ...
    FINE_PER_DAY = 2.0  # ريال لكل يوم تأخير

    def calculate_fine(self) -> float:
        """حساب الغرامة إذا كان العنصر متأخراً."""
        if not self.is_overdue() or self.due_date is None:
            return 0.0
        days_late = (datetime.now() - self.due_date).days
        return days_late * self.FINE_PER_DAY

# استخدام:
item = library.find_item("B001")
fine = item.calculate_fine()
if fine > 0:
    print(f"⚠️ غرامة تأخير: {fine} ر.س")

# ────────────────────────────────────────────────────────────────────────
# [050] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-49
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-024-area.py
"""

# تمرين 1
class Shape:
    def area(self) -> float:
        return 0

class Square(Shape):
    def __init__(self, side: float):
        self.side = side

    def area(self) -> float:
        return self.side ** 2

sq = Square(4)
print(sq.area())  # 16

# تمرين 2
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

p1 = Point(1, 2)
p2 = Point(1, 2)
print(p1)          # Point(x=1, y=2)
print(p1 == p2)     # True

# ────────────────────────────────────────────────────────────────────────
# [051] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-50
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-025-process.py
"""

# تمرين 3
from abc import ABC, abstractmethod

class PaymentMethod(ABC):
    @abstractmethod
    def process(self, amount: float) -> None:
        pass

class CreditCard(PaymentMethod):
    def process(self, amount: float) -> None:
        print(f"💳 تمت معالجة {amount} ر.س عبر البطاقة الائتمانية.")

class PayPal(PaymentMethod):
    def process(self, amount: float) -> None:
        print(f"🅿️ تمت معالجة {amount} ر.س عبر PayPal.")

methods: list[PaymentMethod] = [CreditCard(), PayPal()]
for method in methods:
    method.process(100)

# تمرين 4
from dataclasses import dataclass

@dataclass(frozen=True)
class Product:
    name: str
    price: float
    quantity: int = 1

    def total(self) -> float:
        return self.price * self.quantity

p = Product("لابتوب", 3500, 2)
print(p.total())  # 7000
# p.price = 4000  # 💥 FrozenInstanceError

# ────────────────────────────────────────────────────────────────────────
# [052] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-51
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-026-__init__.py
"""

from abc import ABC, abstractmethod

class Employee(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def calculate_salary(self) -> float:
        pass

    def __str__(self) -> str:
        return f"{self.name}: {self.calculate_salary():.2f} ر.س"

class FullTimeEmployee(Employee):
    def __init__(self, name: str, monthly_salary: float):
        super().__init__(name)
        self.monthly_salary = monthly_salary

    def calculate_salary(self) -> float:
        return self.monthly_salary

class FreelanceEmployee(Employee):
    def __init__(self, name: str, hourly_rate: float, hours_worked: float):
        super().__init__(name)
        self.hourly_rate = hourly_rate
        self.hours_worked = hours_worked

    def calculate_salary(self) -> float:
        return self.hourly_rate * self.hours_worked

class Payroll:
    def __init__(self):
        self.employees: list[Employee] = []

    def add_employee(self, employee: Employee) -> None:
        self.employees.append(employee)

    def total_payroll(self) -> float:
        return sum(emp.calculate_salary() for emp in self.employees)

    def print_report(self) -> None:
        print("📋 تقرير الرواتب:")
        for emp in self.employees:
            print(f"  {emp}")
        print(f"الإجمالي: {self.total_payroll():.2f} ر.س")

payroll = Payroll()
payroll.add_employee(FullTimeEmployee("أحمد", 8000))
payroll.add_employee(FreelanceEmployee("سارة", 50, 120))
payroll.print_report()

# 📋 تقرير الرواتب:
#   أحمد: 8000.00 ر.س
#   سارة: 6000.00 ر.س
# الإجمالي: 14000.00 ر.س

# ────────────────────────────────────────────────────────────────────────
# [053] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-52
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.10: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-027-__init__.py
"""

# ============ الوراثة ============
class Parent:
    def __init__(self, x):
        self.x = x
    def greet(self):
        return f"Parent: {self.x}"

class Child(Parent):
    def __init__(self, x, y):
        super().__init__(x)    # استدعاء __init__ الأساسي
        self.y = y
    def greet(self):
        return f"{super().greet()}, Child: {self.y}"  # تمديد السلوك

# ============ تعدد الأشكال ============
for obj in [ClassA(), ClassB(), ClassC()]:
    obj.common_method()  # كل فئة تُنفّذ بطريقتها

# ============ الفئات المجردة ============
from abc import ABC, abstractmethod

class Base(ABC):
    @abstractmethod
    def required_method(self): ...

# Base()  # 💥 TypeError - لا يمكن إنشاء كائن مباشرة

# ============ Dataclasses ============
from dataclasses import dataclass, field

@dataclass
class Point:
    x: float
    y: float
    tags: list = field(default_factory=list)  # ⚠️ ليس []

@dataclass(frozen=True)   # immutable
@dataclass(order=True)    # يدعم < <= > >=

# ============ Magic Methods إضافية ============
def __call__(self, *args): ...      # obj(args)
def __getitem__(self, i): ...       # obj[i]
def __enter__(self): ...            # with obj as x:
def __exit__(self, *args): ...

# ============ isinstance ============
isinstance(obj, SomeClass)          # ✅ يأخذ الوراثة بعين الاعتبار
type(obj) == SomeClass              # ❌ تطابق دقيق فقط

# ────────────────────────────────────────────────────────────────────────
# [054] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-53
# ────────────────────────────────────────────────────────────────────────
class Manager(Employee):
    def __init__(self, name, salary, team_size):
        # ⚠️ نسيان super().__init__(name, salary)!
        self.team_size = team_size

manager = Manager("نورة", 15000, 8)
print(manager.name)
# 💥 AttributeError: 'Manager' object has no attribute 'name'

# ────────────────────────────────────────────────────────────────────────
# [055] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-54
# ────────────────────────────────────────────────────────────────────────
# ❌ لا يأخذ الوراثة في الحسبان
if type(dog) == Animal:  # False حتى لو dog هو كلب (نوع فرعي من Animal)!
    pass

# ✅ يأخذ الوراثة في الحسبان
if isinstance(dog, Animal):  # True
    pass

# ────────────────────────────────────────────────────────────────────────
# [056] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-55
# ────────────────────────────────────────────────────────────────────────
from abc import ABC, abstractmethod

class Shape(ABC):  # ترث من ABC (Abstract Base Class)
    @abstractmethod
    def area(self) -> float:
        """يجب أن تُخصَّص في كل فئة فرعية."""
        pass

    @abstractmethod
    def perimeter(self) -> float:
        """يجب أن تُخصَّص في كل فئة فرعية."""
        pass

    def describe(self) -> str:
        """method عادية - موروثة كما هي، لا حاجة لتخصيصها."""
        return f"{type(self).__name__}: مساحة={self.area():.2f}, محيط={self.perimeter():.2f}"

class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius

    def area(self) -> float:
        return 3.14159 * self.radius ** 2

    def perimeter(self) -> float:
        return 2 * 3.14159 * self.radius

# shape = Shape()
# 💥 TypeError: Can't instantiate abstract class Shape with abstract methods area, perimeter

circle = Circle(5)
print(circle.describe())  # Circle: مساحة=78.54, محيط=31.42

class Square(Shape):
    def __init__(self, side: float):
        self.side = side
    # ⚠️ لو نسينا تعريف area() أو perimeter() هنا:

# square = Square(5)
# 💥 TypeError: Can't instantiate abstract class Square with abstract method area, perimeter

# ────────────────────────────────────────────────────────────────────────
# [057] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-56
# ────────────────────────────────────────────────────────────────────────
@dataclass
class Student:
    grades: list = []  # 💥 ValueError: mutable default <class 'list'> is not allowed

# ────────────────────────────────────────────────────────────────────────
# [058] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-57
# ────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass

@dataclass(frozen=True)  # يجعل الكائن immutable - كالـ tuple!
class ImmutablePoint:
    x: float
    y: float

p = ImmutablePoint(3, 4)
# p.x = 10
# 💥 FrozenInstanceError: cannot assign to field 'x'

@dataclass(order=True)  # يُضيف __lt__, __le__, __gt__, __ge__ تلقائياً
class Version:
    major: int
    minor: int

v1 = Version(1, 5)
v2 = Version(2, 0)
print(v1 < v2)  # True -- تقارن الحقول بالترتيب المُعرَّف

# ────────────────────────────────────────────────────────────────────────
# [059] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-58
# ────────────────────────────────────────────────────────────────────────
class LibraryItem(ABC):
    # ... (الكود السابق كما هو) ...
    FINE_PER_DAY = 2.0  # ريال لكل يوم تأخير

    def calculate_fine(self) -> float:
        """حساب الغرامة إذا كان العنصر متأخراً."""
        if not self.is_overdue() or self.due_date is None:
            return 0.0
        days_late = (datetime.now() - self.due_date).days
        return days_late * self.FINE_PER_DAY

# استخدام:
item = library.find_item("B001")
fine = item.calculate_fine()
if fine > 0:
    print(f"⚠️ غرامة تأخير: {fine} ر.س")

# ────────────────────────────────────────────────────────────────────────
# [060] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-59
# ────────────────────────────────────────────────────────────────────────
# تمرين 3
from abc import ABC, abstractmethod

class PaymentMethod(ABC):
    @abstractmethod
    def process(self, amount: float) -> None:
        pass

class CreditCard(PaymentMethod):
    def process(self, amount: float) -> None:
        print(f"💳 تمت معالجة {amount} ر.س عبر البطاقة الائتمانية.")

class PayPal(PaymentMethod):
    def process(self, amount: float) -> None:
        print(f"🅿️ تمت معالجة {amount} ر.س عبر PayPal.")

methods: list[PaymentMethod] = [CreditCard(), PayPal()]
for method in methods:
    method.process(100)

# تمرين 4
from dataclasses import dataclass

@dataclass(frozen=True)
class Product:
    name: str
    price: float
    quantity: int = 1

    def total(self) -> float:
        return self.price * self.quantity

p = Product("لابتوب", 3500, 2)
print(p.total())  # 7000
# p.price = 4000  # 💥 FrozenInstanceError

# ────────────────────────────────────────────────────────────────────────
# [061] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-14-chapter-14.ipynb#cell-60
# ────────────────────────────────────────────────────────────────────────
# ============ الوراثة ============
class Parent:
    def __init__(self, x):
        self.x = x
    def greet(self):
        return f"Parent: {self.x}"

class Child(Parent):
    def __init__(self, x, y):
        super().__init__(x)    # استدعاء __init__ الأساسي
        self.y = y
    def greet(self):
        return f"{super().greet()}, Child: {self.y}"  # تمديد السلوك

# ============ تعدد الأشكال ============
for obj in [ClassA(), ClassB(), ClassC()]:
    obj.common_method()  # كل فئة تُنفّذ بطريقتها

# ============ الفئات المجردة ============
from abc import ABC, abstractmethod

class Base(ABC):
    @abstractmethod
    def required_method(self): ...

# Base()  # 💥 TypeError - لا يمكن إنشاء كائن مباشرة

# ============ Dataclasses ============
from dataclasses import dataclass, field

@dataclass
class Point:
    x: float
    y: float
    tags: list = field(default_factory=list)  # ⚠️ ليس []

@dataclass(frozen=True)   # immutable
@dataclass(order=True)    # يدعم < <= > >=

# ============ Magic Methods إضافية ============
def __call__(self, *args): ...      # obj(args)
def __getitem__(self, i): ...       # obj[i]
def __enter__(self): ...            # with obj as x:
def __exit__(self, *args): ...

# ============ isinstance ============
isinstance(obj, SomeClass)          # ✅ يأخذ الوراثة بعين الاعتبار
type(obj) == SomeClass              # ❌ تطابق دقيق فقط
