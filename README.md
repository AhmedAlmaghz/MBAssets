# MaghzBook — Assets

حزمة الأصول الكاملة لمكتبة **MaghzBook**: **14 كتابًا** عربيًا تقنيًا، ولكل كتاب **دورة مطابقة**.
كل ملف هنا جاهز للاستخدام المباشر في التطبيق، وكل الروابط عامّة وقابلة للقراءة دون أي مصادقة.

---

## المحتويات

| المجلد | الوصف |
|---|---|
| `landing/` | **14 صفحة هبوط** — ملف واحد مستقل لكل كتاب (HTML + CSS + JSON-LD مضمّنة، بلا أي ملفات مساندة) |
| `books/` | **14 كتابًا** — لكل كتاب: الغلاف، الكتاب الكامل PDF، النموذج المجاني PDF، وملف blueprint للكتاب |
| `courses/` | **14 دورة** — blueprint الدورة، ودفاتر Jupyter، والسكربتات، وصور الشرح |
| `brand/` | شعار MaghzBook بخلفية شفافة بكل صيغه (أيقونة، كلمة، قفل أفقي) |
| `manifest.json` | فهرس الآلة: كل مسار وروابطه المطلقة وحجمه وبصمة SHA-256 |
| `catalog.json` | فهرس خفيف للعرض والبحث (بلا نصوص الدروس) |

### الإحصاءات

| البند | العدد |
|---|---:|
| كتب | 14 |
| دورات | 14 |
| وحدات (modules) | 118 |
| دروس | 643 |
| مرفقات (دفاتر + سكربتات) | 1,307 |
| أغلفة | 14 |
| كتب كاملة PDF | 14 |
| نماذج مجانية PDF | 14 |
| صفحات هبوط | 14 |

---

## البنية

```text
MaghzBook/
├── landing/
│   └── <slug>.html
├── books/
│   └── <slug>/
│       ├── cover.png
│       ├── book.pdf
│       ├── sample.pdf
│       └── book.blueprint.json
├── courses/
│   └── <slug>/
│       ├── course.blueprint.json
│       ├── notebooks/
│       ├── scripts/
│       └── media/
├── brand/
├── manifest.json
└── catalog.json
```

`<slug>` هو معرّف ثابت للكتاب/الدورة، وهو نفسه `linkedBookSlug` داخل ملف الدورة، ما يجعل الاستيراد
متكرّرًا بأمان (idempotent) عبر `mode: "upsert"`.

---

## الروابط

كل المسارات تُخدَم من هذا المستودع عبر `raw.githubusercontent.com`:

```text
https://raw.githubusercontent.com/AhmedAlmaghz/MBAssets/main/<path>
```

أمثلة:

```text
الغلاف          books/game-theory-strategy/cover.png
الكتاب الكامل    books/game-theory-strategy/book.pdf
النموذج         books/game-theory-strategy/sample.pdf
blueprint كتاب  books/game-theory-strategy/book.blueprint.json
blueprint دورة  courses/game-theory-strategy/course.blueprint.json
صفحة الهبوط     landing/game-theory-strategy.html
```

> ملاحظة: `raw.githubusercontent.com` يخدم الملفات كما هي. أما صفحات الهبوط فعند فتحها في المتصفح
> مباشرةً ستُعرض كنص خام؛ لعرضها كصفحة ويب استخدم GitHub Pages أو أي مستضيف ثابت يشير إلى مجلد
> `landing/`. محتوى الصفحة نفسه مكتمل ومستقل تمامًا.

---

## الاستيراد إلى MaghzBook

لكل كتاب ملفّا blueprint مستقلّان، وكلاهما يحمل نفس المغلّف:

```json
{
  "maghzbook": {
    "schema": "maghzbook.blueprint",
    "version": 1,
    "generator": "MaghzBook Admin"
  },
  "kind": "course",
  "mode": "upsert",
  "options": { "replaceCurriculum": false, "connectPlansBundles": true }
}
```

- **الكتاب** (`kind: "book"`) — الغلاف، الكتاب الكامل، النموذج، الصفحات، السعر، الوسوم.
- **الدورة** (`kind: "course"`) — الوحدات والدروس بنصوصها الكاملة، مع المرفقات لكل درس.

`replaceCurriculum: false` يحفظ تقدّم المتعلّمين عند إعادة الاستيراد، و`connectPlansBundles: true`
يحافظ على ربط الخطط والحزم.

الاستيراد من رابط مباشر:

```text
https://raw.githubusercontent.com/AhmedAlmaghz/MBAssets/main/books/<slug>/book.blueprint.json
https://raw.githubusercontent.com/AhmedAlmaghz/MBAssets/main/courses/<slug>/course.blueprint.json
```

---

## الغلاف والعلامة

أغلفة الكتب في هذا المستودع تحمل علامة **MaghzBook** فقط. العلامة المائية السابقة أُزيلت من كل
صورة، والشعار الجديد مرسوم على خلفية شفافة في `brand/`:

```text
icon-color.png  icon-white.png  icon-navy.png
wordmark-color.png  wordmark-white.png  wordmark-navy.png
logo-full-color.png
lockup-color.png  lockup-white.png  lockup-navy.png
favicon.png
```

- على الخلفيات الداكنة: `lockup-white.png`
- على الخلفيات الفاتحة: `lockup-navy.png`

---

## جودة المحتوى

- كل الدروس بنصوصها **الكاملة** لا ملخّصات، بالعربية والإنجليزية.
- بيانات الكتاب الوصفية بستّ لغات: `ar, en, fr, es, zh, hi`.
- الوحدات والدروس بلغتين: `ar, en`.
- جميع المعرّفات (slugs) مطابقة لنمط المخطّط الرسمي `^[\p{L}\p{N}][\p{L}\p{N}_-]*$` ومُتحقَّق منها
  بمحرّك JavaScript الأصلي.
- كل ملفات JSON مُتحقَّق منها مقابل مخطّط `maghzbook.blueprint` الرسمي: **0 أخطاء، 0 تحذيرات**.

---

## الترخيص

جميع الكتب والمواد محفوظة الحقوق للمؤلّف **م/ أحمد صالح المغز**.
هذا المستودع مخصّص لاستخدام تطبيق MaghzBook وعرض الكتب؛ يُرجى عدم إعادة النشر التجاري دون إذن.
