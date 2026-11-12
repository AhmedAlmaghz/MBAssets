# MaghzBook — Assets

حزمة الأصول الكاملة لمكتبة **MaghzBook**: **14 كتابًا** عربيًا تقنيًا، ولكل كتاب **دورة مطابقة**،
بالإضافة إلى **صفحة هبوط إعلانية وتسويقية** في جذر الحزمة. كل ملف جاهز للاستخدام المباشر في التطبيق،
و**كل الروابط داخلية (نسبية)** بلا أي اعتماد على اسم مضيف ثابت.

---

## المحتويات

| المسار | الوصف |
|---|---|
| `index.html` | **الصفحة الرئيسية الإعلانية والتسويقية** للكتب والدورات — ملف واحد مستقل، بيانات الكتالوج مضمّنة، بحث وتصفية وروابط داخلية |
| `landing/` | 14 صفحة هبوط للكتب + `index.html` فهرس المكتبة — ملف واحد مستقل لكل صفحة (HTML + CSS + JSON-LD، بلا ملفات مساندة) |
| `books/` | 14 كتابًا: الغلاف، الكتاب الكامل PDF، النموذج المجاني PDF، وملف blueprint للكتاب |
| `courses/` | 14 دورة: blueprint الدورة، دفاتر Jupyter، السكربتات، وصور الشرح |
| `brand/` | شعار MaghzBook بصيغه كافة + الخطوط العربية المستخدمة في الصفحة الرئيسية |
| `tools/` | أدوات الروابط: `app-url.js`، `resolve_urls.py`، `verify_internal.py` |
| `catalog.json` | فهرس خفيف للعرض والبحث (بلا نصوص الدروس) |
| `manifest.json` | فهرس الآلة: المسارات والأحجام وبصمات SHA-256 وبيانات بيئة `APP_URL` |
| `.env.example` | قالب متغير البيئة `APP_URL` |

### الإحصاءات

| البند | العدد |
|---|---:|
| كتب | 14 |
| دورات | 14 |
| وحدات (modules) | 118 |
| دروس / فصول | 643 |
| صفحات الكتب | 9,982 |
| مرفقات (دفاتر + سكربتات + صور) | 1,307 |
| أغلفة | 14 |
| كتب كاملة PDF | 14 |
| نماذج مجانية PDF | 14 |
| صفحات هبوط | 14 + فهرس + الرئيسية |

---

## البنية

```text
MaghzBook/
├── index.html               ← الصفحة الإعلانية الرئيسية
├── landing/<slug>.html       ← صفحة هبوط لكل كتاب
├── books/<slug>/
│   ├── cover.png
│   ├── book.pdf
│   ├── sample.pdf
│   └── book.blueprint.json
├── courses/<slug>/
│   ├── course.blueprint.json
│   ├── notebooks/
│   ├── scripts/
│   └── media/
├── brand/
├── tools/
├── manifest.json
├── catalog.json
└── .env.example
```

`<slug>` معرّف ثابت للكتاب/الدورة، وهو نفسه `linkedBookSlug` داخل ملف الدورة، ما يجعل الاستيراد
متكرّرًا بأمان (idempotent) عبر `mode: "upsert"`.

---

## الروابط داخلية بالكامل

لا يوجد أي رابط مطلق لأصل داخل الحزمة. كل رابط هو **مسار نسبي من جذر الحزمة**
(`books/<slug>/cover.png`، `landing/<slug>.html`، …). هذا يجعل الحزمة قابلة للعمل:

- من نفس أصل التطبيق (same-origin)،
- من مستضيف ثابت أو GitHub Pages أو CDN،
- من نظام ملفات محلي (`file://`) للمعاينة،
- دون أي تعديل أو إعادة بناء.

في صفحات HTML يُكتب المسار نسبيًا إلى موضع الصفحة (`../books/…` من `landing/`)، ويحمل كل عنصر
السمة `data-mb-path` بالمسار الكنسي من جذر الحزمة. في ملفات JSON تُخزَّن المسارات الكنسية مباشرة.

> الروابط الخارجية الوحيدة المتبقية هي **مراجع ومصادر** وليست أصولًا: خطوط Google Fonts في صفحات
> الكتب، وروابط الاستشهاد (arXiv، Colab، فصول GitHub الأصلية). هذه لا تُحمَّل من الحزمة.

### متغير البيئة `APP_URL`

عند الحاجة إلى روابط مطلقة — أصل أصول منفصل، CDN، أو استيراد Blueprints من جهة الخادم —
يُستخدم متغير البيئة `APP_URL`، ويُضاف **قبل كل رابط**:

```text
resolve("books/game-theory-strategy/cover.png")

APP_URL=                          → books/game-theory-strategy/cover.png
APP_URL=https://books.example.com → https://books.example.com/books/game-theory-strategy/cover.png
```

| الطريقة | الاستخدام |
|---|---|
| وقت التشغيل | `window.APP_URL = import.meta.env.VITE_APP_URL` قبل تحميل الصفحة |
| وقت البناء | `APP_URL=https://books.example.com python tools/resolve_urls.py --export ../dist` |
| في الموضع | `APP_URL=https://books.example.com python tools/resolve_urls.py --in-place` |
| Node | `const { resolve } = require('./tools/app-url.js'); resolve(path)` |
| المتصفح | `<script src="tools/app-url.js"></script>` ثم `MaghzBookURL.resolve(path)` |

`--export` ينسخ الحزمة كاملة (ربط صلب للملفات الثنائية لتوفير المساحة) ويحقن `APP_URL` في
`<meta name="app-url">` ويُسبقه على كل رابط في HTML و JSON. `--in-place` يعدّل الحزمة نفسها.
كلا الأمرين آمنان عند إعادة التنفيذ.

انسخ `.env.example` إلى `.env` واضبط `APP_URL`، أو مرّره كمتغير بيئة في CI/CD. تركه فارغًا
هو الوضع الافتراضي والموصى به عندما تُخدَم الأصول من نفس أصل التطبيق.

### التحقّق

```bash
python tools/verify_internal.py
```

يتحقّق من خلوّ الحزمة من الروابط المطلقة الداخلية، ومن وجود كل هدف `data-mb-path` وكل مسار
نسبي وكل مسار في JSON على القرص، ومن مطابقة الأحجام وبصمات SHA-256 في `manifest.json`،
ومن سلامة ترتيب `<meta charset>` وحاقن `APP_URL` في كل صفحة HTML.

---

## الاستيراد إلى MaghzBook

لكل كتاب ملفّا blueprint مستقلّان، وكلاهما يحمل نفس المغلّف:

```json
{
  "maghzbook": { "schema": "maghzbook.blueprint", "version": 1, "generator": "MaghzBook Admin" },
  "kind": "course",
  "mode": "upsert",
  "options": { "replaceCurriculum": false, "connectPlansBundles": true }
}
```

- **الكتاب** (`kind: "book"`) — الغلاف، الكتاب الكامل، النموذج، الصفحات، السعر، الوسوم.
- **الدورة** (`kind: "course"`) — الوحدات والدروس بنصوصها الكاملة، مع المرفقات لكل درس.

`replaceCurriculum: false` يحفظ تقدّم المتعلّمين عند إعادة الاستيراد، و`connectPlansBundles: true`
يحافظ على ربط الخطط والحزم.

مسارات الاستيراد الداخلية:

```text
books/<slug>/book.blueprint.json
courses/<slug>/course.blueprint.json
```

وللاستيراد من الخادم حيث يلزم رابط مطلق: `resolve("books/<slug>/book.blueprint.json")`
أو شغّل `tools/resolve_urls.py` مع `APP_URL` كما أعلى.

---

## الغلاف والعلامة

أغلفة الكتب تحمل علامة **MaghzBook** فقط؛ العلامة المائية السابقة أُزيلت من كل صورة،
والشعار الجديد مرسوم على خلفية شفافة في `brand/`:

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
- المعرّفات (slugs) مطابقة لنمط المخطّط الرسمي `^[\p{L}\p{N}][\p{L}\p{N}_-]*$` ومُتحقَّق منها
  بمحرّك JavaScript الأصلي.
- ملفات JSON مُتحقَّق منها مقابل مخطّط `maghzbook.blueprint` الرسمي: **0 أخطاء، 0 تحذيرات**.
- كل الروابط الداخلية مُتحقَّق منها مقابل الملفات الفعلية: `python tools/verify_internal.py`.

---

## الترخيص

جميع الكتب والمواد محفوظة الحقوق للمؤلّف **م/ أحمد صالح المغز**.
هذا المستودع مخصّص لاستخدام تطبيق MaghzBook وعرض الكتب؛ يُرجى عدم إعادة النشر التجاري دون إذن.
