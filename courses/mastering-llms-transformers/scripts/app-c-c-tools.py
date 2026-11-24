#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الملحق ج: المكتبات والأدوات
Mastering Transformers and Large Language Models — C Tools

Book slug : mastering-llms-transformers
Chapter   : app-c-c-tools (lesson c)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/app-c-c-tools.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الملحق ج: المكتبات والأدوات
Mastering Transformers and Large Language Models — C Tools

Book slug : mastering-llms-transformers
Chapter   : app-c-c-tools (lesson c)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/app-c-c-tools.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الملحق ج: المكتبات والأدوات
Mastering Transformers and Large Language Models — C Tools

Book slug : mastering-llms-transformers
Chapter   : app-c-c-tools (lesson c)
Code files: 2 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/appendices/c-tools.md
# ────────────────────────────────────────────────────────────────────────
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")
resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "اشرح الانتباه في جملتين."}],
    temperature=0.2,
)
print(resp.choices[0].message.content)

# ────────────────────────────────────────────────────────────────────────
# [002] content/appendices/c-tools.md
# ────────────────────────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
import faiss

model = SentenceTransformer("intfloat/multilingual-e5-small")
docs = ["الانتباه يحسب أوزانًا لكل رمز.", "RAG يسترجع وثائق قبل التوليد."]
emb = model.encode(docs, normalize_embeddings=True)
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)
scores, ids = index.search(model.encode(["ما هو RAG؟"], normalize_embeddings=True), k=1)
print(scores, ids)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/app-c-c-tools.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")
resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "اشرح الانتباه في جملتين."}],
    temperature=0.2,
)
print(resp.choices[0].message.content)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/app-c-c-tools.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
import faiss

model = SentenceTransformer("intfloat/multilingual-e5-small")
docs = ["الانتباه يحسب أوزانًا لكل رمز.", "RAG يسترجع وثائق قبل التوليد."]
emb = model.encode(docs, normalize_embeddings=True)
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)
scores, ids = index.search(model.encode(["ما هو RAG؟"], normalize_embeddings=True), k=1)
print(scores, ids)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/app-c-c-tools.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الملحق ج: المكتبات والأدوات
Mastering Transformers and Large Language Models — C Tools

Book slug : mastering-llms-transformers
Chapter   : app-c-c-tools (lesson c)
Code files: 2 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/appendices/c-tools.md
# ────────────────────────────────────────────────────────────────────────
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")
resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "اشرح الانتباه في جملتين."}],
    temperature=0.2,
)
print(resp.choices[0].message.content)

# ────────────────────────────────────────────────────────────────────────
# [002] content/appendices/c-tools.md
# ────────────────────────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
import faiss

model = SentenceTransformer("intfloat/multilingual-e5-small")
docs = ["الانتباه يحسب أوزانًا لكل رمز.", "RAG يسترجع وثائق قبل التوليد."]
emb = model.encode(docs, normalize_embeddings=True)
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)
scores, ids = index.search(model.encode(["ما هو RAG؟"], normalize_embeddings=True), k=1)
print(scores, ids)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/app-c-c-tools.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")
resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "اشرح الانتباه في جملتين."}],
    temperature=0.2,
)
print(resp.choices[0].message.content)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/app-c-c-tools.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
import faiss

model = SentenceTransformer("intfloat/multilingual-e5-small")
docs = ["الانتباه يحسب أوزانًا لكل رمز.", "RAG يسترجع وثائق قبل التوليد."]
emb = model.encode(docs, normalize_embeddings=True)
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)
scores, ids = index.search(model.encode(["ما هو RAG؟"], normalize_embeddings=True), k=1)
print(scores, ids)
