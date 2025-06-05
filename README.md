# 🤖 Semantic Cache Response for AI Queries

A pilot project to optimize AI response handling by **caching semantically similar queries**. Built on Sitecore with ONNX-based embeddings.

---

## 🚀 Motivation

Repeated user questions increase AI load, cost, and latency. This project uses semantic caching to serve previously generated answers if a new question is *similar enough*.

---

## 🧠 How It Works

1. A user asks a question.
2. The system generates an ONNX embedding vector.
3. It compares it to cached embeddings.
4. If similarity ≥ `0.80`, reuse the cached answer.
5. Else, fetch new AI response and cache it.

---

## 🛠️ Key Configuration (from `SemanticCache.config`)

```xml
<setting name="SemanticCache.SimilarityThreshold" value="0.80" />
<setting name="SemanticCache.ExpirationHours" value="4" />
<setting name="SemanticCache.MaxEntries" value="1000" />
<setting name="AICache.ModelPath" value="/App_Data/models/model.onnx" />
<setting name="AICache.VocabPath" value="/App_Data/models/vocab.txt" />
