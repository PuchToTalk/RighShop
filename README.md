# RighShop Demo
Multimodal shopping assistant that turns an input (image + optional text + optional user behavior) into **high-quality product matches** with **ranked links**.
![Demo](public/Righ-snap.gif)

> **Goal:** reduce “noise” (irrelevant / low-quality items) and return **actionable purchase options** fast.

---

## What is RighShop?
RighShop is a retrieval + ranking pipeline for shopping discovery:
- **You give:** a photo (e.g., outfit screenshot), a query (“black loafers”), and/or lightweight user context (clicks, preferences).
- **It returns:** a ranked list of products with **filtered links**, metadata, and (optionally) explanations.

It’s built to be practical:
- Works on messy, real-world images (social media, street photos, screenshots).
- Keeps the output **high precision** by filtering spam/low-quality results.
- Supports offline evaluation to iterate quickly.

---

## Key Features
- **Multimodal retrieval**: image + text (and optional behavior signals)
- **Object detection & cropping**: identify items (e.g., shoes/bag/jacket) and isolate them for better search
- **Reverse image search / web retrieval**: fetch candidate products from the web
- **Link filtering & re-ranking**: remove low-quality candidates using LLM/rerank scoring
- **Offline evaluation**: measure precision / true positives on held-out test sets
- **Modular pipeline**: swap detectors, embedders, rerankers, or search providers

---

## Pipeline Overview
```mermaid
flowchart LR
  A[Input Image] --> B[Detect Items / Regions]
  B --> C[Crop / Enhance (optional)]
  C --> D[Generate Object Queries (LLM)]
  D --> E[Candidate Retrieval (Web / Catalog)]
  E --> F[Normalize + Deduplicate]
  F --> G[Link Filter (Quality / Relevance)]
  G --> H[Rerank (Multimodal + Text)]
  H --> I[Top-K Products + Explanations]