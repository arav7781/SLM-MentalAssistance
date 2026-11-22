---
title: Legal
emoji: 🏃
colorFrom: indigo
colorTo: red
sdk: docker
pinned: false
---

# Small legal-terms assistant (Gemma 270M, llama.cpp)

A tiny chatbot API that explains legal terms in plain language. It runs a 4-bit quantised Gemma 270M model (GGUF)
with llama.cpp on CPU, behind FastAPI, and is packaged as a Docker image for Hugging Face Spaces. The front matter
above is the Spaces config.

## API

`POST /chat`

```json
{ "session_id": "abc", "prompt": "What is an affidavit?", "max_tokens": 100 }
```

Returns the answer plus the session's message history. History is kept in memory per `session_id`, with a fixed
system prompt that keeps answers short, accurate and example-driven.

## Run

```bash
docker build -t legal-slm .
docker run -p 8000:8000 -e MODEL_PATH=/models/gemma_270m_q4_0.gguf -v $PWD/models:/models legal-slm
```

`MODEL_PATH` defaults to the quantised model on the Hugging Face Hub (`aravsaxena884/gemma-270m-gguf`).
The server uses all but one CPU core and a 1,024-token context.

## Tech

Python · FastAPI · llama-cpp-python · GGUF (Q4_0) · Docker · Hugging Face Spaces
