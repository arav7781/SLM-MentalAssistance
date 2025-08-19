from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
from typing import List, Dict
import os, multiprocessing
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Legal Terms Chatbot API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (dev mode)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model (local fallback, HF Hub if available)
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "hf://aravsaxena884/gemma-270m-gguf/gemma_270m_q4_0.gguf"  # HF Hub URL
)

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=1024,
    n_threads=max(1, multiprocessing.cpu_count() - 1),
    verbose=False
)

# System prompt for legal assistant
SYSTEM_PROMPT = (
    "You are a legal assistant specialized in explaining legal terms in simple, clear language. "
    "Provide accurate and concise explanations of legal terms or concepts based on user input. "
    "If the user asks about a specific legal term, define it and provide an example. "
    "Maintain a professional yet approachable tone."
)

# In-memory conversation history
conversations: Dict[str, List[Dict[str, str]]] = {}

# Request/Response models
class ChatRequest(BaseModel):
    session_id: str
    prompt: str
    max_tokens: int = 100

class ChatResponse(BaseModel):
    response: str
    history: List[Dict[str, str]]

# API endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Init history if new session
        if request.session_id not in conversations:
            conversations[request.session_id] = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]

        # Add user message
        conversations[request.session_id].append({"role": "user", "content": request.prompt})

        # Build conversation into prompt
        full_prompt = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" 
            for msg in conversations[request.session_id]
        )

        # Run inference
        output = llm(
            prompt=full_prompt,
            max_tokens=request.max_tokens,
            echo=False,
            temperature=0.7,
            top_p=0.9
        )

        # Handle output safely
        response_text = (
            output["choices"][0].get("text") 
            or output["choices"][0].get("content", "")
        ).strip()

        # Add assistant reply
        conversations[request.session_id].append({"role": "assistant", "content": response_text})

        # Trim history (last 10 turns)
        conversations[request.session_id] = conversations[request.session_id][-10:]

        return {
            "response": response_text,
            "history": conversations[request.session_id]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


# Main entrypoint (for local run)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
