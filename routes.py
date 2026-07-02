"""
routes.py — Rotas da API.
Mantém os endpoints separados da lógica de negócio (llm.py e rag.py).
"""

import os
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

import llm
import rag

router = APIRouter()
templates = Jinja2Templates(directory="templates")

API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "").strip()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    response: str
    source: str
    intent: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Página inicial do chatbot."""
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/health")
async def health():
    """Healthcheck para orquestração e contêineres."""
    return {"status": "ok"}


@router.get("/tests", response_class=HTMLResponse)
async def tests_page(request: Request):
    """Página de testes automatizados."""
    from tests import run_all_tests
    results = run_all_tests()
    
    return templates.TemplateResponse("tests.html", {
        "request": request,
        "results": results
    })


@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    _require_api_key()

    message = request.message.strip()
    history = [m.model_dump() for m in request.history]

    # 1. Classifica a intenção via LLM (substitui o roteamento frágil por radicais)
    intent = await llm.classify_intent(message, API_KEY)

    # 2. Recupera contexto relevante (ChromaDB ou fallback)
    context, source = rag.retrieve(intent, message)

    # 3. Gera resposta com histórico completo
    try:
        answer = await llm.generate_answer(
            message=message,
            context=context,
            source=source,
            history=history,
            api_key=API_KEY,
        )
    except httpx_error() as e:
        _handle_http_error(e)

    return ChatResponse(response=answer, source=source, intent=intent)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_api_key() -> None:
    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="DEEPSEEK_API_KEY não configurada. Adicione no arquivo .env.",
        )


def httpx_error():
    """Importação tardia para evitar dependência circular no módulo."""
    import httpx
    return httpx.HTTPStatusError


def _handle_http_error(e) -> None:
    import httpx
    if isinstance(e, httpx.HTTPStatusError):
        status = e.response.status_code
        if status == 429:
            raise HTTPException(status_code=429, detail="Limite de requisições atingido. Tente novamente em instantes.")
        if status == 401:
            raise HTTPException(status_code=401, detail="API Key inválida ou expirada.")
        raise HTTPException(status_code=502, detail=f"Erro na API DeepSeek: HTTP {status}")
    raise HTTPException(status_code=500, detail=str(e))