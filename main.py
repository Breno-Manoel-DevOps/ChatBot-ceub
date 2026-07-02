"""
main.py — Ponto de entrada da aplicação CEUB IA Assistant.

Melhorias implementadas vs. versão anterior:
  ✅ Cliente HTTP único reutilizável (via lifespan)
  ✅ Validação da API Key na inicialização (falha rápido, não na 1ª req)
  ✅ Separação de responsabilidades (llm.py / rag.py / routes.py)
  ✅ Histórico de conversa funcional passado para o LLM
  ✅ Classificação de intenção via LLM (sem radicais frágeis)
  ✅ RAG com ChromaDB semântico + fallback hardcoded
  ✅ Tratamento de erros HTTP específicos (429, 401, timeout)
"""

import os
import sys
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import llm
from routes import router

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()


def _validate_startup() -> None:
    """Encerra a aplicação imediatamente se a configuração estiver incompleta."""
    if not API_KEY:
        print(
            "\n❌ ERRO FATAL: DEEPSEEK_API_KEY não encontrada.\n"
            "   Crie um arquivo .env na raiz do projeto com:\n"
            "   DEEPSEEK_API_KEY=sk-...\n",
            file=sys.stderr,
        )
        sys.exit(1)
    print("✅ API Key carregada com sucesso.")


# ---------------------------------------------------------------------------
# Lifespan — inicializa e finaliza recursos compartilhados
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Cria um único httpx.AsyncClient para toda a vida da aplicação.
    Isso é muito mais eficiente do que criar um cliente por requisição.
    """
    _validate_startup()

    timeout = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)
    limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        llm.set_client(client)
        print("✅ Cliente HTTP inicializado e compartilhado.")
        yield
        print("🔌 Cliente HTTP encerrado.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CEUB IA Assistant",
    description="Chatbot universitário com RAG e DeepSeek",
    version="2.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(router)


# ---------------------------------------------------------------------------
# Entrypoint local
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)