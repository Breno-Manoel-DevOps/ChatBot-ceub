"""
llm.py — Camada de comunicação com a API DeepSeek.
Responsabilidades:
  - Manter um único cliente HTTP reutilizável (injetado via FastAPI lifespan)
  - Classificar a intenção do usuário
  - Gerar a resposta final
"""

import httpx
from typing import Optional


DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-chat"


# ---------------------------------------------------------------------------
# Cliente compartilhado (inicializado no lifespan do FastAPI)
# ---------------------------------------------------------------------------

_client: Optional[httpx.AsyncClient] = None


def set_client(client: httpx.AsyncClient) -> None:
    global _client
    _client = client


def get_client() -> httpx.AsyncClient:
    if _client is None:
        raise RuntimeError("HTTP client não foi inicializado. Verifique o lifespan do FastAPI.")
    return _client


# ---------------------------------------------------------------------------
# Classificação de intenção via LLM
# ---------------------------------------------------------------------------

INTENT_SYSTEM = """
Você é um classificador de intenção para um chatbot universitário do CEUB.
Analise a mensagem do usuário e responda APENAS com uma dessas palavras:

- regimento   → menções, notas, frequência, aprovação, reprovação, abono, regras acadêmicas
- faq         → como acessar sistemas, senha, login, carteirinha, passe estudantil, boleto
- guia        → campi, estacionamento, plataformas, coordenadores, localização
- geral       → qualquer outra coisa

Responda somente a palavra, sem pontuação, sem explicação.
""".strip()


async def classify_intent(message: str, api_key: str) -> str:
    """Retorna a categoria da mensagem: regimento | faq | guia | geral."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": INTENT_SYSTEM},
            {"role": "user", "content": message},
        ],
        "temperature": 0,
        "max_tokens": 10,
    }
    try:
        resp = await get_client().post(
            DEEPSEEK_URL,
            json=payload,
            headers=_auth_headers(api_key),
            timeout=10.0,
        )
        resp.raise_for_status()
        intent = resp.json()["choices"][0]["message"]["content"].strip().lower()
        if intent not in ("regimento", "faq", "guia", "geral"):
            intent = "geral"
        return intent
    except Exception:
        # Se a classificação falhar, usa fallback seguro
        return "geral"


# ---------------------------------------------------------------------------
# Geração de resposta
# ---------------------------------------------------------------------------

ANSWER_SYSTEM = """
Você é o Assistente Oficial do CEUB — educado, preciso e conciso.
Sua ÚNICA fonte de verdade é o CONTEXTO fornecido abaixo.

REGRAS:
1. Se a resposta estiver no contexto, responda com precisão e clareza.
2. Se a resposta NÃO estiver no contexto, diga exatamente:
   "Desculpe, não encontrei essa informação no documento consultado ({source}). 
    Para dúvidas específicas, entre em contato com a Central de Atendimento do CEUB."
3. Nunca invente informações.
4. Use linguagem natural e amigável em português.
5. Quando listar itens, use marcadores simples (•).

CONTEXTO ({source}):
{context}
""".strip()


async def generate_answer(
    message: str,
    context: str,
    source: str,
    history: list[dict],
    api_key: str,
) -> str:
    """Gera a resposta final usando o contexto recuperado do RAG."""
    system_prompt = ANSWER_SYSTEM.format(source=source, context=context)

    # Monta histórico (limita a 10 últimas trocas para não explodir o contexto)
    conversation = _build_conversation(history, message)

    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system_prompt}] + conversation,
        "temperature": 0.15,
    }

    resp = await get_client().post(
        DEEPSEEK_URL,
        json=payload,
        headers=_auth_headers(api_key),
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auth_headers(api_key: str) -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


def _build_conversation(history: list[dict], current_message: str) -> list[dict]:
    """
    Converte o histórico do frontend para o formato messages do DeepSeek.
    Espera histórico no formato: [{"role": "user"|"assistant", "content": "..."}]
    Mantém apenas as últimas 10 trocas (20 mensagens) para controle de tokens.
    """
    trimmed = history[-20:] if len(history) > 20 else history
    conversation = [
        {"role": m["role"], "content": m["content"]}
        for m in trimmed
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    conversation.append({"role": "user", "content": current_message})
    return conversation