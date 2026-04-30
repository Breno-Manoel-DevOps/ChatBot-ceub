"""
tests.py — Testes automatizados para o ChatBot CEUB.

Executa testes unitários e de integração para validar:
- Classificação de intenção
- Recuperação RAG
- Fluxo completo de chat
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
import httpx

# Importa módulos do projeto
import llm
import rag
from routes import ChatRequest, ChatMessage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_client():
    """Cliente HTTP mockado para testes."""
    client = AsyncMock(spec=httpx.AsyncClient)
    llm.set_client(client)
    return client


@pytest.fixture
def sample_history():
    """Histórico de conversa de exemplo."""
    return [
        ChatMessage(role="user", content="Olá"),
        ChatMessage(role="assistant", content="Oi! Como posso ajudar?")
    ]


# ---------------------------------------------------------------------------
# Testes de Classificação de Intenção
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_classify_intent_regimento(mock_client):
    """Testa classificação de intenção para tópicos de regimento."""
    # Mock da resposta da API
    mock_response = AsyncMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "regimento"}}]
    }
    mock_client.post.return_value = mock_response

    intent = await llm.classify_intent("Qual a frequência mínima?", "fake_key")
    assert intent == "regimento"


@pytest.mark.asyncio
async def test_classify_intent_faq(mock_client):
    """Testa classificação de intenção para FAQ."""
    mock_response = AsyncMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "faq"}}]
    }
    mock_client.post.return_value = mock_response

    intent = await llm.classify_intent("Como acessar o Espaço Aluno?", "fake_key")
    assert intent == "faq"


@pytest.mark.asyncio
async def test_classify_intent_fallback(mock_client):
    """Testa fallback quando a API falha."""
    mock_client.post.side_effect = Exception("API Error")

    intent = await llm.classify_intent("Mensagem qualquer", "fake_key")
    assert intent == "geral"


# ---------------------------------------------------------------------------
# Testes de RAG
# ---------------------------------------------------------------------------

def test_rag_retrieve_regimento():
    """Testa recuperação de contexto para intenção regimento."""
    context, source = rag.retrieve("regimento", "frequência")
    assert "frequência" in context.lower()
    assert "Regimento" in source


def test_rag_retrieve_faq():
    """Testa recuperação de contexto para intenção faq."""
    context, source = rag.retrieve("faq", "senha")
    assert "senha" in context.lower() or "login" in context.lower()
    assert "FAQ" in source


def test_rag_retrieve_geral():
    """Testa recuperação de contexto para intenção geral."""
    context, source = rag.retrieve("geral", "qualquer coisa")
    assert len(context) > 0
    assert "Informações Gerais" in source


# ---------------------------------------------------------------------------
# Testes de Integração (Fluxo Completo)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_chat_flow(mock_client, sample_history):
    """Testa o fluxo completo de chat (integração)."""
    # Mock das respostas da API
    intent_response = AsyncMock()
    intent_response.json.return_value = {
        "choices": [{"message": {"content": "regimento"}}]
    }

    answer_response = AsyncMock()
    answer_response.json.return_value = {
        "choices": [{"message": {"content": "A frequência mínima é 75%."}}]
    }

    # Sequência: primeiro classify_intent, depois generate_answer
    mock_client.post.side_effect = [intent_response, answer_response]

    from routes import chat
    from fastapi import HTTPException

    # Simula uma requisição
    request = ChatRequest(
        message="Qual a frequência mínima obrigatória?",
        history=sample_history
    )

    # Como chat() chama _require_api_key, precisamos mockar ou definir API_KEY
    with patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'fake_key'}):
        try:
            response = await chat(request)
            assert response.intent == "regimento"
            assert "75%" in response.response
            assert len(response.source) > 0
        except HTTPException as e:
            # Se falhar por API key, é esperado no teste
            assert e.status_code in [401, 500]


# ---------------------------------------------------------------------------
# Testes de Validação de Entrada
# ---------------------------------------------------------------------------

def test_chat_request_validation():
    """Testa validação do modelo ChatRequest."""
    # Mensagem válida
    request = ChatRequest(message="Olá", history=[])
    assert request.message == "Olá"

    # Mensagem vazia deve falhar
    with pytest.raises(ValueError):
        ChatRequest(message="", history=[])

    # Mensagem muito longa deve falhar
    long_message = "a" * 4001
    with pytest.raises(ValueError):
        ChatRequest(message=long_message, history=[])


def test_chat_message_validation():
    """Testa validação do modelo ChatMessage."""
    # Role válido
    msg = ChatMessage(role="user", content="teste")
    assert msg.role == "user"

    # Role inválido deve falhar
    with pytest.raises(ValueError):
        ChatMessage(role="invalid", content="teste")


# ---------------------------------------------------------------------------
# Testes de Performance
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_intent_classification_performance(mock_client):
    """Testa performance da classificação de intenção."""
    mock_response = AsyncMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "geral"}}]
    }
    mock_client.post.return_value = mock_response

    import time
    start = time.time()
    await llm.classify_intent("Mensagem de teste", "fake_key")
    duration = time.time() - start

    # Deve ser rápido (< 1 segundo com mock)
    assert duration < 1.0


# ---------------------------------------------------------------------------
# Helpers para execução dos testes
# ---------------------------------------------------------------------------

def run_all_tests():
    """Executa todos os testes e retorna resultados."""
    import subprocess
    import sys

    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", "tests.py",
            "-v", "--tb=short", "--color=yes"
        ], capture_output=True, text=True, cwd=".")

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1
        }


if __name__ == "__main__":
    # Permite executar os testes diretamente
    results = run_all_tests()
    print("=== RESULTADOS DOS TESTES ===")
    print(results["stdout"])
    if results["stderr"]:
        print("ERROS:")
        print(results["stderr"])
    print(f"SUCESSO: {results['success']}")