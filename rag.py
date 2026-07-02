"""
rag.py — Camada de Recuperação de Conhecimento (RAG).

Hierarquia de fontes:
  1. ChromaDB com embeddings semânticos (PDFs/textos carregados em /knowledge)
  2. Fallback para base hardcoded (garante funcionamento mesmo sem arquivos externos)

Uso:
  from rag import retrieve
  context, source_label = await retrieve(intent, query)
"""

import os
import pathlib
import asyncio
from typing import Optional

# ---------------------------------------------------------------------------
# Base de conhecimento hardcoded (fallback garantido)
# ---------------------------------------------------------------------------

FALLBACK_DB: dict[str, tuple[str, str]] = {
    "regimento": (
        """REGIMENTO GERAL DO CEUB 2025 — RESUMO:
• Art. 3º: Autonomia acadêmica e administrativa.
• Frequência mínima obrigatória: 75%. Abaixo disso = reprovação direta por falta.
• Sistema de Menções:
  SS (Superior) — Excelente
  MS (Médio Superior) — Bom
  MM (Médio) — Regular → Aprovado
  MI (Médio Inferior) — Abaixo do esperado → Reprovado (contexto-dependente)
  II (Inferior) — Reprovado
  SR (Sem Rendimento) — Reprovado por ausência total
• Aprovação exige menções SS, MS ou MM.
• Aproveitamento de Estudos: mínimo 70% de equivalência de conteúdo.""",
        "📋 Regimento Geral 2025",
    ),
    "faq": (
        """FAQ — CEUB:
• Acesso Espaço Aluno: ea.uniceub.br (RA + senha cadastrada).
• Carteirinha: Digital pelo App Espaço Aluno (disponível iOS e Android).
• Passe Estudantil: Solicitar declaração na Central de Atendimento presencialmente.
• Boleto / financeiro: Disponível na aba "Financeiro" do Espaço Aluno.
• TCC: Obrigatório apenas se constar no Projeto Pedagógico do curso.
• Trancamento de matrícula: Solicitar dentro do prazo no Espaço Aluno.""",
        "⚡ FAQ — Dúvidas Frequentes",
    ),
    "guia": (
        """GUIA DO ESTUDANTE CEUB 2024:
• Campi:
  - Asa Norte: EQN 707/907 — campus principal
  - Taguatinga: QS 1
• Estacionamento: Gratuito no campus Asa Norte mediante apresentação da carteirinha.
• Plataformas:
  - Espaço Aluno (ea.uniceub.br): financeiro e acadêmico
  - Sala Online: aulas EAD e materiais
  - InfoCentral: chamados de TI
• Contatos de coordenadores:
  - Ciência da Computação: fernando.guimaraes@ceub.edu.br
  - Direito: dulce.oliveira@ceub.edu.br""",
        "📖 Guia do Estudante 2024",
    ),
    "geral": (
        """INFORMAÇÕES GERAIS — CEUB:
O UniCEUB é um centro universitário em Brasília com campi na Asa Norte e Taguatinga.
Para dúvidas não cobertas aqui, acesse ea.uniceub.br ou dirija-se à Central de Atendimento.""",
        "ℹ️ Informações Gerais",
    ),
}


# ---------------------------------------------------------------------------
# ChromaDB (opcional — ativado se a biblioteca estiver instalada)
# ---------------------------------------------------------------------------

_chroma_collection = None
_embedder = None
CHROMA_DB_PATH = pathlib.Path(os.getenv("CHROMA_DB_PATH", "chroma_db"))
KNOWLEDGE_DIR = pathlib.Path(os.getenv("KNOWLEDGE_DIR", "knowledge"))  # pasta com PDFs/TXTs do CEUB


def _try_init_chroma() -> bool:
    """Tenta inicializar ChromaDB + SentenceTransformer. Retorna True se OK."""
    global _chroma_collection, _embedder
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer

        CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        _chroma_collection = client.get_or_create_collection(
            name="ceub_knowledge",
            metadata={"hnsw:space": "cosine"},
        )

        _embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        # Indexa documentos da pasta /knowledge se ainda não indexados
        _index_knowledge_files()
        return True
    except ImportError:
        print("[RAG] ChromaDB ou SentenceTransformer não instalados — usando fallback hardcoded.")
        return False
    except Exception as e:
        print(f"[RAG] Falha ao inicializar ChromaDB: {e} — usando fallback hardcoded.")
        return False


def _index_knowledge_files() -> None:
    """Lê arquivos .txt e .pdf de /knowledge e os indexa no ChromaDB."""
    if not KNOWLEDGE_DIR.exists():
        return

    existing_ids: set[str] = set(_chroma_collection.get()["ids"])

    for file_path in KNOWLEDGE_DIR.iterdir():
        doc_id = file_path.name
        if any(id_.startswith(f"{doc_id}::") for id_ in existing_ids):
            continue  # já indexado

        text = _read_file(file_path)
        if not text:
            continue

        chunks = _chunk_text(text, max_chars=800)
        ids = [f"{doc_id}::chunk_{i}" for i in range(len(chunks))]
        embeddings = _embedder.encode(chunks).tolist()

        _chroma_collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{"source": doc_id} for _ in chunks],
        )
        print(f"[RAG] Indexado: {doc_id} ({len(chunks)} chunks)")


def _read_file(path: pathlib.Path) -> Optional[str]:
    if path.suffix == ".txt":
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return None
    if path.suffix == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            return None
    return None


def _chunk_text(text: str, max_chars: int = 800) -> list[str]:
    """Divide o texto em chunks por parágrafos, respeitando o limite de chars."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) > max_chars and current:
            chunks.append(current.strip())
            current = para
        else:
            current = current + "\n" + para if current else para
    if current:
        chunks.append(current.strip())
    return chunks or [text[:max_chars]]


def _chroma_search(query: str, top_k: int = 4) -> Optional[str]:
    """Busca semântica no ChromaDB. Retorna contexto concatenado ou None."""
    if _chroma_collection is None or _embedder is None:
        return None
    try:
        query_embedding = _embedder.encode([query]).tolist()
        results = _chroma_collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, _chroma_collection.count()),
        )
        docs = results.get("documents", [[]])[0]
        if not docs:
            return None
        return "\n\n---\n\n".join(docs)
    except Exception as e:
        print(f"[RAG] Erro na busca ChromaDB: {e}")
        return None


# ---------------------------------------------------------------------------
# Inicialização
# ---------------------------------------------------------------------------

_chroma_ready = _try_init_chroma()


def retrieve(intent: str, query: str) -> tuple[str, str]:
    """
    Retorna (context, source_label) para a intenção e query fornecidas.
    Tenta ChromaDB primeiro; cai no fallback hardcoded se necessário.
    """
    # 1. Tenta ChromaDB semântico
    if _chroma_ready:
        semantic_context = _chroma_search(query)
        if semantic_context:
            return semantic_context, "🔍 Base de Conhecimento (Semântica)"

    # 2. Fallback hardcoded por categoria
    context, label = FALLBACK_DB.get(intent, FALLBACK_DB["geral"])
    return context, label