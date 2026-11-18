import os
import time

from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from retriever import get_vectorstore
from router import embeddings, classify_text, route_question
from config import settings

PATH_DB = settings.PATH_DB

prompt_template = """
Você é um tutor universitário de Circuitos Lógicos.

Sua função é explicar conceitos utilizando exclusivamente o CONTEXTO fornecido.

CONTEXTO:
{base_de_conhecimento}

PERGUNTA:
{pergunta}

REGRAS:
- Responda em português brasileiro, com tom didático de tutor universitário.
- Formate a resposta em Markdown. 
- Use MathJax para fórmulas e símbolos matemáticos. 
- Fórmulas matemáticas devem usar LaTeX entre `$...$`
- Use SOMENTE o CONTEXTO. Não utilize conhecimento externo.
- Não invente informações.
- Se a pergunta pedir N itens (ex.: "três operações"), cubra os N itens com base no CONTEXTO.
- Para cada item, priorize definição e comportamento quando estiverem no CONTEXTO.
- Não diga que falta informação se ela aparecer em qualquer trecho do CONTEXTO.
- Utilize terminologia técnica correta presente no CONTEXTO.
- Quando a pergunta tratar especificamente de uma porta lógica, organize a resposta nas seguintes seções:
  1. Introdução
  2. Simbologia e operação
  3. Tabela verdade
  4. Expressão booleana
  5. Aplicação prática
  6. Resumo

- Na seção "Simbologia e operação", utilize o marcador [[SIMBOLOGIA:NOME_DA_PORTA]], exemplo: [[SIMBOLOGIA:OR]].
- Se a resposta não estiver disponível no CONTEXTO, responda exatamente:
    "Não encontrei essa informação no material."

- Não repita a pergunta, estas regras nem o rótulo "RESPOSTA:".

RESPOSTA:
"""

MESSAGE_NOT_FOUND = "Não encontrei essa informação no material."

def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)

## conectando ao servidor vLLM

def _with_v1(url: str) -> str:
    """Normaliza a URL da API OpenAI para terminar em /v1."""
    url = (url or "").strip().rstrip("/")
    return url if url.endswith("/v1") else url + "/v1"


def extract_generation_stats(response, elapsed_s: float) -> dict:
    """Extrai tokens e velocidade da resposta LangChain."""
    meta = getattr(response, "response_metadata", None) or {}
    usage = meta.get("token_usage") or meta.get("usage") or {}
    completion = (
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or meta.get("eval_count")
    )
    prompt = usage.get("prompt_tokens") or usage.get("input_tokens")
    content = (getattr(response, "content", None) or "").strip()
    chars = len(content)
    if completion is None and chars:
        completion = max(1, chars // 4)
    tok_per_s = round(completion / elapsed_s, 2) if completion and elapsed_s > 0 else None
    return {
        "completion_tokens": completion,
        "prompt_tokens": prompt,
        "chars": chars,
        "tok_per_s": tok_per_s,
        "timed_out": False,
        "error": None,
    }


def create_llm(
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    timeout_s: float | None = None,
) -> ChatOpenAI:
    """Cria cliente LLM compatível com OpenAI Chat API (Ollama, vLLM, etc.)."""
    return ChatOpenAI(
        base_url=base_url or os.getenv("LLM_BASE_URL", settings.LLM_BASE_URL),
        api_key=api_key or os.getenv("LLM_API_KEY", settings.LLM_API_KEY),
        model=model or os.getenv("LLM_MODEL", settings.LLM_MODEL),
        temperature=temperature,
        max_tokens=max_tokens if max_tokens is not None else _env_int("LLM_MAX_TOKENS", settings.DEFAULT_LLM_MAX_TOKENS),
        timeout=timeout_s if timeout_s is not None else _env_float("LLM_TIMEOUT_SECONDS", settings.DEFAULT_LLM_TIMEOUT_S),
    )


llm = create_llm()


def _is_timeout_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return "timeout" in name or "timeout" in msg or "timed out" in msg


def query_api(question: str, llm_client: ChatOpenAI | None = None) -> dict:
    """Executa RAG: recuperação de contexto + geração de resposta."""
    client = llm_client or llm
    collection = ""
    texts_results: list[str] = []
    score = 0.0
    try:
        collection = route_question(question)
        database = get_vectorstore(collection)
        results = database.similarity_search_with_relevance_scores(question, k= settings.k_retrieval)

        texts_results = [result[0].page_content for result in results]
        base_de_conhecimento = "\n\n----\n\n".join(texts_results)
        score = results[0][1] if results else 0.0

        prompt = PromptTemplate.from_template(prompt_template)
        formatted_prompt = prompt.format(
            base_de_conhecimento=base_de_conhecimento,
            pergunta=question,
        )

        t0 = time.perf_counter()
        response = client.invoke([HumanMessage(content=formatted_prompt)])
        elapsed = time.perf_counter() - t0
        generation = extract_generation_stats(response, elapsed)

        return {
            "answer": response.content.strip(),
            "contexts": texts_results,
            "estagio": collection,
            "scores": score,
            "generation": generation,
        }
    except Exception as e:
        timed_out = _is_timeout_error(e)
        print(f"Erro ao consultar o sistema RAG: {e}")
        return {
            "answer": "Erro no processamento do RAG (timeout)." if timed_out else "Erro no processamento do RAG.",
            "contexts": texts_results,
            "estagio": collection,
            "scores": score,
            "generation": {
                "completion_tokens": None,
                "prompt_tokens": None,
                "chars": 0,
                "tok_per_s": None,
                "timed_out": timed_out,
                "error": str(e),
            },
        }


def quest(question: str):
    question = question or input("Escreva sua pergunta: ")

    collection = route_question(question)
    print("Collection escolhida:", collection)

    database = get_vectorstore(collection)
    results = database.similarity_search_with_relevance_scores(question, k= settings.k_results)

    if len(results) == 0 or results[0][1] < settings.t:
        if results:
            print("Resultados: ", results[0][0])
            print("Score: ", results[0][1])
        print("Não sei te dizer isso")
        return

    texts_results = [result[0].page_content for result in results]
    base_de_conhecimento = "\n\n----\n\n".join(texts_results)

    prompt = PromptTemplate.from_template(prompt_template)
    formatted_prompt = prompt.format(
        base_de_conhecimento=base_de_conhecimento,
        pergunta=question,
    )

    response = llm.invoke([HumanMessage(content=formatted_prompt)])
    print("Resposta da IA: ", response.content)
    return response.content.strip()


if __name__ == "__main__":
    quest()
