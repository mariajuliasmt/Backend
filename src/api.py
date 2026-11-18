import json
import urllib.request

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import settings
from formatter import format_response
from pipeline import quest

app = FastAPI(title="TutorIA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Requisicao(BaseModel):
    type: str = "message"
    message: str
    source: str = "unknown"
    timestamp: str = ""


def _llm_root_url() -> str:
    url = settings.LLM_BASE_URL.strip().rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


def _http_get(url: str, headers: dict | None = None):
    request = urllib.request.Request(url, headers=headers or {})
    return urllib.request.urlopen(request, timeout=5)


@app.post("/TutorIA")
def tutoria(dados: Requisicao):
    resposta = quest(dados.message)
    return {"resposta": format_response(resposta)}


@app.get("/health")
def health_check():
    status = {"status": "ok", "llm": "desconhecido", "chromadb": "desconhecido"}
    try:
        with _http_get(f"{_llm_root_url()}/health") as response:
            status["llm"] = "ok" if response.status == 200 else f"erro - http {response.status}"
        with _http_get(
            f"{_llm_root_url()}/v1/models",
            headers={"Authorization": f"Bearer {settings.LLM_API_KEY}"},
        ) as response:
            served = [m.get("id") for m in json.loads(response.read()).get("data", [])]
        if settings.LLM_MODEL not in served:
            status["llm"] = f"erro - modelo '{settings.LLM_MODEL}' não servido; disponíveis: {served}"
    except Exception as exc:
        status["status"] = "erro"
        status["llm"] = f"offline/indisponível: {exc}"

    import os
    if os.path.isdir(settings.PATH_DB):
        status["chromadb"] = "ok"
    else:
        status["status"] = "erro"
        status["chromadb"] = f"diretório não encontrado: {settings.PATH_DB}"
    return status
