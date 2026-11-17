import os
import json
import urllib.request

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.pipeline import quest
from src.formatter import format_response
from src.config import settings

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Requisicao(BaseModel):
    type: str
    message: str
    source: str
    timestamp: str


@app.post("/TutorIA")
def tutoria(dados: Requisicao):

    question = dados.message

    resposta = quest(question)
    resposta_formatada = format_response(resposta)

    return {
        "resposta": resposta_formatada
    }
## health check

def _llm_root_url() -> str:
  """URL do servidor vLLM sem o /v1, o /health fica na raiz, fora do /v1."""
  url = settings.LLM_BASE_URL.strip().rstrip("/")
  if url.endswith("/v1"):
    return url[:-3]
  else:
    return url

@app.get("/health")
def health_check():
    health_status = {
        "status": "ok",
        "llm": "desconhecido",
        "chromadb": "desconhecido"
    }

    # 1. Checando o servidor
    try:
        ## sem url = settings.LLM_BASE_URL, que era a key do Ollama
      
        resposta = _http_get(f"{_llm_root_url()}/health")
      
        if resposta.getcode() == 200:
            health_status["llm"] = "ok"
        else:
            health_status["status"] = "erro"
            health_status["llm"] = f"erro - http {resposta.getcode()}"
            
    except Exception as e:
        health_status["status"] = "erro"
        health_status["llm"] = "offline - verifique se o servidor do Ollama está rodando"

    # 2. Checando o ChromaDB
    try:
        if os.path.exists(settings.PATH_DB):
            health_status["chromadb"] = "ok"
        else:
            health_status["status"] = "erro"
            health_status["chromadb"] = f"erro - pasta '{settings.PATH_DB}' não encontrada"
            
    except Exception as e:
        health_status["status"] = "erro"
        health_status["chromadb"] = "erro de leitura local"

    return health_status


  # 3. Checando se o alias está sendo servido

if health_status["llm"] == "ok":
  try:
    resposta = _http_get(f"{_llm_root_url()}/v1/models",
                         headers={"Authorization": f"Bearer {settings.LLM_API_KEY"})
    servidos = [m["id"] for m in json.loads(resposta.read()).get("data", [])]
    if settings.LLM_MODEL not int servidos:
    health_status["status"] = "erro"
    health_status["llm"] = f"erro - modelo '{settings.LLM_MODEL}' não é servido. São servidos: {servidos})"
except Exception:
health_status["status"] = "erro"
health_status["llm"] = "erro - servidor no ar, porém /v1/models recusou. Verifique a chave da API." 
