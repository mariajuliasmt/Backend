# Executar o E2: vLLM no WSL/Ubuntu + benchmark no Google Colab

## Arquitetura

- **WSL2/Ubuntu:** hospeda Docker Compose e o servidor vLLM. É o único ambiente que
  carrega os pesos dos modelos e precisa de GPU NVIDIA compatível com WSL2,
  driver NVIDIA no Windows e NVIDIA Container Toolkit no Ubuntu.
- **Google Colab:** executa o benchmark E2 como cliente HTTP. Não carrega os
  modelos de geração; acessa a API OpenAI-compatível exposta pelo vLLM.
- **Túnel:** fornece ao Colab uma URL HTTPS pública temporária para a porta 8000
  do WSL. Mantenha-o ativo durante toda a execução.

## 1. Preparar o servidor no WSL/Ubuntu

Na raiz do repositório:

```bash
cd src/vllm
cp env.example .env
```

Edite `.env`:
- `VLLM_MODEL`: identificador correto do modelo no Hugging Face ou caminho local.
- `VLLM_SERVED_NAME`: `llama-int8` ou `mathstral-bf16`.
- `VLLM_API_KEY`: chave forte que também será usada pelo Colab.
- `VLLM_VERSION`: fixe uma versão compatível do vLLM em vez de depender de
  `latest`, depois de validar a imagem com os modelos escolhidos.
- `HF_TOKEN`: só é necessário para modelos que exigem autenticação.

Verifique GPU e Docker:

```bash
nvidia-smi
docker --version
docker compose version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

O último comando valida o acesso do Docker à GPU. Se falhar, resolva primeiro
driver/WSL/NVIDIA Container Toolkit; não tente iniciar o vLLM ainda.

Valide a configuração e suba o servidor:

```bash
docker compose --profile gpu config
docker compose --profile gpu up -d
docker compose logs -f vllm
```

Em outro terminal WSL, teste a API local:

```bash
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8000/v1/models \
  -H "Authorization: Bearer $VLLM_API_KEY"
```

O ID retornado em `/v1/models` precisa ser igual a `VLLM_SERVED_NAME`.
Não exponha diretamente a porta 8000 à internet sem autenticação.

## 2. Expor a API ao Colab

O Colab precisa alcançar o servidor WSL por HTTPS. Use um túnel autenticado
(Cloudflare Tunnel ou alternativa equivalente) apontando para
`http://localhost:8000`. A URL pública deve encaminhar para o serviço local,
não para o próprio Colab.

Teste a partir do Colab, não apenas no WSL:

```bash
curl -fsS "$VLLM_BASE_URL/models" \
  -H "Authorization: Bearer $VLLM_API_KEY"
```

Aqui `VLLM_BASE_URL` deve terminar em `/v1`, por exemplo
`https://seu-host-publico.example/v1`. Nunca publique a chave em células
visíveis, outputs, GitHub ou logs.

## 3. Preparar o Google Colab

Abra um notebook Colab com runtime Python. Clone a branch do PR (ou a branch
que vier a ser integrada):

```python
!git clone --branch fix/e2-vllm-execution https://github.com/mariajuliasmt/Backend.git
%cd Backend
!pip install -r src/requirements.txt -r src/requirements-eval.txt
```

Crie `src/.env` sem gravar os segredos no notebook. Uma forma simples é
solicitar os valores durante a execução:

```python
import getpass
from pathlib import Path

env_path = Path("src/.env")
base_url = input("URL HTTPS do vLLM (inclua /v1): ").strip().rstrip("/")
api_key = getpass.getpass("Chave da API vLLM: ")
openrouter_key = getpass.getpass("Chave OpenRouter (RAGAS): ")

env_path.write_text(
    f"VLLM_BASE_URL={base_url}\n"
    f"VLLM_API_KEY={api_key}\n"
    "LLM_MODEL=llama-int8\n"
    f"LLM_BASE_URL={base_url}\n"
    f"LLM_API_KEY={api_key}\n"
    "LLM_TIMEOUT_SECONDS=180\n"
    "LLM_MAX_TOKENS=350\n"
    "LLM_SLOW_THRESHOLD_S=60\n"
    "EMBEDDING_MODEL=BAAI/bge-m3\n"
    "WEIGHT_EMBEDDING=0.80\nWEIGHT_KEYWORDS=0.20\n"
    "DUPLICATION_THRESHOLD=0.05\nBASE_FOLDER=base\n"
    "WINDOW_SIZE=3\nchunk_SIZE=500\noverlap=75\n"
    "PERSIST_DIRECTORY=DataBase\nPATH_DB=DataBase\n"
    "temperature=0.0\nDEFAULT_LLM_TIMEOUT_S=180\n"
    "DEFAULT_LLM_MAX_TOKENS=350\nDEFAULT_LLM_SLOW_THRESHOLD_S=60\n"
    "k_retrieval=3\nk_results=4\nt=0.5\n"
    f"OPENROUTER_API_KEY={openrouter_key}\n"
    "RAGAS_JUDGE_MODEL=qwen/qwen-2.5-72b-instruct\n"
    "OPENROUTER_BASE_URL=https://openrouter.ai/api/v1\n",
    encoding="utf-8",
)
```

O dataset está em `src/DataSet/dataset.json`. O banco vetorial precisa existir
no runtime Colab em `src/DataBase`. Se não estiver armazenado em um Drive
persistente, gere-o no Colab a partir dos PDFs versionados:

```python
%cd /content/Backend/src
!python store.py
```

A ingestão pode demorar e baixar o modelo de embeddings. Para execuções
repetidas, mantenha `src/DataBase` em armazenamento persistente (por exemplo,
Google Drive) e restaure-o antes do benchmark. Não confunda o armazenamento
do ChromaDB com os pesos do LLM: os pesos permanecem no servidor WSL.

## 4. Validar conexão e executar E2

No Colab:

```python
import os, requests
from dotenv import load_dotenv

load_dotenv("src/.env", override=True)
base = os.environ["VLLM_BASE_URL"].rstrip("/")
key = os.environ["VLLM_API_KEY"]
r = requests.get(base + "/models", headers={"Authorization": f"Bearer {key}"}, timeout=20)
r.raise_for_status()
print(r.json())
```

Confirme que o ID do modelo servido corresponde ao modelo selecionado. Execute
cada modelo separadamente para controlar a comparação:

```python
%cd /content/Backend/src
!python experiments/E2_model_benchmark.py --list
!python experiments/E2_model_benchmark.py --models llama31_8b_int8_vllm --skip-ragas
!python experiments/E2_model_benchmark.py --models mathstral_7b_bf16_vllm --skip-ragas
```

Depois de validar as respostas, execute RAGAS para cada modelo (usa créditos
OpenRouter):

```python
!python experiments/E2_model_benchmark.py --ragas-only llama31_8b_int8_vllm
!python experiments/E2_model_benchmark.py --ragas-only mathstral_7b_bf16_vllm
```

Ou execute a avaliação RAGAS junto com a geração, removendo `--skip-ragas`.
Para retomar uma execução interrompida, repita o mesmo comando; os checkpoints
são gravados em `src/ragas_result/benchmark/`.

## 5. Regras para comparar os resultados

- Mantenha o mesmo dataset, banco ChromaDB, embeddings, parâmetros de geração,
  número de perguntas e juiz RAGAS nos dois modelos.
- Aguarde o aquecimento do servidor antes da primeira medição e evite outras
  cargas concorrentes na GPU.
- Registre separadamente a GPU/VRAM do servidor WSL. O Colab não consegue medir
  a VRAM remota com `nvidia-smi`; `VLLM_VRAM_MB` não deve ser preenchida com
  a VRAM da GPU do Colab.
- A medição de latência feita no Colab inclui a rede/túnel. Registre essa
  condição e não a compare diretamente com latência medida localmente no WSL.
- Para uma comparação experimental controlada, use a mesma região/rota de rede,
  mantenha o túnel estável e execute os modelos em condições equivalentes.
- Não faça commit de `.env`, chaves, checkpoints com dados sensíveis ou logs
  que possam conter segredos.
