# Executar o E2 (vLLM)

O E2 compara os modelos publicados pelo servidor vLLM. O Codespaces executa o
cliente do benchmark, o RAG e a avaliação; a inferência dos modelos ocorre em
uma máquina/VM com GPU NVIDIA que hospeda o vLLM. Não tente carregar os pesos
BF16/INT8 no Codespaces.

## 1. Preparar o Codespaces

Na raiz do repositório:

```bash
python -m pip install --upgrade pip
pip install -r src/requirements.txt -r src/requirements-eval.txt
cp src/.env.example src/.env
```

Edite `src/.env` e informe `VLLM_BASE_URL`, `VLLM_API_KEY`,
`LLM_MODEL` e `OPENROUTER_API_KEY`. Não faça commit do arquivo `.env`.

O diretório persistente do ChromaDB deve estar disponível em
`src/DataBase` (ou ajuste `PERSIST_DIRECTORY` e `PATH_DB`). Se o banco
ainda não foi criado, execute o pipeline de ingestão a partir de `src`:

```bash
cd src
python store.py
cd ..
```

## 2. Validar o servidor vLLM

O servidor precisa estar acessível pelo Codespaces e expor a API OpenAI
compatível. O nome servido deve corresponder ao identificador configurado no
benchmark: `llama-int8` ou `mathstral-bf16`.

```bash
VLLM_ROOT_URL="${VLLM_BASE_URL%/v1}"
curl -fsS "$VLLM_ROOT_URL/v1/models" -H "Authorization: Bearer $VLLM_API_KEY"
```

O comando normaliza a URL e consulta sempre `/v1/models`. O nome publicado
precisa ser idêntico ao nome que o benchmark usa para o modelo selecionado.

## 3. Executar E2

```bash
cd src
python experiments/E2_model_benchmark.py --list
python experiments/E2_model_benchmark.py --models llama31_8b_int8_vllm
python experiments/E2_model_benchmark.py --models mathstral_7b_bf16_vllm
```

Para executar ambos em sequência:

```bash
python experiments/E2_model_benchmark.py
```

Para retomar uma execução interrompida, repita o mesmo comando. O benchmark
salva as respostas e checkpoints por modelo. Para recalcular somente RAGAS:

```bash
python experiments/E2_model_benchmark.py --ragas-only llama31_8b_int8_vllm
```

Use `--skip-ragas` para validar geração e recuperação sem consumir créditos
do juiz OpenRouter. Nesse caso, o checkpoint não terá métricas RAGAS.

## 4. Alternar o modelo

Antes de cada execução, ajuste `LLM_MODEL` para o nome servido correspondente:
`llama-int8` para Llama 3.1 8B W8A8 ou `mathstral-bf16` para Mathstral 7B
BF16. O script também seleciona esse nome automaticamente durante a geração
do modelo correspondente.

## Observações

- A chave de API do cliente deve ser igual à chave configurada com
  `--api-key` no vLLM.
- O benchmark mede latência e tokens/s no cliente; a VRAM remota só é registrada
  se `VLLM_VRAM_MB` for preenchida.
- O túnel precisa permanecer ativo durante toda a execução.
- Não use o servidor de desenvolvimento do Codespaces como substituto de uma
  GPU para inferência dos modelos.
