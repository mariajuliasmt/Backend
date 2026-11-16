"""Utilitários compartilhados pelo pipeline de benchmark E1/E2."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import psutil
import requests
from dotenv import load_dotenv

ENIAC_DIR = Path(__file__).resolve().parent
DATASET_PATH = ENIAC_DIR / "DataSet" / "dataset.json"
BENCHMARK_DIR = ENIAC_DIR / "ragas_result" / "benchmark"
OLLAMA_BASE = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Modelos servidos pelo vLLM (Colab, formatos nativos). "tag" = nome servido (--served-model-name).
MODELS = [
    {"id": "llama31_8b_int8_vllm",   "name": "Llama 3.1 8B W8A8 (vLLM)", "tag": "llama-int8",     "class": "8B", "backend": "vllm", "format": "int8-w8a8"},
    {"id": "mathstral_7b_bf16_vllm", "name": "Mathstral 7B bf16 (vLLM)", "tag": "mathstral-bf16", "class": "7B", "backend": "vllm", "format": "bf16"},
]

MANUAL_SCORE_COLUMNS = [
    "modelo",
    "velocidade_1_5",
    "alucinacao_1_5",
    "ptbr_1_5",
    "observacoes",
]


def load_env() -> None:
    load_dotenv(ENIAC_DIR / ".env")
    os.chdir(ENIAC_DIR)


def model_output_dir(model_id: str) -> Path:
    path = BENCHMARK_DIR / model_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_path(model_id: str) -> Path:
    return model_output_dir(model_id) / "checkpoint.json"


def is_model_completed(model_id: str) -> bool:
    cp = checkpoint_path(model_id)
    if not cp.exists():
        return False
    with open(cp, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("status") == "completed"


def rag_responses_path(model_id: str) -> Path:
    return model_output_dir(model_id) / f"{model_id}_rag_responses.json"


def load_rag_progress(model_id: str) -> list[dict]:
    path = rag_responses_path(model_id)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_partial_hardware(model_id: str) -> dict | None:
    cp = checkpoint_path(model_id)
    if cp.exists():
        with open(cp, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("hardware"):
            return data["hardware"]
    hw_path = model_output_dir(model_id) / f"{model_id}_hardware.json"
    if hw_path.exists():
        with open(hw_path, encoding="utf-8") as f:
            return json.load(f)
    return None


def get_benchmark_status(model_id: str, n_total: int = 45) -> str:
    if is_model_completed(model_id):
        return "concluido"
    n = len(load_rag_progress(model_id))
    if n > 0:
        return f"em andamento ({n}/{n_total})"
    cp = checkpoint_path(model_id)
    if cp.exists():
        with open(cp, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("status") == "in_progress":
            return f"em andamento ({data.get('n_completed', 0)}/{n_total})"
    return "pendente"


def save_rag_checkpoint(
    model: dict,
    results: list[dict],
    hardware: dict | None = None,
    *,
    n_total: int,
    completed: bool = False,
) -> None:
    """Salva progresso parcial após cada pergunta (permite retomar após interrupção)."""
    slug = model["id"]
    out_dir = model_output_dir(slug)

    with open(out_dir / f"{slug}_rag_responses.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if hardware:
        with open(out_dir / f"{slug}_hardware.json", "w", encoding="utf-8") as f:
            json.dump(hardware, f, indent=2, ensure_ascii=False)

    checkpoint: dict[str, Any] = {
        "status": "completed" if completed else "in_progress",
        "model": model,
        "n_completed": len(results),
        "n_total": n_total,
        "updated_at": datetime.now().isoformat(),
    }
    if hardware:
        checkpoint["hardware"] = hardware
    if completed:
        checkpoint["completed_at"] = datetime.now().isoformat()

    with open(checkpoint_path(slug), "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)


def load_dataset() -> pd.DataFrame:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    if not all(c in df.columns for c in ["question", "ground_truth"]):
        raise ValueError("dataset.json deve conter 'question' e 'ground_truth'.")
    return df


def check_ollama() -> dict[str, Any]:
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        r.raise_for_status()
        return {"ok": True, "models": [m["name"] for m in r.json().get("models", [])]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def ollama_pull(model_tag: str) -> None:
    """Baixa modelo via CLI ou API HTTP (fallback quando ollama não está no PATH)."""
    try:
        subprocess.run(["ollama", "pull", model_tag], check=True)
        return
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    print(f"Pull via API: {model_tag}")
    r = requests.post(
        f"{OLLAMA_BASE}/api/pull",
        json={"name": model_tag},
        stream=True,
        timeout=3600,
    )
    r.raise_for_status()
    for line in r.iter_lines():
        if not line:
            continue
        status = json.loads(line).get("status", "")
        if status:
            print(f"  {status}")


def ollama_stop() -> None:
    subprocess.run(["ollama", "stop"], check=False)


def snapshot_vram() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        return {
            "vram_used_mb": int(parts[0]),
            "vram_total_mb": int(parts[1]),
            "gpu_util_pct": int(parts[2]),
        }
    except Exception as e:
        return {"error": str(e)}


def snapshot_ram() -> dict[str, Any]:
    mem = psutil.virtual_memory()
    ollama_ram_mb = 0
    for proc in psutil.process_iter(["name", "memory_info"]):
        try:
            if proc.info["name"] and "ollama" in proc.info["name"].lower():
                ollama_ram_mb += proc.info["memory_info"].rss // (1024 * 1024)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return {
        "ram_used_gb": round(mem.used / (1024**3), 2),
        "ram_total_gb": round(mem.total / (1024**3), 2),
        "ram_pct": mem.percent,
        "ollama_ram_mb": ollama_ram_mb,
    }


def measure_ollama_tps(model_tag: str, prompt: str | None = None) -> dict[str, Any]:
    """Warm-up + medição de tokens/s via API Ollama."""
    prompt = prompt or "Explique em uma frase o que é uma porta lógica AND."
    payload = {
        "model": model_tag,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 128},
    }
    t0 = time.perf_counter()
    r = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=600)
    r.raise_for_status()
    wall_s = time.perf_counter() - t0
    data = r.json()
    eval_count = data.get("eval_count", 0)
    eval_duration_ns = data.get("eval_duration", 0)
    tok_s_api = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns else 0.0
    tok_s_wall = eval_count / wall_s if wall_s else 0.0
    return {
        "eval_count": eval_count,
        "eval_duration_ns": eval_duration_ns,
        "tokens_per_second_api": round(tok_s_api, 2),
        "tokens_per_second_wall": round(tok_s_wall, 2),
        "wall_seconds": round(wall_s, 2),
        "load_duration_ns": data.get("load_duration"),
        "total_duration_ns": data.get("total_duration"),
    }


# ---------------------------------------------------------------------------
# Backend vLLM (API compativel com OpenAI). Modelos com "backend": "vllm" usam
# VLLM_BASE_URL / VLLM_API_KEY (variaveis de ambiente ou .env). Os modelos do
# Ollama continuam exatamente como antes.
# ---------------------------------------------------------------------------
_SAVED_LLM_ENV: dict[str, str | None] | None = None


def _apply_backend_env(model: dict) -> None:
    """Aponta LLM_BASE_URL/LLM_API_KEY para o vLLM (ou restaura os valores anteriores)."""
    global _SAVED_LLM_ENV
    keys = ("LLM_BASE_URL", "LLM_API_KEY")
    if model.get("backend") == "vllm":
        base = os.getenv("VLLM_BASE_URL")
        if not base:
            raise ValueError(
                "VLLM_BASE_URL não definida (URL do túnel do Colab, ex.: https://xxxx.trycloudflare.com)."
            )
        if _SAVED_LLM_ENV is None:
            _SAVED_LLM_ENV = {k: os.environ.get(k) for k in keys}
        base = base.rstrip("/")
        os.environ["LLM_BASE_URL"] = base if base.endswith("/v1") else base + "/v1"
        os.environ["LLM_API_KEY"] = os.getenv("VLLM_API_KEY", "")
    elif _SAVED_LLM_ENV is not None:
        for k, v in _SAVED_LLM_ENV.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _SAVED_LLM_ENV = None


def snapshot_vram_remote() -> dict[str, Any]:
    """VRAM do servidor vLLM (Colab): informe com VLLM_VRAM_MB (valor 'vram_mib' do bench_*.json)."""
    raw = os.getenv("VLLM_VRAM_MB")
    if raw:
        return {"vram_used_mb": int(float(raw)), "fonte": "colab (VLLM_VRAM_MB)"}
    return {"nota": "VRAM do vLLM nao informada (defina VLLM_VRAM_MB)"}


def measure_openai_tps(
    base_url: str, api_key: str, model: str, prompt: str | None = None, max_tokens: int = 128
) -> dict[str, Any]:
    """Medição de tok/s via API OpenAI-compatível (vLLM), com as mesmas chaves de measure_ollama_tps().

    tokens_per_second_api = tokens / tempo após o 1º token (como o eval_duration do Ollama,
    que exclui carga e leitura do prompt); tokens_per_second_wall = tokens / tempo total.
    """
    from openai import OpenAI

    prompt = prompt or "Explique em uma frase o que é uma porta lógica AND."
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=600)
    t0 = time.perf_counter()
    first = None
    n = 0
    stream = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
        messages=[{"role": "user", "content": prompt}],
    )
    for chunk in stream:
        if first is None and chunk.choices and chunk.choices[0].delta.content:
            first = time.perf_counter() - t0
        if chunk.usage:
            n = chunk.usage.completion_tokens
    wall_s = time.perf_counter() - t0
    gen_s = wall_s - first if first is not None and wall_s > first else wall_s
    return {
        "eval_count": n,
        "tokens_per_second_api": round(n / gen_s, 2) if gen_s else 0.0,
        "tokens_per_second_wall": round(n / wall_s, 2) if wall_s else 0.0,
        "wall_seconds": round(wall_s, 2),
        "ttft_seconds": round(first, 3) if first is not None else None,
    }


def _measure_tps(model: dict) -> dict[str, Any]:
    if model.get("backend") == "vllm":
        return measure_openai_tps(
            os.environ["LLM_BASE_URL"], os.environ.get("LLM_API_KEY", ""), model["tag"]
        )
    return measure_ollama_tps(model["tag"])


def _snap_vram(model: dict) -> dict[str, Any]:
    return snapshot_vram_remote() if model.get("backend") == "vllm" else snapshot_vram()


def _snap_ram(model: dict) -> dict[str, Any]:
    return {} if model.get("backend") == "vllm" else snapshot_ram()


def create_rag_llm(model_tag: str):
    from main import create_llm

    return create_llm(model=os.getenv("LLM_MODEL", model_tag))


def run_rag_batch(
    model_tag: str,
    df: pd.DataFrame,
    llm_client=None,
    *,
    model: dict | None = None,
    hardware: dict | None = None,
    resume: bool = True,
) -> list[dict]:
    from main import query_api

    os.environ["LLM_BASE_URL"] = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    os.environ["LLM_MODEL"] = model_tag
    if llm_client is None:
        llm_client = create_rag_llm(model_tag)

    n_total = len(df)
    results: list[dict] = []
    start_index = 0
    if model and resume:
        results = load_rag_progress(model["id"])
        start_index = len(results)
        if start_index >= n_total:
            print(f"  RAG já completo ({n_total}/{n_total}) — pulando geração.")
            return results
        if start_index > 0:
            print(f"  Retomando de {start_index + 1}/{n_total}...")

    slow_threshold = float(os.getenv("LLM_SLOW_THRESHOLD_S", "60"))

    for index, row in df.iloc[start_index:].iterrows():
        t0 = time.perf_counter()
        out = query_api(row["question"], llm_client=llm_client)
        elapsed = time.perf_counter() - t0
        gen = out.get("generation") or {}
        completion_tokens = gen.get("completion_tokens")
        tok_per_s = gen.get("tok_per_s")
        timed_out = bool(gen.get("timed_out"))
        chars = gen.get("chars") or len(out.get("answer", ""))

        results.append(
            {
                "question": row["question"],
                "ground_truth": row["ground_truth"],
                "answer": out.get("answer", ""),
                "contexts": out.get("contexts", []),
                "estagio": out.get("estagio", ""),
                "retrieval_score": out.get("scores", 0),
                "latency_seconds": round(elapsed, 2),
                "completion_tokens": completion_tokens,
                "prompt_tokens": gen.get("prompt_tokens"),
                "answer_chars": chars,
                "tok_per_s": tok_per_s,
                "timed_out": timed_out,
                "generation_error": gen.get("error"),
            }
        )

        tok_label = f"{completion_tokens} tok" if completion_tokens else f"{chars} chars"
        tps_label = f" | {tok_per_s:.1f} tok/s" if tok_per_s else ""
        flags = ""
        if timed_out:
            flags = " | TIMEOUT"
        elif elapsed >= slow_threshold:
            flags = " | AVISO:lento"
        print(
            f"  [{len(results)}/{n_total}] {elapsed:.1f}s | {tok_label}{tps_label}{flags}",
            flush=True,
        )
        if model:
            save_rag_checkpoint(model, results, hardware, n_total=n_total, completed=False)

    return results


def run_ragas(eval_rows: list[dict]) -> pd.DataFrame:
    from datasets import Dataset
    from langchain_huggingface import HuggingFaceEmbeddings
    from openai import AsyncOpenAI
    from ragas import RunConfig, evaluate
    from ragas.llms import llm_factory
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        answer_similarity,
        context_precision,
        context_recall,
        faithfulness,
    )

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY não definida.")

    judge_model = os.getenv("RAGAS_JUDGE_MODEL", "qwen/qwen-2.5-72b-instruct")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    ragas_llm = llm_factory(judge_model, client=client)
    ragas_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    ragas_dataset = Dataset.from_dict(
        {
            "question": [r["question"] for r in eval_rows],
            "answer": [r["answer"] for r in eval_rows],
            "contexts": [r["contexts"] for r in eval_rows],
            "ground_truth": [r["ground_truth"] for r in eval_rows],
        }
    )

    metrics = [
        faithfulness,
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_similarity,
    ]
    run_config = RunConfig(timeout=180, max_retries=3, max_workers=10)
    evaluation = evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
    )
    return evaluation.to_pandas()


def ragas_summary(ragas_df: pd.DataFrame) -> dict[str, float]:
    metric_cols = [
        "faithfulness",
        "answer_correctness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_similarity",
    ]
    summary = {}
    for col in metric_cols:
        if col in ragas_df.columns:
            summary[col] = round(float(ragas_df[col].mean()), 4)
            summary[f"{col}_std"] = round(float(ragas_df[col].std()), 4)
            summary[f"{col}_var"] = round(float(ragas_df[col].var()), 4)
    return summary


def save_model_results(
    model: dict,
    hardware: dict,
    rag_results: list[dict],
    ragas_df: pd.DataFrame | None,
    manual_scores: dict | None = None,
) -> Path:
    out_dir = model_output_dir(model["id"])
    slug = model["id"]

    with open(out_dir / f"{slug}_rag_responses.json", "w", encoding="utf-8") as f:
        json.dump(rag_results, f, indent=2, ensure_ascii=False)

    with open(out_dir / f"{slug}_hardware.json", "w", encoding="utf-8") as f:
        json.dump(hardware, f, indent=2, ensure_ascii=False)

    if ragas_df is not None:
        ragas_df.to_csv(out_dir / f"{slug}_ragas_metrics.csv", index=False)
        summary = ragas_summary(ragas_df)
        with open(out_dir / f"{slug}_ragas_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    if manual_scores:
        with open(out_dir / f"{slug}_manual_scores.json", "w", encoding="utf-8") as f:
            json.dump(manual_scores, f, indent=2, ensure_ascii=False)

    checkpoint = {
        "status": "completed",
        "model": model,
        "completed_at": datetime.now().isoformat(),
        "hardware": hardware,
        "ragas_summary": ragas_summary(ragas_df) if ragas_df is not None else None,
    }
    with open(checkpoint_path(model["id"]), "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)

    return out_dir


def build_manual_scores_template() -> pd.DataFrame:
    rows = [{"modelo": m["name"], **{c: "" for c in MANUAL_SCORE_COLUMNS if c != "modelo"}} for m in MODELS]
    return pd.DataFrame(rows)


def consolidate_results() -> pd.DataFrame:
    rows = []
    for model in MODELS:
        cp = checkpoint_path(model["id"])
        if not cp.exists():
            continue
        with open(cp, "r", encoding="utf-8") as f:
            data = json.load(f)
        hw = data.get("hardware", {})
        tps = hw.get("tps", {})
        ragas = data.get("ragas_summary") or {}
        manual_path = model_output_dir(model["id"]) / f"{model['id']}_manual_scores.json"
        manual = {}
        if manual_path.exists():
            with open(manual_path, "r", encoding="utf-8") as f:
                manual = json.load(f)
        rows.append(
            {
                "modelo": model["name"],
                "tag": model["tag"],
                "classe": model["class"],
                "vram_peak_mb": hw.get("vram_after_load", {}).get("vram_used_mb"),
                "ram_peak_gb": hw.get("ram_during_gen", {}).get("ram_used_gb"),
                "tok_s_medio": tps.get("tokens_per_second_api"),
                "faithfulness": ragas.get("faithfulness"),
                "answer_correctness": ragas.get("answer_correctness"),
                "velocidade_1_5": manual.get("velocidade_1_5"),
                "alucinacao_1_5": manual.get("alucinacao_1_5"),
                "ptbr_1_5": manual.get("ptbr_1_5"),
            }
        )
    return pd.DataFrame(rows)


def run_single_model_benchmark(
    model: dict,
    df: pd.DataFrame,
    *,
    skip_pull: bool = False,
    skip_ragas: bool = False,
    warmup_only: bool = False,
) -> dict[str, Any]:
    """Executa benchmark completo de um modelo."""
    load_env()
    _apply_backend_env(model)
    print(f"\n{'=' * 60}\nModelo: {model['name']} ({model['tag']})\n{'=' * 60}")

    if is_model_completed(model["id"]):
        print("Checkpoint encontrado — pulando (delete checkpoint.json para reexecutar).")
        with open(checkpoint_path(model["id"]), "r", encoding="utf-8") as f:
            return json.load(f)

    partial = load_rag_progress(model["id"])
    resuming = len(partial) > 0

    if resuming:
        print(f"Progresso parcial encontrado: {len(partial)}/{len(df)} perguntas.")
        hardware = load_partial_hardware(model["id"])
        if hardware is None:
            hardware = {
                "model": model,
                "started_at": datetime.now().isoformat(),
                "vram_idle": _snap_vram(model),
                "ram_idle": _snap_ram(model),
            }
    else:
        hardware = {
            "model": model,
            "started_at": datetime.now().isoformat(),
            "vram_idle": _snap_vram(model),
            "ram_idle": _snap_ram(model),
        }

        if not skip_pull and model.get("backend") != "vllm":
            print("Pull do modelo...")
            ollama_pull(model["tag"])

        print("Warm-up (descartado)...")
        _measure_tps(model)

        hardware["vram_after_load"] = _snap_vram(model)
        hardware["ram_after_load"] = _snap_ram(model)

        print("Medição tok/s oficial...")
        hardware["tps"] = _measure_tps(model)
        hardware["vram_during_gen"] = _snap_vram(model)
        hardware["ram_during_gen"] = _snap_ram(model)

    if warmup_only:
        return hardware

    label = "retomando" if resuming else f"{len(df)} perguntas"
    print(f"RAG batch ({label})...")
    rag_results = run_rag_batch(
        model["tag"], df, model=model, hardware=hardware, resume=True
    )

    ragas_df = None
    if not skip_ragas:
        print("RAGAS (judge OpenRouter)...")
        ragas_df = run_ragas(rag_results)

    save_model_results(model, hardware, rag_results, ragas_df)
    print(f"Concluído: {model_output_dir(model['id'])}")
    return {"model": model, "hardware": hardware, "n_responses": len(rag_results)}