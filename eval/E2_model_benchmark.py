"""
E2 — Comparação de modelos locais (RQ2).

Executa o benchmark completo para cada modelo 
Organiza saídas em ragas_result/experiments/e2/<model_id>/

Uso (PowerShell — comando em UMA linha, sem barra invertida):
  # antes: servidor vLLM no Colab + túnel; defina VLLM_BASE_URL, VLLM_API_KEY (e, opcional, VLLM_VRAM_MB)
  python experiments/e2_model_benchmark.py --list
  python experiments/e2_model_benchmark.py --models llama31_8b_int8_vllm
  python experiments/e2_model_benchmark.py --models mathstral_7b_bf16_vllm
  python experiments/e2_model_benchmark.py                      (os dois, em sequência)
  python experiments/e2_model_benchmark.py --ragas-only llama31_8b_int8_vllm

Retomar após interrupção: rode o mesmo comando — progresso salvo a cada pergunta.
Para recomeçar do zero: apague ragas_result/benchmark/<model_id>/checkpoint.json e *_rag_responses.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ENIAC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ENIAC_DIR))

from benchmark_utils import (
    MODELS,
    checkpoint_path,
    consolidate_results,
    get_benchmark_status,
    is_model_completed,
    load_dataset,
    load_env,
    model_output_dir,
    ragas_summary,
    run_ragas,
    run_single_model_benchmark,
    snapshot_ram,
    snapshot_vram,
)

import pandas as pd

E2_OUT = ENIAC_DIR / "ragas_result" / "experiments" / "e2"

# Modelos do E2: Llama 3.1 8B (int8) e Mathstral 7B (bf16), ambos servidos pelo vLLM.
MODEL_IDS = ["llama31_8b_int8_vllm", "mathstral_7b_bf16_vllm"]

COMPOSITE_WEIGHTS = {
    "faithfulness": 0.35,
    "answer_correctness": 0.35,
    "ptbr_1_5_norm": 0.20,  # escala 1–5 normalizada para 0–1
    "refusal_rate_inv": 0.10,  # 1 − taxa_recusa
}


def compute_composite_index(row: dict) -> float | None:
    """I = 0.35·faithfulness + 0.35·answer_correctness + 0.20·(ptbr/5) + 0.10·(1 − recusa)."""
    faith = row.get("faithfulness")
    corr = row.get("answer_correctness")
    ptbr = row.get("ptbr_1_5")
    refusal = row.get("refusal_rate")
    if faith is None or corr is None:
        return None
    ptbr_norm = (ptbr / 5.0) if ptbr is not None else 0.0
    refusal_inv = (1.0 - refusal) if refusal is not None else 1.0
    return round(
        0.35 * faith + 0.35 * corr + 0.20 * ptbr_norm + 0.10 * refusal_inv,
        4,
    )


def consolidate_e2() -> pd.DataFrame:
    """Consolida resultados E2 em DataFrame com métricas + hardware + índice composto."""
    metric_cols = [
        "faithfulness",
        "answer_correctness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_similarity",
    ]
    rows = []
    for model in MODELS:
        cp = checkpoint_path(model["id"])
        if not cp.exists():
            continue
        with open(cp, encoding="utf-8") as f:
            data = json.load(f)
        hw = data.get("hardware", {})
        tps = hw.get("tps", {})
        ragas = data.get("ragas_summary") or {}

        manual_path = model_output_dir(model["id"]) / f"{model['id']}_manual_scores.json"
        manual: dict = {}
        if manual_path.exists():
            with open(manual_path, encoding="utf-8") as f:
                manual = json.load(f)

        refusal_rate = _compute_refusal_rate(model["id"])

        row = {
            "model_id": model["id"],
            "modelo": model["name"],
            "classe": model["class"],
            "servidor": model.get("backend", "ollama"),
            "formato": model.get("format"),
            "tok_s_api": tps.get("tokens_per_second_api"),
            "tok_s_wall": tps.get("tokens_per_second_wall"),
            "vram_idle_mb": hw.get("vram_idle", {}).get("vram_used_mb"),
            "vram_peak_mb": hw.get("vram_after_load", {}).get("vram_used_mb"),
            "ram_peak_gb": hw.get("ram_during_gen", {}).get("ram_used_gb"),
            "refusal_rate": refusal_rate,
            "velocidade_1_5": manual.get("velocidade_1_5"),
            "alucinacao_1_5": manual.get("alucinacao_1_5"),
            "ptbr_1_5": manual.get("ptbr_1_5"),
        }
        for m in metric_cols:
            row[m] = ragas.get(m)

        row["composite_index"] = compute_composite_index(row)
        rows.append(row)

    df = pd.DataFrame(rows)
    E2_OUT.mkdir(parents=True, exist_ok=True)
    out_path = E2_OUT / "e2_model_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"E2 consolidado em: {out_path}")
    return df


def _compute_refusal_rate(model_id: str) -> float | None:
    """Calcula taxa de recusa a partir dos responses salvos."""
    resp_path = model_output_dir(model_id) / f"{model_id}_rag_responses.json"
    if not resp_path.exists():
        return None
    with open(resp_path, encoding="utf-8") as f:
        rows = json.load(f)
    refusals = sum(
        1 for r in rows if "não encontrei essa informação" in r.get("answer", "").lower()
    )
    return round(refusals / len(rows), 4) if rows else None


def run_ragas_only(model_id: str) -> None:
    """Re-executa apenas o RAGAS para um modelo já avaliado."""
    load_env()
    resp_path = model_output_dir(model_id) / f"{model_id}_rag_responses.json"
    if not resp_path.exists():
        raise FileNotFoundError(f"Responses não encontrados: {resp_path}")

    with open(resp_path, encoding="utf-8") as f:
        rows = json.load(f)

    model = next((m for m in MODELS if m["id"] == model_id), None)
    if model is None:
        raise ValueError(f"Modelo '{model_id}' não encontrado em MODELS.")

    print(f"RAGAS-only para {model['name']}...")
    ragas_df = run_ragas(rows)
    out_dir = model_output_dir(model_id)
    ragas_df.to_csv(out_dir / f"{model_id}_ragas_metrics.csv", index=False)
    summary = ragas_summary(ragas_df)
    with open(out_dir / f"{model_id}_ragas_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Atualiza checkpoint com summary
    cp = checkpoint_path(model_id)
    if cp.exists():
        with open(cp, encoding="utf-8") as f:
            data = json.load(f)
        data["ragas_summary"] = summary
        with open(cp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    print("RAGAS concluído:", summary)


def main(
    model_ids: list[str] | None = None,
    *,
    skip_pull: bool = False,
    skip_ragas: bool = False,
    ragas_only: str | None = None,
) -> pd.DataFrame:
    load_env()

    if ragas_only:
        run_ragas_only(ragas_only)
        return consolidate_e2()

    if model_ids:
        unknown = sorted(set(model_ids) - {m["id"] for m in MODELS})
        if unknown:
            raise ValueError(f"Modelo(s) desconhecido(s): {unknown}. Use --list.")

    df_dataset = load_dataset()
    targets = (
        [m for m in MODELS if m["id"] in model_ids]
        if model_ids
        else MODELS
    )

    for model in targets:
        run_single_model_benchmark(
            model,
            df_dataset,
            skip_pull=skip_pull,
            skip_ragas=skip_ragas,
        )

    return consolidate_e2()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="E2 — Comparação de modelos locais")
    parser.add_argument(
        "--models", nargs="+", metavar="MODEL_ID",
        help="IDs dos modelos a executar (ex: llama31_8b_int8_vllm mathstral_7b_bf16_vllm)",
    )
    parser.add_argument("--ragas-only", metavar="MODEL_ID",
                        help="Re-executa apenas RAGAS para o modelo especificado")
    parser.add_argument("--skip-pull", action="store_true")
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--list", action="store_true", help="Lista modelos disponíveis")
    args = parser.parse_args()

    if args.list:
        print("\nModelos disponíveis:")
        for m in MODELS:
            status = get_benchmark_status(m["id"])
            print(f"  {m['id']:25s} {m['name']:25s} [{m['class']}] — {status}")
        sys.exit(0)

    result = main(
        model_ids=args.models,
        skip_pull=args.skip_pull,
        skip_ragas=args.skip_ragas,
        ragas_only=args.ragas_only,
    )
    print("\nResultado E2:")
    print(result.to_string())
