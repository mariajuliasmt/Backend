"""
Troca o modelo usado pelo TutorIA (vLLM).
 
Uso (PowerShell ou bash, na raiz do projeto):
  python switch_models.py --list
  python switch_models.py --status
  python switch_models.py mathstral-bf16                      (só o .env do backend: LLM_MODEL)
  python switch_models.py mathstral-bf16 --docker             (também troca o modelo servido no Docker)
  python switch_models.py llama-int8 --url https://xxxx.trycloudflare.com   (aponta o backend para o túnel do Colab)
 
O alias é o nome servido pelo vLLM (--served-model-name) e precisa ser igual ao LLM_MODEL do backend.
O vLLM serve um modelo por vez: com --docker o container é recriado (leva alguns minutos).
Antes de alterar um .env, é gravada uma cópia em .env.bak.
"""
 
from __future__ import annotations
 
import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
 
# alias -> repositório no Hugging Face, contexto máximo e argumentos extras do vLLM
ALIASES = {
    "llama-int8": {
        "repo": "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8",
        "max_len": "8192",
        "extra": "",
        "descricao": "Llama 3.1 8B Instruct, pesos int8 (W8A8)",
    },
    "mathstral-bf16": {
        "repo": "mistralai/Mathstral-7B-v0.1",
        "max_len": "8192",
        "extra": "",  # se o vLLM exigir: "--tokenizer-mode mistral"
        "descricao": "Mathstral 7B, bf16 (sem quantização)",
    },
}
 
 
# ============ LEITURA E ESCRITA DO .env ============
 
def _linha(key: str) -> re.Pattern:
    """Aceita 'CHAVE=valor', 'CHAVE = valor' e 'CHAVE= valor'."""
    return re.compile(rf"^\s*{re.escape(key)}\s*=\s*(.*)$")
 
 
def get_var(env_path: Path, key: str) -> str | None:
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        m = _linha(key).match(line)
        if m:
            return m.group(1).split(" #")[0].strip()
    return None
 
 
def set_var(env_path: Path, key: str, value: str) -> None:
    text = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if _linha(key).match(line):
            lines[i] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
 
 
def backup(env_path: Path) -> None:
    if env_path.exists():
        shutil.copy(env_path, env_path.with_name(env_path.name + ".bak"))
 
 
def with_v1(url: str) -> str:
    url = url.strip().rstrip("/")
    return url if url.endswith("/v1") else url + "/v1"
 
 
# ============ STATUS ============
 
def served_models(base_url: str, api_key: str) -> list[str] | str:
    """Lista os modelos que o servidor serve agora (ou devolve a mensagem de erro)."""
    try:
        req = urllib.request.Request(
            f"{with_v1(base_url)}/models",
            headers={"Authorization": f"Bearer {api_key}", "User-Agent": "tutoria-switch"},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            return [m["id"] for m in json.loads(r.read()).get("data", [])]
    except Exception as e:
        return f"indisponível ({e})"
 
 
def show_status(env_path: Path, docker_env: Path) -> None:
    print(f"Backend ({env_path}):")
    print(f"  LLM_MODEL    = {get_var(env_path, 'LLM_MODEL')}")
    print(f"  LLM_BASE_URL = {get_var(env_path, 'LLM_BASE_URL')}")
    if docker_env.exists():
        print(f"Docker ({docker_env}):")
        print(f"  VLLM_SERVED_NAME = {get_var(docker_env, 'VLLM_SERVED_NAME')}")
        print(f"  VLLM_MODEL       = {get_var(docker_env, 'VLLM_MODEL')}")
    base, key = get_var(env_path, "LLM_BASE_URL"), get_var(env_path, "LLM_API_KEY") or ""
    if base:
        print(f"Servidor responde com: {served_models(base, key)}")
 
 
# ============ DOCKER ============
 
def recreate_docker(docker_dir: Path, timeout_min: int) -> None:
    if shutil.which("docker") is None:
        sys.exit("Docker não encontrado no PATH.")
    print(">> Recriando o container vLLM...")
    subprocess.run(
        ["docker", "compose", "--profile", "gpu", "up", "-d", "--force-recreate", "vllm"],
        cwd=docker_dir, check=True,
    )
    print(">> Aguardando o servidor (o download do modelo pode demorar)...")
    for _ in range(timeout_min * 6):
        out = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Health.Status}}", "vllm"],
            capture_output=True, text=True,
        ).stdout.strip()
        if out == "healthy":
            print(">> Pronto.")
            return
        time.sleep(10)
    sys.exit("Tempo esgotado. Veja: docker compose --profile gpu logs -f vllm")
 
 
# ============ MAIN ============
 
def main() -> None:
    p = argparse.ArgumentParser(description="Troca o modelo do TutorIA (vLLM)")
    p.add_argument("alias", nargs="?", help=f"um de: {', '.join(ALIASES)}")
    p.add_argument("--list", action="store_true", help="lista os aliases")
    p.add_argument("--status", action="store_true", help="mostra a configuração atual")
    p.add_argument("--url", help="define LLM_BASE_URL no backend (o /v1 é acrescentado se faltar)")
    p.add_argument("--docker", action="store_true", help="troca também o modelo servido no Docker")
    p.add_argument("--env", default=".env", help="arquivo .env do backend (padrão: .env)")
    p.add_argument("--docker-dir", default="vllm", help="pasta do docker-compose.yml (padrão: vllm)")
    p.add_argument("--timeout-min", type=int, default=20, help="espera máxima pelo container (min)")
    args = p.parse_args()
 
    env_path = Path(args.env)
    docker_dir = Path(args.docker_dir)
    docker_env = docker_dir / ".env"
 
    if args.docker and not args.alias:
        sys.exit("--docker precisa de um alias (ex.: python switch_models.py llama-int8 --docker).")
 
    if args.list or not (args.alias or args.status or args.url):
        print("Aliases disponíveis:")
        for name, a in ALIASES.items():
            print(f"  {name:16s} {a['descricao']}  ->  {a['repo']}")
        return
 
    if args.alias and args.alias not in ALIASES:
        sys.exit(f"Alias inválido: {args.alias}. Use --list.")
    if args.docker and not docker_env.exists():
        sys.exit(f"{docker_env} não existe. Copie {docker_dir}/.env.example para {docker_env} e preencha.")
 
    if args.alias or args.url:
        backup(env_path)
        if args.url:
            set_var(env_path, "LLM_BASE_URL", with_v1(args.url))
            print(f"Backend: LLM_BASE_URL={with_v1(args.url)}")
        if args.alias:
            set_var(env_path, "LLM_MODEL", args.alias)
            print(f"Backend: LLM_MODEL={args.alias}")
 
    if args.docker:
        a = ALIASES[args.alias]
        backup(docker_env)
        set_var(docker_env, "VLLM_MODEL", a["repo"])
        set_var(docker_env, "VLLM_SERVED_NAME", args.alias)
        set_var(docker_env, "LLM_MAX_LEN", a["max_len"])
        set_var(docker_env, "LLM_EXTRA_ARGS", a["extra"])
        print(f"Docker: VLLM_SERVED_NAME={args.alias} | VLLM_MODEL={a['repo']}")
        recreate_docker(docker_dir, args.timeout_min)
 
    if args.status:
        show_status(env_path, docker_env)
 
 
if __name__ == "__main__":
    main()
