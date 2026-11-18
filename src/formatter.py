from pathlib import Path
import re
import base64


# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"


# ============================================================
# LOCALIZAÇÃO DA IMAGEM
# ============================================================

def find_symbol_image(porta: str) -> Path | None:
    """
    Localiza a imagem PNG da porta lógica.
    """

    porta = porta.upper().strip()

    image_path = IMAGES_DIR / f"{porta}.png"

    if image_path.exists():
        return image_path

    return None


# ============================================================
# CONVERSÃO DA IMAGEM PARA BASE64
# ============================================================

def image_to_base64(image_path: Path) -> str:   #Lê uma imagem PNG e converte seu conteúdo para Base64.

    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read())

    return encoded.decode("ascii")


# ============================================================
# FORMATAÇÃO DA IMAGEM
# ============================================================

def format_symbol_image(porta: str) -> str: #Localiza a imagem da porta lógica, converte para Base64 e retorna a imagem no formato Markdown
    

    porta = porta.upper().strip()

    image_path = find_symbol_image(porta)

    if image_path is None:
        return f"**Imagem da porta {porta} não encontrada.**"

    image_base64 = image_to_base64(image_path)

    return (
        f"![Porta Lógica {porta}]"
        f"(data:image/png;base64,{image_base64})"
    )


# ============================================================
# FORMATAÇÃO DA RESPOSTA
# ============================================================

def format_response(resposta: str) -> str:

    if not resposta:
        return ""

    # Normaliza possíveis quebras de linha escapadas
    resposta = resposta.replace("\\n", "\n").strip()

    # Aceita:
    #
    # [[SIMBOLOGIA:OR]]
    # [[SIMBOLOGIA:Porta OR]]
    # [[ simbologia : or ]]
    # [[ simbologia : Porta OR ]]
    #
    pattern = (
        r"\[\[\s*SIMBOLOGIA\s*:\s*"
        r"(?:PORTA\s+)?"
        r"(AND|OR|XOR|XNOR|NOT|NAND|NOR)"
        r"\s*\]\]"
    )

    def replace_symbol(match):
        porta = match.group(1).upper()

        return format_symbol_image(porta)

    resposta = re.sub(
        pattern,
        replace_symbol,
        resposta,
        flags=re.IGNORECASE
    )

    return resposta
