from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from tutoria.rag.config import settings


# Descrição do Conteúdo de Cada Collections
STAGES_INFO = {
    "estagio_1": """
        Níveis lógicos: HIGH, LOW, HIGH-Z (Alto, baixo, alta impedância);
        Sistemas de Numeração; 
        Sistemas de Numeração: decimal, binário, octal e hexadecimal; 
        Conversão da representação do número para diferentes sistemas; 
        Operações Aritméticas; 
        Complemento de 2;
        Especificações de funcionamento de um circuito lógico e a sua codificação em termos de suas variáveis de estado;
        Descrição de uma função pela Tabela Verdade;
        Funções Lógicas: AND(multiplicação), NOT(negação), OR(soma);
        Portas Lógicas;
        Bloco Lógico;
        Portas Lógicas com HDL; 
        Introdução a Álgebra de Boole;
        Postulados, Propriedades e Teoremas;
        Simplificações de Expressões Lógicas;
        Projetos de Circuitos Lógicos;
        Projetos com duas variáveis;
        Projetos com três variáveis;
        Projetos com quatro variáveis
    """,

    "estagio_2": """
        Diagrama de Veitch-Karanaugh;
        Projetos de Codificadores e Decodificadores;
        Códigos: BCD, Excesso 3, Gray, etc; 
        Decodificador BCD/Display de Sete Segmentos;
        Codificador Binário/Decimal; 
        Decodificador Decimal/Binário;
        Codificador Binário/Gray; 
        Decodificador Gray/Binário; 
        Codificadores e Decodificadores com HDL; 
        Projetos de Multiplexadores e Demultiplexadores 
        Multiplexador 2x1 e Demultiplexador 1x2; 
        Multiplexador 4x1 e Demultiplexador 1x4; 
        Multiplexador 8x1 e Demultiplexador 1x8; 
        Projetos utilizando Multiplex como bloco básico; 
        Multiplexadores e Demultiplexadores com HDL; 
        Circuitos Lógicos Aritméticos;
        Circuito combinacional;
        Circuito sequencial;
        Meio Somador; 
        Somador Completo; 
        Meio Subtrator; 
        Subtrator Completo; 
        Somador/Subtrator Completo; 
        Circuitos Lógicos Aritméticos com HDL
    """,

    "estagio_3": """
        Circuitos Sequenciais 
        Flip-Flop básico; 
        Flip-Flop JK, tipo D e tipo T; 
        Registradores;
        Projetos de Contadores Assíncronos; 
        Projetos de Contadores Síncronos; 
        Circuitos Sequenciais com HDL; 
        Máquinas de Estado 
        Modelos de Sistemas Digitais;
        Exemplos de Máquinas de Estados
    """
}

KEYWORDS = {
    "estagio_1": [
        "HIGH",
        "LOW",
        "High-Z",
        "nível alto",
        "nível baixo",
        "alta impedância",
        "NÍVEL ALTO",
        "NÍVEL BAIXO",
        "ALTA IMPEDÂNCIA",
        "binário",
        "decimal",
        "octal",
        "hexadecimal",
        "complemento de 2",
        "álgebra de boole",
        "boole",
        "veitch",
        "tabela verdade",
        "porta lógica",
        "AND",
        "OR",
        "XOR",
        "XNOR",
        "NOT",
        "NOR",
        "NAND",
        "nand",
        "nor",
        "xor",
        "and",
        "or",
        "not"
    ],

    "estagio_2": [
        "somador",
        "meio somador",
        "mapa de karnaugh",
        "somador completo",
        "full adder",
        "subtrator",
        "meio subtrator",
        "subtrator completo",
        "multiplexador",
        "demultiplexador",
        "multiplexadores",
        "demultiplexadores",
        "MULTIPLEXADOR",
        "DEMULTIPLEXADOR",
        "MULTIPLEXADORES",
        "DEMULTIPLEXADORES",
        "mux",
        "demux",
        "MUX",
        "DEMUX",
        "codificador",
        "decodificador",
        "gray",
        "bcd",
        "sete segmentos"
    ],

    "estagio_3": [
        "Passado",
        "Futuro",
        "Presente",
        "passado",
        "futuro",
        "presente",
        "memória",
        "sem memória",
        "sequencial",
        "flip-flop",
        "flip flop",
        "flip-flops",
        "flip flops",
        "flip flop-jk",
        "flip flop jk",
        "flip flop-JK",
        "flip flop JK",
        "flip flop tipo d",
        "flip flop tipo t",
        "flip flop-d",
        "flip flop-t",
        "flip flop D",
        "flip flop T",
        "flip flop-D",
        "flip flop-T",
        "FF JK",
        "FF T",
        "FF D",
        "clock",
        "pul-up",
        "pul-down",
        "clock de subida",
        "clock de descida",
        "registrador",
        "registradores",
        "contador",
        "contadores",
        "contador síncrono",
        "contador assíncrono",
        "máquina de estado finitos",
        "máquinas de estado",
        "fsm"
    ]
}
#========================================================

# Modelo de Embeddings
embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

STAGE_NAMES = list(STAGES_INFO.keys())

STAGE_EMBEDDINGS = embeddings.embed_documents(
    list(STAGES_INFO.values())
)

def keyword_score(text):
    text = text.lower()
    scores = {}

    for stage, words in KEYWORDS.items():
        count = 0
        for word in words:
            count += text.count(word.lower())
        scores[stage] = count
    return scores


def normalize_keyword_scores(scores):
    max_score = max(scores.values())
    if max_score == 0:
        return {
            stage: 0.0
            for stage in scores
        }
    return {
        stage: value / max_score
        for stage, value in scores.items()
    }
    

# WORKFLOW 1: CLASSIFICAÇÃO POR AGRUPAMENTO DE CHUNKS
def classify_text(text: str):

    # ===== EMBEDDING SCORE =====
    query_embedding = embeddings.embed_query(
        text.lower()
    )
    embedding_scores = cosine_similarity(
        [query_embedding],
        STAGE_EMBEDDINGS
    )[0]
    # ===== KEYWORD SCORE =====
    keyword_scores = normalize_keyword_scores(
        keyword_score(text)
    )
    # ===== SCORE HÍBRIDO =====
    final_scores = []
    print("\n==============================")
    print("SCORES")

    for i, stage in enumerate(STAGE_NAMES):
        final_score = ((settings.WEIGHT_EMBEDDING * embedding_scores[i])+(settings.WEIGHT_KEYWORDS * keyword_scores[stage]))
        final_scores.append(final_score)

        print(
            f"{stage}"
            f" | embedding={embedding_scores[i]:.4f}"
            f" | keyword={keyword_scores[stage]:.4f}"
            f" | final={final_score:.4f}"
        )

    best_idx = np.argmax(final_scores)
    selected_stage = STAGE_NAMES[best_idx]
    print("Selecionado: ", selected_stage)

    return (
        selected_stage,
        final_scores
    )
# ============== ROUTER  ========================
def route_question(question):
    stage, scores = classify_text(question)
    return stage
# =============================================

def select_target_collections(              #Classifica um texto e decide em quais collections ele deve ser armazenado.
    text: str,                              #Texto do chunk 
    duplication_threshold: float            #Diferença máxima entre os dois melhores scores para permitir duplicação
):
    stage, scores = classify_text(text)

    ranking = sorted(
        zip(STAGE_NAMES, scores),
        key=lambda x: x[1],
        reverse=True
    )

    best_stage = ranking[0][0]
    best_score = ranking[0][1]

    second_stage = ranking[1][0]
    second_score = ranking[1][1]

    difference = best_score - second_score

    if difference < duplication_threshold:

        print(
            f"Chunk duplicado -> "
            f"{best_stage} ({best_score:.4f}) "
            f"e "
            f"{second_stage} ({second_score:.4f})"
        )
        return ([best_stage, second_stage],scores)
    return ([best_stage],scores)
