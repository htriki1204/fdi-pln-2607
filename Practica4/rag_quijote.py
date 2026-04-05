from __future__ import annotations

import os
from functools import lru_cache

import ollama

from buscar_quijote import LIMITE_RESULTADOS, buscar_pasajes_con_modo
from busqueda_semantica import (
    CHUNKING_TOKENS,
    MODELO_EMBEDDINGS,
    buscar_pasajes_semanticos,
)


MODELO_RAG = os.getenv("FDI_PLN_P4_RAG_MODEL", "llama3.2:latest")
MAX_RESULTADOS_CLASICOS = 3
MAX_RESULTADOS_SEMANTICOS = 0


@lru_cache(maxsize=1)
def obtener_cliente_ollama() -> ollama.Client:
    return ollama.Client()


def construir_contexto_rag(
    consulta: str,
    pasajes: list[dict[str, str]],
    max_clasicos: int = MAX_RESULTADOS_CLASICOS,
    max_semanticos: int = MAX_RESULTADOS_SEMANTICOS,
) -> tuple[list[dict[str, str]], str]:
    if max_clasicos > 0:
        resultados_clasicos, modo_clasico = buscar_pasajes_con_modo(pasajes, consulta)
    else:
        resultados_clasicos, modo_clasico = [], "and"

    if max_semanticos > 0:
        resultados_semanticos, _ = buscar_pasajes_semanticos(
            pasajes,
            consulta,
            limite=max_semanticos,
            estrategia_chunking=CHUNKING_TOKENS,
        )
    else:
        resultados_semanticos = []

    contexto: list[dict[str, str]] = []
    textos_vistos: set[str] = set()

    for indice, resultado in enumerate(resultados_clasicos[:max_clasicos], start=1):
        texto = resultado["texto"]
        if texto in textos_vistos:
            continue
        textos_vistos.add(texto)
        contexto.append(
            {
                "referencia": f"C{indice}",
                "fuente": "clasica",
                "encabezado": resultado["encabezado"],
                "texto": texto,
            }
        )

    for indice, resultado in enumerate(resultados_semanticos[:max_semanticos], start=1):
        texto = str(resultado["texto"])
        if texto in textos_vistos:
            continue
        textos_vistos.add(texto)
        contexto.append(
            {
                "referencia": f"S{indice}",
                "fuente": "semantica",
                "encabezado": str(resultado["encabezado"]),
                "texto": texto,
            }
        )

    return contexto, modo_clasico


def construir_prompt_contexto(contexto: list[dict[str, str]]) -> str:
    bloques: list[str] = []

    for entrada in contexto:
        bloques.append(
            "\n".join(
                [
                    f"[{entrada['referencia']}] Fuente: {entrada['fuente']}",
                    f"Encabezado: {entrada['encabezado']}",
                    f"Texto: {entrada['texto']}",
                ]
            )
        )

    return "\n\n".join(bloques)


def responder_con_rag(
    consulta: str,
    pasajes: list[dict[str, str]],
    modelo: str = MODELO_RAG,
) -> dict[str, object]:
    contexto, modo_clasico = construir_contexto_rag(consulta, pasajes)
    if not contexto:
        return {
            "respuesta": "No he encontrado contexto suficiente en el Quijote para responder con seguridad.",
            "contexto": [],
            "modelo": modelo,
            "modo_clasico": modo_clasico,
        }

    prompt_contexto = construir_prompt_contexto(contexto)
    cliente = obtener_cliente_ollama()
    respuesta = cliente.chat(
        model=modelo,
        messages=[
            {
                "role": "system",
                "content": (
                    "Responde en espanol usando solo el contexto recuperado del Quijote. "
                    "No inventes informacion. Si el contexto no basta, dilo claramente. "
                    "Cita siempre las referencias usadas entre corchetes, por ejemplo [C1] o [S2]."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Consulta del usuario: {consulta}\n\n"
                    "Contexto recuperado:\n"
                    f"{prompt_contexto}\n\n"
                    "Da una respuesta breve pero completa, y termina con una linea "
                    "de referencias usadas."
                ),
            },
        ],
        stream=False,
    )

    return {
        "respuesta": respuesta.message.content.strip(),
        "contexto": contexto,
        "modelo": modelo,
        "modo_clasico": modo_clasico,
    }


def formatear_respuesta_rag(resultado_rag: dict[str, object]) -> str:
    respuesta = str(resultado_rag["respuesta"])
    contexto = list(resultado_rag["contexto"])
    modelo = str(resultado_rag["modelo"])

    lineas = [
        "Respuesta RAG",
        f"Modelo: {modelo}",
        "",
        respuesta,
    ]

    if contexto:
        lineas.extend(["", "Pasajes aportados al modelo:", ""])
        for entrada in contexto[:LIMITE_RESULTADOS]:
            lineas.append(
                f"[{entrada['referencia']}] {entrada['encabezado']} ({entrada['fuente']})"
            )

    return "\n".join(lineas).rstrip()
