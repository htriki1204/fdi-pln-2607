from __future__ import annotations

import os
import re
from functools import lru_cache

import ollama

from buscar_quijote import buscar_pasajes_con_modo
from busqueda_semantica import buscar_pasajes_semanticos


MODELO_RAG = os.getenv("FDI_PLN_P4_RAG_MODEL", "llama3.2:3b")
MAX_RESULTADOS_CLASICOS = 3
MAX_RESULTADOS_SEMANTICOS = 2


@lru_cache(maxsize=1)
def obtener_cliente_ollama() -> ollama.Client:
    return ollama.Client()


def construir_contexto_rag(
    consulta: str,
    pasajes: list[dict[str, str]],
    max_clasicos: int = MAX_RESULTADOS_CLASICOS,
    max_semanticos: int = MAX_RESULTADOS_SEMANTICOS,
) -> list[dict[str, str]]:
    if max_clasicos > 0:
        resultados_clasicos, _ = buscar_pasajes_con_modo(pasajes, consulta)
    else:
        resultados_clasicos = []

    if max_semanticos > 0:
        resultados_semanticos, _ = buscar_pasajes_semanticos(
            pasajes,
            consulta,
            limite=max_semanticos,
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

    return contexto


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


def limpiar_respuesta_rag(texto: str) -> str:
    texto_limpio = re.sub(
        r"\n+\s*Referencias usadas:.*\Z",
        "",
        texto.strip(),
        flags=re.IGNORECASE | re.DOTALL,
    )
    return texto_limpio.strip()


def asegurar_referencias_en_respuesta(
    texto: str,
    contexto: list[dict[str, str]],
) -> str:
    if re.search(r"\[(?:C|S)\d+\]", texto):
        return texto

    referencias = " ".join(f"[{entrada['referencia']}]" for entrada in contexto)
    if not referencias:
        return texto

    return f"{texto}\n\nReferencias: {referencias}"


def responder_con_rag(
    consulta: str,
    pasajes: list[dict[str, str]],
    modelo: str = MODELO_RAG,
) -> dict[str, object]:
    contexto = construir_contexto_rag(consulta, pasajes)
    if not contexto:
        return {
            "respuesta": "No he encontrado contexto suficiente en el Quijote para responder con seguridad.",
            "contexto": [],
            "modelo": modelo,
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
                    "Cita las referencias usadas entre corchetes dentro de la respuesta, "
                    "por ejemplo [C1] o [S2], pero no anadas una seccion final de referencias."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Consulta del usuario: {consulta}\n\n"
                    "Contexto recuperado:\n"
                    f"{prompt_contexto}\n\n"
                    "Da una respuesta breve pero completa."
                ),
            },
        ],
        stream=False,
    )

    return {
        "respuesta": asegurar_referencias_en_respuesta(
            limpiar_respuesta_rag(respuesta.message.content),
            contexto,
        ),
        "contexto": contexto,
        "modelo": modelo,
    }
