from __future__ import annotations

import os
import sys
import sysconfig
from functools import lru_cache
from pathlib import Path

import numpy as np
import ollama


LIMITE_RESULTADOS = 5
MODELO_EMBEDDINGS = os.getenv("FDI_PLN_P4_EMBED_MODEL", "nomic-embed-text:latest")
TAMANO_LOTE = 1
TOKENS_POR_CHUNK = 512
SOLAPE_TOKENS = TOKENS_POR_CHUNK // 4


def obtener_ruta_cache_embeddings(
    modelo: str = MODELO_EMBEDDINGS,
    tokens_por_chunk: int = TOKENS_POR_CHUNK,
    solape_tokens: int = SOLAPE_TOKENS,
) -> Path:
    nombre = "".join(
        caracter if caracter.isalnum() or caracter in "._-" else "_"
        for caracter in modelo
    )
    archivo = (
        f"embeddings_quijote_{nombre}_tokens_{tokens_por_chunk}_{solape_tokens}.npz"
    )
    candidatas = [
        Path(__file__).resolve().with_name(archivo),
        Path.cwd() / archivo,
        Path(sysconfig.get_paths().get("data", "")) / archivo,
        Path(sys.prefix) / archivo,
    ]
    vistas: set[Path] = set()

    for candidata in candidatas:
        if candidata in vistas:
            continue
        vistas.add(candidata)
        if candidata.exists():
            return candidata

    return Path.cwd() / archivo


def construir_capitulos(
    pasajes: list[dict[str, str]],
) -> list[dict[str, str | int]]:
    if not pasajes:
        return []

    capitulos: list[dict[str, str | int]] = []
    encabezado_actual = pasajes[0]["encabezado"]
    inicio_actual = 0
    pasajes_actuales: list[dict[str, str]] = []

    def cerrar_capitulo(fin_actual: int) -> None:
        palabras: list[str] = []
        rangos_pasajes: list[dict[str, int]] = []
        cursor = 0

        for indice_pasaje, pasaje in enumerate(pasajes_actuales, start=inicio_actual):
            palabras_pasaje = pasaje["texto"].split()
            inicio_palabras = cursor
            cursor += len(palabras_pasaje)
            palabras.extend(palabras_pasaje)
            rangos_pasajes.append(
                {
                    "indice": indice_pasaje,
                    "inicio_palabras": inicio_palabras,
                    "fin_palabras": cursor,
                }
            )

        capitulos.append(
            {
                "encabezado": encabezado_actual,
                "palabras": palabras,
                "rangos_pasajes": rangos_pasajes,
                "inicio": inicio_actual,
                "fin": fin_actual,
            }
        )

    for indice, pasaje in enumerate(pasajes):
        if pasaje["encabezado"] != encabezado_actual and pasajes_actuales:
            cerrar_capitulo(indice - 1)
            encabezado_actual = pasaje["encabezado"]
            inicio_actual = indice
            pasajes_actuales = [pasaje]
            continue

        pasajes_actuales.append(pasaje)

    if pasajes_actuales:
        cerrar_capitulo(len(pasajes) - 1)

    return capitulos


def obtener_rango_pasajes_en_chunk(
    rangos_pasajes: list[dict[str, int]],
    inicio_palabras: int,
    fin_palabras: int,
) -> tuple[int, int]:
    inicio = rangos_pasajes[0]["indice"]
    fin = rangos_pasajes[-1]["indice"]

    for rango in rangos_pasajes:
        if rango["fin_palabras"] > inicio_palabras:
            inicio = rango["indice"]
            break

    for rango in rangos_pasajes:
        if rango["inicio_palabras"] < fin_palabras:
            fin = rango["indice"]
        else:
            break

    return inicio, fin


def construir_chunks_de_capitulo(
    capitulo: dict[str, object],
    tokens_por_chunk: int,
    solape_tokens: int,
) -> list[dict[str, str | int]]:
    palabras = list(capitulo["palabras"])
    if not palabras:
        return []

    paso = max(1, tokens_por_chunk - solape_tokens)
    chunks: list[dict[str, str | int]] = []
    rangos_pasajes = list(capitulo["rangos_pasajes"])

    for inicio in range(0, len(palabras), paso):
        palabras_chunk = palabras[inicio : inicio + tokens_por_chunk]
        if not palabras_chunk:
            continue

        fin = inicio + len(palabras_chunk)
        inicio_pasaje, fin_pasaje = obtener_rango_pasajes_en_chunk(
            rangos_pasajes,
            inicio_palabras=inicio,
            fin_palabras=fin,
        )
        chunks.append(
            {
                "encabezado": str(capitulo["encabezado"]),
                "texto": " ".join(palabras_chunk),
                "inicio": inicio_pasaje,
                "fin": fin_pasaje,
            }
        )

        if fin >= len(palabras):
            break

    return chunks


def construir_chunks_por_tokens(
    pasajes: list[dict[str, str]],
    tokens_por_chunk: int = TOKENS_POR_CHUNK,
    solape_tokens: int = SOLAPE_TOKENS,
) -> list[dict[str, str | int]]:
    if not pasajes:
        return []

    chunks: list[dict[str, str | int]] = []

    for capitulo in construir_capitulos(pasajes):
        chunks.extend(
            construir_chunks_de_capitulo(
                capitulo,
                tokens_por_chunk=tokens_por_chunk,
                solape_tokens=solape_tokens,
            )
        )

    return chunks


def construir_chunks_semanticos(
    pasajes: list[dict[str, str]],
    tokens_por_chunk: int = TOKENS_POR_CHUNK,
    solape_tokens: int = SOLAPE_TOKENS,
) -> list[dict[str, str | int]]:
    return construir_chunks_por_tokens(
        pasajes,
        tokens_por_chunk=tokens_por_chunk,
        solape_tokens=solape_tokens,
    )


@lru_cache(maxsize=1)
def obtener_cliente_ollama() -> ollama.Client:
    return ollama.Client()


def normalizar_embeddings(matriz: np.ndarray) -> np.ndarray:
    if matriz.size == 0:
        return matriz

    normas = np.linalg.norm(matriz, axis=1, keepdims=True)
    normas[normas == 0] = 1.0
    return matriz / normas


def normalizar_consulta(vector: np.ndarray) -> np.ndarray:
    norma = np.linalg.norm(vector)
    if norma == 0:
        return vector
    return vector / norma


def generar_embeddings_textos(
    textos: list[str],
    modelo: str = MODELO_EMBEDDINGS,
    tamano_lote: int = TAMANO_LOTE,
) -> np.ndarray:
    if not textos:
        return np.empty((0, 0), dtype=np.float32)

    cliente = obtener_cliente_ollama()
    lotes: list[np.ndarray] = []

    for inicio in range(0, len(textos), tamano_lote):
        lote = textos[inicio : inicio + tamano_lote]
        respuesta = cliente.embed(model=modelo, input=lote)
        lotes.append(np.asarray(respuesta.embeddings, dtype=np.float32))

    return normalizar_embeddings(np.vstack(lotes))


def obtener_embedding_consulta(
    consulta: str,
    modelo: str = MODELO_EMBEDDINGS,
) -> np.ndarray:
    cliente = obtener_cliente_ollama()
    respuesta = cliente.embed(model=modelo, input=[consulta])
    vector = np.asarray(respuesta.embeddings[0], dtype=np.float32)
    return normalizar_consulta(vector)


def guardar_cache_embeddings(
    ruta: Path, chunks: list[dict[str, str | int]], embeddings: np.ndarray
) -> None:
    np.savez_compressed(
        ruta,
        embeddings=embeddings.astype(np.float32),
        encabezados=np.asarray([chunk["encabezado"] for chunk in chunks], dtype=object),
        textos=np.asarray([chunk["texto"] for chunk in chunks], dtype=object),
        inicios=np.asarray([chunk["inicio"] for chunk in chunks], dtype=np.int32),
        fines=np.asarray([chunk["fin"] for chunk in chunks], dtype=np.int32),
    )


def cargar_cache_embeddings(
    ruta: Path,
) -> tuple[list[dict[str, str | int]], np.ndarray] | None:
    if not ruta.exists():
        return None

    with np.load(ruta, allow_pickle=True) as datos:
        embeddings = datos["embeddings"].astype(np.float32)
        encabezados = datos["encabezados"].tolist()
        textos = datos["textos"].tolist()
        inicios = datos["inicios"].tolist()
        fines = datos["fines"].tolist()

    chunks = [
        {
            "encabezado": encabezado,
            "texto": texto,
            "inicio": int(inicio),
            "fin": int(fin),
        }
        for encabezado, texto, inicio, fin in zip(encabezados, textos, inicios, fines)
    ]
    return chunks, embeddings


def construir_indice_semantico(
    pasajes: list[dict[str, str]],
    modelo: str = MODELO_EMBEDDINGS,
    tokens_por_chunk: int = TOKENS_POR_CHUNK,
    solape_tokens: int = SOLAPE_TOKENS,
    regenerar: bool = False,
) -> tuple[list[dict[str, str | int]], np.ndarray]:
    chunks = construir_chunks_semanticos(
        pasajes,
        tokens_por_chunk=tokens_por_chunk,
        solape_tokens=solape_tokens,
    )
    ruta_cache = obtener_ruta_cache_embeddings(
        modelo,
        tokens_por_chunk=tokens_por_chunk,
        solape_tokens=solape_tokens,
    )

    if not regenerar:
        cache = cargar_cache_embeddings(ruta_cache)
        if cache is not None:
            chunks_cache, embeddings_cache = cache
            if [chunk["texto"] for chunk in chunks_cache] == [
                chunk["texto"] for chunk in chunks
            ]:
                return chunks_cache, embeddings_cache

    embeddings = generar_embeddings_textos(
        [str(chunk["texto"]) for chunk in chunks],
        modelo=modelo,
    )
    guardar_cache_embeddings(ruta_cache, chunks, embeddings)
    return chunks, embeddings


def calcular_scores_semanticos(
    embedding_consulta: np.ndarray, embeddings_chunks: np.ndarray
) -> np.ndarray:
    if embeddings_chunks.size == 0 or embedding_consulta.size == 0:
        return np.empty(0, dtype=np.float32)

    return embeddings_chunks @ embedding_consulta


def buscar_pasajes_semanticos(
    pasajes: list[dict[str, str]],
    consulta: str,
    limite: int = LIMITE_RESULTADOS,
    modelo: str = MODELO_EMBEDDINGS,
    tokens_por_chunk: int = TOKENS_POR_CHUNK,
    solape_tokens: int = SOLAPE_TOKENS,
    regenerar: bool = False,
) -> tuple[list[dict[str, str | float | int]], str]:
    if not consulta.strip():
        return [], modelo

    chunks, embeddings = construir_indice_semantico(
        pasajes,
        modelo=modelo,
        tokens_por_chunk=tokens_por_chunk,
        solape_tokens=solape_tokens,
        regenerar=regenerar,
    )
    embedding_consulta = obtener_embedding_consulta(consulta, modelo=modelo)
    scores = calcular_scores_semanticos(embedding_consulta, embeddings)

    if scores.size == 0:
        return [], modelo

    indices_ordenados = np.argsort(scores)[::-1][:limite]
    resultados: list[dict[str, str | float | int]] = []

    for indice in indices_ordenados:
        chunk = chunks[int(indice)]
        resultados.append(
            {
                "encabezado": str(chunk["encabezado"]),
                "texto": str(chunk["texto"]),
                "score": float(scores[int(indice)]),
                "inicio": int(chunk["inicio"]),
                "fin": int(chunk["fin"]),
            }
        )

    return resultados, modelo
