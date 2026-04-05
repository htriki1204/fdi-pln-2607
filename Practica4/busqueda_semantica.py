from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import ollama


LIMITE_RESULTADOS = 5
MODELO_EMBEDDINGS = os.getenv("FDI_PLN_P4_EMBED_MODEL", "llama3.2:latest")
TAMANO_CHUNK = 3
SOLAPE_CHUNK = 1
TAMANO_LOTE = 1


def obtener_ruta_cache_embeddings(
    modelo: str = MODELO_EMBEDDINGS,
    tamano_chunk: int = TAMANO_CHUNK,
    solape_chunk: int = SOLAPE_CHUNK,
) -> Path:
    nombre = "".join(
        caracter if caracter.isalnum() or caracter in "._-" else "_"
        for caracter in modelo
    )
    archivo = f"embeddings_quijote_{nombre}_{tamano_chunk}_{solape_chunk}.npz"
    return Path(__file__).resolve().with_name(archivo)


def combinar_encabezados(pasajes_chunk: list[dict[str, str]]) -> str:
    encabezados: list[str] = []

    for pasaje in pasajes_chunk:
        encabezado = pasaje["encabezado"]
        if encabezados and encabezados[-1] == encabezado:
            continue
        encabezados.append(encabezado)

    if not encabezados:
        return "Sin encabezado"

    if len(encabezados) == 1:
        return encabezados[0]

    return f"{encabezados[0]} / {encabezados[-1]}"


def construir_chunks_semanticos(
    pasajes: list[dict[str, str]],
    tamano_chunk: int = TAMANO_CHUNK,
    solape_chunk: int = SOLAPE_CHUNK,
) -> list[dict[str, str | int]]:
    if not pasajes:
        return []

    paso = max(1, tamano_chunk - solape_chunk)
    chunks: list[dict[str, str | int]] = []

    for inicio in range(0, len(pasajes), paso):
        grupo = pasajes[inicio : inicio + tamano_chunk]
        if not grupo:
            continue

        texto = "\n".join(pasaje["texto"] for pasaje in grupo)
        chunks.append(
            {
                "encabezado": combinar_encabezados(grupo),
                "texto": texto,
                "inicio": inicio,
                "fin": inicio + len(grupo) - 1,
            }
        )

        if inicio + tamano_chunk >= len(pasajes):
            break

    return chunks


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
    consulta: str, modelo: str = MODELO_EMBEDDINGS
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
    tamano_chunk: int = TAMANO_CHUNK,
    solape_chunk: int = SOLAPE_CHUNK,
    regenerar: bool = False,
) -> tuple[list[dict[str, str | int]], np.ndarray]:
    chunks = construir_chunks_semanticos(
        pasajes, tamano_chunk=tamano_chunk, solape_chunk=solape_chunk
    )
    ruta_cache = obtener_ruta_cache_embeddings(modelo, tamano_chunk, solape_chunk)

    if not regenerar:
        cache = cargar_cache_embeddings(ruta_cache)
        if cache is not None:
            chunks_cache, embeddings_cache = cache
            if [chunk["texto"] for chunk in chunks_cache] == [
                chunk["texto"] for chunk in chunks
            ]:
                return chunks_cache, embeddings_cache

    embeddings = generar_embeddings_textos(
        [str(chunk["texto"]) for chunk in chunks], modelo=modelo
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
    regenerar: bool = False,
) -> tuple[list[dict[str, str | float | int]], str]:
    if not consulta.strip():
        return [], modelo

    chunks, embeddings = construir_indice_semantico(
        pasajes, modelo=modelo, regenerar=regenerar
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


def formatear_resultados_semanticos(
    consulta: str,
    resultados: list[dict[str, str | float | int]],
    modelo: str,
    limite: int = LIMITE_RESULTADOS,
) -> str:
    if not resultados:
        return f'No se han encontrado pasajes semanticamente similares a "{consulta}".'

    lineas = [
        f'Se han encontrado {len(resultados)} pasajes semanticamente similares a "{consulta}".',
        f"Modelo de embeddings: {modelo}.",
        "",
    ]

    for indice, resultado in enumerate(resultados[:limite], start=1):
        lineas.append(
            f"{indice}. {resultado['encabezado']} (score: {resultado['score']:.4f})"
        )
        lineas.append(str(resultado["texto"]))
        lineas.append("")

    if len(resultados) > limite:
        lineas.append(
            f"Se muestran solo los {limite} primeros resultados de {len(resultados)}."
        )

    return "\n".join(lineas).rstrip()
