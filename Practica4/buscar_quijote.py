from __future__ import annotations

import html
import re
import sys
import sysconfig
from collections import Counter
from functools import lru_cache
from pathlib import Path
from nltk.text import TextCollection

try:
    import spacy
except ImportError:  # pragma: no cover - depende del entorno
    spacy = None


LIMITE_RESULTADOS = 5
PATRON_BLOQUES = re.compile(
    r"<h3\b[^>]*>.*?</h3>|<p\b[^>]*>.*?</p>",
    re.IGNORECASE | re.DOTALL,
)
PATRON_ETIQUETAS = re.compile(r"<[^>]+>")


def obtener_ruta_quijote() -> Path:
    nombre = "2000-h.htm"
    candidatas = [
        Path(__file__).resolve().with_name(nombre),
        Path.cwd() / nombre,
        Path(sysconfig.get_paths().get("data", "")) / nombre,
        Path(sys.prefix) / nombre,
    ]
    vistas: set[Path] = set()
    candidatas_unicas: list[Path] = []

    for candidata in candidatas:
        if candidata in vistas:
            continue
        vistas.add(candidata)
        candidatas_unicas.append(candidata)

    for candidata in candidatas_unicas:
        if candidata.exists():
            return candidata

    return candidatas_unicas[0]


RUTA_QUIJOTE = obtener_ruta_quijote()


def limpiar_html(fragmento: str) -> str:
    fragmento = (
        fragmento.replace("<br />", " ").replace("<br/>", " ").replace("<br>", " ")
    )
    texto = PATRON_ETIQUETAS.sub("", fragmento)
    texto = html.unescape(texto)
    return " ".join(texto.split())


@lru_cache(maxsize=1)
def obtener_nlp():
    if spacy is None:
        raise RuntimeError(
            "spaCy no esta disponible. Ejecuta `uv sync` en Practica4 para instalar las dependencias."
        )

    nlp = spacy.blank("es")
    nlp.add_pipe("lemmatizer", config={"mode": "lookup"})
    nlp.initialize()
    return nlp


@lru_cache(maxsize=6000)
def obtener_lista_lemmas_significativos(texto: str) -> tuple[str, ...]:
    doc = obtener_nlp()(texto)
    lemmas: list[str] = []

    for token in doc:
        if not token.is_alpha or token.is_stop:
            continue

        lemma = token.lemma_.strip().lower() or token.lower_
        lemmas.append(lemma)

    return tuple(lemmas)


@lru_cache(maxsize=6000)
def obtener_lemmas_significativos(texto: str) -> frozenset[str]:
    return frozenset(obtener_lista_lemmas_significativos(texto))


@lru_cache(maxsize=1)
def construir_indice_tfidf(textos_normalizados: tuple[tuple[str, ...], ...]):
    if TextCollection is None:
        raise RuntimeError(
            "NLTK no esta disponible. Ejecuta `uv sync` en Practica4 para instalar las dependencias."
        )

    return TextCollection(textos_normalizados)


def obtener_scores_tfidf(pasajes: list[dict[str, str]], consulta: str) -> list[float]:
    tokens_consulta = obtener_lista_lemmas_significativos(consulta)
    if not tokens_consulta:
        return [0.0] * len(pasajes)

    textos_normalizados = tuple(
        obtener_lista_lemmas_significativos(pasaje["texto"]) for pasaje in pasajes
    )
    coleccion = construir_indice_tfidf(textos_normalizados)
    frecuencias_consulta = Counter(tokens_consulta)
    total_terminos_consulta = len(tokens_consulta)
    scores: list[float] = []

    for tokens_documento in textos_normalizados:
        if not tokens_documento:
            scores.append(0.0)
            continue

        score = 0.0

        for termino, frecuencia in frecuencias_consulta.items():
            peso_consulta = frecuencia / total_terminos_consulta
            score += peso_consulta * coleccion.tf_idf(termino, tokens_documento)

        scores.append(score)

    return scores


def obtener_rangos_lemmas_coincidentes(
    texto: str, consulta: str
) -> list[tuple[int, int]]:
    lemmas_consulta = obtener_lemmas_significativos(consulta)
    if not lemmas_consulta:
        return []

    doc = obtener_nlp()(texto)
    rangos: list[tuple[int, int]] = []

    for token in doc:
        if not token.is_alpha or token.is_stop:
            continue

        lemma = token.lemma_.strip().lower() or token.lower_
        if lemma in lemmas_consulta:
            rangos.append((token.idx, token.idx + len(token.text)))

    return rangos


def extraer_pasajes(ruta_html: Path) -> list[dict[str, str]]:
    contenido = ruta_html.read_text(encoding="utf-8")
    pasajes: list[dict[str, str]] = []
    encabezado_actual = "Sin encabezado"

    for bloque in PATRON_BLOQUES.finditer(contenido):
        texto_bloque = bloque.group(0)

        if texto_bloque.lower().startswith("<h3"):
            encabezado_limpio = limpiar_html(texto_bloque)
            if encabezado_limpio:
                encabezado_actual = encabezado_limpio
            continue

        pasaje = limpiar_html(texto_bloque)
        if pasaje:
            pasajes.append({"encabezado": encabezado_actual, "texto": pasaje})

    return pasajes


def buscar_pasajes_con_modo(
    pasajes: list[dict[str, str]], consulta: str
) -> tuple[list[dict[str, str]], str]:
    lemmas_consulta = obtener_lemmas_significativos(consulta)
    if not lemmas_consulta:
        return [], "and"

    scores_tfidf = obtener_scores_tfidf(pasajes, consulta)
    exactos: list[tuple[float, int, dict[str, str]]] = []
    parciales: list[tuple[int, float, int, dict[str, str]]] = []

    for indice, pasaje in enumerate(pasajes):
        lemmas_pasaje = obtener_lemmas_significativos(pasaje["texto"])

        if lemmas_consulta.issubset(lemmas_pasaje):
            exactos.append((scores_tfidf[indice], indice, pasaje))
            continue

        coincidencias = len(lemmas_consulta & lemmas_pasaje)
        if coincidencias:
            parciales.append((coincidencias, scores_tfidf[indice], indice, pasaje))

    if exactos:
        exactos.sort(key=lambda item: (-item[0], item[1]))
        return [pasaje for _, _, pasaje in exactos], "and"

    parciales.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [pasaje for _, _, _, pasaje in parciales], "or"
