from __future__ import annotations

import html
import re
import sys
from functools import lru_cache
from pathlib import Path

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
    candidatas = [
        Path(__file__).resolve().with_name("2000-h.htm"),
        Path.cwd() / "2000-h.htm",
    ]

    for candidata in candidatas:
        if candidata.exists():
            return candidata

    return candidatas[0]


RUTA_QUIJOTE = obtener_ruta_quijote()


def limpiar_html(fragmento: str) -> str:
    fragmento = fragmento.replace("<br />", " ").replace("<br/>", " ").replace("<br>", " ")
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
def obtener_lemmas_significativos(texto: str) -> frozenset[str]:
    doc = obtener_nlp()(texto)
    lemmas: list[str] = []

    for token in doc:
        if not token.is_alpha or token.is_stop:
            continue

        lemma = token.lemma_.strip().lower() or token.lower_
        lemmas.append(lemma)

    return frozenset(lemmas)


def obtener_rangos_lemmas_coincidentes(texto: str, consulta: str) -> list[tuple[int, int]]:
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

    exactos: list[dict[str, str]] = []
    parciales: list[tuple[int, int, dict[str, str]]] = []

    for indice, pasaje in enumerate(pasajes):
        lemmas_pasaje = obtener_lemmas_significativos(pasaje["texto"])

        if lemmas_consulta.issubset(lemmas_pasaje):
            exactos.append(pasaje)
            continue

        coincidencias = len(lemmas_consulta & lemmas_pasaje)
        if coincidencias:
            parciales.append((coincidencias, indice, pasaje))

    if exactos:
        return exactos, "and"

    parciales.sort(key=lambda item: (-item[0], item[1]))
    return [pasaje for _, _, pasaje in parciales], "or"


def buscar_pasajes(pasajes: list[dict[str, str]], consulta: str) -> list[dict[str, str]]:
    resultados, _ = buscar_pasajes_con_modo(pasajes, consulta)
    return resultados


def formatear_resultados(
    consulta: str,
    resultados: list[dict[str, str]],
    limite: int = LIMITE_RESULTADOS,
    modo_busqueda: str = "and",
) -> str:
    if not resultados:
        return f'No se han encontrado pasajes con "{consulta}".'

    lineas = [f'Se han encontrado {len(resultados)} pasajes con "{consulta}".', ""]

    if modo_busqueda == "or":
        lineas.extend(
            [
                "No hubo una coincidencia completa por lemas.",
                "Se muestran coincidencias parciales con alguno de los lemas buscados.",
                "",
            ]
        )

    for indice, resultado in enumerate(resultados[:limite], start=1):
        lineas.append(f"{indice}. {resultado['encabezado']}")
        lineas.append(resultado["texto"])
        lineas.append("")

    if len(resultados) > limite:
        lineas.append(f"Se muestran solo los {limite} primeros resultados de {len(resultados)}.")

    return "\n".join(lineas).rstrip()


def obtener_consulta() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip()
    return input("Introduce un texto para buscar en Don Quijote: ").strip()


def main() -> None:
    if not RUTA_QUIJOTE.exists():
        print(f"No encuentro el archivo: {RUTA_QUIJOTE}")
        raise SystemExit(1)

    consulta = obtener_consulta()
    if not consulta:
        print("No has introducido ningun texto.")
        raise SystemExit(1)

    pasajes = extraer_pasajes(RUTA_QUIJOTE)
    resultados, modo_busqueda = buscar_pasajes_con_modo(pasajes, consulta)
    print(formatear_resultados(consulta, resultados, modo_busqueda=modo_busqueda))


if __name__ == "__main__":
    main()
