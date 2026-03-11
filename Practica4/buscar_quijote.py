from __future__ import annotations

import html
import re
import sys
from pathlib import Path


RUTA_QUIJOTE = Path(__file__).with_name("2000-h.htm")
PATRON_BLOQUES = re.compile(
    r"<h3\b[^>]*>.*?</h3>|<p\b[^>]*>.*?</p>",
    re.IGNORECASE | re.DOTALL,
)
PATRON_ETIQUETAS = re.compile(r"<[^>]+>")


def limpiar_html(fragmento: str) -> str:
    fragmento = fragmento.replace("<br />", " ").replace("<br/>", " ").replace("<br>", " ")
    texto = PATRON_ETIQUETAS.sub("", fragmento)
    texto = html.unescape(texto)
    return " ".join(texto.split())


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


def buscar_pasajes(pasajes: list[dict[str, str]], consulta: str) -> list[dict[str, str]]:
    consulta_normalizada = consulta.lower()
    return [pasaje for pasaje in pasajes if consulta_normalizada in pasaje["texto"].lower()]


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
        print("No has introducido ningún texto.")
        raise SystemExit(1)

    pasajes = extraer_pasajes(RUTA_QUIJOTE)
    resultados = buscar_pasajes(pasajes, consulta)

    if not resultados:
        print(f'No se han encontrado pasajes con "{consulta}".')
        return

    print(f'Se han encontrado {len(resultados)} pasajes con "{consulta}".')
    print()

    for indice, resultado in enumerate(resultados[:5], start=1):
        print(f"{indice}. {resultado['encabezado']}")
        print(resultado["texto"])
        print()

    if len(resultados) > 5:
        print(f"Se muestran solo los 5 primeros resultados de {len(resultados)}.")


if __name__ == "__main__":
    main()
