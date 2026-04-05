from __future__ import annotations

import argparse
import sys

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Footer, Header, Input, Select, Static

from buscar_quijote import (
    LIMITE_RESULTADOS,
    RUTA_QUIJOTE,
    buscar_pasajes_con_modo,
    extraer_pasajes,
    obtener_rangos_lemmas_coincidentes,
)
from busqueda_semantica import MODELO_EMBEDDINGS, buscar_pasajes_semanticos


ESTILO_RESALTADO = "bold #201a16 on #f0bf5a"
MODO_CLASICO = "clasica"
MODO_EMBEDDINGS = "embeddings"
OPCIONES_MODO = [
    ("1. Busqueda clasica", MODO_CLASICO),
    ("2. Busqueda por embeddings", MODO_EMBEDDINGS),
]


def construir_resultados_enriquecidos(
    consulta: str, resultados: list[dict[str, str]], modo_busqueda: str
) -> Text:
    if not resultados:
        return Text(f'No se han encontrado pasajes con "{consulta}".')

    texto = Text(f'Se han encontrado {len(resultados)} pasajes con "{consulta}".\n\n')

    if modo_busqueda == "or":
        texto.append("No hubo una coincidencia completa por lemas.\n")
        texto.append(
            "Se muestran coincidencias parciales con alguno de los lemas buscados.\n\n"
        )

    for indice, resultado in enumerate(resultados[:LIMITE_RESULTADOS], start=1):
        texto.append(f"{indice}. {resultado['encabezado']}\n")

        pasaje = Text(resultado["texto"])
        for inicio, fin in obtener_rangos_lemmas_coincidentes(
            resultado["texto"], consulta
        ):
            pasaje.stylize(ESTILO_RESALTADO, inicio, fin)

        texto.append_text(pasaje)
        texto.append("\n\n")

    if len(resultados) > LIMITE_RESULTADOS:
        texto.append(
            f"Se muestran solo los {LIMITE_RESULTADOS} primeros resultados de {len(resultados)}."
        )

    return texto


def construir_resultados_semanticos_enriquecidos(
    consulta: str,
    resultados: list[dict[str, str | float | int]],
    modelo: str,
) -> Text:
    if not resultados:
        return Text(
            f'No se han encontrado pasajes semanticamente similares a "{consulta}".'
        )

    texto = Text(
        f'Se han encontrado {len(resultados)} pasajes semanticamente similares a "{consulta}".\n'
    )
    texto.append(f"Modelo de embeddings: {modelo}.\n\n")

    for indice, resultado in enumerate(resultados[:LIMITE_RESULTADOS], start=1):
        texto.append(
            f"{indice}. {resultado['encabezado']} (score: {resultado['score']:.4f})\n"
        )
        texto.append(
            f"Chunk original: pasajes {resultado['inicio']} a {resultado['fin']}\n"
        )
        texto.append(str(resultado["texto"]))
        texto.append("\n\n")

    if len(resultados) > LIMITE_RESULTADOS:
        texto.append(
            f"Se muestran solo los {LIMITE_RESULTADOS} primeros resultados de {len(resultados)}."
        )

    return texto


def parsear_argumentos(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Buscador de pasajes de Don Quijote")
    parser.add_argument("consulta", nargs="*", help="Texto a buscar")
    parser.add_argument(
        "--modo",
        choices=[MODO_CLASICO, MODO_EMBEDDINGS],
        default=MODO_CLASICO,
        help="Modo de busqueda inicial",
    )
    return parser.parse_args(argv)


class BuscadorQuijoteApp(App[None]):
    TITLE = "Don Quijote"
    SUB_TITLE = "Buscador clasico y semantico"
    CSS = """
    Screen {
        background: #f3ede2;
        color: #201a16;
    }

    #intro {
        margin: 1 2 0 2;
        padding: 1 2;
        background: #ead9ba;
        border: tall #8d5b2a;
        color: #3c2413;
    }

    #busqueda {
        margin: 1 2 0 2;
        height: auto;
    }

    #modo {
        width: 28;
        margin-right: 1;
    }

    #consulta {
        width: 1fr;
        margin-right: 1;
    }

    #buscar {
        width: 16;
    }

    #estado {
        margin: 1 2 1 2;
        color: #6c5241;
    }

    #resultados_wrap {
        margin: 0 2 2 2;
        padding: 1 2;
        border: round #8d5b2a;
        background: #fff9ef;
    }

    #resultados {
        width: 100%;
    }
    """
    BINDINGS = [("ctrl+q", "quit", "Salir"), ("ctrl+c", "quit", "Salir")]

    def __init__(
        self, consulta_inicial: str = "", modo_inicial: str = MODO_CLASICO
    ) -> None:
        super().__init__()
        self.consulta_inicial = consulta_inicial
        self.modo_inicial = modo_inicial
        self.pasajes: list[dict[str, str]] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static(
            "Elige un modo: clasico por lemas o semantico por embeddings. Luego escribe una consulta y busca pasajes del Quijote.",
            id="intro",
        )
        with Horizontal(id="busqueda"):
            yield Select(
                OPCIONES_MODO,
                allow_blank=False,
                value=self.modo_inicial,
                id="modo",
            )
            yield Input(
                placeholder="Introduce un texto para buscar en Don Quijote",
                id="consulta",
            )
            yield Button("Buscar", id="buscar", variant="primary")
        yield Static("Cargando pasajes del Quijote...", id="estado")
        with VerticalScroll(id="resultados_wrap"):
            yield Static("Introduce una consulta para comenzar.", id="resultados")
        yield Footer()

    def on_mount(self) -> None:
        consulta = self.query_one("#consulta", Input)
        consulta.value = self.consulta_inicial
        consulta.focus()

        if not RUTA_QUIJOTE.exists():
            mensaje = f"No encuentro el archivo: {RUTA_QUIJOTE}"
            self.mostrar_resultados(mensaje)
            self.actualizar_estado(mensaje)
            self.query_one("#buscar", Button).disabled = True
            consulta.disabled = True
            return

        self.pasajes = extraer_pasajes(RUTA_QUIJOTE)
        self.actualizar_estado(
            f"Archivo cargado: {RUTA_QUIJOTE.name}. Pasajes disponibles: {len(self.pasajes)}. Modos disponibles: clasica y embeddings."
        )

        if self.consulta_inicial:
            self.realizar_busqueda()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "consulta":
            self.realizar_busqueda()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "buscar":
            self.realizar_busqueda()

    def actualizar_estado(self, mensaje: str) -> None:
        self.query_one("#estado", Static).update(Text(mensaje))

    def mostrar_resultados(self, renderizable: str | Text) -> None:
        if isinstance(renderizable, Text):
            self.query_one("#resultados", Static).update(renderizable)
            return

        self.query_one("#resultados", Static).update(Text(renderizable))

    def realizar_busqueda(self) -> None:
        if not self.pasajes:
            return

        consulta = self.query_one("#consulta", Input).value.strip()
        modo = self.query_one("#modo", Select).value
        if not consulta:
            self.actualizar_estado("Esperando una consulta.")
            self.mostrar_resultados("No has introducido ningun texto.")
            return

        if modo == MODO_EMBEDDINGS:
            self.actualizar_estado(
                f'Consulta actual: "{consulta}". Generando o cargando embeddings con {MODELO_EMBEDDINGS}...'
            )
            try:
                resultados_semanticos, modelo = buscar_pasajes_semanticos(
                    self.pasajes, consulta, limite=LIMITE_RESULTADOS
                )
            except Exception as error:
                mensaje_error = (
                    "No se ha podido ejecutar la busqueda por embeddings. "
                    f"Detalle: {error}"
                )
                self.actualizar_estado(mensaje_error)
                self.mostrar_resultados(mensaje_error)
                return

            self.actualizar_estado(
                f'Consulta actual: "{consulta}". Resultados semanticos: {len(resultados_semanticos)}. Modelo: {modelo}.'
            )
            self.mostrar_resultados(
                construir_resultados_semanticos_enriquecidos(
                    consulta, resultados_semanticos, modelo
                )
            )
            return

        resultados, modo_busqueda = buscar_pasajes_con_modo(self.pasajes, consulta)
        mensaje_estado = f'Consulta actual: "{consulta}". Coincidencias encontradas: {len(resultados)}.'

        if modo_busqueda == "or" and resultados:
            mensaje_estado += (
                " Sin coincidencia completa; mostrando coincidencias parciales."
            )

        self.actualizar_estado(mensaje_estado)
        self.mostrar_resultados(
            construir_resultados_enriquecidos(consulta, resultados, modo_busqueda)
        )


def main() -> None:
    argumentos = parsear_argumentos(sys.argv[1:])
    consulta_inicial = " ".join(argumentos.consulta).strip()
    BuscadorQuijoteApp(
        consulta_inicial=consulta_inicial,
        modo_inicial=argumentos.modo,
    ).run()


if __name__ == "__main__":
    main()
