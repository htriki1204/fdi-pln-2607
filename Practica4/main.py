from __future__ import annotations

import re
import sys

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Footer, Header, Input, Static

from buscar_quijote import (
    LIMITE_RESULTADOS,
    RUTA_QUIJOTE,
    buscar_pasajes,
    extraer_pasajes,
)


ESTILO_RESALTADO = "bold #201a16 on #f0bf5a"


def construir_resultados_enriquecidos(consulta: str, resultados: list[dict[str, str]]) -> Text:
    if not resultados:
        return Text(f'No se han encontrado pasajes con "{consulta}".')

    patron = re.compile(re.escape(consulta), re.IGNORECASE)
    texto = Text(f'Se han encontrado {len(resultados)} pasajes con "{consulta}".\n\n')

    for indice, resultado in enumerate(resultados[:LIMITE_RESULTADOS], start=1):
        texto.append(f"{indice}. {resultado['encabezado']}\n")

        pasaje = Text(resultado["texto"])
        for coincidencia in patron.finditer(resultado["texto"]):
            pasaje.stylize(ESTILO_RESALTADO, coincidencia.start(), coincidencia.end())

        texto.append_text(pasaje)
        texto.append("\n\n")

    if len(resultados) > LIMITE_RESULTADOS:
        texto.append(
            f"Se muestran solo los {LIMITE_RESULTADOS} primeros resultados de {len(resultados)}."
        )

    return texto


class BuscadorQuijoteApp(App[None]):
    TITLE = "Don Quijote"
    SUB_TITLE = "Buscador de pasajes"
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

    def __init__(self, consulta_inicial: str = "") -> None:
        super().__init__()
        self.consulta_inicial = consulta_inicial
        self.pasajes: list[dict[str, str]] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static(
            "Escribe un texto, pulsa Enter o el boton Buscar para encontrar los pasajes que coincidan con el texto.",
            id="intro",
        )
        with Horizontal(id="busqueda"):
            yield Input(placeholder="Introduce un texto para buscar en Don Quijote", id="consulta")
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
            f"Archivo cargado: {RUTA_QUIJOTE.name}. Pasajes disponibles: {len(self.pasajes)}. Se mostraran como maximo {LIMITE_RESULTADOS} resultados."
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
        if not consulta:
            self.actualizar_estado("Esperando una consulta.")
            self.mostrar_resultados("No has introducido ningun texto.")
            return

        resultados = buscar_pasajes(self.pasajes, consulta)
        self.actualizar_estado(
            f'Consulta actual: "{consulta}". Coincidencias encontradas: {len(resultados)}.'
        )
        self.mostrar_resultados(construir_resultados_enriquecidos(consulta, resultados))


def main() -> None:
    consulta_inicial = " ".join(sys.argv[1:]).strip()
    BuscadorQuijoteApp(consulta_inicial=consulta_inicial).run()


if __name__ == "__main__":
    main()
