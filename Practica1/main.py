"""Entrypoint del agente: bucle principal de ejecucion."""

from __future__ import annotations

import time

from api_butler import api_get_gente, api_get_info, registrar_identidad
from negociacion import construir_estado, procesar_correo, procesar_turno_sin_correos
from settings import CYCLE_SECONDS, WAIT_WITHOUT_PEERS_SECONDS


def ciclo_principal() -> None:
    """Ejecuta el bucle principal del agente."""
    registrar_identidad()
    print("Agente iniciado")

    while True:
        print("\nNUEVO CICLO")

        info = api_get_info()
        gente = api_get_gente()
        estado = construir_estado(info, gente)

        if not estado["otros"]:
            print("No hay otros agentes en la simulacion todavia. Esperando...")
            time.sleep(WAIT_WITHOUT_PEERS_SECONDS)
            continue

        if not estado["buzon"]:
            procesar_turno_sin_correos(estado)

        for uid, correo in estado["buzon"].items():
            procesar_correo(estado, uid, correo)

        time.sleep(CYCLE_SECONDS)


def main() -> None:
    """Entrypoint CLI."""
    ciclo_principal()


if __name__ == "__main__":
    main()
