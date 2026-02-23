"""Entrypoint del agente: bucle principal de ejecucion."""

from __future__ import annotations

import logging
import time

from api_butler import api_get_gente, api_get_info, registrar_identidad
from logger_config import setup_logging
from negociacion import construir_estado, procesar_correo, procesar_turno_sin_correos
from settings import CYCLE_SECONDS, WAIT_WITHOUT_PEERS_SECONDS

logger = logging.getLogger("agente.main")


def ciclo_principal() -> None:
    """Ejecuta el bucle principal del agente."""
    registrar_identidad()
    logger.info("Agente iniciado")

    try:
        while True:
            logger.info("=== NUEVO CICLO ===")

            info = api_get_info()
            gente = api_get_gente()
            estado = construir_estado(info, gente)

            if not estado["otros"]:
                logger.info(
                    "No hay otros agentes en la simulacion todavia. Esperando..."
                )
                time.sleep(WAIT_WITHOUT_PEERS_SECONDS)
                continue

            if not estado["buzon"]:
                procesar_turno_sin_correos(estado)

            for uid, correo in estado["buzon"].items():
                procesar_correo(estado, uid, correo)

            time.sleep(CYCLE_SECONDS)
    except KeyboardInterrupt:
        logger.info("Interrupcion por teclado. Agente detenido.")


def main() -> None:
    """Entrypoint CLI."""
    setup_logging()
    ciclo_principal()


if __name__ == "__main__":
    main()
