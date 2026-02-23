"""Configuracion global del agente cargada desde variables de entorno."""

from __future__ import annotations

import os


def _int_env(var_name: str, default: int) -> int:
    """Lee un entero desde entorno o devuelve el valor por defecto."""
    raw_value = os.getenv(var_name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


BUTLER_ADDRESS = (
    os.getenv("FDI_PLN__BUTLER_ADDRESS", "127.0.0.1:7719").strip().rstrip("/")
)
BASE_URL = f"http://{BUTLER_ADDRESS}"
MI_ALIAS = os.getenv("FDI_PLN__ALIAS", "hamza_agent")
OLLAMA_HOST = os.getenv("FDI_PLN__OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL_NAME = os.getenv("FDI_PLN__MODEL", "llama3.2:latest")

REQUEST_TIMEOUT = _int_env("FDI_PLN__REQUEST_TIMEOUT", 10)
CYCLE_SECONDS = _int_env("FDI_PLN__CYCLE_SECONDS", 10)
WAIT_WITHOUT_PEERS_SECONDS = _int_env("FDI_PLN__WAIT_WITHOUT_PEERS_SECONDS", 10)
PROACTIVE_COOLDOWN_SECONDS = _int_env("FDI_PLN__PROACTIVE_COOLDOWN_SECONDS", 45)

LOG_LEVEL = os.getenv("FDI_PLN__LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("FDI_PLN__LOG_FILE", "logs/agente.log")
LOG_MAX_BYTES = _int_env("FDI_PLN__LOG_MAX_BYTES", 2_000_000)
LOG_BACKUP_COUNT = _int_env("FDI_PLN__LOG_BACKUP_COUNT", 3)
