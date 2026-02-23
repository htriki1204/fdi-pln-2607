"""Funciones de acceso HTTP al servidor Butler."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

import requests

from settings import BASE_URL, MI_ALIAS, REQUEST_TIMEOUT

logger = logging.getLogger("agente.api")


def parametros_agente() -> dict[str, str]:
    """Devuelve query params para modo monopuesto."""
    return {"agente": MI_ALIAS}


def _request_json(
    method: str,
    endpoint: str,
    payload: dict[str, Any] | None = None,
) -> Any:
    """Hace una llamada HTTP y devuelve JSON si procede."""
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    try:
        response = requests.request(
            method=method,
            url=url,
            params=parametros_agente(),
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        method_upper = method.upper()
        if method_upper == "GET":
            logger.debug("%s /%s -> %s", method_upper, endpoint, response.status_code)
        else:
            logger.info("%s /%s -> %s", method_upper, endpoint, response.status_code)
        return response.json() if response.content else {}
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        if (
            method.upper() == "POST"
            and endpoint.startswith("alias/")
            and status_code == 403
        ):
            logger.info(
                "Alias ya registrado en Butler (%s). Se reutiliza sesion.",
                MI_ALIAS,
            )
            return None
        logger.error("Error %s /%s: %s", method.upper(), endpoint, exc)
        return {} if method.upper() == "GET" else None
    except requests.RequestException as exc:
        logger.error("Error %s /%s: %s", method.upper(), endpoint, exc)
        return {} if method.upper() == "GET" else None


def registrar_identidad() -> None:
    """Registra el alias del agente en Butler."""
    logger.info("Registrando alias en Butler: %s", MI_ALIAS)
    _request_json("POST", f"alias/{MI_ALIAS}")


def api_get_info() -> dict[str, Any]:
    """Recupera estado general del agente desde Butler."""
    result = _request_json("GET", "info")
    return result if isinstance(result, dict) else {}


def api_get_gente() -> list[Any]:
    """Recupera lista de agentes visibles."""
    result = _request_json("GET", "gente")
    return result if isinstance(result, list) else []


def api_post_carta(destinatario: str, asunto: str, cuerpo: str) -> None:
    """Envia una carta a otro agente."""
    payload = {
        "remi": MI_ALIAS,
        "dest": destinatario,
        "asunto": asunto,
        "cuerpo": cuerpo,
        "id": str(uuid.uuid4())[:8],
        "fecha": time.strftime("%Y-%m-%d %H:%M"),
    }
    logger.info(
        "Carta enviada -> dest=%s asunto=%s cuerpo=%s",
        destinatario,
        asunto,
        cuerpo,
    )
    _request_json("POST", "carta", payload=payload)


def api_post_paquete(destinatario: str, recursos: dict[str, int]) -> None:
    """Envia un paquete de recursos a otro agente."""
    logger.info("Paquete enviado -> dest=%s recursos=%s", destinatario, recursos)
    _request_json("POST", f"paquete/{destinatario}", payload=recursos)


def api_delete_mail(uid: str) -> None:
    """Elimina un correo del buzon."""
    logger.debug("Borrando correo uid=%s", uid)
    _request_json("DELETE", f"mail/{uid}")


def parse_resource_map(value: Any) -> dict[str, int]:
    """Convierte una estructura arbitraria en un mapa de recursos valido."""
    if not isinstance(value, dict):
        return {}
    parsed: dict[str, int] = {}
    for key, amount in value.items():
        if isinstance(key, str) and isinstance(amount, int):
            parsed[key] = amount
    return parsed


def parse_mailbox(value: Any) -> dict[str, dict[str, Any]]:
    """Convierte el buzon en uid -> correo."""
    if not isinstance(value, dict):
        return {}
    mailbox: dict[str, dict[str, Any]] = {}
    for uid, mail in value.items():
        if isinstance(uid, str) and isinstance(mail, dict):
            mailbox[uid] = mail
    return mailbox


def parse_other_aliases(gente: list[Any]) -> list[str]:
    """Normaliza la salida de /gente y filtra el alias propio."""
    others: list[str] = []
    for persona in gente:
        alias: str | None = None
        if isinstance(persona, dict) and isinstance(persona.get("alias"), str):
            alias = persona["alias"]
        elif isinstance(persona, str):
            alias = persona

        if alias and alias != MI_ALIAS:
            others.append(alias)
    return others
