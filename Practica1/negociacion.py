"""Logica de negociacion guiada por IA (Ollama) con tools controladas."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from ollama import Client

from api_butler import (
    api_delete_mail,
    api_post_carta,
    api_post_paquete,
    parse_mailbox,
    parse_other_aliases,
    parse_resource_map,
)
from prompts import (
    TOOLS_SCHEMA,
    construir_prompt_sistema,
    construir_user_prompt_correo,
    construir_user_prompt_proactivo,
)
from settings import MI_ALIAS, MODEL_NAME, OLLAMA_HOST, PROACTIVE_COOLDOWN_SECONDS

logger = logging.getLogger("agente.negociacion")
llm = Client(host=OLLAMA_HOST)

estado_global: dict[str, Any] = {
    "recursos": {},
    "objetivo": {},
    "faltantes": {},
    "sobrantes": {},
    "objetivo_cumplido": False,
    "ultima_propuesta_ts": 0.0,
}


def construir_estado(info: dict[str, Any], gente: list[Any]) -> dict[str, Any]:
    """Construye estado de ciclo y calcula faltantes/sobrantes."""
    recursos = parse_resource_map(info.get("Recursos"))
    objetivo = parse_resource_map(info.get("Objetivo"))
    buzon = parse_mailbox(info.get("Buzon"))
    otros = parse_other_aliases(gente)

    faltantes = _calcular_faltantes(recursos, objetivo)
    sobrantes = _calcular_sobrantes(recursos, objetivo)
    objetivo_cumplido = not faltantes

    estado_global["recursos"] = recursos
    estado_global["objetivo"] = objetivo
    estado_global["faltantes"] = faltantes
    estado_global["sobrantes"] = sobrantes
    estado_global["objetivo_cumplido"] = objetivo_cumplido

    if objetivo_cumplido:
        logger.info("=== OBJETIVO CUMPLIDO === Modo adquisicion de oro activo")

    logger.info(
        "Estado | recursos=%s objetivo=%s faltantes=%s sobrantes=%s otros=%s correos=%s objetivo_cumplido=%s",
        recursos,
        objetivo,
        faltantes,
        sobrantes,
        otros,
        len(buzon),
        objetivo_cumplido,
    )

    return {
        "recursos": recursos,
        "objetivo": objetivo,
        "buzon": buzon,
        "otros": otros,
        "faltantes": faltantes,
        "sobrantes": sobrantes,
        "objetivo_cumplido": objetivo_cumplido,
    }


def procesar_turno_sin_correos(estado: dict[str, Any]) -> None:
    """Lanza una accion proactiva por IA cuando no hay correos."""
    ultimo_envio = float(estado_global.get("ultima_propuesta_ts", 0.0))
    ahora = time.time()
    if ahora - ultimo_envio < PROACTIVE_COOLDOWN_SECONDS:
        logger.info("Cooldown activo para propuesta proactiva")
        return

    otros = estado.get("otros", [])
    if not isinstance(otros, list) or not otros:
        logger.info("Sin otros agentes para propuesta proactiva")
        return

    system_prompt = construir_prompt_sistema(estado)
    user_prompt = construir_user_prompt_proactivo(estado)

    tool_calls = _consultar_llm(system_prompt, user_prompt)
    _ejecutar_tool_calls(tool_calls)
    estado_global["ultima_propuesta_ts"] = ahora


def procesar_correo(estado: dict[str, Any], uid: str, correo: dict[str, Any]) -> None:
    """Procesa un correo individual delegando decision en la IA."""
    remitente = str(correo.get("remi", ""))
    asunto = str(correo.get("asunto", ""))
    cuerpo = str(correo.get("cuerpo", ""))

    if remitente in {"Sistema", MI_ALIAS}:
        logger.info("Correo ignorado uid=%s remi=%s asunto=%s", uid, remitente, asunto)
        api_delete_mail(uid)
        return

    logger.info(
        "Leyendo correo uid=%s remi=%s asunto=%s cuerpo=%s",
        uid,
        remitente,
        asunto,
        cuerpo,
    )

    try:
        estado_actual = _estado_dinamico(estado)
        system_prompt = construir_prompt_sistema(estado_actual)
        user_prompt = construir_user_prompt_correo(
            remitente, asunto, cuerpo, estado_actual
        )

        tool_calls = _consultar_llm(system_prompt, user_prompt)
        _ejecutar_tool_calls(tool_calls)
    finally:
        api_delete_mail(uid)


def _calcular_faltantes(
    recursos: dict[str, int], objetivo: dict[str, int]
) -> dict[str, int]:
    """Devuelve recursos que faltan para completar objetivo."""
    faltantes: dict[str, int] = {}
    for material, cantidad_objetivo in objetivo.items():
        actual = recursos.get(material, 0)
        if actual < cantidad_objetivo:
            faltantes[material] = cantidad_objetivo - actual
    return faltantes


def _calcular_sobrantes(
    recursos: dict[str, int], objetivo: dict[str, int]
) -> dict[str, int]:
    """Devuelve recursos que sobran para poder negociar."""
    sobrantes: dict[str, int] = {}
    for material, actual in recursos.items():
        if material == "oro":
            if actual > 0:
                sobrantes[material] = actual
            continue

        extra = actual - objetivo.get(material, 0)
        if extra > 0:
            sobrantes[material] = extra
    return sobrantes


def _consultar_llm(system_prompt: str, user_prompt: str) -> list[Any]:
    """Consulta al modelo y devuelve tool calls."""
    try:
        response = llm.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=TOOLS_SCHEMA,
        )
    except Exception as exc:
        logger.error("Error consultando LLM: %s", exc)
        return []

    message = (
        response.get("message")
        if isinstance(response, dict)
        else getattr(response, "message", None)
    )
    if message is None:
        logger.warning("LLM sin mensaje de salida")
        return []

    tool_calls = (
        message.get("tool_calls")
        if isinstance(message, dict)
        else getattr(message, "tool_calls", None)
    )
    if not isinstance(tool_calls, list):
        logger.info("LLM no devolvio tool calls")
        return []

    logger.info("LLM devolvio %s tool call(s)", len(tool_calls))
    return tool_calls


def _extraer_tool_call(tool_call: Any) -> tuple[str, dict[str, Any]]:
    """Extrae nombre y argumentos de una tool call."""
    function = (
        tool_call.get("function")
        if isinstance(tool_call, dict)
        else getattr(tool_call, "function", None)
    )
    if function is None:
        return "", {}

    name = (
        function.get("name")
        if isinstance(function, dict)
        else getattr(function, "name", "")
    )
    raw_args = (
        function.get("arguments", {})
        if isinstance(function, dict)
        else getattr(function, "arguments", {})
    )

    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            return name, parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return name, {}

    return name, raw_args if isinstance(raw_args, dict) else {}


def _ejecutar_tool_calls(tool_calls: list[Any]) -> None:
    """Ejecuta tool calls devueltas por la IA."""
    if len(tool_calls) > 1:
        logger.warning(
            "LLM devolvio %s tool calls; se ejecutara solo la primera valida",
            len(tool_calls),
        )

    for tool_call in tool_calls:
        name, args = _extraer_tool_call(tool_call)
        if not name:
            logger.warning("Tool call invalida")
            continue

        logger.info("Tool usada -> %s args=%s", name, args)

        if name == "no_accion":
            logger.info("Sin accion: %s", args.get("razon", "sin razon"))
            return

        if name == "enviar_carta":
            _tool_enviar_carta(args)
            return

        if name == "enviar_paquete":
            _tool_enviar_paquete(args)
            return

        logger.warning("Tool desconocida: %s", name)


def _tool_enviar_carta(args: dict[str, Any]) -> None:
    """Valida y ejecuta enviar_carta."""
    destinatario = _coerce_text(args.get("destinatario"))
    asunto = _coerce_text(args.get("asunto"))
    cuerpo = _coerce_text(args.get("cuerpo"))

    if not all(_texto_util(v) for v in (destinatario, asunto, cuerpo)):
        logger.warning("enviar_carta con parametros invalidos")
        return

    api_post_carta(destinatario, asunto, cuerpo)


def _tool_enviar_paquete(args: dict[str, Any]) -> None:
    """Valida y ejecuta enviar_paquete sin romper estado minimo."""
    destinatario = _coerce_text(args.get("destinatario"))
    recursos_raw = args.get("recursos")

    if not destinatario.strip():
        logger.warning("enviar_paquete sin destinatario valido")
        return

    recursos = _normalizar_recursos(recursos_raw)
    if not recursos:
        logger.warning("enviar_paquete sin recursos validos")
        return

    if not _puedo_enviar(recursos):
        logger.warning("Paquete bloqueado: no hay stock suficiente o rompe objetivo")
        return

    api_post_paquete(destinatario, recursos)
    _descontar_stock_local(recursos)

    # Notificar al destinatario que el paquete ha sido enviado
    resumen_enviado = ", ".join(f"{v} {k}" for k, v in recursos.items())
    recursos_esperados = _normalizar_recursos(args.get("recursos_esperados", {}))
    if recursos_esperados:
        resumen_esperado = ", ".join(f"{v} {k}" for k, v in recursos_esperados.items())
        cuerpo_confirmacion = (
            f"Te hemos enviado el paquete acordado ({resumen_enviado}). "
            f"Quedamos a la espera de que nos envies: {resumen_esperado}."
        )
    else:
        cuerpo_confirmacion = (
            f"Te hemos enviado el paquete acordado ({resumen_enviado}). "
            f"Quedamos a la espera de tu parte del intercambio."
        )
    api_post_carta(
        destinatario,
        "Paquete enviado - esperamos tu parte",
        cuerpo_confirmacion,
    )
    logger.info("Carta de confirmacion enviada a %s tras enviar paquete", destinatario)


def _normalizar_recursos(value: Any) -> dict[str, int]:
    """Normaliza mapa de recursos para tools."""
    if not isinstance(value, dict):
        return {}

    parsed: dict[str, int] = {}
    for material, cantidad in value.items():
        if not isinstance(material, str):
            continue
        if isinstance(cantidad, int):
            parsed[material] = cantidad
            continue
        if isinstance(cantidad, str) and cantidad.isdigit():
            parsed[material] = int(cantidad)

    normalized: dict[str, int] = {}
    for material, cantidad in parsed.items():
        if cantidad > 0:
            normalized[material] = cantidad
    return normalized


def _coerce_text(value: Any) -> str:
    """Convierte formatos de salida LLM a texto simple."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("value", "text", "content"):
            maybe = value.get(key)
            if isinstance(maybe, str) and maybe.strip():
                return maybe.strip()
        maybe_type = value.get("type")
        if (
            isinstance(maybe_type, str)
            and maybe_type.strip()
            and maybe_type.strip().lower()
            not in {"string", "object", "integer", "number", "boolean", "array"}
        ):
            return maybe_type.strip()
    return ""


def _texto_util(value: str) -> bool:
    """Filtra placeholders tipicos que no aportan contenido real."""
    normalized = value.strip().lower()
    return normalized not in {"", "string", "texto", "asunto", "cuerpo", "mensaje"}


def _puedo_enviar(envio: dict[str, int]) -> bool:
    """Comprueba que se puede enviar sin romper objetivo propio."""
    recursos = estado_global.get("recursos", {})
    objetivo = estado_global.get("objetivo", {})

    if not isinstance(recursos, dict) or not isinstance(objetivo, dict):
        return False

    for material, cantidad in envio.items():
        actual = int(recursos.get(material, 0))
        if actual < cantidad:
            return False

        if material != "oro" and (actual - cantidad) < int(objetivo.get(material, 0)):
            return False

    return True


def _descontar_stock_local(envio: dict[str, int]) -> None:
    """Descuenta del estado local tras enviar paquete."""
    recursos = estado_global.get("recursos")
    if not isinstance(recursos, dict):
        return

    for material, cantidad in envio.items():
        recursos[material] = max(0, int(recursos.get(material, 0)) - cantidad)

    _recalcular_estado_derivado()


def _recalcular_estado_derivado() -> None:
    """Recalcula faltantes y sobrantes en estado_global."""
    recursos = estado_global.get("recursos")
    objetivo = estado_global.get("objetivo")
    if not isinstance(recursos, dict) or not isinstance(objetivo, dict):
        return

    estado_global["faltantes"] = _calcular_faltantes(recursos, objetivo)
    estado_global["sobrantes"] = _calcular_sobrantes(recursos, objetivo)
    estado_global["objetivo_cumplido"] = not estado_global["faltantes"]


def _estado_dinamico(estado_base: dict[str, Any]) -> dict[str, Any]:
    """Fusiona estado de ciclo con el estado_global mas reciente."""
    _recalcular_estado_derivado()
    estado = dict(estado_base)
    recursos = estado_global.get("recursos")
    objetivo = estado_global.get("objetivo")
    faltantes = estado_global.get("faltantes")
    sobrantes = estado_global.get("sobrantes")

    if isinstance(recursos, dict):
        estado["recursos"] = dict(recursos)
    if isinstance(objetivo, dict):
        estado["objetivo"] = dict(objetivo)
    if isinstance(faltantes, dict):
        estado["faltantes"] = dict(faltantes)
    if isinstance(sobrantes, dict):
        estado["sobrantes"] = dict(sobrantes)
    estado["objetivo_cumplido"] = estado_global.get("objetivo_cumplido", False)
    return estado
