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
from settings import MI_ALIAS, MODEL_NAME, OLLAMA_HOST, PROACTIVE_COOLDOWN_SECONDS

logger = logging.getLogger("agente.negociacion")
llm = Client(host=OLLAMA_HOST)

estado_global: dict[str, Any] = {
    "recursos": {},
    "objetivo": {},
    "faltantes": {},
    "sobrantes": {},
    "ofertas_pendientes": [],
    "ultima_propuesta_ts": 0.0,
}

TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "enviar_carta",
            "description": "Envia una carta de negociacion a otro agente.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destinatario": {"type": "string"},
                    "asunto": {"type": "string"},
                    "cuerpo": {"type": "string"},
                },
                "required": ["destinatario", "asunto", "cuerpo"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "enviar_paquete",
            "description": "Envia recursos acordados a otro agente. Incluye recursos_esperados para indicar que esperas recibir a cambio.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destinatario": {"type": "string"},
                    "recursos": {
                        "type": "object",
                        "additionalProperties": {"type": "integer"},
                    },
                    "recursos_esperados": {
                        "type": "object",
                        "additionalProperties": {"type": "integer"},
                        "description": "Recursos que esperas recibir del destinatario a cambio.",
                    },
                },
                "required": ["destinatario", "recursos"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "no_accion",
            "description": "No hacer ninguna accion este turno.",
            "parameters": {
                "type": "object",
                "properties": {
                    "razon": {"type": "string"},
                },
                "required": ["razon"],
            },
        },
    },
]


def construir_estado(info: dict[str, Any], gente: list[Any]) -> dict[str, Any]:
    """Construye estado de ciclo y calcula faltantes/sobrantes."""
    recursos = parse_resource_map(info.get("Recursos"))
    objetivo = parse_resource_map(info.get("Objetivo"))
    buzon = parse_mailbox(info.get("Buzon"))
    otros = parse_other_aliases(gente)
    materiales = sorted(set(recursos) | set(objetivo))

    faltantes = _calcular_faltantes(recursos, objetivo)
    sobrantes = _calcular_sobrantes(recursos, objetivo)

    estado_global["recursos"] = recursos
    estado_global["objetivo"] = objetivo
    estado_global["faltantes"] = faltantes
    estado_global["sobrantes"] = sobrantes

    logger.info(
        "Estado | recursos=%s objetivo=%s faltantes=%s sobrantes=%s otros=%s correos=%s",
        recursos,
        objetivo,
        faltantes,
        sobrantes,
        otros,
        len(buzon),
    )

    return {
        "recursos": recursos,
        "objetivo": objetivo,
        "buzon": buzon,
        "otros": otros,
        "materiales": materiales,
        "faltantes": faltantes,
        "sobrantes": sobrantes,
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

    system_prompt = _construir_prompt_sistema(estado)

    sobrantes = estado.get("sobrantes", {})
    faltantes = estado.get("faltantes", {})
    sobra_ejemplo = next(iter(sobrantes), None)
    falta_ejemplo = next(iter(faltantes), None)

    if sobra_ejemplo and falta_ejemplo:
        user_prompt = (
            f"No tienes correos. Envia UNA carta a un agente. "
            f"Ofrece 1 {sobra_ejemplo} a cambio de 1 {falta_ejemplo}. "
            f"Usa enviar_carta."
        )
    else:
        user_prompt = (
            "No tienes correos y no tienes recursos sobrantes para ofrecer. "
            "Usa no_accion."
        )

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
        system_prompt = _construir_prompt_sistema(estado)

        # Construir listas legibles para reforzar en user prompt
        recursos_act = estado.get("recursos", {})
        objetivo_act = estado.get("objetivo", {})
        sobrantes_act = estado.get("sobrantes", {})
        faltantes_act = estado.get("faltantes", {})
        no_dar = [
            m
            for m in sorted(set(recursos_act) | set(objetivo_act))
            if recursos_act.get(m, 0) <= objetivo_act.get(m, 0)
        ]
        no_dar_txt = ", ".join(no_dar) if no_dar else "ninguno"
        sobra_txt = (
            ", ".join(f"{v} {k}" for k, v in sobrantes_act.items())
            if sobrantes_act
            else "nada"
        )
        falta_txt = (
            ", ".join(f"{v} {k}" for k, v in faltantes_act.items())
            if faltantes_act
            else "nada"
        )

        user_prompt = (
            f"{remitente} te ha enviado un correo.\n"
            f"Asunto: {asunto}\n"
            f"Cuerpo: {cuerpo}\n"
            f"\n"
            f"Recuerda:\n"
            f"- Te SOBRA: {sobra_txt}. Solo puedes dar de esto.\n"
            f"- Te FALTA: {falta_txt}. Esto es lo que quieres conseguir.\n"
            f"- NUNCA des: {no_dar_txt}.\n"
            f"\n"
            f"Si te pide algo que te SOBRA y te ofrece algo que te FALTA → envia paquete con enviar_paquete.\n"
            f"Si no te renta → haz contraoferta con enviar_carta: ofrece 1 de lo que te sobra por 1 de lo que te falta.\n"
            f"Usa UNA sola tool."
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


def _construir_prompt_sistema(estado: dict[str, Any]) -> str:
    """Construye prompt de sistema claro y simple para LLMs pequenos."""
    recursos = estado.get("recursos", {})
    objetivo = estado.get("objetivo", {})
    faltantes = estado.get("faltantes", {})
    sobrantes = estado.get("sobrantes", {})
    otros = estado.get("otros", [])

    # --- Lista dinamica de recursos que NUNCA se deben dar ---
    nunca_dar: list[str] = []
    for material in sorted(set(recursos) | set(objetivo)):
        actual = recursos.get(material, 0)
        necesito = objetivo.get(material, 0)
        if actual <= necesito:
            nunca_dar.append(material)

    nunca_dar_str = ", ".join(nunca_dar) if nunca_dar else "ninguno"
    sobrantes_str = (
        ", ".join(f"{v} {k}" for k, v in sobrantes.items()) if sobrantes else "nada"
    )
    faltantes_str = (
        ", ".join(f"{v} {k}" for k, v in faltantes.items()) if faltantes else "nada"
    )
    otros_str = ", ".join(otros) if otros else "nadie"

    return (
        f"Tu nombre es {MI_ALIAS}. Intercambias recursos con otros agentes.\n"
        f"\n"
        f"TIENES: {json.dumps(recursos, ensure_ascii=False)}\n"
        f"NECESITAS LLEGAR A: {json.dumps(objetivo, ensure_ascii=False)}\n"
        f"TE FALTA: {faltantes_str}\n"
        f"TE SOBRA (puedes dar): {sobrantes_str}\n"
        f"Otros agentes: {otros_str}\n"
        f"\n"
        f"=== PROHIBIDO ===\n"
        f"NUNCA des estos recursos: {nunca_dar_str}. Los necesitas.\n"
        f"\n"
        f"=== QUE HACER CUANDO RECIBES UN CORREO ===\n"
        f"Alguien te pide un recurso y te ofrece otro a cambio.\n"
        f"Paso 1: Mira si lo que te PIDEN es algo que te SOBRA ({sobrantes_str}).\n"
        f"Paso 2: Mira si lo que te OFRECEN es algo que te FALTA ({faltantes_str}).\n"
        f"Si las DOS cosas se cumplen → el trato te RENTA.\n"
        f"  → Usa enviar_paquete con recursos=lo que te piden y recursos_esperados=lo que te ofrecen. Se enviara automaticamente una carta avisando del paquete y lo que esperas recibir a cambio.\n"
        f"Si NO te renta → Usa enviar_carta para hacer una CONTRAOFERTA simple.\n"
        f"  Contraoferta: ofrece 1 de algo que te SOBRA y pide 1 de algo que te FALTA.\n"
        f"\n"
        f"=== QUE HACER SIN CORREOS ===\n"
        f"Envia UNA carta ofreciendo 1 recurso que te SOBRA a cambio de 1 que te FALTA.\n"
        f"\n"
        f"=== FORMATO ===\n"
        f"- Usa SOLO tools, no texto libre.\n"
        f"- enviar_carta: destinatario, asunto y cuerpo son texto.\n"
        f"- enviar_paquete: recursos es como {{\"tela\": 1}}. Incluye recursos_esperados con lo que te ofrecen a cambio.\n"
        f"- Si no hay buen trato posible, usa no_accion.\n"
    )


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
    for tool_call in tool_calls:
        name, args = _extraer_tool_call(tool_call)
        if not name:
            logger.warning("Tool call invalida")
            continue

        logger.info("Tool usada -> %s args=%s", name, args)

        if name == "no_accion":
            logger.info("Sin accion: %s", args.get("razon", "sin razon"))
            continue

        if name == "enviar_carta":
            _tool_enviar_carta(args)
            continue

        if name == "enviar_paquete":
            _tool_enviar_paquete(args)
            continue

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
