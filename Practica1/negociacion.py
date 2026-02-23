"""Logica de negociacion: estado interno, prompts, LLM y tools."""

from __future__ import annotations

import json
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
from settings import MI_ALIAS, MODEL_NAME, OLLAMA_HOST

CLIENT_LLM = Client(host=OLLAMA_HOST)

# Estado interno del agente.
estado_global: dict[str, Any] = {
    "recursos": {},
    "ofertas_pendientes": [],
}

TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "proponer_intercambio",
            "parameters": {
                "type": "object",
                "properties": {
                    "destinatario": {"type": "string"},
                    "recursos_ofrecidos": {"type": "object"},
                    "recursos_deseados": {"type": "object"},
                },
                "required": ["destinatario", "recursos_ofrecidos", "recursos_deseados"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "enviar_paquete",
            "parameters": {
                "type": "object",
                "properties": {
                    "destinatario": {"type": "string"},
                    "recursos": {
                        "type": "object",
                        "additionalProperties": {"type": "integer"},
                    },
                },
                "required": ["destinatario", "recursos"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "enviar_carta",
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
            "name": "eliminar_correo",
            "parameters": {
                "type": "object",
                "properties": {"uid": {"type": "string"}},
                "required": ["uid"],
            },
        },
    },
]


def construir_estado(info: dict[str, Any], gente: list[Any]) -> dict[str, Any]:
    """Construye el estado actual del agente a partir de /info y /gente."""
    recursos = parse_resource_map(info.get("Recursos"))
    objetivo = parse_resource_map(info.get("Objetivo"))
    buzon = parse_mailbox(info.get("Buzon"))
    otros = parse_other_aliases(gente)
    materiales = sorted(set(recursos) | set(objetivo))

    estado_global["recursos"] = recursos

    print("Estado actual")
    print("Recursos:", recursos)
    print("Objetivo:", objetivo)
    print("Otros:", otros)
    print("Correos:", len(buzon))
    print("Ofertas pendientes:", len(estado_global["ofertas_pendientes"]))

    return {
        "recursos": recursos,
        "objetivo": objetivo,
        "buzon": buzon,
        "otros": otros,
        "materiales": materiales,
    }


def construir_system_prompt(
    estado: dict[str, Any],
    correo_actual: dict[str, Any] | None = None,
) -> str:
    """Genera prompt de sistema adaptado al estado actual."""
    prompt = f"""
Eres un agente comerciante autonomo en un juego de recursos.
Tu alias es {MI_ALIAS}.

ESTADO ACTUAL
Recursos: {json.dumps(estado["recursos"])}
Objetivo: {json.dumps(estado["objetivo"])}
Materiales conocidos: {estado["materiales"]}
Otros jugadores: {estado["otros"]}

REGLAS
- Solo envia paquetes cuando haya un acuerdo confirmado.
- Nunca envies recursos que no tienes.
- Solo negocia intercambios mediante propuestas.
- Recursos deben ser JSON valido.

Tarea:
- Analiza que recursos te sobran.
- Analiza que recursos del objetivo te faltan.
- Propon un intercambio razonable.
- No repitas ofertas pendientes.

Ofertas pendientes:
{json.dumps(estado_global["ofertas_pendientes"], indent=2)}
"""

    if correo_actual is not None:
        prompt += f"""

CORREO ACTUAL
{json.dumps(correo_actual, indent=2)}

Decide unicamente sobre este correo.
"""

    return prompt


def consultar_llm(messages: list[dict[str, str]]) -> list[Any]:
    """Consulta al modelo y devuelve las tool calls resultantes."""
    try:
        response = CLIENT_LLM.chat(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS_SCHEMA,
        )
    except Exception as exc:
        print(f"Error LLM: {exc}")
        return []

    message = (
        response.get("message")
        if isinstance(response, dict)
        else getattr(response, "message", None)
    )
    if message is None:
        return []

    tool_calls = (
        message.get("tool_calls")
        if isinstance(message, dict)
        else getattr(message, "tool_calls", None)
    )
    return tool_calls if isinstance(tool_calls, list) else []


def _parse_tool_call(tool_call: Any) -> tuple[str, dict[str, Any]]:
    """Normaliza una tool call a (nombre, argumentos)."""
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
            args = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            print(f"Error parseando argumentos tool call: {exc}")
            return name, {}
        return name, args if isinstance(args, dict) else {}

    return name, raw_args if isinstance(raw_args, dict) else {}


def _filtrar_recursos_validos(recursos: Any) -> dict[str, int]:
    """Filtra recursos que existen en inventario propio."""
    if not isinstance(recursos, dict):
        print("Formato de recursos invalido:", recursos)
        return {}

    propios = estado_global.get("recursos", {})
    validos: dict[str, int] = {}

    for material, cantidad in recursos.items():
        if (
            not isinstance(material, str)
            or not isinstance(cantidad, int)
            or cantidad <= 0
        ):
            continue
        if propios.get(material, 0) >= cantidad:
            validos[material] = cantidad

    return validos


def _guardar_oferta(
    destinatario: str, ofrecidos: dict[str, int], deseados: dict[str, int]
) -> None:
    """Guarda oferta pendiente y envia carta de propuesta."""
    oferta = {
        "destinatario": destinatario,
        "recursos_ofrecidos": ofrecidos,
        "recursos_deseados": deseados,
        "estado": "pendiente",
    }
    estado_global["ofertas_pendientes"].append(oferta)
    print("Oferta guardada:", oferta)

    cuerpo = f"Propongo intercambio:\nOfrezco: {ofrecidos}\nDeseo: {deseados}"
    api_post_carta(destinatario, "Propuesta de intercambio", cuerpo)


def ejecutar_tool_call(tool_call: Any) -> None:
    """Ejecuta una tool call devuelta por el modelo."""
    name, args = _parse_tool_call(tool_call)
    if not name:
        print("Tool call invalida, se ignora")
        return

    if name == "proponer_intercambio":
        destinatario = args.get("destinatario")
        ofrecidos = parse_resource_map(args.get("recursos_ofrecidos"))
        deseados = parse_resource_map(args.get("recursos_deseados"))
        if isinstance(destinatario, str) and destinatario:
            _guardar_oferta(destinatario, ofrecidos, deseados)
        return

    if name == "enviar_paquete":
        destinatario = args.get("destinatario")
        if not isinstance(destinatario, str) or not destinatario:
            print("enviar_paquete sin destinatario valido")
            return
        recursos_validos = _filtrar_recursos_validos(args.get("recursos", {}))
        if recursos_validos:
            api_post_paquete(destinatario, recursos_validos)
        else:
            print("Paquete bloqueado por validacion")
        return

    if name == "enviar_carta":
        destinatario = args.get("destinatario")
        asunto = args.get("asunto")
        cuerpo = args.get("cuerpo")
        if all(isinstance(v, str) and v for v in (destinatario, asunto, cuerpo)):
            api_post_carta(destinatario, asunto, cuerpo)
        else:
            print("enviar_carta con parametros invalidos")
        return

    if name == "eliminar_correo":
        uid = args.get("uid")
        if isinstance(uid, str) and uid:
            api_delete_mail(uid)
        else:
            print("eliminar_correo sin uid valido")
        return

    print(f"Tool desconocida: {name}")


def procesar_turno_sin_correos(estado: dict[str, Any]) -> None:
    """Procesa un ciclo proactivo cuando no hay correos."""
    print("Buzon vacio, generando oferta proactiva")
    prompt = construir_system_prompt(estado)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Genera una propuesta de intercambio beneficiosa."},
    ]
    for tool_call in consultar_llm(messages):
        ejecutar_tool_call(tool_call)


def procesar_correo(estado: dict[str, Any], uid: str, correo: dict[str, Any]) -> None:
    """Procesa un correo individual del buzon."""
    print("Procesando correo:", uid, "de:", correo.get("remi"))
    prompt = construir_system_prompt(estado, correo_actual=correo)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Procesa este correo."},
    ]
    tool_calls = consultar_llm(messages)
    if not tool_calls:
        print("No se tomo accion para este correo")
        return

    for tool_call in tool_calls:
        ejecutar_tool_call(tool_call)
