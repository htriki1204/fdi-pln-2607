"""Plantillas de prompts y esquema de tools para la negociacion con IA."""

from __future__ import annotations

import json
import random
from typing import Any

from settings import MI_ALIAS

# ---------------------------------------------------------------------------
# Esquema de tools (function-calling) para el LLM
# ---------------------------------------------------------------------------

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
            "description": (
                "Envia recursos acordados a otro agente. "
                "Incluye recursos_esperados para indicar que esperas recibir a cambio."
            ),
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
                        "description": (
                            "Recursos que esperas recibir del destinatario a cambio."
                        ),
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


# ---------------------------------------------------------------------------
# Prompt de sistema
# ---------------------------------------------------------------------------


def construir_prompt_sistema(estado: dict[str, Any]) -> str:
    """Construye prompt de sistema claro y simple para LLMs pequenos."""
    recursos = estado.get("recursos", {})
    objetivo = estado.get("objetivo", {})
    faltantes = estado.get("faltantes", {})
    sobrantes = estado.get("sobrantes", {})
    otros = estado.get("otros", [])
    objetivo_cumplido = estado.get("objetivo_cumplido", False)

    sobrantes_str = _fmt_recursos(sobrantes)
    otros_str = ", ".join(otros) if otros else "nadie"

    if objetivo_cumplido:
        return _prompt_sistema_modo_oro(recursos, objetivo, sobrantes_str, otros_str)

    nunca_dar = _lista_nunca_dar(recursos, objetivo)
    nunca_dar_str = ", ".join(nunca_dar) if nunca_dar else "ninguno"
    faltantes_str = _fmt_recursos(faltantes)

    return f"""\
Tu nombre es {MI_ALIAS}. Intercambias recursos con otros agentes.

TIENES: {json.dumps(recursos, ensure_ascii=False)}
NECESITAS LLEGAR A: {json.dumps(objetivo, ensure_ascii=False)}
TE FALTA: {faltantes_str}
TE SOBRA (puedes dar): {sobrantes_str}
Otros agentes: {otros_str}

=== PROHIBIDO ===
NUNCA des estos recursos: {nunca_dar_str}. Los necesitas.

=== QUE HACER CUANDO RECIBES UN CORREO ===
Alguien te pide un recurso y te ofrece otro a cambio.
Paso 1: Mira si lo que te PIDEN es algo que te SOBRA ({sobrantes_str}).
Paso 2: Mira si lo que te OFRECEN es algo que te FALTA ({faltantes_str}).
Si las DOS cosas se cumplen → el trato te RENTA.
  → Usa enviar_paquete con recursos=lo que te piden y recursos_esperados=lo que \
te ofrecen. Se enviara automaticamente una carta avisando del paquete y lo que \
esperas recibir a cambio.
Si NO te renta → Usa enviar_carta para hacer una CONTRAOFERTA simple.
  Contraoferta: ofrece 1 de algo que te SOBRA y pide 1 de algo que te FALTA.
Todos los intercambios son 1x1: un solo recurso y cantidad 1 por cada parte.

=== QUE HACER SIN CORREOS ===
Envia UNA carta ofreciendo 1 recurso que te SOBRA a cambio de 1 que te FALTA.

=== FORMATO ===
- Usa SOLO tools, no texto libre.
- enviar_carta: destinatario, asunto y cuerpo son texto.
- enviar_paquete: recursos es como {{"tela": 1}}. Incluye recursos_esperados con \
lo que te ofrecen a cambio.
- Si no hay buen trato posible, usa no_accion.
- NUNCA propongas ni envies cantidades mayores que 1.
"""


# ---------------------------------------------------------------------------
# Prompts de usuario
# ---------------------------------------------------------------------------


def construir_user_prompt_proactivo(
    estado: dict[str, Any],
) -> str:
    """Prompt de usuario para turno proactivo (sin correos)."""
    sobrantes = estado.get("sobrantes", {})
    faltantes = estado.get("faltantes", {})
    otros = estado.get("otros", [])
    objetivo_cumplido = estado.get("objetivo_cumplido", False)

    destinatario = random.choice(otros) if otros else None

    if objetivo_cumplido:
        sobra_no_oro = {k: v for k, v in sobrantes.items() if k != "oro"}
        if sobra_no_oro and destinatario:
            material = random.choice(list(sobra_no_oro.keys()))
            return f"""\
No tienes correos. Envia UNA carta a {destinatario}. \
Ofrece 1 {material} a cambio de 1 oro. Usa enviar_carta."""
        return "No tienes recursos sobrantes para intercambiar por oro. Usa no_accion."

    sobra_ejemplo = random.choice(list(sobrantes.keys())) if sobrantes else None
    falta_ejemplo = random.choice(list(faltantes.keys())) if faltantes else None

    if sobra_ejemplo and falta_ejemplo and destinatario:
        return f"""\
No tienes correos. Envia UNA carta a {destinatario}. \
Ofrece 1 {sobra_ejemplo} a cambio de 1 {falta_ejemplo}. Usa enviar_carta."""

    return (
        "No tienes correos y no tienes recursos sobrantes para ofrecer. Usa no_accion."
    )


def construir_user_prompt_correo(
    remitente: str,
    asunto: str,
    cuerpo: str,
    estado: dict[str, Any],
) -> str:
    """Prompt de usuario para procesar un correo recibido."""
    recursos = estado.get("recursos", {})
    objetivo = estado.get("objetivo", {})
    sobrantes = estado.get("sobrantes", {})
    faltantes = estado.get("faltantes", {})
    objetivo_cumplido = estado.get("objetivo_cumplido", False)

    sobra_txt = _fmt_recursos(sobrantes)

    if objetivo_cumplido:
        sobra_no_oro = {k: v for k, v in sobrantes.items() if k != "oro"}
        sobra_no_oro_txt = _fmt_recursos(sobra_no_oro)
        return f"""\
{remitente} te ha enviado un correo.
Asunto: {asunto}
Cuerpo: {cuerpo}

Ya has cumplido tu objetivo. Ahora solo quieres ORO.
- Puedes dar: {sobra_no_oro_txt}.
- Solo aceptas oro a cambio.

Si te ofrece oro a cambio de algo que te sobra  enviar_paquete.
Si no haz contraoferta con enviar_carta: ofrece materiales sobrantes por oro.
Usa UNA sola tool."""

    no_dar = [
        m
        for m in sorted(set(recursos) | set(objetivo))
        if recursos.get(m, 0) <= objetivo.get(m, 0)
    ]
    no_dar_txt = ", ".join(no_dar) if no_dar else "ninguno"
    falta_txt = _fmt_recursos(faltantes)

    return f"""\
{remitente} te ha enviado un correo.
Asunto: {asunto}
Cuerpo: {cuerpo}

Recuerda:
- Te SOBRA: {sobra_txt}. Solo puedes dar de esto.
- Te FALTA: {falta_txt}. Esto es lo que quieres conseguir.
- NUNCA des: {no_dar_txt}.

Si te pide algo que te SOBRA y te ofrece algo que te FALTA → envia paquete con enviar_paquete.
Si no te renta → haz contraoferta con enviar_carta: ofrece 1 de lo que te sobra \
por 1 de lo que te falta.
Intercambia siempre 1x1: una unidad y un solo recurso por lado.
Usa UNA sola tool."""


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------


def _prompt_sistema_modo_oro(
    recursos: dict[str, int],
    objetivo: dict[str, int],
    sobrantes_str: str,
    otros_str: str,
) -> str:
    """Prompt de sistema cuando el objetivo ya esta cumplido (modo oro)."""
    sobra_no_oro = {
        k: v for k, v in recursos.items() if k != "oro" and v > objetivo.get(k, 0)
    }
    intercambiable_str = _fmt_recursos(sobra_no_oro)

    return f"""\
Tu nombre es {MI_ALIAS}. Intercambias recursos con otros agentes.

*** HAS CUMPLIDO TODOS TUS OBJETIVOS. ***
Ahora tu UNICO objetivo es conseguir la mayor cantidad de ORO posible.

TIENES: {json.dumps(recursos, ensure_ascii=False)}
OBJETIVO (ya cumplido): {json.dumps(objetivo, ensure_ascii=False)}
Materiales que puedes intercambiar por oro: {intercambiable_str}
Otros agentes: {otros_str}

=== REGLAS ===
- NUNCA des oro. Solo RECIBES oro.
- Ofrece materiales sobrantes a cambio de oro.
- Si alguien te ofrece oro a cambio de materiales que te sobran → acepta.
- Si no te ofrecen oro → haz contraoferta pidiendo oro.
- Intercambios siempre 1x1: un recurso y cantidad 1.

=== FORMATO ===
- Usa SOLO tools, no texto libre.
- enviar_carta: destinatario, asunto y cuerpo son texto.
- enviar_paquete: recursos es como {{"tela": 1}}. Incluye recursos_esperados \
con lo que esperas a cambio ({{"oro": 1}}).
- Si no hay buen trato posible, usa no_accion.
"""


def _lista_nunca_dar(recursos: dict[str, int], objetivo: dict[str, int]) -> list[str]:
    """Materiales cuyo stock actual no supera el objetivo."""
    return [
        m
        for m in sorted(set(recursos) | set(objetivo))
        if recursos.get(m, 0) <= objetivo.get(m, 0)
    ]


def _fmt_recursos(mapa: dict[str, int]) -> str:
    """Formatea un mapa de recursos como texto legible."""
    if not mapa:
        return "nada"
    return ", ".join(f"{v} {k}" for k, v in mapa.items())
