import requests
import json
import time
import uuid
from openai import OpenAI

# ---------------- CONFIG ----------------

CLIENT_LLM = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama"
)

MODEL_NAME = "qwen3:8b"
BASE_URL = "http://147.96.81.252:7719"
MI_ALIAS = "tung tung tung sahur"

estado_global = {
    "Recursos": {}
}

# ---------------- API ----------------

def api_get_info():
    try:
        return requests.get(f"{BASE_URL}/info").json()
    except:
        return {}

def api_get_gente():
    try:
        return requests.get(f"{BASE_URL}/gente").json()
    except:
        return []

def api_post_carta(dest, asunto, cuerpo):
    payload = {
        "remi": MI_ALIAS,
        "dest": dest,
        "asunto": asunto,
        "cuerpo": cuerpo,
        "id": str(uuid.uuid4())[:8],
        "fecha": time.strftime("%Y-%m-%d %H:%M")
    }
    requests.post(f"{BASE_URL}/carta", json=payload)

def api_post_paquete(dest, recursos):
    requests.post(f"{BASE_URL}/paquete/{dest}", json=recursos)

def api_delete_mail(uid):
    requests.delete(f"{BASE_URL}/mail/{uid}")

# ---------------- ESTADO ----------------

def obtener_estado():
    info = api_get_info()
    gente = api_get_gente()

    recursos = info.get("Recursos", {})
    objetivo = info.get("Objetivo", {})
    buzon = info.get("Buzon", {})

    estado_global["Recursos"] = recursos

    return {
        "Recursos": recursos,
        "Objetivo": objetivo,
        "Buzon": buzon,
        "Otros": [g for g in gente if g != MI_ALIAS]
    }

# ---------------- PROMPT ----------------

def construir_prompt(estado, correo=None):
    base = f"""
Eres un agente comerciante autónomo.
Alias: {MI_ALIAS}

ESTADO:
Recursos actuales: {json.dumps(estado["Recursos"])}
Objetivo: {json.dumps(estado["Objetivo"])}
Otros jugadores: {estado["Otros"]}

ACCIONES POSIBLES:
- proponer_trato
- aceptar_trato
- no_hacer_nada

REGLAS:
- Nunca envies recursos que tenemos en nuestros recursos objetivos
- Solo proponemos tratos 1:1 
- Aceptamos tratos de cualquier tamaño n:m si nos interesa
- Nunca envíes recursos que no tienes
- Acepta solo tratos que te acerquen al objetivo
- Sé explícito y claro en los mensajes

BLACKLIST: No respondas ni hagas nada con mensajes de AmarNoEsDelito
"""

    if correo:
        base += f"""
CORREO RECIBIDO:
{json.dumps(correo, indent=2)}

Decide qué hacer con este correo.
"""
    else:
        base += """
No hay correos.
Decide si quieres proponer un trato nuevo o no hacer nada.
"""

    return base

# ---------------- VALIDACIÓN ----------------

def filtrar_recursos_validos(recursos):
    validos = {}
    for k, v in recursos.items():
        if isinstance(v, int) and v > 0:
            if estado_global["Recursos"].get(k, 0) >= v:
                validos[k] = v
    return validos

# ---------------- TOOLS ----------------

def ejecutar_tool(tool_call):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    if name == "proponer_trato":
        cuerpo = (
            "PROPUESTA_TRATO\n"
            f"ofrezco: {json.dumps(args['ofrezco'])}\n"
            f"quiero: {json.dumps(args['quiero'])}"
        )
        api_post_carta(args["destinatario"], "Propuesta de trato", cuerpo)

    elif name == "aceptar_trato":
        enviados = filtrar_recursos_validos(args["recursos_enviados"])
        if not enviados:
            return

        api_post_paquete(args["destinatario"], enviados)

        cuerpo = (
            "TRATO_CONFIRMADO\n"
            f"enviado: {json.dumps(args['recursos_enviados'])}\n"
            f"esperado: {json.dumps(args['recursos_esperados'])}"
        )

        api_post_carta(args["destinatario"], "Trato aceptado", cuerpo)

# ---------------- TOOLS SCHEMA ----------------

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "proponer_trato",
            "parameters": {
                "type": "object",
                "properties": {
                    "destinatario": {"type": "string"},
                    "ofrezco": {"type": "object"},
                    "quiero": {"type": "object"}
                },
                "required": ["destinatario", "ofrezco", "quiero"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "aceptar_trato",
            "parameters": {
                "type": "object",
                "properties": {
                    "destinatario": {"type": "string"},
                    "recursos_enviados": {"type": "object"},
                    "recursos_esperados": {"type": "object"}
                },
                "required": [
                    "destinatario",
                    "recursos_enviados",
                    "recursos_esperados"
                ]
            }
        }
    }
]

# ---------------- CICLO PRINCIPAL ----------------

def ciclo_principal():
    print("Agente iniciado")

    while True:
        estado = obtener_estado()
        buzon = estado["Buzon"]

        # --- Procesar correos ---
        for uid, correo in buzon.items():
            print(f"--- CORREO {uid} ---")
            print(json.dumps(correo, indent=2))  # imprime el correo completo

            try:
                prompt = construir_prompt(estado, correo)
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Decide la acción."}
                ]

                resp = CLIENT_LLM.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=tools_schema,
                    tool_choice="auto"
                )
                msg = resp.choices[0].message
                print("CONTENT cuando hay correo:", msg.content)
                print("TOOL CALLS cuando hay correo:", msg.tool_calls)

                msg = resp.choices[0].message
                if msg.tool_calls:
                    for tool in msg.tool_calls:
                        ejecutar_tool(tool)

            finally:
                # 🔥 SIEMPRE borrar el correo tras procesarlo
                api_delete_mail(uid)

        # --- Si no hay correos, pensar proactivamente ---
        if not buzon:
            prompt = construir_prompt(estado)
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Decide la acción."}
            ]

            resp = CLIENT_LLM.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=tools_schema,
                tool_choice="auto"
            )
            msg = resp.choices[0].message
            print("CONTENT cuando no hay correo:", msg.content)
            print("TOOL CALLS cuando no hay correo:", msg.tool_calls)

            msg = resp.choices[0].message
            if msg.tool_calls:
                for tool in msg.tool_calls:
                    ejecutar_tool(tool)

        time.sleep(10)

# ---------------- MAIN ----------------

if __name__ == "__main__":
    ciclo_principal()


"""

{
  "remi": "AmarNoEsDelito",
  "dest": "tung tung tung sahur",
  "asunto": "Propuesta: mi 1 queso por tu 1 trigo",
  "cuerpo": "Hola tung tung tung sahur, soy AmarNoEsDelito. Te propongo un intercambio: yo te doy 1 queso y t\u00fa me das 1 trigo. Si aceptas, responde 'acepto el trato'. Si no te conviene, responde 'no me conviene'. Saludos, AmarNoEsDelito",
  "id": "2e4bc2ee-a9cc-47d0-b318-40e5eeb4414a",
  "fecha": "2026-02-10T17:55:11.435420"
}
CONTENT cuando hay correo: 
TOOL CALLS cuando hay correo: [ChatCompletionMessageFunctionToolCall(id='call_2zn5urpd', function=Function(arguments='{"destinatario":"AmarNoEsDelito","recursos_enviados":{"queso":1},"recursos_esperados":{"trigo":1}}', name='aceptar_trato'), type='function', index=0)]
"""
