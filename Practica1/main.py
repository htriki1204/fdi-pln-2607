import requests
import json
import time
import uuid
from openai import OpenAI

# CONFIGURACIÓN
CLIENT_LLM = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama"
)
MODEL_NAME = "llama3.2:3b"
BASE_URL = "http://147.96.81.252:7719"

MI_ALIAS = "grok"

# Estado global para validaciones locales
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

def api_post_carta(destinatario, asunto, cuerpo):
    carta_id = str(uuid.uuid4())[:8]
    payload = {
        "remi": MI_ALIAS,
        "dest": destinatario,
        "asunto": asunto,
        "cuerpo": cuerpo,
        "id": carta_id,
        "fecha": time.strftime("%Y-%m-%d %H:%M")
    }
    requests.post(f"{BASE_URL}/carta", json=payload)

def api_post_paquete(destinatario, recursos):
    requests.post(f"{BASE_URL}/paquete/{destinatario}", json=recursos)

def api_delete_mail(uid):
    requests.delete(f"{BASE_URL}/mail/{uid}")

def registrar_identidad():
    requests.post(f"{BASE_URL}/alias/{MI_ALIAS}")

# ---------------- ESTADO ----------------

def obtener_estado():
    info = api_get_info()
    gente = api_get_gente()
    otros = [p for p in gente if p != MI_ALIAS]

    recursos = info.get("Recursos", {})
    objetivo = info.get("Objetivo", {})
    buzon = info.get("Buzon", {})

    materiales_actuales = list(set(recursos.keys()) | set(objetivo.keys()))

    estado = {
        "Recursos": recursos,
        "Objetivo": objetivo,
        "Buzon": buzon,
        "Otros": otros,
        "Materiales": materiales_actuales
    }

    estado_global["Recursos"] = recursos
    return estado

# ---------------- PROMPT DINÁMICO ----------------

def construir_system_prompt(estado, correo_actual=None):
    base = f"""
Eres un agente comerciante autónomo en un juego de recursos.
Tu alias es {MI_ALIAS}.

--- ESTADO ACTUAL ---
Mis Recursos: {json.dumps(estado['Recursos'])}
Mi Objetivo: {json.dumps(estado['Objetivo'])}
Materiales conocidos actualmente: {estado['Materiales']}
Otros jugadores: {estado['Otros']}
"""

    if correo_actual:
        base += f"""
--- PROCESANDO ESTE CORREO ---
{json.dumps(correo_actual, indent=2)}

REGLAS PARA ESTE CORREO:
- Decide SOLO sobre este correo.
- Si te ofrecen recursos que te ayudan a cumplir el Objetivo:
    - Acepta el trato:
        - enviar_paquete
        - enviar_carta confirmando el acuerdo
        - eliminar_correo
- Si no te interesa:
    - Rechaza educadamente o pide más info
    - eliminar_correo

REGLA SOBRE MATERIALES DESCONOCIDOS:
- Si te ofrecen un material que no aparece en tu Objetivo:
    - No lo aceptes directamente
    - Pide más información
"""
    else:
        base += """
--- NO HAY CORREOS ---
REGLAS:
- Detecta qué recursos te faltan para cumplir el Objetivo.
- Detecta qué recursos te sobran.
- Propón un intercambio simple 1 a 1 con otro jugador usando enviar_carta.
- No envíes paquetes sin un acuerdo previo por carta.
"""

    return base

# ---------------- VALIDACIÓN ----------------

def filtrar_recursos_validos(recursos_a_enviar):
    mis_recursos = estado_global["Recursos"]

    filtrados = {}
    for mat, cant in recursos_a_enviar.items():
        if cant > 0 and mis_recursos.get(mat, 0) >= cant:
            filtrados[mat] = cant
        else:
            print(f" Recurso inválido o insuficiente: {mat} -> {cant}")

    return filtrados

# ---------------- TOOLS ----------------

def ejecutar_tool_call(tool_call):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    print(f" Ejecutando tool: {name}")

    if name == "enviar_carta":
        api_post_carta(args["destinatario"], args["asunto"], args["cuerpo"])

    elif name == "enviar_paquete":
        recursos_raw = args["recursos"]
        if isinstance(recursos_raw, str):
            try:
                recursos_raw = json.loads(recursos_raw)
            except Exception as e:
                print("No se pudo parsear recursos:", recursos_raw, e)
                return

        recursos_filtrados = filtrar_recursos_validos(recursos_raw)
        if recursos_filtrados:
            api_post_paquete(args["destinatario"], recursos_filtrados)
        else:
            print(" No se envió paquete: recursos inválidos.")
        if recursos_filtrados:
            api_post_paquete(args["destinatario"], recursos_filtrados)
        else:
            print("No se envió paquete: recursos inválidos.")

    elif name == "eliminar_correo":
        api_delete_mail(args["uid"])

# ---------------- CICLO ----------------

def ciclo_principal():
    registrar_identidad()
    print("Agente Comercial iniciado")

    while True:
        estado = obtener_estado()
        buzon = estado["Buzon"] or {}

        # Procesar correos uno a uno
        for uid, correo in buzon.items():
            system_prompt = construir_system_prompt(estado, correo)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Procesa este correo."}
            ]

            try:
                response = CLIENT_LLM.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=tools_schema,
                    tool_choice="auto"
                )

                msg = response.choices[0].message

                if msg.tool_calls:
                    for tool in msg.tool_calls:
                        ejecutar_tool_call(tool)
                else:
                    print("No se realizó acción sobre este correo.")

            except Exception as e:
                print("Error LLM (correo):", e)

        # Si no hay correos, proponer trade
        if not buzon:
            system_prompt = construir_system_prompt(estado)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Propón un intercambio beneficioso."}
            ]

            try:
                response = CLIENT_LLM.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=tools_schema,
                    tool_choice="auto"
                )

                msg = response.choices[0].message

                if msg.tool_calls:
                    for tool in msg.tool_calls:
                        ejecutar_tool_call(tool)
                else:
                    print("No se propuso ningún intercambio.")

            except Exception as e:
                print("Error LLM (propuesta):", e)

        time.sleep(10)

# ---------------- TOOLS SCHEMA ----------------

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "enviar_carta",
            "description": "Envía una carta a otro jugador",
            "parameters": {
                "type": "object",
                "properties": {
                    "destinatario": {"type": "string"},
                    "asunto": {"type": "string"},
                    "cuerpo": {"type": "string"}
                },
                "required": ["destinatario", "asunto", "cuerpo"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "enviar_paquete",
            "description": "Envía recursos dinámicos a otro jugador",
            "parameters": {
                "type": "object",
                "properties": {
                    "destinatario": {"type": "string"},
                    "recursos": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "integer"
                        }
                    }
                },
                "required": ["destinatario", "recursos"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "eliminar_correo",
            "description": "Elimina un correo del buzón",
            "parameters": {
                "type": "object",
                "properties": {
                    "uid": {"type": "string"}
                },
                "required": ["uid"]
            }
        }
    }
]

# ---------------- MAIN ----------------

if __name__ == "__main__":
    ciclo_principal()
