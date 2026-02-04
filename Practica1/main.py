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

# Estado global
estado_global = {
    "Recursos": {},
    "OfertasPendientes": []
}

# ---------------- API ----------------

def api_get_info():
    try:
        r = requests.get(f"{BASE_URL}/info")
        print("GET /info ->", r.status_code)
        return r.json()
    except Exception as e:
        print("Error GET /info:", e)
        return {}

def api_get_gente():
    try:
        r = requests.get(f"{BASE_URL}/gente")
        print("GET /gente ->", r.status_code)
        return r.json()
    except Exception as e:
        print("Error GET /gente:", e)
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
    try:
        r = requests.post(f"{BASE_URL}/carta", json=payload)
        print("POST /carta ->", r.status_code, "dest:", destinatario, "asunto:", asunto)
    except Exception as e:
        print("Error enviando carta:", e)

def api_post_paquete(destinatario, recursos):
    try:
        r = requests.post(f"{BASE_URL}/paquete/{destinatario}", json=recursos)
        print("POST /paquete ->", r.status_code, "dest:", destinatario, "recursos:", recursos)
    except Exception as e:
        print("Error enviando paquete:", e)

def api_delete_mail(uid):
    try:
        r = requests.delete(f"{BASE_URL}/mail/{uid}")
        print("DELETE /mail ->", r.status_code, "uid:", uid)
    except Exception as e:
        print("Error borrando mail:", e)

def registrar_identidad():
    try:
        r = requests.post(f"{BASE_URL}/alias/{MI_ALIAS}")
        print("POST /alias ->", r.status_code, "alias:", MI_ALIAS)
    except Exception as e:
        print("Error registrando alias:", e)

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

    print("Estado actual")
    print("Recursos:", recursos)
    print("Objetivo:", objetivo)
    print("Otros:", otros)
    print("Correos:", len(buzon))
    print("Ofertas pendientes:", len(estado_global["OfertasPendientes"]))

    return estado

# ---------------- PROMPT ----------------

def construir_system_prompt(estado, correo_actual=None):
    base = f"""
Eres un agente comerciante autónomo en un juego de recursos.
Tu alias es {MI_ALIAS}.

ESTADO ACTUAL
Recursos: {json.dumps(estado['Recursos'])}
Objetivo: {json.dumps(estado['Objetivo'])}
Materiales conocidos: {estado['Materiales']}
Otros jugadores: {estado['Otros']}

REGLAS
- Solo envía paquetes cuando haya un acuerdo confirmado.
- Nunca envíes recursos que no tienes.
- Solo negocia intercambios mediante propuestas.
- Recursos deben ser JSON válido.
"""

    if correo_actual:
        base += f"""
CORREO ACTUAL
{json.dumps(correo_actual, indent=2)}

Decide únicamente sobre este correo.
"""
    else:
        base += """
No hay correos. Propón un intercambio simple que te acerque a tu objetivo.
"""

    return base

# ---------------- VALIDACIÓN ----------------

def filtrar_recursos_validos(recursos_llm):
    mis_recursos = estado_global["Recursos"]

    if not isinstance(recursos_llm, dict):
        print("Formato de recursos inválido:", recursos_llm)
        return {}

    filtrados = {}
    for mat, cant in recursos_llm.items():
        if not isinstance(cant, int) or cant <= 0:
            continue
        if mat not in mis_recursos or mis_recursos[mat] < cant:
            continue
        filtrados[mat] = cant

    return filtrados

# ---------------- TOOLS ----------------

def guardar_oferta(destinatario, recursos_ofrecidos, recursos_deseados):
    oferta = {
        "destinatario": destinatario,
        "recursos_ofrecidos": json.dumps(recursos_ofrecidos),
        "recursos_deseados": json.dumps(recursos_deseados),
        "estado": "pendiente"
    }
    estado_global["OfertasPendientes"].append(oferta)
    print("Oferta guardada:", oferta)

    # Enviar carta automáticamente
    cuerpo = f"Propongo intercambio:\nOfrezco: {recursos_ofrecidos}\nDeseo: {recursos_deseados}"
    api_post_carta(destinatario, "Propuesta de intercambio", cuerpo)

def ejecutar_tool_call(tool_call):
    name = tool_call.function.name

    try:
        args = json.loads(tool_call.function.arguments)
    except Exception as e:
        print("Error parseando argumentos:", tool_call.function.arguments, e)
        return

    if name == "proponer_intercambio":
        recursos_ofrecidos = args.get("recursos_ofrecidos", {})
        recursos_deseados = args.get("recursos_deseados", {})
        destinatario = args.get("destinatario")
        if destinatario:
            guardar_oferta(destinatario, recursos_ofrecidos, recursos_deseados)

    elif name == "enviar_paquete":
        recursos = filtrar_recursos_validos(args.get("recursos", {}))
        if recursos:
            api_post_paquete(args["destinatario"], recursos)
        else:
            print("Paquete bloqueado por validación")

    elif name == "enviar_carta":
        api_post_carta(args["destinatario"], args["asunto"], args["cuerpo"])

    elif name == "eliminar_correo":
        api_delete_mail(args["uid"])

# ---------------- CICLO PRINCIPAL ----------------

def ciclo_principal():
    registrar_identidad()
    print("Agente iniciado")

    while True:
        print("\n--- NUEVO CICLO ---")
        estado = obtener_estado()
        buzon = estado["Buzon"] or {}

        for uid, correo in buzon.items():
            print("Procesando correo:", uid, "de:", correo.get("remi"))

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
                    print("No se tomó acción para este correo")

            except Exception as e:
                print("Error LLM (correo):", e)

        time.sleep(10)

# ---------------- TOOLS SCHEMA ----------------

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "proponer_intercambio",
            "parameters": {
                "type": "object",
                "properties": {
                    "destinatario": {"type": "string"},
                    "recursos_ofrecidos": {"type": "object"},
                    "recursos_deseados": {"type": "object"}
                },
                "required": ["destinatario", "recursos_ofrecidos", "recursos_deseados"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "enviar_paquete",
            "parameters": {
                "type": "object",
                "properties": {
                    "destinatario": {"type": "string"},
                    "recursos": {"type": "object", "additionalProperties": {"type": "integer"}}
                },
                "required": ["destinatario", "recursos"]
            }
        }
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
                    "cuerpo": {"type": "string"}
                },
                "required": ["destinatario", "asunto", "cuerpo"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "eliminar_correo",
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
