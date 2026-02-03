import requests
import json
import time
import uuid
import os
from openai import OpenAI

# CONFIGURACIÓN
CLIENT_LLM = OpenAI(
    base_url="http://127.0.0.1:11434",
    api_key="ollama"
)
MODEL_NAME = "qwen3-vl:8b"
BASE_URL = "http://147.96.81.252:8000"

# VARIABLES GLOBALES DEL AGENTE
MI_ALIAS = "grok"
ARCHIVO_REPUTACION = "reputacion.json"

class GestorReputacion:
    def __init__(self, archivo):
        self.archivo = archivo
        self.scores = self._cargar()

    def _cargar(self):
        if not os.path.exists(self.archivo):
            return {}
        try:
            with open(self.archivo, 'r') as f:
                return json.load(f)
        except:
            return {}

    def _guardar(self):
        with open(self.archivo, 'w') as f:
            json.dump(self.scores, f, indent=4)

    def obtener_score(self, jugador):
        return self.scores.get(jugador, 0)

    def actualizar_score(self, jugador, delta):
        actual = self.obtener_score(jugador)
        self.scores[jugador] = actual + delta
        self._guardar()
        return self.scores[jugador]

reputacion = GestorReputacion(ARCHIVO_REPUTACION)


def api_get_info():
    try:
        resp = requests.get(f"{BASE_URL}/info")
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

def api_get_gente():
    try:
        resp = requests.get(f"{BASE_URL}/gente")
        return resp.json()
    except Exception as e:
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
        requests.post(f"{BASE_URL}/carta", json=payload)
        return f"Carta enviada a {destinatario} con ID {carta_id}"
    except Exception as e:
        return f"Error enviando carta: {str(e)}"

def api_post_paquete(destinatario, recursos):
    full_pkg = {
        "madera": recursos.get("madera", 0),
        "oro": recursos.get("oro", 0),
        "trigo": recursos.get("trigo", 0),
        "tela": recursos.get("tela", 0),
        "ladrillos": recursos.get("ladrillos", 0),
        "piedra": recursos.get("piedra", 0)
    }
    try:
        requests.post(f"{BASE_URL}/paquete/{destinatario}", json=full_pkg)
        return f"Paquete enviado a {destinatario}: {json.dumps(full_pkg)}"
    except Exception as e:
        return f"Error enviando paquete: {str(e)}"

def api_delete_mail(uid):
    try:
        requests.delete(f"{BASE_URL}/mail/{uid}")
        return f"Correo {uid} eliminado."
    except Exception as e:
        return f"Error eliminando correo: {str(e)}"

def registrar_identidad():
    try:
        requests.post(f"{BASE_URL}/alias/{MI_ALIAS}")
        print(f"Identidad registrada: {MI_ALIAS}")
    except:
        pass

def construir_contexto():
    """Recopila toda la info necesaria para el prompt del usuario"""
    info = api_get_info()
    gente = api_get_gente()
    
    # Filtramos la gente para no enviarnos cartas a nosotros mismos
    otros_jugadores = [p for p in gente if p != MI_ALIAS]
    
    contexto = f"""
    --- ESTADO ACTUAL ---
    Mi Alias: {MI_ALIAS}
    Mis Recursos: {json.dumps(info.get('Recursos', {}))}
    Mis Objetivos: {json.dumps(info.get('Objetivo', {}))}
    
    --- ENTORNO SOCIAL ---
    Otros Jugadores Disponibles: {', '.join(otros_jugadores)}
    
    --- BUZÓN DE ENTRADA ---
    {json.dumps(info.get('Buzon', {}), indent=2)}
    """
    return contexto

def ejecutar_tool_call(tool_call):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    
    print(f"⚙️ Ejecutando Tool: {name}")
    
    if name == "enviar_carta":
        return api_post_carta(args["destinatario"], args["asunto"], args["cuerpo"])
    
    elif name == "enviar_paquete":
        return api_post_paquete(args["destinatario"], args["recursos"])
        
    elif name == "eliminar_correo":
        return api_delete_mail(args["uid"])
        
    return "Tool no encontrada"

def ciclo_principal():
    registrar_identidad()
    print("Agente Comercial Iniciado. Presiona Ctrl+C para detener.")
    
    while True:
        print("\n--- NUEVO CICLO ---")
        
        # 1. Obtener contexto global
        contexto_usuario = construir_contexto()
        print(f"📊 Estado analizado. Recursos actuales vs Objetivos.")

        # 2. Consultar al LLM
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analiza la situación y actúa:\n{contexto_usuario}"}
        ]

        try:
            response = CLIENT_LLM.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=tools_schema,
                tool_choice="auto"
            )
            
            msg = response.choices[0].message
            
            # 3. Verificar si el LLM quiere usar herramientas
            if msg.tool_calls:
                for tool in msg.tool_calls:
                    resultado = ejecutar_tool_call(tool)
                    print(f"   ↳ Resultado: {resultado}")
                    
            else:
                print("El agente decidió no tomar ninguna acción en este turno.")
                if msg.content:
                    print(f"   Pensamiento: {msg.content}")

        except Exception as e:
            print(f"Error en el ciclo LLM: {e}")

        # 4. Esperar antes del siguiente ciclo
        time.sleep(10)

if __name__ == "__main__":
    SYSTEM_PROMPT = """
    Eres un Agente Comerciante Autónomo en un juego de recursos.
    Tu ID es: """ + MI_ALIAS + """
    
    TUS TAREAS EN ORDEN DE PRIORIDAD:
    1. Revisa tu 'Buzon'. Si hay mensajes de oferta:
       - Analiza si el intercambio te beneficia (te acerca a tu 'Objetivo').
       - Si es bueno: Usa 'enviar_paquete' para pagar y 'enviar_carta' para confirmar. Luego 'eliminar_correo'.
       - Si es malo: Usa 'eliminar_correo' (o responde rechazando).
    
    2. Revisa tus 'Recursos' vs 'Objetivo':
       - Identifica qué te falta y qué te sobra.
    
    3. Si te falta algo y no tienes mensajes:
       - Elige un jugador de la lista 'Otros Jugadores'.
       - Usa 'enviar_carta' para proponer un intercambio (Ej: "Te doy X por Y").
    
    4. NO envíes paquetes a menos que sea un trato cerrado o una estrategia clara.
    """

    tools_schema = [
        {
            "type": "function",
            "function": {
                "name": "enviar_carta",
                "description": "Envia una carta/email a otro jugador.",
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
                "description": "Envia recursos (madera, oro, etc) a otro jugador.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "destinatario": {"type": "string"},
                        "recursos": {
                            "type": "object",
                            "properties": {
                                "madera": {"type": "integer"},
                                "oro": {"type": "integer"},
                                "trigo": {"type": "integer"},
                                "tela": {"type": "integer"},
                                "ladrillos": {"type": "integer"},
                                "piedra": {"type": "integer"}
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
                "description": "Borra un correo procesado usando su ID (uid).",
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

    ciclo_principal()