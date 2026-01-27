import requests
import json

BASE_URL = "http://147.96.81.252:8000"

def get_info():
    url = f"{BASE_URL}/info"

    response = requests.get(url)
    return response.json()


def post_alias(nombre: str):
    """Crear alias"""
    url = f"{BASE_URL}/alias/{nombre}"
    response = requests.post(url)
    return response.json()


def delete_alias(nombre: str):
    """Eliminar alias"""
    url = f"{BASE_URL}/alias/{nombre}"
    response = requests.delete(url)
    return response.json()


def post_carta(remi: str, dest: str, asunto: str, cuerpo: str, id_: str):
    """Crear carta"""
    url = f"{BASE_URL}/carta"
    payload = {
        "remi": remi,
        "dest": dest,
        "asunto": asunto,
        "cuerpo": cuerpo,
        "id": id_
    }
    response = requests.post(url, json=payload)
    return response.json()

def post_paquete(destinatario: str, paquete: dict):
    """Enviar un paquete a un destinatario"""
    url = f"{BASE_URL}/paquete/{destinatario}"
    response = requests.post(url, json=paquete)
    return response.json()

def get_gente():
    """Obtener lista de usuarios disponibles"""
    url = f"{BASE_URL}/gente"
    response = requests.get(url)
    return response.json()


if __name__ == "__main__":

    """
    alias = "trumpeta"

    alias_result = post_alias(alias)
    print(f"POST /alias/{alias} ->", alias_result)

        
    # Obtener lista de gente
    print("\nGET /gente ->")
    gente_result = get_gente()
    print(json.dumps(gente_result, indent=4))

    
    
    info_result = get_info()
    print(f"GET /get_info/ ->", info_result)
    print(json.dumps(info_result, indent=4))

    """
    paquete = {
        "madera": 0,
        "oro": 0,
        "trigo": 0
    }
    destinatario = ""
    print("\nPOST /paquete/{} ->".format(destinatario))
    paquete_result = post_paquete(destinatario, paquete)
    print(json.dumps(paquete_result, indent=4))

    

