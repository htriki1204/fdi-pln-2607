import requests
import json

BASE_URL = "http://147.96.81.252:7719"


# --------------------------
# FUNCIONES DE ENDPOINTS CON PRINT
# --------------------------

def post_alias(nombre: str):
    """Crear alias"""
    url = f"{BASE_URL}/alias/{nombre}"
    response = requests.post(url)
    resultado = response.json()
    print(f"\nPOST /alias/{nombre} ->")
    print(json.dumps(resultado, indent=4))
    return resultado


def delete_alias(nombre: str):
    """Eliminar alias"""
    url = f"{BASE_URL}/alias/{nombre}"
    response = requests.delete(url)
    resultado = response.json()
    print(f"\nDELETE /alias/{nombre} ->")
    print(json.dumps(resultado, indent=4))
    return resultado


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
    resultado = response.text
    print(f"\nPOST /carta ->")
    print(json.dumps(resultado, indent=4))
    return resultado


def post_paquete(destinatario: str, paquete: dict):
    """
    Enviar un paquete al destinatario.
    """
    url = f"{BASE_URL}/paquete"
    params = {"dest": destinatario}
    response = requests.post(url, params=params, json=paquete)
    resultado = response.text
    print(f"\nPOST /paquete?dest={destinatario} ->")
    print(json.dumps(resultado, indent=4))
    return resultado



def get_info():
    """Obtener info completa"""
    url = f"{BASE_URL}/info"
    response = requests.get(url)
    resultado = response.json()
    print("\nGET /info ->")
    print(json.dumps(resultado, indent=4))
    return resultado


def get_gente():
    """Obtener lista de usuarios"""
    url = f"{BASE_URL}/gente"
    response = requests.get(url)
    resultado = response.json()
    print("\nGET /gente ->")
    print(json.dumps(resultado, indent=4))
    return resultado


# --------------------------
# MENÚ INTERACTIVO
# --------------------------

def menu():
    while True:
        print("\n===== MENÚ INTERACTIVO =====")
        print("1. Crear alias")
        print("2. Eliminar alias")
        print("3. Crear carta")
        print("4. Enviar paquete")
        print("5. Obtener info")
        print("6. Listar usuarios")
        print("0. Salir")
        opcion = input("Elige una opción: ")

        if opcion == "1":
            nombre = input("Nombre del alias: ")
            post_alias(nombre)

        elif opcion == "2":
            nombre = input("Nombre del alias a eliminar: ")
            delete_alias(nombre)

        elif opcion == "3":
            remi = input("Remitente: ")
            dest = input("Destinatario: ")
            asunto = input("Asunto: ")
            cuerpo = input("Cuerpo: ")
            id_ = input("ID de la carta: ")
            post_carta(remi, dest, asunto, cuerpo, id_)

        elif opcion == "4":
            destinatario = input("Destinatario del paquete: ")
            print("Introduce los recursos del paquete (enter = 0)")
            madera = int(input("Madera: ") or 0)
            tela = int(input("Tela: ") or 0)
            oro = int(input("Oro: ") or 0)
            trigo = int(input("Trigo: ") or 0)
            ladrillos = int(input("Ladrillos: ") or 0)
            piedra = int(input("Piedra: ") or 0)
            paquete = {
                "madera": madera,
                "oro": oro,
                "trigo": trigo,
                "tela": tela,
                "ladrillos": ladrillos,
                "piedra": piedra
            }
            post_paquete(destinatario, paquete)

        elif opcion == "5":
            get_info()

        elif opcion == "6":
            get_gente()

        elif opcion == "0":
            print("Saliendo...")
            break

        else:
            print("Opción no válida. Intenta de nuevo.")


if __name__ == "__main__":
    menu()