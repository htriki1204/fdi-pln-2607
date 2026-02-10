import requests

BASE_URL = "http://147.96.81.252:7719"

def borrar_todo_el_buzon():
    try:
        info = requests.get(f"{BASE_URL}/info").json()
        buzon = info.get("Buzon", {})

        print(f"Correos encontrados: {len(buzon)}")

        for uid in buzon.keys():
            r = requests.delete(f"{BASE_URL}/mail/{uid}")
            print(f"Borrado correo {uid} -> {r.status_code}")

        print("Buzón limpio ✅")

    except Exception as e:
        print("Error borrando buzón:", e)

if __name__ == "__main__":
    borrar_todo_el_buzon()
