
def main():    
    import requests
    url = "http://147.96.81.252:8000"

    nombre = "alvaroYhamza"
    response = requests.post(f"{url}/alias/{nombre}")

    if response.status_code == 200:
        print(response.json())
    else:
        print("Error:", response.status_code, response.text)


if __name__ == "__main__":
    main()
