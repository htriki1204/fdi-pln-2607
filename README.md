# Practica 4 PLN - Buscador del Quijote (Grupo 7)

- Hamza Triki
- Alvaro Ferreno Iglesias

## Descripcion
Aplicacion de terminal para recuperar informacion del Quijote en tres modos:

- busqueda clasica por lemas, stopwords y ranking TF-IDF
- busqueda semantica por embeddings
- RAG con recuperacion clasica y semantica

La entrega principal esta en [Practica4]

## Ejecucion desde el repositorio
```bash
cd Practica4
uv run fdi-pln-2607-p4
```

Tambien se puede arrancar con modo y consulta inicial:

```bash
cd Practica4
uv run fdi-pln-2607-p4 --modo clasica "molinos de viento"
uv run fdi-pln-2607-p4 --modo embeddings "don quijote y los molinos"
uv run fdi-pln-2607-p4 --modo rag "don quijote y los molinos"
```

## Modelos necesarios
La busqueda clasica no necesita IA.

La busqueda por embeddings y el RAG requieren `Ollama` levantado:

```bash
ollama serve
```

Modelos usados por defecto:

- embeddings: `nomic-embed-text:latest`
- RAG: `llama3.2:3b`

Descarga:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

Se pueden cambiar con variables de entorno:

```bash
FDI_PLN_P4_EMBED_MODEL=nomic-embed-text:latest \
FDI_PLN_P4_RAG_MODEL=llama3.2:3b \
uv run fdi-pln-2607-p4
```

## Datos y preprocesado
El wheel incluye:

- el corpus `2000-h.htm`
- una cache de embeddings preprocesada

El codigo puede regenerar los embeddings bajo demanda si la cache no existe o si cambia el modelo.

## Nota sobre el corpus
En [Practica4/documentation.md](/Users/alewar/Documents/Universidad/Cuarto/pln/fdi-pln-2607/Practica4/documentation.md) se documenta el recorte manual del HTML de Gutenberg para dejar solo los bloques relevantes del Quijote.

## Historico

- Practica 1: agente para Butler en [Practica1]
- Practica 3: codificador/decodificador [Practica3]