# Practica 1 PLN - Agente en butler (Grupo 7)

- Hamza Triki
- Alvaro Ferreno Iglesias

## Idea general
El agente busca completar su objetivo de materiales negociando por cartas y paquetes con otros agentes en Butler.

Estrategia base:
- Si recibe una oferta favorable, envia paquete directamente.
- Tras enviar paquete, envia carta de confirmacion con lo que espera recibir.
- Si la oferta no conviene, propone contraoferta simple 1x1.
- Si no hay correos, puede lanzar propuesta proactiva (con cooldown, para no saturar buzones de otros agentes).

## Ejecucion
1. Arrancar Butler.
2. Tener Ollama levantado con un modelo descargado.
3. Ejecutar el agente:

```bash
uv run --env-file .env main.py
```

Para lanzar varias instancias:

```bash
FDI_PLN__ALIAS=agente_1 FDI_PLN__LOG_FILE=logs/agente_1.log uv run --env-file .env main.py
FDI_PLN__ALIAS=agente_2 FDI_PLN__LOG_FILE=logs/agente_2.log uv run --env-file .env main.py
FDI_PLN__ALIAS=agente_3 FDI_PLN__LOG_FILE=logs/agente_3.log uv run --env-file .env main.py
```

## Ejecucion desde wheel (`.whl`)
Instalar el .whl y ejecutar el binario `fdi-pln-2607-p1` (sin `uv run`):

```bash
uv tool install /ruta/al/fdi_pln_2607_p1-1.0-py3-none-any.whl
```

Definir variables de entorno (ejemplo minimo):

```bash
export FDI_PLN__BUTLER_ADDRESS=127.0.0.1:7719
export FDI_PLN__OLLAMA_HOST=http://127.0.0.1:11434
export FDI_PLN__MODEL=ministral-3:8b
export FDI_PLN__ALIAS=agenteGrupo7
```

Lanzar agente:

```bash
fdi-pln-2607-p1
```

## Variables `.env`
Valores actuales del proyecto y fallback en codigo (`settings.py`):

| Variable | Valor en `.env` | Fallback en codigo | Para que sirve |
|---|---:|---:|---|
| `FDI_PLN__BUTLER_ADDRESS` | `127.0.0.1:7719` | `147.96.84.134:7719` | Host:puerto de Butler. |
| `FDI_PLN__ALIAS` | `claude_code` | `agenteGrupo7` | Identidad del agente en Butler. |
| `FDI_PLN__OLLAMA_HOST` | `http://127.0.0.1:11434` | `http://127.0.0.1:11434` | Endpoint de Ollama. |
| `FDI_PLN__MODEL` | `ministral-3:8b` | `llama3.2:latest` | Modelo LLM para decidir tools. |
| `FDI_PLN__REQUEST_TIMEOUT` | `10` | `10` | Timeout HTTP contra Butler (segundos). |
| `FDI_PLN__CYCLE_SECONDS` | `10` | `10` | Espera entre ciclos del bucle principal. |
| `FDI_PLN__WAIT_WITHOUT_PEERS_SECONDS` | `10` | `10` | Espera cuando no hay otros agentes conectados. |
| `FDI_PLN__PROACTIVE_COOLDOWN_SECONDS` | `45` | `45` | Enfriamiento entre propuestas proactivas sin correos. |
| `FDI_PLN__MIN_GOLD_RESERVE` | `5` | - | Reservada para futuras reglas (actualmente no se usa en codigo). |
| `FDI_PLN__LOG_LEVEL` | `INFO` | `INFO` | Nivel de logging. |
| `FDI_PLN__LOG_FILE` | `logs/agente.log` | `logs/agente.log` | Archivo de log. |
| `FDI_PLN__LOG_MAX_BYTES` | `2000000` | `2000000` | Tamano maximo por archivo de log rotativo. |
| `FDI_PLN__LOG_BACKUP_COUNT` | `3` | `3` | Numero de backups de logs rotativos. |

## Arquitectura
Arquitectura modular y simple:

- `main.py`:
  bucle principal por ciclos (`info` + `gente`), decide si procesa correos o accion proactiva.
- `api_butler.py`:
  adaptador HTTP a Butler (`/alias`, `/info`, `/gente`, `/carta`, `/paquete`, `/mail/{uid}`) y parseo defensivo.
- `negociacion.py`:
  logica de negociacion + integracion con Ollama tools.
  valida tools, normaliza argumentos, bloquea envios que rompen objetivo propio y mantiene estado.
- `prompts.py`:
  define `TOOLS_SCHEMA` y construye prompts (sistema + usuario) para responder a correo y modo proactivo (enviar correos).
- `settings.py`:
  carga centralizada de configuracion por variables de entorno (necesario tener un .env o hardcodear variables fallback).
- `logger_config.py`:
  logging en consola + archivo.

## Diseno de prompts
La separacion principal es:

- `System prompt`: reglas de negocio (que esta permitido, que no, criterio de oferta favorable, formato y una sola tool).
- `User prompt`: contexto del turno (correo recibido + estado actual).

Objetivo de este diseño:
- reducir ambiguedad,
- evitar contradicciones entre instrucciones,
- hacer que el modelo devuelva una sola tool call util y ejecutable.

## Comportamiento esperado del agente
- Acepta tratos favorables enviando paquete directamente.
- Si recibe algo como "ya te he enviado X, enviame Y", lo trata como aceptacion y responde con `enviar_paquete` cuando es viable.
- Si no es viable o no conviene, usa `enviar_carta` (contraoferta) o `no_accion`.
- Nunca envia recursos que dejen su stock por debajo del objetivo propio.
