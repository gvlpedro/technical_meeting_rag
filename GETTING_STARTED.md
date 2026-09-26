# Primeros pasos

## USO POR LÍNEA DE COMANDOS

```bash
make up
```

### Tests

La suite rápida (llamadas al LLM mockeadas, segura para ejecutar en cualquier momento):

```bash
make test
```

### Comandos por fase

Generar preguntas de arquitectura para las sesiones de la fecha 2024-05-15 en Silver (modo interactivo):
```bash
make questions-arch DATE=20260515
```

Generar preguntas de data contracts para las sesiones de la fecha 2024-05-15 en Silver (modo interactivo):
```bash
make questions-data-contracts DATE=20260515
```

Proceso completo de clasificación (generación de preguntas / respuestas humanas / generación de ADR) para las sesiones de la fecha 2024-05-15 en Silver:
```bash
make clarify DATE=20260515 INTERACTIVE=1
```

Chatear con el agente (modo interactivo):
```bash
make chat
```

### TESTS DE ACTOR-CRITIC-BOSS

Asegúrate primero de que el esquema está actualizado:
```bash
make migrate
```

Test de actor-critic-boss para preguntas de arquitectura
```bash
make test-adr-acb
```

Test de actor-critic-boss para preguntas de data contracts
```bash
make test-data-contract-acb
```

Test de actor-critic-boss para preguntas de data contracts
```bash
make test-data-contract-acb
```

Test E2E automatizado — ingiere 5 transcripciones reales secuenciales (~10 min)
```bash
make test-gold-arch-evolution
```

## USO DEL FRONTEND

El frontend de Streamlit (`frontend/`) es la "Interfaz de usuario" descrita en el README raíz
— una pestaña para cada cosa: subir/clarificar una transcripción, navegar el historial de
arquitectura y cada ADR publicado, chatear con Gold, y (solo cuando `start_test_mode` está
activo) monitorizar resultados de tests y coste de LLM.

No hay un sistema de usuarios real — `frontend_users` en `app/config.py` es un listado fijo y
hardcodeado. Cada login se asocia a su propio **tenant**, que es lo que realmente aísla los
datos: un fichero que sube `pepe`, o un ADR que publica `pepe`, es invisible para `peter`, y
viceversa.

| Usuario  | Contraseña | Tenant  |
| -------- | ---------- | ------- |
| `pepe`   | `1234`     | `lidr`  |
| `peter`  | `123`      | `lotus` |
| `martin` | `123`      | `lotus` |

`martin` comparte tenant con `peter` (`lotus`), así que los dos ven y editan exactamente los
mismos datos — a diferencia de `pepe`, que está aislado en un tenant separado.

Un único comando arranca todo — Postgres, el backend y el frontend, todo en Docker:

```bash
make up
```

Abre **http://localhost:8522** una vez esté arriba (el frontend espera al healthcheck del
propio backend antes de arrancar, así que dale unos segundos en un `make up` en frío).

Para desarrollo local del frontend — recarga en vivo en cada guardado, sin reconstruir una
imagen Docker cada vez — ejecuta el backend por su cuenta y el frontend por separado:

```bash
make up          # o: uv run uvicorn app.main:app --reload   (backend sin docker)
make frontend    # Streamlit en http://localhost:8501, con recarga en vivo desde tu copia de trabajo
```

Si el backend corre en un sitio distinto a `http://localhost:8010`, apunta el frontend local
hacia él con `BACKEND_URL`:

```bash
BACKEND_URL=http://localhost:8010 make frontend
```

Inicia sesión como `pepe`/`1234` o `peter`/`123`, y luego:

1. **Input transcription** — escribe algo directamente en la caja de "Prompt", sube un `.txt`
   o `.md` para una reunión, o ambas cosas, y pulsa "Process and clarify" (se activa en cuanto
   una de las dos tiene contenido). Si el bucle de clarificación necesita una respuesta
   humana, la pestaña muestra ahí mismo las preguntas pendientes — respóndelas y envíalas para
   terminar la ejecución. Una vez estés conforme con el ADR generado, pulsa "Publish" — sus
   hechos quedan en Gold y son recuperables desde el Chat inmediatamente.
2. **Architecture history** — cada ADR publicado hasta ahora, en una tabla (pulsa "View ADR"
   para abrir uno en una pestaña nueva), más un diagrama Mermaid en vivo de la arquitectura
   actual construido directamente a partir de los propios componentes de Gold — cada nodo
   nombra el ADR que lo tocó por última vez.
3. **Chat with RAG** — pregunta sobre la arquitectura. Recupera de cada hecho publicado hasta
   ahora, limitado a tu propio tenant.
4. **Test monitor** (solo visible cuando `start_test_mode: true` en `app/config.py`, el valor
   por defecto) — los cuatro ficheros `testing_*/output/result.json`, más el uso de
   tokens/coste de cada llamada real al LLM (tabla `llm_costs` en Postgres), desglosado por
   tenant.
