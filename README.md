# technical_meeting_rag

Las reuniones técnicas sobre la arquitectura de un sistema están dispersas en muchas conversaciones separadas, cada una contada desde un ángulo distinto — un ingeniero backend describiendo un servicio, un ingeniero de datos describiendo un pipeline, un ingeniero frontend describiendo una vista — y cada una capturando solo una instantánea parcial e informal de la verdad en ese momento.

Este proyecto construye una **Medallion RAG Architecture** que ingiere transcripciones y una versión más clarificada de estas como base de conocimiento consultable.

# Objetivo

Para cada componente de arquitectura mencionado en las reuniones, clarificar **qué es**, si es **nuevo**, una **evolución** de algo ya conocido, o si permanece **sin cambios**, y producir las dos cosas que Gold genera: un **Architecture Decision Record (ADR)** por cada evolución — capturando el contexto que la motivó, las alternativas consideradas, los trade-offs aceptados, y la decisión en sí — y un **registro versionado de cada componente**, en vez de un resumen plano y sin fecha de lo que se dijo.

# Problemas a resolver

* **Conocimiento fragmentado:** la misma arquitectura se describe en docenas de reuniones separadas, sin un único lugar que refleje la imagen actual y acordada.
* **Ambigüedad:** las afirmaciones hechas en una reunión suelen estar poco especificadas y no se pueden dar por definición final de un componente sin clarificación adicional.
* **Contradicciones:** distintas reuniones — o distintas personas en la misma reunión — describen el mismo componente de forma inconsistente, y nada señala el conflicto.
* **Evolución de la arquitectura en el tiempo:** los componentes se describen en momentos distintos; sin ordenar esa línea temporal, no queda claro qué descripción sigue siendo válida.
* **Confusión entre lo implementado y lo planeado:** las reuniones mezclan lo que ya existe con lo que solo está previsto, y esa distinción es fácil de perder una vez que todo se resume junto.
* **Límites no documentados entre equipos:** los contratos y dependencias entre perfiles (p. ej. lo que Data Engineering espera de Software Engineering) suelen ser implícitos, nunca escritos en ningún sitio.
* **Visibilidad restringida:** parte de la información solo debería ser visible para ciertos perfiles por límites organizativos o de permisos, y un resumen plano y único filtraría o ignoraría esa distinción.
* **Falta de trazabilidad:** una vez que una reunión se resume a mano, normalmente es imposible rastrear una afirmación hasta quién la dijo, cuándo, y en qué conversación.
* **La alineación manual no escala:** conciliar todo lo anterior a mano, reunión tras reunión, no escala a medida que crece la organización y su arquitectura.

# Proceso de clarificación: colaboración entre agentes y humano-en-el-medio

El proyecto refina la comprensión final de la organización pidiendo clarificar los siguientes puntos:

1. **Clarificar ambigüedades:** Preguntar por elementos que no están lo bastante claros y necesitan información adicional para definir correctamente cada componente.
2. **Clarificar contradicciones:** Preguntar por elementos inconsistentes entre distintas versiones del mismo componente que necesitan clarificación o resolución.
3. **Clarificar la evolución de la arquitectura:** Clarificar la línea temporal, ya que los componentes se describen en momentos distintos, así que el proyecto debe ordenar la evolución.
4. **Clarificar subsistemas:** Algunos componentes se describen como parte de un sistema más grande, así que el proyecto debe preguntar por los límites y dependencias entre ellos.
5. **Clarificar lo implementado frente a lo planeado:** Preguntar si los componentes y capacidades ya existen o todavía no se han implementado.

El objetivo no es simplemente resumir una reunión, sino producir un **registro consistente y clarificado de la arquitectura de la organización**, identificando explícitamente huecos, límites, dependencias e inconsistencias.

Consulta el documento [doc/silver_process.md](doc/silver_process.md) para más detalle.

## Flujo de trabajo

Input transcription > Clarification (LLM / Human) > Publish > Enrich RAG

Aunque el flujo de trabajo muestra cómo lo entiende el usuario, internamente los datos siguen un enfoque por capas en PostgreSQL: las transcripciones en bruto se procesan para guardar el dato original, después clarificamos la información en una capa silver, y finalmente cubrimos una síntesis de la arquitectura en la capa gold.

```
                    ┌──────────────────────┐
                    │      RAW / Bronze    │
                    │                      │
Documentos ─────────►│ contenido original   │
                    │ metadatos             │
                    └──────────┬───────────┘
                               │
                               │ proceso del agente de clarificación
                               ▼
                    ┌───────────────────────────┐
                    │ SILVER / Input clarificado│
                    │                           │
                    │ Chunking de resumen        │
                    │ enriquecimiento metadatos  │
                    │ contratos de datos generados│
                    └──────────┬────────────────┘
                               │
                               │ refinamiento semántico
                               ▼
                    ┌────────────────────────────────────┐
                    │ GOLD / ADR: componentes versionados│
                    │                                    │
                    │ Chunking consciente de estructura  │
                    │ Versiones                           │
                    │ enriquecimiento de metadatos       │
                    │ Grafo de componentes                │
                    └────────────────────────────────────┘
```

## Preguntas esperadas a resolver

* ¿Cuándo se introdujo el componente X en la empresa?
* Muéstrame el diagrama de arquitectura general
* ¿Quién es responsable del componente X?
* Añade un nuevo componente Y con dependencias [X, Z] y el siguiente modelo [...]
* Dime la lista de personas que han hablado del componente X
* Dime los detalles del componente X, incluyendo sus dependencias y límites

## Guardrails

Identificar elementos ocultos por permisos o límites organizativos, donde algunos componentes son visibles para ciertos perfiles pero no para otros.

## Ejemplos

* Transcripciones sintéticas: descripciones específicas para gestionar tests unitarios y verificar el comportamiento esperado a partir de distintas descripciones.

## Interfaz de usuario

Una vez el usuario se autentica en una sesión (aislando su información), se muestran distintas pestañas:

* Input transcription: escribe un prompt o sube transcripciones y PDFs de una misma reunión, para que se procesen y clarifiquen.
* Architecture history: navega por cada ADR publicado hasta ahora, y el diagrama de arquitectura actual construido en vivo a partir de los propios componentes de Gold.
* Chat with RAG: una vez se publica un ADR, el usuario puede preguntar sobre la arquitectura y toda la línea temporal de los componentes.
* Test monitor (solo cuando la configuración activa start_test_mode): monitoriza la lista de tests y el consumo de tokens de toda la aplicación.

# Glosario de términos

* **Component** — Cualquier parte de la arquitectura de la organización que se trata en una
  reunión: un servicio, un sistema, un pipeline, una cola. Rastrear lo que se sabe de cada uno,
  reunión tras reunión, es el objetivo central de este proyecto.

* **Data Contract** — La interfaz acordada entre dos componentes: quién produce un dato,
  quién lo consume, y qué forma tiene. Las reuniones suelen describirlos solo de forma
  implícita ("el equipo A envía eventos de pedido al equipo B"); este proyecto los hace
  explícitos y los mantiene versionados.

* **Architecture Decision Record (ADR)** — El registro escrito de una decisión sobre un
  componente: qué cambió, por qué, qué alternativas se consideraron, y qué trade-offs se
  aceptaron. Se produce uno cada vez que la historia de un componente evoluciona.

* **Entity** — El nombre general para cualquier cosa de la que este proyecto guarda
  historial: un componente, un data contract, o la arquitectura en su conjunto. Cada entidad
  tiene su propia identidad y su propio historial de versiones, independiente de las demás.

* **Evolution** — Lo que le pasó a una entidad entre una reunión y la siguiente: es
  completamente **nueva**, **cambió** (una evolución de algo ya conocido), se **eliminó**, o
  permaneció **sin cambios**. Cada evolución queda registrada — nada se sobrescribe nunca en
  silencio.

* **Clarification** — Una pregunta planteada sobre algo que una reunión dejó ambiguo,
  contradictorio o sin resolver, junto con la respuesta que lo resuelve — venga esa respuesta
  de una reunión posterior o de una persona a la que se le pregunta directamente.

* **Bronze / Silver / Gold** — Las tres etapas por las que pasa un dato: Bronze es una
  reunión exactamente tal y como se dijo; Silver es esa reunión una vez que sus ambigüedades
  se clarifican y se convierte en un documento ADR; Gold es la imagen versionada de cada
  entidad, construida a partir de todos los ADR de Silver a lo largo del tiempo.

# Arquitectura

Una única aplicación FastAPI respaldada por Postgres (`pgvector` para embeddings) y un agente
de LangGraph que lleva un batch de transcripciones a través de todo el pipeline Bronze → Silver
→ Gold en una sola ejecución; Streamlit es la capa de interfaz (ver `## Interfaz de usuario`
arriba y `frontend/`), que llama al backend a través de `app/routers/frontend.py`. No hay un
sistema de usuarios real detrás del login — ver la sección "Frontend usage" de
[GETTING_STARTED.md](GETTING_STARTED.md) para las dos cuentas de ejemplo y sus tenants.

### Diagrama del sistema

![Arquitectura general: el usuario sube un transcript o pregunta al Frontend, que llama al Backend, que orquesta LangGraph. LangGraph llama al LLM y escribe en Postgres siguiendo el patrón medallion: Bronze, luego Silver, luego Gold. El Chat lee directamente de Gold.](doc/dataviz_architecture_general.svg)

### Grafo de agentes

![Ciclo de agentes de LangGraph: carga Bronze, genera y clasifica preguntas, si falta algo por responder un humano contesta, el Actor redacta el ADR, el Critic lo revisa, y si queda una afirmación sin verificar se vuelve a preguntar al humano una vez; si no, se publica en Silver y Gold.](doc/dataviz_langgraph_cycle.svg)

### Capas

Una ejecución de LangGraph lleva un batch de transcripciones a través de las tres capas — Gold
no es un pipeline separado, son los últimos tres nodos de la misma ejecución del grafo que
escribió Silver.

| Capa | Gestiona | Tabla(s) | Módulo |
| --- | --- | --- | --- |
| Bronze | Ingesta en bruto, sin interpretación | `bronze_documents` | `ingestion/service.py` |
| Silver | Bucle de clarificación Actor–Critic–Boss, ADR versionado por fuente | `silver_documents`, `silver_clarifications`, `silver_chunks` | `agents/graph.py` |
| Gold | Identidad de entidad entre reuniones y libro de evolución | `gold_evolution`, `gold_aliases` | `agents/stages/gold/service.py` |

### Decisiones técnicas clave

Decisiones de ingeniería tomadas para este propio proyecto — que no hay que confundir con los
ADR que el pipeline produce *sobre las reuniones que ingiere*.

**Estándar de código:** cada comentario y docstring del código Python de este repositorio sigue
**SE100 — Simple English** (frases cortas y claras, palabras comunes, sin modismos ni jerga).
Ver `doc/coding_standards.md` para la regla completa y ejemplos.

* **RAG, no CAG.** El corpus crece sin límite a medida que se ingieren nuevas reuniones con el
  tiempo, y el acceso a la descripción de un componente debe restringirse por perfil/límite de
  permisos (ver `## Guardrails` arriba). Meter todo el corpus en un contexto cacheado rompe
  ambas cosas: no escala más allá de una ventana de contexto, y no puede ocultar un chunk a una
  consulta que no debería poder responderlo. La recuperación por consulta (retrieval-per-query)
  es lo que hace manejables tanto el crecimiento como el límite de acceso.

* **Capas medallion (Bronze → Silver → Gold), las tres implementadas.** Bronze nunca
  interpreta; Silver ejecuta el bucle de clarificación y produce un ADR versionado por
  transcripción; Gold resuelve cada componente/data-contract mencionado a una identidad
  estable entre reuniones y mantiene un libro de evolución de solo-añadir por entidad —
  implementado en `agents/stages/gold/service.py` y los últimos tres nodos de `agents/graph.py`.

* **Actor–Critic–Boss como una sola ejecución de LangGraph, no tres servicios.**
  `synthesize_document` (Actor) redacta el ADR, `critic_document` (Critic) se ejecuta siempre y
  marca afirmaciones por severidad, y `boss_decide` (Boss, sin llamada a LLM — una política
  determinista) o bien degrada in situ las afirmaciones de baja severidad o escala una material
  a un humano. Mantener esto como una sola ejecución de grafo significa que el bucle de
  redacción comparte estado (`revision_attempted`) directamente en vez de coordinarlo entre
  límites de servicio, y `revision_attempted` limita la escalada a un solo reintento por fuente
  para que el bucle no pueda girar indefinidamente sobre una afirmación que la respuesta del
  humano no resolvió de verdad.

* **La generación de preguntas son dos etapas de LLM secuenciales, no una.** La primera
  llamada lee la transcripción y hace una pregunta por cada componente realmente nombrado,
  mientras solo *identifica* (nunca especifica del todo) cada data contract en juego
  (`prompts/architecture_questions/combined.jinja`). La segunda coge exactamente esos
  contratos identificados y redacta el conjunto completo de preguntas de completitud ODCS para
  cada uno (`prompts/data_contract_questions/questions.jinja`). Separarlas arregló un fallo
  observado en el que una única pasada combinada solía cubrir mal los data contracts — ver
  `doc/cost_analysis.md`.

* **La clarificación es LLM-first; a un humano solo se le pregunta lo que el LLM no pudo
  resolver.** Una llamada de clasificación ordena cada pregunta redactada en `answered` /
  `unknown` / `needs_clarification`; solo el último grupo llega a un humano, agrupado en un
  único `interrupt()` de LangGraph. El mismo nodo `ask_human` se reutiliza tanto para rellenar
  huecos en la etapa de clasificación como para una escalada del Boss — un único mecanismo de
  interrupción, un único checkpointer respaldado por Postgres, sin una segunda vía de pausa
  para el segundo caso.

* **La resolución de identidad en Gold es determinista, no una llamada a LLM.** Un nombre
  mencionado se resuelve por coincidencia exacta en `gold_aliases.alias`, luego por
  coincidencia difusa con `pg_trgm`, y solo entonces se acuña un nuevo `entity_id`. Esto
  mantiene la resolución de identidad auditable (cada variante de alias con la que se llamó a
  un componente es una fila que puedes inspeccionar) y barata — sin ida y vuelta al LLM solo
  para decidir si "the auth service" y "Auth Service" son la misma entidad.

* **El versionado de Gold es comparar-hash-y-versionar por entidad, no por documento.**
  `SilverDocument` versiona el ADR completo; `GoldEvolution` versiona cada componente, data
  contract, y arquitectura-por-fuente de forma independiente vía `entity_hash`. Un cambio en
  la narrativa de un componente no hace subir de versión a las demás entidades mencionadas en
  el mismo documento, así que "cómo ha evolucionado X" es una consulta contra el propio
  historial de versiones de X, nunca un diff entre entidades no relacionadas.

* **Sin claves foráneas entre capas.** `source_component`/`source_adr_version`/
  `ingestion_date` son columnas planas copiadas hacia delante en el momento de escritura
  (Bronze → Silver → Gold), la misma convención en todas partes. Esto cambia la aplicación de
  integridad referencial por consultas de trazabilidad sin joins y corrección de event-time
  (`ingestion_date` sigue siendo la fecha de la reunión, no un timestamp de procesamiento), lo
  cual importa más aquí porque nada en este pipeline borra ni reasigna el padre de una fila
  jamás.

* **Router de LiteLLM con fallback de OpenAI → Anthropic** (`llm/router.py`), para que la
  caída de un único proveedor no detenga la ingesta ni la clarificación.

# Métricas clave

Quality gates definidos para el pipeline RAG (Fase 6 de `.tmp/tasks.md` — todavía no
implementados; se registran aquí para que el objetivo quede explícito antes de escribir los
tests):

| Métrica | Qué detecta | Umbral |
| --- | --- | --- |
| Corrección de top-k / métrica de distancia / filtro | Un top-k desalineado en uno, un orden de similitud incorrecto, un filtro de perfil/sesión dejando pasar los chunks equivocados | Coincidencia exacta contra expectativas calculadas a mano |
| Recall@k del índice ANN frente a fuerza bruta | Un índice pgvector mal ajustado (HNSW/IVFFlat) descartando en silencio el único chunk que importaba | ≥ 0.95 |
| RAGAS faithfulness | La respuesta contiene una afirmación que los chunks recuperados no respaldan (alucinación) | Mínimo documentado por métrica, bloquea merges (`eval/thresholds.yaml`) |
| RAGAS context precision | Los chunks recuperados son en su mayoría relleno irrelevante | ″ |
| RAGAS context recall | A los chunks recuperados les falta algo que la respuesta de referencia necesitaba | ″ |
| Fuga de guardrail | Un chunk restringido por perfil llega a una respuesta generada para otro perfil, incluso parafraseado | Tolerancia cero — cualquier fuga es un fallo, no un umbral |


# Integración

MCP para cada salida, para agentes internos

# Próximos pasos

* Solo hay dos usuarios incluidos; esto debería evolucionar para gestionar múltiples usuarios y una arquitectura multi-tenant.
* Reducir costes de las llamadas a LLM poniendo una cache de embeddings y respuestas
