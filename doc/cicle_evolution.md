# Ciclo de evolución de componentes y contratos

Cada componente y cada contrato de datos vive en `gold_evolution` como una serie de versiones
append-only (nunca se sobreescribe una versión existente). Una versión nueva solo se crea cuando
hay un cambio real que merece quedar registrado — no en cada ADR que simplemente vuelve a
mencionar el elemento.

Cada versión tiene dos fuentes de verdad independientes, que deben leerse juntas para saber si
algo "cambió" de verdad:

| Fuente | Qué es | Quién la decide |
|---|---|---|
| **Narrativa** (`narrative`) | Prosa libre que describe el elemento en esa ADR. | El LLM de extracción, en cada ADR, de nuevo — su redacción puede variar aunque el hecho sea idéntico. |
| **Payload** (`payload`) | Hechos estructurales: dependencias, contratos asociados (componentes); productor/consumidor y `odcs_spec` (contratos). | Cálculo determinista a partir de esa misma extracción — nunca varía por redacción. |

---

## Componentes

Estado (`ComponentStatus`, `agents/shared.py`): `new` · `modified` · `unchanged` · `removed` · `unknown`.

| Estado | Qué significa | ¿Genera versión nueva? |
|---|---|---|
| **new** | El componente aparece por primera vez. | Sí — siempre versión 1. |
| **modified** | El comportamiento/responsabilidad del componente cambió respecto a su versión anterior, **o** su payload estructural cambió (ver regla especial abajo), **o** alguno de sus contratos de datos asociados (input u output) es nuevo o fue modificado en esta misma ADR — aunque el LLM describiera al componente como "unchanged". | Sí — versión anterior + 1. |
| **unchanged** | Ni la narrativa tiene contenido nuevo real, ni el payload (dependencias, contratos input/output) difiere de la versión anterior. | No — se descarta, la versión actual se mantiene. |
| **removed** | El componente ya no forma parte de la arquitectura. | Sí — queda registrado como la última versión conocida. |
| **unknown** | La extracción no pudo determinar el estado con confianza. | Nunca llega a `gold_evolution` (se descarta antes de persistir, ver `agents.graph.persist_gold_evolution`). |

### Regla especial: el payload manda sobre la narrativa

Un componente puede tener su propio comportamiento narrado sin cambios, mientras que algo de su
payload sí cambia. Hay dos formas distintas en que esto pasa, y ambas cuentan como "modified":

1. **Su propia lista de contratos/dependencias cambia** — por ejemplo, empieza a producir un
   contrato nuevo, o deja de consumir uno que antes usaba (`input_contract_ids`/
   `output_contract_ids`/`dependency_ids` distintos a los de la versión anterior).
2. **Un contrato ya asociado (mismo id, sigue apareciendo en su lista) cambia por su cuenta** —
   por ejemplo, `registration-login-purchase` sufre un `break-change` en esta ADR; el componente
   que lo consume sigue teniendo exactamente el mismo `input_contract_ids`, pero lo que hay
   detrás de ese id cambió, así que el componente queda afectado igualmente.

Regla: si `operation == "unchanged"` pero (1) el payload propio del componente difiere del de la
versión anterior, **o** (2) alguno de los contratos que ya tenía asociados fue extraído en esta
misma ADR con una acción distinta de `unchanged`/`unknown` (es decir, ese contrato también generó
versión nueva), `operation` se corrige a `"modified"` antes de persistir — los hechos
estructurales, propios o heredados de sus contratos, tienen prioridad sobre el juicio narrativo
del LLM.

Esta corrección automática solo aplica a componentes, no a contratos (ver más abajo por qué).

### Por qué "unchanged" con la misma narrativa y el mismo payload no debe versionar

Cada ADR es una llamada nueva al LLM: aunque el hecho sea exactamente el mismo ("este componente
sigue igual"), la redacción de la narrativa puede variar de una ADR a otra. Comparar la
narrativa recién generada contra la anterior, tal cual, produce falsos positivos — versiones
nuevas que no representan ningún cambio real. Por eso, cuando `operation == "unchanged"`, la
comparación de hash usa la narrativa **de la versión anterior**, no la recién extraída — así solo
un cambio real de payload (o una corrección a "modified" por la regla de arriba) puede disparar
una versión nueva.

---

## Contratos de datos

Acción (`ContractAction`, `agents/shared.py`): `new` · `forward-update` · `break-change` ·
`unchanged` · `deprecated` · `removed` · `unknown`.

| Acción | Qué significa | ¿Genera versión nueva? |
|---|---|---|
| **new** | El contrato aparece por primera vez. | Sí — siempre versión 1. |
| **forward-update** | Cambio compatible hacia atrás: solo añade o relaja campos. | Sí — versión anterior + 1. |
| **break-change** | Cambio incompatible: elimina o renombra un campo, o convierte uno opcional en obligatorio. | Sí — versión anterior + 1. |
| **unchanged** | Ni la narrativa tiene contenido nuevo real, ni el payload (`producer`/`consumer`/`odcs_spec`) difiere de la versión anterior. | No — se descarta, igual que en componentes. |
| **deprecated** | Marcado para retirarse, pero todavía en uso. | Sí — versión anterior + 1. |
| **removed** | El contrato se retira definitivamente. | Sí — queda registrado como la última versión conocida. |
| **unknown** | La extracción no pudo determinar la acción con confianza. | Nunca llega a `gold_evolution`, igual que en componentes. |

### Por qué los contratos NO tienen la misma corrección automática que los componentes

`ContractAction` no tiene un "modified" genérico — distingue explícitamente entre
`forward-update` (compatible) y `break-change` (incompatible), una clasificación que requiere
juicio semántico sobre el propio cambio de schema. Si el payload de un contrato cambia mientras
está marcado "unchanged", forzar automáticamente uno de esos dos valores sin ese juicio arriesga
etiquetar mal un cambio real (por ejemplo, marcar como compatible algo que en realidad rompe
compatibilidad). Por eso, para contratos, solo se aplica la protección contra ruido de redacción
(comparar contra la narrativa anterior cuando `operation == "unchanged"`) — la clasificación de
`forward-update` vs. `break-change` sigue siendo, como hoy, un juicio del LLM sin verificación
automática contra la transcripción (limitación ya conocida, ver el propio comentario de
`ContractAction` en `agents/shared.py`).

---

## Ejemplo real que motivó este documento

En una misma sesión de prueba, el contrato `product-catalog` acumuló 4 versiones — v1 (`new`) y
v2/v3/v4, las tres marcadas `unchanged` — con `producer_id`/`consumer_id` y el resto del payload
**idénticos** en las cuatro. Las tres versiones "unchanged" eran puro ruido: la única diferencia
entre ellas era la redacción de la narrativa, generada de nuevo en cada ADR. Este documento y las
reglas de arriba son la corrección a ese comportamiento.
