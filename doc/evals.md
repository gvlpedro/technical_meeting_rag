# Evaluación, explicado simple


## Dos capas de evaluación, y por qué no basta con una

Generar un ADR, o responder una pregunta en el chat, no es determinista: la misma
entrada puede dar una salida distinta entre una llamada y la siguiente. No se puede
comprobar eso con una simple igualdad (`assert resultado == "lo que esperaba"`), pero
tampoco se puede dejar de comprobar nada solo porque sea probabilístico. El proyecto usa
dos capas distintas, cada una para lo que de verdad puede comprobar:

**La suite rápida (`tests/`, 230 tests)** no llama a ningún LLM. Comprueba la mecánica
determinista: aislamiento entre tenants, resolución de identidades, versionado,
guardrails de seguridad, el pipeline de recuperación. Esta es la que se ejecuta en cada
cambio — incluido el flujo de integración continua — porque es rápida, gratuita y
siempre da el mismo resultado.

**Los golden-sets (`agents/stages/*/testing/`)** sí llaman a un LLM real, con casos
reales preparados a mano, y la salida la juzga un segundo LLM actuando como crítico —
nunca una igualdad exacta de texto, porque dos redacciones válidas de la misma ADR nunca
serán idénticas palabra por palabra. Esto cuesta dinero y tiempo, así que no se ejecuta
en cada cambio — se ejecuta a propósito, cuando se quiere medir si un cambio de prompt o
de modelo mejora o empeora la calidad real.

## Las cuatro cosas que pueden fallar, y que se evalúan por separado

Una respuesta mala puede venir de cuatro sitios distintos, y cada uno necesita un
arreglo distinto:

1. **Mecánica de recuperación** — ¿se pidió lo correcto a la base de datos?
2. **Corrección del filtrado** — ¿la búsqueda devolvió de verdad lo que debía, sin
   mezclar datos de otro tenant?
3. **Calidad de la respuesta** — dado el contexto correcto, ¿la respuesta es fiel a lo
   que ese contexto dice, sin inventar nada por su cuenta?
4. **Guardrails** — ¿se filtró o se enseñó alguna vez algo que no debía verse?

Un sistema puede acertar en tres de estos cuatro puntos y seguir estando roto. Por eso
se evalúan por separado, no como una sola caja negra de "funciona / no funciona".

## Qué cubre cada golden-set hoy

| Etapa | Casos reales | Qué comprueba |
|---|---|---|
| Preguntas de arquitectura | 10 | Identificar componentes y ambigüedades reales en una transcripción. |
| Preguntas de contratos de datos | 4 | Detectar contratos nuevos, parcialmente definidos, o con versión rota. |
| Clasificación de preguntas | 6 | Decidir si una pregunta ya está respondida, es irresoluble, o necesita a un humano. |
| Generación de la ADR (Actor-Critic-Boss) | 3 | El documento final, revisado por el propio Critic contra la transcripción. |
| Evolución de arquitectura en Gold | 10 | Que el historial de versiones de un componente quede bien reconciliado entre reuniones. |
| Chat con RAG | 8 | Que una respuesta del chat esté fundamentada en lo recuperado, no inventada. |

41 casos reales en total, repartidos en seis suites — cada una con su propio criterio de
corrección, no uno genérico para todas.

## Resultados medidos en el tiempo

`TESTING_REPORT.md`, en la raíz del repositorio, guarda la evolución real de las
métricas clave a lo largo de varias fechas: puntuación del Critic, fidelidad y cobertura
de contexto (RAGAS), porcentaje de citas del chat verificadas correctamente, y porcentaje
de la suite rápida que pasa. La tendencia documentada es de mejora sostenida — por
ejemplo, la fidelidad RAGAS pasó de 0.60 a un rango 0.92–1.00 conforme se corrigieron
guardrails y se afinaron los prompts.

## Qué no cubre todavía esta evaluación

- No hay un guardrail de visibilidad por perfil/rol todavía — el README lo plantea como
  objetivo, pero ningún golden-set lo comprueba porque el propio modelo de datos no
  guarda esa información aún.
- Los golden-sets no se ejecutan en el flujo de integración continua — tienen coste real
  en tokens y no son deterministas, así que se corren a mano, no en cada commit.
- La clasificación `forward-update` / `break-change` de un contrato de datos es un
  juicio del LLM sin verificación automática contra la transcripción — una limitación ya
  conocida y documentada donde ocurre, no una omisión de este documento.
