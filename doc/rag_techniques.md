# Técnicas de RAG, agentes y testing

Este documento explica las técnicas que recuperación semántica (RAG), los agentes que la alimentan 
y la forma en la que se prueba el sistema.

## Medallion RAG

El sistema se basa en un procesamiento de por capas de transcripciones, 
la capa final (la capa consumible) son **hechos versionados**: cada componente de arquitectura, 
cada contrato de datos y la arquitectura como un todo se extraen como entidades con identidad propia, 
y cada vez que algo cambia se guarda una nueva versión de esa entidad para entender
 tanto "¿qué es X hoy?" como "¿cómo llegó X a ser lo que es?".

Cada capa tiene un único trabajo y solo confía en que la anterior ya lo hizo bien:

- **Ingesta (raw):** guarda la transcripción tal cual, sin interpretarla. El único tratamiento es
  trocearla y calcular embeddings para poder localizarla más adelante; ningún modelo de lenguaje
  emite todavía un juicio sobre su contenido.
- **Silver:** convierte ese texto crudo en un documento fiable. Detecta huecos de información,
  genera preguntas de clarificación en dos fases (arquitectura y contratos de datos) cuando hace falta, 
  y usa el ciclo actor-crítico-decisor para redactar y revisar hasta producir un ADR versionado 
  donde cada afirmación está respaldada o marcada como no confirmada.
- **Gold:** es la capa consumible por el RAG. A partir del ADR ya aprobado extrae los hechos
  concretos — componentes, contratos, arquitectura — cada uno con identidad propia e historial de
  versiones. Es la única capa sobre la que se busca por similitud semántica.

Dentro de Gold, la identidad de cada hecho se resuelve en cascada: coincidencia exacta con un
nombre ya visto, si no aparece coincidencia aproximada por similitud de texto, y solo si ninguna
de las dos aparece se acuña una identidad nueva — así una misma entidad se reconoce aunque cambie
de nombre entre reuniones, sin fusionar por error dos cosas que solo se llaman parecido.

## Agentes: un grafo con roles

El razonamiento no ocurre en una única llamada a un modelo de lenguaje, sino en un grafo de
estados donde cada etapa tiene una responsabilidad acotada y puede decidir interrumpir el flujo
para pedir información a un humano en lugar de adivinar. Esa capacidad de pausar y esperar una
respuesta — en vez de seguir adelante con una suposición — es central: cuando la información que
hace falta no está en la transcripción, el sistema no la inventa, la pregunta.

Sobre ese grafo corre un patrón de tres roles para todo lo que se redacta: un actor que produce
un primer borrador, un crítico que lo revisa buscando contradicciones o afirmaciones sin respaldo
en la fuente, y un rol de decisión que, ante un desacuerdo real entre ambos, decide si se
reintenta automáticamente o si hace falta que un humano resuelva la disputa. Ninguna pieza de
contenido importante queda aceptada solo porque un modelo la generó una vez; hay una segunda
mirada estructural antes de darla por buena.

Lo que no se resuelve explícitamente durante ese proceso no se deja ambiguo en silencio: una
afirmación que dependía de una pregunta sin responder queda marcada como tal en el documento
final, de forma visible, en vez de mezclarse con el resto del texto como si estuviera confirmada.

Cada llamada que necesita devolver datos con una forma concreta — el estado de un componente, la
acción sobre un contrato, la lista de dependencias — se fuerza a través de un esquema tipado, no
de texto libre que luego haya que interpretar. Eso hace que buena parte de los errores de
formato se detecten antes de llegar a ninguna parte, y que el mismo vocabulario cerrado de
estados se comparta entre las distintas capas del sistema en vez de que cada una reinvente el
suyo.

Además, ninguna mención que el modelo dice haber visto en la transcripción se acepta a ciegas:
se verifica mecánicamente que el nombre aparezca de verdad en el texto fuente. Es un control
barato que atrapa alucinaciones concretas — nombres inventados — sin depender de que el propio
modelo se autoevalúe.

Por último, el conocimiento se construye en capas que solo confían en la capa de abajo una vez
que esa capa ya superó su propio control de calidad: primero los datos crudos, después el
documento ya clarificado y aprobado por el ciclo actor-crítico-decisor, y solo entonces se
extraen de ese documento aprobado los hechos versionados que alimentan la recuperación. Cada capa
hereda la confianza que la anterior ya se ganó, en vez de repetir el mismo trabajo de validación
tres veces.

## Testing: separar lo determinista de lo probabilístico

La regla más importante en las pruebas de todo este sistema es no dejar que un juicio
probabilístico decida si un test pasa o falla. Cada verificación se divide en dos partes: una
determinista — ¿existe el dato esperado?, ¿tiene el valor correcto?, ¿el vocabulario usado
pertenece al conjunto cerrado permitido? — que sí bloquea el test, y una evaluación de calidad
hecha por otro modelo de lenguaje — ¿la respuesta generada es clara?, ¿resume bien la evolución?
— que se registra para que una persona la lea, pero nunca hace fallar la prueba por sí sola. La
razón es simple: un juez que a veces se equivoca no puede ser el árbitro de si el sistema
funciona, porque entonces el test mismo se vuelve intermitente.

Las pruebas también se dividen según qué están verificando. Hay un conjunto rápido y gratuito que
sustituye las llamadas al modelo de lenguaje por respuestas fijas y predecibles: sirve para
probar la mecánica de almacenamiento y versionado sin pagar ni esperar por una llamada real, y
puede correr en cada cambio de código. Y hay un conjunto más lento y costoso que sí llama al
modelo real, con una base de datos real y transcripciones reales: ese es el único capaz de decir
si el razonamiento del sistema sobre lenguaje natural es efectivamente bueno, no solo si el
mecanismo de guardado funciona. Ninguno sustituye al otro; prueban cosas distintas a propósito.

Para el segundo tipo de prueba se construyó un escenario que no es un caso suelto sino una
historia continua: una misma arquitectura crece durante varios encuentros sucesivos, empezando
por un único componente y sumando piezas, de forma que un mismo contrato de datos atraviese todo
su ciclo de vida posible — nace, se actualiza de forma compatible, sufre un cambio que rompe
compatibilidad, se marca en desuso y finalmente se retira — dentro de una sola corrida. Es la
forma más económica de comprobar que el versionado se sostiene sobre una historia real y no solo
sobre una escritura aislada.

Por último, cuando una de estas pruebas costosas falla, no basta con volver a correrla y
esperar que esta vez salga bien: antes de cualquier verificación se vuelca a disco todo lo que el
modelo produjo en el camino — qué identificó, qué extrajo, qué escribió — para poder diagnosticar
la causa exacta sin necesidad de pagar y esperar de nuevo solo para averiguar qué pasó.
