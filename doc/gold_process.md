# Gold, explicado simple


## Qué produce Gold

Silver produce un documento por reunión. Gold hace algo distinto: junta **todas** las
reuniones publicadas hasta ahora, de todas las fechas, y mantiene un registro único de
cómo es la arquitectura real — qué componentes existen, qué contratos de datos los
conectan, y cómo ha ido cambiando todo eso con el tiempo.

Mientras que un documento de Silver solo sabe de la reunión que lo originó, Gold sabe de
la suma de todas ellas. Es la capa que responde "¿qué sabemos de verdad, a día de hoy,
sobre este componente?" — no "¿qué dijo esta reunión concreta?".

## Los datos que mantiene Gold

Gold mantiene, conceptualmente, dos cosas:

**Un libro de evolución.** Una fila por cada versión real de cada componente, cada
contrato de datos, y de la arquitectura general — nunca se borra ni se sobreescribe una
versión ya existente, solo se añaden nuevas. Cada versión guarda su propia narrativa,
sus hechos estructurales (de qué depende un componente, o quién produce y quién consume
un contrato), qué reunión concreta la originó, y desde cuándo es válida. Esto permite
preguntar no solo "¿cómo es X ahora?" sino también "¿cómo era X en una fecha concreta?".

**Un directorio de identidades.** Dos reuniones distintas pueden referirse al mismo
componente con nombres ligeramente distintos ("Timeline Svc" frente a
"timeline-service"). Este directorio reconoce que son la misma pieza, para que su
historial no se fragmente en dos identidades separadas que en realidad son una sola
cosa.

## Qué hace que se cree una versión nueva

Una versión nueva solo se crea cuando hay un cambio real que vale la pena dejar
registrado — nunca por el simple hecho de que una reunión vuelva a mencionar algo que
ya existía. Concretamente:

- Si una reunión confirma que algo es nuevo, que ha cambiado, o que se ha retirado, eso
  sí genera una versión nueva.
- Si una reunión menciona algo pero nunca llega a confirmar qué le pasó, esa mención
  simplemente no se registra — no se inventa un estado a partir de una mención de paso.
- Si lo único que cambia entre dos reuniones es la redacción con la que se describe
  exactamente el mismo hecho, eso tampoco cuenta como cambio — la narrativa se mantiene
  fija hasta que de verdad pasa algo nuevo, para que la misma idea contada con otras
  palabras no genere ruido de versiones sin sentido.

## El efecto en cadena

Un componente puede verse afectado por una reunión que nunca lo menciona directamente.
Si un contrato de datos que el componente ya usaba cambia en una reunión posterior —por
ejemplo, gana un nuevo consumidor—, el componente recibe su propia versión nueva
también, aunque esa reunión concreta no hable de él en ningún momento. Tiene sentido:
la historia del componente sí se ve afectada por ese cambio, incluso si el protagonista
de esa reunión fue el contrato, no el componente.

## Cómo reconoce la misma pieza aunque se la nombre distinto

Gold intenta resolver un nombre nuevo contra lo que ya conoce, primero buscando una
coincidencia exacta, y si no la hay, una coincidencia por semejanza en el nombre — así
un renombrado real sigue reconociéndose como la misma pieza. La contrapartida de esto es
conocida: dos cosas genuinamente distintas, pero con nombres muy parecidos, pueden
confundirse como si fueran la misma. Es un compromiso inherente a reconocer renombrados
de forma automática, no un fallo puntual de un caso concreto.

## Cómo se usa para responder preguntas en el chat

Cuando alguien pregunta en el chat, Gold busca por significado (qué tan parecida es la
pregunta a cada narrativa guardada) y, a la vez, por coincidencia textual exacta,
combinando ambos resultados — así una pregunta que usa palabras distintas a las del
registro, y otra que repite el término exacto, tienen cada una su propia vía para
encontrar lo relevante.

La búsqueda es honesta en dos pasos: primero intenta con un criterio estricto de
relevancia, y solo si eso no encuentra nada, lo intenta una segunda vez sin ese filtro,
antes de rendirse. Si de verdad no hay nada relevante, el chat lo dice explícitamente
en vez de responder con el candidato menos malo. Y cada referencia que la respuesta cita
se comprueba contra lo que realmente se encontró — nunca se da por buena solo porque el
propio modelo diga que la usó.

## La arquitectura actual, en vivo

El diagrama de "arquitectura actual" que ve un usuario se construye en el momento,
directamente desde el estado más reciente de Gold — nunca a partir del dibujo que trazó
una única reunión, y nunca desde una copia guardada de antes. Así siempre refleja la
suma de todo lo confirmado hasta ahora, no solo lo que se dijo la última vez.

## Preguntas sobre el pasado y la evolución

Además de "¿cómo es esto ahora?", Gold puede responder "¿cómo era esto en una fecha
concreta?" — apoyándose en que cada versión sabe desde y hasta cuándo fue la vigente. Y
puede situar una pieza en el tiempo frente a las demás: qué existía antes de ella, y qué
llegó después, útil para entender una cadena de sustituciones o una migración por
etapas.

## Qué no intenta hacer Gold

- Gold no vuelve a leer ni a interpretar la transcripción de una reunión por su cuenta
  — ese juicio es enteramente de Silver. Gold solo reconcilia lo que Silver ya decidió,
  reunión a reunión.
- Una versión nueva nunca aparece sin motivo: siempre es porque la propia pieza cambió,
  o porque algo de lo que depende cambió. Nunca es arbitraria.
- Gold no inventa una relación —una dependencia, quién produce o consume un contrato—
  que ninguna reunión haya afirmado realmente.
