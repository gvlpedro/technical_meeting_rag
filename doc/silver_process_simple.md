# Silver, explicado simple


## Qué produce Silver

Por cada reunión subida en un mismo día, Silver produce **un documento de decisión**
(un ADR — una especie de acta razonada: qué se decidió, por qué, y qué cambia). Ese
documento se rellena solo con lo que la propia reunión, más las respuestas que dé un
humano, permiten afirmar con seguridad.

Si algo no se puede confirmar, esa parte del documento se deja explícitamente como "no
cubierto" — nunca se inventa. Un documento con varias secciones sin cubrir no es un fallo:
es exactamente lo que se espera cuando una reunión no llegó a tratar ese tema.

## Los datos que viajan por el proceso

Silver mantiene, conceptualmente, tres cosas:

**El documento en sí, versionado.** Si la misma reunión se vuelve a procesar y el
resultado es idéntico al de la última vez, no se crea nada nuevo. Si de verdad cambia
—porque las respuestas fueron distintas—, se guarda como una versión nueva, sin borrar
ni tocar la anterior. Así siempre se puede consultar cómo era el documento en cualquier
momento de su historia, no solo la última versión.

**El historial de preguntas y respuestas.** Cada pregunta que se llegó a hacer, y lo que
se contestó, queda registrado para siempre — incluso si la misma reunión se procesa dos
veces y la segunda vez se responde distinto. Nada se sobrescribe aquí: es un registro que
solo crece.

**Una copia del documento lista para buscarse.** El ADR completo (no troceado) se prepara
para que, más adelante, se pueda encontrar por significado — no solo por las palabras
exactas que usa — cuando alguien pregunte por él en el chat.

## Qué intenta resolver el bucle de clarificación

El objetivo no es solo resumir la reunión, sino dejar un registro consistente del estado
de la arquitectura, señalando explícitamente lo que falta. Para eso, las preguntas que se
generan buscan resolver cinco cosas:

1. **Ambigüedades** — algo que se mencionó pero no quedó lo bastante claro como para
   darlo por definido.
2. **Contradicciones** — dos partes de la misma reunión, o de reuniones distintas, que
   cuentan lo mismo de forma incompatible.
3. **La evolución en el tiempo** — como las reuniones describen la arquitectura en
   momentos distintos, hace falta poder ordenar qué vino antes y qué vino después.
4. **Los límites entre subsistemas** — cuando un componente se describe como parte de
   algo más grande, hay que preguntar dónde empieza y termina cada pieza, y de qué depende.
5. **Lo implementado frente a lo planeado** — distinguir con claridad qué ya existe hoy
   de qué todavía no se ha construido.

## Cómo decide si preguntar o no

Cada pregunta que se genera termina clasificada en una de tres situaciones:

- **La propia reunión ya la responde**, de forma directa o claramente implícita. En ese
  caso se registra la respuesta sin molestar a nadie.
- **No se sabe, y probablemente nunca se va a saber a partir de esta reunión.** Aquí no
  se pregunta a nadie — se deja constancia de que es un dato desconocido y se sigue
  adelante.
- **No se sabe, pero alguien que participó en la reunión probablemente sí lo sabe.** Solo
  este tercer caso llega a convertirse en una pregunta real para un humano.

Todas las preguntas que de verdad necesitan a un humano se agrupan y se hacen de una sola
vez, no una por una. Y si, al responder, el humano tampoco sabe la respuesta, esa pregunta
pasa a marcarse como "no se sabe" — y no se vuelve a preguntar después.

## El recorrido, paso a paso

1. **Se cargan las transcripciones** de ese día, todas juntas.
2. **Se generan las preguntas**: primero sobre la arquitectura y los componentes que se
   mencionan, y después — solo si hizo falta — sobre el detalle completo de los contratos
   de datos que se identificaron.
3. **Se clasifican automáticamente** todas esas preguntas, según la regla explicada arriba.
4. **Si algo necesita a un humano, el proceso se pausa** y se le presentan todas las
   preguntas pendientes de golpe. Al responder, el proceso continúa donde lo dejó.
5. **Se redacta el documento** con todo lo que ya se sabe — la reunión más las respuestas
   obtenidas.
6. **Se revisa ese documento** frase por frase contra la reunión original, buscando
   cualquier afirmación que en realidad no esté respaldada.
7. **Si se encuentra algo sin respaldo:** lo poco importante se marca directamente como
   dudoso en el propio documento, sin molestar a nadie. Lo importante se pregunta una vez
   más al humano y el documento se vuelve a redactar con esa respuesta. Si, tras esa
   segunda vuelta, todavía queda algo sin poder confirmarse del todo, ya no se pregunta una
   tercera vez — se marca como dudoso y el proceso sigue adelante, para no quedarse
   bloqueado indefinidamente.
8. **Se guarda la versión final** del documento. A partir de aquí, ese documento pasa a
   alimentar la capa Gold, que es quien reconcilia esta información con la de todas las
   demás reuniones.

## Qué no intenta hacer Silver

- Silver no decide por sí solo cuál es el estado "verdadero" de un componente comparando
  todas las reuniones entre sí — eso es tarea de la capa Gold, no de Silver.
- El número de versión de un documento de Silver solo indica si el texto de **ese**
  documento concreto cambió respecto a la vez anterior. No representa, por sí mismo, la
  evolución real de un componente a lo largo del tiempo.
- Silver prefiere siempre una respuesta honesta e incompleta — dejar algo explícitamente
  como "no se sabe" — antes que rellenar un documento con algo que no está respaldado por
  la reunión.
