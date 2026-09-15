# Präsentationsleitfaden – Logistisches Neuron V3

**Gesamt: 10 Minuten = 8 Minuten Vortrag + 2 Minuten Diskussion.**

## Aufteilung

| Zeit | Person | Inhalt |
| --- | --- | --- |
| 0:00–0:30 | Person 1 | Aufgabe und Ziel |
| 0:30–2:00 | Person 1 | Tab 1: Rechenweg und Sigmoid |
| 2:00–4:00 | Person 1 | Tab 2: Gewicht und Bias |
| 4:00–7:30 | Person 2 | Tab 3: Harte Schwelle annähern |
| 7:30–8:00 | Person 2 | Drei Kernaussagen |
| 8:00–10:00 | Beide | Diskussion |

- Person 1 erklärt die Grundlagen; Person 2 zeigt die gezielte Anwendung.
- Jede Person bedient die Demo während ihres eigenen Teils.
- Ein durchgehendes Beispiel verwenden: **Umschalten bei Eingabe 3**.

## Person 1 – Einstieg · 30 Sekunden

- Ein einzelnes Neuron mit einer Eingabe untersuchen.
- Drei Fragen nennen: Wie rechnet es? Was verändern Gewicht und Bias? Wie lässt sich eine harte Schwelle annähern?
- Links: Ausgabe ohne zusätzliche Aktivierung. Rechts: Ausgabe mit Sigmoid.
- Parameter werden von Hand eingestellt; das Neuron wird hier nicht trainiert.

**Kurzer Einstieg:**

> „Wir zeigen, wie ein einzelnes Neuron rechnet und wie wir es so einstellen können, dass es sich fast wie ein Ein-/Aus-Schalter verhält.“

## Person 1 – Tab 1: So rechnet das Neuron · 1½ Minuten

### Zeigen

- Tab 1 öffnen: Gewicht **1**, Bias **0**.
- Die drei Wertefelder für Eingabe, Gewicht und Bias zeigen.
- Die beiden Rechenkarten erklären: **Formel → eingesetzte Zahlen → Ausgabe**.
- Eingabe nacheinander auf **−2, 0, 2** setzen; die orangefarbenen Punkte verfolgen.

### Darauf eingehen

- Zuerst: `z = w · x + b` – Eingabe gewichten und Bias addieren.
- Links ist dieser Wert z bereits die Ausgabe; es entsteht eine Gerade.
- Rechts wird derselbe Wert z in `y = 1 / (1 + e^(−z))` eingesetzt.
- Die Sigmoidfunktion ist nichtlinear und bildet z auf einen Wert zwischen 0 und 1 ab.
- Beispiel **x = 2**: links **2**, rechts ungefähr **0,881**.
- Bei **x = 0**: links **0**, rechts **0,5**.
- Nur x verändern: Der Punkt bewegt sich, die Kurven bleiben gleich.

**Merksatz:**

> „Die Sigmoidfunktion verarbeitet den Zwischenwert. Sie macht aus der Geraden die S-förmige Gesamtfunktion.“

## Person 1 – Tab 2: Gewicht und Bias · 2 Minuten

### A. Gewicht verändern · etwa 1 Minute

- Tab 2 öffnen: **w = 1**, **b = 0**.
- Gewicht von **1 auf 5** erhöhen; Bias bei **0** lassen.
- Links: Die Steigung der Geraden nimmt zu. Dabei die automatisch angepasste y-Achse beachten.
- Rechts: Der Übergang der S-Kurve wird steiler.
- Die Kurvenmitte bleibt **in diesem Beispiel mit b = 0** bei x = 0.
- Größeres positives Gewicht bedeutet einen abrupteren Übergang.

### B. Bias verändern · etwa 1 Minute

- Gewicht wieder auf **1** stellen.
- Bias von **0 auf −3** verändern.
- Links: Die Gerade verschiebt sich nach unten.
- Rechts: Die Kurvenmitte verschiebt sich nach rechts auf **x = 3**.
- „Eingabe auf Kurvenmitte“ drücken: links **0**, rechts **0,5**.
- Bezug zur Formel: Die Mitte liegt dort, wo `w · x + b = 0`, also bei `x = −b/w` für w ungleich 0.

**Merksatz und Übergabe:**

> „Gewicht und Bias bestimmen gemeinsam den Verlauf. Wir haben die Mitte jetzt bei drei. Als Nächstes zeigen wir, wie der Übergang dort immer stärker einem harten Schalter ähnelt.“

## Person 2 – Tab 3: Einen Schalter annähern · 3½ Minuten

### A. Ziel und Referenz erklären · etwa 45 Sekunden

- Tab 3 öffnen: **t = 3**, **α = 1**.
- Auf die graue gestrichelte Kurve zeigen: **Das ist die harte Schwelle.**
- Harte Schwelle: **unter 3 → 0; ab 3 → 1**.
- Auf die blaue Kurve zeigen: **Das ist die tatsächliche Sigmoid-Ausgabe.**
- Beide klar unterscheiden: Die blaue Kurve soll sich der grauen annähern.

### B. t und α mit der bekannten Formel verbinden · etwa 1 Minute

- Wertefelder zeigen: **x** ist die Eingabe, **t** die gewünschte Umschaltstelle, **α** die Steilheit.
- **t sagt, wo der Übergang liegt; α sagt, wie abrupt er ist.**
- Auf die Zuordnung unter den Wertefeldern zeigen: `w = α` und `b = −α · t`.
- Linke Rechenkarte: `z = α · (x − t) (= w · x + b)`.
- `x − t` beschreibt den vorzeichenbehafteten Abstand zur Schwelle; α verstärkt ihn.
- Beispiel **x = 2, t = 3, α = 1**: `1 · (2 − 3)` ergibt **z = −1**, danach **y ≈ 0,269**.
- Klarstellen: **Gleiches Neuron wie in Tab 2. Gewicht und Bias werden jetzt passend zusammen eingestellt.**

### C. Annäherung live zeigen · etwa 1 Minute

- Schwellwert kurz von **3 auf 2** verändern: Der Übergang wandert mit.
- Schwellwert wieder auf **3** setzen und dort belassen.
- Steilheit von **1 auf 5, dann auf 20** erhöhen.
- Die blaue Kurve nähert sich außerhalb der Schwelle der grauen Referenz an.
- Auf die Parameterzuordnung zeigen: Bei **α = 20** werden **w = 20** und **b = −60** verwendet.
- „Übergang vergrößern“ aktivieren; danach x im Zahlenfeld auf **2,9**, dann **3,1** setzen.
- Bei α = 20 liegen die Ausgaben ungefähr bei **0,119** und **0,881**. Der Übergang ist also weiterhin weich.

### D. Grenze der Annäherung zeigen · etwa 45 Sekunden

- „Eingabe auf Schwelle“ drücken: **x = t = 3**.
- Links: `α · (3 − 3)` ergibt immer **0**.
- Rechts: Sigmoid von 0 ergibt immer **0,5**, unabhängig von der Steilheit.
- Die harte Referenz liefert an dieser Stelle **1**; das Neuron liefert **0,5**.
- Das ist kein Fehler: Die Aufgabe verlangt eine **Annäherung**, keine identische Funktion.
- Keine Aussage wie „Ab jetzt gibt Sigmoid nur noch 0 oder 1 aus“ verwenden.

**Merksatz:**

> „Je größer Alpha, desto schmaler wird der Übergang. Genau an der Schwelle bleibt die Ausgabe trotzdem 0,5.“

## Person 2 – Abschluss · 30 Sekunden

- **Tab 1:** Affine Berechnung plus nichtlineare Sigmoid-Aktivierung.
- **Tab 2:** Gewicht und Bias verändern den Verlauf der Funktion.
- **Tab 3:** Durch `w = α` und `b = −α · t` entsteht eine immer steilere Annäherung an die gewünschte Schwelle.

> „Damit haben wir die drei Lernziele gezeigt. Welche Fragen gibt es dazu?“

## Beide – Diskussion · 2 Minuten

| Mögliche Frage | Kurze Antwort | Wer? |
| --- | --- | --- |
| Was passiert ohne Sigmoid? | Die Ausgabe ist direkt w · x + b und nicht auf 0 bis 1 begrenzt. | Person 1 |
| Was passiert bei negativem Gewicht? | Die S-Kurve fällt statt zu steigen. | Person 1 |
| Was passiert bei Gewicht 0? | Die Ausgabe hängt nicht mehr von x ab: links b, rechts Sigmoid(b). | Person 1 |
| Warum wird Bias bei größerem α mitverändert? | Damit die Mitte bei t bleibt; sonst kann sich die Umschaltstelle verschieben. | Person 2 |
| Warum ist die Ausgabe bei t noch 0,5? | Dort ist z = α · (t − t) = 0 und Sigmoid(0) = 0,5. | Person 2 |
| Wie bekommt man exakt 0 oder 1? | Durch einen zusätzlichen Entscheidungsschritt: y ≥ 0,5 → 1, sonst 0. Dieser Schritt ist nicht die Sigmoidfunktion selbst. | Person 2 |

## Vor der Vorführung

- Demo vorher starten und Fenster passend zum Bildschirm einstellen.
- Für genaue Beispielwerte die Zahlenfelder benutzen; beim freien Zeigen die Regler.
- Tabwechsel setzt die Werte zurück. Innerhalb eines Tabs weiterarbeiten, ohne erneut auf dessen Button zu klicken.
- Den gesamten Ablauf einmal mit Stoppuhr auf **acht Minuten** proben.
- Sonderfälle aus der Diskussion nicht zusätzlich in den Hauptvortrag aufnehmen.
- Die linke y-Achse skaliert automatisch; die rechte bleibt fest. Beim Gewichtsvergleich nicht nur den sichtbaren Winkel der Geraden beurteilen.
