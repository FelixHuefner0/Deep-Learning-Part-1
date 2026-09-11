## Execute code

```powershell
.\.venv\Scripts\python.exe .\logistic_neuron_threshold_v2.py
```

## Themen

### Gewicht, Bias, Alpha und Threshold einfach erklärt

Der bekannte Ausdruck ist `z = w * x + b`: Die Eingabe `x` wird mit dem Gewicht `w` multipliziert, dann wird der Bias `b` addiert.

Das logistische Neuron berechnet zuerst genau diesen Ausdruck. Danach kommt die Sigmoidfunktion: `y = sigmoid(z)`. Sie macht daraus eine Ausgabe zwischen 0 und 1. Bei `z = 0` ist die Ausgabe genau 0,5.

**Alpha und Threshold sind keine zusätzlichen Parameter.** Sie sind eine andere Möglichkeit, Gewicht und Bias festzulegen:

```text
z = alpha * (x - t)
  = alpha * x - alpha * t

Also: w = alpha und b = -alpha * t
```

- **Threshold `t` bedeutet Schwelle:** Hier liegt die Mitte des Übergangs, also die Stelle, an der das Neuron 0,5 ausgibt.
- **Alpha bestimmt die Steilheit:** Bei positivem Alpha gilt: Je größer Alpha, desto steiler der Übergang von nahe 0 zu nahe 1.

Die Idee ist: „Ich möchte den Übergang an einer bestimmten Stelle haben und einstellen, wie steil er ist.“ Dafür sind die Namen `t` und `alpha` praktisch.

### Beispiel aus der Base-Datei

Im Code stehen `t = -3` und `alpha = 5`. Daraus werden:

```text
w = 5
b = -5 * (-3) = 15
z = 5 * (x + 3) = 5 * x + 15
```

Das ist weiterhin der normale Ausdruck `w * x + b`.

| Eingabe x | Zwischenwert z | Ausgabe sigmoid(z) |
|---|---:|---:|
| -4: unter der Schwelle | -5 | ungefähr 0,007 |
| -3: genau an der Schwelle | 0 | 0,5 |
| -2: über der Schwelle | 5 | ungefähr 0,993 |

Erhöhst du Alpha zum Beispiel von 5 auf 20, sieht die Kurve stärker wie eine Stufe aus. Die Mitte bleibt bei `x = -3`, weil der Bias mit `b = -alpha * t` passend mitgeändert wird. Genau an der Schwelle bleibt die Ausgabe immer 0,5; für jedes endliche Alpha bleibt die Kurve glatt.

**Alpha und t stecken also im Ausdruck vor der Sigmoidfunktion. Die Sigmoidfunktion selbst bleibt gleich.** Auch eine Gerade könnte man als `alpha * (x - t)` schreiben: Dort wäre `t` ihre Nullstelle. Erst die anschließende Sigmoidfunktion macht daraus den S-förmigen Übergang.

