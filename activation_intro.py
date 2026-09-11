"""Small 2-2-1 network comparison shown before the logistic neuron demo."""

import numpy as np


def network_score(points, nonlinear, pattern="XOR", improved=False):
    """Hand-picked weights; final class is score >= 0 in both networks."""
    from logistic_neuron_threshold_v2 import sigmoid

    total = np.asarray(points).sum(axis=-1)
    hidden = np.stack((10 * (total - .5), 10 * (total - 1.5)), axis=-1)
    if nonlinear:
        hidden = sigmoid(hidden)
    if improved and not nonlinear and pattern == "XOR":
        return hidden[..., 0] - .5
    return hidden[..., 0] - (hidden[..., 1] if pattern == "XOR" else 0) - .5


class ActivationIntro:
    def __init__(self):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button
        from matplotlib.patches import FancyBboxPatch
        from logistic_neuron_threshold_v2 import ButtonSelector

        self.proceed = False
        self.improved = False
        self.points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.fig = plt.figure(figsize=(16, 12), facecolor="#f4f6fa")
        for left, fill, edge in [(.04, "#ffffff", "#d6e2eb"), (.52, "#fcfaff", "#d9cceb")]:
            self.fig.add_artist(FancyBboxPatch(
                (left, .105), .455, .703, transform=self.fig.transFigure,
                boxstyle="round,pad=0.008,rounding_size=0.018",
                facecolor=fill, edgecolor=edge, linewidth=1.5, zorder=-10))
        self.fig.canvas.manager.set_window_title("Warum Aktivierungsfunktionen?")
        self.fig.suptitle("Was verändert die Aktivierung im Netz?", fontsize=21, weight="bold", y=.975)
        self.pattern = ButtonSelector(self.fig, ["OR", "XOR"], ["1 · OR", "2 · XOR"],
                                      [[.055, .883, .10, .043], [.159, .883, .10, .043]])
        self.task = self.fig.text(.28, .898, "", fontsize=12)
        self.fig.text(.055, .842, "BEISPIELEINGABE", fontsize=9, color="#64748b", weight="bold")
        self.examples = ButtonSelector(self.fig, list(range(4)), ["(0, 0)", "(0, 1)", "(1, 0)", "(1, 1)"],
                                       [[.20 + i * .075, .829, .07, .037] for i in range(4)])
        self.examples.set_active(3)
        self.target_text = self.fig.text(.52, .84, "", fontsize=11, weight="bold")
        self.diagrams = [self.fig.add_axes([left, .425, .425, .37]) for left in [.055, .535]]
        self.axes = [self.fig.add_axes([left, .18, .425, .195]) for left in [.055, .535]]
        self.improve_button = Button(self.fig.add_axes([.12, .112, .295, .029]),
                                     "Verbessern auf 3/4", color="#eef2f6", hovercolor="#dceaf4")
        self.improve_button.label.set_fontsize(10)
        for spine in self.improve_button.ax.spines.values():
            spine.set_visible(False)
        self.improve_button.on_clicked(self.toggle_improvement)
        self.note = self.fig.text(.055, .069, "", fontsize=11, weight="bold")
        self.fig.text(.055, .039, "Blau = Klasse 0 · Orange = Klasse 1 · Hintergrund = Netz · Ring = gewählte Eingabe · × = Fehler",
                      fontsize=9, color="#64748b")
        self.fig.text(.055, .017, "Gewichte für OR / XOR von Hand gewählt — in einem trainierten Netz werden sie aus Beispielen gelernt.",
                      fontsize=9, color="#64748b")
        self.button = Button(self.fig.add_axes([.73, .025, .24, .048]), "Weiter: logistisches Neuron →",
                             color="#eef2f6", hovercolor="#dceaf4")
        for spine in self.button.ax.spines.values():
            spine.set_visible(False)
        self.button.on_clicked(self.continue_demo)
        self.pattern.on_clicked(self.update)
        self.examples.on_clicked(self.update)
        self.fig.canvas.mpl_connect("button_press_event", self.select_point)
        self.update()

    def toggle_improvement(self, _=None):
        if self.pattern.value_selected == "XOR":
            self.improved = not self.improved
            self.update()

    def select_point(self, event):
        if event.inaxes in self.axes and event.xdata is not None and event.ydata is not None:
            distances = np.linalg.norm(self.points - [event.xdata, event.ydata], axis=1)
            if distances.min() < .25:
                self.examples.set_active(int(distances.argmin()))

    def continue_demo(self, _=None):
        import matplotlib.pyplot as plt
        self.proceed = True
        plt.close(self.fig)

    def draw_network(self, ax, point, nonlinear, pattern, target):
        from matplotlib.patches import FancyBboxPatch
        from logistic_neuron_threshold_v2 import sigmoid

        ax.clear()
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        blue, purple = "#176b9c", "#7952b3"
        ax.text(0, .96, "Mit Sigmoid" if nonlinear else "Ohne Aktivierung", fontsize=15, weight="bold")
        adjusted = self.improved and pattern == "XOR"
        ax.text(0, .875,
                ("Ausgabe verbessert: w₄ von −1 auf 0 gesetzt" if not nonlinear else "Unverändert: ursprüngliche Gewichte mit Sigmoid")
                if adjusted else "Alle Gewichte und Bias gleich · Nur die Aktivierung unterscheidet sich",
                fontsize=8, color="#64748b")
        def frame(x, y, width, height, fill="white", edge="#d6e2eb"):
            ax.add_patch(FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.009,rounding_size=0.018",
                         facecolor=fill, edgecolor=edge, linewidth=1.2, zorder=1, clip_on=False))

        def arrow(start, end, label=None):
            ax.annotate("", xy=end, xytext=start,
                        arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=1.3), zorder=2)
            if label:
                ax.text(start[0] + .012, start[1] - .03, label, fontsize=9, color=purple, weight="bold")

        def step(x, y, width, title, formula, result, accent=False):
            frame(x, y, width, .115, "#f1ebfa" if accent else "#eef4f8")
            ax.text(x + .012, y + .088, title, fontsize=8, color="#64748b")
            ax.text(x + .012, y + .052, formula, fontsize=9, color="#475569")
            ax.text(x + .012, y + .012, result, fontsize=12, color=purple if accent else blue, weight="bold")

        z = 10 * (point.sum() - np.array([.5, 1.5]))
        h = sigmoid(z) if nonlinear else z
        ax.text(.005, .79, "EINGABEN", fontsize=8, color="#64748b", weight="bold")
        for i, y in enumerate([.635, .395]):
            frame(.005, y, .115, .09, "#eef4f8")
            ax.text(.062, y + .045, f"x{i+1} = {point[i]}", ha="center", va="center", fontsize=11, weight="bold", color=blue)

        for i, y in enumerate([.585, .345]):
            frame(.235, y, .655 if nonlinear else .40, .215, edge=purple if nonlinear else "#b8cfdf")
            kind = "Logistisches Neuron" if nonlinear else "Lineares Neuron"
            ax.text(.25, y + .178, f"{kind} {i+1}", fontsize=10, weight="bold", color=purple if nonlinear else blue)
            ax.text(.25, y + .142, f"w₁ = 10, w₂ = 10   |   b = −{5 + 10*i}", fontsize=9, color="#475569")
            step(.25, y + .01, .29 if nonlinear else .37, "① Gewichten + Bias", f"10·{point[0]} + 10·{point[1]} − {5 + 10*i}",
                 f"z{i+1} = {z[i]:.1f}" if nonlinear else f"h{i+1} = z{i+1} = {z[i]:.1f}")
            if nonlinear:
                step(.585, y + .01, .29, "② Sigmoid", f"σ({z[i]:.1f})", f"h{i+1} = {h[i]:.3f}", True)
                arrow((.545, y + .065), (.58, y + .065))
            for input_y in [.68, .44]:
                arrow((.125, input_y), (.23, y + .065))
            arrow((.895 if nonlinear else .64, y + .065), (.96, .30), f"h{i+1}")

        score = float(network_score(point, nonlinear, pattern, self.improved))
        prediction = int(score >= 0)
        subtract = pattern == "XOR" and not (adjusted and not nonlinear)
        formula = f"{h[0]:.3f} − {h[1]:.3f} − 0,5" if subtract else f"{h[0]:.3f} + 0 · {h[1]:.3f} − 0,5"
        weights = f"w₃ = 1, w₄ = {'−1' if subtract else '0'}     Bias: b = −0,5"
        frame(.005, .025, .975, .265, edge="#b8cfdf")
        ax.text(.02, .255, "Ausgabeneuron · linear", fontsize=11, weight="bold", color=blue)
        ax.text(.02, .215, "Eingänge: h₁, h₂    |    " + weights, fontsize=9, color="#475569")
        step(.02, .063, .54, "Gewichten + Bias: s = w₃h₁ + w₄h₂ + b", formula, f"Score s = {score:.3f}")
        arrow((.565, .12), (.62, .12))
        step(.625, .063, .335, "Klassenzuordnung", "Klasse 1 bei s ≥ 0",
             f"Klasse {prediction}    {'✓ richtig' if prediction == target else '✗ falsch'}")
        ax.text(.02, -.012,
                "Nur links wurde w₄ angepasst · Alle anderen Parameter bleiben gleich" if adjusted else
                "Identische Parameter in beiden Varianten – auch im Ausgabeneuron", fontsize=8, color="#64748b")

    def update(self, _=None):
        from matplotlib.colors import ListedColormap

        pattern = self.pattern.value_selected
        self.improve_button.ax.set_visible(pattern == "XOR")
        self.improve_button.set_active(pattern == "XOR")
        self.improve_button.label.set_text("Zurück zu gleichen Gewichten (2/4)" if self.improved else "Verbessern auf 3/4")
        targets = np.array([0, 1, 1, 0 if pattern == "XOR" else 1])
        selected = self.examples.value_selected
        point, target = self.points[selected], targets[selected]
        self.task.set_text("Mindestens eine Eingabe ist 1." if pattern == "OR" else "Genau eine Eingabe ist 1.")
        self.target_text.set_text(f"Für ({point[0]}, {point[1]}) soll die Klasse {target} herauskommen.")
        xx, yy = np.meshgrid(np.linspace(-.4, 1.4, 200), np.linspace(-.4, 1.4, 200))
        grid = np.stack((xx, yy), axis=-1)
        colors = ListedColormap(["#176b9c", "#db791e"])
        for ax, diagram, nonlinear in zip(self.axes, self.diagrams, (False, True)):
            self.draw_network(diagram, point, nonlinear, pattern, target)
            ax.clear()
            score = network_score(grid, nonlinear, pattern, self.improved)
            predicted = network_score(self.points, nonlinear, pattern, self.improved) >= 0
            ax.contourf(xx, yy, (score >= 0).astype(int), levels=[-.5, .5, 1.5], cmap=colors, alpha=.16)
            ax.contour(xx, yy, score, levels=[0], colors=["#64748b"], linewidths=1.3)
            ax.scatter(*self.points.T, c=targets, cmap=colors, vmin=0, vmax=1, s=130, edgecolors="white", linewidths=2, zorder=3)
            ax.scatter(*point, s=310, facecolors="none", edgecolors="#7952b3", linewidths=2, zorder=4)
            wrong = self.points[predicted != targets]
            if len(wrong):
                ax.scatter(*wrong.T, s=80, marker="x", color="crimson", zorder=5)
            ax.set(title=f"{sum(predicted == targets)} / 4 richtig", xlabel="x₁", ylabel="x₂",
                   xlim=(-.4, 1.4), ylim=(-.4, 1.4), xticks=[0, 1], yticks=[0, 1], aspect="equal")
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#cbd5e1")
        self.note.set_text("OR: Eine Gerade reicht. Jetzt oben zu XOR wechseln →" if pattern == "OR" else
                           "XOR: Ohne Aktivierung maximal 3/4 – keine Gerade trennt die diagonalen Klassen.")
        self.fig.canvas.draw_idle()


