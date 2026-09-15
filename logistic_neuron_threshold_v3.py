"""V3: Ein Neuron mit und ohne Sigmoid. Start: python logistic_neuron_threshold_v3.py

Python >= 3.10. Abhängigkeiten: python -m pip install -r requirements.txt
--snapshot datei.png speichert das ganze Fenster; --step 1/2/3 wählt den Einstieg.
--self-test prüft Mathematik und echte Qt-Bedienelemente ohne sichtbares Fenster.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys

import numpy as np


def sigmoid(z):
    """Überlaufsichere Sigmoidfunktion, auch für große negative Eingaben."""
    values = np.asarray(z, dtype=float)
    e = np.exp(-np.abs(values))
    result = np.where(values >= 0, 1 / (1 + e), e / (1 + e))
    return float(result) if result.ndim == 0 else result


@dataclass
class LogisticNeuron1D:
    weight: float = 1.0
    bias: float = 0.0

    def affine(self, x):
        return self.weight * np.asarray(x) + self.bias

    def eval(self, x):
        return sigmoid(self.affine(x))


def threshold_neuron(threshold, alpha):
    if not np.isfinite([threshold, alpha]).all() or alpha <= 0:
        raise ValueError("Schwellwert muss endlich, Steilheit positiv und endlich sein.")
    return LogisticNeuron1D(alpha, -alpha * threshold)


def build_window(initial_step=1):
    """GUI erst nach CLI-Auswertung laden; Qt-Fenster unabhängig vom IDE-Plotfenster."""
    from PySide6 import QtCore, QtWidgets
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    class NumberControl(QtWidgets.QWidget):
        changed = QtCore.Signal(float)

        def __init__(self, label, low, high, value):
            super().__init__()
            row = QtWidgets.QHBoxLayout(self)
            row.setContentsMargins(0, 3, 0, 3)
            self.caption = QtWidgets.QLabel(label)
            self.caption.setMinimumWidth(145)
            self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.slider.setRange(0, 10000)
            self.slider.setMinimumWidth(140)
            self.spin = QtWidgets.QDoubleSpinBox()
            self.spin.setDecimals(3)
            self.spin.setSingleStep(.1)
            self.spin.setMinimumWidth(105)
            self.low, self.high = low, high
            self.spin.setRange(low, high)
            for widget in (self.caption, self.slider, self.spin):
                row.addWidget(widget)
            row.setStretch(1, 1)
            self.slider.valueChanged.connect(self.from_slider)
            self.spin.valueChanged.connect(self.from_spin)
            self.set_value(value)

        @property
        def value(self):
            return self.spin.value()

        def set_value(self, value):
            value = float(np.clip(value, self.low, self.high))
            blockers = [QtCore.QSignalBlocker(self.spin), QtCore.QSignalBlocker(self.slider)]
            self.spin.setValue(value)
            self.slider.setValue(round((self.value-self.low)/(self.high-self.low)*10000))
            del blockers

        def set_range(self, low, high):
            previous = self.value
            blocker = QtCore.QSignalBlocker(self.spin)
            self.low, self.high = low, high
            self.spin.setRange(low, high)
            del blocker
            self.set_value(previous)

        def from_slider(self, position):
            self.set_value(self.low + position/10000*(self.high-self.low))
            self.changed.emit(self.value)

        def from_spin(self, value):
            self.set_value(value)
            self.changed.emit(self.value)

    class Demo(QtWidgets.QMainWindow):
        STEPS = ["1  So rechnet das Neuron", "2  Gewicht und Bias", "3  Einen Schalter annähern"]

        def __init__(self):
            super().__init__()
            self.setWindowTitle("Logistisches Neuron · V3 Basics")
            self.resize(1240, 900)
            self.setMinimumSize(800, 600)
            self.step = 1
            self.background = None
            self.last_limits = None
            self.full_draws = 0
            root = QtWidgets.QWidget()
            root.setMinimumSize(980, 880)
            self.content = root
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            scroll.setWidget(root)
            self.setCentralWidget(scroll)
            layout = QtWidgets.QVBoxLayout(root)
            layout.setContentsMargins(24, 18, 24, 16)
            layout.setSpacing(10)
            title = QtWidgets.QLabel("Was verändert die Sigmoidfunktion?")
            title.setObjectName("title")
            layout.addWidget(title)
            nav = QtWidgets.QHBoxLayout()
            self.step_buttons = []
            group = QtWidgets.QButtonGroup(self)
            group.setExclusive(True)
            for index, name in enumerate(self.STEPS, 1):
                button = QtWidgets.QPushButton(name)
                button.setCheckable(True)
                button.clicked.connect(lambda checked=False, i=index: self.select_step(i))
                group.addButton(button)
                nav.addWidget(button)
                self.step_buttons.append(button)
            layout.addLayout(nav)

            self.figure = Figure(figsize=(11, 3.6), facecolor="white")
            self.canvas = FigureCanvasQTAgg(self.figure)
            self.canvas.setMinimumHeight(320)
            layout.addWidget(self.canvas, 1)
            self.left, self.right = self.figure.subplots(1, 2)
            self.figure.subplots_adjust(left=.075, right=.97, bottom=.20, top=.80, wspace=.24)
            for ax in (self.left, self.right):
                ax.grid(alpha=.18)
                ax.set_xlabel("Eingabe x")
                ax.axhline(0, color="#94a3b8", lw=.8)
                ax.set_axisbelow(True)
                ax.spines[["top", "right"]].set_visible(False)
            self.left.set_title("Ohne Aktivierungsfunktion", fontsize=13, pad=12)
            self.right.set_title("Mit Sigmoid", fontsize=13, pad=12)
            self.left.set_ylabel("Ausgabe ohne Sigmoid")
            self.right.set_ylabel("Ausgabe mit Sigmoid")
            self.right.set_ylim(-.08, 1.08)
            self.right.set_yticks([0, .5, 1])
            self.right.axhline(.5, color="#94a3b8", lw=.8, ls=":")
            self.line_left, = self.left.plot([], [], color="#2563eb", lw=2.5, animated=True)
            self.line_right, = self.right.plot([], [], color="#2563eb", lw=2.5, animated=True)
            self.hard, = self.right.plot([], [], "--", color="#64748b", lw=1.8, animated=True)
            self.hard_open, = self.right.plot([], [], "o", mfc="white", mec="#64748b", animated=True)
            self.hard_closed, = self.right.plot([], [], "o", color="#64748b", animated=True)
            self.point_left, = self.left.plot([], [], "o", color="#ea580c", ms=8, animated=True)
            self.point_right, = self.right.plot([], [], "o", color="#ea580c", ms=8, animated=True)
            self.guides = [ax.axvline(0, color="#ea580c", alpha=.35, lw=1, animated=True)
                           for ax in (self.left, self.right)]
            self.artists = [self.hard, self.hard_open, self.hard_closed,
                            self.line_left, self.line_right, *self.guides,
                            self.point_left, self.point_right]
            self.canvas.mpl_connect("draw_event", self.on_draw)
            self.timer = QtCore.QTimer(self)
            self.timer.setInterval(16)
            self.timer.setSingleShot(True)
            self.timer.timeout.connect(self.render)

            inputs = QtWidgets.QHBoxLayout()
            self.input_values = {}
            for key, caption in [("x", "Eingabe x"), ("w", "Gewicht w"), ("b", "Bias b")]:
                card = QtWidgets.QLabel()
                card.setObjectName("inputValue")
                card.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                card.setMinimumHeight(56)
                card.setToolTip(caption)
                inputs.addWidget(card, 1)
                self.input_values[key] = card
            layout.addLayout(inputs)
            self.parameter_mapping = QtWidgets.QLabel()
            self.parameter_mapping.setObjectName("parameterMapping")
            self.parameter_mapping.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.parameter_mapping.setMinimumHeight(28)
            layout.addWidget(self.parameter_mapping)

            calculations = QtWidgets.QHBoxLayout()
            calculations.setSpacing(16)
            self.calculations = []
            self.formulas = []
            for title_text, formula in [
                ("Ohne Sigmoid", "z = w · x + b"),
                ("Mit Sigmoid", "y = 1 / (1 + e<sup>−z</sup>)"),
            ]:
                box = QtWidgets.QFrame()
                box.setObjectName("calculation")
                column = QtWidgets.QVBoxLayout(box)
                column.setContentsMargins(16, 10, 16, 10)
                column.setSpacing(5)
                heading = QtWidgets.QLabel(title_text)
                heading.setObjectName("caption")
                general = QtWidgets.QLabel(formula)
                general.setObjectName("formula")
                general.setWordWrap(True)
                self.formulas.append(general)
                substituted = QtWidgets.QLabel()
                substituted.setObjectName("substitution")
                result = QtWidgets.QLabel()
                result.setObjectName("result")
                for label in (heading, general, substituted, result):
                    column.addWidget(label)
                calculations.addWidget(box, 1)
                self.calculations.append((substituted, result))
            layout.addLayout(calculations)
            self.controls = {
                "x": NumberControl("Eingabe x", -6, 6, 2),
                "weight": NumberControl("Gewicht w", -8, 8, 1),
                "bias": NumberControl("Bias b", -6, 6, 0),
                "threshold": NumberControl("Schwellwert t", -4, 4, 3),
                "alpha": NumberControl("Steilheit α", .1, 50, 1),
            }
            for control in self.controls.values():
                layout.addWidget(control)
                control.changed.connect(self.schedule)
            actions = QtWidgets.QHBoxLayout()
            self.midpoint_button = QtWidgets.QPushButton("Eingabe auf Kurvenmitte")
            self.midpoint_button.clicked.connect(self.move_to_midpoint)
            actions.addWidget(self.midpoint_button)
            self.zoom = QtWidgets.QCheckBox("Übergang vergrößern (t ± 0,5)")
            self.zoom.toggled.connect(self.zoom_changed)
            actions.addWidget(self.zoom)
            actions.addStretch()
            reset = QtWidgets.QPushButton("Schritt zurücksetzen")
            reset.clicked.connect(lambda: self.select_step(self.step))
            actions.addWidget(reset)
            layout.addLayout(actions)
            self.setStyleSheet("""
                QMainWindow, QWidget { background: white; color: #172033; font-size: 14px; }
                QLabel#title { font-size: 25px; font-weight: 700; }
                QLabel#inputValue { background: #fff7ed; border-radius: 8px; padding: 5px; }
                QFrame#calculation { background: #eff6ff; border-radius: 8px; }
                QFrame#calculation QLabel { background: transparent; }
                QLabel#caption { color: #526176; font-size: 12px; }
                QLabel#formula { font-size: 19px; }
                QLabel#parameterMapping { color: #526176; font-size: 15px; }
                QLabel#substitution { color: #526176; font-size: 16px; }
                QLabel#result { color: #1d4ed8; font-size: 21px; font-weight: 600; }
                QPushButton { background: #f1f5f9; border: 1px solid #d7e0ea; border-radius: 6px; padding: 9px; }
                QPushButton:checked { background: #1d4ed8; color: white; border-color: #1d4ed8; }
                QPushButton:hover { border-color: #2563eb; }
                QPushButton:disabled { color: #94a3b8; }
                QDoubleSpinBox { border: 1px solid #cbd5e1; border-radius: 4px; padding: 5px; }
                QSlider::groove:horizontal { height: 6px; background: #dbe4ef; border-radius: 3px; }
                QSlider::sub-page:horizontal { background: #2563eb; border-radius: 3px; }
                QSlider::handle:horizontal { background: #2563eb; width: 16px; margin: -5px 0; border-radius: 8px; }
            """)
            self.select_step(initial_step)

        def model(self):
            c = self.controls
            if self.step == 3:
                return threshold_neuron(c["threshold"].value, c["alpha"].value)
            return LogisticNeuron1D(c["weight"].value, c["bias"].value)

        def select_step(self, step):
            self.step = step
            self.timer.stop()
            self.step_buttons[step-1].setChecked(True)
            self.controls["x"].set_range(-6, 6)
            for key, value in {"x": 2, "weight": 1, "bias": 0, "threshold": 3, "alpha": 1}.items():
                self.controls[key].set_value(value)
            blocker = QtCore.QSignalBlocker(self.zoom)
            self.zoom.setChecked(False)
            del blocker
            for key, control in self.controls.items():
                control.setVisible(key == "x" or (step == 2 and key in ("weight", "bias"))
                                   or (step == 3 and key in ("threshold", "alpha")))
            self.zoom.setVisible(step == 3)
            self.parameter_mapping.setVisible(step == 3)
            self.formulas[0].setText(
                "z = α · (x − t) <span style='font-size:16px; color:#526176'>(= w · x + b)</span>"
                if step == 3 else "z = w · x + b")
            self.midpoint_button.setVisible(step != 1)
            self.midpoint_button.setText("Eingabe auf Schwelle" if step == 3 else "Eingabe auf Kurvenmitte")
            handles = [self.line_right, self.point_right]
            labels = ["Sigmoid", "Gewählte Eingabe"]
            self.hard.set_visible(step == 3)
            if step == 3:
                handles.append(self.hard)
                labels.append("Harte Schwelle")
            self.right.legend(handles, labels, loc="upper left", fontsize=8, framealpha=.9)
            self.last_limits = None
            self.render()

        def schedule(self, _=None):
            # Ereignisse zusammenfassen, keinen neuen Timer für jedes Mausereignis starten.
            if not self.timer.isActive():
                self.timer.start()

        def zoom_changed(self, checked):
            if checked:
                self.controls["x"].set_value(self.controls["threshold"].value)
            self.schedule()

        def move_to_midpoint(self):
            model = self.model()
            if model.weight:
                self.controls["x"].set_value(-model.bias/model.weight)
                self.schedule()

        def on_draw(self, _event):
            self.full_draws += 1
            self.background = self.canvas.copy_from_bbox(self.figure.bbox)
            # draw_event kann aus einem Qt-Paint kommen: dort keinen erneuten Paint auslösen.
            self.draw_artists(blit=False)

        def draw_artists(self, blit=True):
            if self.background is None:
                return
            self.canvas.restore_region(self.background)
            for artist in self.artists:
                if artist.get_visible():
                    artist.axes.draw_artist(artist)
            if blit:
                self.canvas.blit(self.figure.bbox)

        def render(self):
            self.timer.stop()
            model = self.model()
            threshold = self.controls["threshold"].value
            low, high = (-6., 6.)
            if self.step == 3 and self.zoom.isChecked():
                low, high = threshold-.5, threshold+.5
            self.controls["x"].set_range(low, high)
            x = self.controls["x"].value
            z, y = float(model.affine(x)), model.eval(x)
            # Grobe Skalenstufen halten die Achse während vieler Sliderbewegungen stabil.
            peak = max(abs(float(model.affine(low))), abs(float(model.affine(high))), 4.)
            extent = 2. ** math.ceil(math.log2(peak * 1.12))
            limits = (low, high, extent)
            grid = np.linspace(low, high, 1201)
            self.line_left.set_data(grid, model.affine(grid))
            self.line_right.set_data(grid, model.eval(grid))
            self.point_left.set_data([x], [z])
            self.point_right.set_data([x], [y])
            for guide in self.guides:
                guide.set_xdata([x, x])
            for artist in (self.hard, self.hard_open, self.hard_closed):
                artist.set_visible(self.step == 3)
            self.hard.set_data([low, threshold, np.nan, threshold, high], [0, 0, np.nan, 1, 1])
            self.hard_open.set_data([threshold], [0])
            self.hard_closed.set_data([threshold], [1])
            def number(value, digits=3):
                return f"{value:.{digits}f}".rstrip("0").rstrip(".") if value else "0"

            def operand(value):
                text = number(value)
                return f"({text})" if value < 0 else text

            values = ([("x", "Eingabe x", x), ("w", "Schwellwert t", threshold),
                       ("b", "Steilheit α", model.weight)] if self.step == 3 else
                      [("x", "Eingabe x", x), ("w", "Gewicht w", model.weight),
                       ("b", "Bias b", model.bias)])
            for key, caption, value in values:
                self.input_values[key].setText(f"{caption}<br><span style='font-size:22px; font-weight:600'>{number(value)}</span>")
                self.input_values[key].setToolTip(caption)
            self.parameter_mapping.setText(
                f"Gewicht: <b>w = α = {number(model.weight)}</b>"
                f" &nbsp;&nbsp;&nbsp; Bias: <b>b = −α · t = −{number(model.weight)} × "
                f"{operand(threshold)} = {number(model.bias, 6)}</b>")
            sign = "−" if model.bias < 0 else "+"
            self.calculations[0][0].setText(
                f"Eingesetzt: {number(model.weight)} × ({number(x)} − {operand(threshold)})"
                if self.step == 3 else
                f"Eingesetzt: {operand(model.weight)} × {operand(x)} {sign} {number(abs(model.bias))}")
            self.calculations[0][1].setText(f"Ausgabe z: {number(z, 6)}")
            self.calculations[1][0].setText(f"Eingesetzt: 1 / (1 + e<sup>−({number(z, 6)})</sup>)")
            self.calculations[1][1].setText(f"Ausgabe y: {y:.6f}")
            midpoint = -model.bias/model.weight if model.weight else None
            reachable = midpoint is not None and low <= midpoint <= high
            self.midpoint_button.setEnabled(reachable)
            self.midpoint_button.setToolTip(
                "Bei Gewicht 0 gibt es keine eindeutige Kurvenmitte." if midpoint is None else
                "Die Kurvenmitte liegt außerhalb des sichtbaren Bereichs." if not reachable else
                f"Kurvenmitte bei x = {midpoint:.3f}")
            if limits != self.last_limits:
                self.left.set_xlim(low, high)
                self.right.set_xlim(low, high)
                self.left.set_ylim(-extent, extent)
                self.last_limits = limits
                self.canvas.draw()
            elif self.background is None:
                self.canvas.draw()
            else:
                self.draw_artists()

        def save_snapshot(self, path):
            self.render()
            QtWidgets.QApplication.processEvents()
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not self.content.grab().save(str(path), "PNG"):
                raise OSError(f"Bild konnte nicht gespeichert werden: {path}")

    return Demo()


def self_test(window):
    """Prüft Fachlogik, Regler, Schrittwechsel, Zoom, Resize und den schnellen Zeichenpfad."""
    from PySide6 import QtWidgets
    import unittest

    class Tests(unittest.TestCase):
        def test_sigmoid(self):
            np.testing.assert_allclose(sigmoid([-np.log(3), 0, np.log(3)]), [.25, .5, .75])
            with np.errstate(over="raise", invalid="raise"):
                self.assertTrue(np.isfinite(sigmoid([-1e6, 1e6])).all())

        def test_threshold(self):
            for t in [-3, 0, 3]:
                n = threshold_neuron(t, 20)
                self.assertEqual(n.eval(t), .5)
                self.assertLess(n.eval(t-.2), .02)
                self.assertGreater(n.eval(t+.2), .98)
            with self.assertRaises(ValueError):
                threshold_neuron(3, 0)

        def test_input_and_fast_render(self):
            window.select_step(1)
            before = window.full_draws
            c = window.controls["x"]
            c.slider.setValue(7500)  # echte Qt-Signalverbindung: x = 3
            window.render()
            self.assertEqual(c.value, 3)
            self.assertEqual(window.point_left.get_ydata()[0], 3)
            self.assertAlmostEqual(window.point_right.get_ydata()[0], sigmoid(3))
            self.assertEqual(window.full_draws, before)

        def test_parameters_and_zero(self):
            window.select_step(2)
            window.controls["bias"].spin.setValue(-3)
            window.move_to_midpoint()
            window.render()
            self.assertEqual(window.controls["x"].value, 3)
            self.assertEqual(window.point_right.get_ydata()[0], .5)
            window.controls["weight"].spin.setValue(0)
            window.render()
            self.assertFalse(window.midpoint_button.isEnabled())
            np.testing.assert_allclose(window.line_left.get_ydata(), -3)
            window.controls["weight"].spin.setValue(-2)
            window.render()
            self.assertTrue(np.all(np.diff(window.line_right.get_ydata()) < 0))

        def test_threshold_zoom_and_reset(self):
            window.select_step(3)
            window.controls["alpha"].spin.setValue(50)
            window.zoom.setChecked(True)
            window.render()
            window.controls["x"].slider.setValue(5100)
            window.render()
            self.assertAlmostEqual(window.controls["x"].value, 3.01)
            self.assertEqual(window.model().bias, -150)
            window.controls["threshold"].spin.setValue(-4)
            window.render()
            self.assertTrue(-4.5 <= window.controls["x"].value <= -3.5)
            window.move_to_midpoint()
            window.render()
            self.assertEqual(window.point_right.get_ydata()[0], .5)
            window.select_step(1)
            self.assertEqual(window.controls["x"].value, 2)
            self.assertEqual(window.model(), LogisticNeuron1D())
            self.assertFalse(window.hard.get_visible())

        def test_resize_and_bounds(self):
            window.select_step(2)
            for w, b in [(8, 6), (-8, -6), (.01, 6)]:
                window.controls["weight"].spin.setValue(w)
                window.controls["bias"].spin.setValue(b)
                window.render()
                low, high = window.left.get_ylim()
                values = window.line_left.get_ydata()
                self.assertTrue(np.all((values >= low) & (values <= high)))
            window.resize(1000, 800)
            QtWidgets.QApplication.processEvents()
            window.render()
            self.assertIsNotNone(window.background)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    return result.wasSuccessful()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.snapshot or args.self_test:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["QT_API"] = "pyside6"
    try:
        from PySide6 import QtWidgets
    except ImportError:
        print("Bitte Abhängigkeiten installieren: python -m pip install -r requirements.txt", file=sys.stderr)
        return 2
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    window = build_window(args.step)
    window.show()
    app.processEvents()
    if args.self_test and not self_test(window):
        return 1
    if args.snapshot:
        window.select_step(args.step)
        window.save_snapshot(args.snapshot)
        print(args.snapshot.resolve())
    if args.self_test or args.snapshot:
        window.close()
        return 0
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
