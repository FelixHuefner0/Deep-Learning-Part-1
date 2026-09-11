"""Logistic neuron: interactive course demonstration, exports and self-tests.

Run this file to open the demo. Use --help for non-interactive options.
Install dependencies with: python -m pip install -r requirements.txt
The default GUI is a separate Qt window (PySide6), including when run in an IDE.
No training or internet connection is needed after installation.
The original LogisticNeuron1D(weight, bias).eval(x) / .plot() API is retained.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import sys
import unittest

import numpy as np


def configure_backend(*, headless=False, gui_backend="QtAgg"):
    """Choose explicitly BEFORE pyplot creates figures; avoid static IDE backends.

    PyCharm can select module://backend_interagg, which renders widget drawings
    as PNGs. Merely rejecting the plain Agg backend does not catch that case.
    Tests/exports deliberately use Agg and do not need the GUI dependencies.
    """
    import matplotlib

    if not headless and gui_backend == "QtAgg":
        os.environ.setdefault("QT_API", "pyside6")
    matplotlib.use("Agg" if headless else gui_backend, force=True)


def sigmoid(z):
    """Stable logistic activation for a scalar or array (no exp(+large))."""
    values = np.asarray(z, dtype=float)
    # Both np.where branches are evaluated: make the exponential safe first.
    e = np.exp(-np.abs(values))
    result = np.where(values >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
    return float(result) if result.ndim == 0 else result


@dataclass
class LogisticNeuron1D:
    """One affine transformation followed by the fixed sigmoid activation."""

    weight: float = 1.0
    bias: float = 0.0

    def __post_init__(self):
        if not np.isfinite([self.weight, self.bias]).all():
            raise ValueError("Gewicht und Bias müssen endlich sein.")

    def affine(self, x):
        """First step: z = weight * x + bias (affine, not generally linear)."""
        result = self.weight * np.asarray(x, dtype=float) + self.bias
        return float(result) if result.ndim == 0 else result

    def eval(self, x):
        """Second step: y = sigmoid(z) = 1 / (1 + exp(-z))."""
        return sigmoid(self.affine(x))

    @property
    def midpoint(self):
        """Unique x with output 0.5, or None for weight zero."""
        return -self.bias / self.weight if self.weight != 0 else None

    def plot(self, x_min=-10, x_max=10, num_points=1000):
        """Compatibility helper for the original examples; opens one plot."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        plot_neurons(ax, [self], x_min=x_min, x_max=x_max,
                     num_points=num_points, title="Logistisches Neuron")
        fig.tight_layout()
        plt.show()
        return fig, ax


def neuron_for_threshold(threshold, alpha):
    """sigma(alpha * (x - threshold)); alpha > 0 means rising threshold."""
    if not np.isfinite([threshold, alpha]).all() or alpha <= 0:
        raise ValueError("Schwelle muss endlich und alpha endlich sowie > 0 sein.")
    return LogisticNeuron1D(weight=float(alpha), bias=float(-alpha * threshold))


def hard_threshold(x, threshold):
    """Reference convention: 0 for x < threshold, 1 for x >= threshold."""
    result = (np.asarray(x, dtype=float) >= threshold).astype(float)
    return float(result) if result.ndim == 0 else result


def transition_width(weight):
    """Distance between sigmoid outputs 0.1 and 0.9; undefined for w=0."""
    return None if weight == 0 else 2 * np.log(9) / abs(weight)


def _draw_hard_threshold(ax, threshold, x_min, x_max):
    """Separate segments and open/closed dots: no fictitious vertical values."""
    color = "#bc6515"
    if x_min <= threshold <= x_max:
        ax.plot([x_min, threshold], [0, 0], "--", color=color, label="Harte Schwelle")
        ax.plot([threshold, x_max], [1, 1], "--", color=color)
        ax.plot(threshold, 0, "o", mfc=ax.get_facecolor(), mec=color, zorder=5)
        ax.plot(threshold, 1, "o", color=color, zorder=5)
    else:
        value = hard_threshold((x_min + x_max) / 2, threshold)
        ax.plot([x_min, x_max], [value, value], "--", color=color, label="Harte Schwelle")


def plot_neurons(ax, neurons, *, x_min=-10, x_max=10, num_points=2001,
                 title="Vergleich", threshold=None):
    """Reusable static comparison with shared axes and explicit parameters."""
    if x_min >= x_max or num_points < 2:
        raise ValueError("x_min < x_max und mindestens zwei Punkte erforderlich.")
    grid = np.linspace(x_min, x_max, num_points)
    for neuron in neurons:
        ax.plot(grid, neuron.eval(grid), lw=2,
                label=f"w={neuron.weight:g}, b={neuron.bias:g}")
    if threshold is not None:
        _draw_hard_threshold(ax, threshold, x_min, x_max)
        ax.axvline(threshold, color="0.5", lw=0.8)
    ax.axhline(0.5, color="0.7", lw=0.8)
    ax.set(xlim=(x_min, x_max), ylim=(-0.06, 1.06), xlabel="Eingabe x",
           ylabel="Ausgabe y", title=title, yticks=[0, 0.5, 1])
    ax.grid(alpha=0.18)
    ax.legend(fontsize=9, loc="best")
    return ax


def export_comparisons(directory):
    """Reproducible figures for the presentation; no interactive backend needed."""
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    experiments = [
        ("01_weight", "Gewicht ändern, Bias = 0",
         [LogisticNeuron1D(w, 0) for w in [0.5, 1, 5, 20]], None),
        ("02_bias", "Bias ändern, Gewicht = 1",
         [LogisticNeuron1D(1, b) for b in [-5, 0, 5]], None),
        ("03_special_cases", "Negatives Gewicht und Gewicht 0",
         [LogisticNeuron1D(w, 0) for w in [-5, 0, 5]], None),
        ("04_threshold", "Feste Schwelle t = -3",
         [neuron_for_threshold(-3, a) for a in [1, 5, 20, 100]], -3),
    ]
    paths = []
    for name, title, neurons, threshold in experiments:
        if threshold is None:
            fig, ax = plt.subplots(figsize=(9, 5.5))
            plot_neurons(ax, neurons, title=title)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            plot_neurons(axes[0], neurons, title=title, threshold=threshold)
            plot_neurons(axes[1], neurons, x_min=-3.5, x_max=-2.5,
                         title="Zoom: am Schwellwert bleibt y = 0,5", threshold=threshold)
        fig.tight_layout()
        path = directory / f"{name}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


class ButtonSelector:
    """Exclusive button group with the selection API used by the demo."""

    def __init__(self, fig, values, labels, positions):
        from matplotlib.widgets import Button

        self.values = tuple(values)
        self.value_selected = self.values[0]
        self.callbacks = []
        self.buttons = []
        for index, (label, position) in enumerate(zip(labels, positions)):
            button = Button(fig.add_axes(position), label, hovercolor="#dceaf4")
            button.label.set_fontsize(10)
            for spine in button.ax.spines.values():
                spine.set_visible(False)
            button.on_clicked(lambda event, i=index: self.set_active(i))
            self.buttons.append(button)
        self._style()

    def _style(self):
        for value, button in zip(self.values, self.buttons):
            selected = value == self.value_selected
            button.color = "#176b9c" if selected else "#eef2f6"
            button.hovercolor = "#12577f" if selected else "#dceaf4"
            button.ax.set_facecolor(button.color)
            button.label.set_color("white" if selected else "#334155")
            button.label.set_fontweight("bold" if selected else "normal")

    def set_active(self, index):
        self.value_selected = self.values[index]
        self._style()
        for callback in self.callbacks:
            callback(self.value_selected)
        self.buttons[0].ax.figure.canvas.draw_idle()

    def on_clicked(self, callback):
        self.callbacks.append(callback)


class LogisticDemo:
    """Matplotlib UI; model and math above remain independent of the widgets."""

    FREE = "Gewicht / Bias"
    THRESHOLD = "Feste Schwelle"
    PRESETS = ("Standard", "Steil", "Verschoben", "Negativ", "Konstant", "Schwelle -3")

    def __init__(self, export_dir="exports"):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider

        self.export_dir = Path(export_dir)
        self._updating = False
        self._background = None
        self._needs_full_draw = True
        self._render_pending = False
        self._last_view = None
        self.fig = plt.figure(figsize=(15, 9.5), facecolor="white")
        self.fig.canvas.manager.set_window_title("Logistisches Neuron - interaktive Demo")
        self.fig.suptitle("Ein Neuron, zwei Rechenschritte", y=0.98,
                          fontsize=20, fontweight="bold")
        self.fig.text(0.5, 0.937, "Gewicht und Bias verändern den Eingang der Sigmoidfunktion. Die Aktivierung selbst bleibt gleich.",
                      ha="center", fontsize=11, color="0.3")
        self.ax_affine = self.fig.add_axes([0.065, 0.585, 0.26, 0.29])
        self.ax_activation = self.fig.add_axes([0.39, 0.585, 0.23, 0.29])
        self.ax_output = self.fig.add_axes([0.69, 0.585, 0.285, 0.29])
        self.values = self.fig.text(0.065, 0.505, "", fontsize=12, fontweight="bold")
        self.explanation = self.fig.text(0.065, 0.467, "", fontsize=10)
        self.status = self.fig.text(0.065, 0.027, "Regler ziehen; PNG speichert den aktuellen Zustand.", fontsize=10)

        # Two pairs share positions, but only the pair for the selected mode is active.
        def slider(label, lo, hi, initial, row, step):
            control = Slider(self.fig.add_axes([0.15, row, 0.37, 0.024]), label,
                             lo, hi, valinit=initial, valstep=step, valfmt="%1.1f",
                             color="#176b9c", track_color="#e2e8f0",
                             handle_style={"facecolor": "white", "edgecolor": "#176b9c", "size": 8})
            control.label.set_fontsize(10)
            control.valtext.set_color("#176b9c")
            control.valtext.set_fontweight("bold")
            control.drawon = False  # One shared renderer; no full redraw per widget.
            control.vline.set_visible(False)
            return control

        self.sliders = {
            "weight": slider("Gewicht w", -10, 10, 1, 0.387, 0.1),
            "bias": slider("Bias b", -6, 6, 0, 0.318, 0.1),
            "alpha": slider("Steilheit α", 0.1, 100, 5, 0.387, 0.1),
            "threshold": slider("Schwelle t", -5, 5, -3, 0.318, 0.1),
            "x": slider("Eingabe x", -10, 10, 0, 0.249, 0.1),
            "radius": slider("Halbe x-Breite", 0.1, 10, 10, 0.180, 0.1),
        }
        self.range_text = self.fig.text(0.065, 0.147, "", fontsize=9, color="0.35")
        self.fig.text(0.065, 0.128, "Grenzen passen sich an; Eingabepunkt und Mitte bleiben sichtbar.", fontsize=9, color="0.35")
        self.fig.text(0.065, 0.429, "PARAMETER", fontsize=9, color="#64748b", weight="bold")
        self.fig.text(0.64, 0.429, "DARSTELLUNG", fontsize=9, color="#64748b", weight="bold")
        self.mode = ButtonSelector(
            self.fig, [self.FREE, self.THRESHOLD],
            ["w / b · Frei", "α / t · Feste Schwelle"],
            [[0.64, 0.367, 0.16, 0.046], [0.802, 0.367, 0.173, 0.046]])
        self.fig.text(0.64, 0.340, "Bei α / t ist die harte Referenz immer sichtbar.", fontsize=9, color="#64748b")
        self.fig.text(0.64, 0.294, "BEISPIELE", fontsize=9, color="#64748b", weight="bold")
        self.presets = ButtonSelector(
            self.fig, self.PRESETS, self.PRESETS,
            [[0.64 + (i % 2) * 0.171, 0.234 - (i // 2) * 0.052, 0.164, 0.042]
             for i in range(len(self.PRESETS))])
        self.buttons = {}
        for label, x, width, callback in [
            ("x = Mitte", 0.065, 0.13, self.move_to_midpoint),
            ("Zurücksetzen", 0.22, 0.13, self.reset),
            ("PNG speichern", 0.64, 0.155, self.save_snapshot),
        ]:
            button = Button(self.fig.add_axes([x, 0.075, width, 0.043]), label,
                            color="#eef2f6", hovercolor="#dceaf4")
            button.label.set_color("#334155")
            button.label.set_fontsize(10)
            for spine in button.ax.spines.values():
                spine.set_visible(False)
            button.on_clicked(callback)
            self.buttons[label] = button
        self._create_plot_artists()
        self._animated = [*self._plot_artists, self.values, self.explanation,
                          self.range_text, self.status]
        for control in self.sliders.values():
            # Slider has no public handle accessor; isolated here for Matplotlib 3.10+.
            self._animated.extend([control.poly, control._handle, control.valtext])
        self._use_blit = self.fig.canvas.supports_blit
        for artist in self._animated:
            artist.set_animated(self._use_blit)
        self.fig.canvas.mpl_connect("draw_event", self._on_draw)
        self.fig.canvas.mpl_connect("resize_event", self._invalidate)
        self.fig.canvas.mpl_connect("close_event", self._on_close)
        self._timer = self.fig.canvas.new_timer(interval=16)
        self._timer.single_shot = True
        self._timer.add_callback(self._render)
        for control in self.sliders.values():
            control.on_changed(self.update)
        self.mode.on_clicked(self.update)
        self.presets.on_clicked(self.apply_preset)
        self.update()

    def _create_plot_artists(self):
        """Create axes, grid, curves and markers ONCE. Updates only change data."""
        from matplotlib.ticker import MaxNLocator

        blue, orange = "#176b9c", "#bc6515"
        for ax in (self.ax_affine, self.ax_activation, self.ax_output):
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#cbd5e1")
            ax.grid(alpha=0.18)
            ax.tick_params(labelsize=9)
            ax.set_navigate(False)  # Zoom slider owns the bounded viewport.
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        a = self.ax_affine
        a.set(xlim=(-10, 10), ylim=(-12, 12), xlabel="Eingabe x", ylabel="Zwischenwert z",
              title="1. Affin: z = w x + b")
        a.axhline(0, color="0.6", lw=0.8)
        self.affine_line, = a.plot([], [], color=blue, lw=2.2)
        self.affine_point, = a.plot([], [], "o", color=orange, ms=7)
        note = a.text(0.03, 0.95, "Fester Ausschnitt: z von -12 bis 12", transform=a.transAxes,
                      va="top", fontsize=8, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
        a = self.ax_activation
        a.set(xlim=(-12, 12), ylim=(-0.06, 1.06), xlabel="Zwischenwert z", ylabel="Aktivierung",
              title="2. Sigmoid: y = sigma(z)", yticks=[0, 0.5, 1])
        z_grid = np.linspace(-12, 12, 601)
        a.plot(z_grid, sigmoid(z_grid), color=blue, lw=2.2)
        a.axhline(0.5, color="0.6", lw=0.8)
        a.axvline(0, color="0.6", lw=0.8)
        self.activation_point, = a.plot([], [], "o", color=orange, ms=7)
        a = self.ax_output
        a.set(xlim=(-10, 10), ylim=(-0.06, 1.06), xlabel="Eingabe x", ylabel="Ausgabe y",
              title="Gesamt: y = sigma(w x + b)", yticks=[0, 0.5, 1])
        a.axhline(0.5, color="0.6", lw=0.8)
        self.output_line, = a.plot([], [], color=blue, lw=2.2, label="Neuron")
        self.middle_line = a.axvline(0, color="0.6", lw=0.8)
        self.middle_point, = a.plot([], [], "s", color=blue, ms=5)
        self.hard_left, = a.plot([], [], "--", color=orange, label="Harte Schwelle")
        self.hard_right, = a.plot([], [], "--", color=orange)
        self.hard_open, = a.plot([], [], "o", mfc="white", mec=orange)
        self.hard_closed, = a.plot([], [], "o", color=orange)
        self.output_point, = a.plot([], [], "o", color=orange, ms=7, label="Gewählte Eingabe")
        self.legend_free = a.legend(handles=[self.output_line, self.output_point], loc="upper left", fontsize=8)
        a.add_artist(self.legend_free)
        self.legend_hard = a.legend(handles=[self.output_line, self.hard_left, self.output_point],
                                     loc="upper left", fontsize=8)
        self._plot_artists = [self.affine_line, self.affine_point, note,
                              self.activation_point, self.output_line, self.middle_line,
                              self.middle_point, self.hard_left, self.hard_right,
                              self.hard_open, self.hard_closed, self.output_point,
                              self.legend_free, self.legend_hard]

    def _set_bounds(self, key, lo, hi):
        """Change a slider's actual draggable range and clamp its stored value."""
        slider = self.sliders[key]
        old = float(slider.val)
        bounds_changed = slider.valmin != lo or slider.valmax != hi
        if bounds_changed:
            if key in ("bias", "x"):
                first = int(np.ceil((lo - 1e-10) * 10))
                last = int(np.floor((hi + 1e-10) * 10))
                slider.valstep = np.arange(first, last + 1, dtype=float) / 10
            slider.valmin, slider.valmax = float(lo), float(hi)
            slider.ax.set_xlim(lo, hi)
            slider.poly.set_x(lo)
        value = float(np.clip(old, lo, hi))
        if not bounds_changed and value == old:
            return False
        events = slider.eventson
        slider.eventson = False
        try:
            slider.set_val(value)
        finally:
            slider.eventson = events
        return value != old

    def _constrain_controls(self):
        """Keep midpoint in view and input inside both x and z plot ranges."""
        locked = self.mode.value_selected == self.THRESHOLD
        # Clamp programmatic set_val too (Slider only clamps mouse input itself).
        for key, bounds in {"weight": (-10, 10), "alpha": (0.1, 100),
                            "threshold": (-5, 5), "radius": (0.1, 10)}.items():
            self._set_bounds(key, *bounds)
        radius = self.sliders["radius"].val
        center = self.sliders["threshold"].val if locked else 0.0
        lo, hi = max(-10.0, center-radius), min(10.0, center+radius)
        bias_clamped = False
        if not locked:
            w = self.sliders["weight"].val
            # 15% margin keeps the transition away from the plot boundary.
            bias_limit = min(6.0, 0.85 * radius * abs(w)) if w != 0 else 4.0
            bias_clamped = self._set_bounds("bias", -bias_limit, bias_limit)
        neuron = self.current_neuron()
        margin = 0.02 * (hi-lo)  # Keep the entire point marker inside the frame.
        xlo, xhi = lo+margin, hi-margin
        if neuron.weight != 0:
            z_endpoints = sorted(((-10-neuron.bias)/neuron.weight,
                                  (10-neuron.bias)/neuron.weight))
            xlo, xhi = max(xlo, z_endpoints[0]), min(xhi, z_endpoints[1])
        x_clamped = self._set_bounds("x", xlo, xhi)
        if bias_clamped or x_clamped:
            names = " und ".join(name for name, changed in [("Bias", bias_clamped), ("Eingabe x", x_clamped)] if changed)
            self.status.set_text(f"{names} an den sichtbaren Bereich angepasst.")
        return locked, neuron, lo, hi

    def _draw_dynamic(self):
        for artist in self._animated:
            if artist.get_visible() and (artist.axes is None or artist.axes.get_visible()):
                self.fig.draw_artist(artist)

    def _on_draw(self, event):
        if self.fig.canvas.is_saving() or not self._use_blit:
            self._background = None
            return
        self._background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self._needs_full_draw = False
        self._draw_dynamic()

    def _invalidate(self, _=None):
        self._background = None
        self._needs_full_draw = True

    def _on_close(self, _=None):
        self._timer.stop()

    def _request_render(self):
        # Agg has no GUI event loop; tests render synchronously when requested.
        if not self._render_pending:
            self._render_pending = True
            self._timer.start()

    def _render(self):
        self._timer.stop()
        self._render_pending = False
        canvas = self.fig.canvas
        if self._needs_full_draw or self._background is None or not self._use_blit:
            self._needs_full_draw = False
            canvas.draw()
        else:
            canvas.restore_region(self._background)
            self._draw_dynamic()
            canvas.blit(self.fig.bbox)

    def current_neuron(self):
        if self.mode.value_selected == self.THRESHOLD:
            return neuron_for_threshold(self.sliders["threshold"].val, self.sliders["alpha"].val)
        return LogisticNeuron1D(self.sliders["weight"].val, self.sliders["bias"].val)

    def update(self, _=None):
        if self._updating:
            return
        locked, neuron, lo, hi = self._constrain_controls()
        for key in ("weight", "bias", "alpha", "threshold"):
            active = (key in ("alpha", "threshold")) == locked
            if self.sliders[key].ax.get_visible() != active:
                self.sliders[key].ax.set_visible(active)
                self._invalidate()
            self.sliders[key].set_active(active)
        point_x = float(self.sliders["x"].val)
        point_z = neuron.affine(point_x)
        point_y = neuron.eval(point_x)
        if self._last_view != (lo, hi):
            self.ax_affine.set_xlim(lo, hi)
            self.ax_output.set_xlim(lo, hi)
            self._last_view = (lo, hi)
            self._invalidate()
        # Extra samples near the transition keep alpha=100 accurate at wide zoom.
        grid = np.linspace(lo, hi, 601)
        if neuron.midpoint is not None:
            local = neuron.midpoint + np.linspace(-8, 8, 401) / abs(neuron.weight)
            grid = np.unique(np.r_[grid, local[(local >= lo) & (local <= hi)]])
        self.affine_line.set_data([lo, hi], neuron.affine([lo, hi]))
        self.affine_point.set_data([point_x], [point_z])
        self.activation_point.set_data([point_z], [point_y])
        self.output_line.set_data(grid, neuron.eval(grid))
        self.output_point.set_data([point_x], [point_y])
        has_midpoint = neuron.midpoint is not None
        self.middle_line.set_visible(has_midpoint)
        self.middle_point.set_visible(has_midpoint)
        if has_midpoint:
            self.middle_line.set_xdata([neuron.midpoint, neuron.midpoint])
            self.middle_point.set_data([neuron.midpoint], [0.5])
        show_hard = locked
        for artist in (self.hard_left, self.hard_right, self.hard_open, self.hard_closed):
            artist.set_visible(show_hard)
        if show_hard:
            t = self.sliders["threshold"].val
            self.hard_left.set_data([lo, t], [0, 0])
            self.hard_right.set_data([t, hi], [1, 1])
            self.hard_open.set_data([t], [0])
            self.hard_closed.set_data([t], [1])
        self.legend_free.set_visible(not show_hard)
        self.legend_hard.set_visible(show_hard)
        b = self.sliders["bias"]
        x = self.sliders["x"]
        bias_range = "Bias automatisch" if locked else f"Bias: [{b.valmin:.3g}, {b.valmax:.3g}]"
        self.range_text.set_text(f"{bias_range}    |    Eingabe x: [{x.valmin:.3g}, {x.valmax:.3g}]")

        self.values.set_text(f"x = {point_x:.2f}    |    z = {neuron.weight:g} × ({point_x:.2f}) "
                             f"+ ({neuron.bias:g}) = {point_z:.3f}    |    y = {point_y:.6f}")
        width = transition_width(neuron.weight)
        if neuron.midpoint is None:
            info = f"w = 0: konstante Ausgabe sigma(b) = {point_y:.6f}; keine eindeutige Kurvenmitte."
        else:
            info = (f"Mitte: x = {neuron.midpoint:g}, y = 0,5    |    "
                    f"Steigung dort: {neuron.weight / 4:g}    |    10-90%-Breite: {width:.4f}")
        if locked:
            info += f"\nBias automatisch: b = -alpha × t = {neuron.bias:g}. H_t(t) = 1; Neuron(t) = 0,5."
        else:
            info += "\nBias wird bei Bedarf begrenzt, damit die Mitte im Bild bleibt."
        self.explanation.set_text(info)
        self._request_render()

    def apply_preset(self, name):
        settings = {
            "Standard": (0, {"weight": 1, "bias": 0, "x": 0, "radius": 10}),
            "Steil": (0, {"weight": 10, "bias": 0, "x": 0.1, "radius": 10}),
            "Verschoben": (0, {"weight": 1, "bias": -5, "x": 5, "radius": 10}),
            "Negativ": (0, {"weight": -5, "bias": 0, "x": 1, "radius": 10}),
            "Konstant": (0, {"weight": 0, "bias": 0, "x": 2, "radius": 10}),
            "Schwelle -3": (1, {"alpha": 5, "threshold": -3, "x": -3, "radius": 2}),
        }
        mode_index, values = settings[name]
        keep_threshold = self.mode.value_selected == self.THRESHOLD
        if keep_threshold:
            if name in ("Negativ", "Konstant"):
                self.status.set_text(
                    f"{name}: benötigt Gewicht / Bias. Die feste steigende Schwelle bleibt erhalten.")
                self._request_render()
                return
            # Examples change steepness, while the user's threshold stays fixed.
            mode_index = 1
            values = {"alpha": values.get("alpha", values.get("weight", 1)),
                      "x": self.sliders["threshold"].val,
                      "radius": values["radius"]}
        self._updating = True
        try:
            self.mode.set_active(mode_index)
            for key, value in values.items():
                self.sliders[key].set_val(value)
        finally:
            self._updating = False
        if keep_threshold:
            self.status.set_text(f"Beispiel {name}: Steilheit angepasst; Schwelle bleibt erhalten.")
        else:
            self.status.set_text(f"Beispiel geladen: {name}. Regler passen den Ausgangszustand an.")
        self.update()

    def reset(self, _=None):
        self._updating = True
        try:
            for slider in self.sliders.values():
                slider.reset()
            self.mode.set_active(0)
            self.presets.set_active(0)
        finally:
            self._updating = False
        self.status.set_text("Standardzustand wiederhergestellt.")
        self.update()

    def move_to_midpoint(self, _=None):
        midpoint = self.current_neuron().midpoint
        if midpoint is None:
            self.status.set_text("Bei w = 0 gibt es keine eindeutige Mitte.")
        else:
            # set_val accepts exact values, even between the user drag increments.
            self.sliders["x"].set_val(midpoint)
            self.status.set_text("Eingabe auf die Kurvenmitte gesetzt: y = 0,5.")
        self._request_render()

    def save_figure(self, path, **kwargs):
        """Include animated figure text and widget artists in static exports."""
        flags = [artist.get_animated() for artist in self._animated]
        try:
            for artist in self._animated:
                artist.set_animated(False)
            self.fig.savefig(path, **kwargs)
        finally:
            for artist, flag in zip(self._animated, flags):
                artist.set_animated(flag)
            self._invalidate()
            self._request_render()

    def save_snapshot(self, _=None):
        try:
            self.export_dir.mkdir(parents=True, exist_ok=True)
            path = self.export_dir / f"neuron_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
            self.save_figure(path, dpi=160, facecolor="white")
            self.status.set_text(f"PNG gespeichert: {path}")
            print(f"PNG gespeichert: {path.resolve()}")
            return path
        except OSError as exc:
            self.status.set_text(f"Speichern fehlgeschlagen: {exc}")
            return None
        finally:
            self._request_render()


def run_self_tests():
    """Independent mathematical properties plus headless UI state integration."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    class NeuronTests(unittest.TestCase):
        def test_backend_selection_overrides_ide(self):
            from unittest.mock import patch

            # An IDE-provided backend must not decide the GUI path. Mocking
            # use() also keeps this test runnable on machines without Qt.
            with patch("matplotlib.use") as use:
                configure_backend()
                use.assert_called_once_with("QtAgg", force=True)
            with patch("matplotlib.use") as use:
                configure_backend(headless=True)
                use.assert_called_once_with("Agg", force=True)
            with patch("matplotlib.use") as use:
                configure_backend(gui_backend="TkAgg")
                use.assert_called_once_with("TkAgg", force=True)

        def test_known_values_and_arrays(self):
            np.testing.assert_allclose(sigmoid([-np.log(3), 0, np.log(3)]), [.25, .5, .75])
            self.assertIsInstance(sigmoid(0), float)
            self.assertEqual(LogisticNeuron1D(2, -6).affine(4), 2)

        def test_extreme_values(self):
            with np.errstate(over="raise", invalid="raise"):
                result = sigmoid(np.array([-1e6, -1000, 0, 1000, 1e6]))
                self.assertTrue(np.isfinite(result).all())
                self.assertEqual(LogisticNeuron1D(100, -500).eval(-10), 0)

        def test_direction_and_constant(self):
            grid = np.linspace(-1, 1, 100)
            self.assertTrue((np.diff(LogisticNeuron1D(2).eval(grid)) > 0).all())
            self.assertTrue((np.diff(LogisticNeuron1D(-2).eval(grid)) < 0).all())
            np.testing.assert_allclose(LogisticNeuron1D(0, np.log(3)).eval(grid), .75)
            self.assertIsNone(LogisticNeuron1D(0).midpoint)

        def test_threshold_and_boundary_convention(self):
            for t in [-3, 0, 5]:
                for alpha in [.1, 5, 100]:
                    n = neuron_for_threshold(t, alpha)
                    self.assertAlmostEqual(n.eval(t), .5)
                    self.assertLess(n.eval(t-.1), .5)
                    self.assertGreater(n.eval(t+.1), .5)
            np.testing.assert_equal(hard_threshold([2.9, 3, 3.1], 3), [0, 1, 1])

        def test_transition_width_and_error_bound(self):
            n = neuron_for_threshold(-3, 20)
            half = transition_width(n.weight) / 2
            np.testing.assert_allclose(n.eval([-3-half, -3+half]), [.1, .9])
            alpha = np.log(99) / .1
            n = neuron_for_threshold(3, alpha)
            np.testing.assert_allclose(n.eval([2.9, 3.1]), [.01, .99])

        def test_validation(self):
            for alpha in [0, -1, np.nan, np.inf]:
                with self.assertRaises(ValueError):
                    neuron_for_threshold(0, alpha)
            with self.assertRaises(ValueError):
                LogisticNeuron1D(np.inf)

        def test_ui_modes_presets_and_reset(self):
            app = LogisticDemo()
            try:
                app.mode.set_active(1)
                app.sliders["threshold"].set_val(3)
                app.sliders["alpha"].set_val(100)
                self.assertEqual(app.current_neuron().bias, -300)
                self.assertFalse(app.sliders["bias"].active)
                self.assertFalse(app.sliders["bias"].ax.get_visible())
                app.move_to_midpoint()
                self.assertEqual(app.current_neuron().eval(app.sliders["x"].val), .5)
                for index, name in enumerate(app.PRESETS):
                    app.presets.set_active(index)
                    self.assertTrue(np.isfinite(app.current_neuron().eval(0)))
                    self.assertEqual(app.mode.value_selected, app.THRESHOLD)
                    self.assertEqual(app.sliders["threshold"].val, 3)
                    self.assertTrue(app.hard_left.get_visible())
                    self.assertTrue(app.legend_hard.get_visible())
                app.sliders["radius"].set_val(.1)
                app.reset()
                self.assertEqual(app.current_neuron(), LogisticNeuron1D())
                self.assertEqual(app.sliders["radius"].val, 10)
                self.assertFalse(app.hard_left.get_visible())
                self.assertTrue(app.sliders["weight"].active)
            finally:
                plt.close(app.fig)

        def test_visible_bounds_for_parameter_changes(self):
            app = LogisticDemo()
            try:
                # Reproduce the user's offscreen example, then zoom and cross w=0.
                cases = [(0, "weight", .8), (0, "bias", -30.5),
                         (0, "radius", .1), (0, "weight", 0), (0, "bias", 99),
                         (0, "weight", -.05), (0, "x", 100),
                         (1, "alpha", 100), (1, "threshold", 5), (1, "x", -100)]
                for mode, key, value in cases:
                    app.mode.set_active(mode)
                    app.sliders[key].set_val(value)
                    neuron = app.current_neuron()
                    x = app.sliders["x"].val
                    lo, hi = app.ax_output.get_xlim()
                    self.assertLessEqual(lo, x)
                    self.assertLessEqual(x, hi)
                    self.assertLessEqual(abs(neuron.affine(x)), 10 + 1e-10)
                    if neuron.midpoint is not None:
                        self.assertLessEqual(lo, neuron.midpoint)
                        self.assertLessEqual(neuron.midpoint, hi)
                    self.assertAlmostEqual(app.output_point.get_ydata()[0], neuron.eval(x))
            finally:
                plt.close(app.fig)

        def test_artists_reused_and_fast_path(self):
            from unittest.mock import patch
            app = LogisticDemo()
            try:
                app._render()
                identities = tuple(id(a) for a in app._plot_artists)
                with patch.object(app.ax_output, "clear", side_effect=AssertionError("Axes rebuilt")):
                    with patch.object(app.fig.canvas, "draw", wraps=app.fig.canvas.draw) as draw:
                        app.sliders["weight"].set_val(5)
                        app._render()
                        self.assertEqual(draw.call_count, 0)
                self.assertEqual(identities, tuple(id(a) for a in app._plot_artists))
                app.sliders["radius"].set_val(2)
                app._render()  # Rebuild cached ticks/background after zoom.
                self.assertEqual(app.ax_output.get_xlim(), (-2, 2))
            finally:
                plt.close(app.fig)

        def test_real_slider_mouse_events(self):
            from matplotlib.backend_bases import MouseEvent
            app = LogisticDemo()
            try:
                app.fig.canvas.draw()
                slider = app.sliders["weight"]
                px, py = slider.ax.transData.transform((7, .5))
                for kind in ["button_press_event", "button_release_event"]:
                    event = MouseEvent(kind, app.fig.canvas, px, py, button=1)
                    app.fig.canvas.callbacks.process(kind, event)
                self.assertAlmostEqual(app.current_neuron().weight, 7)
            finally:
                plt.close(app.fig)

    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(NeuronTests))
    return result.wasSuccessful()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Mathematik und UI-Zustände ohne Fenster prüfen")
    parser.add_argument("--export-dir", type=Path, help="Vier feste Vergleichsgrafiken exportieren, ohne Fenster")
    parser.add_argument("--snapshot", type=Path, help="Aktuellen Demo-Startzustand ohne Fenster als PNG speichern")
    parser.add_argument("--preset", choices=LogisticDemo.PRESETS, default="Standard", help="Startbeispiel")
    parser.add_argument("--output-dir", type=Path, default=Path("exports"), help="Ziel für den PNG-Knopf")
    parser.add_argument("--backend", choices=["QtAgg", "TkAgg"], default="QtAgg",
                        help="Interaktives Fenster: QtAgg (PySide6, Standard) oder TkAgg")
    args = parser.parse_args(argv)
    if args.self_test and not run_self_tests():
        return 1
    headless = bool(args.self_test or args.export_dir or args.snapshot)
    try:
        configure_backend(headless=headless, gui_backend=args.backend)
    except (ImportError, RuntimeError) as exc:
        print(f"Fenster-Backend konnte nicht gestartet werden: {exc}", file=sys.stderr)
        print("Installiere die requirements.txt im verwendeten Python-Interpreter.\n"
              "Starte das Skript im lokalen Terminal einer grafischen Sitzung.", file=sys.stderr)
        return 2
    import matplotlib.pyplot as plt
    if args.export_dir:
        for path in export_comparisons(args.export_dir):
            print(path.resolve())
    if args.snapshot:
        app = LogisticDemo(args.output_dir)
        app.presets.set_active(app.PRESETS.index(args.preset))
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        app.save_figure(args.snapshot, dpi=130)
        plt.close(app.fig)
        print(args.snapshot.resolve())
    if not headless:
        try:
            from activation_intro import ActivationIntro
            intro = ActivationIntro()
            plt.show(block=True)
            if not intro.proceed:
                return 0
            app = LogisticDemo(args.output_dir)  # Keep widgets alive until window closes.
            app.presets.set_active(app.PRESETS.index(args.preset))
            print(f"Interaktives Fenster: {plt.get_backend()}. Regler im separaten Fenster bedienen.")
            plt.show(block=True)
        except (ImportError, RuntimeError) as exc:
            print(f"Interaktives Fenster konnte nicht geöffnet werden: {exc}\n"
                  "1. python -m pip install -r requirements.txt\n"
                  "2. In der IDE 'Show plots in tool window' deaktivieren.\n"
                  "3. Skript im lokalen Terminal mit grafischer Sitzung starten.\n"
                  "Optional mit installiertem Tk: --backend TkAgg\n"
                  "Ohne Fenster: --self-test, --snapshot oder --export-dir.", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
