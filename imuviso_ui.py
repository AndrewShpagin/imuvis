"""Interactive preview for the vision/acceleration feedback experiment.

Run without arguments to use ``height.csv`` from the working directory (or
beside this file), or pass a CSV path.  On Windows, a CSV can also be dropped
onto ``imuviso_ui.cmd``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import queue
import threading
import tempfile
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Any

# Keep Matplotlib startup independent of an unwritable user profile cache.
_MPL_CACHE = Path(tempfile.gettempdir()) / "imuviso-matplotlib-cache"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from imuviso import (
    SMALL_ANGLE_RATIO_WARNING,
    average_2d_distance,
    compute_feedback_preview,
    least_squares_affine_trend,
    load_flight_csv,
    resolve_default_csv,
    trajectory_distance_metrics,
)


DEFAULT_GRAVITY = None
DEFAULT_GAIN = 0.3
DEFAULT_ITERATIONS = 4
DEFAULT_ACCELERATION_SIGN = 1
DEBOUNCE_MS = 110


@dataclass(frozen=True)
class _PreviewRequest:
    sequence: int
    csv_path: Path
    start: int
    count: int
    scale_percent: float


@dataclass(frozen=True)
class _PreviewResponse:
    sequence: int
    result: dict[str, Any] | None = None
    error: str | None = None


@dataclass(frozen=True)
class _EndpointPoint:
    tick: int
    scaled_x: float
    integrated_x: float
    scaled_y: float
    integrated_y: float
    initial_bias_x: float
    final_bias_x: float
    initial_bias_y: float
    final_bias_y: float


def _history_point_from_result(result: dict[str, Any]) -> _EndpointPoint:
    """Extract synchronized endpoint and fitted-bias values from one window."""
    axes = result["axes"]
    return _EndpointPoint(
        tick=int(result["start"]) + int(result["count"]) - 1,
        scaled_x=float(axes["x"]["scaled_vision"][-1]),
        integrated_x=float(
            axes["x"]["states"][-1]["integrated_position"][-1]
        ),
        scaled_y=float(axes["y"]["scaled_vision"][-1]),
        integrated_y=float(
            axes["y"]["states"][-1]["integrated_position"][-1]
        ),
        initial_bias_x=float(axes["x"]["states"][0]["bias_m_s2"]),
        final_bias_x=float(axes["x"]["states"][-1]["bias_m_s2"]),
        initial_bias_y=float(axes["y"]["states"][0]["bias_m_s2"]),
        final_bias_y=float(axes["y"]["states"][-1]["bias_m_s2"]),
    )


def _store_endpoint_point(
    history: list[_EndpointPoint], point: _EndpointPoint, capacity: int
) -> None:
    """Store one endpoint, dropping history after resets or large jumps."""
    if history:
        tick_step = point.tick - history[-1].tick
        if tick_step < 0 or tick_step > 10:
            history.clear()
    if history and point.tick == history[-1].tick:
        history[-1] = point
    else:
        history.append(point)
    if len(history) > capacity:
        del history[:-capacity]


class ImuvisoUI:
    """Tk application which previews feedback iterations without writing files."""

    def __init__(
        self,
        root: tk.Misc,
        initial_csv: str | Path | None = None,
        *,
        initial_start: int = 0,
        initial_count: int = 500,
        initial_scale: float = 100.0,
        gravity: float | None = DEFAULT_GRAVITY,
        gain: float = DEFAULT_GAIN,
        iterations: int = DEFAULT_ITERATIONS,
        acceleration_sign: int = DEFAULT_ACCELERATION_SIGN,
        dnd_files: str | None = None,
    ) -> None:
        self.root = root
        self.gravity = gravity
        self.gain = gain
        self.iterations = iterations
        self.rows = 2
        self.show_all_iterations_var = tk.BooleanVar(value=False)
        self.acceleration_sign = acceleration_sign
        self._dnd_files = dnd_files
        self._closing = False
        self._playing = False
        self._reset_start = max(0, int(initial_start))
        self._sequence = 0
        self._suspend_changes = False
        self._syncing_timeline = False
        self._debounce_id: str | None = None
        self._requests: queue.Queue[_PreviewRequest | None] = queue.Queue()
        self._responses: queue.SimpleQueue[_PreviewResponse] = queue.SimpleQueue()
        self._endpoint_history: list[_EndpointPoint] = []
        self._endpoint_context: tuple[Any, ...] | None = None

        self.path_var = tk.StringVar()
        self.scale_var = tk.DoubleVar(value=initial_scale)
        self.start_var = tk.StringVar(value=str(initial_start))
        self.step_var = tk.StringVar(value="1")
        self.timeline_var = tk.DoubleVar(value=initial_start)
        self.count_var = tk.StringVar(value=str(initial_count))
        self.status_var = tk.StringVar(value="Choose a CSV to begin.")

        self._configure_window()
        self._build_controls()
        self._build_plots()
        self._install_change_handlers()
        self._install_drop_target()

        self._worker = threading.Thread(
            target=self._worker_loop,
            name="imuviso-preview",
            daemon=True,
        )
        self._worker.start()
        self.root.after(40, self._poll_responses)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        try:
            default_path = resolve_default_csv(initial_csv)
        except (OSError, ValueError, TypeError) as exc:
            self._set_status(str(exc), error=True)
        else:
            if default_path is not None:
                self.path_var.set(str(default_path))
            else:
                self._set_status(
                    "Choose a CSV. If height.csv is present it is selected automatically.",
                    error=False,
                )

    def _configure_window(self) -> None:
        self.root.title("IMU / vision feedback preview")
        self.root.geometry("1280x900")
        self.root.minsize(900, 650)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        style = ttk.Style(self.root)
        style.configure("Status.TLabel", foreground="#303030")
        style.configure("Error.Status.TLabel", foreground="#a40000")

    def _build_controls(self) -> None:
        controls = ttk.Frame(self.root, padding=(10, 9, 10, 6))
        controls.grid(row=0, column=0, sticky="nsew")
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="CSV").grid(row=0, column=0, padx=(0, 6))
        self.path_entry = ttk.Entry(controls, textvariable=self.path_var)
        self.path_entry.grid(row=0, column=1, sticky="ew")
        ttk.Button(controls, text="Browse...", command=self.browse).grid(
            row=0, column=2, padx=(6, 0)
        )

        timeline = ttk.Frame(controls)
        timeline.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        timeline.columnconfigure(1, weight=1)
        ttk.Label(timeline, text="Timeline (start tick)").grid(
            row=0, column=0, padx=(0, 8)
        )
        self.timeline_slider = tk.Scale(
            timeline, from_=0, to=0, variable=self.timeline_var,
            orient="horizontal", resolution=1, showvalue=True,
            highlightthickness=0, state="disabled",
        )
        self.timeline_slider.grid(row=0, column=1, sticky="ew")

        parameters = ttk.Frame(controls)
        parameters.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        parameters.columnconfigure(2, weight=1)

        ttk.Label(parameters, text="Scale (%)").grid(row=0, column=0, sticky="w")
        self.scale_spinbox = ttk.Spinbox(
            parameters,
            from_=1.0,
            to=300.0,
            increment=1.0,
            textvariable=self.scale_var,
            width=8,
        )
        self.scale_spinbox.grid(row=0, column=1, padx=(6, 8))
        self.scale_slider = tk.Scale(
            parameters,
            length=120,
            from_=10.0,
            to=200.0,
            variable=self.scale_var,
            orient="horizontal",
            resolution=1.0,
            showvalue=False,
            highlightthickness=0,
        )
        self.scale_slider.grid(row=0, column=2, sticky="ew", padx=(0, 18))

        ttk.Label(parameters, text="Start tick").grid(row=0, column=3, sticky="e")
        self.start_spinbox = ttk.Spinbox(
            parameters,
            from_=0,
            to=100_000_000,
            increment=1,
            textvariable=self.start_var,
            width=10,
        )
        self.start_spinbox.grid(row=0, column=4, padx=(6, 18))

        ttk.Label(parameters, text="Step").grid(row=0, column=5, sticky="e")
        self.step_spinbox = ttk.Spinbox(
            parameters, from_=1, to=100_000_000, increment=1,
            textvariable=self.step_var, width=6,
        )
        self.step_spinbox.grid(row=0, column=6, padx=(6, 12))

        self.play_button = ttk.Button(
            parameters, text="Play", command=self.toggle_playback
        )
        self.play_button.grid(row=0, column=7, padx=(0, 6))
        ttk.Button(parameters, text="Reset", command=self.reset_playback).grid(
            row=0, column=8, padx=(0, 18)
        )

        ttk.Label(parameters, text="Tick count").grid(row=0, column=9, sticky="e")
        self.count_spinbox = ttk.Spinbox(
            parameters,
            from_=6,
            to=1_000_000,
            increment=10,
            textvariable=self.count_var,
            width=10,
        )
        self.count_spinbox.grid(row=0, column=10, padx=(6, 8))
        ttk.Button(parameters, text="Refresh", command=self.request_preview_now).grid(
            row=0, column=11
        )

        drop_text = (
            "Drop a CSV anywhere in this window, or drop it onto imuviso_ui.cmd."
            if self._dnd_files
            else "For drag-and-drop, drop a CSV onto imuviso_ui.cmd "
            "(in-window drop is enabled when tkinterdnd2 is installed)."
        )
        ttk.Label(controls, text=drop_text, foreground="#555555").grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(7, 0)
        )
        sign_text = "+DX*G/H" if self.acceleration_sign == 1 else "-DX*G/H"
        ttk.Label(
            controls,
            text=("Geometry: coordinate += DX; acceleration += " + sign_text
                  + "; then bias is refitted and acceleration is reintegrated."),
            foreground="#555555" if self.acceleration_sign == 1 else "#a40000",
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(3, 0))
        ttk.Label(
            controls,
            text=("Realtime endpoints and bias: one estimate per completed window; "
                  "history clears on reset, incompatible settings, or a tick gap > 10."),
            foreground="#555555",
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(3, 0))
        ttk.Checkbutton(
            controls, text="Show all iterations", variable=self.show_all_iterations_var,
            command=self._toggle_iterations,
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(4, 0))


    def _build_position_axes(self) -> None:
        figure_height = min(18.0, max(7.0, 2.6 * self.rows))
        self.position_figure.clear()
        self.position_figure.set_size_inches(11, figure_height)
        self._position_suptitle = self.position_figure.suptitle(
            "Position detail: offset and constant speed removed for display only"
        )
        self.position_axes = self.position_figure.subplots(
            self.rows, 2, sharex=True, sharey="col", squeeze=False
        )
        self.position_lines: dict[tuple[int, str], dict[str, Any]] = {}
        for row in range(self.rows):
            iteration = row if self.show_all_iterations_var.get() else (0 if row == 0 else self.iterations)
            integration_label = (
                "Initial rebiased acceleration integration"
                if row == 0 else f"Rebiased acceleration integration, iteration {iteration}"
            )
            for column, axis_name in enumerate(("x", "y")):
                ax = self.position_axes[row][column]
                original, = ax.plot(
                    [], [], color="0.25", linewidth=0.9,
                    linestyle=":", marker=".", markersize=2.5,
                    markevery=10, zorder=4,
                    label="Raw vision (dotted; overlaps scaled at 100%)"
                )
                coordinate = None
                if row > 0:
                    coordinate, = ax.plot(
                        [], [], color="tab:blue", linewidth=1,
                        label=f"Updated feedback coordinate, iteration {iteration}"
                    )
                scaled, = ax.plot(
                    [], [], color="tab:green", linewidth=1.2,
                    linestyle="--", zorder=3,
                    label="Initial scaled vision (RMSE reference)"
                )
                initial_integrated = None
                if row > 0:
                    initial_integrated, = ax.plot(
                        [], [], color="tab:purple", linewidth=1.4,
                        linestyle="--",
                        label="Initial rebiased acceleration integration"
                    )
                integrated, = ax.plot(
                    [], [], color="tab:orange", linewidth=1.7,
                    label=integration_label
                )
                ax.set_title(
                    f"{axis_name.upper()} iteration {iteration}\nWaiting for data"
                )
                ax.set_ylabel("Detrended position (m)")
                ax.grid(True, alpha=0.35)
                ax.legend(loc="upper left", fontsize="small")
                self.position_lines[(row, axis_name)] = {
                    "original": original,
                    "scaled": scaled,
                    "coordinate": coordinate,
                    "initial_integrated": initial_integrated,
                    "integrated": integrated,
                }
        for ax in self.position_axes[-1]:
            ax.set_xlabel("Time from window start (s)")
        self.position_figure.tight_layout(rect=(0, 0, 1, 0.90))

    def _toggle_iterations(self) -> None:
        self.rows = self.iterations + 1 if self.show_all_iterations_var.get() else 2
        self._build_position_axes()
        self._resize_position_view()
        self.position_canvas.draw_idle()
        self.request_preview_now()

    def _resize_position_view(self, event: Any = None) -> None:
        width = max(1, self.position_scroll.winfo_width())
        height = max(1, self.position_scroll.winfo_height())
        if self.show_all_iterations_var.get():
            height = max(height, self.rows * 360)
        self.position_scroll.itemconfigure(self._position_window, width=width, height=height)
        self.position_scroll.configure(scrollregion=(0, 0, width, height))

    def _build_plots(self) -> None:
        notebook = ttk.Notebook(self.root)
        notebook.grid(row=1, column=0, sticky="nsew", padx=10)

        position_frame = ttk.Frame(notebook)
        comparison_frame = ttk.Frame(notebook)
        bias_frame = ttk.Frame(notebook)
        notebook.add(position_frame, text="Position feedback")
        notebook.add(comparison_frame, text="Realtime endpoints")
        notebook.add(bias_frame, text="Realtime acceleration bias")

        self.position_figure = Figure(figsize=(11, 7), dpi=100)
        self._build_position_axes()
        self.position_scroll = tk.Canvas(position_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(position_frame, orient="vertical", command=self.position_scroll.yview)
        scrollbar.pack(side="right", fill="y")
        self.position_scroll.pack(side="left", fill="both", expand=True)
        self.position_scroll.configure(yscrollcommand=scrollbar.set)
        position_content = ttk.Frame(self.position_scroll)
        self._position_window = self.position_scroll.create_window(0, 0, window=position_content, anchor="nw")
        self.position_scroll.bind("<Configure>", self._resize_position_view)
        self.position_canvas = FigureCanvasTkAgg(
            self.position_figure, master=position_content
        )
        self.position_canvas.get_tk_widget().pack(fill="both", expand=True)

        self.comparison_figure = Figure(figsize=(11, 6), dpi=100)
        self._comparison_suptitle = self.comparison_figure.suptitle(
            "Mean 2D integration-to-initial-scaled-vision distance: waiting for samples"
        )
        comparison_grid = self.comparison_figure.subplots(
            1, 2, sharex=True, squeeze=False
        )
        self.comparison_axes = comparison_grid[0]
        self.comparison_lines: dict[str, dict[str, Any]] = {}
        for column, axis_name in enumerate(("x", "y")):
            ax = self.comparison_axes[column]
            scaled, = ax.plot(
                [], [], color="0.55", linewidth=1, marker=".", markersize=5,
                label="Initial scaled vision endpoint"
            )
            integrated, = ax.plot(
                [], [], color="tab:blue", linewidth=1.5, marker=".",
                markersize=5, label="Final integrated endpoint"
            )
            ax.set_title(f"{axis_name.upper()}: waiting for endpoint samples")
            ax.set_xlabel("CSV tick")
            ax.set_ylabel("Position (m)")
            ax.grid(True, alpha=0.35)
            ax.legend(loc="upper left")
            self.comparison_lines[axis_name] = {
                "scaled": scaled,
                "integrated": integrated,
            }
        self.comparison_figure.tight_layout(rect=(0, 0, 1, 0.93))
        self.comparison_canvas = FigureCanvasTkAgg(
            self.comparison_figure, master=comparison_frame
        )
        self.comparison_canvas.get_tk_widget().pack(fill="both", expand=True)

        self.bias_figure = Figure(figsize=(11, 6), dpi=100)
        self._bias_suptitle = self.bias_figure.suptitle(
            "Fitted constant acceleration bias: waiting for samples"
        )
        bias_grid = self.bias_figure.subplots(
            1, 2, sharex=True, squeeze=False
        )
        self.bias_axes = bias_grid[0]
        self.bias_lines: dict[str, dict[str, Any]] = {}
        for column, axis_name in enumerate(("x", "y")):
            ax = self.bias_axes[column]
            initial, = ax.plot(
                [], [], color="0.5", linewidth=1.1, linestyle="--",
                marker=".", markersize=5, label="Iteration 0 fitted bias"
            )
            final, = ax.plot(
                [], [], color="tab:red", linewidth=1.6, marker=".",
                markersize=5, label=f"Iteration {self.iterations} fitted bias"
            )
            ax.axhline(0.0, color="0.75", linewidth=0.8, zorder=0)
            ax.set_title(f"{axis_name.upper()}: waiting for bias samples")
            ax.set_xlabel("Window endpoint CSV tick")
            ax.set_ylabel("Fitted constant bias (m/s²)")
            ax.grid(True, alpha=0.35)
            ax.legend(loc="upper left")
            self.bias_lines[axis_name] = {
                "initial": initial,
                "final": final,
            }
        self.bias_figure.tight_layout(rect=(0, 0, 1, 0.93))
        self.bias_canvas = FigureCanvasTkAgg(
            self.bias_figure, master=bias_frame
        )
        self.bias_canvas.get_tk_widget().pack(fill="both", expand=True)

        status_frame = ttk.Frame(self.root, padding=(10, 5, 10, 8))
        status_frame.grid(row=2, column=0, sticky="ew")
        status_frame.columnconfigure(0, weight=1)
        self.status_label = ttk.Label(
            status_frame, textvariable=self.status_var, style="Status.TLabel"
        )
        self.status_label.grid(row=0, column=0, sticky="w")

    def _install_change_handlers(self) -> None:
        self.path_var.trace_add("write", self._on_path_changed)
        self.timeline_var.trace_add("write", self._on_timeline_changed)
        self.start_var.trace_add("write", self._sync_timeline)
        for variable in (self.scale_var, self.start_var, self.count_var):
            variable.trace_add("write", self._schedule_preview)
        for widget in (
            self.path_entry,
            self.scale_spinbox,
            self.start_spinbox,
            self.count_spinbox,
        ):
            widget.bind("<Return>", self.request_preview_now)

    def _on_timeline_changed(self, *_args: Any) -> None:
        if self._syncing_timeline or self._suspend_changes:
            return
        self._set_playing(False)
        self.start_var.set(str(int(self.timeline_var.get())))

    def _sync_timeline(self, *_args: Any) -> None:
        try:
            start = int(self.start_var.get())
        except (ValueError, tk.TclError):
            return
        self._syncing_timeline = True
        try:
            self.timeline_var.set(start)
        finally:
            self._syncing_timeline = False

    def _on_path_changed(self, *_args: Any) -> None:
        self.timeline_slider.configure(state="disabled")
        self._set_playing(False)
        self._clear_endpoint_history()
        self._schedule_preview()

    def _install_drop_target(self) -> None:
        if not self._dnd_files:
            return
        try:
            self.root.drop_target_register(self._dnd_files)  # type: ignore[attr-defined]
            self.root.dnd_bind("<<Drop>>", self._on_drop)  # type: ignore[attr-defined]
        except (AttributeError, tk.TclError):
            # The command-file fallback remains available if a particular Tk
            # build cannot register the optional tkinterdnd2 extension.
            self._dnd_files = None

    def _on_drop(self, event: Any) -> str:
        try:
            paths = self.root.tk.splitlist(event.data)
        except (AttributeError, tk.TclError):
            paths = (str(event.data).strip("{}"),)
        if paths:
            candidate = Path(paths[0])
            if candidate.suffix.lower() != ".csv":
                self._set_status("Drop a .csv file.", error=True)
            else:
                self.path_var.set(str(candidate))
        return "break"

    def browse(self) -> None:
        current = Path(self.path_var.get()).expanduser()
        initial_dir = current.parent if current.parent.is_dir() else Path.cwd()
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Choose flight CSV",
            initialdir=str(initial_dir),
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
        )
        if selected:
            self.path_var.set(selected)

    def _schedule_preview(self, *_args: Any) -> None:
        if self._suspend_changes:
            return
        self._sequence += 1
        if self._debounce_id is not None:
            self.root.after_cancel(self._debounce_id)
        sequence = self._sequence
        self._debounce_id = self.root.after(
            DEBOUNCE_MS, lambda: self._queue_preview(sequence)
        )

    def request_preview_now(self, _event: Any = None) -> str:
        self._sequence += 1
        if self._debounce_id is not None:
            self.root.after_cancel(self._debounce_id)
            self._debounce_id = None
        self._queue_preview(self._sequence)
        return "break"

    def toggle_playback(self) -> None:
        """Start or pause playback using the selected tick step."""
        if self._playing:
            self._set_playing(False)
            return
        if self._playback_step() is None:
            return
        self._set_playing(True)
        self.request_preview_now()

    def reset_playback(self) -> None:
        """Pause playback and restore the start tick used at launch."""
        self._set_playing(False)
        self._clear_endpoint_history()
        self._suspend_changes = True
        try:
            self.start_var.set(str(self._reset_start))
        finally:
            self._suspend_changes = False
        self.request_preview_now()

    def _set_playing(self, playing: bool) -> None:
        self._playing = playing
        if hasattr(self, "play_button"):
            self.play_button.configure(text="Pause" if playing else "Play")

    def _clear_endpoint_history(self, *, redraw: bool = True) -> None:
        """Clear the shared realtime endpoint and fitted-bias history."""
        self._endpoint_history.clear()
        self._endpoint_context = None
        if hasattr(self, "comparison_lines"):
            for axis_name in ("x", "y"):
                for line in self.comparison_lines[axis_name].values():
                    line.set_data([], [])
                self.comparison_axes[0 if axis_name == "x" else 1].set_title(
                    f"{axis_name.upper()}: waiting for endpoint samples"
                )
            self._comparison_suptitle.set_text(
                "Mean 2D integration-to-initial-scaled-vision distance: "
                "waiting for samples"
            )
            if redraw:
                self.comparison_canvas.draw_idle()
        if hasattr(self, "bias_lines"):
            for column, axis_name in enumerate(("x", "y")):
                for line in self.bias_lines[axis_name].values():
                    line.set_data([], [])
                self.bias_axes[column].set_title(
                    f"{axis_name.upper()}: waiting for bias samples"
                )
            self._bias_suptitle.set_text(
                "Fitted constant acceleration bias: waiting for samples"
            )
            if redraw:
                self.bias_canvas.draw_idle()

    def _update_endpoint_history(self, result: dict[str, Any]) -> None:
        count = int(result["count"])
        endpoint_tick = int(result["start"]) + count - 1
        context = (
            tuple(result["source_key"]),
            count,
            float(result["scale_percent"]),
        )
        if context != self._endpoint_context:
            self._clear_endpoint_history(redraw=False)
            self._endpoint_context = context

        axes = result["axes"]
        point = _history_point_from_result(result)

        _store_endpoint_point(self._endpoint_history, point, count)

        ticks = [item.tick for item in self._endpoint_history]
        for column, axis_name in enumerate(("x", "y")):
            scaled = [
                getattr(item, f"scaled_{axis_name}")
                for item in self._endpoint_history
            ]
            integrated = [
                getattr(item, f"integrated_{axis_name}")
                for item in self._endpoint_history
            ]
            lines = self.comparison_lines[axis_name]
            lines["scaled"].set_data(ticks, scaled)
            lines["integrated"].set_data(ticks, integrated)
            rmse, mean_absolute_error = trajectory_distance_metrics(
                integrated, scaled
            )
            mean_error = sum(
                integrated_value - scaled_value
                for scaled_value, integrated_value in zip(
                    scaled, integrated
                )
            ) / len(integrated)
            final_iteration = axes[axis_name]["states"][-1]["iteration"]
            self.comparison_axes[column].set_title(
                f"{axis_name.upper()}: realtime endpoints, iteration "
                f"{final_iteration}\n"
                f"{len(ticks)}/{count} points; latest tick {endpoint_tick}\n"
                f"integration vs initial scaled vision: RMSE {rmse:.4g} m; "
                f"mean |error| {mean_absolute_error:.4g} m\n"
                f"mean signed error {mean_error:+.4g} m",
                fontsize="medium",
            )

        mean_2d_distance = average_2d_distance(
            [item.integrated_x for item in self._endpoint_history],
            [item.scaled_x for item in self._endpoint_history],
            [item.integrated_y for item in self._endpoint_history],
            [item.scaled_y for item in self._endpoint_history],
        )
        self._comparison_suptitle.set_text(
            f'Scale {result["scale_percent"]:g}%: average 2D integration-to-'
            f'initial-scaled-vision distance {mean_2d_distance:.4g} m '
            f'(lower is closer)\n'
            f'Window average height: {result["average_height_m"]:.3f} m'
        )

        for column, axis_name in enumerate(("x", "y")):
            initial_biases = [
                getattr(item, f"initial_bias_{axis_name}")
                for item in self._endpoint_history
            ]
            final_biases = [
                getattr(item, f"final_bias_{axis_name}")
                for item in self._endpoint_history
            ]
            lines = self.bias_lines[axis_name]
            lines["initial"].set_data(ticks, initial_biases)
            lines["final"].set_data(ticks, final_biases)
            mean_final = sum(final_biases) / len(final_biases)
            final_std = math.sqrt(sum(
                (value - mean_final) ** 2 for value in final_biases
            ) / len(final_biases))
            latest_change = final_biases[-1] - initial_biases[-1]
            final_iteration = axes[axis_name]["states"][-1]["iteration"]
            self.bias_axes[column].set_title(
                f"{axis_name.upper()}: fitted constant acceleration bias\n"
                f"{len(ticks)}/{count} windows; latest tick {endpoint_tick}\n"
                f"latest iter 0 {initial_biases[-1]:+.4g}; iter "
                f"{final_iteration} {final_biases[-1]:+.4g}; "
                f"change {latest_change:+.4g} m/s²\n"
                f"final history mean {mean_final:+.4g}; "
                f"σ {final_std:.3g} m/s²",
                fontsize="medium",
            )

        latest_bias_magnitude = math.hypot(
            point.final_bias_x, point.final_bias_y
        )
        mean_bias_magnitude = sum(
            math.hypot(item.final_bias_x, item.final_bias_y)
            for item in self._endpoint_history
        ) / len(self._endpoint_history)
        self._bias_suptitle.set_text(
            f'Scale {result["scale_percent"]:g}%: each point is one whole-'
            f'window bias estimate after geometric DX*G/H correction\n'
            f'Final bias magnitude: latest {latest_bias_magnitude:.4g} m/s²; '
            f'history mean {mean_bias_magnitude:.4g} m/s²; '
            f'average height {result["average_height_m"]:.3f} m'
        )

    def _playback_step(self) -> int | None:
        try:
            step = int(self.step_var.get())
            if step < 1:
                raise ValueError
        except (ValueError, tk.TclError):
            self._set_playing(False)
            self._set_status("Step must be a positive whole number of ticks.", error=True)
            return None
        return step

    def _advance_playback(self, result: dict[str, Any]) -> None:
        """Queue the next start tick only after the current result is drawn."""
        if not self._playing or self._closing:
            return
        step = self._playback_step()
        if step is None:
            return
        start = int(result["start"])
        count = int(result["count"])
        last_full_window_start = max(0, int(result["row_count"]) - count)
        if start >= last_full_window_start:
            self._set_playing(False)
            self.status_var.set(
                self.status_var.get() + " Playback reached the last full window."
            )
            return

        self._suspend_changes = True
        try:
            self.start_var.set(str(min(start + step, last_full_window_start)))
        finally:
            self._suspend_changes = False
        self.request_preview_now()

    def _queue_preview(self, sequence: int) -> None:
        self._debounce_id = None
        try:
            path_text = self.path_var.get().strip()
            if not path_text:
                raise ValueError("Choose a CSV file.")
            csv_path = Path(path_text).expanduser()
            scale_percent = float(self.scale_var.get())
            start = int(self.start_var.get())
            count = int(self.count_var.get())
            if not 0 < scale_percent <= 1000:
                raise ValueError("Scale must be greater than 0 and at most 1000%.")
            if start < 0:
                raise ValueError("Start tick must be zero or greater.")
            if count < 6:
                raise ValueError("Tick count must be at least 6.")
        except (ValueError, TypeError, tk.TclError) as exc:
            self._set_playing(False)
            self._set_status(str(exc), error=True)
            return

        self._set_status("Updating preview...", error=False)
        self._requests.put(
            _PreviewRequest(sequence, csv_path, start, count, scale_percent)
        )

    def _worker_loop(self) -> None:
        cached_key: tuple[Path, int, int] | None = None
        cached_flight: Any = None
        while True:
            request = self._requests.get()
            if request is None:
                return

            # Discard requests which became obsolete before work began.
            while True:
                try:
                    newer = self._requests.get_nowait()
                except queue.Empty:
                    break
                if newer is None:
                    return
                request = newer

            try:
                resolved_path = request.csv_path.resolve(strict=True)
                stat = resolved_path.stat()
                cache_key = (resolved_path, stat.st_size, stat.st_mtime_ns)
                if cache_key != cached_key:
                    cached_flight = load_flight_csv(resolved_path)
                    cached_key = cache_key
                if cached_flight.row_count < 6:
                    raise ValueError("CSV must contain at least 6 data rows.")
                count = min(max(request.count, 6), cached_flight.row_count)
                start = min(request.start, cached_flight.row_count - count)
                result = compute_feedback_preview(
                    cached_flight,
                    start,
                    count,
                    request.scale_percent,
                    self.gravity,
                    self.gain,
                    self.iterations,
                    self.acceleration_sign,
                )
                result["row_count"] = cached_flight.row_count
                result["source_key"] = (
                    str(resolved_path), stat.st_size, stat.st_mtime_ns
                )
            except Exception as exc:  # Display data/fit failures in the UI.
                response = _PreviewResponse(request.sequence, error=str(exc))
            else:
                response = _PreviewResponse(request.sequence, result=result)
            self._responses.put(response)

    def _poll_responses(self) -> None:
        if self._closing:
            return
        latest: _PreviewResponse | None = None
        while True:
            try:
                response = self._responses.get_nowait()
            except queue.Empty:
                break
            if response.sequence == self._sequence:
                latest = response
        if latest is not None:
            if latest.error is not None:
                self._set_playing(False)
                self._set_status(latest.error, error=True)
            elif latest.result is not None:
                try:
                    self._apply_result(latest.result)
                except Exception as exc:
                    self._set_playing(False)
                    self._set_status(f"Cannot draw preview: {exc}", error=True)
        self.root.after(40, self._poll_responses)

    def _apply_result(self, result: dict[str, Any]) -> None:
        self._suspend_changes = True
        try:
            self.timeline_slider.configure(
                to=max(0, result["row_count"] - result["count"]),
                state="normal",
            )
            self.start_spinbox.configure(
                to=max(0, result["row_count"] - result["count"])
            )
            self.count_spinbox.configure(to=result["row_count"])
            self.start_var.set(str(result["start"]))
            self.count_var.set(str(result["count"]))
        finally:
            self._suspend_changes = False
        t = result["t"]
        self._update_endpoint_history(result)
        for column, axis_name in enumerate(("x", "y")):
            axis_result = result["axes"][axis_name]
            states = axis_result["states"]
            original_vision = axis_result["original_vision"]
            scaled_vision = axis_result["scaled_vision"]
            initial_roughness = states[0]["roughness_m"]
            raw_display_trend = least_squares_affine_trend(
                t, original_vision
            )
            coordinate_display_trend = least_squares_affine_trend(
                t, states[0]["coordinate"]
            )
            displayed_states = states if self.show_all_iterations_var.get() else (states[0], states[-1])
            for row, state in enumerate(displayed_states):
                position_ax = self.position_axes[row][column]
                position = self.position_lines[(row, axis_name)]
                comparison_center = (
                    scaled_vision + state["integrated_position"]
                ) / 2.0
                # This transform is display-only. Sharing one affine trend
                # preserves every scaled-vision/integration difference.
                comparison_display_trend = least_squares_affine_trend(
                    t, comparison_center
                )
                position["original"].set_data(
                    t, original_vision - raw_display_trend
                )
                position["scaled"].set_data(
                    t, scaled_vision - comparison_display_trend
                )
                if position["coordinate"] is not None:
                    position["coordinate"].set_data(
                        t, state["coordinate"] - coordinate_display_trend
                    )
                if position["initial_integrated"] is not None:
                    position["initial_integrated"].set_data(
                        t, states[0]["integrated_position"]
                        - comparison_display_trend
                    )
                position["integrated"].set_data(
                    t, state["integrated_position"]
                    - comparison_display_trend
                )
                roughness_percent = (
                    100.0 * state["roughness_m"] / initial_roughness
                    if initial_roughness > 0 else 0.0
                )
                position_ax.set_title(
                    f'{axis_name.upper()} iter {state["iteration"]}: '
                    f'integration vs initial scaled vision\n'
                    f'absolute RMSE {state["scaled_vision_rmse_m"]:.4g} m; '
                    f'mean |error| '
                    f'{state["scaled_vision_mean_abs_distance_m"]:.4g} m\n'
                    f'bias {state["bias_m_s2"]:.4g} m/s^2; vision roughness '
                    f'{state["roughness_m"]:.4g} m ({roughness_percent:.1f}%)\n'
                    f'max |DX/H| '
                    f'{state["max_abs_total_coordinate_change_over_height"]:.3g}',
                    fontsize="medium",
                )

        final_x = result["axes"]["x"]["states"][-1]
        final_y = result["axes"]["y"]["states"][-1]
        average_2d_fit_distance = average_2d_distance(
            final_x["integrated_position"],
            result["axes"]["x"]["scaled_vision"],
            final_y["integrated_position"],
            result["axes"]["y"]["scaled_vision"],
        )
        self._position_suptitle.set_text(
            f'Scale {result["scale_percent"]:g}%: absolute final average 2D '
            f'integration-to-scaled-vision distance '
            f'{average_2d_fit_distance:.4g} m; '
            f'average height {result["average_height_m"]:.3f} m\n'
            f'Display only: offset and constant speed removed separately; '
            f'endpoints and metrics remain absolute'
        )

        for ax in self.position_axes.flat:
            ax.relim()
            ax.autoscale_view()
        for ax in self.comparison_axes:
            ax.relim()
            ax.autoscale_view()
        for ax in self.bias_axes:
            ax.relim()
            ax.autoscale_view()
        self.position_canvas.draw_idle()
        self.comparison_canvas.draw_idle()
        self.bias_canvas.draw_idle()

        seconds = float(result.get("compute_seconds", 0.0))
        summaries = [result["axes"][axis]["summary"] for axis in ("x", "y")]
        max_ratio = max(
            summary["max_abs_total_coordinate_change_over_height"]
            for summary in summaries)
        warnings = []
        if max_ratio > SMALL_ANGLE_RATIO_WARNING:
            max_overestimate = max(
                summary["max_small_angle_linearization_overestimate_percent"]
                for summary in summaries)
            warnings.append(
                f"small-angle warning: max |DX/H|={max_ratio:.3g} "
                f"(~{max_overestimate:.2g}% high)")
        nonmonotonic = [
            axis.upper() for axis, summary in zip(("x", "y"), summaries)
            if not summary["scaled_vision_rmse_improved_every_iteration"]]
        if nonmonotonic:
            warnings.append(
                "initial-scaled-vision RMSE did not improve every iteration for "
                + "/".join(nonmonotonic))
        warning_text = "  WARNING: " + "; ".join(warnings) if warnings else ""
        self._set_status(
            f'Showing ticks {result["start"]} through '
            f'{result["start"] + result["count"] - 1} '
            f'({result["count"]} samples), scale '
            f'{result["scale_percent"]:g}%, average height '
            f'{result["average_height_m"]:.3f} m, G={result["gravity_m_s2"]:.5g} '
            f'm/s² ({result["gravity_source"]}) - computed in {seconds:.3f} s.'
            f'{warning_text}',
            error=bool(warnings),
        )
        self._advance_playback(result)

    def _set_status(self, message: str, *, error: bool) -> None:
        self.status_var.set(message)
        if hasattr(self, "status_label"):
            self.status_label.configure(
                style="Error.Status.TLabel" if error else "Status.TLabel"
            )

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._set_playing(False)
        if self._debounce_id is not None:
            self.root.after_cancel(self._debounce_id)
            self._debounce_id = None
        self._requests.put(None)
        self.root.destroy()


def _make_root() -> tuple[tk.Misc, str | None]:
    """Create a normal Tk root, optionally enhanced by tkinterdnd2."""
    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD
    except ImportError:
        return tk.Tk(), None
    return TkinterDnD.Tk(), DND_FILES


def launch_interactive(
    initial_path: str | Path | None = None,
    *,
    initial_start: int = 0,
    initial_count: int = 500,
    initial_scale: float = 100.0,
    gravity: float | None = DEFAULT_GRAVITY,
    gain: float = DEFAULT_GAIN,
    iterations: int = DEFAULT_ITERATIONS,
    acceleration_sign: int = DEFAULT_ACCELERATION_SIGN,
) -> None:
    """Open the interactive feedback preview for an optional CSV path."""
    root, dnd_files = _make_root()
    ImuvisoUI(
        root,
        initial_path,
        initial_start=initial_start,
        initial_count=initial_count,
        initial_scale=initial_scale,
        gravity=gravity,
        gain=gain,
        iterations=iterations,
        acceleration_sign=acceleration_sign,
        dnd_files=dnd_files,
    )
    root.mainloop()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively preview vision/acceleration feedback iterations."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        help="flight CSV (defaults to height.csv when present)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    launch_interactive(_parse_args().csv_path)
