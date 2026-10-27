"""Example plotting scripts for machine drives."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from numpy.typing import ArrayLike
from collections import deque
import threading

from motulator.common.model._simulation import SimulationResults
from motulator.common.utils._plotting import (
    configure_axes,
    save_and_show,
    set_latex_style,
    set_screen_style,
)
from motulator.common.utils._utils import BaseValues, complex2abc


# %%
def _setup_plot(latex: bool) -> tuple[float, float]:
    """Setup plot style and return figure dimensions."""
    if latex:
        set_latex_style()
        height = plt.rcParams["figure.figsize"][1] * 2.2
    else:
        set_screen_style()
        height = plt.rcParams["figure.figsize"][1] * 1.8

    width = plt.rcParams["figure.figsize"][0]
    return width, height


# %%
def _plot_speeds(ax, ctrl, mdl, base: BaseValues) -> None:
    """Plot angular speeds."""
    if hasattr(ctrl.ref, "w_M") and np.any(ctrl.ref.w_M):
        ax.plot(
            ctrl.t,
            ctrl.ref.w_M / base.w_M,
            "--",
            ds="steps-post",
            label=rf"$\omega_{M}^\mathrm{{ref}}$",
        )
    ax.plot(mdl.t, mdl.machine.w_M / base.w_M, label=rf"$\omega_{M}$")
    ax.plot(
        ctrl.t, ctrl.fbk.w_M / base.w_M, label=rf"$\hat{{\omega}}_{M}$", ds="steps-post"
    )
    ax.legend()


def _plot_torques(ax, ctrl, mdl, base: BaseValues) -> None:
    """Plot torques."""
    # Plot torque reference only if it is nonzero
    if hasattr(ctrl.ref, "tau_M") and np.any(ctrl.ref.tau_M):
        ax.plot(
            ctrl.t,
            ctrl.ref.tau_M / base.tau,
            "--",
            label=rf"$\tau_{M}^\mathrm{{ref}}$",
            ds="steps-post",
        )
    ax.plot(mdl.t, mdl.machine.tau_M / base.tau, label=rf"$\tau_{M}$")
    # Plot estimated torque only if it is nonzero
    if hasattr(ctrl.fbk, "tau_M") and np.any(ctrl.fbk.tau_M):
        ax.plot(
            ctrl.t,
            ctrl.fbk.tau_M / base.tau,
            label=rf"$\hat{{\tau}}_{M}$",
            ds="steps-post",
        )
    if hasattr(mdl.mechanics, "tau_L_tot"):
        ax.plot(
            mdl.t,
            mdl.mechanics.tau_L_tot / base.tau,
            ":",
            label=r"$\tau_\mathrm{L}^\mathrm{tot}$",
        )
    ax.legend(loc="upper right")


def _plot_currents(ax, ctrl, base: BaseValues) -> None:
    """Plot currents."""
    if hasattr(ctrl.ref, "i_s") and np.any(ctrl.ref.i_s):
        ax.plot(
            ctrl.t,
            ctrl.ref.i_s.real / base.i,
            "--",
            label=r"$i_\mathrm{d}^\mathrm{ref}$",
            ds="steps-post",
        )
    ax.plot(
        ctrl.t, ctrl.fbk.i_s.real / base.i, label=r"$i_\mathrm{d}$", ds="steps-post"
    )
    if hasattr(ctrl.ref, "i_s") and np.any(ctrl.ref.i_s):
        ax.plot(
            ctrl.t,
            ctrl.ref.i_s.imag / base.i,
            "--",
            label=r"$i_\mathrm{q}^\mathrm{ref}$",
            ds="steps-post",
        )
    ax.plot(
        ctrl.t, ctrl.fbk.i_s.imag / base.i, label=r"$i_\mathrm{q}$", ds="steps-post"
    )
    ax.legend(loc="upper right")


def _plot_voltages(ax, ctrl, base: BaseValues) -> None:
    """Plot voltages."""
    ax.plot(
        ctrl.t,
        np.abs(ctrl.fbk.u_s) / base.u,
        label=r"$\hat{u}_\mathrm{s}$",
        ds="steps-post",
    )
    ax.plot(
        ctrl.t,
        ctrl.fbk.u_dc / np.sqrt(3) / base.u,
        "--",
        label=r"$u_\mathrm{dc}/\sqrt{3}$",
        ds="steps-post",
    )
    ax.legend(loc="lower right")
    _, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax)


def _plot_flux_linkages(ax, ctrl, mdl, base: BaseValues) -> None:
    """Plot flux linkages."""
    if hasattr(ctrl.ref, "psi_s"):
        ax.plot(
            ctrl.t,
            np.abs(ctrl.ref.psi_s) / base.psi,
            "--",
            label=r"$\psi_\mathrm{s}^\mathrm{ref}$",
            ds="steps-post",
        )
    ax.plot(mdl.t, np.abs(mdl.machine.psi_s_ab) / base.psi, label=r"$\psi_\mathrm{s}$")

    if hasattr(mdl.machine, "psi_R_ab"):
        ax.plot(
            mdl.t,
            np.abs(mdl.machine.psi_R_ab) / base.psi,
            "-.",
            label=r"$\psi_\mathrm{R}$",
        )
    if hasattr(ctrl.fbk, "psi_s"):
        ax.plot(
            ctrl.t,
            np.abs(ctrl.fbk.psi_s) / base.psi,
            label=r"$\hat{\psi}_\mathrm{s}$",
            ds="steps-post",
        )
    ax.legend(loc="upper right")
    _, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax)


def add_sample(self, t: float, values: dict[str, float]) -> None:
    with self.lock:
        for name, value in values.items():
            if name not in self.data:
                continue

            self.data[name].append((t, value))


def _update_plot(self, _frame):
    """Update lines and maintain the rolling time window."""
    with self.lock:
        # Copy data while locked so the simulation can continue
        # publishing samples during plotting.
        snapshots = {
            name: list(samples)
            for name, samples in self.data.items()
        }

    latest_times = [
        samples[-1][0]
        for samples in snapshots.values()
        if samples
    ]

    if not latest_times:
        return list(self.lines.values())

    t_now = max(latest_times)
    t_min = max(0.0, t_now - self.window)

    for name, samples in snapshots.items():
        if not samples:
            continue

        # Retain only points inside the visible time window.
        visible = [
            (t, value)
            for t, value in samples
            if t >= t_min
        ]

        if visible:
            times, values = zip(*visible)
            self.lines[name].set_data(times, values)

    for ax in self.axes:
        ax.set_xlim(t_min, max(t_min + self.window, t_now))

    return list(self.lines.values())


def show(self):
    """Start the Matplotlib GUI event loop."""
    plt.show()


def init_plotting(    
    self,
    window: float = 5.0,
    subplots: list[str] | None = None,
    base: BaseValues | None = None,
    y_lims: list[tuple[float, float] | None] | None = None,
    y_ticks: list[ArrayLike | None] | None = None,
    latex: bool = False,
    interval: int = 50,
    ):

    self.window = window
    self.lock = threading.Lock()

    # Define supported subplot types and their axis labels.
    subplot_map = {
        "speed": ("Speed (rad/s)", "Speed (p.u.)"),
        "torque": ("Torque (Nm)", "Torque (p.u.)"),
        "current": ("Current (A)", "Current (p.u.)"),
        "voltage": ("Voltage (V)", "Voltage (p.u.)"),
        "flux": ("Flux linkage (Vs)", "Flux linkage (p.u.)"),
    }

    if subplots is None:
        subplots = ["speed", "torque", "current", "voltage", "flux"]

    if not subplots:
        raise ValueError("At least one subplot must be selected.")

    for name in subplots:
        if name not in subplot_map:
            raise ValueError(f"Unknown subplot name: {name}")

    self.subplots = subplots

    # Use the existing plotting style.
    width, height = _setup_plot(latex)
    height *= len(subplots) / 5

    # Configure per-unit labels when base values are supplied.
    self.pu_vals = base is not None
    self.base = base if base is not None else BaseValues.unity()

    # Initialize subplot axes.
    self.fig, axes = plt.subplots(
        len(subplots),
        1,
        figsize=(width, height),
        sharex=True,
        squeeze=False,
    )
    self.axes = axes[:, 0]

    # Store recent samples independently for each subplot.
    self.data = {
        name: deque(maxlen=10000)
        for name in subplots
    }

    # Initialize one line per subplot.
    self.lines = {}

    for i, (name, ax) in enumerate(zip(subplots, self.axes)):
        si_label, pu_label = subplot_map[name]

        self.lines[name], = ax.plot(
            [],
            [],
            label=name.capitalize(),
        )

        ax.set_ylabel(pu_label if self.pu_vals else si_label)
        ax.set_xlim(0, window)
        ax.grid(True)
        ax.legend()

        if y_lims is not None and i < len(y_lims):
            if y_lims[i] is not None:
                ax.set_ylim(*y_lims[i])

        if y_ticks is not None and i < len(y_ticks):
            if y_ticks[i] is not None:
                ax.set_yticks(y_ticks[i])

    self.axes[-1].set_xlabel("Time (s)")
    self.fig.tight_layout()

    # Animate existing lines rather than recreating them.
    self.animation = FuncAnimation(
        self.fig,
        self._update_plot,
        interval=interval,
        blit=False,
        cache_frame_data=False,
    )    


#def plot(
#    res: SimulationResults,
#    base: BaseValues | None = None,
#    subplots: list[str] | None = None,
#    t_lims: tuple[float, float] | None = None,
#    t_ticks: ArrayLike | None = None,
#    y_lims: list[tuple[float, float] | None] | None = None,
#    y_ticks: list[ArrayLike | None] | None = None,
#    latex: bool = False,
#    save_path: str | Path | None = None,
#    **savefig_kwargs,
#) -> None:
#    """
#    Plot example figures with selectable subplots.
#
#    Parameters
#    ----------
#    res : SimulationResults
#        Simulation results.
#    base : BaseValues, optional
#        Base values for scaling the waveforms. If not given, the waveforms are plotted
#        in SI units.
#    subplots : list[str], optional
#        List of subplot names to include. Valid names: "speed", "torque", "current",
#        "voltage", "flux". Defaults to all.
#    t_lims : tuple[float, float], optional
#        Time axis limits. If None, uses full time range.
#    t_ticks : ArrayLike, optional
#        Time axis tick locations.
#    y_lims : list[tuple[float, float] | None], optional
#        y-axis limits for each subplot.
#    y_ticks : list[ArrayLike | None], optional
#        y-axis tick locations for each subplot.
#    latex : bool, optional
#        Use LaTeX fonts for the labels. Enabling this option requires a working LaTeX
#        installation, defaults to False.
#    save_path : str | Path, optional
#        Path to save the figure. If None, the figure is not saved.
#    **savefig_kwargs
#        Additional keyword arguments passed to plt.savefig().
#
#    """
#    # Plot selected subplots
#    for ax, name in zip(axes, subplots, strict=True):
#        plot_func, label_si, label_pu = subplot_map[name]
#        if name in ["speed", "torque", "flux"]:
#            plot_func(ax, res.ctrl, res.mdl, base)
#        elif name == "current":
#            plot_func(ax, res.ctrl, base)
#        elif name == "voltage":
#            plot_func(ax, res.ctrl, base)
#        ax.set_ylabel(label_pu if pu_vals else label_si)
#
#    # Configure all axes
#    configure_axes(axes, t_lims, t_ticks, y_lims, y_ticks)
#
#    fig.align_ylabels()
#    axes[-1].set_xlabel("Time (s)")
#
#    save_and_show(save_path, **savefig_kwargs)

