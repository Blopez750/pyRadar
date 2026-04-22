
import numpy as np
import json
import os


# =============================================================================
# Settings persistence
# =============================================================================
# GUI settings (dynamic range, CFAR params, MTI toggle, etc.) are saved to a
# JSON file in the project root so they survive between sessions.  The file
# is loaded automatically when the GUI is built (init_rd_gui), and saved
# whenever the user clicks "Apply All" or toggles a processing button.
#
# Settings saved:  dyn_range, cfar, mti, range_norm, fft_all_bins,
#                  cfar_guard, cfar_training, cfar_bias_db,
#                  zero_range_bins, max_range_m, vel_bins
#
# Hardware controls (beam angle, TX gain, element phase) are NOT persisted
# because they depend on the physical setup which may change between sessions.
#
_SETTINGS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'rd_gui_settings.json',
)


def _load_gui_settings():
    """Load persisted GUI settings from disk, or return empty dict."""
    try:
        with open(_SETTINGS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_gui_settings(settings):
    """Persist GUI settings to disk."""
    try:
        with open(_SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
    except OSError as e:
        print(f"Warning: could not save GUI settings: {e}")


# ---------------------------------------------------------------------------
# FMCW Range — PyQtGraph GUI  (replaces old matplotlib plot_fmcw_data)
# ---------------------------------------------------------------------------

def init_fmcw_range_gui():
    """Build the 3-panel FMCW Range PyQtGraph GUI.

    Returns a dict of handles used by update_fmcw_range_gui().
    """
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets, QtCore

    app = ensure_qt_app()

    win = pg.GraphicsLayoutWidget(title="FMCW Range")
    win.resize(1100, 900)

    # ── Panel 0: Raw RX Subarray Data ────────────────────────────────────
    p0 = win.addPlot(row=0, col=0, title="Raw RX Subarray Data")
    p0.setLabel('bottom', "Sample Index")
    p0.setLabel('left', "Amplitude")
    p0.addLegend(offset=(60, 10))
    p0.showGrid(x=True, y=True, alpha=0.3)
    p0.setXRange(0, 2000)
    curve_sub1 = p0.plot(pen=pg.mkPen('r', width=1), name="Subarray 1 (RX)")
    curve_sub2 = p0.plot(pen=pg.mkPen('g', width=1), name="Subarray 2 (RX)")
    curve_sub4 = p0.plot(pen=pg.mkPen('#4488ff', width=1), name="Subarray 4 (RX)")

    # ── Panel 1: Combined Data Stream ────────────────────────────────────
    p1 = win.addPlot(row=1, col=0, title="Combined Data Stream")
    p1.setLabel('bottom', "Sample Index")
    p1.setLabel('left', "Amplitude")
    p1.addLegend(offset=(60, 10))
    p1.showGrid(x=True, y=True, alpha=0.3)
    p1.setXRange(0, 2000)
    curve_sum = p1.plot(pen=pg.mkPen('#aa44ff', width=1), name="Combined Data")

    # ── Panel 2: Range Profile ─────────────────────────────────────────
    p2 = win.addPlot(row=2, col=0, title="Range Profile")
    p2.setLabel('bottom', "Range", units="m")
    p2.setLabel('left', "Magnitude")
    p2.showGrid(x=True, y=True, alpha=0.3)
    p2.setXRange(0, 100)
    curve_beat_fft = p2.plot(pen=pg.mkPen('#44cc44', width=1))
    peak_line = pg.InfiniteLine(angle=90, pen=pg.mkPen('r', width=1, style=QtCore.Qt.DashLine))
    p2.addItem(peak_line)
    peak_label = pg.TextItem(anchor=(1, 0), color='r')
    peak_label.setFont(pg.QtGui.QFont("Arial", 10))
    p2.addItem(peak_label)
    dist_label = pg.TextItem(anchor=(1, 0), color='#4488ff')
    dist_label.setFont(pg.QtGui.QFont("Arial", 12, pg.QtGui.QFont.Bold))
    p2.addItem(dist_label)

    win.showMaximized()

    return {
        'app': app,
        'win': win,
        # Panel 0
        'p0': p0, 'curve_sub1': curve_sub1, 'curve_sub2': curve_sub2, 'curve_sub4': curve_sub4,
        # Panel 1
        'p1': p1, 'curve_sum': curve_sum,
        # Panel 2 (beat FFT)
        'p2': p2, 'curve_beat_fft': curve_beat_fft,
        'peak_line': peak_line, 'peak_label': peak_label, 'dist_label': dist_label,
        '_closed': False,
    }


def update_fmcw_range_gui(gui, sub1, sub2, sub4, sum_data,
                           R_calculated, range_axis, yf_mag,
                           s_beat, sample_rate, T, BW):
    """Push one frame of data into the FMCW Range GUI."""
    c = 3e8

    # Panel 0: raw subarray traces (first 2000 samples)
    n_show = min(2000, len(sub1))
    gui['curve_sub1'].setData(sub1[:n_show].real)
    gui['curve_sub2'].setData(sub2[:n_show].real)
    gui['curve_sub4'].setData(sub4[:n_show].real)

    # Panel 1: combined data
    n_show_sum = min(2000, len(sum_data))
    gui['curve_sum'].setData(sum_data[:n_show_sum].real)

    # Panel 2: range profile in dB
    if s_beat is not None and sample_rate is not None:
        N_beat = len(s_beat)
        beat_fft = np.fft.fft(s_beat)
        beat_freq_axis = np.fft.fftfreq(N_beat, 1 / sample_rate)

        positive_mask = beat_freq_axis >= 0
        beat_freq_positive = beat_freq_axis[positive_mask]
        beat_fft_positive = np.abs(beat_fft[positive_mask])

        # Convert to range axis
        range_axis_plot = c * T * beat_freq_positive / (2 * BW)

        beat_fft_mag = beat_fft_positive / N_beat

        gui['curve_beat_fft'].setData(range_axis_plot, beat_fft_mag)

        gui['peak_line'].setValue(R_calculated)
        gui['peak_label'].setText(f"Peak: {R_calculated:.2f} m")
        gui['peak_label'].setPos(R_calculated + 1, np.max(beat_fft_mag) * 0.9)
        gui['dist_label'].setText(f"Range: {R_calculated:.2f} m")
        max_range = np.max(range_axis_plot)
        gui['dist_label'].setPos(max_range * 0.95, np.max(beat_fft_mag) * 0.9)

    gui['app'].processEvents()

def init_fmcw_radar_viewer(
    angles_deg,
    ranges_m,
    *,
    beat_freq_hz=None,           # Optional. If provided, enables bottom beat-spectrum panel
    initial_heatmap=None,        # Optional (N_ranges, N_angles). If None, zeros used.
    initial_spectra=None,        # Optional (N_freq, N_angles). If None, created on first update.
    window_title="FMCW Radar – Live",
    cmap_name="inferno",
    levels=None,                 # None=auto (5–99th pct). Or (vmin, vmax).
    show_crosshair=True,
    start_event_loop=False       # Usually False for streaming; call processEvents() in your loop.
):
    """
    Create the FMCW radar viewer and return a dict of handles for streaming updates.

    Shapes:
        angles_deg: (N_angles,)
        ranges_m:   (N_ranges,)
        initial_heatmap: (N_ranges, N_angles) or None
        beat_freq_hz: (N_freq,) or None
        initial_spectra: (N_freq, N_angles) or None
    """
    import numpy as np
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets

    # --- Sanity / shapes ---
    angles = np.asarray(angles_deg, float)
    ranges = np.asarray(ranges_m, float)
    n_angles = len(angles)
    n_ranges = len(ranges)

    if initial_heatmap is None:
        H = np.zeros((n_ranges, n_angles), dtype=float)
    else:
        H = np.asarray(initial_heatmap, float)
        if H.shape != (n_ranges, n_angles):
            raise ValueError(f"initial_heatmap shape {H.shape} must be (len(ranges_m), len(angles_deg)) = {(n_ranges, n_angles)}")

    if beat_freq_hz is not None:
        f_axis = np.asarray(beat_freq_hz, float)
        n_freq = len(f_axis)
        if initial_spectra is None:
            S = np.full((n_freq, n_angles), np.nan, dtype=float)
        else:
            S = np.asarray(initial_spectra, float)
            if S.shape != (n_freq, n_angles):
                raise ValueError(f"initial_spectra shape {S.shape} must be (len(beat_freq_hz), len(angles_deg)) = {(n_freq, n_angles)}")
    else:
        f_axis = None
        S = None

    # --- Build UI ---
    app = pg.mkQApp(window_title)
    win = pg.GraphicsLayoutWidget(title=window_title)
    win.resize(1000, 700)
    
 
    # Top: heatmap
    plot_heat = win.addPlot(row=0, col=0)
    plot_heat.setLabel('bottom', 'Azimuth (deg)')
    plot_heat.setLabel('left', 'Range (m)')
    plot_heat.setTitle("PRESS Q TO EXIT --> Range vs Azimuth")
    vb = plot_heat.getViewBox()
    vb.invertY(True)   # 0 m at top, range increases downward (matches working commit)
    win.showMinimized()

    img = pg.ImageItem()
    plot_heat.addItem(img)

    try:
        cmap = pg.colormap.get(cmap_name)
    except Exception:
        cmap = pg.colormap.get('inferno')
    img.setColorMap(cmap)

    # Levels
    if levels is None:
        vmin, vmax = -60.0, 0.0
    else:
        vmin, vmax = levels

    # Map image rect (bin-centered)
    dx = (angles[-1] - angles[0]) / (n_angles - 1) if n_angles > 1 else 1.0
    dy = (ranges[-1] - ranges[0]) / (n_ranges - 1) if n_ranges > 1 else 1.0
    x0 = angles[0] - 0.5 * dx
    y0 = ranges[0] - 0.5 * dy
    W = dx * n_angles
    Hh = dy * n_ranges

    H_dB_init = 20.0 * np.log10(np.clip(H, 1e-6, None))
    img.setImage(H_dB_init.T, levels=(vmin, vmax))
    img.setRect(QtCore.QRectF(x0, y0, W, Hh))

    # Align Y-axis ticks to bin boundaries
    bin_step = float(ranges[1] - ranges[0]) if n_ranges > 1 else 1.0
    plot_heat.getAxis('left').setTickSpacing(float(bin_step * 5), float(bin_step))
    plot_heat.setYRange(y0, y0 + Hh, padding=0)

    # Colorbar
    try:
        cbar = pg.ColorBarItem(colorMap=cmap, values=(vmin, vmax))
        cbar.setImageItem(img)
        win.addItem(cbar, row=0, col=1)
    except Exception:
        cbar = None

    # Crosshair
    if show_crosshair:
        angle_line = pg.InfiniteLine(pos=float(angles[0]), angle=90,
                                     movable=True,
                                     pen=pg.mkPen((0, 255, 255), width=2, style=QtCore.Qt.DashLine))
        plot_heat.addItem(angle_line)
    else:
        angle_line = None

    # Bottom: beat spectrum
    if f_axis is not None:
        plot_beat = win.addPlot(row=1, col=0)
        plot_beat.setLabel('bottom', 'Beat Frequency (Hz)')
        plot_beat.setLabel('left', 'Amplitude')
        plot_beat.setTitle("Beat Spectrum @ angle: --")
        beat_curve = plot_beat.plot([], [], pen=pg.mkPen((0, 255, 255), width=2))
    else:
        plot_beat = None
        beat_curve = None

    # Helper: update beat for a given angle value
    def update_for_angle(angle_val):
        if f_axis is None or S is None:
            return
        idx = int(np.clip(np.argmin(np.abs(angles - angle_val)), 0, n_angles - 1))
        if angle_line is not None:
            angle_line.setValue(float(angles[idx]))
        plot_beat.setTitle(f"Beat Spectrum @ {angles[idx]:.2f}°")
        y = S[:, idx]
        if np.all(np.isnan(y)):
            beat_curve.setData([], [])
        else:
            beat_curve.setData(f_axis, y)

    # Mouse move live update
    if plot_heat is not None and f_axis is not None:
        def on_mouse_moved(evt):
            pos = evt[0] if isinstance(evt, tuple) else evt
            if plot_heat.sceneBoundingRect().contains(pos):
                mouse_point = vb.mapSceneToView(pos)
                update_for_angle(mouse_point.x())

        proxy = pg.SignalProxy(plot_heat.scene().sigMouseMoved, rateLimit=60, slot=on_mouse_moved)
    else:
        proxy = None

    # Angle line drag
    if angle_line is not None and f_axis is not None:
        angle_line.sigPositionChanged.connect(lambda: update_for_angle(angle_line.value()))

    win.show()
    if start_event_loop:
        pg.exec()

    return {
        # UI
        'app': app, 'window': win, 'plot_heat': plot_heat, 'image_item': img,
        'colorbar': cbar, 'plot_beat': plot_beat, 'beat_curve': beat_curve,
        'angle_line': angle_line, 'mouse_proxy': proxy,
        # Data & axes
        'H': H, 'S': S, 'angles': angles, 'ranges': ranges, 'f_axis': f_axis,
        # Config
        'levels': (vmin, vmax)
    }


def update_fmcw_radar_viewer(
    handles,
    angle_idx,
    *,
    heatmap_col=None,
    spectrum=None,
    rescale=False,
    show_this_angle=True,
    fixed_levels=None,
    dyn_range=None,
):
    import numpy as np

    H = handles['H']
    S = handles['S']
    angles = handles['angles']
    f_axis = handles['f_axis']
    img = handles['image_item']
    cbar = handles.get('colorbar', None)

    n_ranges, n_angles = H.shape
    angle_idx = int(np.clip(angle_idx, 0, n_angles - 1))

    # Update heatmap column
    if heatmap_col is not None:
        col = np.asarray(heatmap_col, float)
        if col.shape[0] != n_ranges:
            raise ValueError(f"heatmap_col length {col.shape[0]} must equal number of ranges {n_ranges}")

        H[:, angle_idx] = col

        # Convert to dB for display (linear magnitudes span many orders)
        H_dB = 20.0 * np.log10(np.clip(H, 1e-6, None))

        # Level scaling in dB: peak minus dynamic range
        if fixed_levels is not None:
            handles['levels'] = fixed_levels
        elif dyn_range is not None:
            peak_dB = float(np.nanmax(H_dB))
            if np.isfinite(peak_dB):
                vmax = peak_dB
                vmin = peak_dB - dyn_range
            else:
                vmax = 0.0
                vmin = -60.0
            handles['levels'] = (vmin, vmax)
        else:
            vmin = float(np.nanmin(H_dB))
            vmax = float(np.nanmax(H_dB))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = -60.0, 0.0
            handles['levels'] = (vmin, vmax)
        lv = handles['levels']
        img.setImage(H_dB.T, levels=lv, autoLevels=False)
        if cbar is not None:
            cbar.setLevels(lv)

    # Update spectrum
    if spectrum is not None and S is not None and f_axis is not None:
        spec = np.asarray(spectrum, float)
        if spec.shape[0] != S.shape[0]:
            n = min(spec.shape[0], S.shape[0])
            spec = spec[:n]
            S[:n, angle_idx] = spec
        else:
            S[:, angle_idx] = spec

        if show_this_angle and handles.get('plot_beat') is not None:
            handles['beat_curve'].setData(f_axis[:len(spec)], spec)
            if handles.get('angle_line') is not None:
                _n = len(angles)
                handles['angle_line'].setValue(float(angles[_n - 1 - angle_idx]))
            handles['plot_beat'].setTitle(f"Beat Spectrum @ {angles[angle_idx]:.2f}°")


# ---------------------------------------------------------------------------
# Range-Doppler GUI
# ---------------------------------------------------------------------------

def ensure_qt_app():
    """Create or return an existing QApplication with Windows platform-plugin fix."""
    import os
    if os.name == "nt":
        try:
            import PyQt5
            pyqt_root = os.path.dirname(PyQt5.__file__)
            qt_plugins_dir = os.path.join(pyqt_root, "Qt5", "plugins")
            qt_platforms_dir = os.path.join(qt_plugins_dir, "platforms")
            if os.path.isdir(qt_platforms_dir):
                os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = qt_platforms_dir
                os.environ.setdefault("QT_PLUGIN_PATH", qt_plugins_dir)
        except Exception:
            pass

    from pyqtgraph.Qt import QtWidgets
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def init_rd_gui(
    sray, conv, iq, subarray_modes, rx_phase_cal, tx_phase_cal,
    az_angle, el_angle, output_freq, r_res, num_chirps,
    track_state,
    enable_cfar=False, mti_filter=True,
    zero_range_bins=0, max_range_m=100,
):
    """Build the full Range-Doppler PyQtGraph GUI and return a handles dict.

    The returned dict contains every widget, state dict, and plot item the
    main processing loop needs.  All hardware-interacting callbacks are wired
    up internally via closure over the function parameters.

    Layout:
      ┌────────────────────────────────────────────────────────────┐
      │  [Top control bar: DynRange, CFAR, MTI, Pause, params]  │
      ├──────────────────────────────────────────┬─────────────────┤
      │                                          │ Side panel   │
      │   Tab 1: RD Heatmap + PPI                 │ (HW ctrls,   │
      │   Tab 2: Signal Processing plots            │  Track info, │
      │                                          │  Perf stats) │
      └──────────────────────────────────────────┴─────────────────┘

    Settings persistence:
      On startup, loads saved settings from rd_gui_settings.json in the
      project root.  The "⚙ Apply All" button saves all current settings.
      Toggle buttons (CFAR, MTI, R-Norm) auto-save on click.
    """
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets, QtCore
    app = ensure_qt_app()

    # ── Main window ──────────────────────────────────────────────────────
    main_widget = QtWidgets.QWidget()
    main_layout = QtWidgets.QVBoxLayout()
    main_widget.setLayout(main_layout)

    # ── Top control panel ────────────────────────────────────────────────
    control_panel = QtWidgets.QWidget()
    control_layout = QtWidgets.QHBoxLayout()
    control_panel.setLayout(control_layout)

    control_layout.addWidget(QtWidgets.QLabel("Dyn Range (dB):"))
    dyn_range_input = QtWidgets.QLineEdit("20")
    dyn_range_input.setMaximumWidth(50)
    dyn_range_input.setToolTip("Color scale spans from (peak - this value) to peak")
    control_layout.addWidget(dyn_range_input)

    cfar_btn = QtWidgets.QPushButton()
    cfar_btn.setFixedWidth(150)
    cfar_btn.setCheckable(True)
    control_layout.addWidget(cfar_btn)

    mti_btn = QtWidgets.QPushButton()
    mti_btn.setFixedWidth(140)
    mti_btn.setCheckable(True)
    control_layout.addWidget(mti_btn)

    mti3_btn = QtWidgets.QPushButton()
    mti3_btn.setFixedWidth(155)
    mti3_btn.setCheckable(True)
    mti3_btn.setToolTip("3-pulse MTI canceller [1,-2,1].\n"
                        "Deeper clutter notch (~40 dB) than 2-pulse (~20 dB).\n"
                        "Mutually exclusive with 2-pulse MTI.")
    control_layout.addWidget(mti3_btn)

    pause_btn = QtWidgets.QPushButton("⏸ Pause")
    pause_btn.setFixedWidth(130)
    pause_btn.setCheckable(True)
    control_layout.addWidget(pause_btn)

    control_layout.addWidget(QtWidgets.QLabel("  |  CFAR Gd:"))
    cfar_guard_input = QtWidgets.QLineEdit("2")
    cfar_guard_input.setMaximumWidth(35)
    control_layout.addWidget(cfar_guard_input)

    control_layout.addWidget(QtWidgets.QLabel("Tr:"))
    cfar_train_input = QtWidgets.QLineEdit("8")
    cfar_train_input.setMaximumWidth(35)
    control_layout.addWidget(cfar_train_input)

    control_layout.addWidget(QtWidgets.QLabel("Bias(dB):"))
    cfar_bias_input = QtWidgets.QLineEdit("10")
    cfar_bias_input.setMaximumWidth(45)
    control_layout.addWidget(cfar_bias_input)

    control_layout.addWidget(QtWidgets.QLabel("  |  Zero Range Bins:"))
    zero_bins_input = QtWidgets.QLineEdit(str(zero_range_bins))
    zero_bins_input.setMaximumWidth(50)
    control_layout.addWidget(zero_bins_input)

    control_layout.addWidget(QtWidgets.QLabel("  Max Range (m):"))
    max_range_input = QtWidgets.QLineEdit(str(max_range_m))
    max_range_input.setMaximumWidth(50)
    control_layout.addWidget(max_range_input)

    control_layout.addWidget(QtWidgets.QLabel("  Vel Bins:"))
    vel_bins_input = QtWidgets.QLineEdit(str(num_chirps))
    vel_bins_input.setMaximumWidth(50)
    control_layout.addWidget(vel_bins_input)

    control_layout.addWidget(QtWidgets.QLabel("  Gate Rσ:"))
    gate_range_sigma_input = QtWidgets.QLineEdit(
        str(track_state.get('gate_vis_sigma_range', 2.0)))
    gate_range_sigma_input.setMaximumWidth(50)
    gate_range_sigma_input.setToolTip(
        "Red gate box range half-height scale in sigma units")
    control_layout.addWidget(gate_range_sigma_input)

    # Universal apply button — reads all text inputs, updates state, saves
    apply_all_btn = QtWidgets.QPushButton("\u2699 Apply All")
    apply_all_btn.setFixedWidth(120)
    apply_all_btn.setStyleSheet("font-weight: bold;")
    control_layout.addWidget(apply_all_btn)

    fft_bins_btn = QtWidgets.QPushButton()
    fft_bins_btn.setFixedWidth(220)
    fft_bins_btn.setCheckable(True)
    control_layout.addWidget(fft_bins_btn)

    range_norm_btn = QtWidgets.QPushButton("R-Norm: OFF")
    range_norm_btn.setFixedWidth(155)
    range_norm_btn.setCheckable(True)
    control_layout.addWidget(range_norm_btn)

    nci_btn = QtWidgets.QPushButton("NCI: OFF")
    nci_btn.setFixedWidth(130)
    nci_btn.setCheckable(True)
    nci_btn.setToolTip("Non-coherent integration — EMA of dB RD map across frames.\n"
                       "Reduces noise floor, makes weak targets visible.\n"
                       "Alpha 0.0–1.0: lower = more averaging, higher = faster response.")
    control_layout.addWidget(nci_btn)

    control_layout.addWidget(QtWidgets.QLabel("α:"))
    nci_alpha_input = QtWidgets.QLineEdit("0.15")
    nci_alpha_input.setMaximumWidth(40)
    nci_alpha_input.setToolTip("NCI EMA weight (0.05–1.0). Lower = more averaging.")
    control_layout.addWidget(nci_alpha_input)

    cw_btn = QtWidgets.QPushButton("Waveform: Real")
    cw_btn.setFixedWidth(170)
    cw_btn.setCheckable(True)
    cw_btn.setToolTip("Toggle between Real and Complex (IQ) chirp waveform.\n"
                      "Real: cos chirp, one-sided spectrum, LPF needed.\n"
                      "Complex: analytic chirp exp(jφ), full spectrum, no image.")
    control_layout.addWidget(cw_btn)

    instructions = QtWidgets.QLabel("  |  SPACE=pause, [/] ±5dB, -/= ±1dB, Q=quit")
    control_layout.addWidget(instructions)
    control_layout.addStretch()

    main_layout.addWidget(control_panel)

    # ── Content area: side panel (left) + graphs (right) ─────────────────
    content_widget = QtWidgets.QWidget()
    content_layout = QtWidgets.QHBoxLayout()
    content_layout.setContentsMargins(0, 0, 0, 0)
    content_layout.setSpacing(4)
    content_widget.setLayout(content_layout)

    side_panel = QtWidgets.QWidget()
    side_panel.setFixedWidth(320)
    side_layout = QtWidgets.QVBoxLayout()
    side_layout.setContentsMargins(4, 6, 4, 4)
    side_layout.setAlignment(QtCore.Qt.AlignTop)
    side_panel.setLayout(side_layout)

    hw_label = QtWidgets.QLabel("Hardware Controls")
    hw_label.setStyleSheet("font-weight: bold;")
    side_layout.addWidget(hw_label)

    rx_lna_btn = QtWidgets.QPushButton("RX: ON")
    rx_lna_btn.setFixedWidth(155)
    rx_lna_btn.setCheckable(True)
    rx_lna_btn.setChecked(True)
    side_layout.addWidget(rx_lna_btn)

    tx_pa_btn = QtWidgets.QPushButton("TX: ON")
    tx_pa_btn.setFixedWidth(155)
    tx_pa_btn.setCheckable(True)
    tx_pa_btn.setChecked(True)
    side_layout.addWidget(tx_pa_btn)

    tx_atten_btn = QtWidgets.QPushButton("TX Atten: OFF")
    tx_atten_btn.setFixedWidth(155)
    tx_atten_btn.setCheckable(True)
    tx_atten_btn.setChecked(False)
    side_layout.addWidget(tx_atten_btn)

    side_layout.addWidget(QtWidgets.QLabel(""))
    dac_label = QtWidgets.QLabel("DAC Outputs")
    dac_label.setStyleSheet("font-weight: bold;")
    side_layout.addWidget(dac_label)

    _subarray_to_dac = {1: 3, 2: 2, 3: 1, 4: 0}
    _initial_tx_dac_chs = set(
        _subarray_to_dac[sa] for sa, m in subarray_modes.items() if m.lower() == "tx"
    )
    dac_checkboxes = []
    for ch in range(4):
        cb = QtWidgets.QCheckBox(f"CH{ch}")
        cb.setChecked(ch in _initial_tx_dac_chs)
        side_layout.addWidget(cb)
        dac_checkboxes.append(cb)

    side_layout.addWidget(QtWidgets.QLabel(""))
    tx_gain_label = QtWidgets.QLabel("TX PA Gain")
    tx_gain_label.setStyleSheet("font-weight: bold;")
    side_layout.addWidget(tx_gain_label)

    tx_gain_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    tx_gain_slider.setMinimum(0)
    tx_gain_slider.setMaximum(127)
    tx_gain_slider.setValue(127)
    tx_gain_slider.setFixedWidth(155)
    side_layout.addWidget(tx_gain_slider)

    tx_gain_value_label = QtWidgets.QLabel("127 / 127")
    tx_gain_value_label.setAlignment(QtCore.Qt.AlignCenter)
    side_layout.addWidget(tx_gain_value_label)

    side_layout.addWidget(QtWidgets.QLabel(""))
    _track_header = QtWidgets.QLabel("Tracking")
    _track_header.setStyleSheet("font-weight: bold;")
    side_layout.addWidget(_track_header)

    track_btn = QtWidgets.QPushButton("Track: OFF")
    track_btn.setFixedWidth(155)
    track_btn.setCheckable(True)
    side_layout.addWidget(track_btn)

    track_status_label = QtWidgets.QLabel("Track: ACQUIRING")
    side_layout.addWidget(track_status_label)
    beam_az_label = QtWidgets.QLabel(f"Beam Az: {az_angle:.1f}\u00b0")
    beam_el_label = QtWidgets.QLabel(f"Beam El: {el_angle:.1f}\u00b0")
    side_layout.addWidget(beam_az_label)
    side_layout.addWidget(beam_el_label)

    # ── Spectrogram controls ─────────────────────────────────────────
    side_layout.addWidget(QtWidgets.QLabel(""))
    _spec_header = QtWidgets.QLabel("Spectrogram")
    _spec_header.setStyleSheet("font-weight: bold;")
    side_layout.addWidget(_spec_header)

    _spec_range_row = QtWidgets.QHBoxLayout()
    _spec_range_row.addWidget(QtWidgets.QLabel("Range (m):"))
    spec_range_input = QtWidgets.QLineEdit("8.0")
    spec_range_input.setMaximumWidth(50)
    spec_range_input.setToolTip("Target range for spectrogram slice (metres).\n"
                                 "Set to the range of your target to see its\n"
                                 "Doppler signature evolve over time.")
    _spec_range_row.addWidget(spec_range_input)
    _spec_range_row.addStretch()
    _spec_range_container = QtWidgets.QWidget()
    _spec_range_container.setLayout(_spec_range_row)
    side_layout.addWidget(_spec_range_container)

    _spec_depth_row = QtWidgets.QHBoxLayout()
    _spec_depth_row.addWidget(QtWidgets.QLabel("Depth:"))
    spec_depth_input = QtWidgets.QLineEdit("200")
    spec_depth_input.setMaximumWidth(50)
    spec_depth_input.setToolTip("Number of frames in the spectrogram waterfall.")
    _spec_depth_row.addWidget(spec_depth_input)
    _spec_depth_row.addStretch()
    _spec_depth_container = QtWidgets.QWidget()
    _spec_depth_container.setLayout(_spec_depth_row)
    side_layout.addWidget(_spec_depth_container)

    spec_auto_btn = QtWidgets.QPushButton("Auto Range: OFF")
    spec_auto_btn.setFixedWidth(155)
    spec_auto_btn.setCheckable(True)
    spec_auto_btn.setToolTip("Auto-select range bin from CFAR peak detection.")
    side_layout.addWidget(spec_auto_btn)

    side_layout.addWidget(QtWidgets.QLabel(""))
    perf_label = QtWidgets.QLabel("Capture: -- ms")
    perf_label.setStyleSheet("font-weight: bold;")
    side_layout.addWidget(perf_label)
    proc_label = QtWidgets.QLabel("Process: -- ms")
    proc_label.setStyleSheet("font-weight: bold;")
    side_layout.addWidget(proc_label)
    # Per-brick timing breakdown (monospace for column alignment)
    timing_detail_label = QtWidgets.QLabel("")
    timing_detail_label.setStyleSheet(
        "font-family: Consolas, monospace; font-size: 10px; color: #aaa;")
    side_layout.addWidget(timing_detail_label)
    fps_label = QtWidgets.QLabel("FPS: --")
    fps_label.setStyleSheet("font-weight: bold;")
    side_layout.addWidget(fps_label)

    side_layout.addStretch()
    content_layout.addWidget(side_panel)

    # ── Tabbed display ───────────────────────────────────────────────────
    tab_widget = QtWidgets.QTabWidget()

    # Tab 1: Radar (Range-Doppler heatmap + PPI)
    radar_tab = pg.GraphicsLayoutWidget()

    rd_plot = radar_tab.addPlot(row=0, col=0, rowspan=3, title="Range-Doppler Map")
    rd_plot.setLabel('left', 'Range (m)')
    rd_plot.setLabel('bottom', 'Velocity (m/s)')
    rd_img = pg.ImageItem()
    rd_plot.addItem(rd_img)
    rd_cmap = pg.colormap.get('inferno')
    rd_img.setColorMap(rd_cmap)
    rd_cbar = pg.ColorBarItem(colorMap=rd_cmap, values=(0, 10), limits=(0, 20))
    rd_cbar.setImageItem(rd_img, insert_in=rd_plot)
    rd_stats = pg.TextItem(color='w', anchor=(0, 0))
    rd_plot.addItem(rd_stats)

    det_scatter = pg.ScatterPlotItem(
        size=12, pen=pg.mkPen('g', width=2),
        brush=pg.mkBrush(0, 255, 0, 80), symbol='crosshair')
    rd_plot.addItem(det_scatter)
    det_labels = []

    gate_rect = QtWidgets.QGraphicsRectItem(0, 0, 1, 1)
    gate_rect.setPen(pg.mkPen('r', width=2, style=QtCore.Qt.DashLine))
    gate_rect.setBrush(pg.mkBrush(None))
    gate_rect.setVisible(False)
    rd_plot.addItem(gate_rect)

    # ── PPI (Plan Position Indicator) plot ───────────────────────────────
    _max_steer = track_state.get('max_beam_angle', 45.0)
    _ppi_x_max = max_range_m * np.sin(np.deg2rad(_max_steer))

    ppi_plot = radar_tab.addPlot(row=0, col=1, rowspan=3, title="PPI — Top-Down View")
    ppi_plot.setLabel('left', 'Range (m)')
    ppi_plot.setLabel('bottom', 'Cross-Range (m)')
    ppi_plot.setAspectLocked(True)
    ppi_plot.showGrid(x=True, y=True, alpha=0.2)
    ppi_plot.setXRange(-_ppi_x_max, _ppi_x_max, padding=0.05)
    ppi_plot.setYRange(0, max_range_m, padding=0.05)

    # Beam coverage cone (filled arc showing max steering extent)
    _arc_angles = np.linspace(-np.deg2rad(_max_steer), np.deg2rad(_max_steer), 100)
    _arc_x = max_range_m * np.sin(_arc_angles)
    _arc_y = max_range_m * np.cos(_arc_angles)
    _arc_x = np.concatenate([[0], _arc_x, [0]])
    _arc_y = np.concatenate([[0], _arc_y, [0]])
    ppi_plot.plot(_arc_x, _arc_y, pen=pg.mkPen((80, 80, 80), width=1, style=QtCore.Qt.DashLine))

    # Range rings
    for _ring_frac in [0.25, 0.5, 0.75, 1.0]:
        _rr = max_range_m * _ring_frac
        _ring_a = np.linspace(-np.deg2rad(_max_steer), np.deg2rad(_max_steer), 60)
        _ring_x = _rr * np.sin(_ring_a)
        _ring_y = _rr * np.cos(_ring_a)
        ppi_plot.plot(_ring_x, _ring_y, pen=pg.mkPen((60, 60, 60), width=1))

    # Current beam line
    ppi_beam_line = ppi_plot.plot([0, 0], [0, max_range_m],
                                  pen=pg.mkPen('y', width=2, style=QtCore.Qt.DashLine))

    # Target marker (current position)
    ppi_target = pg.ScatterPlotItem(
        size=14, pen=pg.mkPen('lime', width=2),
        brush=pg.mkBrush(0, 255, 0, 160), symbol='o')
    ppi_plot.addItem(ppi_target)

    # Target trail (history of positions)
    _ppi_trail_max = 50
    ppi_trail = pg.ScatterPlotItem(
        size=10, pen=pg.mkPen(None),
        brush=pg.mkBrush(0, 255, 0, 60), symbol='o')
    ppi_plot.addItem(ppi_trail)
    ppi_trail_data = {'x': [], 'y': []}

    tab_widget.addTab(radar_tab, "Radar")

    # Tab 2: Signal Processing
    sp_tab = pg.GraphicsLayoutWidget()

    sumiq_plot = sp_tab.addPlot(row=0, col=0, title="Sum Data (Real) - Raw Captured")
    sumiq_plot.setLabel('left', 'Amplitude')
    sumiq_plot.setLabel('bottom', 'Sample')
    sumiq_plot.showGrid(x=True, y=True, alpha=0.3)
    sumiq_curve = sumiq_plot.plot(pen=pg.mkPen('c', width=1))

    fft_plot = sp_tab.addPlot(row=1, col=0, title="Radar Data (FFT)")
    fft_plot.setLabel('left', 'Magnitude (dB)')
    fft_plot.setLabel('bottom', 'Range Bin')
    fft_plot.showGrid(x=True, y=True, alpha=0.3)
    fft_curves = []

    rxfft_plot = sp_tab.addPlot(row=2, col=0, title="Received Data FFT (1 chirp, dB)")
    rxfft_plot.setLabel('left', 'Magnitude (dB)')
    rxfft_plot.setLabel('bottom', 'Frequency (MHz)')
    rxfft_plot.showGrid(x=True, y=True, alpha=0.3)
    rxfft_curve = rxfft_plot.plot(pen=pg.mkPen((255, 180, 0), width=1))

    tab_widget.addTab(sp_tab, "Signal Processing")

    # Tab 3: Spectrogram (time-Doppler waterfall at a selected range bin)
    spec_tab = pg.GraphicsLayoutWidget()
    spec_plot = spec_tab.addPlot(row=0, col=0,
                                  title="Micro-Doppler Spectrogram")
    spec_plot.setLabel('left', 'Velocity (m/s)')
    spec_plot.setLabel('bottom', 'Time (frames)')
    spec_plot.showGrid(x=True, y=True, alpha=0.3)
    spec_img = pg.ImageItem()
    spec_plot.addItem(spec_img)
    _spec_lut = pg.colormap.get('inferno').getLookupTable(nPts=256)
    spec_img.setLookupTable(_spec_lut)
    spec_cbar = pg.ColorBarItem(
        colorMap='inferno', interactive=False, orientation='right')
    spec_cbar.setImageItem(spec_img)
    spec_tab.addItem(spec_cbar, row=0, col=1)
    tab_widget.addTab(spec_tab, "Spectrogram")

    content_layout.addWidget(tab_widget)
    main_layout.addWidget(content_widget)

    main_widget.setWindowTitle("Range-Doppler Processing - Interactive Controls")
    main_widget.resize(2020, 1050)
    main_widget.show()

    # ── Mutable state dicts ──────────────────────────────────────────────
    color_scale = {'dyn_range': 20.0}
    paused = {'state': False}
    zero_bins_config = {'bins': zero_range_bins}
    max_range_config = {'range_m': max_range_m}
    vel_bins_config = {'bins': num_chirps}
    processing_flags = {
        'cfar': bool(enable_cfar),
        'mti': bool(mti_filter),
        'mti3': False,
        'range_norm': False,
        'nci': False,
        'complex_waveform': False,
    }
    nci_state = {
        'alpha': 0.3,
        'accum': None,  # accumulated dB RD map (same shape as radar_data)
    }
    fft_display = {'include_additional_velocity_bins': False}

    spectrogram_state = {
        'range_m': 8.0,           # target range in metres
        'depth': 200,             # waterfall depth in frames
        'buffer': None,           # ring buffer (depth x doppler_bins) in dB
        'write_idx': 0,           # next row to write
        'auto_range': False,      # auto-select range from peak
    }
    cfar_params = {
        'guard_cells': (4, 4),
        'training_cells': (8, 8),
        'bias_db': 10.0,
    }

    # ── Load saved settings and apply to state + widgets ─────────────────
    _saved = _load_gui_settings()
    if _saved:
        color_scale['dyn_range'] = _saved.get('dyn_range', color_scale['dyn_range'])
        processing_flags['cfar'] = _saved.get('cfar', processing_flags['cfar'])
        processing_flags['mti'] = _saved.get('mti', processing_flags['mti'])
        processing_flags['mti3'] = _saved.get('mti3', processing_flags['mti3'])
        processing_flags['range_norm'] = _saved.get('range_norm', processing_flags['range_norm'])
        processing_flags['nci'] = _saved.get('nci', processing_flags['nci'])
        processing_flags['complex_waveform'] = _saved.get('complex_waveform', processing_flags['complex_waveform'])
        nci_state['alpha'] = _saved.get('nci_alpha', nci_state['alpha'])
        fft_display['include_additional_velocity_bins'] = _saved.get(
            'fft_all_bins', fft_display['include_additional_velocity_bins'])
        cfar_params['guard_cells'] = tuple(
            _saved.get('cfar_guard', list(cfar_params['guard_cells'])))
        cfar_params['training_cells'] = tuple(
            _saved.get('cfar_training', list(cfar_params['training_cells'])))
        cfar_params['bias_db'] = _saved.get('cfar_bias_db', cfar_params['bias_db'])
        zero_bins_config['bins'] = _saved.get('zero_range_bins', zero_bins_config['bins'])
        max_range_config['range_m'] = _saved.get('max_range_m', max_range_config['range_m'])
        _saved_vel = _saved.get('vel_bins', vel_bins_config['bins'])
        vel_bins_config['bins'] = min(_saved_vel, num_chirps)
        track_state['gate_vis_sigma_range'] = _saved.get(
            'gate_vis_sigma_range', track_state.get('gate_vis_sigma_range', 2.0))
        # Update input widgets to reflect loaded values
        dyn_range_input.setText(str(color_scale['dyn_range']))
        cfar_guard_input.setText(str(cfar_params['guard_cells'][0]))
        cfar_train_input.setText(str(cfar_params['training_cells'][0]))
        cfar_bias_input.setText(str(cfar_params['bias_db']))
        zero_bins_input.setText(str(zero_bins_config['bins']))
        max_range_input.setText(str(max_range_config['range_m']))
        vel_bins_input.setText(str(vel_bins_config['bins']))
        gate_range_sigma_input.setText(str(track_state['gate_vis_sigma_range']))
        range_norm_btn.setText(f"R-Norm: {'ON' if processing_flags['range_norm'] else 'OFF'}")
        range_norm_btn.setChecked(processing_flags['range_norm'])
        nci_btn.setText(f"NCI: {'ON' if processing_flags['nci'] else 'OFF'}")
        nci_btn.setChecked(processing_flags['nci'])
        nci_alpha_input.setText(str(nci_state['alpha']))
        cw_btn.setText(f"Waveform: {'Complex' if processing_flags['complex_waveform'] else 'Real'}")
        cw_btn.setChecked(processing_flags['complex_waveform'])
        # Spectrogram
        spectrogram_state['range_m'] = _saved.get('spec_range_m', spectrogram_state['range_m'])
        spectrogram_state['depth'] = _saved.get('spec_depth', spectrogram_state['depth'])
        spec_range_input.setText(str(spectrogram_state['range_m']))
        spec_depth_input.setText(str(spectrogram_state['depth']))
        print(f"Loaded saved GUI settings from {_SETTINGS_FILE}")

    # ── Callbacks ────────────────────────────────────────────────────────
    def refresh_processing_button_labels():
        cfar_btn.setText(f"CFAR: {'ON' if processing_flags['cfar'] else 'OFF'}")
        mti_btn.setText(f"MTI-2: {'ON' if processing_flags['mti'] else 'OFF'}")
        mti3_btn.setText(f"MTI-3: {'ON' if processing_flags['mti3'] else 'OFF'}")
        cfar_btn.setChecked(processing_flags['cfar'])
        mti_btn.setChecked(processing_flags['mti'])
        mti3_btn.setChecked(processing_flags['mti3'])

    def toggle_cfar():
        processing_flags['cfar'] = not processing_flags['cfar']
        refresh_processing_button_labels()
        print(f"CFAR {'enabled' if processing_flags['cfar'] else 'disabled'}")

    def toggle_mti():
        processing_flags['mti'] = not processing_flags['mti']
        if processing_flags['mti']:
            processing_flags['mti3'] = False  # mutually exclusive
        refresh_processing_button_labels()
        print(f"MTI 2-pulse {'enabled' if processing_flags['mti'] else 'disabled'}")

    def toggle_mti3():
        processing_flags['mti3'] = not processing_flags['mti3']
        if processing_flags['mti3']:
            processing_flags['mti'] = False  # mutually exclusive
        refresh_processing_button_labels()
        print(f"MTI 3-pulse {'enabled' if processing_flags['mti3'] else 'disabled'}")

    def toggle_range_norm():
        processing_flags['range_norm'] = not processing_flags['range_norm']
        range_norm_btn.setText(f"R-Norm: {'ON' if processing_flags['range_norm'] else 'OFF'}")
        range_norm_btn.setChecked(processing_flags['range_norm'])
        print(f"Range normalisation {'enabled' if processing_flags['range_norm'] else 'disabled'}")

    def toggle_nci():
        processing_flags['nci'] = not processing_flags['nci']
        nci_btn.setText(f"NCI: {'ON' if processing_flags['nci'] else 'OFF'}")
        nci_btn.setChecked(processing_flags['nci'])
        if not processing_flags['nci']:
            nci_state['accum'] = None  # reset accumulator on disable
        print(f"Non-coherent integration {'enabled' if processing_flags['nci'] else 'disabled'}")

    def toggle_complex_waveform():
        processing_flags['complex_waveform'] = not processing_flags['complex_waveform']
        _label = 'Complex' if processing_flags['complex_waveform'] else 'Real'
        cw_btn.setText(f"Waveform: {_label}")
        cw_btn.setChecked(processing_flags['complex_waveform'])
        print(f"Waveform set to {_label}")

    def toggle_spec_auto_range():
        spectrogram_state['auto_range'] = not spectrogram_state['auto_range']
        spec_auto_btn.setText(
            f"Auto Range: {'ON' if spectrogram_state['auto_range'] else 'OFF'}")
        spec_auto_btn.setChecked(spectrogram_state['auto_range'])
        print(f"Spectrogram auto-range {'enabled' if spectrogram_state['auto_range'] else 'disabled'}")

    def update_spec_range():
        try:
            val = float(spec_range_input.text())
            if val < 0:
                print("Spectrogram range must be >= 0")
                return
            spectrogram_state['range_m'] = val
            spectrogram_state['buffer'] = None  # reset waterfall
            spectrogram_state['write_idx'] = 0
            print(f"Spectrogram range set to {val:.1f} m")
        except ValueError:
            print("Invalid spectrogram range value")

    def update_spec_depth():
        try:
            val = int(spec_depth_input.text())
            if val < 10:
                print("Spectrogram depth must be >= 10")
                return
            spectrogram_state['depth'] = val
            spectrogram_state['buffer'] = None  # reset waterfall
            spectrogram_state['write_idx'] = 0
            print(f"Spectrogram depth set to {val} frames")
        except ValueError:
            print("Invalid spectrogram depth value")

    def update_nci_alpha():
        try:
            alpha = float(nci_alpha_input.text())
            alpha = max(0.01, min(1.0, alpha))
            nci_state['alpha'] = alpha
            nci_state['accum'] = None  # reset on alpha change
            nci_alpha_input.setText(str(alpha))
            print(f"NCI alpha set to {alpha}")
        except ValueError:
            print("Invalid NCI alpha value")

    def toggle_tracking():
        track_state['enabled'] = not track_state['enabled']
        track_btn.setText(f"Track: {'ON' if track_state['enabled'] else 'OFF'}")
        track_btn.setChecked(track_state['enabled'])
        if not track_state['enabled']:
            reset_track(track_state)
            gate_rect.setVisible(False)
            sray.steer_rx(0.0, 0.0, cal_dict=rx_phase_cal)
            sray.steer_tx(0.0, 0.0, cal_dict=tx_phase_cal)
            beam_az_label.setText("Beam Az: 0.0\u00b0")
            beam_el_label.setText("Beam El: 0.0\u00b0")
            track_status_label.setText("Track: OFF")
            det_scatter.setData([], [])
            print("Tracking DISABLED \u2014 beam reset to boresight")
        else:
            track_status_label.setText("Track: ACQUIRING")
            print("Tracking ENABLED \u2014 CFAR must be ON")

    def toggle_rx_lna():
        enabled = rx_lna_btn.isChecked()
        for device in sray.devices.values():
            if device.mode == "rx":
                for channel in device.channels:
                    channel.rx_enable = enabled
        sray.latch_rx_settings()
        rx_lna_btn.setText(f"RX: {'ON' if enabled else 'OFF'}")
        print(f"RX channels {'enabled' if enabled else 'disabled'}")

    def toggle_tx_pa():
        enabled = tx_pa_btn.isChecked()
        for device in sray.devices.values():
            if device.mode == "tx":
                for channel in device.channels:
                    channel.tx_enable = enabled
        sray.latch_tx_settings()
        tx_pa_btn.setText(f"TX: {'ON' if enabled else 'OFF'}")
        print(f"TX channels {'enabled' if enabled else 'disabled'}")

    def toggle_tx_atten():
        enabled = tx_atten_btn.isChecked()
        for device in sray.devices.values():
            if device.mode == "tx":
                for channel in device.channels:
                    channel.tx_attenuator = enabled
        sray.latch_tx_settings()
        tx_atten_btn.setText(f"TX Atten: {'ON' if enabled else 'OFF'}")
        print(f"TX attenuators {'enabled' if enabled else 'disabled'}")

    def update_dac_channels():
        enabled_chs = set(i for i, cb in enumerate(dac_checkboxes) if cb.isChecked())
        zeros = np.zeros_like(iq)
        tx_data = [iq if ch in enabled_chs else zeros for ch in range(4)]
        conv.tx_destroy_buffer()
        conv.tx_enabled_channels = [0, 1, 2, 3]
        conv.tx(tx_data)
        print(f"DAC chirp on channels: {sorted(enabled_chs)}, "
              f"zeros on: {sorted(set(range(4)) - enabled_chs)}")

    def update_tx_gain(value):
        tx_gain_value_label.setText(f"{value} / 127")
        for device in sray.devices.values():
            if device.mode == "tx":
                for channel in device.channels:
                    channel.tx_gain = value
        sray.latch_tx_settings()
        print(f"TX PA gain set to {value}")

    def refresh_fft_button_label():
        if fft_display['include_additional_velocity_bins']:
            fft_bins_btn.setText("FFT: All Velocity Bins")
            fft_bins_btn.setChecked(True)
        else:
            fft_bins_btn.setText("FFT: Zero-Vel Bin Only")
            fft_bins_btn.setChecked(False)

    def toggle_fft_bins_display():
        fft_display['include_additional_velocity_bins'] = not fft_display['include_additional_velocity_bins']
        refresh_fft_button_label()
        if fft_display['include_additional_velocity_bins']:
            print("FFT plot mode: showing all velocity bins")
        else:
            print("FFT plot mode: showing zero-velocity bin only")

    refresh_processing_button_labels()
    refresh_fft_button_label()

    def update_color_scale():
        try:
            new_dr = float(dyn_range_input.text())
            if new_dr < 1:
                print("Dynamic range must be >= 1 dB")
                return
            color_scale['dyn_range'] = new_dr
            print(f"Dynamic range set to {new_dr:.1f} dB")
        except ValueError:
            print("Invalid dynamic range value")

    def toggle_pause():
        paused['state'] = not paused['state']
        if paused['state']:
            pause_btn.setText("▶ Resume")
            print("PAUSED - Adjust settings and press Resume or SPACE to continue")
        else:
            pause_btn.setText("⏸ Pause")
            print("RESUMED")

    def update_zero_bins():
        try:
            new_bins = int(zero_bins_input.text())
            if new_bins < 0:
                print("Zero range bins must be >= 0")
                return
            zero_bins_config['bins'] = new_bins
            zeroed_range_m = new_bins * r_res
            print(f"Zero range bins updated to {new_bins} ({zeroed_range_m:.1f} m)")
        except ValueError:
            print("Invalid zero range bins value")

    def update_max_range():
        try:
            new_range = float(max_range_input.text())
            if new_range <= 0:
                print("Max range must be > 0")
                return
            max_range_config['range_m'] = new_range
            print(f"Max range updated to {new_range:.1f} m")
        except ValueError:
            print("Invalid max range value")

    def update_vel_bins():
        try:
            new_bins = int(vel_bins_input.text())
            if new_bins < 2:
                print("Velocity bins must be >= 2")
                return
            new_bins = min(new_bins, num_chirps)
            if new_bins % 2 != 0:
                new_bins += 1
            vel_bins_config['bins'] = new_bins
            print(f"Velocity bins updated to {new_bins}")
        except ValueError:
            print("Invalid velocity bins value")

    def update_gate_range_sigma():
        try:
            sigma = float(gate_range_sigma_input.text())
            if sigma <= 0:
                print("Gate Rσ must be > 0")
                return
            track_state['gate_vis_sigma_range'] = sigma
            print(f"Gate range sigma updated to {sigma:.2f}")
        except ValueError:
            print("Invalid Gate Rσ value")

    def update_cfar_params():
        try:
            guard = int(cfar_guard_input.text())
            train = int(cfar_train_input.text())
            bias = float(cfar_bias_input.text())
            if guard < 0 or train < 1:
                print("CFAR guard must be >= 0 and training must be >= 1")
                return
            if train <= guard:
                print("CFAR training should be greater than guard")
                return
            if bias < 0:
                print("CFAR bias must be >= 0 dB")
                return
            cfar_params['guard_cells'] = (guard, guard)
            cfar_params['training_cells'] = (train, train)
            cfar_params['bias_db'] = bias
            print(f"CFAR params updated: guard={guard}, train={train}, bias={bias:.1f} dB")
        except ValueError:
            print("Invalid CFAR params")

    # ── Settings save helper ──────────────────────────────────────────────
    def _save_current_state():
        """Persist all current GUI state to disk."""
        _save_gui_settings({
            'dyn_range': color_scale['dyn_range'],
            'cfar': processing_flags['cfar'],
            'mti': processing_flags['mti'],
            'mti3': processing_flags['mti3'],
            'range_norm': processing_flags['range_norm'],
            'nci': processing_flags['nci'],
            'nci_alpha': nci_state['alpha'],
            'complex_waveform': processing_flags['complex_waveform'],
            'fft_all_bins': fft_display['include_additional_velocity_bins'],
            'cfar_guard': list(cfar_params['guard_cells']),
            'cfar_training': list(cfar_params['training_cells']),
            'cfar_bias_db': cfar_params['bias_db'],
            'zero_range_bins': zero_bins_config['bins'],
            'max_range_m': max_range_config['range_m'],
            'vel_bins': vel_bins_config['bins'],
            'gate_vis_sigma_range': track_state.get('gate_vis_sigma_range', 2.0),
            'spec_range_m': spectrogram_state['range_m'],
            'spec_depth': spectrogram_state['depth'],
        })

    def apply_all():
        """Read all text inputs, update state dicts, and save to disk."""
        update_color_scale()
        update_cfar_params()
        update_zero_bins()
        update_max_range()
        update_vel_bins()
        update_gate_range_sigma()
        update_nci_alpha()
        update_spec_range()
        update_spec_depth()
        _save_current_state()
        print("All settings applied and saved.")

    # ── Signal wiring ────────────────────────────────────────────────────
    # Universal apply button + Enter key in any text input
    apply_all_btn.clicked.connect(apply_all)
    dyn_range_input.returnPressed.connect(apply_all)
    cfar_guard_input.returnPressed.connect(apply_all)
    cfar_train_input.returnPressed.connect(apply_all)
    cfar_bias_input.returnPressed.connect(apply_all)
    zero_bins_input.returnPressed.connect(apply_all)
    max_range_input.returnPressed.connect(apply_all)
    vel_bins_input.returnPressed.connect(apply_all)
    gate_range_sigma_input.returnPressed.connect(apply_all)
    # Toggle buttons: immediate effect + auto-save
    pause_btn.clicked.connect(toggle_pause)
    cfar_btn.clicked.connect(toggle_cfar)
    cfar_btn.clicked.connect(_save_current_state)
    mti_btn.clicked.connect(toggle_mti)
    mti_btn.clicked.connect(_save_current_state)
    mti3_btn.clicked.connect(toggle_mti3)
    mti3_btn.clicked.connect(_save_current_state)
    range_norm_btn.clicked.connect(toggle_range_norm)
    range_norm_btn.clicked.connect(_save_current_state)
    nci_btn.clicked.connect(toggle_nci)
    nci_btn.clicked.connect(_save_current_state)
    cw_btn.clicked.connect(toggle_complex_waveform)
    cw_btn.clicked.connect(_save_current_state)
    nci_alpha_input.returnPressed.connect(apply_all)
    fft_bins_btn.clicked.connect(toggle_fft_bins_display)
    fft_bins_btn.clicked.connect(_save_current_state)
    # Spectrogram controls
    spec_auto_btn.clicked.connect(toggle_spec_auto_range)
    spec_auto_btn.clicked.connect(_save_current_state)
    spec_range_input.returnPressed.connect(apply_all)
    spec_depth_input.returnPressed.connect(apply_all)
    # Hardware controls (not persisted — always start from known state)
    rx_lna_btn.clicked.connect(toggle_rx_lna)
    tx_pa_btn.clicked.connect(toggle_tx_pa)
    tx_atten_btn.clicked.connect(toggle_tx_atten)
    for cb in dac_checkboxes:
        cb.stateChanged.connect(update_dac_channels)
    tx_gain_slider.valueChanged.connect(update_tx_gain)
    track_btn.clicked.connect(toggle_tracking)

    # ── Return handles ───────────────────────────────────────────────────
    return {
        # Qt application
        'app': app,
        'main_widget': main_widget,
        # State dicts
        'color_scale': color_scale,
        'paused': paused,
        'zero_bins_config': zero_bins_config,
        'max_range_config': max_range_config,
        'vel_bins_config': vel_bins_config,
        'processing_flags': processing_flags,
        'nci_state': nci_state,
        'fft_display': fft_display,
        'cfar_params': cfar_params,
        # Radar tab plots
        'rd_plot': rd_plot,
        'rd_img': rd_img,
        'rd_cbar': rd_cbar,
        'rd_stats': rd_stats,
        'det_scatter': det_scatter,
        'det_labels': det_labels,
        'gate_rect': gate_rect,
        # PPI plot
        'ppi_plot': ppi_plot,
        'ppi_beam_line': ppi_beam_line,
        'ppi_target': ppi_target,
        'ppi_trail': ppi_trail,
        'ppi_trail_data': ppi_trail_data,
        'ppi_trail_max': _ppi_trail_max,
        # Tracking
        'track_state': track_state,
        'track_status_label': track_status_label,
        'beam_az_label': beam_az_label,
        'beam_el_label': beam_el_label,
        # SP tab plots
        'tab_widget': tab_widget,
        'sumiq_plot': sumiq_plot,
        'sumiq_curve': sumiq_curve,
        'fft_plot': fft_plot,
        'fft_curves': fft_curves,
        'rxfft_plot': rxfft_plot,
        'rxfft_curve': rxfft_curve,
        # Spectrogram tab
        'spec_plot': spec_plot,
        'spec_img': spec_img,
        'spectrogram_state': spectrogram_state,
        # Side panel labels
        'perf_label': perf_label,
        'proc_label': proc_label,
        'timing_detail_label': timing_detail_label,
        'fps_label': fps_label,
        # Input widgets (read by main loop)
        'pause_btn': pause_btn,
        'dyn_range_input': dyn_range_input,
        # Callbacks exposed for keyboard shortcuts
        'toggle_pause': toggle_pause,
    }


