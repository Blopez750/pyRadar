# =============================================================================
# fmcw_scan.py — FMCW azimuth scanning pilot with heatmap display
# =============================================================================
# This pilot scans the radar beam through a range of azimuth angles and builds
# a 2-D heatmap (angle × range) similar to a B-scope display.
#
# At each scan angle the beam is electronically steered by programming new
# phase values into the ADAR1000 array, then one ADC capture is taken.  The
# received data is dechirped (mixed with conj(iq)), FFT'd to produce a beat
# spectrum, and the magnitude is placed into the corresponding column of the
# heatmap.
#
# Key concepts:
#   - **Chirp slope k = BW / T**: links beat frequency to range.
#   - **Range binning**: the beat spectrum is interpolated onto a uniform
#     range grid (r_centers) so adjacent angles share the same range axis.
#   - **Baseline clutter mesh**: before the live scan, a multi-frame average
#     of the empty-scene heatmap is captured.  Subtracting this baseline
#     during the live scan removes static clutter (walls, furniture, etc.)
#     and highlights moving / new targets.
#   - **Cross-talk removal**: the internal leakage peak is circularly shifted
#     to DC and zeroed (see signal_processing.circular_shift_fft).
# =============================================================================

"""FMCW azimuth scanning pilot with PyQtGraph heatmap display."""
import numpy as np
import time

from radar_utils.hardware_setup import extract_rx_subarrays, build_rx_channel_config
from radar_utils.calibration import data_capture_cal
from radar_utils.signal_processing import heatmap_gen, beat_calc
from radar_utils.radar_plotting import init_fmcw_radar_viewer, update_fmcw_radar_viewer
from radar_utils.utils import maximize_by_title, minimise_by_title


def FMCWScan(conv, sray, cal_ant_fix, subarray_modes, iq, BW, PRF,
             scan_min, scan_max, scan_step, analog_phase_cal, tx_phase_cal,
             cluttermesh=True):
    """
    FMCW live azimuth scan.

    The beam is electronically steered from scan_min to scan_max in steps of
    scan_step degrees.  At each angle a single ADC capture is dechirped,
    FFT'd, and the magnitude is placed into the heatmap at the corresponding
    azimuth column.  The heatmap axes are angle (x) and range (y).

    Parameters:
        conv:             AD9081 converter object
        sray:             Stingray ADAR1000 array object
        cal_ant_fix:      digital phase calibration corrections
        subarray_modes:   dict {1:"rx", 2:"rx", 3:"tx", 4:"rx"}
        iq:               ideal IQ chirp waveform (used for dechirp)
        BW:               chirp bandwidth (Hz)
        PRF:              pulse repetition frequency (Hz) — used for pacing
        scan_min/max:     azimuth scan limits (degrees)
        scan_step:        angular step size (degrees)
        analog_phase_cal: RX analog phase calibration dict
        tx_phase_cal:     TX phase calibration dict
        cluttermesh:      if True, capture a baseline clutter mesh before scanning
      - Notch around cross_talk_freq (optional adaptive drift tracking)
      - PRF pacing and Qt 'Q' shortcut to quit
      - Robust to changing capture length (rebuilds FFT axis and S)
    """
    from pyqtgraph.Qt import QtWidgets, QtGui, QtCore
    import pyqtgraph as pg

    # ── Constants & config ────────────────────────────────────────────────
    c = 3e8
    fs = conv.rx_sample_rate
    N_ref = len(iq)
    T = N_ref / fs
    # Chirp slope k = BW / T  (Hz per second)
    # This links beat frequency to target range:  R = c * f_beat / (2 * k)
    k = BW / T

    mag_floor_db = -30.0
    r_max        = 4.8
    # Range resolution: R_res = c / (2 * BW)
    # This sets the width of each range bin in the heatmap.
    dr_target    = 0.6

    range_state = {'r_max': r_max, 'changed': False}

    angle_vals = np.arange(scan_min, scan_max + 1e-12, scan_step)
    num_angles = len(angle_vals)

    r_edges   = np.arange(0.0, r_max + dr_target, dr_target)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    n_bins    = len(r_centers)

    heatmap = np.full((n_bins, num_angles), 1e-6, dtype=np.float32)

    cross_talk_freq = 0.0  # Calibration disabled — no notch applied

    # ── Initialize viewer ─────────────────────────────────────────────────
    handles = init_fmcw_radar_viewer(
        angles_deg=angle_vals,
        ranges_m=r_centers,
        beat_freq_hz=None,
        initial_heatmap=heatmap,
        start_event_loop=False,
    )

    try:
        handles['image_item'].getViewBox().invertY(False)
        handles['image_item'].getViewBox().invertX(True)
    except Exception:
        pass

    # Levels will auto-scale to the true data range in update_fmcw_radar_viewer

    handles['stop'] = False
    try:
        QtWidgets.QShortcut(
            QtGui.QKeySequence("Q"),
            handles['window'],
            activated=lambda: handles.__setitem__('stop', True),
        )
    except Exception:
        pass

    # ── Side panel with TX Gain slider ────────────────────────────────────
    win = handles['window']
    win.hide()

    scan_container = QtWidgets.QWidget()
    scan_container.setWindowTitle("FMCW Radar \u2013 Live")
    scan_container_layout = QtWidgets.QHBoxLayout()
    scan_container_layout.setContentsMargins(0, 0, 0, 0)
    scan_container_layout.setSpacing(0)

    scan_side_panel = QtWidgets.QWidget()
    scan_side_panel.setFixedWidth(170)
    scan_side_layout = QtWidgets.QVBoxLayout()
    scan_side_layout.setContentsMargins(6, 8, 6, 6)
    scan_side_layout.setAlignment(QtCore.Qt.AlignTop)
    scan_side_panel.setLayout(scan_side_layout)

    scan_hw_label = QtWidgets.QLabel("Hardware Controls")
    scan_hw_label.setStyleSheet("font-weight: bold;")
    scan_side_layout.addWidget(scan_hw_label)

    scan_tx_gain_label = QtWidgets.QLabel("TX PA Gain")
    scan_tx_gain_label.setStyleSheet("font-weight: bold; margin-top: 8px;")
    scan_side_layout.addWidget(scan_tx_gain_label)

    scan_tx_gain_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    scan_tx_gain_slider.setMinimum(0)
    scan_tx_gain_slider.setMaximum(127)
    scan_tx_gain_slider.setValue(127)
    scan_tx_gain_slider.setFixedWidth(155)
    scan_side_layout.addWidget(scan_tx_gain_slider)

    scan_tx_gain_value_label = QtWidgets.QLabel("127 / 127")
    scan_tx_gain_value_label.setAlignment(QtCore.Qt.AlignCenter)
    scan_side_layout.addWidget(scan_tx_gain_value_label)

    scan_display_label = QtWidgets.QLabel("Display Controls")
    scan_display_label.setStyleSheet("font-weight: bold; margin-top: 12px;")
    scan_side_layout.addWidget(scan_display_label)

    scan_max_range_label = QtWidgets.QLabel("Max Range (m)")
    scan_max_range_label.setStyleSheet("font-weight: bold; margin-top: 8px;")
    scan_side_layout.addWidget(scan_max_range_label)

    scan_max_range_spin = QtWidgets.QDoubleSpinBox()
    scan_max_range_spin.setMinimum(dr_target)
    scan_max_range_spin.setMaximum(30.0)
    scan_max_range_spin.setSingleStep(dr_target)
    scan_max_range_spin.setValue(r_max)
    scan_max_range_spin.setDecimals(1)
    scan_max_range_spin.setSuffix(" m")
    scan_max_range_spin.setFixedWidth(155)
    scan_side_layout.addWidget(scan_max_range_spin)

    scan_dynrange_label = QtWidgets.QLabel("Dyn Range (dB)")
    scan_dynrange_label.setStyleSheet("font-weight: bold; margin-top: 8px;")
    scan_side_layout.addWidget(scan_dynrange_label)

    color_scale = {'dyn_range': 60.0}

    scan_dynrange_spin = QtWidgets.QDoubleSpinBox()
    scan_dynrange_spin.setMinimum(1.0)
    scan_dynrange_spin.setMaximum(200.0)
    scan_dynrange_spin.setSingleStep(5.0)
    scan_dynrange_spin.setValue(color_scale['dyn_range'])
    scan_dynrange_spin.setDecimals(0)
    scan_dynrange_spin.setFixedWidth(155)
    scan_side_layout.addWidget(scan_dynrange_spin)

    def scan_update_dynrange(value):
        color_scale['dyn_range'] = value

    scan_dynrange_spin.valueChanged.connect(scan_update_dynrange)

    scan_zerobins_label = QtWidgets.QLabel("Zero Range Bins")
    scan_zerobins_label.setStyleSheet("font-weight: bold; margin-top: 8px;")
    scan_side_layout.addWidget(scan_zerobins_label)

    zero_bins_state = {'count': 0}

    scan_zerobins_spin = QtWidgets.QSpinBox()
    scan_zerobins_spin.setMinimum(0)
    scan_zerobins_spin.setMaximum(50)
    scan_zerobins_spin.setValue(0)
    scan_zerobins_spin.setFixedWidth(155)
    scan_side_layout.addWidget(scan_zerobins_spin)

    def scan_update_zerobins(value):
        zero_bins_state['count'] = value

    scan_zerobins_spin.valueChanged.connect(scan_update_zerobins)

    scan_target_label = QtWidgets.QLabel("Target Estimate")
    scan_target_label.setStyleSheet("font-weight: bold; margin-top: 12px;")
    scan_side_layout.addWidget(scan_target_label)

    scan_range_value_label = QtWidgets.QLabel("Range: --")
    scan_range_value_label.setStyleSheet(
        "font-size: 14px; color: black; font-weight: bold;")
    scan_side_layout.addWidget(scan_range_value_label)

    scan_angle_value_label = QtWidgets.QLabel("Angle: --")
    scan_angle_value_label.setStyleSheet(
        "font-size: 14px; color: black; font-weight: bold;")
    scan_side_layout.addWidget(scan_angle_value_label)

    scan_side_layout.addStretch()

    win.setParent(scan_container)
    scan_container_layout.addWidget(scan_side_panel)
    scan_container_layout.addWidget(win)
    scan_container.setLayout(scan_container_layout)
    scan_container.resize(1200, 800)
    scan_container.show()
    win.show()
    handles['container'] = scan_container

    def scan_update_tx_gain(value):
        scan_tx_gain_value_label.setText(f"{value} / 127")
        for device in sray.devices.values():
            if device.mode == "tx":
                for channel in device.channels:
                    channel.tx_gain = value
        sray.latch_tx_settings()

    scan_tx_gain_slider.valueChanged.connect(scan_update_tx_gain)

    def scan_update_max_range(value):
        range_state['r_max'] = value
        range_state['changed'] = True

    scan_max_range_spin.valueChanged.connect(scan_update_max_range)

    # PRF pacing
    sleep_time = 0.0
    if PRF and PRF > 0:
        T_rep = 1.0 / PRF
        sleep_time = max(0.0, T_rep - T)

    # Full-range grid for baseline
    r_max_full     = 30.0
    r_edges_full   = np.arange(0.0, r_max_full + dr_target, dr_target)
    r_centers_full = 0.5 * (r_edges_full[:-1] + r_edges_full[1:])

    # -------------------------------------------------------------------
    # Baseline clutter mesh capture (optional)
    # -------------------------------------------------------------------
    # Average N_baseline_frames with nobody in the scene to build a
    # per-angle, per-range reference.  Subtracting this during live scan
    # suppresses static returns (walls, furniture, mounts) and highlights
    # new / moving objects.
    N_baseline_frames = 20
    baseline_accum = np.zeros((n_bins, num_angles), dtype=np.float32)

    if cluttermesh:
        input("--> Capturing Baseline Environmental Clutter. "
              "Move out of the way and press Enter...")
        baseline_accum_full = np.zeros(
            (len(r_centers_full), num_angles), dtype=np.float32)
        for frame_idx in range(N_baseline_frames):
            for angle_idx, scan_angle in enumerate(angle_vals):
                sray.steer_rx(scan_angle, 0, cal_dict=analog_phase_cal)
                sray.steer_tx(scan_angle, 0, cal_dict=tx_phase_cal)

                data = None
                for _attempt in range(3):
                    try:
                        data = data_capture_cal(conv, cal_ant_fix)
                        break
                    except (BrokenPipeError, OSError):
                        time.sleep(0.05)
                if data is None:
                    continue

                sub1, sub2, sub4, sum_data = extract_rx_subarrays(
                    data, subarray_modes)

                heatmap_dummy_full = np.zeros(
                    (len(r_centers_full), num_angles), dtype=np.float32)
                heatmap_dummy_full, spec_db, xf = heatmap_gen(
                    sum_data, iq, cross_talk_freq, r_centers_full,
                    mag_floor_db=mag_floor_db,
                    angle_idx=angle_idx, heatmap=heatmap_dummy_full,
                    k=k, T=T,
                )
                baseline_accum_full[:, angle_idx] += heatmap_dummy_full[:, angle_idx]

        H_baseline = baseline_accum_full / N_baseline_frames
        handles['H_baseline'] = H_baseline
        handles['H_baseline_r_centers'] = r_centers_full.copy()

    maximize_by_title("FMCW Radar \u2013 Live")

    # ── Main scan loop ────────────────────────────────────────────────────
    try:
        while not handles.get('stop', False):
            # Rebuild range arrays if user changed max range
            if range_state['changed']:
                r_max = range_state['r_max']
                r_edges = np.arange(0.0, r_max + dr_target, dr_target)
                r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
                n_bins = len(r_centers)
                heatmap = np.full((n_bins, num_angles), 1e-6,
                                  dtype=np.float32)
                baseline_accum = np.zeros((n_bins, num_angles),
                                          dtype=np.float32)
                handles['H'] = heatmap.copy()
                handles['ranges'] = r_centers.copy()
                _angles = handles['angles']
                _n_angles = len(_angles)
                _dx = ((_angles[-1] - _angles[0]) / (_n_angles - 1)
                       if _n_angles > 1 else 1.0)
                _dy = ((r_centers[-1] - r_centers[0]) / (n_bins - 1)
                       if n_bins > 1 else 1.0)
                _x0 = _angles[0] - 0.5 * _dx
                _y0 = r_centers[0] - 0.5 * _dy
                _W = _dx * _n_angles
                _Hh = _dy * n_bins
                handles['image_item'].setRect(QtCore.QRectF(_x0, _y0, _W, _Hh))
                handles['plot_heat'].setYRange(0, r_max, padding=0)
                handles['plot_heat'].getAxis('left').setTickSpacing(
                    float(dr_target * 5), float(dr_target))
                range_state['changed'] = False

            for angle_idx, scan_angle in enumerate(angle_vals):
                if handles.get('stop', False):
                    break

                sray.steer_rx(scan_angle, 0, cal_dict=analog_phase_cal)
                sray.steer_tx(scan_angle, 0, cal_dict=tx_phase_cal)

                # Retry ADC capture on transient IIO broken-pipe errors
                data = None
                for _attempt in range(3):
                    try:
                        data = data_capture_cal(conv, cal_ant_fix)
                        break
                    except (BrokenPipeError, OSError):
                        time.sleep(0.05)
                if data is None:
                    continue

                sub1, sub2, sub4, sum_data = extract_rx_subarrays(
                    data, subarray_modes)

                heatmap_dummy = np.zeros_like(baseline_accum)
                heatmap_dummy, spec_db, xf = heatmap_gen(
                    sum_data, iq, cross_talk_freq, r_centers,
                    mag_floor_db=mag_floor_db,
                    angle_idx=angle_idx, heatmap=heatmap_dummy,
                    k=k, T=T,
                )

                # Lazily create beat panel when xf is first known
                if handles.get('f_axis') is None or handles['f_axis'] is None:
                    handles['f_axis'] = xf.copy()
                    plot_beat = handles['window'].addPlot(row=1, col=0)
                    plot_beat.setLabel('bottom', 'Range (m)')
                    plot_beat.setLabel('left', 'Amplitude (dB)')
                    plot_beat.setTitle("Beat Spectrum @ angle: --")
                    beat_curve = plot_beat.plot(
                        [], [], pen=pg.mkPen((0, 255, 255), width=2))
                    handles['plot_beat'] = plot_beat
                    handles['beat_curve'] = beat_curve
                    handles['S'] = np.full(
                        (len(xf), num_angles), np.nan, dtype=float)

                    plot_adc = handles['window'].addPlot(row=2, col=0)
                    plot_adc.setLabel('bottom', 'Sample')
                    plot_adc.setLabel('left', 'Amplitude')
                    plot_adc.setTitle("Summed ADC Data (I/Q)")
                    plot_adc.addLegend()
                    adc_curve_i = plot_adc.plot(
                        [], [], pen=pg.mkPen((0, 200, 255), width=1), name='I')
                    adc_curve_q = plot_adc.plot(
                        [], [], pen=pg.mkPen((255, 100, 0), width=1), name='Q')
                    handles['plot_adc'] = plot_adc
                    handles['adc_curve_i'] = adc_curve_i
                    handles['adc_curve_q'] = adc_curve_q

                    _iq_one = iq[:N_ref]
                    _iq_t = np.arange(len(_iq_one)) / fs
                    plot_ideal = handles['window'].addPlot(row=3, col=0)
                    plot_ideal.setLabel('bottom', 'Time (s)')
                    plot_ideal.setLabel('left', 'Amplitude')
                    plot_ideal.setTitle("Ideal Transmit Waveform")
                    plot_ideal.addLegend()
                    handles['plot_ideal'] = plot_ideal
                    handles['ideal_curve_i'] = plot_ideal.plot(
                        _iq_t, _iq_one.real,
                        pen=pg.mkPen((0, 255, 120), width=1), name='I')
                    handles['ideal_curve_q'] = plot_ideal.plot(
                        _iq_t, _iq_one.imag,
                        pen=pg.mkPen((255, 200, 0), width=1), name='Q')
                    handles['ideal_t'] = _iq_t

                # Convert beat frequency axis to range (m) and clip
                range_axis = (c * xf) / (2.0 * k)
                range_mask = range_axis <= range_state['r_max']
                range_limited = range_axis[range_mask]
                spec_db_limited = spec_db[range_mask]

                _baseline = handles.get('H_baseline')
                _baseline_r = handles.get('H_baseline_r_centers')
                if _baseline is not None and _baseline_r is not None:
                    baseline_col = np.interp(
                        r_centers, _baseline_r, _baseline[:, angle_idx],
                        left=0.0, right=0.0)
                    heatmap_dummy[:, angle_idx] = np.abs(
                        heatmap_dummy[:, angle_idx] - baseline_col)

                # Zero the lowest N range bins (suppress near-field / cross-talk)
                zb = zero_bins_state['count']
                if zb > 0:
                    heatmap_dummy[:min(zb, n_bins), angle_idx] = 1e-6

                update_fmcw_radar_viewer(
                    handles, angle_idx,
                    heatmap_col=heatmap_dummy[:, angle_idx],
                    spectrum=spec_db, rescale=True,
                    show_this_angle=True,
                    dyn_range=color_scale['dyn_range'],
                )
                handles['beat_curve'].setData(range_limited, spec_db_limited)

                if handles.get('adc_curve_i') is not None:
                    n_samples = len(sum_data)
                    sample_axis = np.arange(n_samples)
                    handles['adc_curve_i'].setData(sample_axis, sum_data.real)
                    handles['adc_curve_q'].setData(sample_axis, sum_data.imag)

                handles['app'].processEvents()
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # Post-scan target estimate: after sweeping all angles, find the
            # brightest cell in the heatmap to estimate the target's range
            # and angle.
            H_current = handles['H']
            if H_current.size > 0:
                update_fmcw_radar_viewer(
                    handles, 0, dyn_range=color_scale['dyn_range'])

                max_flat = np.argmax(H_current)
                peak_r_idx, peak_a_idx = np.unravel_index(max_flat, H_current.shape)
                peak_range_m = r_centers[peak_r_idx] if peak_r_idx < len(r_centers) else 0.0
                peak_angle_deg = angle_vals[peak_a_idx] if peak_a_idx < len(angle_vals) else 0.0
                scan_range_value_label.setText(f"Range: {peak_range_m:.2f} m")
                scan_angle_value_label.setText(
                    f"Angle: {-1.0 * peak_angle_deg:.1f}\u00b0")
                handles['app'].processEvents()

        minimise_by_title("FMCW Radar \u2013 Live")

    except KeyboardInterrupt:
        pass

    except Exception as e:
        raise
