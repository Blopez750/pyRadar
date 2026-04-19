"""Standalone test procedures: drone capture, elevation scan, azimuth scan."""
import numpy as np
from radar_utils.utils import is_key_pressed
import time

from radar_utils.signal_processing import RDRConfig, freq_process
from radar_utils.hardware_setup import get_radar_data


def drone_test_procedure(
    conv, sray, iq, cal_ant_fix, subarray_modes, rx_phase_cal, tx_phase_cal,
    num_chirps, PRF, BW, output_freq, save_dir, az_angle=0, el_angle=0,
    num_captures=100, mti_filter=True, max_velocity_bins=20, max_range_m=100,
    enable_plot=True,
):
    """
    Single-run drone test procedure for range-doppler radar.

    Captures radar data for a single test condition with configurable beam steering.
    User provides a filename for the test data.

    Workflow:
    - Creates a timestamped folder for the test
    - User provides filename for test data
    - User positions the drone
    - System steers beam to specified azimuth/elevation
    - Captures data with range-doppler processing
    - Saves HDF5 file and JSON parameters file to timestamped folder

    Parameters:
        conv, sray, iq, cal_ant_fix, subarray_modes, rx_phase_cal, tx_phase_cal:
            Hardware and calibration objects.
        num_chirps, PRF, BW, output_freq: Radar waveform parameters.
        save_dir: Base directory for test data.
        az_angle, el_angle: Beam steering angles (degrees).
        num_captures: Number of captures (None = indefinite).
        mti_filter: Enable MTI filtering.
        max_velocity_bins: Velocity display bins.
        max_range_m: Max display range (m).
        enable_plot: Enable real-time plotting.

    Returns:
        str: Path to timestamped test folder.
    """
    import os
    import json
    import h5py
    from datetime import datetime

    os.makedirs(save_dir, exist_ok=True)

    test_timestamp = datetime.now()
    test_folder_name = test_timestamp.strftime('%Y-%m-%d_%H-%M-%S')
    test_folder_path = os.path.join(save_dir, test_folder_name)
    os.makedirs(test_folder_path, exist_ok=True)

    print("\n" + "=" * 60)
    print("DRONE TEST DATA CAPTURE")
    print("=" * 60)
    print(f"Test folder: {test_folder_name}")
    filename = input("Enter filename for this test (without extension): ").strip()
    if not filename:
        filename = "drone_test"
        print(f"Using default filename: {filename}")

    # ── Range-Doppler configuration ──────────────────────────────────────
    PRI_ms = 1000 / PRF
    signal_freq = 10e6
    min_scale = 225
    max_scale = 250

    (good_ramp_samples, start_offset_samples, N_frame, dist,
     r_res, v_res, max_doppler_freq, max_doppler_vel) = RDRConfig(
        conv, PRI_ms, BW, num_chirps, signal_freq, output_freq)

    print(f"RDRConfig results: r_res={r_res}, v_res={v_res}")

    # ── Optional PyQtGraph display ───────────────────────────────────────
    if enable_plot:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtWidgets, QtCore

        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        win = pg.GraphicsLayoutWidget(show=True)
        win.resize(1800, 1000)
        win.setWindowTitle("Drone Test - Range-Doppler Processing")

        rd_plot = win.addPlot(row=0, col=0, rowspan=3,
                              title="Range-Doppler Map")
        rd_plot.setLabel('left', 'Range (m)')
        rd_plot.setLabel('bottom', 'Velocity (m/s)')
        rd_img = pg.ImageItem()
        rd_plot.addItem(rd_img)
        rd_cmap = pg.colormap.get('inferno')
        rd_img.setColorMap(rd_cmap)
        rd_cbar = pg.ColorBarItem(colorMap=rd_cmap, values=(0, 10),
                                  limits=(0, 20))
        rd_cbar.setImageItem(rd_img, insert_in=rd_plot)
        rd_stats = pg.TextItem(color='w', anchor=(0, 0))
        rd_plot.addItem(rd_stats)

        sumiq_plot = win.addPlot(row=0, col=1,
                                 title="Sum Data (Real) - Raw Captured")
        sumiq_plot.setLabel('left', 'Amplitude')
        sumiq_plot.setLabel('bottom', 'Sample')
        sumiq_plot.showGrid(x=True, y=True, alpha=0.3)
        sumiq_curve = sumiq_plot.plot(pen=pg.mkPen('c', width=1))

        fft_plot = win.addPlot(row=1, col=1, title="Radar Data (FFT)")
        fft_plot.setLabel('left', 'Magnitude (dB)')
        fft_plot.setLabel('bottom', 'Range Bin')
        fft_plot.showGrid(x=True, y=True, alpha=0.3)
        fft_curves = []

        idealiq_plot = win.addPlot(row=2, col=1,
                                   title="Ideal IQ (Real) - 1 Chirp")
        idealiq_plot.setLabel('left', 'Amplitude')
        idealiq_plot.setLabel('bottom', 'Sample')
        idealiq_plot.showGrid(x=True, y=True, alpha=0.3)
        idealiq_curve = idealiq_plot.plot(pen=pg.mkPen('c', width=1))

        samples_per_chirp = good_ramp_samples
        idealiq_curve.setData(np.arange(samples_per_chirp),
                              np.real(iq[:samples_per_chirp]))
        idealiq_plot.setXRange(0, samples_per_chirp, padding=0.02)
        print("Real-time plotting enabled.")
    else:
        print("Real-time plotting disabled for faster data collection.")

    # ── Display test configuration ───────────────────────────────────────
    print(f"\nTest Configuration:")
    print(f"  Azimuth: {az_angle}\u00b0")
    print(f"  Elevation: {el_angle}\u00b0")
    print(f"  Beam Steering: TX and RX")
    if num_captures is None:
        print(f"  Number of captures: Indefinite (press 'q' to stop)")
    else:
        print(f"  Number of captures: {num_captures}")
    print(f"  Range resolution: {r_res:.3f} m")
    print(f"  Velocity resolution: {v_res:.3f} m/s")
    print(f"  MTI filter: {'Enabled' if mti_filter else 'Disabled'}")
    print(f"  Real-time plot: {'Enabled' if enable_plot else 'Disabled'}")
    print("=" * 60)

    print(f"\nBeam will be steered to Az={az_angle}\u00b0, El={el_angle}\u00b0")
    input("Position your target and press ENTER to start capture...")

    print(f"\nSteering beam to Az={az_angle}\u00b0, El={el_angle}\u00b0...")
    sray.steer_rx(az_angle, el_angle, cal_dict=rx_phase_cal)
    sray.steer_tx(az_angle, el_angle, cal_dict=tx_phase_cal)

    if num_captures is None:
        print(f"\nStarting indefinite captures (press 'q' to stop)...")
    else:
        print(f"\nStarting {num_captures} captures...")

    raw_adc_data = []
    subarray1_data = []
    subarray2_data = []
    subarray4_data = []
    capture_idx = 0
    quit_requested = False

    try:
        while True:
            if num_captures is not None and capture_idx >= num_captures:
                break

            if is_key_pressed('q'):
                print("\n  'q' pressed - stopping captures...")
                quit_requested = True
                time.sleep(0.3)
                break

            if num_captures is None:
                if (capture_idx + 1) % 10 == 0 or capture_idx == 0:
                    print(f"  Capture {capture_idx + 1}... (press 'q' to stop)")
            else:
                if (capture_idx + 1) % 10 == 0 or capture_idx == 0:
                    print(f"  Capture {capture_idx + 1}/{num_captures}...")

            t_capture_start = time.perf_counter()

            rx_bursts, sum_data, subarray_data, _ = get_radar_data(
                conv, cal_ant_fix, subarray_modes, num_chirps, PRI_ms,
                start_offset_samples, good_ramp_samples,
                coherent_integration=False)
            raw_adc_data.append(sum_data)
            subarray1_data.append(subarray_data[1])
            subarray2_data.append(subarray_data[2])
            subarray4_data.append(subarray_data[4])

            # Dechirp
            num_samples_per_chirp = rx_bursts.shape[1]
            iq_chirp = iq[:num_samples_per_chirp]
            rx_bursts_mixed = rx_bursts * np.conj(iq_chirp)

            # Range-Doppler processing (fix: unpack tuple return)
            radar_data, _ = freq_process(
                rx_bursts_mixed, min_scale, max_scale,
                use_window=True, mti_filter=mti_filter)

            # ── Plot update ──────────────────────────────────────────────
            if enable_plot:
                samples_to_plot = min(good_ramp_samples, len(sum_data))
                sumiq_curve.setData(
                    np.arange(samples_to_plot),
                    np.real(sum_data[:samples_to_plot]))
                sumiq_plot.setTitle(
                    f'Sum Data (Real) - Capture {capture_idx + 1}')
                for curve in fft_curves:
                    fft_plot.removeItem(curve)
                fft_curves.clear()
                for range_idx in range(radar_data.shape[0]):
                    curve = fft_plot.plot(
                        radar_data[range_idx, :],
                        pen=pg.mkPen((100, 200, 255, 100), width=1))
                    fft_curves.append(curve)
                fft_plot.setTitle(
                    f'Radar Data (FFT) - All {radar_data.shape[0]} '
                    f'range bins - Capture {capture_idx + 1}')

                num_doppler = radar_data.shape[0]
                num_range_bins = radar_data.shape[1]

                if max_velocity_bins < num_doppler:
                    center = num_doppler // 2
                    half = max_velocity_bins // 2
                    start_idx = max(0, center - half)
                    end_idx = min(num_doppler, center + half)
                    rd_cropped = radar_data[start_idx:end_idx, :]
                else:
                    rd_cropped = radar_data

                max_range_bins = min(num_range_bins,
                                     int(max_range_m / r_res))
                rd_cropped = rd_cropped[:, :max_range_bins]

                vel_bins = rd_cropped.shape[0]
                vel_half_span = vel_bins / 2 * v_res
                vel_min = -vel_half_span
                vel_max = vel_half_span
                range_max_display = rd_cropped.shape[1] * r_res

                if (not np.isfinite(vel_min) or not np.isfinite(vel_max)
                        or not np.isfinite(range_max_display)):
                    print(f"WARNING: Invalid axis values detected!")
                    capture_idx += 1
                    continue

                vmin = min_scale
                vmax = max_scale

                if capture_idx % 10 == 0:
                    actual_min = np.min(rd_cropped)
                    actual_max = np.max(rd_cropped)
                    print(f"Data range: min={actual_min:.2f}, "
                          f"max={actual_max:.2f} "
                          f"(clipped to {vmin} - {vmax})")

                rd_img.setImage(rd_cropped, autoLevels=False)
                rd_img.setLevels([vmin, vmax])
                rd_img.setRect(pg.QtCore.QRectF(
                    vel_min, 0, vel_max - vel_min, range_max_display))
                rd_plot.setXRange(vel_min, vel_max, padding=0)
                rd_plot.setYRange(0, range_max_display, padding=0)

                peak_idx = np.unravel_index(
                    np.argmax(rd_cropped), rd_cropped.shape)
                peak_vel = (peak_idx[0] - vel_bins / 2) * v_res
                peak_range = peak_idx[1] * r_res
                peak_db = np.max(rd_cropped)

                rd_stats.setText(
                    f"Capture: {capture_idx + 1}\n"
                    f"Peak: {peak_db:.1f} dB\n"
                    f"Range: {peak_range:.1f} m\n"
                    f"Vel: {peak_vel:.1f} m/s\n"
                    f"Scale: {vmin:.1f} to {vmax:.1f} dB")
                app.processEvents()
                rd_cbar.setLevels(values=(vmin, vmax))

            t_capture = time.perf_counter() - t_capture_start
            print(f"  Capture {capture_idx + 1} completed in "
                  f"{t_capture*1000:.2f} ms")
            capture_idx += 1

    except KeyboardInterrupt:
        print("\n\nCapture interrupted by user (Ctrl+C)")

    actual_num_captures = len(raw_adc_data)
    print(f"\nCompleted {actual_num_captures} captures.")

    # ── Save HDF5 ────────────────────────────────────────────────────────
    print(f"\nSaving data to HDF5...")
    h5_filename = f"{filename}.h5"
    h5_filepath = os.path.join(test_folder_path, h5_filename)

    raw_adc_array = np.array(raw_adc_data)
    subarray1_array = np.array(subarray1_data)
    subarray2_array = np.array(subarray2_data)
    subarray4_array = np.array(subarray4_data)

    with h5py.File(h5_filepath, 'w') as f:
        f.create_dataset('raw_adc_data', data=raw_adc_array,
                         compression='gzip', compression_opts=4)
        f.create_dataset('subarray1_data', data=subarray1_array,
                         compression='gzip', compression_opts=4)
        f.create_dataset('subarray2_data', data=subarray2_array,
                         compression='gzip', compression_opts=4)
        f.create_dataset('subarray4_data', data=subarray4_array,
                         compression='gzip', compression_opts=4)

        iq_one_chirp = iq[:good_ramp_samples]
        f.create_dataset('iq_reference', data=iq_one_chirp,
                         compression='gzip', compression_opts=4)

        f.attrs['filename'] = filename
        f.attrs['azimuth'] = az_angle
        f.attrs['elevation'] = el_angle
        f.attrs['num_captures'] = actual_num_captures
        f.attrs['timestamp'] = time.time()
        f.attrs['num_chirps'] = num_chirps
        f.attrs['PRF'] = PRF
        f.attrs['BW'] = BW
        f.attrs['output_freq'] = output_freq
        f.attrs['range_resolution_m'] = r_res
        f.attrs['velocity_resolution_mps'] = v_res
        f.attrs['mti_filter'] = mti_filter
        f.attrs['max_velocity_bins'] = max_velocity_bins
        f.attrs['max_range_m'] = max_range_m

        active_subarrays = [sa_id for sa_id, mode in subarray_modes.items()
                            if isinstance(mode, str) and mode.lower() == 'rx']
        f.attrs['active_subarrays'] = str(active_subarrays)
        f.attrs['subarray_mapping'] = (
            "subarray1_data=Subarray1(data[3]), "
            "subarray2_data=Subarray2(data[1]), "
            "subarray4_data=Subarray4(data[2])")

    print(f"Data saved to: {h5_filepath}")

    # ── Save JSON parameters ─────────────────────────────────────────────
    print(f"Saving test parameters to JSON...")
    json_filename = f"{filename}_parameters.json"
    json_filepath = os.path.join(test_folder_path, json_filename)

    test_parameters = {
        "test_info": {
            "timestamp": test_timestamp.isoformat(),
            "timestamp_unix": time.time(),
            "folder_name": test_folder_name,
            "filename": filename,
            "h5_file": h5_filename,
            "json_file": json_filename,
        },
        "beam_steering": {
            "azimuth_deg": az_angle,
            "elevation_deg": el_angle,
        },
        "radar_config": {
            "num_chirps": num_chirps,
            "PRF_Hz": PRF,
            "PRI_ms": PRI_ms,
            "BW_Hz": BW,
            "output_freq_Hz": output_freq,
            "signal_freq_Hz": signal_freq,
            "rx_buffer_size": conv.rx_buffer_size,
        },
        "processing": {
            "mti_filter": mti_filter,
            "max_velocity_bins": max_velocity_bins,
            "max_range_m": max_range_m,
            "min_scale_dB": min_scale,
            "max_scale_dB": max_scale,
        },
        "resolutions": {
            "range_resolution_m": r_res,
            "velocity_resolution_mps": v_res,
            "max_doppler_freq_Hz": max_doppler_freq,
            "max_doppler_vel_mps": max_doppler_vel,
        },
        "capture_info": {
            "num_captures_requested": num_captures,
            "num_captures_actual": actual_num_captures,
            "good_ramp_samples": good_ramp_samples,
            "start_offset_samples": start_offset_samples,
            "N_frame": N_frame,
        },
        "subarrays": {
            "modes": {str(k): v for k, v in subarray_modes.items()},
            "active_rx_subarrays": active_subarrays,
            "mapping": {
                "subarray1": "data[3]",
                "subarray2": "data[1]",
                "subarray4": "data[2]",
            },
        },
        "data_files": {
            "h5_filepath": h5_filepath,
            "json_filepath": json_filepath,
            "datasets": {
                "raw_adc_data":
                    f"shape: ({actual_num_captures}, {raw_adc_array.shape[1]})",
                "subarray1_data":
                    f"shape: ({actual_num_captures}, {subarray1_array.shape[1]})",
                "subarray2_data":
                    f"shape: ({actual_num_captures}, {subarray2_array.shape[1]})",
                "subarray4_data":
                    f"shape: ({actual_num_captures}, {subarray4_array.shape[1]})",
                "iq_reference":
                    f"shape: ({len(iq_one_chirp)},)",
            },
        },
    }

    with open(json_filepath, 'w') as json_file:
        json.dump(test_parameters, json_file, indent=4)

    print(f"Parameters saved to: {json_filepath}")

    if enable_plot:
        win.close()

    print(f"\n{'=' * 60}")
    print("DATA CAPTURE COMPLETE")
    print(f"{'=' * 60}")
    print(f"Test folder: {test_folder_path}")
    print(f"Total captures: {actual_num_captures}")
    print(f"H5 file: {h5_filename}")
    print(f"JSON file: {json_filename}")
    print(f"{'=' * 60}")

    return test_folder_path


def elevation_scan_test(sray, tx_phase_cal, el_min=-30, el_max=30,
                        el_step=1, az_angle=0, delay_ms=100):
    """
    Continuous elevation scanning test for TX beam steering.

    Repeatedly scans TX beam in elevation. No data capture.
    Press 'q' to stop.
    """
    print("=" * 60)
    print("ELEVATION SCAN TEST - TX BEAM STEERING")
    print("=" * 60)
    print(f"Elevation range: {el_min}\u00b0 to {el_max}\u00b0")
    print(f"Step size: {el_step}\u00b0")
    print(f"Azimuth (fixed): {az_angle}\u00b0")
    print(f"Delay: {delay_ms} ms")
    print("\nPress 'q' to stop scanning")
    print("=" * 60)

    scan_count = 0

    try:
        while True:
            scan_count += 1
            print(f"\nScan cycle #{scan_count}")

            for el_angle in np.arange(el_min, el_max + el_step / 2, el_step):
                if is_key_pressed('q'):
                    print("\nStopping scan (user pressed 'q')")
                    return

                sray.steer_tx(az_angle, el_angle, cal_dict=tx_phase_cal)
                print(f"  Elevation: {el_angle:+6.1f}\u00b0", end='\r')
                time.sleep(delay_ms / 1000.0)

            print()

    except KeyboardInterrupt:
        print("\n\nStopping scan (keyboard interrupt)")

    print("=" * 60)
    print("ELEVATION SCAN TEST COMPLETE")
    print(f"Total scan cycles: {scan_count}")
    print("=" * 60)


def azimuth_scan_test(sray, tx_phase_cal, az_min=-30, az_max=30,
                      az_step=1, el_angle=0, delay_ms=100):
    """
    Continuous azimuth scanning test for TX beam steering.

    Repeatedly scans TX beam in azimuth. No data capture.
    Press 'q' to stop.
    """
    print("=" * 60)
    print("AZIMUTH SCAN TEST - TX BEAM STEERING")
    print("=" * 60)
    print(f"Azimuth range: {az_min}\u00b0 to {az_max}\u00b0")
    print(f"Step size: {az_step}\u00b0")
    print(f"Elevation (fixed): {el_angle}\u00b0")
    print(f"Delay: {delay_ms} ms")
    print("\nPress 'q' to stop scanning")
    print("=" * 60)

    scan_count = 0

    try:
        while True:
            scan_count += 1
            print(f"\nScan cycle #{scan_count}")

            for az_angle in np.arange(az_min, az_max + az_step / 2, az_step):
                if is_key_pressed('q'):
                    print("\nStopping scan (user pressed 'q')")
                    return

                sray.steer_tx(az_angle, el_angle, cal_dict=tx_phase_cal)
                print(f"  Azimuth: {az_angle:+6.1f}\u00b0", end='\r')
                time.sleep(delay_ms / 1000.0)

            print()

    except KeyboardInterrupt:
        print("\n\nStopping scan (keyboard interrupt)")

    print("=" * 60)
    print("AZIMUTH SCAN TEST COMPLETE")
    print(f"Total scan cycles: {scan_count}")
    print("=" * 60)
