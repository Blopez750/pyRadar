# =============================================================================
# fmcw_range.py — Single-beam FMCW range measurement pilot
# =============================================================================
# This pilot demonstrates the simplest FMCW radar measurement: detecting the
# range to a single target along the current beam direction (no scanning or
# Doppler processing).
#
# Procedure:
#   **Live loop**: captures IQ data, dechirps by multiplying with conj(iq),
#   FFTs the beat signal, removes interference, and estimates range from
#   the peak beat frequency using:  R = c · T · f_beat / (2 · BW)
# =============================================================================

"""FMCW single-beam range measurement pilot with PyQtGraph display."""
import numpy as np
from radar_utils.utils import is_key_pressed

from radar_utils.hardware_setup import extract_rx_subarrays, build_rx_channel_config, SUBARRAY_TO_ADC
from radar_utils.calibration import data_capture_cal
from radar_utils.radar_plotting import init_fmcw_range_gui, update_fmcw_range_gui


def FMCWRange(conv, tddn, cal_ant_fix, subarray_modes, iq, BW, PRF, duty_cycle):
    """
    Computes the range profile from FMCW radar data using only RX subarrays.

    Parameters:
        conv: Radar device interface.
        tddn: TDD controller (unused, kept for interface compatibility).
        cal_ant_fix: Calibration antenna setting.
        subarray_modes: dict  {1:"rx", 2:"rx", 3:"tx", 4:"rx"}.
        iq: Ideal IQ chirp waveform.
        BW: Bandwidth (Hz).
        PRF: Pulse repetition frequency (Hz).
        duty_cycle: Duty cycle (unused, kept for interface compatibility).
    """
    c = 3e8
    # T must be the single chirp duration, NOT the tiled-buffer duration.
    # iq is tiled (num_chirps repetitions), so len(iq)/fs = T_total = num_chirps × T_chirp.
    # PRF = fs / samples_per_chirp  →  T_chirp = 1/PRF.
    # Using T_total here makes the LPF cutoff 32× too narrow (passes only ~4.7 m instead
    # of 150 m) AND stretches the range axis by 32× (15 m target appears at 480 m).
    T = 1 / PRF  # single chirp duration [s]

    # ── Subarray / ADC configuration summary ─────────────────────────────
    rx_adc_channels, subarray_to_data_index = build_rx_channel_config(subarray_modes)
    all_adc_channels = set(SUBARRAY_TO_ADC.values())
    disabled_adc_channels = sorted(all_adc_channels - set(rx_adc_channels))
    rx_subarrays = [sa for sa, m in subarray_modes.items() if m.lower() == "rx"]
    tx_subarrays = [sa for sa, m in subarray_modes.items() if m.lower() == "tx"]
    print(f"  TX subarrays: {tx_subarrays}  |  RX subarrays: {rx_subarrays}")
    print(f"  Enabled ADC channels: {rx_adc_channels}  |  Disabled ADC channels: {disabled_adc_channels}")
    print(f"  Subarray → data index: {subarray_to_data_index}")
    print(f"  Chirp duration T: {T*1e6:.2f} µs  ({len(iq)} samples)")
    print(f"  Chirp slope k: {BW/T:.3e} Hz/s")
    print(f"  Range resolution: {c / (2 * BW):.2f} m")
    print(f"  Maximum unambiguous range: {c * T * conv.rx_sample_rate / (4 * BW):.2f} m")

    # ── PyQtGraph GUI ────────────────────────────────────────────────────
    gui = init_fmcw_range_gui()

    while True:
        data = data_capture_cal(conv, cal_ant_fix)
        sub1, sub2, sub4, sum_data = extract_rx_subarrays(data, subarray_modes)

        # Dechirp: multiply received signal by the TX chirp replica
        n = min(len(sum_data), len(iq))
        _iq_is_complex = np.iscomplexobj(iq)
        if _iq_is_complex:
            s_beat_raw = sum_data[:n] * np.conj(iq[:n])
        else:
            s_beat_raw = sum_data[:n] * iq[:n]

        N = len(s_beat_raw)

        if _iq_is_complex:
            # Complex dechirp: beat is at negative frequency (f = −k·τ).
            # Take second half of FFT (negative-freq bins) and flip so
            # range increases with index, matching the real-chirp path.
            S = np.fft.fft(s_beat_raw)
            xf_full = np.fft.fftfreq(N, 1 / conv.rx_sample_rate)
            xf = np.abs(xf_full[N // 2:][::-1])     # |neg freqs|, ascending
            yf = np.abs(S[N // 2:][::-1])
        else:
            # Real waveform: LPF to reject sum-frequency product
            S = np.fft.fft(s_beat_raw)
            freqs = np.fft.fftfreq(N, 1 / conv.rx_sample_rate)
            max_beat_hz = 2 * BW * 150 / (c * T)
            S[np.abs(freqs) > max_beat_hz] = 0
            s_beat = np.fft.ifft(S).real
            yf = np.fft.fft(s_beat)
            xf = np.fft.fftfreq(N, 1 / conv.rx_sample_rate)
            xf = xf[:N // 2]
            yf = np.abs(yf[:N // 2])

        # Convert frequency axis to range: R = c * T * f / (2 * BW)
        range_axis = c * T * xf / (2 * BW)

        # Zero DC and first bin (near-field / cross-talk below ~0.6 m)
        yf[0] = 0
        yf[1] = 0

        # Normalize
        yf_max = np.max(yf)
        if yf_max > 0:
            yf = yf / yf_max

        # Peak detection in range domain
        peak_bin = np.argmax(yf)
        R_calculated = range_axis[peak_bin]
        beat_freq = xf[peak_bin]

        print(f"Peak range: {R_calculated:.2f} m  |  Beat freq: {beat_freq/1e3:.2f} kHz  |  Bin: {peak_bin}")

        update_fmcw_range_gui(gui, sub1, sub2, sub4, sum_data,
                              R_calculated, range_axis, yf,
                              s_beat, conv.rx_sample_rate, T, BW)

        if is_key_pressed('q') or not gui['win'].isVisible():
            print("Exiting FMCW Range processing.")
            gui['win'].close()
            break
