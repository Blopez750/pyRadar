# =============================================================================
# main.py — Entry point for the X-Band phased-array radar (Stingray / XBDP)
# =============================================================================
#
#  ┌─────────────────────────────────────────────────────────────────────┐
#  │                      SYSTEM OVERVIEW                                │
#  │                                                                     │
#  │  This radar is a monostatic, X-band, phased-array FMCW system:      │
#  │                                                                     │
#  │    • Carrier frequency : 10.4 GHz  (X-band: 8–12 GHz)               │
#  │    • Waveform          : FMCW (Frequency-Modulated Continuous Wave) │
#  │    • Bandwidth         : 250 MHz → range resolution ≈ 0.6 m         │
#  │    • Antenna            : 32-element phased array (4 × 8 grid)      │
#  │    • Beamforming       : analogue (ADAR1000 per-element gain/phase) │
#  │    • Digitiser         : AD9081 MxFE (4 ADC + 4 DAC channels)       │
#  │    • Platform          : Xilinx ZCU102 FPGA + Stingray board        │
#  │                                                                     │
#  │  Signal chain (transmit):                                           │
#  │    FPGA DAC → AD9081 TX NCO → XUD1A up-converter → ADAR1000         │
#  │    beamformer → antenna elements                                    │
#  │                                                                     │
#  │  Signal chain (receive):                                            │
#  │    antenna elements → ADAR1000 beamformer (LNA + phase shifter)     │
#  │    → XUD1A down-converter → AD9081 RX ADC → FPGA DMA → host PC      │
#  │                                                                     │
#  │  FMCW principle (quick summary):                                    │
#  │    The transmitter sends a chirp — a signal whose frequency sweeps  │
#  │    linearly from f0 to f0+BW over chirp period T = 1/PRF.           │
#  │    A target at range R reflects the chirp back with a round-trip    │
#  │    delay τ = 2R/c.  Mixing (multiplying) the received signal with   │
#  │    a copy of the transmitted chirp produces a constant-frequency    │
#  │    "beat" tone:                                                     │
#  │                                                                     │
#  │        f_beat = (BW / T) · τ  =  2·R·BW / (c·T)                     │
#  │                                                                     │
#  │    An FFT converts the beat signal from time to frequency, and      │
#  │    each frequency bin maps directly to a range.                     │
#  │                                                                     │
#  │  Monopulse angle estimation:                                        │
#  │    By forming Sum (Σ) and Difference (Δ) beams from pairs of        │
#  │    subarrays, the ratio Δ/Σ gives a single-measurement angle        │
#  │    estimate (azimuth or elevation) without mechanical scanning.     │
#  │                                                                     │
#  └─────────────────────────────────────────────────────────────────────┘
#
# This is the top-level script that ties everything together:
#   1. Configure radar parameters (bandwidth, chirps, duty cycle, etc.)
#   2. Run antenna calibration (or load a saved one)
#   3. Connect to the hardware (AD9081, ADAR1000, XUD1A, TDDN)
#   4. Launch one of the Pilot demos (FMCW Range, FMCW Scan)
#
# "Pilot" = a ready-to-run demonstration that showcases a specific radar mode.
#
# Newcomers: start by reading the menus at the bottom of this file, then
# trace upward to see how each option calls into the radar_utils/ library.
# =============================================================================

import os
import subprocess

# ── Temporary FPGA network link ──────────────────────────────────────────────
from radar_utils.network import ensure_fpga_network
ensure_fpga_network()

try:
    import PyQt5
    os.environ.setdefault(
        'QT_QPA_PLATFORM_PLUGIN_PATH',
        os.path.join(os.path.dirname(PyQt5.__file__), 'Qt5', 'plugins', 'platforms')
    )
except ImportError:
    pass

# ── Explicit imports (Ctrl-click any name to jump to its definition) ─────────
from radar_utils.calibration    import enable_stingray_channel, disable_stingray_channel
from radar_utils.hardware_setup import build_rx_channel_config, setup_ad9081, setup_xud1a, setup_stingray
from radar_utils.sync_config    import sys_sync, sync_disable
from radar_utils.tx_rx_cal      import setup, rx_cal_full, tx_cal_full
from radar_utils.cal_manager    import save_calibration, load_latest_calibration, apply_calibration, purge_stale_calibrations
from radar_utils.signal_processing import set_fft_window
from pilot_functions import FMCWRange, FMCWScan
import adi
from custom_libs.adar1000 import adar1000

# =============================================================================
# Configuration parameters — edit these to match your hardware & experiment
# =============================================================================
#
# The numbers below set the physical operating point of the radar.  Here is
# how each parameter relates to the radar's fundamental performance:
#
#   BW (chirp bandwidth):
#       Range resolution  R_res = c / (2 · BW)
#       With BW = 250 MHz → R_res = 3×10⁸ / (2 × 250×10⁶) = 0.6 m
#       Wider bandwidth → finer range resolution, but requires more ADC BW.
#
#   num_chirps (number of chirps per coherent processing interval):
#       Velocity resolution  v_res = λ / (2 · N · PRI)
#       More chirps → finer velocity resolution.  Also sets the Doppler FFT
#       size: with 16 chirps we get 16 velocity bins before zero-padding.
#
#   PRF (pulse repetition frequency, computed from buffer_size / num_chirps):
#       Max unambiguous velocity  v_max = λ / (4 · PRI)  =  PRF · λ / 4
#       Higher PRF → higher max velocity, but shorter max unambiguous range.
#       Max unambiguous range  R_ua = c / (2 · PRF)
#
#   duty_cycle:
#       Fraction of the PRI during which the TX is active.  FMCW uses 1.0
#       (transmit continuously).  Pulsed mode uses < 1.0.
#
#   source_freq_ghz:
#       The carrier frequency of the external signal generator used during
#       calibration.  During normal operation the TX waveform is generated
#       internally by the AD9081 DAC + XUD1A up-converter.
#
uri = "ip:192.168.0.101"            # IP address of the FPGA / ZCU102
N_rx = 2**12                         # ADC buffer size (samples per capture)
BW = 250e6                          # Chirp bandwidth in Hz (wider → finer range resolution)
num_chirps = 1                     # Chirps per buffer (more → finer velocity resolution)
duty_cycle = 1.0                     # TX duty cycle (1.0 = continuous wave / FMCW)
source_freq_ghz = 10.4              # Point-source frequency for calibration (GHz)
buffer_size = N_rx
mode = "FMCW"                        # Waveform mode: "FMCW", "Pulsed", or "CW"
IQ_SAVE_DIR = r"D:\Stingray"            # Base directory for saving/loading IQ data
taper_type = "uniform"                   # Array taper: "uniform", "hamming", "hanning", "blackman", "taylor", or "chebyshev"
fft_window = "none"                      # FFT window: "none", "hamming", "hanning", "blackman", or "kaiser"

set_fft_window(fft_window)

# Subarray and channel configuration
# The Stingray board has 4 subarrays of 8 antenna elements each (32 total).
# The elements are arranged in a 4-row × 8-column planar grid:
#
#   Rows 1-2 (upper): Subarrays 1 & 2  (receive)
#   Rows 3-4 (lower): Subarrays 3 & 4  (subarray 3 = transmit, 4 = receive)
#
# With 3 RX subarrays we can form:
#   • Azimuth monopulse   : Σ_az = sub1 + sub4,  Δ_az = sub1 − sub4
#   • Elevation monopulse : Σ_el = sub1 + sub2,  Δ_el = sub2 − sub1
#   • Full coherent sum   : sub1 + sub2 + sub4  (maximum SNR for RD map)
#
subarray_modes = {1: "rx", 2: "rx", 3: "tx", 4: "rx"}
channel_modes  = {1: "rx", 2: "rx", 3: "tx", 4: "rx"}

# =============================================================================
# Shared state — replaces scattered 'global' variables
# =============================================================================
# cal  — calibration results (antenna offsets, per-element phase/gain corrections)
# hw   — live hardware handles (ADC, beamformer, timing engine)

cal = {
    "cal_ant_fix":   None,      # Antenna path-length correction (list of 4 floats)
    "loFreq":        14.9e9,    # LO frequency (Hz), updated by calibration
    "rx_phase_cal":  None,      # Per-element RX phase correction dict
    "tx_phase_cal":  None,      # Per-element TX phase correction dict
    "rx_gain_dict":  None,      # Per-element RX gain correction dict
    "rx_atten_dict": None,      # Per-element RX attenuation correction dict
    "tx_gain_dict":  None,      # Per-element TX gain correction dict
    "tx_atten_dict": None,      # Per-element TX attenuation correction dict
}

hw = {
    "tddn":     None,           # TDDN timing engine (TX/RX switching)
    "conv":     None,           # AD9081 MxFE (ADC + DAC)
    "PRF":      None,           # Pulse Repetition Frequency (Hz)
    "sray":     None,           # ADAR1000 beamformer array ("Stingray")
    "subarray": None,           # Subarray-to-element mapping dict
}


# =============================================================================
# Calibration functions
# =============================================================================

def run_rx_calibration():
    """Run RX-only calibration: measures per-element phase & gain offsets."""
    print("\nRunning Rx Calibration Only\n")
    url, sray, conv, tddn, subarray, subarray_ref, subarray_targ, adc_map, adc_ref = setup(N_rx)
    enable_stingray_channel(sray, subarray)
    disable_stingray_channel(sray, subarray)

    cal["cal_ant_fix"], cal["loFreq"], cal["rx_phase_cal"], cal["rx_gain_dict"], cal["rx_atten_dict"] = \
        rx_cal_full(url, sray, conv, subarray, subarray_ref, subarray_targ,
                    adc_map, adc_ref, source_freq_ghz, subarray_modes)
    cal["tx_phase_cal"]  = None
    cal["tx_gain_dict"]  = None
    cal["tx_atten_dict"] = None

    print("\n--> Rx Calibration is complete - Disable the RF Source and remove from the setup\n")
    input("--> Press Enter once the RF Source is removed\n")

    save_calibration(
        cal["cal_ant_fix"], cal["loFreq"], cal["rx_phase_cal"], cal["tx_phase_cal"], sray,
        rx_gain_dict=cal["rx_gain_dict"], rx_atten_dict=cal["rx_atten_dict"],
        tx_gain_dict=cal["tx_gain_dict"], tx_atten_dict=cal["tx_atten_dict"],
    )


def run_full_calibration():
    """Run full RX + TX calibration sequence."""
    print("\nRunning Full Calibration\n")
    url, sray, conv, tddn, subarray, subarray_ref, subarray_targ, adc_map, adc_ref = setup(N_rx)
    disable_stingray_channel(sray, subarray)

    cal["cal_ant_fix"], cal["loFreq"], cal["rx_phase_cal"], cal["rx_gain_dict"], cal["rx_atten_dict"] = \
        rx_cal_full(url, sray, conv, subarray, subarray_ref, subarray_targ,
                    adc_map, adc_ref, source_freq_ghz, subarray_modes)

    print("\n" + "=" * 60)
    print("--> RX Calibration Complete")
    print("--> Turn OFF the point source / RF signal generator")
    print("--> Connect the antenna to J9 on the Stingray board for TX cal")
    print("=" * 60)
    input("--> Press Enter once ready to proceed with TX Calibration\n")

    cal["tx_phase_cal"], cal["tx_gain_dict"], cal["tx_atten_dict"] = \
        tx_cal_full(url, sray, conv, subarray, source_freq_ghz, subarray_modes)

    save_calibration(
        cal["cal_ant_fix"], cal["loFreq"], cal["rx_phase_cal"], cal["tx_phase_cal"], sray,
        rx_gain_dict=cal["rx_gain_dict"], rx_atten_dict=cal["rx_atten_dict"],
        tx_gain_dict=cal["tx_gain_dict"], tx_atten_dict=cal["tx_atten_dict"],
    )


def load_default_cal_settings():
    """Load safe defaults when no calibration is available."""
    print("\nNo Calibration Executed - Loading Default Settings\n")
    cal["cal_ant_fix"]   = [0, 0, 0, 0]
    cal["loFreq"]        = 14.9e9
    cal["rx_phase_cal"]  = None
    cal["tx_phase_cal"]  = None
    cal["rx_gain_dict"]  = None
    cal["rx_atten_dict"] = None
    cal["tx_gain_dict"]  = None
    cal["tx_atten_dict"] = None


def load_saved_calibration():
    """Load the most recent calibration from the cal files/ directory.

    Values are stored but not applied to hardware yet — that happens
    in connect_and_configure_xbdp() after the beamformer is initialised.
    """
    print("\nLoading Latest Saved Calibration\n")
    cal_data = load_latest_calibration()

    if cal_data is None:
        print("No saved calibration found. Using default settings.")
        load_default_cal_settings()
        return

    (cal["cal_ant_fix"], cal["loFreq"], cal["rx_phase_cal"], cal["tx_phase_cal"],
     cal["rx_gain_dict"], cal["rx_atten_dict"],
     cal["tx_gain_dict"], cal["tx_atten_dict"]) = apply_calibration(cal_data, None)

    print("Note: Calibration values loaded. They will be applied to hardware after connection.\n")


# =============================================================================
# Hardware connection
# =============================================================================

def connect_and_configure_xbdp():
    """Connect to the XBDP hardware and apply calibration.

    Connection order matters:
      1. TDDN (timing / TDD engine)   — controls TX/RX switching timing
      2. AD9081 MxFE (ADC + DAC)      — digitises received echoes, generates TX waveform
      3. XUD1A (up/down converter)    — shifts IF ↔ X-band RF
      4. ADAR1000 (beamformer array)  — sets per-element gain/phase for beam steering

    PRF is derived from the ADC sample rate and buffer geometry:
        PRF = fs / (buffer_size / num_chirps)
    This is the rate at which chirps repeat, and it sets the max unambiguous
    velocity:  v_max = λ / (4 · PRI)  where PRI = 1/PRF.
    """
    hw["tddn"] = adi.tddn(uri)

    rx_adc_channels, _ = build_rx_channel_config(subarray_modes)
    hw["conv"] = setup_ad9081(uri, rx_enabled_channels=rx_adc_channels)
    print(f"\t --> RX enabled ADC channels: {rx_adc_channels} (disabled TX-only channels)")

    setup_xud1a(hw["conv"], channel_modes, cal["loFreq"])

    hw["PRF"] = hw["conv"].rx_sample_rate / (buffer_size / num_chirps)
    print("\t --> PRF in KHz", hw["PRF"])

    hw["sray"], hw["subarray"] = setup_stingray(uri, subarray_modes)
    hw["sray"].frequency = source_freq_ghz * 1e9
    enable_stingray_channel(hw["sray"], hw["subarray"])

    # Apply TX defaults if no TX calibration data
    if cal["tx_gain_dict"] is None and cal["tx_phase_cal"] is None:
        print("\n--> No TX calibration found — applying TX defaults: gain=127, phase=0, attenuator=OFF for all TX elements")
        for element in hw["sray"].elements.values():
            element.tx_gain = 127
            element.tx_phase = 0
            element.tx_attenuator = False  # Ensure binary attenuator is OFF (~23 dB loss if left engaged)
        hw["sray"].latch_tx_settings()
        print("--> TX defaults applied\n")

    # Apply RX defaults if no RX calibration data
    if cal["rx_gain_dict"] is None and cal["rx_phase_cal"] is None:
        print("\n--> No RX calibration found — applying RX defaults: gain=127, phase=0, attenuator=OFF for all RX elements")
        for element in hw["sray"].elements.values():
            element.rx_gain = 127
            element.rx_phase = 0
            element.rx_attenuator = False  # Ensure binary attenuator is OFF
        hw["sray"].latch_rx_settings()
        print("--> RX defaults applied\n")

    # Apply array taper (amplitude window) over the active RX aperture
    if taper_type != "uniform":
        hw["sray"].apply_taper(taper_type, scale_existing=True, subarray_modes=subarray_modes)
        print(f"--> Applied '{taper_type}' taper to RX elements\n")


# =============================================================================
# Pilot launcher functions
# =============================================================================

def run_fmcw_range():
    """Launch the FMCW Range pilot — shows a 1-D range profile."""
    print("\nRunning the FMCW Range Pilot Demo.....\n")
    iq = sys_sync(hw["conv"], hw["tddn"], hw["PRF"], num_chirps, BW, duty_cycle, mode, subarray_modes)
    hw["tddn"].sync_soft = 1
    FMCWRange(hw["conv"], hw["tddn"], cal["cal_ant_fix"], subarray_modes, iq, BW, hw["PRF"], duty_cycle)
    sync_disable(hw["conv"], hw["tddn"], hw["sray"], hw["subarray"])


def run_fmcw_scan():
    """Launch the FMCW Scan pilot — azimuth sweep with range heatmap."""
    print("\nRunning the FMCW Scan Pilot Demo.....\n")
    scan_min = -25
    scan_max = 25
    scan_step = 5

    clutter_choice = input("Enable clutter mesh? (Y/n): ").strip().lower()
    cluttermesh = clutter_choice not in ("n", "no")

    iq = sys_sync(hw["conv"], hw["tddn"], hw["PRF"], num_chirps, BW, duty_cycle, mode, subarray_modes)
    hw["tddn"].sync_soft = 1
    FMCWScan(hw["conv"], hw["sray"], cal["cal_ant_fix"], subarray_modes, iq, BW, hw["PRF"],
             scan_min, scan_max, scan_step, cal["rx_phase_cal"], cal["tx_phase_cal"], cluttermesh=cluttermesh)
    sync_disable(hw["conv"], hw["tddn"], hw["sray"], hw["subarray"])


# =============================================================================
# Menus
# =============================================================================

calibration_options = {
    "1": run_rx_calibration,
    "2": run_full_calibration,
    "3": load_saved_calibration,
    "4": load_default_cal_settings,
}

pilot_options = {
    "1": run_fmcw_range,
    "2": run_fmcw_scan,
}


def calibration_menu():
    while True:
        print("\n=== XBDP Calibration Menu ===")
        print("1. Rx Calibration")
        print("2. Full (Rx & Tx) Calibration")
        print("3. Load saved calibration")
        print("4. Load default calibration")
        print("q. Quit\n")

        choice = input("Choose an option: ").strip().lower()

        if choice == "q":
            print("Exiting...")
            return

        action = calibration_options.get(choice)
        if action:
            action()
            break
        else:
            print("Invalid choice. Please try again.\n")


def pilot_menu_loop():
    while True:
        print("\n=== XBDP Pilot Demo Menu ===")
        print("1. Run the FMCW Range Pilot")
        print("2. Run the FMCW Scan Pilot")
        print("q. Quit\n")

        choice = input("Choose an option: ").strip().lower()

        if choice == "q":
            print("Exiting...")
            break

        action = pilot_options.get(choice)
        if action:
            action()
            break
        else:
            print("Invalid choice. Please try again.\n")


if __name__ == "__main__":
    purge_stale_calibrations()
    calibration_menu()
    connect_and_configure_xbdp()
    pilot_menu_loop()