# Copyright (C) 2025 Analog Devices, Inc.
#
# SPDX short identifier: ADIBSD

# =============================================================================
# tx_rx_cal.py — CW-mode hardware initialisation and calibration orchestration
# =============================================================================
# This module provides two high-level calibration entry points:
#
#   rx_cal_full()  — RX signal-chain calibration
#   tx_cal_full()  — TX signal-chain calibration
#
# Both routines set the system into continuous-wave (CW) mode, where the TDDN
# timing engine is configured to hold the TX/RX paths permanently active
# (polarity-inverted channels stay LOW, all others stay HIGH with
# frame_length = 1 ms, no off-time).  CW mode gives the calibration
# algorithms a steady-state signal to measure against, unlike the pulsed FMCW
# waveform used during normal operation.
#
# Calibration workflow (RX path):
#   1.  setup()          → instantiate AD9081, ADAR1000, TDDN in CW mode
#   2.  updown_init()    → switch XUD1A to RX path
#   3.  setup_adf4371()  → tune ADF4371 PLL to LO = f_source + NCO + 4 GHz
#   4.  rx_gain()        → equalise per-element amplitude via gain codes
#   5.  find_phase_delay_fixed_ref() → align ADC-channel phases to a ref
#   6.  phase_analog()   → fine-align element phases within each subarray
#
# TX calibration follows a similar flow but measures output power with an
# external LTC2314-14 ADC (envelope detector) instead of the AD9081 RX ADC.
#
# Hardware addressed:
#   AD9081   – MxFE converter (ADC + DAC + NCO)
#   XUD1A    – up/down converter (RX/TX T/R switch, ADF4371 PLL, LNA bias)
#   ADAR1000 – 4-channel analogue beamformer ICs (8 ICs, 32 elements)
#   TDDN     – FPGA-based timing engine
#   LTC2314  – 14-bit ADC on the TX envelope detector (TX cal only)
# =============================================================================
from .calibration import *
import adi
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from custom_libs.adar1000 import adar1000_array
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

def setup(N_rx):
    """
    Master hardware bootstrap for the X-Band Stingray development kit.

    Creates device handles (AD9081, ADAR1000 array, TDDN), puts the TDDN into
    CW mode, and returns the physical array topology needed by the calibration
    routines.

    Parameters
    ----------
    N_rx : int
        ADC buffer size in complex samples (e.g. 2**20 = 1 048 576).

    Returns
    -------
    url, sray, conv, tddn,
    subarray, subarray_ref, subarray_targ, adc_map, adc_ref
    """
    url = "ip:192.168.0.101"

    # ── Physical array topology ──────────────────────────────────────────
    # The 4×8 element grid is partitioned into four 8-element subarrays.
    # Each row lists the 1-based element IDs for that subarray. When facing the array:
    #
    #   Subarray 1  (top-right):    elements  1, 2, 5, 6, 9,10,13,14
    #   Subarray 2  (bottom-right):   elements  3, 4, 7, 8,11,12,15,16
    #   Subarray 3  (bottom-left):elements 19,20,23,24,27,28,31,32
    #   Subarray 4  (top-left): elements 17,18,21,22,25,26,29,30
    #
    # In the default monopulse config: sub 1,2,4 = RX  |  sub 3 = TX.
    subarray = np.array([
        [1, 2, 5, 6, 9, 10, 13, 14],  # subarray 1
        [3, 4, 7, 8, 11, 12, 15, 16],  # subarray 2
        [19, 20, 23, 24, 27, 28, 31, 32],  # subarray 3
        [17, 18, 21, 22, 25, 26, 29, 30]   # subarray 4
    ])
    # Reference elements — one per subarray.  During phase calibration every
    # other element in the subarray is swept while the reference stays fixed.
    # The four references also serve as the inter-subarray alignment anchors.
    subarray_ref = np.array([2, 4, 18, 20])  # subarray reference elements

    # ADC channel wiring.  The AD9081 has 4 RX ADC channels (0-3) that map
    # to the four subarrays in a non-sequential order dictated by the PCB
    # traces:  subarray 1→ADC 3, subarray 2→ADC 1, subarray 3→ADC 0,
    #          subarray 4→ADC 2.
    adc_map      = np.array([3, 1, 0, 2])  # ADC channel per subarray row
    adc_ref      = 3  # ADC reference channel (subarray 1's ADC channel)

    # Setup Stingray for RX mode
    d = ~np.isin(subarray, subarray_ref)
    subarray_targ = subarray[d] # analog target channels
    subarray_targ = np.reshape(subarray_targ, (subarray.shape[0],-1)) # matrix of subarray target channels to enable/disable wrt reference
    
    sray, conv, tddn = device_init(N_rx, url, default=True) # Initialize the device with default settings
    
    return url, sray, conv, tddn, subarray, subarray_ref, subarray_targ, adc_map, adc_ref

def device_init(N_rx, url="ip:192.168.0.101", default=True):
    """
    Instantiate and configure the three core peripherals:
      1. AD9081 (conv)  — MxFE ADC/DAC + digital NCOs
      2. ADAR1000 (sray) — 8-chip beamformer array (32 elements)
      3. TDDN (tddn)     — FPGA timing engine (set to CW mode here)

    When *default=True* the AD9081 NCO frequencies are explicitly set to
    known-good defaults:
      - RX main NCO  = 500 MHz  (shifts the IF into baseband)
      - TX main NCO  = 4.5 GHz  (sets the DAC carrier)
      - Channel NCOs = 0 Hz     (no fine offset)

    The TDDN is programmed for CW mode: all gating channels are held
    permanently active so the calibration tones are continuous.

    Parameters
    ----------
    N_rx : int
        ADC buffer size (complex samples).
    url : str
        IIO context URI for the ZCU102 FPGA.
    default : bool
        If True, apply safe-default NCO / gain / buffer settings.

    Returns
    -------
    sray, conv, tddn
    """
    if default:
        # Setup AD9081 RX, TDDN Engine & ADAR1000
        print("")
        print("--> Connecting to", url, "...")
        print("")
        conv = adi.ad9081(uri = url)
        tddn = adi.tddn(uri = url)
        sray = adar1000_array(
            uri = url,
            chip_ids = ["adar1000_csb_1_1", "adar1000_csb_1_2", "adar1000_csb_1_3", "adar1000_csb_1_4",
                        "adar1000_csb_2_1", "adar1000_csb_2_2", "adar1000_csb_2_3", "adar1000_csb_2_4"],
            device_map = [[1, 3, 5, 7], [2, 4, 6, 8]],
            element_map = np.array([[1, 5, 9, 13, 17, 21, 25, 29],
                        [2, 6, 10, 14, 18, 22, 26, 30],
                        [3, 7, 11, 15, 19, 23, 27, 31],
                        [4, 8, 12, 16, 20, 24, 28, 32]]),
            device_element_map = {
                1: [2, 6, 5, 1],
                2: [4, 8, 7, 3],
                3: [10, 14, 13, 9],
                4: [12, 16, 15, 11],
                5: [18, 22, 21, 17],
                6: [20, 24, 23, 19],
                7: [26, 30, 29, 25],
                8: [28, 32, 31, 27],
            },
        )
        for device in sray.devices.values():
            device.mode = "rx"
        print("--> Setting up chip")
        conv._rxadc.set_kernel_buffers_count(1)
        conv._txdac.set_kernel_buffers_count(1)
        conv._ctx.set_timeout(0)

        if conv.rx_main_nco_frequencies !=[100000000] * 4:
            conv.rx_channel_nco_frequencies = [0] * 4
            conv.tx_channel_nco_frequencies = [0] * 4
            conv.rx_main_nco_frequencies      = [100000000] * 4
            conv.tx_main_nco_frequencies      = [100000000] * 4
            conv.tx_channel_nco_gain_scales   = [1.0] * 4
            conv.tx_channel_nco_phases = [0] * 4
            conv.tx_main_nco_phases    = [0] * 4
            conv.rx_channel_nco_phases = [0] * 4
            conv.rx_main_nco_phases    = [0] * 4
            conv.rx_nyquist_zone = ["odd"] * 4
            conv.rx_main_6dB_digital_gains = [0] * 4

        conv.rx_channel_nco_frequencies = [0] * 4
        conv.tx_channel_nco_frequencies = [0] * 4
        # RX main NCO = 500 MHz: digitally shifts the IF signal into baseband
        # after the XUD1A mixer brings RF down to ~500 MHz.
        conv.rx_main_nco_frequencies = [int(500e6)] * 4
        # TX main NCO = 4.5 GHz: the DAC outputs a CW carrier at 4.5 GHz
        # which the XUD1A upconverts to X-band (~10.4 GHz).
        conv.tx_main_nco_frequencies = [int(4.5e9)] * 4
        conv.rx_enabled_channels = [0,1,2,3]
        conv.tx_enabled_channels = [0,1,2,3]
        conv.rx_nyquist_zone = ["odd"] * 4
        conv.rx_buffer_size = int(N_rx)
        conv.tx_cyclic_buffer = True
        conv.rx_cyclic_buffer = False
        conv.tx_ddr_offload   = False
        # Startup and connect TDDN
        tddn.enable = False
        tddn.startup_delay_ms = 0
        # Configure top level engine
        frame_length_ms = 1
        tddn.frame_length_ms = frame_length_ms
        # Configure component channels
        off_time = frame_length_ms - 0.1
        # Setup TDDN Channel for CW mode
        tddn_channels = {
            "TX_OFFLOAD_SYNC": 0,
            "RX_OFFLOAD_SYNC": 1,
            "TDD_ENABLE": 2,
            "RX_MXFE_EN": 3,
            "TX_MXFE_EN": 4,
            "TX_STINGRAY_EN": 5
        }
        # Assign channel properties for CW (continuous-wave) mode.
        # In CW mode all RF paths are permanently active:
        #   ch 0,1 (offload syncs): polarity=True  → always HIGH
        #   ch 2,5 (TDD_EN, TX_EN): polarity=False → inverted → active-low
        #   ch 3,4 (RX/TX MXFE):   polarity=True, off_raw=10 → essentially CW
        for key, value in tddn_channels.items():
            if value == 0 or value == 1:
                tddn.channel[value].on_raw = 0
                tddn.channel[value].off_raw = 0
                tddn.channel[value].on_ms = 0
                tddn.channel[value].off_ms = 0
                tddn.channel[value].polarity = True
                tddn.channel[value].enable = True
            elif value == 2 or value == 5:
                tddn.channel[value].on_raw = 0
                tddn.channel[value].off_raw = 0
                tddn.channel[value].on_ms = 0
                tddn.channel[value].off_ms = 0
                tddn.channel[value].polarity = False
                tddn.channel[value].enable = True
            else:
                tddn.channel[value].on_raw = 0
                tddn.channel[value].off_raw = 10
                tddn.channel[value].polarity = True
                tddn.channel[value].enable = True
        tddn.enable = True # Fire up TDD engine
        tddn.sync_internal = True # software enable TDD mode
    else:
        # Init AD9081 RX, TDDN Engine & ADAR1000
        conv = adi.ad9081(uri = url)
        tddn = adi.tddn(uri = url)
        sray = adar1000_array(
            uri = url,
            chip_ids = ["adar1000_csb_1_1", "adar1000_csb_1_2", "adar1000_csb_1_3", "adar1000_csb_1_4",
                        "adar1000_csb_2_1", "adar1000_csb_2_2", "adar1000_csb_2_3", "adar1000_csb_2_4"],
            device_map = [[1, 3, 5, 7], [2, 4, 6, 8]],
            element_map = np.array([[1, 5, 9, 13, 17, 21, 25, 29],
                        [2, 6, 10, 14, 18, 22, 26, 30],
                        [3, 7, 11, 15, 19, 23, 27, 31],
                        [4, 8, 12, 16, 20, 24, 28, 32]]),
            device_element_map = {
                1: [2, 6, 5, 1],
                2: [4, 8, 7, 3],
                3: [10, 14, 13, 9],
                4: [12, 16, 15, 11],
                5: [18, 22, 21, 17],
                6: [20, 24, 23, 19],
                7: [26, 30, 29, 25],
                8: [28, 32, 31, 27],
            },
        )
        # Startup and connect TDDN (non-default path — same CW config)
        tddn.enable = False
        tddn.startup_delay_ms = 0
        frame_length_ms = 1
        tddn.frame_length_ms = frame_length_ms
        off_time = frame_length_ms - 0.1
        # TDDN channels for CW mode (see default path above for semantics)
        tddn_channels = {
            "TX_OFFLOAD_SYNC": 0,
            "RX_OFFLOAD_SYNC": 1,
            "TDD_ENABLE": 2,
            "RX_MXFE_EN": 3,
            "TX_MXFE_EN": 4,
            "TX_STINGRAY_EN": 5
        }
        for key, value in tddn_channels.items():
            if value == 0 or value == 1:
                tddn.channel[value].on_raw = 0
                tddn.channel[value].off_raw = 0
                tddn.channel[value].on_ms = 0
                tddn.channel[value].off_ms = 0
                tddn.channel[value].polarity = True
                tddn.channel[value].enable = True
            elif value == 2 or value == 5:
                tddn.channel[value].on_raw = 0
                tddn.channel[value].off_raw = 0
                tddn.channel[value].on_ms = 0
                tddn.channel[value].off_ms = 0
                tddn.channel[value].polarity = False
                tddn.channel[value].enable = True
            else:
                tddn.channel[value].on_raw = 0
                tddn.channel[value].off_raw = 10
                tddn.channel[value].polarity = True
                tddn.channel[value].enable = True
        tddn.enable = True # Fire up TDD engine
        tddn.sync_internal = True # software enable TDD mode
    return sray, conv, tddn
    

def updown_init(ctx, mode):
    """
    Switch the XUD1A up/down converter board between RX and TX paths.

    The XUD1A has four T/R switches (one per subarray RF front-end).  Writing
    0 to a voltage channel selects the RX (down-convert) path; writing 1
    selects the TX (up-convert) path.

    Also sets:
      - PLL select = 1  (choose ADF4371 output band for X-band LO)
      - RX gain mode = 1  (high-gain LNA setting)

    Parameters
    ----------
    ctx : iio.Context
        The raw IIO context (conv._ctrl.ctx).
    mode : str or int
        'rx'/0 for receive path, 'tx'/1 for transmit path.
    """
    if isinstance(mode, str):
        if mode.lower() == "rx":
            mode = str(0)
        elif mode.lower() == "tx":
            mode = str(1)
        elif mode.lower() not in ["rx", "tx"]:
            raise ValueError("mode must be 'rx' or 'tx'")
    elif isinstance(mode, int):
        mode = str(mode)
        if mode not in [0, 1]:
            raise ValueError("mode must be 0 or 1")
    else:
        raise ValueError("mode must be 'rx' or 0 for RX, 'tx' or 1 for TX")
    
    # Find the XUD control device and the TX/RX channels
    xud = ctx.find_device("xud_control")
    txrx1 = xud.find_channel("voltage1", True)
    txrx2 = xud.find_channel("voltage2", True)
    txrx3 = xud.find_channel("voltage3", True)
    txrx4 = xud.find_channel("voltage4", True)

    # 0 for rx, 1 for tx
    txrx1.attrs["raw"].value = mode # Subarray 4
    txrx2.attrs["raw"].value = mode # Subarray 3
    txrx3.attrs["raw"].value = mode # Subarray 1
    txrx4.attrs["raw"].value = mode # Subarray 2

    # Find the PLL select and RX gain mode channels
    PLLselect = xud.find_channel("voltage5", True)
    rxgainmode = xud.find_channel("voltage0", True)

    # Set the PLL select and RX gain mode
    PLLselect.attrs["raw"].value = "1"
    rxgainmode.attrs["raw"].value = "1"

def setup_adf4371(ctx, adc, source_freq_ghz):
    """
    Programme the ADF4371 PLL to generate the X-band LO.

    The receive down-conversion chain is:

        RF in  ──►  XUD1A mixer  ──►  AD9081 ADC
                        │
                     LO from ADF4371

    The LO must be set so the desired RF signal lands inside the ADC's
    first Nyquist zone after the RX main NCO digital shift:

        LO = f_source + RX_main_NCO + 4.0 GHz

    For the default config (f_source = 10.4 GHz, NCO = 500 MHz):
        LO = 10.4e9 + 0.5e9 + 4.0e9 = 14.9 GHz

    The 4.0 GHz offset accounts for the XUD1A's internal IF frequency plan.

    Parameters
    ----------
    ctx : iio.Context
        Raw IIO context.
    adc : adi.ad9081
        Converter object (used to read RX main NCO frequency).
    source_freq_ghz : float
        Desired RF centre frequency in GHz (e.g. 10.4).

    Returns
    -------
    loFreq : float
        Programmed LO frequency in Hz.
    """
    adf4371 = ctx.find_device("adf4371-0")
    # LO = f_source + RX_main_NCO + 4 GHz  (XUD1A IF offset)
    rx_lo = str(int(((float(source_freq_ghz) * 1e9) + adc.rx_main_nco_frequencies[0]) + 4.0e9))
    loFreq = (float(source_freq_ghz) * 1e9) + adc.rx_main_nco_frequencies[0]+4.0e9
    print("\t --> ",loFreq)
    print("\t --> ",rx_lo)
    # Find channel attribute for TX & RX and set frequency and powerdown
    XUDLO = adf4371.find_channel("altvoltage2", True)
    XUDLO.attrs["frequency"].value = str(rx_lo)
    XUDLO.attrs["powerdown"].value = "0"
    return loFreq
def rx_cal_full(url, sray, conv, subarray, subarray_ref, subarray_targ,
                adc_map, adc_ref, source_freq_ghz, subarray_modes=None):
    """
    Full RX signal-chain calibration.

    Requires an external CW RF source at *source_freq_ghz*.  The routine:

      1. Switches the XUD1A to RX mode and tunes the ADF4371 LO.
      2. Sets all ADAR1000 elements to max RX gain (code 127) and 0° phase.
      3. **Gain equalisation** — measures per-element amplitude and writes
         gain codes so all elements contribute equally to the sum beam.
      4. **ADC-channel phase alignment** — sweeps a reference element's
         phase to find the offset that aligns all four ADC channels.
      5. **Analog phase calibration** — sweeps each non-reference element
         within a subarray to null out manufacturing phase errors.
      6. Captures a post-cal IQ snapshot and saves it to ``rx_cal_data.csv``.

    Parameters
    ----------
    url : str
        IIO URI.
    sray : adar1000_array
        Stingray beamformer handle.
    conv : adi.ad9081
        MxFE converter handle.
    subarray, subarray_ref, subarray_targ : np.ndarray
        Physical array topology from setup().
    adc_map : np.ndarray
        ADC channel index per subarray row.
    adc_ref : int
        ADC channel used as the inter-subarray reference.
    source_freq_ghz : float
        CW source frequency in GHz.
    subarray_modes : dict or None
        If provided, only subarrays marked 'rx' are calibrated.

    Returns
    -------
    cal_ant_fix : np.ndarray
        Per-ADC-channel digital phase corrections (complex phasors).
    loFreq : float
        Programmed LO frequency in Hz.
    analog_phase_cal : dict
        Per-element analog phase corrections.
    gain_dict, atten_dict : dict
        Per-element gain code and attenuator settings.
    """
    print("")
    print("--> Turn on RF Source...")
    print("")
    input('--> Press Enter to continue...')
    print("")
    # Set up the RX path for calibration
    mode = "rx" # Set to rx mode for RX cals
    # Set up AD9081 RX path for calibration
    updown_init(conv._ctrl.ctx,mode) # Set up the RX path for calibration
    loFreq = setup_adf4371(conv._ctrl.ctx,conv,source_freq_ghz) # Setup ADF4371 for RX path
    print("\t --> ARRAY_MODE = RX. Setting all devices to rx mode")
    SELF_BIASED_LNAs = True # Set to True if using self-biased LNAs
    # Set up the RX path for calibration
    for device in sray.devices.values():
        device.mode = "rx" # Set mode to Rx for all devices in stingray
        if SELF_BIASED_LNAs:
            # Allow the external LNAs to self-bias
            device.lna_bias_out_enable = False
        else:
            # Set the external LNA bias
            device.lna_bias_on = -0.7
    for element in sray.elements.values():
        element.rx_gain = 127 # 127: Highest gain; 0: Lowest gain
        element.rx_attenuator = 0 # 1: Attentuation on; 0: Attentuation off
        element.rx_phase = 0 # Set all phases to 0
    sray.latch_rx_settings() # Latch SPI settings to devices

    # Determine which subarray rows to calibrate based on subarray_modes.
    # subarray_modes keys are 1-based (1=subarray1 ... 4=subarray4).
    # find_phase_delay_fixed_ref always uses the full subarray_ref/adc_ref to align all 4 ADC channels.
    if subarray_modes is not None:
        rx_indices = sorted([k - 1 for k, v in subarray_modes.items()
                             if isinstance(v, str) and v.lower() == "rx"])
        print(f"\t --> Calibrating only RX subarrays: {[i + 1 for i in rx_indices]}")
    else:
        rx_indices = list(range(len(subarray)))

    cal_subarray = subarray[rx_indices, :]
    cal_adc_map = adc_map[rx_indices]

    # Match each selected subarray row to its reference element by value, not by index.
    # subarray_ref ordering does not necessarily correspond to subarray row order.
    cal_subarray_ref = np.array([
        subarray_ref[np.isin(subarray_ref, row)][0]
        for row in cal_subarray
    ])
    d = ~np.isin(cal_subarray, cal_subarray_ref)
    cal_subarray_targ = cal_subarray[d]
    cal_subarray_targ = np.reshape(cal_subarray_targ, (cal_subarray.shape[0], -1))

    # sray.steer_rx(azimuth=0, elevation=0) # Broadside
    delay_phases = np.arange(-180,181,1) # sweep phase from -180 to 180 in 1 degree steps.
    # Pass only the RX subarrays so gain_codes normalises against the true minimum
    # of the active elements — TX-subarray elements are excluded entirely.
    gain_dict, atten_dict, mag_pre_cal, mag_post_cal = rx_gain(sray, conv, cal_subarray, cal_adc_map, sray.element_map)
    print_subarray_values(gain_dict, cal_subarray, "RX Gain")

    # ADC-channel alignment uses full subarray_ref so all 4 data channels are corrected.
    cal_ant_fix = find_phase_delay_fixed_ref(sray, conv, subarray_ref, adc_ref, delay_phases)
    analog_phase_cal = phase_analog(sray, conv, adc_map, adc_ref, subarray_ref, subarray_targ, cal_ant_fix)

    # Build a mapping: subarray number (1-based) → ADC channel index.
    # Used for the phase print below and for the calibration plot.
    rx_subarray_labels = {rx_indices[i] + 1: cal_adc_map[i] for i in range(len(rx_indices))}

    # Print phase differences between active RX subarrays AFTER all phase calibrations.
    print("\n\t --> Phase differences between active RX subarrays (post all phase cals):")
    enable_stingray_channel(sray, cal_subarray)
    phase_check_data = np.array(data_capture_cal(conv, cal_ant_fix))
    disable_stingray_channel(sray, cal_subarray)
    ref_subarray_num = list(rx_subarray_labels.keys())[0]
    ref_adc_ch = rx_subarray_labels[ref_subarray_num]
    ref_phase = np.angle(phase_check_data[ref_adc_ch, 100], deg=True)
    for sa_num, adc_ch in rx_subarray_labels.items():
        sa_phase = np.angle(phase_check_data[adc_ch, 100], deg=True)
        diff = (sa_phase - ref_phase + 180) % 360 - 180
        marker = " <-- reference" if sa_num == ref_subarray_num else f"  Δ = {diff:+.1f}°"
        print(f"\t     Subarray {sa_num} (ADC ch {adc_ch}): {sa_phase:.1f}°{marker}")
    print("")

    # Enable only the designated RX subarray elements for the verification capture
    enable_stingray_channel(sray, cal_subarray)
    rx_cal_data = np.transpose(np.array(data_capture_cal(conv, cal_ant_fix)))  # Post-process data with gain and phase calibration
    disable_stingray_channel(sray, cal_subarray)

    # Build plot lines only for active RX subarrays, using the correct ADC channel per subarray.
    # adc_map[row_index] gives the ADC channel for that subarray row.
    colors = ["red", "blue", "green", "orange", "purple"]
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.xlim(0, 4095)
    for i, (sa_num, adc_ch) in enumerate(rx_subarray_labels.items()):
        ax.plot(rx_cal_data[:, adc_ch].real, colors[i % len(colors)],
                label=f"RX Cal Data Subarray {sa_num} (ADC ch {adc_ch})")
    ax.legend(loc="upper right")

    # Set titles and labels
    ax.set_title('RX Calibration Data')
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.grid(visible=True)

    
    # Create a small axes for the button: [left, bottom, width, height] in figure coords (0–1)
    btn_ax = fig.add_axes([0.87, 0.02, 0.08, 0.06])  # adjust to taste
    close_btn = Button(btn_ax, "Close", color="#eeeeee", hovercolor="#dddddd")

    def on_close_clicked(event):
        # Close just this figure
        plt.close(fig)
        # Alternatively: fig.canvas.manager.close()  # also closes the window

    close_btn.on_clicked(on_close_clicked)


    # Display the plot
    plt.show()
    return cal_ant_fix, loFreq, analog_phase_cal, gain_dict, atten_dict
def tx_cal_full(url, sray, conv, subarray, source_freq_ghz, subarray_modes=None):
    """
    Full TX signal-chain calibration.

    Uses the LTC2314-14 envelope detector ADC to measure radiated power.
    The routine:

      1. Switches XUD1A to TX mode and tunes the ADF4371.
      2. Sends a DC baseband tone (all ones × 2^14) through the AD9081 DAC.
         The TX main NCO at 4.5 GHz upconverts this to a CW carrier, then
         the XUD1A mixer shifts it to X-band.
      3. **Pre-cal power snapshot** — records array power before any cal.
      4. **Phase calibration** — sweeps each element's TX phase and finds
         the setting that maximises the LTC detector reading (constructive
         interference → coherent sum).
      5. **Post-phase-cal power check** — ideally +9 dB for 8 coherent
         elements (20·log10(8) ≈ 9.03 dB).
      6. **Gain calibration** — equalises per-element amplitude for
         sidelobe control.

    Parameters
    ----------
    url : str
        IIO URI.
    sray : adar1000_array
        Stingray beamformer handle.
    conv : adi.ad9081
        MxFE converter handle.
    subarray : np.ndarray
        Physical array topology.
    source_freq_ghz : float
        CW source frequency in GHz.
    subarray_modes : dict or None
        If provided, only subarrays marked 'tx' are calibrated.

    Returns
    -------
    tx_phase_cal : dict
        Per-element TX phase corrections.
    tx_gain_dict, tx_atten_dict : dict
        Per-element TX gain code and attenuator settings.
    """
    print("ARRAY_MODE = TX. Setting all devices to tx mode")
    mode  = "tx" # Set to tx mode for TX cals
    updown_init(conv._ctrl.ctx, mode) # Set up the TX path for calibration
    setup_adf4371(conv._ctrl.ctx,conv, source_freq_ghz) # Setup ADF4371 for TX path
    for device in sray.devices.values():
        device.mode = "tx"
        # Enable the Tx path for each channel and set the external PA bias
        for channel in device.channels:
            channel.pa_bias_on = -1.06
    for element in sray.elements.values():
        element.tx_gain = 127 # 127: Highest gain; 0: Lowest gain
        element.tx_phase = 0 # Set all phases to 0
        element.tx_attenuator = 0 # 1: Attentuation off; 0: Attentuation on 
    sray.latch_tx_settings() # Latch SPI settings to devices

    # Filter to only calibrate the subarrays designated as TX in subarray_modes.
    if subarray_modes is not None:
        tx_indices = sorted([k - 1 for k, v in subarray_modes.items()
                             if isinstance(v, str) and v.lower() == "tx"])
        print(f"\t --> Calibrating only TX subarrays: {[i + 1 for i in tx_indices]}")
    else:
        tx_indices = list(range(len(subarray)))
    cal_subarray_tx = subarray[tx_indices, :]

    N_rx = conv.rx_buffer_size # number of samples to capture
    fs = conv.tx_sample_rate # sample rate of the TX path

    # TX Main NCO is at 4.5 GHz; ADF4371 upconverts that to 10 GHz.
    # A DC baseband signal (all ones) gets shifted cleanly to 4.5 GHz by the NCO
    # and then to 10 GHz by the mixer — no aliasing.
    # (Using cos(2*pi*10GHz*t) aliases badly since 10 GHz >> Nyquist of the DAC.)
    # AD9081 uses Q1.15 fixed-point NCO internally: 2^14 is the conventional full-scale
    # input value. Values above this saturate silently at 2^14 after the NCO multiply.
    tx_sig = np.ones(N_rx) * 2 ** 14

    conv.tx([tx_sig, tx_sig, tx_sig, tx_sig]) # Send the signal to the TX path

    ltc = adi.ltc2314_14(uri=url) # Connect to the LTC2314-14 ADC

    import time
    N_avg = 50  # LTC averages per power snapshot
    ref_elem = int(cal_subarray_tx[0, 0])

    # --- Step 0: Measure DC baseline ---
    # The LTC2314 power detector has a large DC offset (~50 % of ADC full
    # scale) with no RF present.  Subtracting this baseline in the linear /
    # raw-count domain isolates the actual RF signal component — without it,
    # phase-sweep variations are swamped by the constant offset.
    disable_stingray_channel(sray, cal_subarray_tx)
    for element in sray.elements.values():
        element.tx_gain = 0
    sray.latch_tx_settings()
    time.sleep(0.1)
    baseline_raw = float(np.mean([get_ltc_raw(ltc) for _ in range(N_avg)]))

    # Restore full gain
    for element in sray.elements.values():
        element.tx_gain = 127
    sray.latch_tx_settings()

    # Measure with all elements on to verify array signal above baseline
    enable_stingray_channel(sray, cal_subarray_tx)
    time.sleep(0.1)
    all_on_raw = float(np.mean([get_ltc_raw(ltc) for _ in range(N_avg)]))
    signal_raw = all_on_raw - baseline_raw

    print(f"\n\t === BASELINE DIAGNOSTIC ===")
    print(f"\t  Baseline (elements OFF):       {baseline_raw:.0f} counts")
    print(f"\t  All elements ON:               {all_on_raw:.0f} counts")
    print(f"\t  Signal above baseline:         {signal_raw:.0f} counts")
    if signal_raw > 100:
        print(f"\t  [OK] Array signal clearly above baseline.")
    else:
        print(f"\t  [WARN] Very low signal above baseline — check antenna / coupling.")
    print(f"\t ===========================\n")

    # --- Step 1: Pre-calibration power (baseline-subtracted) ---
    pre_cal_raw = float(np.mean([get_ltc_raw(ltc) for _ in range(N_avg)]))
    pre_cal_signal = pre_cal_raw - baseline_raw
    print(f"\t --> Pre-cal signal (raw - baseline): {pre_cal_signal:.0f} counts")
    disable_stingray_channel(sray, cal_subarray_tx)

    # --- Step 2: Phase calibration at full gain with baseline subtraction ---
    tx_phase_cal = tx_phase(sray, ltc, cal_subarray_tx,
                            baseline_raw=baseline_raw)

    # --- Step 3: Post-phase-cal — coherent buildup test ---
    # The LTC2314 power detector circuit is LOGARITHMIC: raw ADC counts are
    # proportional to power in dB, NOT linear amplitude or power.
    # Correct metric: count DIFFERENCES give dB differences.
    #   dB_change ≈ (count_B - count_A) / counts_per_dB
    # The scale factor counts_per_dB is estimated from the buildup curve.

    all_elems = cal_subarray_tx.flatten().astype(int)

    # Build up the array one element at a time with calibrated phases.
    # Each step adds one coherent element; the expected per-step power
    # increase is 10*log10((n+1)² / n²) = 20*log10((n+1)/n).
    print(f"\n\t === COHERENT BUILDUP TEST ===")
    print(f"\t  Enabling elements one at a time with cal phases.")
    print(f"\t  {'N':>3s}  {'Elem':>4s}  {'Signal':>8s}  {'Expected':>10s}  {'Measured':>10s}")

    disable_stingray_channel(sray, cal_subarray_tx)

    buildup_signals = []
    enabled_so_far = []
    for idx, elem_num in enumerate(all_elems):
        enable_stingray_channel(sray, np.array([int(elem_num)]))
        phase = tx_phase_cal.get(int(elem_num), 0)
        set_tx_phase(sray, int(elem_num), int(phase))
        sray.latch_tx_settings()
        time.sleep(0.03)
        enabled_so_far.append(int(elem_num))
        raw = float(np.mean([get_ltc_raw(ltc) for _ in range(N_avg)]))
        sig = raw - baseline_raw
        buildup_signals.append(sig)

        n = idx + 1
        # Expected dB above single-element: 20*log10(n)
        expected_db = 20.0 * np.log10(n) if n > 1 else 0.0
        # Measured dB above single-element (estimated from count difference)
        if buildup_signals[0] > 0 and sig > 0:
            measured_diff = sig - buildup_signals[0]  # count difference from 1-elem
        else:
            measured_diff = 0.0
        print(f"\t  {n:>3d}  {elem_num:>4d}  {sig:>8.0f}  {expected_db:>+9.1f} dB  {measured_diff:>+8.0f} cts")

    # Estimate counts_per_dB from the buildup: 1-element vs N-element
    sig_1 = buildup_signals[0]
    sig_N = buildup_signals[-1]
    n_total = len(all_elems)
    expected_total_db = 20.0 * np.log10(n_total)  # 18.06 dB for 8 elements
    if sig_N > sig_1 and expected_total_db > 0:
        counts_per_dB = (sig_N - sig_1) / expected_total_db
    else:
        counts_per_dB = 0.0

    print(f"\t  ─────────────────────────────────────────────────")
    print(f"\t  1-element signal:   {sig_1:.0f} counts")
    print(f"\t  {n_total}-element signal:   {sig_N:.0f} counts")
    print(f"\t  Count increase:     {sig_N - sig_1:.0f} counts")
    print(f"\t  Expected for coherent {n_total} elem: +{expected_total_db:.1f} dB")
    if counts_per_dB > 0:
        measured_total_db = (sig_N - sig_1) / counts_per_dB
        print(f"\t  Detector scale:     {counts_per_dB:.1f} counts/dB")
        print(f"\t  Measured gain:      +{measured_total_db:.1f} dB  (by definition)")
    else:
        print(f"\t  [WARN] Could not estimate detector scale — no signal increase.")

    # Compare pre-cal to post-cal using log-detector math
    if counts_per_dB > 0:
        improvement_db = (sig_N - pre_cal_signal) / counts_per_dB
        print(f"\n\t  Pre-cal signal:     {pre_cal_signal:.0f} counts (all at 0°)")
        print(f"\t  Post-cal signal:    {sig_N:.0f} counts (cal phases)")
        print(f"\t  Improvement:        {improvement_db:+.1f} dB")
    print(f"\t =============================\n")

    # --- Step 3b: NULL TEST — shift all cal phases by 180° (destructive) ---
    # Expected: power drops by 10*log10(36/n²) below calibrated peak.
    # For n=8 total, 7 anti-phase + 1 in-phase → net = (n-2)V anti-phase.
    # Power ratio = (n-2)²/n² → 10*log10(36/64) = -2.5 dB for n=8.
    for elem, phase in tx_phase_cal.items():
        set_tx_phase(sray, int(elem), int((phase + 180) % 360))
    sray.latch_tx_settings()
    time.sleep(0.05)
    null_raw = float(np.mean([get_ltc_raw(ltc) for _ in range(N_avg)]))
    null_signal = null_raw - baseline_raw

    n_shifted = len(tx_phase_cal)  # elements that got +180° (all except ref)
    n_unshifted = n_total - n_shifted  # reference only
    expected_null_db = 10.0 * np.log10(max((n_shifted - n_unshifted) ** 2, 1)
                                        / (n_total ** 2))

    if counts_per_dB > 0:
        null_delta_db = (null_signal - sig_N) / counts_per_dB
    else:
        null_delta_db = 0.0

    print(f"\t --> NULL test signal (cal+180°): {null_signal:.0f} counts")
    print(f"\t --> Delta from calibrated:       {null_delta_db:+.1f} dB")
    print(f"\t --> Expected drop ({n_shifted}v{n_unshifted}):      {expected_null_db:+.1f} dB")
    if counts_per_dB > 0 and null_delta_db <= (expected_null_db + 1.5):
        print(f"\t --> [PASS] Phase control confirmed — null matches theory.")
    elif counts_per_dB > 0:
        print(f"\t --> [WARN] Null weaker than expected — partial coherence only.")
    print("")

    # Restore calibrated phases
    for elem, phase in tx_phase_cal.items():
        set_tx_phase(sray, int(elem), int(phase))
    set_tx_phase(sray, ref_elem, 0)
    sray.latch_tx_settings()
    disable_stingray_channel(sray, cal_subarray_tx)

    # --- Step 4: Gain calibration at full gain ---
    tx_gain_dict, tx_atten_dict = tx_gain_cal(sray, ltc, cal_subarray_tx)

    return tx_phase_cal, tx_gain_dict, tx_atten_dict