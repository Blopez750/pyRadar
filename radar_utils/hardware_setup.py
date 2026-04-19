# =============================================================================
# hardware_setup.py — Hardware initialisation and radar data acquisition
# =============================================================================
#
# This module configures the three main hardware subsystems of the X-band
# phased-array radar:
#
#   1. **AD9081 MxFE** — the mixed-signal front-end that contains the ADCs
#      (analogue-to-digital converters) and DACs.  It digitises the received
#      signal from each subarray and generates the transmit waveform.
#      • 4 ADC channels at ~250 MSPS (one per subarray)
#      • 4 DAC channels for TX waveform generation
#      • NCO (Numerically Controlled Oscillator) stages for digital
#        up/down-conversion inside the chip
#
#   2. **XUD1A up/down converter** — translates between the AD9081's IF
#      (intermediate frequency) band and the 10 GHz X-band RF frequency.
#      • Uses an ADF4371 PLL to generate the LO (local oscillator) signal
#      • Each subarray channel can independently be set to RX or TX mode
#      • RX path: X-band RF → down-mix to IF → AD9081 ADC
#      • TX path: AD9081 DAC → up-mix to X-band RF → antenna
#
#   3. **ADAR1000 beamformer array ("Stingray")** — an 8-chip, 32-element
#      phased-array front-end.  Each ADAR1000 chip controls 4 antenna
#      elements and can set per-element:
#      • Phase (0–360° in ~2.8° steps) — for beam steering
#      • Gain  (0–127 register codes)  — for amplitude tapering / equalisation
#      • Mode  (RX, TX, or OFF)        — per-channel transmit/receive control
#      The 32 elements are arranged in a 4-row × 8-column planar grid,
#      grouped into 4 subarrays of 8 elements each.
#
# The module also provides helper functions to extract per-subarray data from
# the raw ADC buffer and to form the "burst matrix" (chirps × samples) used
# by the signal-processing pipeline.
#
# Subarray-to-ADC wiring (physical board layout):
#   Subarray 1 → ADC channel 3
#   Subarray 2 → ADC channel 1
#   Subarray 3 → ADC channel 0  (TX — not used for RX)
#   Subarray 4 → ADC channel 2
# =============================================================================

import adi
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from custom_libs.adar1000 import adar1000_array
from .calibration import data_capture_cal
import numpy as np

# Physical hardware mapping: subarray number → AD9081 ADC channel index.
# This is a wiring fact of the XBDP board and never changes.
# In the Stingray array there are 4 subarrays of 8 elements each.
# Each subarray is connected to one of the AD9081's 4 ADC channels,
# but the wiring is NOT in order — hence this lookup table.
SUBARRAY_TO_ADC = {1: 3, 2: 1, 3: 0, 4: 2}

def build_rx_channel_config(subarray_modes):
    """Derive which ADC channels to enable and the subarray → data-index mapping.

    When only RX subarrays are enabled on the ADC, adc.rx() returns fewer
    arrays and the indices shift.  For example if subarrays 1,2,4 are RX and
    subarray 3 is TX, the ADC only returns 3 arrays instead of 4.  This
    function figures out which position in that shortened list corresponds
    to which subarray so callers can do ``data[subarray_to_data_index[sa]]``.

    Parameters:
        subarray_modes: dict  {1: "rx", 2: "rx", 3: "tx", 4: "rx"}

    Returns:
        rx_enabled_channels:    sorted list of ADC channel indices to enable
        subarray_to_data_index: dict {subarray_id: index_in_adc_rx_output}
                                (only contains RX subarrays)
    """
    rx_adc_channels = sorted(
        SUBARRAY_TO_ADC[sa]
        for sa, mode in subarray_modes.items()
        if isinstance(mode, str) and mode.lower() == "rx"
    )
    adc_to_position = {ch: idx for idx, ch in enumerate(rx_adc_channels)}
    subarray_to_data_index = {
        sa: adc_to_position[SUBARRAY_TO_ADC[sa]]
        for sa, mode in subarray_modes.items()
        if isinstance(mode, str) and mode.lower() == "rx"
    }
    return rx_adc_channels, subarray_to_data_index

def setup_ad9081(
    url,
    rx_buffer_size=2**16,
    tx_cyclic_buffer=True,
    rx_cyclic_buffer=False,
    tx_ddr_offload=False,
    rx_channel_nco_frequencies=None,
    tx_channel_nco_frequencies=None,
    rx_main_nco_frequencies=None,
    tx_main_nco_frequencies=None,
    rx_enabled_channels=None,
):
    """Configure the AD9081 MxFE (mixed-signal front-end).

    The AD9081 contains both high-speed ADCs and DACs.  Key settings:
      - **NCO frequencies**: digital up/down-conversion stages inside the chip.
        The *main* NCO shifts the entire Nyquist band; the *channel* NCO
        provides fine frequency placement per virtual channel.
      - **rx_buffer_size**: number of IQ samples per DMA transfer — determines
        how many chirps fit in one capture (buffer_size / samples_per_chirp).
      - **cyclic buffers**: when True, the DAC continuously replays the TX
        waveform without CPU intervention.

    Returns:
        conv: configured AD9081 device object (used as ``conv`` elsewhere)
    """
    # Initialize the AD9081 device
    conv = adi.ad9081(url)

    # Set default values if not provided
    if rx_channel_nco_frequencies is None:
        rx_channel_nco_frequencies = [0] * 4
    if tx_channel_nco_frequencies is None:
        tx_channel_nco_frequencies = [0] * 4
    if rx_main_nco_frequencies is None:
        rx_main_nco_frequencies = [int(500e6)] * 4
    if tx_main_nco_frequencies is None:
        tx_main_nco_frequencies = [int(4.5e9)] * 4
    if rx_enabled_channels is None:
        rx_enabled_channels = [0, 1, 2, 3]

    # Configure the AD9081 device
    conv.rx_channel_nco_frequencies = rx_channel_nco_frequencies
    conv.tx_channel_nco_frequencies = tx_channel_nco_frequencies
    conv.rx_main_nco_frequencies = rx_main_nco_frequencies
    conv.tx_main_nco_frequencies = tx_main_nco_frequencies
    conv.rx_enabled_channels = rx_enabled_channels
    conv.rx_nyquist_zone = ["odd"] * 4
    conv.rx_buffer_size = rx_buffer_size
    conv.tx_cyclic_buffer = tx_cyclic_buffer
    conv.rx_cyclic_buffer = rx_cyclic_buffer
    conv.tx_ddr_offload = tx_ddr_offload
    conv.tx_channel_nco_gain_scales = [1.0] * 4
    conv.rx_main_6dB_digital_gains = [1] * 4

    print("")
    print("--> AD9081 MxFE Configuration")
    print(f"\t --> AD9081 configured with RX buffer size: {rx_buffer_size}")
    print(f"\t --> Sampling rate: {conv.rx_sample_rate} Hz")
    print(f"\t --> RX Channel NCO Frequencies: {rx_channel_nco_frequencies}")
    print(f"\t --> TX Channel NCO Frequencies: {tx_channel_nco_frequencies}")
    print(f"\t --> RX Main NCO Frequencies: {rx_main_nco_frequencies}")
    print(f"\t --> TX Main NCO Frequencies: {tx_main_nco_frequencies}")
    print(f"\t --> Enabled RX Channels: {rx_enabled_channels}")
    print(f"\t --> RX Nyquist Zone: {conv.rx_nyquist_zone}")
    print(f"\t --> TX Cyclic Buffer: {tx_cyclic_buffer}")
    print(f"\t --> RX Cyclic Buffer: {rx_cyclic_buffer}")
    print(f"\t --> TX DDR Offload: {tx_ddr_offload}")
    print("")

    return conv

def setup_xud1a(conv, channel_modes, rx_lo):
    """Configure the XUD1A up/down converter board.

    The XUD1A sits between the AD9081 and the antenna array.  It performs
    analogue frequency conversion:
      - **RX path**: mixes the ~10 GHz received signal down to the AD9081's
        IF band using a local oscillator (LO) set by the ADF4371 PLL.
      - **TX path**: mixes the AD9081's IF output up to X-band for transmission.

    Each of the 4 subarrays can be independently set to RX or TX mode by
    writing a control register on the XUD IIO device.

    Parameters:
        conv: AD9081 device (used to get the IIO context)
        channel_modes: dict {1: "rx", 2: "rx", 3: "tx", 4: "rx"}
        rx_lo: LO frequency in Hz
    """
    # Validate input
    if not isinstance(channel_modes, dict):
        raise ValueError("channel_modes must be a dictionary with keys as subarray numbers (1-4) and values as 'rx' or 'tx'.")

    for subarray, mode in channel_modes.items():
        if subarray not in [1, 2, 3, 4]:
            raise ValueError(f"Invalid subarray number: {subarray}. Valid subarrays are 1, 2, 3, and 4.")
        if mode.lower() not in ["rx", "tx"]:
            raise ValueError(f"Invalid mode for subarray {subarray}: {mode}. Mode must be 'rx' or 'tx'.")

    if not isinstance(rx_lo, (float)) or rx_lo <= 0:
        raise ValueError("rx_lo must be a positive number representing the frequency in Hz.")
    ctx = conv._ctrl.ctx
    # Find the XUD control device and its channels
    xud = ctx.find_device("xud_control")
    txrx1 = xud.find_channel("voltage1", True)  # Subarray 4
    txrx2 = xud.find_channel("voltage2", True)  # Subarray 3
    txrx3 = xud.find_channel("voltage3", True)  # Subarray 1
    txrx4 = xud.find_channel("voltage4", True)  # Subarray 2

    # Map subarrays to their corresponding control channels
    subarray_to_channel = {
        1: txrx3,
        2: txrx4,
        3: txrx2,
        4: txrx1
    }

    # Configure each subarray
    for subarray, mode in channel_modes.items():
        subarray_to_channel[subarray].attrs["raw"].value = "1" if mode.lower() == "tx" else "0"

    # Configure PLL select and RX gain mode (shared settings)
    PLLselect = xud.find_channel("voltage5", True)
    rxgainmode = xud.find_channel("voltage0", True)

    PLLselect.attrs["raw"].value = "1"  # Enable PLL
    rxgainmode.attrs["raw"].value = "1"  # Enable RX gain mode

    # Set the LO frequency
    adf4371 = ctx.find_device("adf4371-0")
    XUDLO = adf4371.find_channel("altvoltage2", True)
    XUDLO.attrs["frequency"].value = str(int(rx_lo))  # Set LO frequency
    XUDLO.attrs["powerdown"].value = "0"  # Ensure LO is powered on

    print("")
    print("--> Up/Down Converter Board Configuration")
    print(f"\t --> XUD1A subarrays configured: {channel_modes}")
    print(f"\t --> LO frequency set to: {rx_lo / 1e9:.2f} GHz")
    print("")

def setup_stingray(uri, subarray_modes):
    """Configure the ADAR1000 phased-array beamformer ("Stingray").

    The Stingray board contains 8 ADAR1000 ICs, each controlling 4 antenna
    elements, for a total of 32 elements arranged in a 4-row × 8-column grid.
    The 32 elements are grouped into 4 subarrays:
      - Subarrays 1 & 2: upper rows  (elements 1–16)
      - Subarrays 3 & 4: lower rows  (elements 17–32)

    Each subarray can be set to RX, TX, or OFF.  In a typical configuration
    three subarrays receive and one transmits.  The returned ``sray`` object
    is used to steer the beam and apply calibration gains/phases.

    Parameters:
        uri: IIO URI for the ADAR1000 device (e.g. "ip:192.168.2.1")
        subarray_modes: dict {1: "rx", 2: "rx", 3: "tx", 4: "rx"}

    Returns:
        sray:     configured adar1000_array object
        subarray: 4×8 element-number map (which elements belong to which subarray)
    """
    # Initialize the ADAR1000 array
    sray = adar1000_array(
    uri=uri,
    chip_ids=["adar1000_csb_1_1", "adar1000_csb_1_2", "adar1000_csb_1_3", "adar1000_csb_1_4", "adar1000_csb_2_1", "adar1000_csb_2_2", "adar1000_csb_2_3", "adar1000_csb_2_4"],
    device_map=[[1, 3, 5, 7], [2, 4, 6, 8]],
    element_map=[[1, 5, 9, 13, 17, 21, 25, 29],
                 [2, 6, 10, 14, 18, 22, 26, 30],
                 [3, 7, 11, 15, 19, 23, 27, 31],
                 [4, 8, 12, 16, 20, 24, 28, 32]],
    device_element_map={
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
    # Map subarrays to devices
    subarray_to_devices = {
        1: [1, 3],  # Subarray 1
        2: [2, 4],  # Subarray 2
        3: [6, 8],  # Subarray 3
        4: [5, 7],  # Subarray 4
    }

    subarray = np.array([
        [1, 2, 5, 6, 9, 10, 13, 14],  # subarray 1
        [3, 4, 7, 8, 11, 12, 15, 16],  # subarray 2
        [19, 20, 23, 24, 27, 28, 31, 32],  # subarray 3
        [17, 18, 21, 22, 25, 26, 29, 30]   # subarray 4
    ])

    # Configure subarrays based on the subarray_modes dictionary
    for sa_idx, subarray_mode in subarray_modes.items():
        if sa_idx not in subarray_to_devices:
            raise ValueError(f"Invalid subarray number: {sa_idx}. Valid subarrays are 1, 2, 3, and 4.")
        if subarray_mode.lower() not in ["rx", "tx", "off"]:
            raise ValueError(f"Invalid mode for subarray {sa_idx}: {subarray_mode}. Mode must be 'rx', 'tx', or 'off'.")
        SELF_BIASED_LNAs = True
        for device_id in subarray_to_devices[sa_idx]:
            device = sray.devices[device_id]
            if subarray_mode.lower() == "rx":
                device.mode = "rx"
                if SELF_BIASED_LNAs:
                    # Allow the external LNAs to self-bias
                    device.lna_bias_out_enable = False
                else:
                    # Set the external LNA bias
                    device.lna_bias_on = -0.7
                for channel in device.channels:
                    channel.rx_enable = True
                    channel.tx_enable = False

            elif subarray_mode.lower() == "tx":
                device.mode = "tx"
                for channel in device.channels:
                    channel.tx_enable = True
                    channel.rx_enable = False
                    channel.pa_bias_on = -1.1
 
            elif subarray_mode.lower() == "off":
                for channel in device.channels:
                    channel.rx_enable = False
                    channel.tx_enable = False 
    # Latch settings for RX or TX
    for subarray_mode in subarray_modes.values():
        if subarray_mode.lower() == "rx":
            sray.latch_rx_settings()
        elif subarray_mode.lower() == "tx":
            sray.latch_tx_settings()

    print(f"\t --> ADAR1000 array configured with subarray modes: {subarray_modes}")
    return sray, subarray


def extract_rx_subarrays(data, subarray_modes):
    """Demux the raw ADC capture into individual RX subarray signals.

    The AD9081 returns a flat list of arrays (one per enabled ADC channel).
    This function maps each array back to its physical subarray using the
    SUBARRAY_TO_ADC wiring table, and also returns the coherent sum of all
    RX subarrays (used as the Σ channel for monopulse or range processing).

    Parameters:
        data: list of arrays from data_capture_cal / data_capture_test
        subarray_modes: dict  {1: "rx", 2: "rx", 3: "tx", 4: "rx"}

    Returns:
        sub1, sub2, sub4: individual RX subarray IQ arrays
        sum_data:         sub1 + sub2 + sub4  (coherent sum)
    """
    _, subarray_to_data_index = build_rx_channel_config(subarray_modes)

    rx_subarrays = {}
    for subarray_id, mode in subarray_modes.items():
        if isinstance(mode, str) and mode.lower() == "rx":
            data_index = subarray_to_data_index.get(subarray_id)
            if data_index is not None and data_index < len(data):
                rx_subarrays[subarray_id] = data[data_index]
            else:
                print(f"Warning: Invalid or missing data index for Subarray {subarray_id}")

    sub1 = rx_subarrays.get(1, np.zeros_like(data[0]))
    sub2 = rx_subarrays.get(2, np.zeros_like(data[0]))
    sub4 = rx_subarrays.get(4, np.zeros_like(data[0]))

    # ADC channels can differ by ±1 sample; truncate to common length.
    min_len = min(len(sub1), len(sub2), len(sub4))
    sub1, sub2, sub4 = sub1[:min_len], sub2[:min_len], sub4[:min_len]

    sum_data = sub1 + sub2 + sub4

    return sub1, sub2, sub4, sum_data


def get_radar_data(conv, cal_ant_fix, subarray_modes, num_bursts, PRI_ms,
                   start_offset_samples, good_ramp_samples,
                   coherent_integration=False):
    """Acquire one frame of radar data and slice it into a burst matrix.

    The ADC captures a continuous stream of IQ samples.  An FMCW radar
    transmits repeated frequency-swept chirps, each lasting PRI_ms milliseconds.
    This function carves the continuous buffer into rows of
    (num_bursts × samples_per_chirp), producing the 2-D "burst matrix" that
    the Range-Doppler FFT expects.

    Coherent integration (optional):
      Two consecutive ADC captures are stitched end-to-end before slicing.
      This doubles the number of chirps, which doubles Doppler resolution
      (Δv = λ / (2·N·PRI)) at the cost of doubled latency.

    Parameters:
        conv:                  AD9081 converter object
        cal_ant_fix:           calibration corrections to apply immediately
        subarray_modes:        dict {1: "rx", ...}
        num_bursts:            number of chirps to extract
        PRI_ms:                pulse repetition interval (ms)
        start_offset_samples:  samples to skip at the start of each chirp
                               (avoids transient from chirp retrace)
        good_ramp_samples:     usable samples per chirp
        coherent_integration:  if True, capture twice and stitch

    Returns:
        rx_bursts:        (num_bursts, samples_per_chirp) sum-channel burst matrix
        sum_data:         raw summed IQ stream
        subarray_data:    dict {1: sub1, 2: sub2, 4: sub4} raw streams
        subarray_bursts:  dict of per-subarray burst matrices (non-coherent only)
    """
    N_frame = int(PRI_ms / 1000 * float(conv.rx_sample_rate))
    samples_per_chirp = N_frame

    if coherent_integration:
        # First capture
        data1 = data_capture_cal(conv, cal_ant_fix)
        sub1_1, sub2_1, sub4_1, sum_data1 = extract_rx_subarrays(data1, subarray_modes)

        # Second capture
        data2 = data_capture_cal(conv, cal_ant_fix)
        sub1_2, sub2_2, sub4_2, sum_data2 = extract_rx_subarrays(data2, subarray_modes)

        # Stitch
        sum_data = np.concatenate([sum_data1, sum_data2])
        sub1 = np.concatenate([sub1_1, sub1_2])
        sub2 = np.concatenate([sub2_1, sub2_2])
        sub4 = np.concatenate([sub4_1, sub4_2])

        total_bursts = num_bursts * 2

        rx_bursts = np.zeros((total_bursts, samples_per_chirp), dtype=complex)
        for burst in range(total_bursts):
            start_index = start_offset_samples + burst * N_frame
            stop_index = start_index + samples_per_chirp
            if stop_index <= len(sum_data):
                rx_bursts[burst] = sum_data[start_index:stop_index]

        subarray_data = {1: sub1, 2: sub2, 4: sub4}
        return rx_bursts, sum_data, subarray_data

    else:
        # ── Standard single-capture path ───────────────────────────────
        # The ADC captures a continuous stream of IQ samples.  The FMCW
        # waveform consists of num_bursts identical chirps, each N_frame
        # samples long.  We carve the stream into rows of a 2-D "burst
        # matrix" (num_bursts × samples_per_chirp) using vectorised index
        # arithmetic rather than a Python loop for speed.
        #
        # start_offset_samples skips the initial chirp retrace transient;
        # good_ramp_samples is the usable portion of each chirp.
        data = data_capture_cal(conv, cal_ant_fix)
        sub1, sub2, sub4, sum_data = extract_rx_subarrays(data, subarray_modes)

        # Vectorized burst extraction: build a 2-D index array where
        # row i selects samples [start + i*N_frame .. start + i*N_frame + N_chirp)
        burst_indices = np.arange(num_bursts)
        start_indices = start_offset_samples + burst_indices * N_frame
        sample_offsets = np.arange(samples_per_chirp)
        all_indices = start_indices[:, np.newaxis] + sample_offsets[np.newaxis, :]
        rx_bursts = sum_data[all_indices]

        sub1_bursts = sub1[all_indices]
        sub2_bursts = sub2[all_indices]
        sub4_bursts = sub4[all_indices]

        subarray_data = {1: sub1, 2: sub2, 4: sub4}
        subarray_bursts = {1: sub1_bursts, 2: sub2_bursts, 4: sub4_bursts}

        return rx_bursts, sum_data, subarray_data, subarray_bursts