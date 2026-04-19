# =============================================================================
# calibration.py — RX & TX signal-chain calibration for an X-band phased array
# =============================================================================
#
# WHY CALIBRATION IS NECESSARY:
#   A phased-array radar has many antenna elements.  For the beam to form
#   correctly (constructive interference in the desired direction, destructive
#   elsewhere), ALL elements must be phase-aligned and amplitude-matched.
#
#   In practice, each element's signal path has slightly different amplitude
#   and phase due to:
#     • PCB trace-length differences
#     • ADAR1000 chip-to-chip variation
#     • Connector and cable tolerances
#     • Antenna element manufacturing spread
#
#   Without calibration, these mismatches degrade the beam pattern:
#     • Higher sidelobes (false targets at wrong angles)
#     • Reduced main-beam gain (lower detection range)
#     • Beam pointing error (target appears at wrong angle)
#
# CALIBRATION APPROACHES:
#
#   1. **Gain equalisation** — measure each element's amplitude, compute
#      the offset from the weakest element, and programme the ADAR1000 VGA
#      (variable-gain amplifier, 7-bit register 0–127) to equalise all
#      amplitudes.  Uses a pre-characterised polynomial curve that maps
#      dB offset → VGA code.
#
#   2. **Phase alignment (RX)** — two methods:
#        a) *Analog null-steering*: sweep a phase offset on one element
#           until the coherent sum with a reference element reaches a
#           minimum (destructive null).  The offset at the null is the
#           phase error.  Resolution: ~2.8° (ADAR1000 phase register).
#        b) *Digital NCO*: read the complex IQ sample's angle directly
#           and programme the AD9081's NCO to cancel the residual offset
#           with milli-degree resolution.
#
#   3. **Phase alignment (TX)** — *maximum-power method*: sweep the phase
#      of each TX element while transmitting alongside a reference element,
#      and find the phase that produces the highest combined power at an
#      external detector (LTC2314 ADC on the Stingray board).  The peak
#      phase is the correction value.
#
# CALIBRATION PROCEDURE (external setup required):
#   • RX cal: place a known point source (signal generator) at boresight
#     and measure each element's response.
#   • TX cal: connect the Stingray TX output to an external power detector
#     (J9 connector) and sweep each element's phase.
#
# =============================================================================

import numpy as np
import re
import os
from scipy.signal import correlate

####################################################################################################
#                       Functions used throughout the calibration process                          #
####################################################################################################

def enable_stingray_channel(obj, elements=None, man_input=False):
    """
    Enables the specified Stingray channel based on the mode. If no elements are passed, ask for user input
    """
    if elements is None and man_input:
        user_input = input("Enter a comma-separated list of channels to turn on (1-32): ")
        elements = [int(x.strip()) for x in user_input.split(',') if 1 <= int(x) <= 32]

    if man_input == False:
        elements = np.array(elements).flatten()

    for device in obj.devices.values():
        if device.mode == "rx":
            for channel in device.channels:

                str_channel = str(channel)
                value = int(strip_to_last_two_digits(str_channel))

                # Check if the channel is in the list of elements to enable
                # If it is, enable the channel
                for elem in elements:
                    if elem == value:
                        channel.rx_enable = True

        elif device.mode == "tx":
            for channel in device.channels:

                str_channel = str(channel)
                value = int(strip_to_last_two_digits(str_channel))

                # Check if the channel is in the list of elements to enable
                # If it is, enable the channel
                for elem in elements:
                    if elem == value:
                        channel.tx_enable = True
        else:
            raise ValueError('Mode of operation must be either "rx" or "tx"')
 
def disable_stingray_channel(obj, elements=None, man_input=False):
    """
    Disables the specified Stingray channel based on the mode. If no elements are passed, ask for user input
    """
    if elements is None and man_input:
        user_input = input("Enter a comma-separated list of channels to turn off (1-32): ")
        elements = [int(x.strip()) for x in user_input.split(',') if 1 <= int(x) <= 32]

    if man_input == False:
        elements = np.array(elements).flatten()

    for device in obj.devices.values():
        if device.mode == "rx":
            for channel in device.channels:

                str_channel = str(channel)
                value = int(strip_to_last_two_digits(str_channel))

                # Check if the channel is in the list of elements to disable
                # If it is, disable the channel
                for elem in elements:
                    if elem == value:
                        channel.rx_enable = False

        elif device.mode == "tx":
            for channel in device.channels:

                str_channel = str(channel)
                value = int(strip_to_last_two_digits(str_channel))

                # Check if the channel is in the list of elements to disable
                # If it is, disable the channel
                for elem in elements:
                    if elem == value:
                        channel.tx_enable = False
        else:
            raise ValueError('Mode of operation must be either "rx" or "tx"')
        
# ---------------------------------------------------------------------------
# Data capture helpers
# ---------------------------------------------------------------------------
# The AD9081 MxFE (mixed-signal front-end) digitises the analogue signals from
# the phased-array antenna elements.  adc.rx() returns a list of complex IQ
# arrays — one per enabled ADC channel.  "Destroying" the buffer before a new
# capture ensures we get fresh samples rather than stale data.

def data_capture(adc):
    """Raw ADC capture — no calibration applied."""
    adc.rx_destroy_buffer()  # discard any old DMA data
    for i in range(1):
        data = adc.rx()       # request one new buffer from the ADC
    return data

def data_capture_test(adc, cal_values):
    """Capture with phase-only calibration (gain factors left at unity)."""
    adc.rx_destroy_buffer()
    data = adc.rx()
    # Apply calibration: phase corrections from cal_values, unity gain [1,1,1,1]
    data = cal_data(data, cal_values, [1, 1, 1, 1])
    return data

def data_capture_cal(adc, cal_values):
    """Capture with phase-only calibration (same as data_capture_test but keeps
    the DMA buffer alive for back-to-back captures used in calibration sweeps)."""
    adc.rx_destroy_buffer()
    for i in range(1):
        data = adc.rx()
    # Apply only the phase correction — gain stays at 1.0
    data = cal_data(data, cal_values, [1, 1, 1, 1])
    return data
 
def phase_delayer(data, delay):
    """Rotate complex IQ data by *delay* degrees.

    Multiplying a complex signal by exp(j·θ) shifts its phase by θ.
    This is the fundamental operation used to steer a phased-array beam
    or to correct per-element phase errors during calibration.
    """
    delayed_data = data * np.exp(1j * np.deg2rad(delay))
    return delayed_data

def cal_data(data, phaseCAL, gainCal):
    """Apply per-channel gain and phase calibration to raw ADC data.

    For each ADC channel i:
      1. Scale amplitude by gainCal[i]  (amplitude equalisation)
      2. Rotate phase by phaseCAL[i] degrees  (phase alignment)
    """
    for i in range(len(data)):
        data[i] = phase_delayer(data[i] * gainCal[i], phaseCAL[i])
    return data

def gain_codes(obj, analog_mag_pre_cal, mode):
    """Determine VGA gain register codes to equalise element amplitudes.

    Each ADAR1000 element has a variable-gain amplifier (VGA) controlled by
    a 7-bit register (0–127).  This function takes the measured amplitude of
    every element (in dBFS), computes how much each one needs to be adjusted
    relative to the weakest element, and looks up the correct register code
    using pre-characterised polynomial curves (one for the main gain path,
    one for the attenuated path).

    Parameters:
        analog_mag_pre_cal: array of element amplitudes in dBFS
        mode: "rx" or "tx" — selects the correct polynomial set

    Returns:
        gain_codes_cal: VGA codes (0–127) for each element
        atten:          1 if the element also needs the attenuator engaged
    """
    atten = np.zeros(np.shape(analog_mag_pre_cal))
   
    # Polynomial fit coefficients for Rx and Tx modes
    if mode == "rx":
        poly_atten1 = [-4.178368227245296e-09, -3.124456767699238e-07, -7.218061870232358e-06,
                     1.146280656652001e-05, 0.003079353177989, 0.048281159204065,
                     0.247215102895886, 0.176811045216789, 10.163992861226674, 127.1237461140638]
        poly_atten0 = [4.12957161960063e-10, 1.11191262836380e-07, 1.34714959988008e-05,
                     0.000967813015434471, 0.0456701602403594, 1.47865205676699,
                     33.2281071820574, 510.768971360134, 5126.75849430329, 30268.0388934082, 79815.3362477404]
    elif mode == "tx":
        poly_atten1 = [2.11066024918707e-11, 5.70009839945272e-09, 5.49937839434060e-07,
                     2.73783996444945e-05, 0.000800103462132557, 0.0143895847100511,
                     0.159688022065331, 1.06808692792217, 4.50865732789487,
                     21.0313863981928, 127.541504127078]
        poly_atten0 = [5.00901188825329e-10, 1.44673726954726e-07, 1.86493751074489e-05,
                     0.00141263342869073, 0.0696128076080524, 2.33122230277517,
                     53.7090182591208, 840.244043642996, 8539.25517554357,
                     50904.6416796854, 135350.366810770]
 
    # Calculate delta in dB
    mag_min = np.min(analog_mag_pre_cal)
    mag_cal_diff = analog_mag_pre_cal - mag_min
    mag_cal_poly = np.zeros(np.shape(analog_mag_pre_cal))
   
    # Find correct gain code values based on which polynomial should be used
    # Adjust attenuators accordingly
    for i in range(np.size(analog_mag_pre_cal)):

        if mag_cal_diff.flat[i] < 23:
            mag_cal_poly.flat[i] = np.floor(np.polyval(poly_atten1, -1 * mag_cal_diff.flat[i]))

        elif mag_cal_diff.flat[i] == np.inf:
            mag_cal_poly.flat[i] = 0
            atten.flat[i] = 1
            
        else:
            mag_cal_poly.flat[i] = np.floor(np.polyval(poly_atten0, -1 * mag_cal_diff.flat[i]))
            atten.flat[i] = 1
 
    # set min and max clipping to 0 and 127, respectively
    mag_cal_poly = np.clip(mag_cal_poly, 0, 127)
 
    gain_codes_cal = mag_cal_poly
 
    return gain_codes_cal, atten 

def strip_to_last_two_digits(input_string):
    """
    Extract the last two digits from a string.
    """
    # Find all sequences of digits in the string
    all_numbers = re.findall(r'\d+', input_string)

    # Join them together and take the last two digits
    last_two_digits = ''.join(all_numbers)[-2:]
    
    return last_two_digits


def _to_native_scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value

def create_dict(new_keys, array):
    """
    Create a dictionary with new keys and values from array.
    """
    result_dict = {}

    for i in range(array.shape[0]):

        for j in range(array.shape[1]):

            key = int(_to_native_scalar(new_keys[i][j]))
            value = _to_native_scalar(array[i][j])
            result_dict[key] = value

    return result_dict


def print_subarray_values(values_dict, subarray, label="Values"):
    """
    Print calibration values grouped by subarray for easy physical mapping.
    """
    print(f"\t --> {label} by subarray:")
    for subarray_idx, row in enumerate(np.asarray(subarray), start=1):
        entries = []
        for elem in np.asarray(row).flatten():
            element_id = int(_to_native_scalar(elem))
            value = _to_native_scalar(values_dict.get(element_id))
            entries.append(f"{element_id}:{value}")
        print(f"\t     Subarray {subarray_idx}: " + ", ".join(entries))

def wrap_to_360(angle):
    """Wrap angle to the range [0, 360)."""
    return angle % 360

def ind2sub(array_shape, index):
    """Convert a linear index to row and column indices."""
    rows = index % array_shape[0]
    cols = index // array_shape[0]
    return rows, cols

####################################################################################################
#               RX signal-chain calibration functions for the X-Band Development Kit               #
####################################################################################################

def calc_dbfs(data):
    """Compute the dBFS (decibels relative to full scale) spectrum of ADC data.

    dBFS is a standard way to express how close a digital signal's amplitude
    is to the maximum value the ADC can represent.  0 dBFS = full scale.

    Steps:
      1. Apply a Hamming window to reduce spectral leakage (sidelobe artefacts).
      2. Take the FFT and normalise by the window's coherent gain.
      3. Convert magnitude to dB, referenced to the ADC's full-scale peak.

    The ADC is 12-bit signed, so full-scale peak = 2^11 = 2048.
    """
    from radar_utils.signal_processing import _get_window
    NumSamples = len(data)
    win = _get_window(NumSamples)
    y = data * win
    s_fft = np.fft.fft(y) / np.sum(win)   # normalise by window's coherent gain
    s_shift = np.fft.fftshift(s_fft)       # centre DC in the middle of the spectrum

    s_mag = np.abs(s_shift)
    s_mag[s_mag == 0] = 1e-12              # guard against log(0)

    # 2^11 = 2048 = full-scale peak for a 12-bit signed ADC
    s_dbfs = 20 * np.log10(s_mag / (2**11))
    return s_dbfs

def find_phase_delay_sliding_ref(obj, adc, subarray_ref, adc_map, delay_phases):
    """Measure per-ADC-channel phase offsets using the **null-steering** method
    with a sliding reference.

    Null-steering principle:
      Two signals with identical phase will cancel (null) when subtracted.
      We sweep a phase offset on one channel and find the value that
      minimises the *peak* of (channel_A − channel_B).  That offset is
      the channel's phase error.

    'Sliding reference' means each channel is referenced to its immediate
    neighbour rather than to a single fixed channel.  This accumulates
    the correction chain-style: ch0→ch1→ch2→ch3.
    """

    # Enable the Stingray reference channels and capture data
    enable_stingray_channel(obj,subarray_ref)
    data = np.array(data_capture(adc))

    # Create a list to store the calibration values for each antenna
    # Initialize the first antenna's calibration value to 0
    cal_ant = []
    cal_ant.append(0)

    for i in range(len(data)-1):
        peak_sum = []
        for phase_delay in delay_phases:

            # Apply the phase delay to the first and second antennas
            first_ant = phase_delayer(data[adc_map[i]], phase_delay*i+cal_ant[i])
            second_ant = phase_delayer(data[adc_map[i+1]], phase_delay*(i+1))

            # Calculate the delayed sum of the two antennas
            # and find the maximum value
            delayed_sum = calc_dbfs(first_ant - second_ant)
            peak_sum.append(np.max(delayed_sum))

        # Find the minimum value in the peak sum and its index
        # This index corresponds to the phase delay that minimizes the difference
        null_val = np.min(peak_sum)
        null_index = np.where(peak_sum==null_val)

        # Get the phase delay value that corresponds to the minimum peak sum
        # and append it to the calibration values list
        cal_value = delay_phases[null_index]
        cal_ant.append(cal_value[0])

    # Disable the Stingray reference channels
    disable_stingray_channel(obj,subarray_ref)
    return cal_ant

def find_phase_delay_fixed_ref(obj, adc, subarray_ref, adc_ref, delay_phases):
    """Measure per-ADC-channel phase offsets using the **null-steering** method
    with a fixed reference channel.

    Same principle as the sliding-reference variant, but every channel is
    compared directly against one constant reference channel (adc_ref).
    This avoids accumulated rounding errors from chaining.
    """
    # Enable the Stingray reference channels and capture data
    enable_stingray_channel(obj,subarray_ref)
    data = np.array(data_capture(adc))

    # Create a list to store the calibration values for each antenna
    # Initialize the first antenna's calibration value to 0
    cal_ant = []
    cal_ant.append(0)

    # Apply a zero phase delay to the reference antenna
    first_ant = phase_delayer(data[adc_ref], cal_ant[0])

    for i in range(len(data)-1):
        peak_sum = []
        for phase_delay in delay_phases:

            # Apply the phase delay second antennas
            second_ant = phase_delayer(data[i], phase_delay)

            # Calculate the delayed sum of the two antennas
            delayed_sum = calc_dbfs(first_ant - second_ant)

            # Find the maximum value
            peak_sum.append(np.max(delayed_sum))
        
        # Find the minimum value in the peak sum and its index
        null_val = np.min(peak_sum)
        null_index = np.where(peak_sum==null_val)

        # Get the phase delay value that corresponds to the minimum peak sum
        # and append it to the calibration values list
        cal_value = delay_phases[null_index]
        cal_ant.append(cal_value[0])

    # Disable the Stingray reference channels
    disable_stingray_channel(obj,subarray_ref)

    # Roll the calibration values to align with the reference antenna
    # This is done because data[adc_ref] corresponds to subarray 4
    return np.roll(cal_ant, -1)

def phase_digital(obj, adc, adc_ref, subarray_ref):
    """Measure and programme fine digital phase offsets in the AD9081's NCOs.

    After the coarse analog phase calibration, there are still small residual
    phase errors between the four ADC channels inside the AD9081 MxFE.  These
    are corrected digitally by programming each channel's NCO (Numerically
    Controlled Oscillator) with a phase offset in **milli-degrees** (1/1000°).

    Procedure:
      1. Enable all reference elements so every ADC channel sees the same signal.
      2. Capture one buffer and read the complex phase at sample 100.
      3. Compute each channel's phase relative to the reference channel.
      4. Write the corrections (in milli-degrees) into the AD9081's NCO registers.
    """
    # Enable analog array_reference channels for NCO calibration
    enable_stingray_channel(obj, subarray_ref)

    # Capture ADC data
    data = np.array(data_capture(adc))

    # Extract sample 100 from each IQ, phase in degrees
    phase_compare = np.angle(data[:, 100], deg = True)

    # Measure phase delta with respect to reference and scale to millidegrees
    digital_phase_cal = (np.mod(phase_compare - phase_compare[adc_ref] + 180, 360) - 180) * 1e3

    # Disable analog array_reference channels
    disable_stingray_channel(obj, subarray_ref)

    # write NCO phases to AD9081
    adc.rx_main_nco_phases = (np.round(digital_phase_cal).astype(int)).tolist()
 
    return digital_phase_cal
 
def rx_gain(obj, adc, subarray, adc_map, element_map):
    """Equalise RX gain across all antenna elements.

    Each of the 32 antenna elements has a slightly different signal-path gain.
    This function measures each element's amplitude individually (one at a time),
    uses a polynomial model (gain_codes()) to compute the VGA register code that
    brings it in line with the weakest element, then programmes the hardware and
    verifies by remeasuring.

    Only the subarrays present in *subarray* (the RX subset) are calibrated;
    TX subarrays are skipped.
    """

    # Capture ADC data with initial gain, attenuation, and phase settings
    data = rx_single_channel_data(obj, adc, subarray, adc_map)

    # Extract only the rows for the subarrays being calibrated.
    # rx_single_channel_data returns a 32-row buffer; rows for uncalibrated subarrays
    # stay zero, which would corrupt the dB minimum used by gain_codes.
    active_elements = subarray.flatten().astype(int)
    data_active = data[active_elements - 1, :]  # 0-indexed rows for active elements only

    # Measure analog magnitude pre-calibration
    analog_mag_pre_cal = get_analog_mag(data_active)

    # Reshape the analog magnitude to match the (active) subarray shape in column-major order
    analog_mag_pre_cal = np.reshape(analog_mag_pre_cal, np.shape(subarray), order = 'F')

    # Calculate gain codes and attenuation values based on pre-calibration magnitude
    gain_codes_cal, atten_cal = gain_codes(obj, analog_mag_pre_cal, "rx")

    # Create dictionary for active elements only (keyed by element number)
    gain_dict = create_dict(subarray, gain_codes_cal)
    atten_dict = create_dict(subarray, atten_cal)

    for element in obj.elements.values():
        """
        Iterate through each element in the Stingray object
        Convert the element to a string and extract the last two digits
        This is used to map the element to its corresponding gain and attenuation values
        in the dictionaries created above
        """
        str_channel = str(element)
        value = int(strip_to_last_two_digits(str_channel))

        # Only apply to elements that were calibrated; skip TX-mode subarray elements
        if value in gain_dict:
            element.rx_attenuator = atten_dict[value]
            element.rx_gain = gain_dict[value]

            obj.latch_rx_settings() # Latch SPI settings to devices
   
    # Capture ADC data with calibrated gain codes and attenuation values
    data = rx_single_channel_data(obj, adc, subarray, adc_map)
    data_active = data[active_elements - 1, :]
   
    # Measure analog magnitude post-calibration
    analog_mag_post_cal = get_analog_mag(data_active)
    analog_mag_post_cal = np.reshape(analog_mag_post_cal, np.shape(subarray), order = 'F')
 
    return gain_dict, atten_dict, analog_mag_pre_cal, analog_mag_post_cal

def phase_analog(sray_obj, adc_obj, adc_map, adc_ref, subarray_ref, subarray_targ, dig_phase):
    """Measure and correct the analog phase offset of every antenna element.

    This is the per-element equivalent of the coarse inter-ADC phase calibration.
    One reference element transmits/receives simultaneously with one target element
    at a time.  By comparing the complex phase of the two received signals, we
    learn the target element's phase error and programme it into the ADAR1000's
    per-element phase register so all elements are phase-aligned.

    Subarray 1 is calibrated against a reference in subarray 2, while
    subarrays 2–4 are calibrated against a reference in subarray 1.  This
    avoids using an element from the same subarray as its own reference.
    """
    analog_phase = np.zeros((4, 8))  # Initialize phase array
    element_map = np.array([
        [1, 5, 9, 13, 17, 21, 25, 29],
        [2, 6, 10, 14, 18, 22, 26, 30],
        [3, 7, 11, 15, 19, 23, 27, 31],
        [4, 8, 12, 16, 20, 24, 28, 32]
    ])

    # break out subarray 1 cal, vs subarrays 2,3,4 cal
    for ii in range(2): 
        for jj in range(subarray_targ.shape[1]):
            dummy_array = np.zeros((4, 8))

            if ii == 0:

                # When calibrating subarray 1, use the reference channel from subarray 2
                tmp_array_ref = subarray_ref[1]

                # Assign the target channels from subarray 1 to tmp_targ
                tmp_targ = subarray_targ[0, :]

                # Enable the reference channel in subarray 2
                enable_stingray_channel(sray_obj, tmp_array_ref)

                # Iterate through subarray 1 and enable one channel at a time (excludes the reference channel)
                enable_stingray_channel(sray_obj, tmp_targ[jj])

                # Grab row and column indices for specific channel in subarray 1
                row, col = ind2sub(dummy_array.shape, tmp_targ[jj] - 1)

            else:

                # When calibrating subarrays 2, 3, and 4, use the reference channel from subarray 1
                tmp_array_ref = subarray_ref[0]

                # Assign the target channels from subarray 2, 3, and 4 to tmp_targ
                tmp_targ = subarray_targ[1:4]

                # Enable the reference channel in subarray 1
                enable_stingray_channel(sray_obj, tmp_array_ref)

                # Enable the target channels in subarray 2, 3, and 4
                enable_stingray_channel(sray_obj, tmp_targ[:, jj])

                # Grab row and column indices for specfic channel in subarray 2, 3, and 4
                row, col = ind2sub(dummy_array.shape, tmp_targ[:, jj] - 1)

            # Capture ADC data for the enabled channels
            data = data_capture_cal(adc_obj, dig_phase)
            data = np.array(data).T

            # Extract the phase information from the captured data and convert to degrees
            phase_compare = np.angle(data[100, :]) * 180 / np.pi

            if ii == 0:
                # When calibrating subarray 1, use the reference channel from subarray 2
                analog_phase[row, col] = wrap_to_360(phase_compare[adc_map[0]] - phase_compare[adc_map[1]])
            else:
                for n in range(1, len(adc_map)):
                    # n=1→sub2, n=2→sub3, n=3→sub4; use each subarray's own ADC channel
                    analog_phase[row[n-1], col[n-1]] = wrap_to_360(phase_compare[adc_map[n]] - phase_compare[adc_ref])

            if ii == 0:
                # Disable the target channel in subarray 1
                disable_stingray_channel(sray_obj, tmp_targ[jj])
            else:
                # Disable the target channels in subarrays 2, 3, and 4
                disable_stingray_channel(sray_obj, tmp_targ[:, jj])

        # Disable the reference channel being used for calibration
        disable_stingray_channel(sray_obj, tmp_array_ref)

    analog_phase_dict = create_dict(element_map, analog_phase)

    for element in sray_obj.elements.values():
        str_channel = str(element)
        value = int(strip_to_last_two_digits(str_channel))

        # Assign the calculated phase to the element
        element.rx_phase = analog_phase_dict[value]
        sray_obj.latch_rx_settings()  # Latch SPI settings to devices

    return analog_phase_dict

def rx_single_channel_data(obj, adc, array, adc_map):
        """Capture raw ADC data one element at a time.

        To measure each element's individual gain, we enable only *one* element
        per subarray at a time, capture the ADC buffer, and store the result
        in the corresponding row of a 32×4096 matrix (32 elements, 4096 samples).
        The mapping from ADC channel index to subarray row is provided by adc_map.
        """
        rx_data = np.zeros((32,4096), dtype = complex)  # Allocate memory
        for a in range(np.size(array,1)):

            # Enable one reference channel per subarray
            enable_stingray_channel(obj, array[:,a])

            # Pull data from ADC
            data = np.array(data_capture(adc))
            data = data[:, :np.size(rx_data,1)] # remove data past 4096th column for FFT

            # Initialize temporary array for ADC data
            # This is used to map the ADC data to the correct subarray
            new_data = np.zeros(np.shape(data), dtype = complex)

            for i in range(len(adc_map)):
                # Map data to the correct ADC channels
                new_data[i,:] = data[adc_map[i],:]

            for index, row in enumerate(array[:,a]):

                # Map the data to the correct row in the rx_data matrix
                # The row is determined by the index of the subarray
                rx_data[row - 1,:] = new_data[index,:]

            # Disable target channels
            disable_stingray_channel(obj, array[:,a])
        return rx_data
 
def get_analog_mag(data, nfft=4096):
    """Extract the peak spectral magnitude (dBFS) for each element.

    For each row of the data matrix (one row per element), we compute the
    FFT and find the fundamental-tone magnitude in dBFS (decibels relative
    to full scale for a 16-bit ADC).  The result is a 1×N vector of dBFS
    values used by gain_codes() to decide how much gain correction each
    element needs.

    This replaces the previous genalyzer-based implementation with pure
    numpy, extracting the same 'A:mag_dbfs' value.
    """
    full_scale = 2 ** 15  # 16-bit two's complement full scale

    def _tone_mag_dbfs(iq_row):
        """Return the magnitude of the strongest tone in dBFS."""
        N = min(len(iq_row), nfft)
        # Use real and imaginary parts as separate channels (matches
        # the genalyzer real_data/imag_data split)
        real_part = np.real(iq_row[:N]).astype(np.float64)
        imag_part = np.imag(iq_row[:N]).astype(np.float64)
        # Complex FFT from I/Q
        sig = real_part + 1j * imag_part
        spectrum = np.fft.fft(sig, n=nfft)
        mag = np.abs(spectrum) / nfft
        # Peak magnitude (fundamental tone) excluding DC
        mag[0] = 0.0
        peak_mag = np.max(mag)
        if peak_mag < 1e-12:
            return -120.0
        return 20.0 * np.log10(peak_mag / full_scale)

    if data.ndim == 1:
        return _tone_mag_dbfs(data)

    analog_mag = np.zeros((1, data.shape[0]))
    for i in range(data.shape[0]):
        analog_mag[0, i] = _tone_mag_dbfs(data[i, :])
    return analog_mag

####################################################################################################
#               TX signal-chain calibration functions for the X-Band Development Kit               #
####################################################################################################

def get_ltc_raw(ltc):
    """Read the raw 14-bit code from the LTC2314-14 power detector ADC."""
    result = ltc._get_iio_attr("voltage0", "raw", False, ltc._ctrl)
    while result == 0:
        result = ltc._get_iio_attr("voltage0", "raw", False, ltc._ctrl)
    return int(result)


def get_ltc_voltage(ltc):
    """Read the TX power detector voltage from the LTC2314-14 ADC.

    The LTC2314 is a 14-bit ADC connected to an RF power detector.
    Its reading is proportional to the combined TX power radiated by
    the enabled elements.  The raw code is converted to dBFS so we can
    compare power levels across different element/phase combinations.
    """
    result = get_ltc_raw(ltc)
    result = 20 * np.log10(result/((2 ** 14)-1))
    return result

def set_tx_phase(sray, ant, phase):
    """
    Set the TX phase for a specific channel in the Stingray array.
    """

    for element in sray.elements.values():
        # Convert the element to a string and extract the last two digits
        str_channel = str(element)
        value = int(strip_to_last_two_digits(str_channel))

        # Check if the value matches the antenna number
        if value == ant:
            # If it does, set the tx_phase attribute of the element
            element.tx_phase = phase

    sray.latch_tx_settings()

def tx_gain_cal(sray, ltc, subarray):
    """Equalise TX amplitudes across the transmit subarray.

    Same concept as RX gain calibration, but measured with an external power
    detector (LTC2314).  Each TX element is enabled alone, the detector
    reading is recorded, and a polynomial fit maps the amplitude spread
    to VGA gain register codes that flatten the array's aperture illumination.
    """

    # Disable all channels
    disable_stingray_channel(sray, subarray)
    detect = []

    # Iterate only over the TX subarray elements
    for m in range(subarray.shape[0]):
        for n in range(subarray.shape[1]):

            # Turn each channel on and capture the LTC voltage
            enable_stingray_channel(sray, subarray[m][n])
            detect.append(get_ltc_voltage(ltc))

            # Turn each channel off
            disable_stingray_channel(sray, subarray[m][n])

    # Convert the list of detected values to a numpy array and reshape it to match the TX subarray
    detect = np.array(detect)
    detect = np.reshape(detect, subarray.shape)

    # Apply polynomial fit to the detected values for normalization
    tx_gain_cal_vals, tx_atten_cals = gain_codes(sray, detect, "tx")

    # Create dictionary keyed only by TX subarray elements
    gain_dict = create_dict(subarray, tx_gain_cal_vals)
    atten_dict = create_dict(subarray, tx_atten_cals)
    print_subarray_values(gain_dict, subarray, "TX Gain")
    for element in sray.elements.values():

        str_channel = str(element)
        value = int(strip_to_last_two_digits(str_channel))

        if value in gain_dict:
            element.tx_attenuator = atten_dict[value]
            element.tx_gain = gain_dict[value]
            sray.latch_tx_settings() # Latch SPI settings to devices
    
    return gain_dict, atten_dict
    
def tx_phase(sray, ltc, subarray, step_thru=False, baseline_raw=0):
    """
    Calibrate the TX phase for each channel in the Stingray array.
    Uses the maximum-power method: sweep phase of the target element alongside
    the fixed reference, find the phase that produces maximum combined power —
    that is the coherent alignment phase.

    The baseline_raw parameter is the DC offset of the power detector with
    no RF signal.  It is subtracted from every LTC reading so the peak-
    finding operates on the actual RF signal component, not the total
    reading which is dominated by the DC offset.
    """
    # Disable all channels and enable only the reference channel (first TX element)
    disable_stingray_channel(sray, subarray)
    enable_stingray_channel(sray, subarray[0, 0])
    element_to_phase = {}
    N_avg = 5  # LTC averages per phase step — reduces noise in peak detection
    for m in range(subarray.shape[0]):
        for n in range(subarray.shape[1]):
            elem = int(subarray[m][n])
            if n == 0 and m == 0:

                # Reference channel: assign 0 degrees
                set_tx_phase(sray, int(subarray[m][n]), int(0))
                print(f'Calibrated Element {subarray[m][n]} (reference)')

            else:
                # Enable target channel alongside the reference
                enable_stingray_channel(sray, subarray[m][n])
                detect = []

                for i in range(0, 360, 1):

                    # Set the TX phase to i degrees and average N_avg raw LTC
                    # readings minus the DC baseline.
                    set_tx_phase(sray, int(subarray[m][n]), int(i))
                    raw_avg = float(np.mean([get_ltc_raw(ltc) for _ in range(N_avg)]))
                    detect.append(raw_avg - baseline_raw)

                # Maximum-power method: the phase at which combined power is highest
                # is when the target element is coherent with the reference.
                _d = np.array(detect)
                cal_phase = int(np.argmax(_d))
                peak_val = _d[cal_phase]
                null_val = float(np.min(_d))
                # The detector is logarithmic (counts ∝ dB), so the contrast
                # in dB equals the count difference, NOT 10*log10(ratio).
                # We report the raw peak-to-null count swing; the caller can
                # convert with the counts_per_dB scale factor if known.
                swing = peak_val - null_val
                set_tx_phase(sray, int(subarray[m][n]), int(cal_phase))
                element_to_phase[elem] = int(cal_phase)

                # Turn the channel off and print the channel has been calibrated
                disable_stingray_channel(sray, subarray[m][n])
                print(f'Calibrated Element {subarray[m][n]:>2d}  '
                      f'phase={cal_phase:>3d}°  '
                      f'peak={peak_val:.0f}  null={null_val:.0f}  '
                      f'swing={swing:.0f} cts')
    return element_to_phase
        
####################################################################################################
#               Functions that were written to sanity check our calibrations                       #
####################################################################################################
# NOTE: These are not used in RX/TX calibration 
# but are useful for debugging and sanity checking the calibrations.

def find_phase_difference(waveform1, waveform2, sampling_rate):
    """
    Calculate the phase difference between two waveforms using cross-correlation.
    """
    # Calculate the cross-correlation between the two waveforms
    correlation = correlate(waveform1, waveform2, mode='full')
    
    # Find the index of the maximum correlation value
    max_corr_index = np.argmax(correlation)
    
    # Calculate the lag for the maximum correlation value
    lag = max_corr_index - len(waveform1) + 1
    
    # Calculate the phase difference for the maximum correlation value
    phase_difference = (lag / sampling_rate) * 2 * np.pi
    
    # Convert phase difference to degrees
    phase_difference_degrees = np.degrees(phase_difference)
    phase_difference_degrees = wrap_to_360(phase_difference_degrees)
    
    return correlation, correlation[max_corr_index], lag, phase_difference_degrees

# Calculates gain calibration values for each channel
def calcGainCal(data, maxValue):
    gainCal = []
    for i in range(len(data)):
        gainCal.append(maxValue / np.max(data[i]))
    return np.abs(gainCal)