# =============================================================================
# signal_processing.py — DSP algorithms for FMCW phased-array radar
# =============================================================================
# This file contains all the signal-processing building blocks that turn raw
# ADC samples into target detections with range, velocity, and angle.
#
# ─────────────────────────────────────────────────────────────────────────────
#  FMCW Radar Fundamentals
# ─────────────────────────────────────────────────────────────────────────────
#
#  FMCW (Frequency-Modulated Continuous Wave):
#    The radar transmits a "chirp" — a signal whose instantaneous frequency
#    sweeps linearly from f0 to f0+BW over time T:
#        f_tx(t) = f0 + (BW/T) · t       for 0 ≤ t < T
#
#    A target at range R reflects this chirp back with a round-trip delay:
#        τ = 2R / c                      (c = speed of light ≈ 3×10⁸ m/s)
#
#    The receiver multiplies (mixes) the echo with the transmitted chirp.
#    Because both signals are chirps with the same slope (BW/T), their
#    instantaneous frequency difference is constant:
#        f_beat = (BW / T) · τ  =  2 · R · BW / (c · T)
#
#    This "beat tone" frequency is proportional to target range.  An FFT
#    of the beat signal converts time → frequency → range.
#
#    Range resolution (minimum separable distance between two targets):
#        R_res = c / (2 · BW)
#    With BW = 250 MHz:  R_res = 3e8 / (2 · 250e6) = 0.6 m
#
# ─────────────────────────────────────────────────────────────────────────────
#  Range-Doppler Processing (2-D FFT)
# ─────────────────────────────────────────────────────────────────────────────
#
#    Multiple chirps are collected into a 2-D matrix (chirps × samples).
#
#    "Fast-time" = sample index within one chirp.  FFT along this axis
#    resolves range.  Each bin k corresponds to:
#        R(k) = k · c / (2 · BW)   (after keeping the positive-frequency half)
#
#    "Slow-time" = chirp index across the coherent processing interval (CPI).
#    A moving target at radial velocity v adds a phase shift of
#        Δφ = 4π · v · PRI / λ
#    between consecutive chirps.  FFT along slow-time resolves this phase
#    progression into velocity bins:
#        v_res = λ / (2 · N · PRI)    (N = number of chirps)
#        v_max = λ / (4 · PRI)         (max unambiguous velocity)
#
#    The result is the Range-Doppler map — a 2-D image where each cell
#    (doppler_bin, range_bin) represents the radar return at that
#    (velocity, range) combination.  Targets appear as bright peaks.
#
# ─────────────────────────────────────────────────────────────────────────────
#  CFAR (Constant False Alarm Rate)
# ─────────────────────────────────────────────────────────────────────────────
#
#    A fixed threshold would produce many false alarms in high-noise regions
#    and miss weak targets in quiet regions.  CFAR adapts its threshold to
#    the local noise level so the probability of false alarm (P_fa) stays
#    constant across the entire map.
#
#    For each cell under test (CUT):
#      1. Skip the "guard cells" immediately around the CUT (so the target's
#         own energy doesn't inflate the noise estimate).
#      2. Average the "training cells" further out → this estimates the local
#         noise floor in dB.
#      3. Threshold = noise_estimate + bias_db.
#      4. If CUT > threshold → detection.
#
#    The guard/training/bias parameters trade off detection sensitivity
#    against false-alarm rate.
#
# ─────────────────────────────────────────────────────────────────────────────
#  Monopulse Angle Estimation
# ─────────────────────────────────────────────────────────────────────────────
#
#    Monopulse uses two overlapping sub-beams:
#      • Sum beam (Σ)        = sub_A + sub_B   (maximum gain on boresight)
#      • Difference beam (Δ) = sub_A − sub_B   (null on boresight)
#
#    For two subarrays with phase-centre spacing d, a target at angle θ off
#    boresight produces a complex ratio:
#        Δ/Σ = j · tan(π · d · sin(θ) / λ)
#
#    The imaginary part of Δ/Σ (the "discriminant") is a monotonic function
#    of θ within the main beam, so it can be inverted to recover θ.
#
#    This works in a single pulse (hence "mono-pulse") — no scanning needed.
#    The technique is applied independently for azimuth and elevation using
#    different subarray pairs.
#
#    Azimuth:   Σ_az = sub1 + sub4,  Δ_az = sub1 − sub4  (left/right pair)
#    Elevation: Σ_el = sub1 + sub2,  Δ_el = sub2 − sub1  (upper/lower pair)
#
# =============================================================================

import numpy as np
import scipy.fft as _spfft
from scipy.signal import resample_poly as _resample_poly

# ── Configurable FFT window ─────────────────────────────────────────────────
_fft_window_type = "none"

def set_fft_window(window_type):
    """Set the FFT window type used by all signal processing functions.

    Parameters
    ----------
    window_type : str
        "none", "hamming", "hanning", "blackman", or "kaiser"
    """
    global _fft_window_type
    _fft_window_type = window_type.strip().lower()

def _get_window(n):
    """Return a 1-D window array of length *n* based on the current setting."""
    if _fft_window_type in ("none", "rectangular"):
        return np.ones(n)
    elif _fft_window_type == "hamming":
        return np.hamming(n)
    elif _fft_window_type == "hanning":
        return np.hanning(n)
    elif _fft_window_type == "blackman":
        return np.blackman(n)
    elif _fft_window_type == "kaiser":
        return np.kaiser(n, beta=14)
    else:
        return np.ones(n)

def apply_range_normalization(rd_db, r_res, exponent=4, start_bin=5):
    """Compensate R^n path loss so distant targets are visible.

    In dB the received power falls as  n·10·log10(R).  This function adds
    a range-dependent correction curve to flatten the map:
        correction[k] = exponent · 10 · log10(k · r_res)
    Bins below *start_bin* are left untouched to avoid a log(0) singularity
    and to match the zero-range blanking already applied elsewhere.

    Parameters:
        rd_db:      2-D array (doppler_bins, range_bins) in dB
        r_res:      range resolution in metres
        exponent:   path-loss exponent (4 = two-way R^4 radar equation)
        start_bin:  first bin to normalise (bins 0..start_bin-1 unchanged)

    Returns:
        rd_norm: copy of rd_db with range normalisation applied
    """
    num_range_bins = rd_db.shape[1]
    correction = np.zeros(num_range_bins)
    bins = np.arange(start_bin, num_range_bins)
    correction[start_bin:] = exponent * 10.0 * np.log10(bins * r_res)
    # Subtract the correction at start_bin so the curve starts at 0 dB
    correction[start_bin:] -= correction[start_bin]
    return rd_db + correction[np.newaxis, :]


def rd_monopulse_angle(sum_cell, delta_cell, output_freq, d=0.06):
    """
    Compute monopulse angle estimate from a single Range-Doppler cell.

    For two subarrays separated by d, the monopulse ratio is:
        Δ/Σ = j · tan(π · d · sin(θ) / λ)
    The angle information is in the imaginary part (the discriminant),
    NOT in np.angle() which is always ±90° for a purely imaginary number.

    Parameters:
        sum_cell:   complex scalar – Σ channel value at the target cell
        delta_cell: complex scalar – Δ channel value at the target cell
        output_freq: float – carrier frequency in Hz
        d:          float – subarray phase-center spacing in metres

    Returns:
        angle_deg: float – estimated angle off boresight in degrees
    """
    c = 3e8
    wavelength = c / output_freq
    ratio = delta_cell / (sum_cell + 1e-12)
    # Discriminant: Im(Δ/Σ) = tan(π·d·sinθ/λ)
    discriminant = np.imag(ratio)
    # Invert: sinθ = (λ/(π·d)) · arctan(discriminant)
    sin_theta = (wavelength / (np.pi * d)) * np.arctan(discriminant)
    sin_theta = float(np.clip(sin_theta, -1, 1))
    angle_deg = np.rad2deg(np.arcsin(sin_theta))
    return angle_deg

def freq_process_complex_batch(data_list, use_window=True, mti_filter=False, mti_3pulse=False,
                               bg_profile=None, max_range_bins=None,
                               complex_waveform=False):
    """Batch Range-Doppler FFT: processes multiple subarrays in one NumPy call.

    Instead of looping over subarrays one at a time, the data is stacked into
    a 3-D tensor (K subarrays × chirps × samples) and all FFTs run together.
    This is significantly faster thanks to NumPy's vectorised FFT.

    Processing chain (same physics as freq_process, but batched):
      1. **DC removal** per chirp (subtract mean of each row).
         Reason: ADC and mixer offsets produce a large DC component that,
         after the range FFT, would appear as a strong peak at range bin 0.
         Subtracting the mean eliminates this artefact.

      2. **Fast-time decimation** (when max_range_bins is set).
         If the display only needs M range bins, we anti-alias-filter and
         downsample (decimate) the fast-time axis BEFORE the FFT so the
         FFT length is proportional to M instead of the raw ADC length.
         Range-bin spacing ΔR = c/(2·BW) is independent of FFT length, so
         resolution is unchanged — only the maximum unambiguous range
         decreases (which is fine since we truncate anyway).
         Typical speed-up: 65 536 → 2 048 samples = 32× shorter FFT.

      3. **Blackman window** along fast-time (samples axis).
         Reason: a finite-length FFT of an un-windowed signal produces
         spectral leakage — energy from a strong target spills into nearby
         bins, potentially masking weaker targets.  The Blackman window has
         very low sidelobes (~−58 dB) at the cost of a slightly wider main
         lobe (1.7× vs rectangular).  Coherent gain is compensated after
         the FFT to preserve the correct signal amplitude.

      4. **Range FFT** (axis=2 / fast-time) → range bins.
         Only the positive-frequency half is kept because the FMCW beat
         signal is real-valued after dechirp, so the spectrum is symmetric.
         Each bin k maps to range R(k) = k · R_res.

         **Range-bin truncation** (when max_range_bins is set):
         Any bins beyond max_range_bins are discarded after the FFT.
         With decimation active, only a small margin of extra bins exists,
         so this step is inexpensive.

      5. **Background subtraction** (optional).
         An EMA (exponential moving average) of the range profile across
         many frames estimates the static scene (cross-talk, walls).  This
         profile is complex-valued, so subtracting it cancels coherent
         static returns.  Moving targets have a different phase each frame
         and are *not* cancelled.

      6. **MTI 2-pulse canceller** (optional, axis=1 / slow-time).
         Computes np.diff(range_fft, axis=chirps).  Because stationary
         targets have zero Doppler shift, consecutive chirps see the same
         phase → their difference is zero, cancelling clutter.  Moving
         targets have a phase rotation between chirps and survive.
         The factor ×2.0 compensates the 6 dB loss from differencing.

      7. **Hann window** along slow-time (Doppler axis).
         Without windowing, the rectangular window's sidelobes are only
         −13 dB — strong zero-Doppler clutter leaks into adjacent velocity
         bins, burying slow-moving targets.  The Hann window pushes
         sidelobes to ~−32 dB, significantly improving visibility of
         pedestrian-speed targets at the cost of slightly coarser velocity
         resolution (1.5×).

      8. **Doppler FFT** (axis=1 / slow-time) → velocity bins.
         fftshift centres zero-velocity in the middle of the array.
         Positive Doppler = target approaching, negative = receding.

    Parameters:
        data_list:       list of K arrays, each (num_chirps, num_samples)
        use_window:      apply Blackman window along fast-time (recommended)
        mti_filter:      enable 2-pulse MTI canceller to suppress stationary clutter
        mti_3pulse:      enable 3-pulse MTI canceller (deeper clutter notch, ~40 dB)
        bg_profile:      complex mean range profile for inter-frame subtraction
        max_range_bins:  if set, truncate the range axis to this many bins right
                         after the range FFT.  All downstream processing (MTI,
                         windowing, Doppler FFT) operates on the truncated array,
                         which can be orders of magnitude smaller than the full
                         FFT output.  Set to int(max_range_m / r_res) + margin.
        complex_waveform: if True, keep full FFT spectrum (complex/analytic chirp
                         has no Hermitian symmetry).  If False (default), keep
                         only the positive-frequency half (real chirp).

    Returns:
        rd_list:  list of K complex Doppler-FFT matrices (doppler_bins, range_bins)
        rfm_list: list of K mean range-FFT vectors (used to update background EMA)
    """
    K = len(data_list)
    # Stack into (K, num_chirps, num_samples)
    stacked = np.stack(data_list, axis=0)

    # DC removal per chirp
    stacked = stacked - np.mean(stacked, axis=2, keepdims=True)

    # ── Fast-time decimation (before windowing & FFT) ────────────────
    # The full ADC buffer often contains far more samples per chirp than
    # the display range requires.  E.g. 65 536 samples → 32 768 range
    # bins, but only ~512 are ever displayed.  Computing a 65 536-pt FFT
    # and discarding 99 % of the bins wastes enormous CPU time.
    #
    # Instead we anti-alias-filter and downsample (decimate) the fast-
    # time axis so the FFT length matches the number of bins we need.
    #
    # Key insight: range-bin spacing ΔR = c / (2·BW) is independent of
    # FFT length, so decimation does NOT change range resolution — it
    # only reduces the maximum unambiguous range, which is acceptable
    # because we are already truncating to max_range_bins after the FFT.
    #
    # We target 4 × max_range_bins samples (2× oversample margin) and
    # round the decimation factor to a power of 2 for filter efficiency.
    # resample_poly applies a Kaiser-windowed FIR anti-aliasing filter
    # internally, preventing distant targets from aliasing into view.
    #
    # Typical speed-up: 65 536 → 2 048 samples = 32× shorter FFT,
    # cutting Range-FFT time from ~250 ms to ~5-10 ms.
    _dec_factor = 1
    if max_range_bins is not None and max_range_bins > 0:
        _target_len = max(256, 4 * max_range_bins)  # 2× oversample margin
        _dec_factor = stacked.shape[2] // _target_len
        if _dec_factor >= 2:
            _dec_factor = int(2 ** int(np.log2(_dec_factor)))
        else:
            _dec_factor = 1

    if _dec_factor > 1:
        stacked = _resample_poly(stacked, up=1, down=_dec_factor, axis=2)

    # Blackman window on (possibly decimated) fast-time axis.
    coherent_gain = 1.0
    if use_window:
        num_samples = stacked.shape[2]
        window = _get_window(num_samples)
        stacked = stacked * window  # broadcast (K, chirps, samples) * (samples,)
        coherent_gain = np.sum(window) / num_samples

    # Range FFT along fast-time (axis=2), keep positive half.
    # After decimation the FFT length is typically ~2 048 instead of
    # 65 536, so the 48 per-subarray×chirp FFTs finish in a few ms.
    # scipy.fft with workers=-1 further parallelises across CPU cores.
    range_fft = _spfft.fft(stacked, axis=2, workers=-1)
    if complex_waveform:
        # Complex dechirp: rx × conj(tx) produces beat at NEGATIVE frequency
        # (f_beat = −k·τ).  The negative-frequency bins sit in the second
        # half of the FFT output [N//2 … N-1].  We extract that half and
        # flip it so that range increases with bin index, matching the
        # real-chirp convention.  Result: same N//2 range bins, but free
        # of the image-frequency ambiguity present in real-valued processing.
        num_range_bins = range_fft.shape[2] // 2
        range_fft = np.flip(range_fft[:, :, num_range_bins:], axis=2)
    else:
        # Real chirp: spectrum is symmetric — keep only positive half.
        num_range_bins = range_fft.shape[2] // 2
        range_fft = range_fft[:, :, :num_range_bins]
    range_fft = range_fft / (coherent_gain + 1e-12)

    # ── Early range-bin truncation ───────────────────────────────────────
    # With large ADC buffers (e.g. 2^20 samples, 16 chirps → 65 536
    # samples/chirp → 32 768 range bins), the vast majority of bins lie
    # far beyond the radar's useful range.  Truncating now avoids carrying
    # those bins through MTI, Hann windowing, Doppler FFT, and beamforming.
    # Typical speedup: 50-100× for downstream operations.
    if max_range_bins is not None and max_range_bins < num_range_bins:
        range_fft = range_fft[:, :, :max_range_bins]
        num_range_bins = max_range_bins

    # Mean range profile per subarray
    range_fft_mean = np.mean(range_fft, axis=1)  # (K, num_range_bins)

    if bg_profile is not None:
        range_fft = range_fft - bg_profile[np.newaxis, np.newaxis, :]

    if mti_3pulse:
        # 3-pulse canceller: convolve each range bin along slow-time with [1, -2, 1].
        # This is equivalent to two cascaded 2-pulse cancellers and provides
        # ~40 dB clutter rejection at zero Doppler vs ~20 dB for 2-pulse.
        # The filter weights [1, -2, 1] have unity DC gain of 0 (perfect null)
        # and the ×3.0 scale compensates the ~9.5 dB processing loss.
        kernel = np.array([[[1, -2, 1]]], dtype=range_fft.dtype)  # (1,3,1)
        kernel = kernel.transpose(0, 1, 2)  # shape (1, 3, 1) for axis=1 convolution
        from scipy.signal import fftconvolve
        mti_output = fftconvolve(range_fft, np.array([1, -2, 1], dtype=range_fft.dtype).reshape(1, 3, 1),
                                 mode='valid', axes=1) * 3.0
        # Pad 2 lost chirps (valid mode loses kernel_len-1 = 2 rows)
        pad = np.zeros((K, 2, num_range_bins), dtype=range_fft.dtype)
        range_fft = np.concatenate([mti_output, pad], axis=1)
    elif mti_filter:
        mti_output = np.diff(range_fft, axis=1) * 2.0
        pad = np.zeros((K, 1, num_range_bins), dtype=range_fft.dtype)
        range_fft = np.concatenate([mti_output, pad], axis=1)

    # Hann window on slow-time (Doppler) axis.
    # A rectangular window lets zero-Doppler clutter leak into adjacent velocity
    # bins at only -13 dB, burying slow-moving targets.  Hann brings this to
    # -32 dB, significantly improving visibility of pedestrian-speed targets.
    num_chirps_dim = range_fft.shape[1]
    doppler_win = _get_window(num_chirps_dim)           # shape: (num_chirps,)
    coherent_gain_doppler = np.sum(doppler_win) / num_chirps_dim
    range_fft = range_fft * doppler_win[np.newaxis, :, np.newaxis]

    # Doppler FFT along slow-time (axis=1)
    doppler_fft = _spfft.fftshift(_spfft.fft(range_fft, axis=1, workers=-1), axes=1)
    doppler_fft = doppler_fft / (coherent_gain_doppler + 1e-12)

    # Split back into per-subarray results
    rd_list = [doppler_fft[k] for k in range(K)]
    rfm_list = [range_fft_mean[k] for k in range(K)]
    return rd_list, rfm_list

def freq_process(data, min_scale, max_scale, use_window=True, mti_filter=False, mti_3pulse=False,
                 bg_profile=None, complex_waveform=False):
    """Single-subarray Range-Doppler processing.

    Takes a 2-D burst matrix (chirps × samples) and returns the Range-Doppler
    map in dB.  This is the core DSP chain for FMCW radar:

      1. DC removal         — subtract per-chirp mean to eliminate DC offset
      2. Windowing           — Blackman window along fast-time (samples) axis
                               to suppress FFT sidelobes (~-58 dB)
      3. Range FFT           — FFT along fast-time; each bin corresponds to a
                               range determined by its beat frequency
      4. Background subtract — remove the EMA (exponential moving average)
                               of the range profile across frames; cancels
                               static cross-talk / leakage that is coherent
                               frame-to-frame while preserving moving targets
      5. MTI filter          — 2-pulse canceller: np.diff along slow-time;
                               subtracts consecutive chirps so stationary
                               targets (zero Doppler) cancel out
      6. Doppler FFT         — FFT along slow-time; resolves target velocity
      7. Convert to dB and clip

    Parameters:
        bg_profile: complex ndarray (num_range_bins,) or None.
            The EMA background; moving targets have random phase each frame
            so they survive subtraction.

    Returns:
        range_doppler_data: 2-D dB map clipped to [min_scale, max_scale]
        range_fft_mean:     mean range-FFT profile (caller uses this to update EMA)
    """
    # Remove DC offset from each chirp (per-sample mean along fast-time)
    data_dc_removed = data - np.mean(data, axis=1, keepdims=True)

    # Apply windowing to range dimension (fast-time) to reduce sidelobes
    # Window each chirp independently to avoid amplitude imbalance across chirps
    data_windowed = data_dc_removed
    coherent_gain = 1.0

    if use_window:
        # Apply window along range dimension (axis=1) for each chirp
        num_samples = data.shape[1]
        window = _get_window(num_samples)
        data_windowed = data_dc_removed * window  # Broadcasting: window applied to each chirp
        # Calculate coherent gain for proper FFT normalization
        coherent_gain = np.sum(window) / num_samples

    # Step 1: Range FFT (FFT along fast-time/samples dimension for each chirp)
    # data shape: (num_chirps, num_samples)
    range_fft = np.fft.fft(data_windowed, axis=1)

    if complex_waveform:
        # Complex dechirp beat is at negative frequency; extract second
        # half of FFT (negative-freq bins) and flip for ascending range.
        num_range_bins = range_fft.shape[1] // 2
        range_fft = np.flip(range_fft[:, num_range_bins:], axis=1)
    else:
        # Real chirp: only positive-frequency half (Hermitian symmetry)
        num_range_bins = range_fft.shape[1] // 2
        range_fft = range_fft[:, :num_range_bins]

    # Normalize by coherent gain to preserve signal levels
    range_fft = range_fft / (coherent_gain + 1e-12)

    # range_fft shape: (num_chirps, num_range_bins)

    # Compute the mean range-FFT across chirps (returned to caller for EMA background tracking)
    range_fft_mean = np.mean(range_fft, axis=0)  # shape: (num_range_bins,)

    # Inter-frame background subtraction: remove static cross-talk / leakage profile.
    # The background is the EMA of range_fft_mean across many frames.
    # Moving targets have varying phase each frame so they are NOT cancelled.
    if bg_profile is not None:
        range_fft = range_fft - bg_profile[np.newaxis, :]  # subtract from every chirp

    # Apply MTI filtering if enabled
    # MTI works in slow-time (across chirps) for each range bin
    if mti_3pulse:
        # 3-pulse canceller [1, -2, 1]: deeper clutter notch (~40 dB)
        from scipy.signal import fftconvolve
        mti_output = fftconvolve(range_fft, np.array([1, -2, 1], dtype=range_fft.dtype).reshape(3, 1),
                                 mode='valid', axes=0) * 3.0
        pad = np.zeros((2, range_fft.shape[1]), dtype=range_fft.dtype)
        range_fft = np.vstack([mti_output, pad])
    elif mti_filter:
        # Vectorized 2-pulse canceller: np.diff along slow-time axis
        mti_output = np.diff(range_fft, axis=0) * 2.0
        # Pad with zeros row to keep same shape for Doppler FFT
        range_fft = np.vstack([mti_output, np.zeros((1, range_fft.shape[1]), dtype=range_fft.dtype)])
    
    doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=0),axes=0)
    range_doppler_data = 20 *np.log10(np.abs(doppler_fft)+1e-6)

    range_doppler_data = np.clip(range_doppler_data, min_scale, max_scale)  # clip the data to control the spectrogram scale
    return range_doppler_data, range_fft_mean

def beat_calc(data, fs, T, BW):
    """Estimate beat frequency and target range from an FMCW beat signal.

    The "beat signal" is what you get after mixing (multiplying) the received
    echo with a copy of the transmitted chirp.  For a target at range R the
    beat tone sits at:
        f_beat = 2 · R · BW / (c · T)

    This function FFTs the beat signal, finds the peak frequency, and
    converts it to range using:  R = c · T · f_beat / (2 · BW)

    Parameters:
        data: complex beat signal (1-D array)
        fs:   sampling frequency (Hz)
        T:    chirp duration (seconds)
        BW:   chirp bandwidth (Hz)

    Returns:
        beat_freq:    peak beat frequency (Hz)
        R_calculated: estimated target range (metres)
        xf:           frequency axis (Hz)
        yf:           magnitude spectrum
    """
    import numpy as np
    c = 3e8
    N = len(data)

    # Windowing and DC removal
    window = _get_window(N)
    data = data * window
    data = data - np.mean(data)

    # FFT and coherent gain compensation
    cg = np.sum(window) / N
    yf = np.fft.fft(data)
    xf = np.fft.fftfreq(N, 1 / fs)

    xf = xf[:N // 2]
    yf = np.abs(yf[:N // 2]) / (cg + 1e-12)

    # Peak detection
    beat_freq = xf[np.argmax(yf)]

    # Range calculation
    R_calculated = (c * T * beat_freq) / (2 * BW)

    return beat_freq, R_calculated, xf, yf

def circular_shift_fft(yf, xf, cross_talk_freq):
    """Shift the FFT so the cross-talk tone moves to bin 0 (effectively removing it).

    Internal cross-talk (TX leaking into RX inside the board) appears as a
    strong spectral peak at a fixed frequency.  By circularly shifting the
    spectrum so that peak sits at DC, then zeroing the DC bin, we suppress
    the cross-talk without affecting the rest of the spectrum.

    Parameters:
        yf:              FFT magnitude array
        xf:              frequency axis
        cross_talk_freq: frequency of the internal cross-talk tone (Hz)

    Returns:
        yf_shifted: circularly shifted FFT magnitude
    """
    # Find the index of the cross-talk frequency
    shift_index = np.argmin(np.abs(xf - cross_talk_freq))
    
    # Perform circular shift on yf
    yf_shifted = np.roll(yf, -shift_index)
    
    return yf_shifted

def heatmap_gen(sum_data, iq, cross_talk_freq, r_centers, mag_floor_db,angle_idx, heatmap, k,T):
    """Generate one column of the azimuth-range heatmap (used by FMCW scan).

    Steps:
      1. **Dechirp** — multiply received data by the TX chirp replica (real × real).
         A low-pass filter rejects the sum-frequency product, leaving only
         the beat tone at a frequency proportional to target range.
      2. Run beat_calc to get the frequency spectrum.
      3. Circular-shift to remove the cross-talk tone and zero DC bins.
      4. Convert beat frequency axis to range: R = c · f / (2 · k)  where k = BW/T.
      5. Interpolate the spectrum onto the desired range grid (r_centers).
    """
    c = 3e8
    fs = 250e6
    BW = 250e6

    # ADC channels can differ by ±1 sample from the ideal chirp length; align.
    N = min(len(sum_data), len(iq))
    sum_data = sum_data[:N]
    iq = iq[:N]

    _iq_is_complex = np.iscomplexobj(iq)
    if _iq_is_complex:
        # Complex waveform: conjugate-multiply dechirp, no LPF needed
        xcal = sum_data * np.conj(iq)
    else:
        # Dechirp: real × real multiplication produces beat + sum-frequency terms.
        xcal = sum_data * iq

        # Low-pass filter to reject the sum-frequency product (cos*cos → beat + 2×carrier)
        S = np.fft.fft(xcal)
        freqs = np.fft.fftfreq(N, 1 / fs)
        max_beat_hz = 2 * BW * 150 / (c * T)   # ~150 m max range cutoff
        S[np.abs(freqs) > max_beat_hz] = 0
        xcal = np.fft.ifft(S).real

    beat_freq, R_calculated, xf, yf = beat_calc(xcal, fs, T, BW)

    yf = circular_shift_fft(yf, xf, cross_talk_freq)
    noise_floor = np.median(yf)
    yf[:2] = noise_floor

    # 9) Frequency -> range (r = c * f_b / (2k)) and bin via interpolation
    range_axis = (c * xf) / (2.0 * k)
    # Interpolate onto desired range grid; out-of-bounds -> noise floor
    spec_binned = np.interp(r_centers, range_axis, yf,
                            left=noise_floor, right=noise_floor)

    heatmap[:, angle_idx] = spec_binned
    return heatmap, yf, xf

def init_fft(N):
    fs = 250e6
    w = _get_window(N).astype(np.float32)
    cg = w.sum() / N  # coherent gain
    X = np.fft.fftfreq(N, 1.0 / fs)
    xf = X[:N // 2]
    return w, cg, xf


def normalize_heatmap_data(data, floor_db=-30.0, clip_db=10.0):
    """
    Normalize data using logarithmic scaling (dB) and clip to a defined range.
    Parameters:
        data: numpy array
            Raw magnitude data.
        floor_db: float
            Minimum dB value to clip to.
        clip_db: float
            Maximum dB value to clip to.
    Returns:
        numpy array
            Normalized data between 0 and 1.
    """
    # Avoid log(0) by adding a small epsilon
    epsilon = 1e-12
    data_db = 20 * np.log10(np.abs(data) + epsilon)
    data_db = np.clip(data_db, floor_db, clip_db)
    return (data_db - floor_db) / (clip_db - floor_db)


def RDRConfig(conv, PRI_ms, BW, num_chirps, signal_freq, output_freq):
    """Compute Range-Doppler radar configuration parameters.

    This function derives all the physical quantities needed by the RD
    processing loop from the fundamental radar parameters.  Every formula
    below comes directly from FMCW radar theory.

    Key equations:
      • Chirp slope:        k = BW / T_chirp          [Hz/s]
      • Range resolution:   R_res = c / (2 · BW)       [m]
        — depends *only* on bandwidth, not on carrier frequency.
        — wider BW → finer resolution (can separate closer targets).
      • Velocity resolution: v_res = λ / (2 · N · PRI)  [m/s]
        — more chirps (N) or longer dwell → finer velocity resolution.
      • Max unambiguous velocity: v_max = λ / (4 · PRI) [m/s]
        — higher PRF → higher v_max (but shorter max range).
      • Max unambiguous range: R_ua = c / (2 · PRF)    [m]
        — there is a fundamental trade-off between R_ua and v_max.

    Parameters:
        conv:         AD9081 converter (used to read the ADC sample rate)
        PRI_ms:       pulse repetition interval in milliseconds
        BW:           chirp bandwidth in Hz
        num_chirps:   number of chirps in one coherent processing interval
        signal_freq:  intermediate frequency (NCO) in Hz
        output_freq:  RF carrier frequency in Hz (used for wavelength λ)

    Returns:
        good_ramp_samples:     usable ADC samples per chirp
        start_offset_samples:  samples to skip at chirp start (retrace guard)
        N_frame:               total samples per PRI
        dist:                  range axis array (metres)
        R_res:                 range resolution (metres)
        v_res:                 velocity resolution (m/s)
        max_doppler_freq:      max Doppler frequency (Hz)
        max_doppler_vel:       max unambiguous velocity (m/s)
    """
    ramp_time_s = PRI_ms / 1000
    fs = conv.rx_sample_rate
    begin_offset_time = 0
    good_ramp_samples = int((ramp_time_s - begin_offset_time) * conv.rx_sample_rate)

    start_offset_time = begin_offset_time
    start_offset_samples = int(start_offset_time * conv.rx_sample_rate)

    PRI_s = PRI_ms / 1e3
    PRF = 1 / PRI_s

    N_frame = int(PRI_s * float(conv.rx_sample_rate))
    c = 3e8
    wavelength = c / output_freq
    slope = BW / ramp_time_s
    freq = np.linspace(-fs / 2, fs / 2, N_frame)
    dist = (freq - signal_freq) * c / (2 * slope)

    R_res = c / (2 * BW)
    v_res = wavelength / (2 * num_chirps * PRI_s)

    max_doppler_freq = PRF / 2
    max_doppler_vel = max_doppler_freq * wavelength / 2

    return good_ramp_samples, start_offset_samples, N_frame, dist, R_res, v_res, max_doppler_freq, max_doppler_vel


def apply_cfar_2d(rd_map_db, guard_cells=(1, 1), training_cells=(6, 4), bias_db=25.0):
    """1-D Cell-Averaging CFAR applied per Doppler bin on a Range-Doppler map.

    CFAR (Constant False Alarm Rate) adapts its detection threshold to the
    local noise floor so that the probability of a false alarm stays constant
    regardless of the noise level at each range bin.

    How it works (for each range bin r):
      1. Define a window around r:  [r-rg ... r-guard ... r ... r+guard ... r+rg]
         - guard_cells: bins immediately around the CUT that are excluded
           (to prevent the target's own energy from biasing the noise estimate)
         - training_cells: bins further out that estimate the local noise
      2. Average the training cells' dB values = noise_estimate
      3. Threshold = noise_estimate + bias_db
      4. If rd_map_db[r] > threshold → detection.

    The implementation uses cumulative sums (cumsum) instead of a Python loop
    over range bins for speed.

    Parameters:
        rd_map_db:      2-D dB Range-Doppler map (doppler_bins, range_bins)
        guard_cells:    (doppler, range) guard width — only range component used
        training_cells: (doppler, range) training width — only range component used
        bias_db:        dB above local noise mean to set threshold

    Returns:
        cfar_map_db:     RD map with non-detections suppressed to floor
        detections_mask: boolean 2-D mask of detected cells
    """
    _, range_guard = guard_cells
    _, range_train = training_cells

    rows, cols = rd_map_db.shape
    rg = range_guard + range_train
    num_train = 2 * range_train

    if range_train < 1 or rg >= cols:
        floor_db = float(np.min(rd_map_db))
        return np.full_like(rd_map_db, floor_db), np.zeros((rows, cols), dtype=bool)

    cs = np.cumsum(rd_map_db, axis=1)

    def _range_sum(a, b):
        s = cs[:, b - 1].copy()
        if a > 0:
            s -= cs[:, a - 1]
        return s

    detections = np.zeros((rows, cols), dtype=bool)

    for r in range(rg, cols - rg):
        left_sum = _range_sum(r - rg, r - range_guard)
        right_sum = _range_sum(r + range_guard + 1, r + rg + 1)
        threshold = (left_sum + right_sum) / num_train + bias_db
        detections[:, r] = rd_map_db[:, r] > threshold

    floor_db = float(np.min(rd_map_db))
    cfar_map_db = np.where(detections, rd_map_db, floor_db)
    return cfar_map_db, detections


def compute_monopulse_angles(rd_sum_az, rd_delta_az, rd_sum_el, rd_delta_el,
                             peak_d, peak_r, output_freq, d_az=0.06, d_el=0.03):
    """Compute azimuth and elevation angles at a detected Range-Doppler cell.

    Monopulse uses two overlapping beams (Sum Σ and Difference Δ) to estimate
    a target's angle in a single measurement (no scanning required).  The ratio
    Δ/Σ is a monotonic function of angle within the main beam, so inverting it
    yields the target's angular offset from boresight.

    A guard check ensures the Σ channel has enough energy (> 4× noise floor)
    before trusting the angle estimate; otherwise it returns 0° to avoid
    erratic outputs from noise-dominated cells.

    Parameters:
        rd_sum_az, rd_delta_az: complex RD maps for azimuth Σ/Δ channels
        rd_sum_el, rd_delta_el: complex RD maps for elevation Σ/Δ channels
        peak_d, peak_r:         Doppler and range bin indices of the detection
        output_freq:            carrier frequency (Hz) — used to compute λ
        d_az:                   azimuth subarray phase-center spacing (m)
        d_el:                   elevation subarray phase-center spacing (m)

    Returns:
        (az_deg, el_deg): monopulse angle estimates in degrees
    """
    sum_az_cell = rd_sum_az[peak_d, peak_r]
    delta_az_cell = rd_delta_az[peak_d, peak_r]
    sum_el_cell = rd_sum_el[peak_d, peak_r]
    delta_el_cell = rd_delta_el[peak_d, peak_r]

    sigma_mag_az = float(np.abs(sum_az_cell))
    sigma_mag_el = float(np.abs(sum_el_cell))
    sigma_noise_az = float(np.percentile(np.abs(rd_sum_az), 25))
    sigma_noise_el = float(np.percentile(np.abs(rd_sum_el), 25))
    sigma_min_az = sigma_noise_az * 4.0
    sigma_min_el = sigma_noise_el * 10.0   # higher guard — 2 el subarrays are noisy

    wavelength = 3e8 / output_freq
    max_mono_az = np.rad2deg(np.arcsin(min(wavelength / (2 * d_az), 1.0)))
    max_mono_el = np.rad2deg(np.arcsin(min(wavelength / (2 * d_el), 1.0)))

    if sigma_mag_az > sigma_min_az:
        az_deg = rd_monopulse_angle(sum_az_cell, delta_az_cell, output_freq, d=d_az)
        az_deg = float(np.clip(az_deg, -max_mono_az, max_mono_az))
    else:
        az_deg = 0.0

    if sigma_mag_el > sigma_min_el:
        el_deg = rd_monopulse_angle(sum_el_cell, delta_el_cell, output_freq, d=d_el)
        el_deg = float(np.clip(el_deg, -max_mono_el, max_mono_el))
    else:
        el_deg = 0.0

    return az_deg, el_deg