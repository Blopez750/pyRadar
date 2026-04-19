import numpy as np
import math
from scipy.signal import chirp
from .tx_rx_cal import *


TX_OFFLOAD_SYNC = 0     # DMA trigger: push TX waveform from host to DAC
RX_OFFLOAD_SYNC = 1     # DMA trigger: pull RX samples from ADC to host
TDD_ENABLE      = 2     # Master enable for the TDD timing engine
RX_MXFE_EN      = 3     # AD9081 RX path enable
TX_MXFE_EN      = 4     # AD9081 TX path enable
TX_STINGRAY_EN  = 5     # Stingray PA (power amplifier) enable


# Maps subarray number (1-4) to AD9081 DAC channel index (0-3)
_SUBARRAY_TO_DAC = {1: 3, 2: 2, 3: 1, 4: 0}

def sys_sync(conv, tddn, PRF, num_chirps, BW, duty_cycle, mode, subarray_modes=None, complex_waveform=False):
    """Configure the FPGA timing engine (TDDN) and load the TX waveform.

    The TDDN (Time-Division Duplex Network) is a hardware timing engine on
    the ZCU102 FPGA that precisely synchronises the transmit and receive
    paths.  It generates 6 timing channels that control:
      • When the DAC starts transmitting the chirp waveform
      • When the ADC starts capturing receive samples
      • When the PA (power amplifier) on the Stingray board is enabled

    For FMCW mode (duty_cycle = 1.0), TX and RX are on simultaneously for
    the entire frame.  The DAC continuously replays the chirp waveform
    (cyclic buffer) while the ADC captures one buffer of
    num_chirps × samples_per_chirp samples.

    For Pulsed mode (duty_cycle < 1.0), the TX is gated on/off with the
    specified duty cycle, and the ADC captures during the full PRI.

    This function also generates the baseband chirp waveform:
        iq(t) = A · chirp(t, f0=0, f1=BW, T=PRI)
    The chirp sweeps from 0 to BW Hz over one PRI.  It is tiled num_chirps
    times and loaded into the DAC.  The XUD1A up-converter then translates
    this baseband chirp to the X-band carrier frequency.

    Parameters:
        conv:            AD9081 converter object
        tddn:            TDDN timing engine object
        PRF:             pulse repetition frequency (Hz) = 1/PRI
        num_chirps:      number of chirps per buffer
        BW:              chirp bandwidth (Hz)
        duty_cycle:      TX duty cycle (1.0 for FMCW, < 1.0 for pulsed)
        mode:            "FMCW", "CW", or "Pulsed"
        subarray_modes:  dict {1:"rx",...} — used to route the chirp only
                         to TX DAC channels, silence RX channels
        complex_waveform: if True, generate an analytic (complex IQ) chirp
                         exp(j·2π·f(t)·t) instead of a real-valued chirp.
                         Complex waveforms use both I and Q DAC channels and
                         eliminate the image frequency, giving full unambiguous
                         bandwidth.

    Returns:
        iq: the baseband chirp waveform array (tiled num_chirps times)
    """
    tddn.enable = 0
    start_freq = 0
    end_freq = BW
    PRI_ms = 1000 / PRF
    ramp_time_s = PRI_ms / 1000
    fs = conv.rx_sample_rate
    conv.tx_cyclic_buffer = True
    conv.rx_cyclic_buffer = False
    conv.tx_ddr_offload   = False
    print("\t --> ",PRI_ms)
    print("\t --> ",ramp_time_s)
    t = np.linspace(0, ramp_time_s, int(fs * ramp_time_s), endpoint=False)
    A = 2**14  # AD9081 conventional full scale (Q1.15 NCO — values above 2^14 clip)
    print("\t --> Operation Mode: ",mode)
    if mode == "FMCW" or mode == "fmcw":
        conv.rx_buffer_size = int((PRI_ms / 1000) * conv.rx_sample_rate)*num_chirps
        tddn.burst_count          = 0
        frame_length_ms          = conv.rx_buffer_size / conv.rx_sample_rate * 1e3
        rx_time                   = frame_length_ms
        tx_time                   = frame_length_ms

        ######################################################

        N_tx                      = int((tx_time * conv.rx_sample_rate) / 1e3)
        N_rx                      = conv.rx_buffer_size
        conv.rx_buffer_size       = N_rx
        tddn.startup_delay_ms     = 0
        tddn.frame_length_raw     = conv.rx_buffer_size
        tddn.sync_external        = 1


        for chan in [TX_MXFE_EN,TDD_ENABLE,RX_MXFE_EN,TX_STINGRAY_EN]:
            tddn.channel[chan].on_ms    = 0
            tddn.channel[chan].off_ms   = 0
            tddn.channel[chan].polarity = 0
            tddn.channel[chan].enable   = 1
        
        for chan in [RX_OFFLOAD_SYNC]:
            tddn.channel[chan].on_raw   = 399
            tddn.channel[chan].off_raw  = 400
            tddn.channel[chan].polarity = 0
            tddn.channel[chan].enable   = 1
        for chan in [TX_OFFLOAD_SYNC]:
            tddn.channel[chan].on_raw   = 0
            tddn.channel[chan].off_raw  = 1
            tddn.channel[chan].polarity = 0
            tddn.channel[chan].enable   = 1
        tddn.enable = 1
            
        conv.tx_destroy_buffer()
        if complex_waveform:
            # Complex (analytic) chirp: exp(j·2π·∫f(t)dt) where f sweeps 0→BW
            # Phase integral of a linear chirp: φ(t) = 2π·(f0·t + 0.5·(f1-f0)/T·t²)
            phase = 2 * np.pi * (start_freq * t + 0.5 * (end_freq - start_freq) / ramp_time_s * t**2)
            iq = A * np.exp(1j * phase)
        else:
            iq = A * chirp(t, f0=start_freq, f1=end_freq, t1=ramp_time_s, method='linear')
        iq = np.tile(iq, num_chirps)
        print(iq.shape)
    elif mode == "CW" or mode == "cw":
        conv.rx_buffer_size = int((PRI_ms / 1000) * conv.rx_sample_rate)
        tddn.burst_count         = 0
        frame_length_ms          = conv.rx_buffer_size * 1e3 / conv.rx_sample_rate
        rx_time                  = frame_length_ms
        tx_time                  = frame_length_ms

        ######################################################

        N_tx                      = int((tx_time * conv.tx_sample_rate) / 1000)
        N_rx                      = int((rx_time * conv.rx_sample_rate) / 1000)
        conv.rx_buffer_size       = N_rx
        tddn.startup_delay_ms     = 0
        tddn.frame_length_raw     = int(frame_length_ms/1e3 * 250e6)
        tddn.sync_external        = 1


        for chan in [TX_MXFE_EN,TDD_ENABLE,RX_MXFE_EN,TX_STINGRAY_EN]:
            tddn.channel[chan].on_ms    = 0
            tddn.channel[chan].off_ms   = 0
            tddn.channel[chan].polarity = 0
            tddn.channel[chan].enable   = 1

        for chan in [TX_OFFLOAD_SYNC,RX_OFFLOAD_SYNC]:
            tddn.channel[chan].on_raw   = 0
            tddn.channel[chan].off_raw  = 1
            tddn.channel[chan].polarity = 0
            tddn.channel[chan].enable   = 1

        tddn.enable = 1
        i = A * np.cos(2 * math.pi * BW * t)
        q = A * np.sin(2 * math.pi * BW * t)
        iq = i + 1j * q
        tddn.sync_soft = 1
        
        
    elif mode == "Pulsed" or mode == "pulsed":
        conv.rx_buffer_size = int((PRI_ms / 1000) * conv.rx_sample_rate*num_chirps)
        tddn.burst_count          = 0
        frame_length_ms          = conv.rx_buffer_size / conv.rx_sample_rate * 1e3 / num_chirps
        rx_time                  = conv.rx_buffer_size / conv.rx_sample_rate * 1e3
        tx_time     = frame_length_ms*duty_cycle

        ######################################################

        N_tx                      = int((tx_time * conv.tx_sample_rate) / 1e3)
        N_rx                      = conv.rx_buffer_size
        tddn.startup_delay_ms     = 0
        tddn.frame_length_raw     = int(frame_length_ms/1e3 * 250e6)
        tddn.sync_external        = 1



        for chan in [TX_MXFE_EN,TDD_ENABLE,RX_MXFE_EN,TX_STINGRAY_EN]:
            tddn.channel[chan].on_ms    = 0
            tddn.channel[chan].off_ms   = tx_time
            tddn.channel[chan].polarity = 0
            tddn.channel[chan].enable   = 1

        for chan in [TX_OFFLOAD_SYNC,RX_OFFLOAD_SYNC]:
            tddn.channel[chan].on_raw   = 0
            tddn.channel[chan].off_raw  = 1
            tddn.channel[chan].polarity = 0
            tddn.channel[chan].enable   = 1

        tddn.enable = 1
        A = 2**14
        iq = A*chirp(t, f0=start_freq, f1=end_freq, t1=ramp_time_s, method='linear')
    else:
        raise ValueError("Invalid mode specified. Use 'FMCW', 'CW', 'Pulsed'.")

    print(f"\t --> TX buffer length:          {N_tx}")
    print(f"\t --> Generated signal time[ms]: {tx_time}")
    print(f"\t --> RX_recieve time[ms]:       {((1/conv.rx_sample_rate) * conv.rx_buffer_size)*1000}")
    print(f"\t --> TDD_frame time[ms]:        {tddn.frame_length_ms}")
    print(f"\t --> TDD_frame time[raw]:       {tddn.frame_length_raw}")
    print(f"\t --> RX buffer length:          {conv.rx_buffer_size}")
    print(f"\t --> RX Sampling_rate:          {conv.rx_sample_rate}")
    print(f"\t --> PRI_ms:                    {PRI_ms}")

    # Always transmit on all 4 channels; send zeros on non-TX channels to silence them.
    # This avoids the AD9081 defaulting to an internal CW tone when the buffer is destroyed.
    if subarray_modes is not None:
        tx_channels = set([_SUBARRAY_TO_DAC[sa] for sa, m in subarray_modes.items() if m.lower() == "tx"])
    else:
        tx_channels = {0, 1, 2, 3}  # fallback: all channels active
    zeros = np.zeros_like(iq)
    tx_data = [iq if ch in tx_channels else zeros for ch in range(4)]
    print(f"\t --> TX DAC channels (chirp):   {sorted(tx_channels)}")
    conv.tx_enabled_channels = [0, 1, 2, 3]
    conv.tx(tx_data)

    return iq

def sync_disable(conv,tddn,sray,subarray):
    """Shut down the radar hardware gracefully.

    Disables the TDD timing engine, destroys the TX and RX DMA buffers,
    and disables all Stingray antenna elements.  This is called after every
    pilot demo to leave the hardware in a safe idle state.
    """
    tddn.enable = 0
    for chan in [TX_OFFLOAD_SYNC, RX_OFFLOAD_SYNC, TDD_ENABLE, RX_MXFE_EN, TX_MXFE_EN, TX_STINGRAY_EN]:
        tddn.channel[chan].on_ms    = 0
        tddn.channel[chan].off_ms   = 0
        tddn.channel[chan].polarity = 0
        tddn.channel[chan].enable   = 1
    tddn.enable = 1
    tddn.enable = 0
    conv.tx_destroy_buffer()
    conv.rx_destroy_buffer()
    disable_stingray_channel(sray, subarray)
    print("done")