---
description: "Use when working on pyRadar radar toolkit code, x-band hardware documentation, FMCW radar signal processing, ADAR1000 beamforming, AD9081 MxFE configuration, XUD1A converter setup, ADTR1107 front-end, RF chain analysis, or x-band phased array platform development."
applyTo: "x-band/pyRadar/**, docs/solutions/platforms/x_band/**"
---

# X-Band pyRadar Development Rules & System Reference

## Code Rules

### Cross-Platform Compatibility

- All code must be compatible with both **Windows and Linux**. Do not use OS-specific APIs without providing a cross-platform fallback (e.g., `keyboard` module has a Qt fallback for Linux non-root).

### Module Placement

Place new code in the correct module based on its concern. Do not mix responsibilities across modules.

| Concern | Module |
|---|---|
| Plotting / GUI | `radar_utils/radar_plotting.py` |
| Signal processing / DSP | `radar_utils/signal_processing.py` |
| Hardware init / MxFE / XUD1A / ADAR1000 config | `radar_utils/hardware_setup.py` |
| Target tracking / Kalman filter | `radar_utils/tracking.py` |
| Per-element calibration (gain/phase) | `radar_utils/calibration.py` |
| Calibration file I/O (JSON save/load) | `radar_utils/cal_manager.py` |
| TDD timing / chirp waveform / TX config | `radar_utils/sync_config.py` |
| FPGA Ethernet / network config | `radar_utils/network.py` |
| Window management / keyboard utilities | `radar_utils/utils.py` |
| New radar demo applications | `pilot_functions/` |
| ADAR1000 driver extensions | `custom_libs/adar1000.py` |
| Offline post-processing tools | `tools/` |

### Organization & Documentation

- **Favor functions**: Extract logic into reusable functions rather than leaving inline code blocks. Keep code organized and modular.
- **Document changes in code**: When code is added, removed, or changed, add inline comments describing the change. Include context such as what was modified and why.
- **Import convention**: Use `from radar_utils.module import function` for clear IDE navigation.

## Documentation Rules

These rules apply to all Markdown files in `docs/solutions/platforms/x_band/`.

### Scope

- Only reference **x-band platform** content. Do not reference other platforms (phaser, pluto, mako, etc.) unless explicitly comparing.

### Formatting

- Follow the **existing formatting and flow** of sibling documentation pages. Match heading levels, section ordering, and style.
- **Never use raw HTML**. Use MyST Markdown directives instead:
  - Admonitions: `` ```{note} ``, `` ```{warning} ``, `` ```{tip} ``, `` ```{important} ``, `` ```{caution} ``
  - Tabs: `` ````{tab-set} `` with `` ```{tab-item} Label ``
  - Tables: `` ```{list-table} `` (preferred) or pipe tables
  - Collapsible: `` ```{dropdown} ``
  - Code: `` ```{code-block} language ``
  - Math blocks: `` ```{math} `` with LaTeX syntax
  - Inline math: `$...$` (dollarmath enabled in `conf.py`)
- **Always use `{image}` for image files**:
  ```
  ```{image} path/to/image.png
  :alt: Description
  :width: 600px
  :align: center
  ```
  Never use `![alt](path)` Markdown image syntax or `<img>` HTML tags.
- Figures are auto-numbered (`numfig = True` in `conf.py`).

### Keeping Docs in Sync with Code

- Whenever there is a **new implementation of code or code is removed** in pyRadar, the corresponding documentation in `docs/solutions/platforms/x_band/` must be updated to reflect the changes. This includes:
  - New features or functions → add documentation
  - Removed features → remove or mark as deprecated in docs
  - Changed behavior → update descriptions and examples

## System Specifications

### Platform Overview

**X-Band Phased Array Development Platform** — A monostatic FMCW radar with 32-element analog beamforming and hybrid digital/analog architecture.

**Three main hardware boards:**

| Board | Part Number | Role |
|---|---|---|
| MxFE Evaluation Board | AD9081-FMCA-EBZ | Quad 12-bit ADC (4 GSPS) + Quad 16-bit DAC (12 GSPS) |
| X/C Band Up/Down Converter | ADXUD1AEBZ (XUD1A) | 4-channel frequency conversion (X-band ↔ C-band IF) |
| Analog Beamforming Board | ADAR1000EVAL1Z (Stingray) | 32-channel beamformer with ADTR1107 front-end ICs |

**FPGA**: Xilinx ZCU102 (Zynq UltraScale+)

### Frequency Plan

| Parameter | Value |
|---|---|
| Carrier Frequency | 10.4 GHz (X-Band) |
| LO Frequency | 14.9 GHz (ADF4371 PLL) |
| IF Frequency | ~4.5 GHz (C-Band, between XUD1A and AD9081) |
| FMCW Chirp Bandwidth | 250 MHz |
| Wavelength (λ) | 28.8 mm |
| RX Main NCO | 500 MHz |
| TX Main NCO | 4.5 GHz |
| RX Nyquist Zone | Odd |

### Antenna Array

- **Tiles**: 2× ADAR1000EVAL1Z-ANT with 10 GHz lattice spacing
- **Elements**: 32 total in a 4×8 planar grid
- **Element Spacing**: 15 mm (~λ/2 at 10.4 GHz)
- **Subarrays**: 4 subarrays of 8 elements each
- **Typical Operating Mode**: 3 subarrays RX (24 elements), 1 subarray TX (8 elements)

**Array Gain vs. Number of Elements:**

| Elements | Gain (dB) | Elements | Gain (dB) |
|---|---|---|---|
| 1 | 5.0 | 8 (1 subarray) | 14.0 |
| 16 (2 subarrays) | 17.0 | 24 (3 sub, typical RX) | 19.3 |
| 32 (full array) | 20.1 | | |

**Subarray-to-ADC Wiring:**

| Subarray | Role (typical) | ADC Channel | XUD1A Channel |
|---|---|---|---|
| Subarray 1 | RX | ADC 3 | Ch 1 (voltage3) |
| Subarray 2 | RX | ADC 1 | Ch 2 (voltage4) |
| Subarray 3 | TX | ADC 0 | Ch 3 (voltage2) |
| Subarray 4 | RX | ADC 2 | Ch 4 (voltage1) |

### Component Specifications

#### ADTR1107 — Front-End IC (32×, one per element)

6–18 GHz front-end with integrated PA, LNA, and reflective SPDT switch. Positioned between antenna elements and ADAR1000. 50 Ω matched I/O. Directional coupler on PA output for power detection.

| Parameter | TX Mode | RX Mode |
|---|---|---|
| Small-Signal Gain | +22 dB | +18 dB |
| Psat | 25 dBm | — |
| Noise Figure | — | 2.5 dB |

#### ADAR1000 — Beamformer IC (8×, 4 channels each)

8–16 GHz, 4-channel beamforming core. Half-duplex TX/RX. Values at nominal bias.

**Transmit (RF_IO → TX1–TX4):**

| Parameter | 9.5 GHz | 11.5 GHz |
|---|---|---|
| Max Single Channel Gain | 21 dB | 19 dB |
| Output P1dB | 10 dBm | 10 dBm |
| Psat | 14 dBm | 14 dBm |
| OIP3 | 20 dBm | 21 dBm |
| Channel-to-Channel Isolation | −40 dB | −40 dB |

**Receive (RX1–RX4 → RF_IO):**

| Parameter | 9.5 GHz | 11.5 GHz |
|---|---|---|
| Max Single Channel Gain | 10 dB | 9 dB |
| Max Electronic Gain | 16 dB | 15 dB |
| Max Coherent Gain | 22 dB | 21 dB |
| Noise Figure | 8 dB | 8 dB |
| Input P1dB | −16 dBm | −16 dBm |
| Input IP3 | −7 dBm | −7 dBm |

**Common:**

| Parameter | Value |
|---|---|
| VGA Gain Range | ≥31 dB (7-bit register, 0–127) |
| Gain Resolution | ≤0.5 dB |
| RMS Gain Error | 0.2 dB |
| Phase Adjustment Range | 360° (7-bit register, 0–127) |
| Phase Resolution | 2.8° |
| RMS Phase Error | 2° |
| Binary Attenuator | ~23 dB additional range |
| Power Detectors | 4 per IC, −20 to +10 dBm, 8-bit |
| TR Switching Time | 180 ns |

#### ZX10-2-183-S+ — 2:1 Power Combiner/Splitter (4×, Mini-Circuits)

One per subarray, between the 2 ADAR1000 RF outputs and the XUD1A SMA I/O. Insertion loss ~3.5 dB (3 dB theoretical + ~0.5 dB excess).

#### ADXUD1AEBZ (XUD1A) — Up/Down Converter

4-channel X-to-C band converter. Per-channel TX/RX mode switching. On-board IF bandpass filters (FL3/FL6/FL7/FL8).

**Internal signal chain per channel:** B096QC2S → ADRF5020 switch → HMC903 amps → EQY-6-24+ equalizer → LFCN-123+ filter → HMC652 buffers → HMC773A mixer → BFCN-5200+ filter → ADL8111 → HMC8411. TX path includes HMC383 PA.

**LO chain:** ADF4371 PLL → ADRF5020 → EP2K1+ splitters → HMC963 multiplier/amp, distributed to all 4 channels.

**Performance @ 10 GHz:**

| Parameter | RX High Gain | RX Low Gain | TX |
|---|---|---|---|
| Gain (dB) | +15.5 | +0.5 | −14.3 |
| Noise Figure (dB) | 15.5 | 18.3 | 27.4 |
| IIP3 / OIP3 (dBm) | IIP3 = 8.5 | IIP3 = 9.6 | OIP3 = 11.6 |

#### VLF-8400+ — RX IF Low-Pass Filter (4×, Mini-Circuits)

Between XUD1A IF output and AD9081 ADC inputs. Passband insertion loss ~1 dB.

#### VLF-5500+ — TX IF Low-Pass Filter (4×, Mini-Circuits)

Between AD9081 DAC outputs and XUD1A IF input. Passband insertion loss ~1 dB.

#### AD9081 — MxFE (Mixed-Signal Front End)

**ADC:** 4 channels, 12-bit, 4 GSPS max sample rate.

| Parameter | Value |
|---|---|
| Full-Scale Input | 1.4 V p-p |
| Noise Figure | 26.8 dB |
| Noise Density | −147.5 dBFS/Hz |
| Analog Input Bandwidth (−3 dB) | 7.5 GHz |

**DAC:** 4 channels, 16-bit, 12 GSPS max sample rate.

| Parameter | Value |
|---|---|
| Full-Scale Output Current | 6.43–37.75 mA |
| NSD (single-tone, 3.7 GHz) | −155.1 dBc/Hz |
| SFDR (single-tone, 3.7 GHz) | −70 dBc |
| Usable Analog Bandwidth | 8 GHz |

### RF Chain Gain Analysis

#### RX Chain (per element, single subarray, XUD1A high gain mode, ~10 GHz)

```
Antenna Element (5 dBi)
  → ADTR1107 RX         (+18 dB,   NF 2.5 dB)
  → ADAR1000 RX          (+10 dB,   NF 8 dB,    IP1dB −16 dBm)
  → ZX10-2-183-S+ 2:1    (−3.5 dB)
  → XUD1A RX High Gain   (+15.5 dB, NF 15.5 dB, IIP3 8.5 dBm)
  → VLF-8400+ LPF        (−1 dB)
  → AD9081 ADC            (NF 26.8 dB)
```

**Per-element chain gain (before array combining): +18 + 10 − 3.5 + 15.5 − 1 = +39 dB**

With 24-element RX array gain (~19.3 dB) the total effective gain is ~58.3 dB from element input to digitized output.

#### TX Chain (per element, single subarray, ~10 GHz)

```
AD9081 DAC
  → VLF-5500+ LPF        (−1 dB)
  → XUD1A TX              (−14.3 dB, OIP3 11.6 dBm)
  → ZX10-2-183-S+ 1:2    (−3.5 dB)
  → ADAR1000 TX           (+20 dB,   OP1dB 10 dBm)
  → ADTR1107 TX           (+22 dB,   Psat 25 dBm)
  → Antenna Element       (5 dBi)
```

**Per-element chain gain: −1 − 14.3 − 3.5 + 20 + 22 = +23.2 dB**

With 8-element TX subarray gain (~14 dB) the total effective EIRP gain is ~37.2 dB from DAC output.

### Key Radar Performance

| Parameter | Value | Formula |
|---|---|---|
| Range Resolution | 0.6 m | $\Delta R = c / (2 \cdot BW)$ |
| Chirps per CPI | 16 | — |
| Velocity Resolution | ~0.27 m/s | $\Delta v = \lambda / (2 \cdot N_{chirps} \cdot PRI)$ |
| Max Unambiguous Range | ~150 m | Buffer/geometry dependent |
| Max Unambiguous Velocity | $\lambda / (4 \cdot PRI)$ | — |
| Element Phase Step | 2.8° | ADAR1000 register |
| Half-Wave Spacing (λ/2) | 14.4 mm | Element spacing reference |

### Calibration System

- **Per-element calibration**: 32 elements, each with independent RX and TX phase/gain corrections
- **Polynomial VGA mapping**: `poly_atten1` (0–23 dB main path), `poly_atten0` (>23 dB with attenuator engaged)
- **Digital NCO phase correction**: AD9081 NCO provides milli-degree resolution for per-subarray digital alignment
- **Persistence**: JSON files with timestamps in `cal files/` directory, auto-purge of stale calibrations
- **Stored fields**: `rx_phase_cal`, `tx_phase_cal`, `rx_gain_dict`, `tx_gain_dict`, `rx_atten_dict`, `tx_atten_dict`, `cal_ant_fix` (4 subarray digital phases), `loFreq`, `sray_settings`
- **TX power detector**: External LTC2314-14 (14-bit ADC) for PA envelope measurement

### Signal Processing Defaults

From `rd_gui_settings.json`:

| Parameter | Default |
|---|---|
| Dynamic Range | 8.0 dB |
| CFAR | Enabled |
| MTI | Enabled |
| NCI Alpha | 0.3 |
| CFAR Guard Cells | [2, 2] |
| CFAR Training Cells | [8, 8] |
| CFAR Bias | 19.0 dB |
| Zero Range Bins | 16 |
| Max Display Range | 30.0 m |
| Velocity Bins | 6 |

### Datasheet References

| Component | Datasheet |
|---|---|
| ADAR1000 | [Data Sheet Rev B](https://www.analog.com/media/en/technical-documentation/data-sheets/ADAR1000.pdf) |
| ADTR1107 | [Data Sheet Rev C](https://www.analog.com/media/en/technical-documentation/data-sheets/adtr1107.pdf) |
| AD9081 MxFE | [Data Sheet Rev 0](https://www.analog.com/media/en/technical-documentation/data-sheets/ad9081.pdf) |
| ADF4371 PLL | [Data Sheet](https://www.analog.com/media/en/technical-documentation/data-sheets/adf4371.pdf) |
| ADXUD1AEBZ | [User Guide (wiki)](https://wiki.analog.com/resources/eval/user-guides/xud1a) |
| AD9081-FMCA-EBZ | [UG-1578 User Guide](https://www.analog.com/media/en/technical-documentation/user-guides/ad9081-ad9082-ug-1578.pdf) |