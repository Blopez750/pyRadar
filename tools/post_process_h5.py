"""
Post-processing script for HDF5 drone test data.
Loads raw ADC data from 3 subarray channels and displays them.
"""

import h5py
import numpy as np
import json
import os
import time
import sys
from scipy.signal import chirp

# Add parent directory to path so radar_utils is importable from tools/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from radar_utils.signal_processing import freq_process

def post_process_position(h5_filepath, capture_idx=0, plot_mode='adc', use_mti=True, zero_range_bins=0, max_range_m=100):
    """
    Post-process raw ADC data from HDF5 file with multiple visualization modes.
    
    Parameters:
    -----------
    h5_filepath : str
        Path to the HDF5 file to process
    capture_idx : int
        Index of the capture to display (default: 0)
    plot_mode : str
        Visualization mode: 'adc' for raw ADC data, 'range_doppler' for Range-Doppler heatmap (default: 'adc')
    use_mti : bool
        Enable MTI (Moving Target Indicator) filtering for range_doppler mode (default: True)
    zero_range_bins : int
        Number of range bins to zero out (starting from bin 0) to remove near-field clutter (default: 0)
    max_range_m : float
        Maximum range in meters to display on Range-Doppler heatmap (default: 100)
    """
    
    print("="*60)
    print(f"POST-PROCESSING HDF5 DATA - {plot_mode.upper()} MODE")
    if plot_mode == 'range_doppler':
        print(f"MTI Filter: {'ENABLED' if use_mti else 'DISABLED'}")
    print("="*60)
    print(f"Loading: {h5_filepath}")
    
    # Try to load JSON parameters file
    h5_dir = os.path.dirname(h5_filepath)
    h5_basename = os.path.basename(h5_filepath)
    json_filename = h5_basename.replace('.h5', '_parameters.json')
    json_filepath = os.path.join(h5_dir, json_filename)
    
    if os.path.exists(json_filepath):
        print(f"Loading parameters from: {json_filepath}")
        with open(json_filepath, 'r') as json_file:
            params = json.load(json_file)
        
        print("\n" + "="*60)
        print("TEST CONDITIONS")
        print("="*60)
        print(f"Timestamp: {params['test_info']['timestamp']}")
        print(f"Folder: {params['test_info']['folder_name']}")
        print(f"Filename: {params['test_info']['filename']}")
        print(f"\nBeam Steering:")
        print(f"  Azimuth: {params['beam_steering']['azimuth_deg']}°")
        print(f"  Elevation: {params['beam_steering']['elevation_deg']}°")
        print(f"\nRadar Config:")
        print(f"  Num Chirps: {params['radar_config']['num_chirps']}")
        print(f"  PRF: {params['radar_config']['PRF_Hz']} Hz")
        print(f"  Bandwidth: {params['radar_config']['BW_Hz']/1e6} MHz")
        print(f"  Output Freq: {params['radar_config']['output_freq_Hz']/1e9} GHz")
        print(f"  RX Buffer Size: {params['radar_config']['rx_buffer_size']}")
        print(f"\nResolutions:")
        print(f"  Range: {params['resolutions']['range_resolution_m']:.3f} m")
        print(f"  Velocity: {params['resolutions']['velocity_resolution_mps']:.3f} m/s")
        print(f"\nCapture Info:")
        print(f"  Requested: {params['capture_info']['num_captures_requested']}")
        print(f"  Actual: {params['capture_info']['num_captures_actual']}")
        print(f"\nSubarrays:")
        print(f"  Modes: {params['subarrays']['modes']}")
        print(f"  Active RX: {params['subarrays']['active_rx_subarrays']}")
        print("="*60)
    else:
        print(f"Warning: JSON parameters file not found at {json_filepath}")
    
    # Open HDF5 file and read data
    with h5py.File(h5_filepath, 'r') as f:
        # Print available datasets
        print(f"\nAvailable datasets in HDF5:")
        for key in f.keys():
            print(f"  {key}: shape {f[key].shape}")
        
        # Print HDF5 metadata
        print(f"\nHDF5 Metadata:")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")
        
        num_captures = f.attrs.get('num_captures', f['raw_adc_data'].shape[0])
        
        # Validate capture index
        if capture_idx >= num_captures:
            print(f"\nWarning: capture_idx {capture_idx} >= num_captures {num_captures}")
            capture_idx = 0
            print(f"Using capture_idx = {capture_idx}")
        
        print(f"\nDisplaying capture index: {capture_idx} of {num_captures}")
        
        # Pre-load ALL capture data into memory for fast access
        print(f"\nPre-loading all {num_captures} captures into memory...")
        t_preload_start = time.perf_counter()
        all_subarray1 = f['subarray1_data'][:]
        all_subarray2 = f['subarray2_data'][:]
        all_subarray4 = f['subarray4_data'][:]
        all_sum_data = f['raw_adc_data'][:]
        t_preload = time.perf_counter() - t_preload_start
        print(f"Pre-load complete in {t_preload:.2f}s")
        
        # Get initial data
        subarray1 = all_subarray1[capture_idx]
        subarray2 = all_subarray2[capture_idx]
        subarray4 = all_subarray4[capture_idx]
        sum_data = all_sum_data[capture_idx]
        
        print(f"\nData loaded:")
        print(f"  Subarray 1: {len(subarray1)} samples per capture")
        print(f"  Subarray 2: {len(subarray2)} samples per capture")
        print(f"  Subarray 4: {len(subarray4)} samples per capture")
        print(f"  Sum: {len(sum_data)} samples per capture")
        print(f"  Total memory: ~{(all_subarray1.nbytes + all_subarray2.nbytes + all_subarray4.nbytes + all_sum_data.nbytes) / 1024**2:.1f} MB")
        
        # ==================== PLOTTING SETUP ====================
        if plot_mode == 'adc':
            plot_adc_data(all_subarray1, all_subarray2, all_subarray4, all_sum_data, 
                         num_captures, capture_idx, h5_filepath)
        elif plot_mode == 'range_doppler':
            if not os.path.exists(json_filepath):
                print(f"\nError: JSON parameters file required for range_doppler mode")
                print(f"Expected: {json_filepath}")
                return
            plot_range_doppler(all_sum_data, params, num_captures, capture_idx, 
                             h5_filepath, use_mti, zero_range_bins, max_range_m)
        else:
            print(f"Error: Unknown plot_mode '{plot_mode}'. Use 'adc' or 'range_doppler'")
            return


def plot_adc_data(all_subarray1, all_subarray2, all_subarray4, all_sum_data, 
                 num_captures, capture_idx, h5_filepath):
    """Plot raw ADC data from all subarrays."""
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets, QtCore
    
    # Get initial data
    subarray1 = all_subarray1[capture_idx]
    subarray2 = all_subarray2[capture_idx]
    subarray4 = all_subarray4[capture_idx]
    sum_data = all_sum_data[capture_idx]
    
    # ==================== PYQTGRAPH SETUP ====================
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    
    win = pg.GraphicsLayoutWidget(show=True)
    win.resize(1600, 1000)
    win.setWindowTitle(f"Raw ADC Data - {os.path.basename(h5_filepath)}")
    
    # Row 0: Subarray 1
    plot1 = win.addPlot(row=0, col=0, title=f"Subarray 1 (Real) - Capture {capture_idx}")
    plot1.setLabel('left', 'ADC Codes')
    plot1.setLabel('bottom', 'Sample')
    plot1.showGrid(x=True, y=True, alpha=0.3)
    curve1 = plot1.plot(np.real(subarray1), pen=pg.mkPen('r', width=1))
    
    # Row 0: Subarray 2
    plot2 = win.addPlot(row=0, col=1, title=f"Subarray 2 (Real) - Capture {capture_idx}")
    plot2.setLabel('left', 'ADC Codes')
    plot2.setLabel('bottom', 'Sample')
    plot2.showGrid(x=True, y=True, alpha=0.3)
    curve2 = plot2.plot(np.real(subarray2), pen=pg.mkPen('g', width=1))
    
    # Row 1: Subarray 4
    plot4 = win.addPlot(row=1, col=0, title=f"Subarray 4 (Real) - Capture {capture_idx}")
    plot4.setLabel('left', 'ADC Codes')
    plot4.setLabel('bottom', 'Sample')
    plot4.showGrid(x=True, y=True, alpha=0.3)
    curve4 = plot4.plot(np.real(subarray4), pen=pg.mkPen('b', width=1))
    
    # Row 1: Sum of all subarrays
    plot_sum = win.addPlot(row=1, col=1, title=f"Sum of All Subarrays (Real) - Capture {capture_idx}")
    plot_sum.setLabel('left', 'ADC Codes')
    plot_sum.setLabel('bottom', 'Sample')
    plot_sum.showGrid(x=True, y=True, alpha=0.3)
    curve_sum = plot_sum.plot(np.real(sum_data), pen=pg.mkPen('c', width=1))
    
    # State variable for current capture
    current_capture = [capture_idx]
    
    def update_plot():
        """Update all plots with data from next capture"""
        t_start = time.perf_counter()
        
        current_capture[0] = (current_capture[0] + 1) % num_captures
        idx = current_capture[0]
        
        # Load new data from pre-loaded memory arrays
        t_load_start = time.perf_counter()
        new_sub1 = all_subarray1[idx]
        new_sub2 = all_subarray2[idx]
        new_sub4 = all_subarray4[idx]
        new_sum = all_sum_data[idx]
        t_load = time.perf_counter() - t_load_start
        
        # Update curves
        t_curves_start = time.perf_counter()
        curve1.setData(np.real(new_sub1))
        curve2.setData(np.real(new_sub2))
        curve4.setData(np.real(new_sub4))
        curve_sum.setData(np.real(new_sum))
        t_curves = time.perf_counter() - t_curves_start
        
        # Update titles
        t_titles_start = time.perf_counter()
        plot1.setTitle(f"Subarray 1 (Real) - Capture {idx}")
        plot2.setTitle(f"Subarray 2 (Real) - Capture {idx}")
        plot4.setTitle(f"Subarray 4 (Real) - Capture {idx}")
        plot_sum.setTitle(f"Sum of All Subarrays (Real) - Capture {idx}")
        t_titles = time.perf_counter() - t_titles_start
        
        t_total = time.perf_counter() - t_start
        
        print(f"Capture {idx + 1}/{num_captures} | Total: {t_total*1000:.2f}ms | Load: {t_load*1000:.2f}ms | Curves: {t_curves*1000:.2f}ms | Titles: {t_titles*1000:.2f}ms")
    
    # Create timer for automatic updates (as fast as possible)
    timer = QtCore.QTimer()
    timer.timeout.connect(update_plot)
    timer.start(0)  # 0ms = update as fast as possible
    
    print("\n" + "="*60)
    print("DISPLAY READY - Looping through captures as fast as possible")
    print("Close window to exit")
    print("="*60)
    
    # Run the application
    app.exec_()


def plot_range_doppler(all_sum_data, params, num_captures, capture_idx, h5_filepath, use_mti, zero_range_bins=0, max_range_m=100):
    """Plot Range-Doppler heatmap from radar data."""
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets, QtCore
    
    # Extract radar parameters
    num_chirps = params['radar_config']['num_chirps']
    PRF = params['radar_config']['PRF_Hz']
    BW = params['radar_config']['BW_Hz']
    output_freq = params['radar_config']['output_freq_Hz']
    rx_buffer_size = params['radar_config']['rx_buffer_size']
    
    # Calculate processing parameters
    samples_per_chirp = rx_buffer_size // num_chirps
    r_res = params['resolutions']['range_resolution_m']
    v_res = params['resolutions']['velocity_resolution_mps']
    
    # Processing parameters (matching drone_test_procedure)
    min_scale = 180
    max_scale = 215
    
    # Generate ideal IQ chirp for dechirping (mixing)
    # This is critical for FMCW radar processing
    PRI_ms = 1000 / PRF
    ramp_time_s = PRI_ms / 1000
    # Assume sample rate from buffer size (standard is 250 MHz)
    fs = samples_per_chirp / ramp_time_s
    t = np.linspace(0, ramp_time_s, samples_per_chirp, endpoint=False)
    A = 2**15
    start_freq = 0
    end_freq = BW
    iq_chirp = A * chirp(t, f0=start_freq, f1=end_freq, t1=ramp_time_s, method='linear')
    
    print(f"\nRange-Doppler Processing Parameters:")
    print(f"  Num Chirps: {num_chirps}")
    print(f"  Samples per Chirp: {samples_per_chirp}")
    print(f"  Sample Rate: {fs/1e6:.1f} MHz")
    print(f"  Range Resolution: {r_res:.3f} m")
    print(f"  Velocity Resolution: {v_res:.3f} m/s")
    print(f"  MTI Filter: {'ENABLED' if use_mti else 'DISABLED'}")
    print(f"  Scale Range: {min_scale} to {max_scale} dB")
    print(f"  IQ Chirp Length: {len(iq_chirp)} samples")
    print(f"  Max Range Display: {max_range_m} m")
    if zero_range_bins > 0:
        zeroed_range_m = zero_range_bins * r_res
        print(f"  Zeroing first {zero_range_bins} range bins ({zeroed_range_m:.1f} m) to remove near-field clutter")
    if use_mti:
        print(f"\n  NOTE: MTI filters out static targets - only moving objects will appear!")
        print(f"        If data was collected with no motion, try use_mti=False")
    
    # ==================== PYQTGRAPH SETUP ====================
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    
    # Create main window with control panel
    main_widget = QtWidgets.QWidget()
    main_layout = QtWidgets.QVBoxLayout()
    main_widget.setLayout(main_layout)
    
    # Control panel for color scale
    control_panel = QtWidgets.QWidget()
    control_layout = QtWidgets.QHBoxLayout()
    control_panel.setLayout(control_layout)
    
    control_layout.addWidget(QtWidgets.QLabel("Color Scale (dB):  Min:"))
    vmin_input = QtWidgets.QLineEdit(str(min_scale))
    vmin_input.setMaximumWidth(60)
    control_layout.addWidget(vmin_input)
    
    control_layout.addWidget(QtWidgets.QLabel("  Max:"))
    vmax_input = QtWidgets.QLineEdit(str(max_scale))
    vmax_input.setMaximumWidth(60)
    control_layout.addWidget(vmax_input)
    
    auto_scale_checkbox = QtWidgets.QCheckBox("Auto-scale")
    auto_scale_checkbox.setChecked(False)
    control_layout.addWidget(auto_scale_checkbox)
    
    apply_btn = QtWidgets.QPushButton("Apply")
    apply_btn.setMaximumWidth(80)
    control_layout.addWidget(apply_btn)
    
    pause_btn = QtWidgets.QPushButton("⏸ Pause")
    pause_btn.setMaximumWidth(100)
    pause_btn.setCheckable(True)
    control_layout.addWidget(pause_btn)
    
    control_layout.addWidget(QtWidgets.QLabel("  |  Zero Range Bins:"))
    zero_bins_input = QtWidgets.QLineEdit(str(zero_range_bins))
    zero_bins_input.setMaximumWidth(50)
    control_layout.addWidget(zero_bins_input)
    
    apply_zero_btn = QtWidgets.QPushButton("Apply")
    apply_zero_btn.setMaximumWidth(60)
    control_layout.addWidget(apply_zero_btn)
    
    control_layout.addWidget(QtWidgets.QLabel("  Max Range (m):"))
    max_range_input = QtWidgets.QLineEdit(str(max_range_m))
    max_range_input.setMaximumWidth(50)
    control_layout.addWidget(max_range_input)
    
    apply_range_btn = QtWidgets.QPushButton("Apply")
    apply_range_btn.setMaximumWidth(60)
    control_layout.addWidget(apply_range_btn)
    
    instructions = QtWidgets.QLabel("  |  SPACE=pause, Arrow keys=navigate, Q=quit")
    control_layout.addWidget(instructions)
    control_layout.addStretch()
    
    main_layout.addWidget(control_panel)
    
    win = pg.GraphicsLayoutWidget()
    win.resize(1800, 1000)
    main_layout.addWidget(win)
    
    main_widget.setWindowTitle(f"Range-Doppler Processing - {os.path.basename(h5_filepath)} - Interactive Controls")
    main_widget.resize(1850, 1050)
    main_widget.show()
    
    # Color scale variables
    color_scale = {'vmin': float(min_scale), 'vmax': float(max_scale), 'auto': False}
    paused = {'state': False}
    zero_bins_config = {'bins': zero_range_bins}
    max_range_config = {'range_m': max_range_m}
    
    def update_color_scale():
        try:
            new_vmin = float(vmin_input.text())
            new_vmax = float(vmax_input.text())
            
            # If user manually changed the values, disable auto-scale
            if new_vmin != color_scale['vmin'] or new_vmax != color_scale['vmax']:
                color_scale['auto'] = False
                auto_scale_checkbox.setChecked(False)
                print(f"Manual color scale set: {new_vmin:.1f} to {new_vmax:.1f} dB")
            
            color_scale['vmin'] = new_vmin
            color_scale['vmax'] = new_vmax
            update_plot()  # Refresh with new scale
        except ValueError:
            print("Invalid color scale values")
    
    def update_auto_scale_state():
        # When checkbox is toggled, update the auto state
        color_scale['auto'] = auto_scale_checkbox.isChecked()
        if color_scale['auto']:
            print("Auto-scale enabled")
        else:
            print(f"Auto-scale disabled - Manual: {color_scale['vmin']:.1f} to {color_scale['vmax']:.1f} dB")
        update_plot()  # Refresh with new setting
    
    def toggle_pause():
        paused['state'] = not paused['state']
        if paused['state']:
            pause_btn.setText("▶ Resume")
            print("PAUSED")
        else:
            pause_btn.setText("⏸ Pause")
            print("RESUMED")
    
    def update_zero_bins():
        try:
            new_bins = int(zero_bins_input.text())
            if new_bins < 0:
                print("Zero range bins must be >= 0")
                return
            zero_bins_config['bins'] = new_bins
            zeroed_range_m = new_bins * r_res
            print(f"Zero range bins updated to {new_bins} ({zeroed_range_m:.1f} m)")
            update_plot()  # Refresh with new zeroing
        except ValueError:
            print("Invalid zero range bins value")
    
    def update_max_range():
        try:
            new_range = float(max_range_input.text())
            if new_range <= 0:
                print("Max range must be > 0")
                return
            max_range_config['range_m'] = new_range
            print(f"Max range updated to {new_range:.1f} m")
            update_plot()  # Refresh with new max range
        except ValueError:
            print("Invalid max range value")
    
    apply_btn.clicked.connect(update_color_scale)
    vmin_input.returnPressed.connect(update_color_scale)
    vmax_input.returnPressed.connect(update_color_scale)
    auto_scale_checkbox.stateChanged.connect(update_auto_scale_state)
    pause_btn.clicked.connect(toggle_pause)
    apply_zero_btn.clicked.connect(update_zero_bins)
    zero_bins_input.returnPressed.connect(update_zero_bins)
    apply_range_btn.clicked.connect(update_max_range)
    max_range_input.returnPressed.connect(update_max_range)
    
    # Row 0-2, Col 0: Range-Doppler heatmap (left, spans 3 rows)
    rd_plot = win.addPlot(row=0, col=0, rowspan=3, title="Range-Doppler Map")
    rd_plot.setLabel('left', 'Range (m)')
    rd_plot.setLabel('bottom', 'Velocity (m/s)')
    rd_img = pg.ImageItem()
    rd_plot.addItem(rd_img)
    rd_cmap = pg.colormap.get('inferno')
    rd_img.setColorMap(rd_cmap)
    rd_cbar = pg.ColorBarItem(colorMap=rd_cmap, values=(min_scale, max_scale), 
                             limits=(min_scale-10, max_scale+10))
    rd_cbar.setImageItem(rd_img, insert_in=rd_plot)
    rd_stats = pg.TextItem(color='w', anchor=(0, 0))
    rd_plot.addItem(rd_stats)
    
    # Row 0, Col 1: Sum Data (Real) - Raw captured 1 chirp (top right)
    sumiq_plot = win.addPlot(row=0, col=1, title="Sum Data (Real)")
    sumiq_plot.setLabel('left', 'Amplitude')
    sumiq_plot.setLabel('bottom', 'Sample')
    sumiq_plot.showGrid(x=True, y=True, alpha=0.3)
    sumiq_curve = sumiq_plot.plot(pen=pg.mkPen('c', width=1))
    
    # Row 1, Col 1: Radar Data FFT - All doppler bins overlaid (middle right)
    fft_plot = win.addPlot(row=1, col=1, title="Radar Data FFT")
    fft_plot.setLabel('left', 'Magnitude (dB)')
    fft_plot.setLabel('bottom', 'Range Bin')
    fft_plot.showGrid(x=True, y=True, alpha=0.3)
    fft_curves = []
    
    # Row 2, Col 1: Ideal IQ (Real) - Reference chirp (bottom right)
    idealiq_plot = win.addPlot(row=2, col=1, title="Ideal IQ (Real)")
    idealiq_plot.setLabel('left', 'Amplitude')
    idealiq_plot.setLabel('bottom', 'Sample')
    idealiq_plot.showGrid(x=True, y=True, alpha=0.3)
    idealiq_curve = idealiq_plot.plot(pen=pg.mkPen('g', width=1))
    # Plot ideal IQ reference chirp once (it doesn't change)
    idealiq_curve.setData(np.arange(len(iq_chirp)), np.real(iq_chirp))
    
    # State variable for current capture
    current_capture = [capture_idx]
    
    def process_and_display_rd(capture_data):
        """Process single capture and return Range-Doppler data."""
        # Reshape data into chirps
        total_samples = len(capture_data)
        expected_samples = num_chirps * samples_per_chirp
        
        if total_samples < expected_samples:
            print(f"Warning: Not enough samples. Expected {expected_samples}, got {total_samples}")
            # Pad with zeros
            capture_data = np.pad(capture_data, (0, expected_samples - total_samples), 'constant')
        elif total_samples > expected_samples:
            # Truncate
            capture_data = capture_data[:expected_samples]
        
        # Reshape into (num_chirps, samples_per_chirp)
        rx_bursts = capture_data.reshape(num_chirps, samples_per_chirp)
        
        # CRITICAL: Mix (dechirp) with ideal IQ signal
        # In FMCW radar, we multiply received signal by conjugate of transmitted chirp
        # to extract beat frequencies proportional to target range
        rx_bursts_mixed = rx_bursts * np.conj(iq_chirp)
        
        # Debug: Check mixed signal strength
        mixed_power = np.mean(np.abs(rx_bursts_mixed))
        
        # Use current color scale settings
        current_min = color_scale['vmin']
        current_max = color_scale['vmax']
        
        # Process through freq_process (Range FFT + Doppler FFT)
        range_doppler_data = freq_process(rx_bursts_mixed, current_min, current_max, 
                                         use_window=True, mti_filter=use_mti)
        
        # Zero out first N range bins if requested (removes near-field clutter/crosstalk)
        current_zero_bins = zero_bins_config['bins']
        if current_zero_bins > 0:
            # range_doppler_data shape: (doppler_bins, range_bins)
            # Zero all doppler bins for the first N range bins
            range_doppler_data[:, :current_zero_bins] = current_min
        
        # Crop to max range
        num_range_bins = range_doppler_data.shape[1]
        current_max_range = max_range_config['range_m']
        max_range_bins = min(num_range_bins, int(current_max_range / r_res))
        range_doppler_data = range_doppler_data[:, :max_range_bins]
        
        # Debug: Check data range before and after clipping
        data_min = np.min(range_doppler_data)
        data_max = np.max(range_doppler_data)
        
        return range_doppler_data, rx_bursts, mixed_power, data_min, data_max
    
    def update_plot():
        """Update Range-Doppler plot with data from next capture."""
        t_start = time.perf_counter()
        
        current_capture[0] = (current_capture[0] + 1) % num_captures
        idx = current_capture[0]
        
        # Process captured data
        rd_data, rx_bursts, mixed_power, data_min, data_max = process_and_display_rd(all_sum_data[idx])
        
        # rd_data shape: (doppler_bins, range_bins)
        num_doppler = rd_data.shape[0]
        num_range_bins = rd_data.shape[1]
        
        # Calculate velocity and range axes
        vel_bins = num_doppler
        vel_half_span = vel_bins / 2 * v_res
        vel_min = -vel_half_span
        vel_max = vel_half_span
        range_max_display = num_range_bins * r_res
        
        # Update Range-Doppler heatmap
        # Use user-defined color scale or auto-scale
        if color_scale['auto']:
            # Use narrower dynamic range to make targets pop
            actual_min = np.min(rd_data)
            actual_max = np.max(rd_data)
            vmin = max(180, actual_min)
            vmax = vmin + 25
            if vmax > 270:
                vmax = 270
                vmin = 245
            # Update dictionary and input fields
            color_scale['vmin'] = vmin
            color_scale['vmax'] = vmax
            vmin_input.setText(f"{vmin:.1f}")
            vmax_input.setText(f"{vmax:.1f}")
        else:
            vmin = color_scale['vmin']
            vmax = color_scale['vmax']
        
        rd_img.setImage(rd_data, autoLevels=False)
        rd_img.setLevels([vmin, vmax])
        rd_img.setRect(pg.QtCore.QRectF(vel_min, 0, vel_max - vel_min, range_max_display))
        
        rd_plot.setXRange(vel_min, vel_max, padding=0)
        rd_plot.setYRange(0, range_max_display, padding=0)
        
        # Find peak
        peak_idx = np.unravel_index(np.argmax(rd_data), rd_data.shape)
        peak_vel = (peak_idx[0] - vel_bins / 2) * v_res
        peak_range = peak_idx[1] * r_res
        peak_db = np.max(rd_data)
        
        # Update stats overlay
        mti_status = "MTI ON" if use_mti else "MTI OFF"
        zero_bins_m = zero_bins_config['bins'] * r_res
        rd_stats.setText(f"Capture: {idx+1}/{num_captures}\nPeak: {peak_db:.1f} dB\n" + 
                        f"Range: {peak_range:.1f} m\nVel: {peak_vel:.1f} m/s\n" +
                        f"Scale: {vmin:.1f} to {vmax:.1f} dB\nZero bins: {zero_bins_config['bins']} ({zero_bins_m:.1f}m)\nMax range: {max_range_config['range_m']:.1f}m")
        rd_stats.setPos(vel_min + 0.5, range_max_display - 2)
        
        # Update colorbar
        rd_cbar.setLevels(values=(vmin, vmax))
        
        # Update Sum Data plot (first chirp real part)
        first_chirp = rx_bursts[0, :]
        sumiq_curve.setData(np.arange(len(first_chirp)), np.real(first_chirp))
        
        # Update FFT overlay plot (all doppler bins)
        for curve in fft_curves:
            fft_plot.removeItem(curve)
        fft_curves.clear()
        for doppler_idx in range(num_doppler):
            curve = fft_plot.plot(rd_data[doppler_idx, :], 
                                 pen=pg.mkPen((100, 200, 255, 100), width=1))
            fft_curves.append(curve)
        
        t_total = time.perf_counter() - t_start
        print(f"Capture {idx+1}/{num_captures} | Frame: {t_total*1000:.2f}ms | " +
              f"Peak: {peak_db:.1f}dB @ R={peak_range:.1f}m, V={peak_vel:.1f}m/s | " +
              f"Mixed: {mixed_power:.1f} | Data range: {data_min:.1f} to {data_max:.1f} dB")
    
    # Initial plot
    rd_data, rx_bursts, mixed_power, data_min, data_max = process_and_display_rd(all_sum_data[capture_idx])
    num_doppler = rd_data.shape[0]
    num_range_bins = rd_data.shape[1]
    vel_bins = num_doppler
    vel_half_span = vel_bins / 2 * v_res
    vel_min = -vel_half_span
    vel_max = vel_half_span
    range_max_display = num_range_bins * r_res
    
    # Debug output for initial capture
    print(f"\nInitial Capture {capture_idx+1}/{num_captures}:")
    print(f"  Mixed signal power: {mixed_power:.1f}")
    print(f"  Data range (after FFT): {data_min:.1f} to {data_max:.1f} dB")
    print(f"  Initial color scale: {color_scale['vmin']} to {color_scale['vmax']} dB")
    if data_max <= color_scale['vmin']:
        print(f"  WARNING: All data is at or below min_scale!")
        print(f"  This suggests MTI is removing all signals (no moving targets detected)")
    
    # Use initial color scale settings
    vmin = color_scale['vmin']
    vmax = color_scale['vmax']
    
    # Initial display
    rd_img.setImage(rd_data, autoLevels=False)
    rd_img.setLevels([vmin, vmax])
    rd_img.setRect(pg.QtCore.QRectF(vel_min, 0, vel_max - vel_min, range_max_display))
    rd_plot.setXRange(vel_min, vel_max, padding=0)
    rd_plot.setYRange(0, range_max_display, padding=0)
    
    # Initial Sum Data plot (first chirp)
    first_chirp = rx_bursts[0, :]
    sumiq_curve.setData(np.arange(len(first_chirp)), np.real(first_chirp))
    
    # Create timer for automatic updates (only runs when not paused)
    def timer_tick():
        if not paused['state']:
            update_plot()
    
    timer = QtCore.QTimer()
    timer.timeout.connect(timer_tick)
    timer.start(1000)  # Update every 1000ms (1s)
    
    print("\n" + "="*60)
    print(f"RANGE-DOPPLER DISPLAY READY - MTI: {'ON' if use_mti else 'OFF'}")
    print("Looping through captures...")
    print("Close window to exit")
    print("="*60)
    
    # Run the application
    app.exec_()


if __name__ == "__main__":
    # Example usage
    # Replace with your actual HDF5 file path
    h5_file = r"D:\Stingray\2026-02-06_13-04-18\StraighAndBackNCSFarm.h5"
    
    # Display raw ADC data (default mode)
    # post_process_position(h5_file, capture_idx=0, plot_mode='adc')
    
    # Display Range-Doppler heatmap with MTI disabled
    post_process_position(h5_file, capture_idx=0, plot_mode='range_doppler', use_mti=True, zero_range_bins=0, max_range_m=500)
    
    # Display Range-Doppler heatmap with MTI and zero out first 50 range bins (30m if r_res=0.6m)
    # post_process_position(h5_file, capture_idx=0, plot_mode='range_doppler', use_mti=True, zero_range_bins=50, max_range_m=100)
    
    # Display Range-Doppler without MTI and zero first 20 bins to remove near-field clutter
    # post_process_position(h5_file, capture_idx=0, plot_mode='range_doppler', use_mti=False, zero_range_bins=20, max_range_m=150)
