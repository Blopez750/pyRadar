"""
Calibration Manager Module

Handles saving and loading of calibration data for the Stingray radar system.

Calibration files are stored as JSON in the `cal files/` directory, named with
a timestamp: `cal_YYYYMMDD_HHMMSS.json`.  Each file contains:
  • cal_ant_fix      — per-subarray digital phase corrections (list of 4 floats)
  • loFreq           — LO frequency used during calibration (Hz)
  • rx_phase_cal     — per-element RX phase corrections (dict, element_id → degrees)
  • tx_phase_cal     — per-element TX phase corrections (dict, element_id → degrees)
  • rx/tx_gain_dict  — per-element VGA gain codes (dict, element_id → 0–127)
  • rx/tx_atten_dict — per-element attenuator flags (dict, element_id → 0 or 1)
  • sray_settings    — snapshot of all ADAR1000 register values at cal time

Stale calibrations (from previous days) are automatically purged on startup
by `purge_stale_calibrations()` to prevent applying outdated corrections.
"""

import numpy as np
import json
import os
from datetime import datetime
import glob


def convert_to_serializable(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def save_calibration(cal_ant_fix, loFreq, rx_phase_cal, tx_phase_cal, sray, rx_gain_dict=None, rx_atten_dict=None, tx_gain_dict=None, tx_atten_dict=None, cal_dir="cal files"):
    """
    Save all calibration values to a JSON file with timestamp
    
    Parameters:
        cal_ant_fix: Fixed antenna calibration values
        loFreq: LO frequency
        rx_phase_cal: RX phase calibration dictionary
        tx_phase_cal: TX phase calibration dictionary
        sray: Stingray object to extract current gain/phase/NCO settings
        rx_gain_dict: RX gain calibration dictionary
        rx_atten_dict: RX attenuation calibration dictionary
        tx_gain_dict: TX gain calibration dictionary
        tx_atten_dict: TX attenuation calibration dictionary
        cal_dir: Directory to save calibration files
    """
    # Create calibration directory if it doesn't exist
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cal_path = os.path.join(script_dir, cal_dir)
    os.makedirs(cal_path, exist_ok=True)

    # If TX calibration was not run (RX-only cal), build default TX dicts so
    # the saved file doesn't carry over stale hardware values (e.g. gain=0).
    if tx_gain_dict is None and sray is not None:
        tx_gain_dict = {elem_id: 127 for elem_id in sray.elements}
        print("--> TX gain dict not provided — saving defaults: all elements gain=127")
    if tx_phase_cal is None and sray is not None:
        tx_phase_cal = {elem_id: 0 for elem_id in sray.elements}
        print("--> TX phase cal not provided — saving defaults: all elements phase=0")
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cal_{timestamp}.json"
    filepath = os.path.join(cal_path, filename)
    
    # Collect all calibration data
    cal_data = {
        "timestamp": timestamp,
        "datetime": datetime.now().isoformat(),
        "loFreq": float(loFreq),
        "cal_ant_fix": convert_to_serializable(cal_ant_fix),
        "rx_phase_cal": convert_to_serializable(rx_phase_cal),
        "tx_phase_cal": convert_to_serializable(tx_phase_cal),
        "rx_gain_dict": convert_to_serializable(rx_gain_dict),
        "rx_atten_dict": convert_to_serializable(rx_atten_dict),
        "tx_gain_dict": convert_to_serializable(tx_gain_dict),
        "tx_atten_dict": convert_to_serializable(tx_atten_dict),
        "sray_settings": {}
    }
    
    # Extract Stingray element settings (gain, phase, attenuation)
    if sray is not None:
        element_settings = {}
        for elem_id, element in sray.elements.items():
            # Use calibrated TX values if available (avoids saving stale hardware state)
            saved_tx_gain = tx_gain_dict[elem_id] if (tx_gain_dict and elem_id in tx_gain_dict) else int(element.tx_gain)
            saved_tx_phase = tx_phase_cal[elem_id] if (tx_phase_cal and elem_id in tx_phase_cal) else float(element.tx_phase)
            element_settings[str(elem_id)] = {
                "rx_gain": int(element.rx_gain),
                "rx_phase": float(element.rx_phase),
                "rx_attenuator": int(element.rx_attenuator),
                "tx_gain": saved_tx_gain,
                "tx_phase": saved_tx_phase,
                "tx_attenuator": int(element.tx_attenuator)
            }
        cal_data["sray_settings"]["elements"] = element_settings
        
        # Extract device settings (NCO phases, if accessible)
        device_settings = {}
        for dev_id, device in sray.devices.items():
            device_settings[str(dev_id)] = {
                "mode": str(device.mode)
            }
        cal_data["sray_settings"]["devices"] = device_settings
    
    # Save to JSON file
    with open(filepath, 'w') as f:
        json.dump(cal_data, f, indent=4)
    
    print(f"Calibration saved to: {filepath}")
    return filepath


def purge_stale_calibrations(cal_dir="cal files"):
    """
    Delete any calibration files whose date (encoded in the filename) does not
    match today's date.  Files are expected to follow the naming convention
    ``cal_YYYYMMDD_HHMMSS.json``.

    Parameters:
        cal_dir: Directory containing calibration files
    """
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cal_path = os.path.join(script_dir, cal_dir)

    cal_files = glob.glob(os.path.join(cal_path, "cal_*.json"))
    if not cal_files:
        return

    today = datetime.now().strftime("%Y%m%d")
    removed = []
    for filepath in cal_files:
        filename = os.path.basename(filepath)          # e.g. cal_20260204_144435.json
        parts = filename.split("_")                    # ['cal', '20260204', '144435.json']
        if len(parts) >= 2 and parts[1] != today:
            os.remove(filepath)
            removed.append(filename)

    if removed:
        print(f"Removed {len(removed)} stale calibration file(s) from a previous day:")
        for name in removed:
            print(f"  {name}")
    else:
        print("No stale calibration files found.")


def load_latest_calibration(cal_dir="cal files"):
    """
    Load the most recent calibration file
    
    Parameters:
        cal_dir: Directory containing calibration files
        
    Returns:
        cal_data: Dictionary containing all calibration values, or None if no files found
    """
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cal_path = os.path.join(script_dir, cal_dir)
    
    # Find all calibration files
    cal_files = glob.glob(os.path.join(cal_path, "cal_*.json"))
    
    if not cal_files:
        print(f"No calibration files found in {cal_path}")
        return None
    
    # Get the most recent file
    latest_file = max(cal_files, key=os.path.getmtime)
    
    # Load the calibration data
    with open(latest_file, 'r') as f:
        cal_data = json.load(f)
    
    print(f"Loaded calibration from: {latest_file}")
    print(f"Calibration date: {cal_data['datetime']}")
    
    return cal_data


def apply_calibration(cal_data, sray):
    """
    Apply calibration values to the Stingray system element by element
    
    Parameters:
        cal_data: Dictionary containing calibration values
        sray: Stingray object to apply settings to
        
    Returns:
        cal_ant_fix, loFreq, rx_phase_cal, tx_phase_cal, rx_gain_dict, rx_atten_dict, tx_gain_dict, tx_atten_dict
    """
    if cal_data is None:
        print("No calibration data to apply")
        return None, None, None, None, None, None, None, None
    
    # Extract calibration values
    cal_ant_fix = cal_data.get("cal_ant_fix")
    loFreq = cal_data.get("loFreq")
    rx_phase_cal = cal_data.get("rx_phase_cal")
    tx_phase_cal = cal_data.get("tx_phase_cal")
    
    # Process gain/atten dictionaries - convert to proper format
    def process_dict(data):
        """Convert calibration dict to element_id -> value mapping"""
        if data is None:
            return None
        if isinstance(data, dict):
            # Convert string keys to int and return as dict
            return {int(k): v for k, v in data.items()}
        elif isinstance(data, list):
            # If it's already a list, return as-is
            return data
        return data
    
    # Try to get gain/atten from top-level first, then from sray_settings.elements
    rx_gain_dict = process_dict(cal_data.get("rx_gain_dict"))
    rx_atten_dict = process_dict(cal_data.get("rx_atten_dict"))
    tx_gain_dict = process_dict(cal_data.get("tx_gain_dict"))
    tx_atten_dict = process_dict(cal_data.get("tx_atten_dict"))
    
    # If not found at top level, extract from sray_settings.elements
    if rx_gain_dict is None or rx_atten_dict is None or tx_gain_dict is None or tx_atten_dict is None:
        sray_settings = cal_data.get("sray_settings", {})
        elements = sray_settings.get("elements", {})
        if elements:
            if rx_gain_dict is None:
                rx_gain_dict = {int(k): v.get("rx_gain") for k, v in elements.items() if "rx_gain" in v}
            if rx_atten_dict is None:
                rx_atten_dict = {int(k): v.get("rx_attenuator") for k, v in elements.items() if "rx_attenuator" in v}
            if tx_gain_dict is None:
                tx_gain_dict = {int(k): v.get("tx_gain") for k, v in elements.items() if "tx_gain" in v}
            if tx_atten_dict is None:
                tx_atten_dict = {int(k): v.get("tx_attenuator") for k, v in elements.items() if "tx_attenuator" in v}
    
    # Print global calibration values
    print("\n=== Global Calibration Values ===")
    print(f"LO Frequency: {loFreq/1e9:.3f} GHz")
    print(f"NCO Phase Corrections (cal_ant_fix): {cal_ant_fix}")
    
    # Convert string keys back to integers for phase dictionaries
    if rx_phase_cal is not None:
        rx_phase_cal = {int(k): v for k, v in rx_phase_cal.items()}
    if tx_phase_cal is not None:
        tx_phase_cal = {int(k): v for k, v in tx_phase_cal.items()}
    
    # Apply calibration element by element
    if sray is not None:
        print("\n=== Applying Calibration to Stingray Elements ===")
        
        for elem_id in sray.elements:
            element = sray.elements[elem_id]
            print(f"\nElement {elem_id}:")
            
            # Apply RX gain (if available)
            if rx_gain_dict is not None and elem_id in rx_gain_dict:
                element.rx_gain = int(rx_gain_dict[elem_id])
                print(f"  RX Gain: {element.rx_gain}")
            
            # Apply RX phase (if available)
            if rx_phase_cal is not None and elem_id in rx_phase_cal:
                element.rx_phase = rx_phase_cal[elem_id]
                print(f"  RX Phase: {element.rx_phase:.2f}°")
            
            # Apply RX attenuation (if available)
            if rx_atten_dict is not None and elem_id in rx_atten_dict:
                element.rx_attenuator = int(rx_atten_dict[elem_id])
                print(f"  RX Atten: {element.rx_attenuator}")
            
            # Apply TX gain (if available)
            if tx_gain_dict is not None and elem_id in tx_gain_dict:
                element.tx_gain = int(tx_gain_dict[elem_id])
                print(f"  TX Gain: {element.tx_gain}")
            
            # Apply TX phase (if available)
            if tx_phase_cal is not None and elem_id in tx_phase_cal:
                element.tx_phase = tx_phase_cal[elem_id]
                print(f"  TX Phase: {element.tx_phase:.2f}°")
            
            # Apply TX attenuation (if available)
            if tx_atten_dict is not None and elem_id in tx_atten_dict:
                element.tx_attenuator = int(tx_atten_dict[elem_id])
                print(f"  TX Atten: {element.tx_attenuator}")
        
        print("\n✓ Calibration applied successfully")
        print("Note: You may need to call sray.latch_rx_settings() or sray.latch_tx_settings()")
    
    return cal_ant_fix, loFreq, rx_phase_cal, tx_phase_cal, rx_gain_dict, rx_atten_dict, tx_gain_dict, tx_atten_dict
