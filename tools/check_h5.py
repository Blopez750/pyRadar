"""
Quick diagnostic script to check HDF5 file contents
"""
import h5py
import numpy as np

h5_file = r"D:\Stingray\drone_test_20260121_092407\position_2_az0_el0.h5"

print("Checking HDF5 file contents...")
print("="*60)

with h5py.File(h5_file, 'r') as f:
    print("Datasets:")
    for key in f.keys():
        print(f"  {key}: shape = {f[key].shape}, dtype = {f[key].dtype}")
    
    print("\nAttributes:")
    for key in f.attrs.keys():
        print(f"  {key} = {f.attrs[key]}")
    
    # Load first capture
    print("\nFirst capture stats:")
    first_capture = f['raw_adc_data'][0]
    print(f"  Shape: {first_capture.shape}")
    print(f"  Dtype: {first_capture.dtype}")
    print(f"  Min: {np.min(np.abs(first_capture))}")
    print(f"  Max: {np.max(np.abs(first_capture))}")
    print(f"  Mean: {np.mean(np.abs(first_capture))}")
    print(f"  Is complex: {np.iscomplexobj(first_capture)}")
