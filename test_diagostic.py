import numpy as np

files = [
    "calibration_data/rgb_intrinsics.npz",
    "calibration_data/projector/projector_intrinsics.npz",
    "calibration_data/rgbd_extrinsics/rgbd_calibration.npz"
]

for filename in files:
    print("\n", filename)

    with np.load(filename) as data:
        for key in data.files:
            print(key, data[key].shape, data[key].dtype)
            print(data[key] if data[key].size < 15 else "")
