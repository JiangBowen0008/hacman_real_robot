from pyk4a import PyK4A
import time
import matplotlib.pyplot as plt
import open3d as o3d

# Initialize the camera
k4a = PyK4A(device_id=0)
k4a.start()

# Capture a single frame (implement error checking in real code)
for _ in range(100):
    try:
        capture = k4a.get_capture()
        if capture.depth is not None:
            # Read pcd from k4a
            img = capture.color
            depth = capture.depth
            plt.imshow(img)
            plt.show()

        else:
            print("No depth frame captured")
    except:
        print("Failed to capture PCD from camera")
        time.sleep(0.1)