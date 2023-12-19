import cv2
import numpy as np
from pyk4a import PyK4A
from pyk4a.calibration import CalibrationType
import matplotlib.pyplot as plt

def get_kinect_ir_frame(device, visualize=False):
    """
    Capture an IR frame from the Kinect camera.
    """
    # Capture an IR frame
    for _ in range(20):
        device.get_capture()
        capture = device.get_capture()
        if capture is not None:
            ir_frame = capture.ir
            ir_frame = np.clip(ir_frame, 0, 5e3) / 5e3  # Clip and normalize
            # cv2.imshow('IR', ir_frame)
            if visualize:
                plt.imshow(ir_frame)
                plt.show()
            return ir_frame
    else:
        return None
    

def detect_aruco_markers(ir_frame, debug=False):
    """
    Detect ArUco markers in an IR frame and visualize the detection.
    """
    gray = cv2.convertScaleAbs(ir_frame, alpha=(255.0/ir_frame.max()))
    # gray = cv2.convertScaleAbs(ir_frame)
    
    # Load the predefined dictionary
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    
    # Detect markers
    corners, ids, rejected = detector.detectMarkers(gray)   # top-left, top-right, bottom-right, and bottom-left corners

    # Visualize markers
    vis_image = cv2.aruco.drawDetectedMarkers(gray.copy(), corners, ids)
    # cv2.destroyAllWindows()
    cv2.imshow('ArUco Marker Detection', vis_image)
    cv2.waitKey(0 if debug else 1)

    return corners, ids

def estimate_transformation(corners, ids, camera_matrix, dist_coeffs):
    """
    Estimate the transformation matrix A given ArUco marker detections.

    These should be known or calibrated beforehand:
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        dist_coeffs = np.array([k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4])  #  of 4, 5, 8 or 12 elements.
    """
    if ids is not None and len(ids) > 0:
        # Assuming marker size is known
        marker_size = 0.045  # In meters

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)

        # For demonstration, we'll use the first detected marker
        rvec, tvec = rvecs[0], tvecs[0]

        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)
        # Form the transformation matrix
        transform_mat = np.eye(4)
        transform_mat[:3, :3] = rmat
        transform_mat[:3, 3] = tvec.squeeze()
        return transform_mat

    return None

def main():
    # Initialize the camera
    cam_id = 3
    k4a = PyK4A(device_id=cam_id)
    k4a.start()

    camera_matrix = k4a.calibration.get_camera_matrix(CalibrationType.DEPTH)
    dist_coeffs = k4a.calibration.get_distortion_coefficients(CalibrationType.DEPTH)
    print(f"Camera matrix: {camera_matrix}")
    print(f"Distortion coefficients: {dist_coeffs}")

    # Capture IR frame from Kinect
    ir_frame = get_kinect_ir_frame(k4a)

    if ir_frame is not None:
        # Detect ArUco markers and get visualization
        corners, ids = detect_aruco_markers(ir_frame, debug=True)

        # Estimate transformation
        if ids is not None and len(ids) > 0:
            transform_matrix = estimate_transformation(corners, ids, camera_matrix, dist_coeffs)
            if transform_matrix is not None:
                print("Transformation Matrix A:")
                print(transform_matrix)
            else:
                print("Could not estimate transformation.")
        else:
            print("No ArUco marker detected.")
    else:
        print("Failed to capture IR frame.")

if __name__ == "__main__":
    main()