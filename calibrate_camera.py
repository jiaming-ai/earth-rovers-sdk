import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
import requests
import base64
import time

def calibrate_kb8_camera(image_folder, pattern_size=(9, 6), square_size=1.0):
    """
    Calibrate a camera using the KB8 model with distortion parameters up to 4th order.
    
    Args:
        image_folder: Path to folder containing checkerboard images
        pattern_size: Size of checkerboard pattern (inner corners)
        square_size: Size of each square in the checkerboard (in units)
        
    Returns:
        camera_matrix: Camera intrinsic matrix (3x3)
        dist_coeffs: Distortion coefficients (k1, k2, p1, p2, k3, k4)
        rvecs: Rotation vectors for each image
        tvecs: Translation vectors for each image
        reprojection_error: Mean reprojection error
    """
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Get list of calibration images
    images = glob.glob(os.path.join(image_folder, '*.jpg')) + \
             glob.glob(os.path.join(image_folder, '*.png'))
    
    if not images:
        raise ValueError(f"No images found in {image_folder}")
    
    # Process each image
    successful_images = []
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"Failed to load image: {fname}")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Store points
            objpoints.append(objp)
            imgpoints.append(corners2)
            successful_images.append(fname)
            
            # Optional: Draw and display the corners
            if False:  # Set to True to visualize corners
                img_with_corners = cv2.drawChessboardCorners(img.copy(), pattern_size, corners2, ret)
                cv2.imshow('Detected Corners', img_with_corners)
                cv2.waitKey(500)
    
    if not objpoints:
        raise ValueError("No valid checkerboard patterns found in images")
    
    print(f"Successfully processed {len(successful_images)} out of {len(images)} images")
    
    # Initialize camera matrix
    h, w = gray.shape
    camera_matrix = np.array([
        [w, 0, w/2],
        [0, w, h/2],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Use CALIB_RATIONAL_MODEL flag to include up to 4th order distortion coefficients
    flags = cv2.CALIB_RATIONAL_MODEL
    
    # Initialize distortion coefficients
    dist_coeffs = np.zeros(8)  # 8 parameters for distortion up to 4th order
    
    # Perform calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], camera_matrix, dist_coeffs,
        flags=flags)
    
    # Store the full 8-coefficient vector for internal calculations
    dist_coeffs_full = dist_coeffs.copy()
    
    # We're interested in the first 6 coefficients for KB8 model with 4th order distortion
    # (k1, k2, p1, p2, k3, k4)
    dist_coeffs = dist_coeffs[:6]
    
    # Calculate reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        # Use the full distortion coefficients for projectPoints
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], 
                                         camera_matrix, dist_coeffs_full)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    
    reprojection_error = total_error / len(objpoints)
    print(f"Mean reprojection error: {reprojection_error}")
    
    return camera_matrix, dist_coeffs, rvecs, tvecs, reprojection_error

def visualize_distortion(camera_matrix, dist_coeffs, image_size):
    """
    Visualize the KB8 camera distortion model.
    
    Args:
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        image_size: Size of the image (width, height)
    """
    # Create output directory if it doesn't exist
    output_dir = 'data/camera_calibration'
    os.makedirs(output_dir, exist_ok=True)
    
    w, h = image_size
    
    # Create a grid of points
    x, y = np.meshgrid(np.linspace(0, w-1, 20), np.linspace(0, h-1, 20))
    points = np.column_stack((x.flatten(), y.flatten()))
    
    # Convert points for undistortPoints function
    points_reshaped = points.reshape(-1, 1, 2).astype(np.float32)
    
    # Pad the distortion coefficients to 8 elements if needed
    # First, make sure dist_coeffs is flattened to 1D
    dist_coeffs = np.ravel(dist_coeffs)
    
    # Now pad to 8 elements
    dist_coeffs_padded = np.zeros(8)
    dist_coeffs_padded[:len(dist_coeffs)] = dist_coeffs
    
    # Normalize points (remove camera matrix effect)
    points_normalized = cv2.undistortPoints(
        points_reshaped, 
        camera_matrix, 
        dist_coeffs_padded
    ).reshape(-1, 2)
    
    # Project points applying distortion
    points_distorted = cv2.projectPoints(
        np.hstack((points_normalized, np.zeros((points_normalized.shape[0], 1)))),
        np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs_padded)[0].reshape(-1, 2)
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(points[:, 0], points[:, 1], c='b', marker='.', label='Original Grid')
    plt.scatter(points_distorted[:, 0], points_distorted[:, 1], c='r', marker='.', label='Distorted Grid')
    
    # Connect corresponding points
    for i in range(len(points)):
        plt.plot([points[i, 0], points_distorted[i, 0]], 
                 [points[i, 1], points_distorted[i, 1]], 'g-', alpha=0.3)
    
    plt.title('KB8 Camera Distortion Model (Up to 4th Order)')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'kb8_distortion_model.png'))
    plt.show()

def undistort_image(image, camera_matrix, dist_coeffs):
    """
    Undistort an image using the KB8 camera parameters.
    
    Args:
        image: Input distorted image
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
    
    Returns:
        undistorted_img: Output undistorted image
    """
    h, w = image.shape[:2]
    
    # First, make sure dist_coeffs is flattened to 1D
    dist_coeffs = np.ravel(dist_coeffs)
    
    # Pad the distortion coefficients to 8 elements if needed
    dist_coeffs_padded = np.zeros(8)
    dist_coeffs_padded[:len(dist_coeffs)] = dist_coeffs
    
    # Get optimal new camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs_padded, (w, h), 1, (w, h))
    
    # Undistort the image
    undistorted_img = cv2.undistort(image, camera_matrix, dist_coeffs_padded, None, new_camera_matrix)
    
    # Crop the image if desired
    x, y, w, h = roi
    if all([x, y, w, h]):  # Check if ROI is valid
        undistorted_img = undistorted_img[y:y+h, x:x+w]
    
    return undistorted_img

def save_calibration_results(camera_matrix, dist_coeffs, reprojection_error, output_file='kb8_calibration.npz'):
    """
    Save KB8 calibration results to file.
    
    Args:
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        reprojection_error: Mean reprojection error
        output_file: Output file name
    """
    # Create output directory if it doesn't exist
    output_dir = 'data/camera_calibration'
    os.makedirs(output_dir, exist_ok=True)
    
    # Update output file path
    output_file = os.path.join(output_dir, os.path.basename(output_file))
    
    # Make sure dist_coeffs is flattened to 1D
    dist_coeffs = np.ravel(dist_coeffs)
    
    # Save as NumPy archive
    np.savez(output_file, 
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs,
             reprojection_error=reprojection_error)
    
    # Save human-readable text file
    with open(output_file.replace('.npz', '.txt'), 'w') as f:
        f.write("KB8 Camera Calibration Results\n")
        f.write("==============================\n\n")
        f.write("Camera Matrix (Intrinsic parameters):\n")
        f.write(str(camera_matrix) + "\n\n")
        
        # Label distortion coefficients
        k1, k2, p1, p2, k3, k4 = dist_coeffs.ravel()
        f.write("Distortion Coefficients:\n")
        f.write(f"k1 (2nd order radial): {k1}\n")
        f.write(f"k2 (4th order radial): {k2}\n")
        f.write(f"p1 (tangential): {p1}\n")
        f.write(f"p2 (tangential): {p2}\n")
        f.write(f"k3 (6th order radial): {k3}\n")
        f.write(f"k4 (8th order radial): {k4}\n\n")
        
        f.write(f"Mean Reprojection Error: {reprojection_error}")
    
    print(f"Calibration results saved to {output_file} and {output_file.replace('.npz', '.txt')}")

def collect_images(num_images=40, save_folder='calibration_images', pattern_size=(9, 6)):
    """
    Collect calibration images from the robot camera at 2Hz.
    Only saves images where a checkerboard pattern is detected.
    
    Args:
        num_images: Number of images to collect
        save_folder: Folder to save images in
        pattern_size: Size of the checkerboard pattern (inner corners)
    
    Returns:
        True if successful, False otherwise
    """
    # Create folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)
    
    # API endpoint for camera feed
    screenshot_url = "http://localhost:8000/v2/front"
    
    print(f"Collecting {num_images} calibration images with checkerboard pattern ({pattern_size[0]}x{pattern_size[1]})...")
    print("Position checkerboard in different orientations in front of camera")
    print("Tips for detection:")
    print(" - Ensure good lighting with minimal shadows")
    print(" - Hold board flat and fully visible in the frame")
    print(" - Try different distances (not too close, not too far)")
    print(" - Avoid motion blur by holding the board steady")
    print("Press 'q' to stop collecting")
    
    # Create window to show preview
    window_name = "Camera Calibration Capture"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    images_captured = 0
    frames_processed = 0
    delay_seconds = 0.1  # 2Hz capture rate
    
    # Add adaptive thresholding to improve detection in difficult lighting
    use_adaptive = False
    
    try:
        while images_captured < num_images:
            frames_processed += 1
            # Get current time for filename
            timestamp = time.time()
            
            # Get camera frame
            try:
                response = requests.get(screenshot_url, timeout=1.0)
                
                if response.status_code == 200:
                    # Parse response
                    data = response.json()
                    front_frame_b64 = data["front_frame"]
                    
                    # Decode image
                    front_frame_bytes = base64.b64decode(front_frame_b64)
                    front_frame_np = np.frombuffer(front_frame_bytes, np.uint8)
                    frame = cv2.imdecode(front_frame_np, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # Check for checkerboard pattern
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        
                        # Try different methods to detect the checkerboard
                        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
                        
                        # If standard method fails, try with adaptive thresholding
                        if not ret and use_adaptive:
                            # Apply adaptive threshold to improve contrast
                            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                          cv2.THRESH_BINARY, 11, 2)
                            ret, corners = cv2.findChessboardCorners(thresh, pattern_size, None)
                        
                        # Create a copy of the frame for display
                        display_frame = frame.copy()
                        
                        if ret:
                            # Pattern found, save the image
                            filename = os.path.join(save_folder, f"calib_{timestamp:.2f}.png")
                            cv2.imwrite(filename, frame)
                            
                            # Draw the corners on the display frame
                            cv2.drawChessboardCorners(display_frame, pattern_size, corners, ret)
                            
                            images_captured += 1
                            pattern_status = f"PATTERN FOUND! Captured: {images_captured}/{num_images}"
                            text_color = (0, 255, 0)  # Green for success
                            print(f"Captured image {images_captured}/{num_images}")
                        else:
                            # Pattern not found
                            pattern_status = f"No checkerboard pattern found. Captured: {images_captured}/{num_images}"
                            text_color = (0, 0, 255)  # Red for failure
                        
                        # Add information overlay
                        cv2.putText(
                            display_frame,
                            pattern_status,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            text_color,
                            2
                        )
                        
                        cv2.putText(
                            display_frame,
                            f"Frames processed: {frames_processed}",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 0),
                            2
                        )
                        
                        cv2.putText(
                            display_frame,
                            f"Looking for {pattern_size[0]}x{pattern_size[1]} pattern",
                            (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 0),
                            2
                        )
                        
                        # Show preview
                        cv2.imshow(window_name, display_frame)
                else:
                    print(f"Error fetching camera feed: {response.status_code}")
            except Exception as e:
                print(f"Exception fetching camera feed: {e}")
            
            # Check for key press
            key = cv2.waitKey(int(delay_seconds * 1000)) & 0xFF
            if key == ord('q'):
                print("Image collection stopped by user")
                break
            elif key == ord('a'):
                # Toggle adaptive thresholding
                use_adaptive = not use_adaptive
                print(f"Adaptive thresholding {'enabled' if use_adaptive else 'disabled'}")
            
            # Wait for the next capture (maintain 2Hz)
            time.sleep(delay_seconds)
        
        cv2.destroyAllWindows()
        
        print(f"Collection complete! {images_captured} images with checkerboard pattern saved to {save_folder}")
        print(f"Total frames processed: {frames_processed}")
        return True if images_captured > 0 else False
        
    except KeyboardInterrupt:
        print("Image collection stopped by user")
        cv2.destroyAllWindows()
        return images_captured > 0

def main():
    # Define parameters
    image_folder = 'calibration_images'  # Folder containing checkerboard images
    pattern_size = (9, 6)  # Number of inner corners in the checkerboard pattern
    square_size = 2.8  # Size of each square in cm (adjust as needed)
    
    # Create output directory
    output_dir = 'data/camera_calibration'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create folder for calibration images if it doesn't exist
    os.makedirs(image_folder, exist_ok=True)
    
    # Check if there are images in the folder
    images = glob.glob(os.path.join(image_folder, '*.jpg')) + \
             glob.glob(os.path.join(image_folder, '*.png'))
    
    if not images:
        print(f"No images found in {image_folder}")
        collect = input("Would you like to collect calibration images now? (y/n): ")
        if collect.lower() == 'y':
            # Collect calibration images - PASS THE SAME PATTERN SIZE 
            if collect_images(save_folder=image_folder, pattern_size=pattern_size):
                # Re-check for images
                images = glob.glob(os.path.join(image_folder, '*.jpg')) + \
                         glob.glob(os.path.join(image_folder, '*.png'))
            else:
                print("Failed to collect images. Please put checkerboard calibration images in this folder and run the script again.")
                return
        else:
            print("Please put checkerboard calibration images in this folder and run the script again.")
            return
    
    # Perform calibration
    print("Calibrating KB8 camera model with distortion up to 4th order...")
    camera_matrix, dist_coeffs, rvecs, tvecs, reprojection_error = calibrate_kb8_camera(
        image_folder, pattern_size, square_size)
    
    # Print results
    print("\nCalibration Results:")
    print("-" * 40)
    print("Camera Matrix (Intrinsic parameters):")
    print(camera_matrix)
    print("\nDistortion Coefficients (k1, k2, p1, p2, k3, k4):")
    print(dist_coeffs.ravel())
    print(f"\nMean Reprojection Error: {reprojection_error}")
    
    # Visualize distortion model
    image = cv2.imread(images[0])
    if image is not None:
        image_size = image.shape[1], image.shape[0]
        print("\nGenerating distortion model visualization...")
        visualize_distortion(camera_matrix, dist_coeffs, image_size)
    
        # Undistort a sample image
        print("Applying undistortion to sample image...")
        undistorted = undistort_image(image, camera_matrix, dist_coeffs)
        
        # Save original and undistorted images
        cv2.imwrite(os.path.join(output_dir, 'original.png'), image)
        cv2.imwrite(os.path.join(output_dir, 'undistorted.png'), undistorted)
        
        # Create side-by-side comparison
        comparison = np.zeros((image.shape[0], image.shape[1]*2, 3), dtype=np.uint8)
        comparison[:, :image.shape[1]] = image
        comparison[:, image.shape[1]:] = undistorted
        
        # Add dividing line and labels
        cv2.line(comparison, (image.shape[1], 0), (image.shape[1], image.shape[0]), (0, 255, 0), 2)
        cv2.putText(comparison, "Original", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "Undistorted", (image.shape[1] + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Save comparison
        cv2.imwrite(os.path.join(output_dir, 'distortion_correction.png'), comparison)
        print(f"Distortion correction visualization saved to '{output_dir}/distortion_correction.png'")
    
    # Save calibration results
    save_calibration_results(camera_matrix, dist_coeffs, reprojection_error)
    
    print("\nKB8 camera calibration completed successfully!")

if __name__ == "__main__":
    main()