import os
import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from clear.data_process.compass_calibration import CompassCalibrator
import seaborn as sns
from scipy.spatial import ConvexHull
import json
import re
import glob

def read_csv_data(file_path, data_type):
    """
    Read data from CSV file
    
    Args:
        file_path: Path to the CSV file
        data_type: Type of data ('gps', 'mag', 'rpm', or 'imu')
        
    Returns:
        List of dictionaries with the data
    """
    data = []
    
    if data_type == 'imu':
        # Handle IMU data with nested JSON-like structure
        
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)  # Skip header row
            
            for row in reader:
                if len(row) < 4:  # Ensure we have enough columns
                    continue
                
                
                # Extract compass data (magnetometer)
                compass_str = row[0]
                try:
                    # Clean up the string to make it valid JSON
                    compass_str = compass_str.replace('""', '"')  # Fix double quotes
                    # Extract the array content using regex
                    match = re.search(r'\[\[(.*?)\]\]', compass_str)
                    if match:
                        compass_values = match.group(1).split(', ')
                        # Remove remaining quotes
                        x = float(compass_values[0].replace('"', ''))
                        y = float(compass_values[1].replace('"', ''))
                        z = float(compass_values[2].replace('"', ''))
                        timestamp = float(compass_values[3].replace('"', ''))  # Last column is timestamp
                        
                        data.append({
                            'timestamp': timestamp,
                            'x': x,
                            'y': y,
                            'z': z
                        })
                except Exception as e:
                    print(f"Error parsing compass data: {e}")
                    continue
    else:
        # Handle standard CSV formats
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                entry = {}
                
                if data_type == 'gps':
                    entry['latitude'] = float(row['latitude'])
                    entry['longitude'] = float(row['longitude'])
                    entry['timestamp'] = float(row['timestamp'])/1000

                elif data_type == 'rpm':
                    entry['rpm_1'] = float(row['rpm_1'])
                    entry['rpm_2'] = float(row['rpm_2'])
                    entry['rpm_3'] = float(row['rpm_3'])
                    entry['rpm_4'] = float(row['rpm_4'])
                    entry['timestamp'] = float(row['timestamp'])/1000
                    
                data.append(entry)
    
    return sorted(data, key=lambda x: x['timestamp'])

def get_raw_headings(mag_data):
    """
    Calculate raw headings from magnetometer data without any calibration
    
    Args:
        mag_data: List of magnetometer readings
        
    Returns:
        List of dictionaries with timestamps and headings
    """
    raw_headings = []
    
    for entry in mag_data:
        # Convert directly to NED frame (North-East-Down)
        # According to the schema in the original code:
        # - Positive X = Vertical Up
        # - Positive Y = West
        # - Positive Z = South
        north = -entry['z']  # North is opposite of South (Z)
        east = -entry['y']   # East is opposite of West (Y)
        
        # Calculate heading (clockwise from North)
        heading = math.degrees(math.atan2(east, north))
        heading = (heading + 360) % 360
        
        raw_headings.append({
            'timestamp': entry['timestamp'],
            'heading': heading
        })
    
    return raw_headings

def compare_calibration_methods(gps_csv, imu_csv, rpm_csv, output_dir='./output',
                              wheel_diameter=130, wheel_base=200):
    """
    Run different calibration methods and compare their results
    
    Args:
        gps_csv: Path to GPS data CSV
        imu_csv: Path to IMU data CSV (containing magnetometer data)
        rpm_csv: Path to RPM data CSV
        output_dir: Directory to save results
        wheel_diameter: Diameter of wheels in mm
        wheel_base: Distance between wheels in mm
        
    Returns:
        Dictionary with results for each method
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read data from CSV files
    print("Loading data from CSV files...")
    gps_data = read_csv_data(gps_csv, 'gps')
    mag_data = read_csv_data(imu_csv, 'imu')  # Using imu_csv for magnetometer data
    rpm_data = read_csv_data(rpm_csv, 'rpm')
    
    print(f"Loaded {len(gps_data)} GPS points, {len(mag_data)} magnetometer points, {len(rpm_data)} RPM points")
    
    # Initialize results dictionary
    results = {}
    
    # Define all calibration methods to test
    calibration_methods = [
        {
            'name': 'Raw Heading (No Calibration)',
            'use_ellipsoid_fit': False,
            'use_gps_calibration': False,
            'use_static_calibration': False,
            'use_kalman_filter': False
        },
        {
            'name': 'Static Calibration Only',
            'use_ellipsoid_fit': False,
            'use_gps_calibration': False,
            'use_static_calibration': True,
            'use_kalman_filter': False
        },
        {
            'name': 'Ellipsoid Fitting Only',
            'use_ellipsoid_fit': True,
            'use_gps_calibration': False,
            'use_static_calibration': False,
            'use_kalman_filter': False
        },
        {
            'name': 'GPS Calibration Only',
            'use_ellipsoid_fit': False,
            'use_gps_calibration': True,
            'use_static_calibration': False,
            'use_kalman_filter': False
        },
        {
            'name': 'Ellipsoid + GPS',
            'use_ellipsoid_fit': True,
            'use_gps_calibration': True,
            'use_static_calibration': False,
            'use_kalman_filter': False
        },
        {
            'name': 'GPS + Static',
            'use_ellipsoid_fit': False,
            'use_gps_calibration': True,
            'use_static_calibration': True,
            'use_kalman_filter': False
        },
        {
            'name': 'Ellipsoid + GPS + Kalman',
            'use_ellipsoid_fit': True,
            'use_gps_calibration': True,
            'use_static_calibration': False,
            'use_kalman_filter': True
        },
        {
            'name': 'GPS + Static + Kalman',
            'use_ellipsoid_fit': False,
            'use_gps_calibration': True,
            'use_static_calibration': True,
            'use_kalman_filter': True
        },
        {
            'name': 'Ellipsoid + Kalman',
            'use_ellipsoid_fit': True,
            'use_gps_calibration': False,
            'use_static_calibration': False,
            'use_kalman_filter': True
        },
        {
            'name': 'GPS + Kalman',
            'use_ellipsoid_fit': False,
            'use_gps_calibration': True,
            'use_static_calibration': False,
            'use_kalman_filter': True
        }
    ]
    
    # Process each calibration method
    for method in calibration_methods:
        method_name = method['name']
        print(f"\nMethod: {method_name}")
        
        try:
            # Create calibrator with specified parameters
            calibrator = CompassCalibrator(
                use_ellipsoid_fit=method['use_ellipsoid_fit'],
                use_gps_calibration=method['use_gps_calibration'],
                use_static_calibration=method['use_static_calibration'],
                use_kalman_filter=method['use_kalman_filter'],
                wheel_diameter=wheel_diameter,
                wheel_base=wheel_base
            )
            
            # Special case for raw heading (no calibration)
            if method_name == 'Raw Heading (No Calibration)':
                headings = get_raw_headings(mag_data)
                # Calculate validation metrics against GPS
                aligned_data = calibrator._align_timestamps(gps_data, mag_data, None)
                metrics = calibrator._validate_calibration(aligned_data, headings)
            else:
                # Standard calibration process
                headings = calibrator.calibrate(gps_data, mag_data, rpm_data)
                metrics = calibrator.get_validation_metrics()
            
            # Store results
            results[method_name] = {
                'headings': headings,
                'metrics': metrics,
                'params': calibrator.get_calibration_params() if hasattr(calibrator, 'calibration_params') else None
            }
            
        except Exception as e:
            print(f"Error in {method_name} calibration: {e}")
            # Add placeholder results
            results[method_name] = {
                'headings': [],
                'metrics': {'error_count': 0, 'mean_error': float('nan'), 'max_error': float('nan')},
                'params': None
            }
    
    # Generate visualizations and comparison metrics
    print("\nGenerating visualizations and comparison metrics...")
    
    # 1. Visualize magnetometer data before and after calibration
    try:
        # Use ellipsoid fitting results for visualization
        ellipsoid_method = [m for m in calibration_methods if m['use_ellipsoid_fit'] and 
                          not m['use_gps_calibration'] and 
                          not m['use_static_calibration'] and
                          not m['use_kalman_filter']]
        
        if ellipsoid_method and ellipsoid_method[0]['name'] in results:
            calibrator_for_viz = CompassCalibrator(
                use_ellipsoid_fit=True,
                use_gps_calibration=False,
                use_static_calibration=False,
                use_kalman_filter=False
            )
            calibrator_for_viz.ellipsoid_params = calibrator_for_viz._perform_ellipsoid_fitting(mag_data)
            visualize_magnetometer_data(mag_data, calibrator_for_viz, os.path.join(output_dir, "magnetometer_calibration.png"))
    except Exception as e:
        print(f"Error visualizing magnetometer data: {e}")
    
    # 2. Create individual heading visualizations
    for method_name, method_results in results.items():
        try:
            safe_method_name = method_name.replace(' ', '_').lower()
            visualize_headings(gps_data, method_results['headings'], 
                             os.path.join(output_dir, f"{safe_method_name}_headings.png"),
                             title=f"Compass Headings - {method_name}",
                             metrics=method_results['metrics'])
        except Exception as e:
            print(f"Error visualizing headings for {method_name}: {e}")
    
    # 3. Create combined visualization
    try:
        create_combined_visualization(gps_data, results, os.path.join(output_dir, "combined_comparison.png"))
    except Exception as e:
        print(f"Error creating combined visualization: {e}")
    
    # 4. Create metrics comparison chart
    try:
        create_metrics_comparison(results, os.path.join(output_dir, "metrics_comparison.png"))
    except Exception as e:
        print(f"Error creating metrics comparison: {e}")
    
    # 5. Generate metrics table
    try:
        metrics_df = generate_metrics_table(results)
        metrics_df.to_csv(os.path.join(output_dir, "calibration_metrics.csv"), index=False)
        
        print("\nCalibration Metrics Comparison:")
        print("-" * 80)
        print(metrics_df.to_string(index=False))
        print("-" * 80)
    except Exception as e:
        print(f"Error generating metrics table: {e}")
    
    print(f"\nResults saved to {output_dir}")
    
    return results

def visualize_magnetometer_data(mag_data, calibrator, output_path):
    """
    Visualize magnetometer data before and after calibration
    
    Args:
        mag_data: Raw magnetometer data
        calibrator: Calibrator with ellipsoid parameters
        output_path: Path to save the output image
    """
    # Extract raw magnetometer data
    x_raw = np.array([entry['x'] for entry in mag_data])
    y_raw = np.array([entry['y'] for entry in mag_data])
    z_raw = np.array([entry['z'] for entry in mag_data])
    
    # Check if ellipsoid parameters are valid
    if calibrator.ellipsoid_params is None:
        print("Warning: Ellipsoid parameters are None. Creating fallback visualization.")
        # Create a simple visualization of raw data only
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 3, figure=fig)
        
        # 3D plot of raw data
        ax1 = fig.add_subplot(gs[0, 0:2], projection='3d')
        ax1.scatter(x_raw, y_raw, z_raw, c='blue', alpha=0.5)
        ax1.set_title('Raw Magnetometer Data (3D)')
        ax1.set_xlabel('X (Up)')
        ax1.set_ylabel('Y (West)')
        ax1.set_zlabel('Z (South)')
        
        # 2D projections
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.scatter(x_raw, y_raw, c='blue', alpha=0.5)
        ax3.set_title('Raw Data (X-Y Plane)')
        ax3.set_xlabel('X (Up)')
        ax3.set_ylabel('Y (West)')
        ax3.grid(True)
        
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.scatter(x_raw, z_raw, c='blue', alpha=0.5)
        ax4.set_title('Raw Data (X-Z Plane)')
        ax4.set_xlabel('X (Up)')
        ax4.set_ylabel('Z (South)')
        ax4.grid(True)
        
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.scatter(y_raw, z_raw, c='blue', alpha=0.5)
        ax5.set_title('Raw Data (Y-Z Plane)')
        ax5.set_xlabel('Y (West)')
        ax5.set_ylabel('Z (South)')
        ax5.grid(True)
        
        plt.suptitle('Magnetometer Data (Calibration Failed)', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_path, dpi=300)
        plt.close()
        return
    
    # Apply calibration
    x_cal = []
    y_cal = []
    z_cal = []
    
    ellipsoid_params = calibrator.ellipsoid_params
    offset_x = ellipsoid_params['offset_x']
    offset_y = ellipsoid_params['offset_y']
    offset_z = ellipsoid_params['offset_z']
    transform = np.array(ellipsoid_params['transform'])
    
    for entry in mag_data:
        # Center data
        x_centered = entry['x'] - offset_x
        y_centered = entry['y'] - offset_y
        z_centered = entry['z'] - offset_z
        
        # Apply transform
        v = np.array([x_centered, y_centered, z_centered])
        calibrated_v = transform.dot(v)
        
        x_cal.append(calibrated_v[0])
        y_cal.append(calibrated_v[1])
        z_cal.append(calibrated_v[2])
    
    # Convert to numpy arrays
    x_cal = np.array(x_cal)
    y_cal = np.array(y_cal)
    z_cal = np.array(z_cal)
    
    # Create 3D plot
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # 3D plots
    ax1 = fig.add_subplot(gs[0, 0:2], projection='3d')
    ax1.scatter(x_raw, y_raw, z_raw, c='blue', alpha=0.5)
    ax1.set_title('Raw Magnetometer Data (3D)')
    ax1.set_xlabel('X (Up)')
    ax1.set_ylabel('Y (West)')
    ax1.set_zlabel('Z (South)')
    
    try:
        # Try to add convex hull to raw data
        points = np.column_stack((x_raw, y_raw, z_raw))
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            ax1.plot3D(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'r-', alpha=0.3)
    except:
        # Skip if convex hull fails
        pass
    
    ax2 = fig.add_subplot(gs[0, 2], projection='3d')
    ax2.scatter(x_cal, y_cal, z_cal, c='red', alpha=0.5)
    ax2.set_title('Calibrated Magnetometer Data (3D)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    try:
        # Try to add convex hull to calibrated data
        points = np.column_stack((x_cal, y_cal, z_cal))
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            ax2.plot3D(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'g-', alpha=0.3)
    except:
        # Skip if convex hull fails
        pass
    
    # 2D projections
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(x_raw, y_raw, c='blue', alpha=0.5)
    ax3.set_title('Raw Data (X-Y Plane)')
    ax3.set_xlabel('X (Up)')
    ax3.set_ylabel('Y (West)')
    ax3.grid(True)
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(x_raw, z_raw, c='blue', alpha=0.5)
    ax4.set_title('Raw Data (X-Z Plane)')
    ax4.set_xlabel('X (Up)')
    ax4.set_ylabel('Z (South)')
    ax4.grid(True)
    
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.scatter(y_raw, z_raw, c='blue', alpha=0.5)
    ax5.set_title('Raw Data (Y-Z Plane)')
    ax5.set_xlabel('Y (West)')
    ax5.set_ylabel('Z (South)')
    ax5.grid(True)
    
    # Add calibration parameters
    info_text = (
        f"Hard Iron Offsets:\n"
        f"X: {offset_x:.2f}\n"
        f"Y: {offset_y:.2f}\n"
        f"Z: {offset_z:.2f}\n\n"
        f"Soft Iron Transform:\n"
        f"{transform[0,0]:.2f} {transform[0,1]:.2f} {transform[0,2]:.2f}\n"
        f"{transform[1,0]:.2f} {transform[1,1]:.2f} {transform[1,2]:.2f}\n"
        f"{transform[2,0]:.2f} {transform[2,1]:.2f} {transform[2,2]:.2f}"
    )
    fig.text(0.02, 0.5, info_text, fontsize=10, family='monospace')
    
    plt.suptitle('Magnetometer Calibration: Before vs After Ellipsoid Fitting', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()

def visualize_headings(gps_data, heading_data, output_path, title="Compass Headings", 
                      metrics=None, max_arrows=150):
    """
    Visualize compass headings as arrows at GPS locations
    
    Args:
        gps_data: List of dictionaries with GPS data
        heading_data: List of dictionaries with heading data
        output_path: Path to save the output image
        title: Title for the plot
        metrics: Optional metrics to display
        max_arrows: Maximum number of arrows to display
    """
    # Create a list of GPS timestamps for efficient lookup
    gps_timestamps = [entry['timestamp'] for entry in gps_data]
    
    # Match headings with GPS positions using closest timestamp within threshold
    threshold = 0.8  # seconds
    matched_data = []
    
    for entry in heading_data:
        ts = entry['timestamp']
        # Find closest GPS timestamp
        closest_idx = None
        min_diff = float('inf')
        
        for i, gps_ts in enumerate(gps_timestamps):
            diff = abs(ts - gps_ts)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
        
        # Only use if within threshold
        if closest_idx is not None and min_diff < threshold:
            gps_entry = gps_data[closest_idx]
            lat, lon = gps_entry['latitude'], gps_entry['longitude']
            heading = entry['heading']
            matched_data.append((ts, lat, lon, heading))
    
    # Sort by timestamp
    matched_data.sort()
    
    if not matched_data:
        print(f"Warning: No matching data points found for {title}")
        # Create a blank image with error message
        plt.figure(figsize=(12, 10))
        plt.text(0.5, 0.5, f"No matching data points found for {title}", 
                ha='center', va='center', fontsize=16)
        plt.axis('off')
        plt.savefig(output_path, dpi=300)
        plt.close()
        return
    
    # Extract data for plotting
    timestamps, lats, lons, headings = zip(*matched_data)
    
    # Convert to numpy arrays
    lats = np.array(lats)
    lons = np.array(lons)
    headings = np.array(headings)
    
    # Sparsify if too many points
    if len(lats) > max_arrows:
        step = len(lats) // max_arrows
        indices = np.arange(0, len(lats), step)
        lats = lats[indices]
        lons = lons[indices]
        headings = headings[indices]
    
    # Convert heading to direction components
    # u = east component, v = north component
    u = np.sin(np.radians(headings))
    v = np.cos(np.radians(headings))
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Plot GPS track
    all_lats = [entry['latitude'] for entry in gps_data]
    all_lons = [entry['longitude'] for entry in gps_data]
    plt.plot(all_lons, all_lats, 'b-', alpha=0.3, linewidth=1.5, label='GPS Track')
    
    # Plot heading arrows
    plt.quiver(lons, lats, u, v, color='red', scale=25, width=0.003, 
              alpha=0.8, label='Heading', pivot='mid')
    
    # Mark start and end points
    plt.scatter(all_lons[0], all_lats[0], color='green', s=100, label='Start')
    plt.scatter(all_lons[-1], all_lats[-1], color='red', s=100, label='End')
    
    # Build title with metrics if provided
    if metrics:
        # Handle potentially NaN or missing metrics
        mean_error = metrics.get('mean_error', float('nan'))
        median_error = metrics.get('median_error', float('nan'))
        max_error = metrics.get('max_error', float('nan'))
        
        # Format as strings with error handling
        mean_str = f"{mean_error:.2f}°" if not math.isnan(mean_error) else "N/A"
        median_str = f"{median_error:.2f}°" if not math.isnan(median_error) else "N/A"
        max_str = f"{max_error:.2f}°" if not math.isnan(max_error) else "N/A"
        
        metrics_text = f"\nMean Error: {mean_str}, Median: {median_str}, Max: {max_str}"
        title = title + metrics_text
    
    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300)
    plt.close()

def create_combined_visualization(gps_data, results, output_path):
    """
    Create a combined visualization of all calibration methods
    
    Args:
        gps_data: GPS data
        results: Dictionary with results for each method
        output_path: Path to save the output image
    """
    # Limit the number of methods to show to avoid overcrowding
    # Prioritize showing the most interesting methods
    priority_methods = [
        'Raw Heading (No Calibration)',
        'Ellipsoid Fitting Only',
        'Ellipsoid + GPS',
        'Ellipsoid + GPS + Kalman'
    ]
    
    # Filter results to include only priority methods that exist
    filtered_results = {}
    for method in priority_methods:
        if method in results:
            filtered_results[method] = results[method]
    
    # If we don't have enough priority methods, add others
    if len(filtered_results) < 4:
        for method, result in results.items():
            if method not in filtered_results and len(filtered_results) < 4:
                filtered_results[method] = result
    
    # Create figure with subplots (2x2 grid)
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(2, 2, figure=fig)
    
    # Get GPS track data
    all_lats = [entry['latitude'] for entry in gps_data]
    all_lons = [entry['longitude'] for entry in gps_data]
    
    # Create a map of timestamps to GPS positions
    gps_map = {entry['timestamp']: (entry['latitude'], entry['longitude']) 
             for entry in gps_data}
    
    # Process each method
    for i, (method_name, method_results) in enumerate(filtered_results.items()):
        # Skip if we have more than 4 methods (shouldn't happen with our filtering)
        if i >= 4:
            break
            
        # Determine subplot position
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        
        # Match headings with GPS positions
        matched_data = []
        for entry in method_results['headings']:
            ts = entry['timestamp']
            if ts in gps_map:
                lat, lon = gps_map[ts]
                heading = entry['heading']
                matched_data.append((lat, lon, heading))
        
        if not matched_data:
            ax.text(0.5, 0.5, f"No matched data for {method_name}", 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Extract data for plotting
        lats, lons, headings = zip(*matched_data)
        
        # Sparsify if too many points
        max_arrows = 100
        if len(lats) > max_arrows:
            step = len(lats) // max_arrows
            indices = np.arange(0, len(lats), step)
            lats = np.array(lats)[indices]
            lons = np.array(lons)[indices]
            headings = np.array(headings)[indices]
        else:
            lats = np.array(lats)
            lons = np.array(lons)
            headings = np.array(headings)
        
        # Convert heading to direction components
        u = np.sin(np.radians(headings))
        v = np.cos(np.radians(headings))
        
        # Plot GPS track
        ax.plot(all_lons, all_lats, 'b-', alpha=0.3, linewidth=1.5)
        
        # Plot heading arrows
        ax.quiver(lons, lats, u, v, color='red', scale=25, width=0.003, alpha=0.8, pivot='mid')
        
        # Mark start and end points
        ax.scatter(all_lons[0], all_lats[0], color='green', s=50, label='Start')
        ax.scatter(all_lons[-1], all_lats[-1], color='red', s=50, label='End')
        
        # Get metrics
        metrics = method_results['metrics']
        mean_error = metrics.get('mean_error', float('nan'))
        mean_str = f"{mean_error:.2f}°" if not math.isnan(mean_error) else "N/A"
        metrics_text = f"Mean Error: {mean_str}"
        
        # Set subplot title
        ax.set_title(f"{method_name}\n{metrics_text}")
        ax.grid(True)
        
        # Only show legend on first plot
        if i == 0:
            ax.legend()
        
        # Only show axis labels on bottom row and left column
        if row == 1:
            ax.set_xlabel('Longitude')
        if col == 0:
            ax.set_ylabel('Latitude')
    
    plt.suptitle('Comparison of Compass Calibration Methods', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle
    
    # Save figure
    plt.savefig(output_path, dpi=300)
    plt.close()

def create_metrics_comparison(results, output_path):
    """
    Create a bar chart comparing the metrics of different calibration methods
    
    Args:
        results: Dictionary with results for each method
        output_path: Path to save the output image
    """
    # Extract metrics
    methods = []
    mean_errors = []
    median_errors = []
    max_errors = []
    
    valid_data = False  # Flag to check if any valid data exists
    
    for method_name, method_results in results.items():
        metrics = method_results['metrics']
        
        # Get metrics with safe fallback
        mean_error = metrics.get('mean_error', float('nan'))
        median_error = metrics.get('median_error', float('nan'))
        max_error = metrics.get('max_error', float('nan'))
        
        # Check if we have at least one valid metric
        if not (math.isnan(mean_error) and math.isnan(median_error) and math.isnan(max_error)):
            valid_data = True
        
        methods.append(method_name)
        mean_errors.append(mean_error if not math.isnan(mean_error) else 0)
        median_errors.append(median_error if not math.isnan(median_error) else 0)
        max_errors.append(max_error if not math.isnan(max_error) else 0)
    
    # If all metrics are NaN, create a message plot instead of bar chart
    if not valid_data:
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, "No valid metrics available for comparison", 
                ha='center', va='center', fontsize=16)
        plt.axis('off')
        plt.savefig(output_path, dpi=300)
        plt.close()
        return
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Method': methods,
        'Mean Error': mean_errors,
        'Median Error': median_errors,
        'Maximum Error': max_errors
    })
    
    # Convert to long format for seaborn
    df_long = pd.melt(df, id_vars=['Method'], 
                     value_vars=['Mean Error', 'Median Error', 'Maximum Error'],
                     var_name='Metric', value_name='Error (degrees)')
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Use seaborn for better visuals
    sns.set_style("whitegrid")
    ax = sns.barplot(x='Method', y='Error (degrees)', hue='Metric', data=df_long)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f°', padding=3)
    
    # Customize plot
    plt.title('Compass Calibration Error Comparison', fontsize=16)
    plt.xlabel('Calibration Method', fontsize=12)
    plt.ylabel('Error (degrees)', fontsize=12)
    
    # Set y-limit with safety check
    y_max = max(max_errors) if max_errors and max(max_errors) > 0 else 10
    plt.ylim(0, y_max * 1.1)  # Add some headroom
    
    plt.legend(title='Metric')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=30, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def generate_metrics_table(results):
    """
    Generate a metrics comparison table
    
    Args:
        results: Dictionary with results for each method
        
    Returns:
        DataFrame with metrics for each method
    """
    metrics_data = []
    
    for method_name, method_results in results.items():
        metrics = method_results['metrics']
        metrics_data.append({
            'Method': method_name,
            'Mean Error (°)': metrics.get('mean_error', float('nan')),
            'Median Error (°)': metrics.get('median_error', float('nan')),
            'Max Error (°)': metrics.get('max_error', float('nan')),
            'Data Points': metrics.get('error_count', 0)
        })
    
    return pd.DataFrame(metrics_data)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare compass calibration methods')
    parser.add_argument('--data-dir', default='./data', help='Directory containing data files')
    parser.add_argument('--output', default='./calibration_results', help='Output directory')
    parser.add_argument('--wheel-diameter', type=float, default=130, help='Wheel diameter in mm')
    parser.add_argument('--wheel-base', type=float, default=200, help='Wheel base in mm')
    
    args = parser.parse_args()
    
    # Automatically discover files in the data directory
    data_dir = args.data_dir
    
    # Find files matching the patterns
    gps_files = glob.glob(os.path.join(data_dir, "gps_data_*.csv"))
    imu_files = glob.glob(os.path.join(data_dir, "imu_data_*.csv"))
    rpm_files = glob.glob(os.path.join(data_dir, "control_data_*.csv"))
    
    if not gps_files:
        raise FileNotFoundError(f"No GPS data file found in {data_dir}. Expected pattern: gps_data_*.csv")
    if not imu_files:
        raise FileNotFoundError(f"No IMU data file found in {data_dir}. Expected pattern: imu_data_*.csv")
    if not rpm_files:
        raise FileNotFoundError(f"No control/RPM data file found in {data_dir}. Expected pattern: control_data_*.csv")
    
    # Use the first matching file of each type
    gps_file = gps_files[0]
    imu_file = imu_files[0]
    rpm_file = rpm_files[0]
    
    print(f"Using files:\n  GPS: {os.path.basename(gps_file)}\n  IMU: {os.path.basename(imu_file)}\n  RPM: {os.path.basename(rpm_file)}")
    
    results = compare_calibration_methods(
        gps_file, imu_file, rpm_file, args.output,
        args.wheel_diameter, args.wheel_base
    )