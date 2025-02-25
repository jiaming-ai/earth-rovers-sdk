import os
import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from clear.data_process.compass import CompassCalibrator
import seaborn as sns
from scipy.spatial import ConvexHull

def read_csv_data(file_path, data_type):
    """
    Read data from CSV file
    
    Args:
        file_path: Path to the CSV file
        data_type: Type of data ('gps', 'mag', or 'rpm')
        
    Returns:
        List of dictionaries with the data
    """
    data = []
    
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            entry = {'timestamp': float(row['timestamp'])}
            
            if data_type == 'gps':
                entry['latitude'] = float(row['latitude'])
                entry['longitude'] = float(row['longitude'])
            elif data_type == 'mag':
                entry['x'] = float(row['x'])
                entry['y'] = float(row['y'])
                entry['z'] = float(row['z'])
            elif data_type == 'rpm':
                entry['rpm_1'] = float(row['rpm_1'])
                entry['rpm_2'] = float(row['rpm_2'])
                entry['rpm_3'] = float(row['rpm_3'])
                entry['rpm_4'] = float(row['rpm_4'])
                
            data.append(entry)
    
    return sorted(data, key=lambda x: x['timestamp'])

def compare_calibration_methods(gps_csv, mag_csv, rpm_csv, output_dir='./output',
                              wheel_diameter=130, wheel_base=200):
    """
    Run different calibration methods and compare their results
    
    Args:
        gps_csv: Path to GPS data CSV
        mag_csv: Path to magnetometer data CSV
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
    mag_data = read_csv_data(mag_csv, 'mag')
    rpm_data = read_csv_data(rpm_csv, 'rpm')
    
    print(f"Loaded {len(gps_data)} GPS points, {len(mag_data)} magnetometer points, {len(rpm_data)} RPM points")
    
    # Initialize results dictionary
    results = {}
    
    # 1. Ellipsoid fitting only
    print("\nMethod 1: Ellipsoid fitting only...")
    calibrator1 = CompassCalibrator(use_kalman_filter=False, 
                                  wheel_diameter=wheel_diameter, 
                                  wheel_base=wheel_base)
    
    # Run ellipsoid-only calibration
    ellipsoid_headings = run_ellipsoid_only_calibration(calibrator1, mag_data, gps_data)
    ellipsoid_metrics = calibrator1.get_validation_metrics()
    
    results['Ellipsoid Only'] = {
        'headings': ellipsoid_headings,
        'metrics': ellipsoid_metrics,
        'params': calibrator1.get_calibration_params()
    }
    
    # 2. GPS calibration without Kalman filter
    print("\nMethod 2: GPS calibration without Kalman filter...")
    calibrator2 = CompassCalibrator(use_kalman_filter=False, 
                                  wheel_diameter=wheel_diameter, 
                                  wheel_base=wheel_base)
    gps_headings = calibrator2.calibrate(gps_data, mag_data, None)
    gps_metrics = calibrator2.get_validation_metrics()
    
    results['GPS (No Kalman)'] = {
        'headings': gps_headings,
        'metrics': gps_metrics,
        'params': calibrator2.get_calibration_params()
    }
    
    # 3. GPS calibration with Kalman filter
    print("\nMethod 3: GPS calibration with Kalman filter...")
    calibrator3 = CompassCalibrator(use_kalman_filter=True, 
                                  wheel_diameter=wheel_diameter, 
                                  wheel_base=wheel_base)
    gps_kalman_headings = calibrator3.calibrate(gps_data, mag_data, None)
    gps_kalman_metrics = calibrator3.get_validation_metrics()
    
    results['GPS with Kalman'] = {
        'headings': gps_kalman_headings,
        'metrics': gps_kalman_metrics,
        'params': calibrator3.get_calibration_params()
    }
    
    # 4. RPM with Kalman filter
    print("\nMethod 4: GPS+RPM with Kalman filter...")
    calibrator4 = CompassCalibrator(use_kalman_filter=True, 
                                  wheel_diameter=wheel_diameter, 
                                  wheel_base=wheel_base)
    rpm_kalman_headings = calibrator4.calibrate(gps_data, mag_data, rpm_data)
    rpm_kalman_metrics = calibrator4.get_validation_metrics()
    
    results['GPS+RPM with Kalman'] = {
        'headings': rpm_kalman_headings,
        'metrics': rpm_kalman_metrics,
        'params': calibrator4.get_calibration_params()
    }
    
    # Generate visualizations and comparison metrics
    print("\nGenerating visualizations and comparison metrics...")
    
    # 1. Visualize magnetometer data before and after calibration
    visualize_magnetometer_data(mag_data, calibrator1, os.path.join(output_dir, "magnetometer_calibration.png"))
    
    # 2. Create individual heading visualizations
    for method_name, method_results in results.items():
        safe_method_name = method_name.replace(' ', '_').lower()
        visualize_headings(gps_data, method_results['headings'], 
                         os.path.join(output_dir, f"{safe_method_name}_headings.png"),
                         title=f"Compass Headings - {method_name}",
                         metrics=method_results['metrics'])
    
    # 3. Create combined visualization
    create_combined_visualization(gps_data, results, os.path.join(output_dir, "combined_comparison.png"))
    
    # 4. Create metrics comparison chart
    create_metrics_comparison(results, os.path.join(output_dir, "metrics_comparison.png"))
    
    # 5. Generate metrics table
    metrics_df = generate_metrics_table(results)
    metrics_df.to_csv(os.path.join(output_dir, "calibration_metrics.csv"), index=False)
    
    print("\nCalibration Metrics Comparison:")
    print("-" * 80)
    print(metrics_df.to_string(index=False))
    print("-" * 80)
    
    print(f"\nResults saved to {output_dir}")
    
    return results

def run_ellipsoid_only_calibration(calibrator, mag_data, gps_data):
    """
    Run ellipsoid-only calibration
    
    Args:
        calibrator: CompassCalibrator instance
        mag_data: Magnetometer data
        gps_data: GPS data for validation
    
    Returns:
        Calibrated headings
    """
    # First perform ellipsoid fitting
    ellipsoid_params = calibrator._perform_ellipsoid_fitting(mag_data)
    calibrator.ellipsoid_params = ellipsoid_params
    
    # Apply calibration manually
    calibrated_headings = []
    
    for entry in mag_data:
        # Get parameters
        offset_x = ellipsoid_params['offset_x']
        offset_y = ellipsoid_params['offset_y']
        offset_z = ellipsoid_params['offset_z']
        transform = np.array(ellipsoid_params['transform'])
        
        # Center data (hard iron correction)
        x_centered = entry['x'] - offset_x
        y_centered = entry['y'] - offset_y
        z_centered = entry['z'] - offset_z
        
        # Apply transform (soft iron correction)
        v = np.array([x_centered, y_centered, z_centered])
        calibrated_v = transform.dot(v)
        
        # Convert to NED frame
        north = -calibrated_v[2]  # Convert Z (South) to North
        east = -calibrated_v[1]   # Convert Y (West) to East
        
        # Calculate heading
        heading = math.degrees(math.atan2(east, north))
        heading = (heading + 360) % 360
        
        calibrated_headings.append({
            'timestamp': entry['timestamp'],
            'heading': heading
        })
    
    # Calculate validation metrics
    aligned_data = calibrator._align_timestamps(gps_data, mag_data, None)
    calibrator.validation_metrics = calibrator._validate_calibration(aligned_data, calibrated_headings)
    
    return calibrated_headings

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
    # Create a map of timestamps to GPS positions
    gps_map = {entry['timestamp']: (entry['latitude'], entry['longitude']) 
             for entry in gps_data}
    
    # Match headings with GPS positions
    matched_data = []
    for entry in heading_data:
        ts = entry['timestamp']
        if ts in gps_map:
            lat, lon = gps_map[ts]
            heading = entry['heading']
            matched_data.append((ts, lat, lon, heading))
    
    # Sort by timestamp
    matched_data.sort()
    
    if not matched_data:
        print(f"Warning: No matching data points found for {title}")
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
        metrics_text = (
            f"\nMean Error: {metrics.get('mean_error', 'N/A'):.2f}°, "
            f"Median: {metrics.get('median_error', 'N/A'):.2f}°, "
            f"Max: {metrics.get('max_error', 'N/A'):.2f}°"
        )
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
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(2, 2, figure=fig)
    
    # Get GPS track data
    all_lats = [entry['latitude'] for entry in gps_data]
    all_lons = [entry['longitude'] for entry in gps_data]
    
    # Create a map of timestamps to GPS positions
    gps_map = {entry['timestamp']: (entry['latitude'], entry['longitude']) 
             for entry in gps_data}
    
    # Process each method
    for i, (method_name, method_results) in enumerate(results.items()):
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
        metrics_text = f"Mean Error: {metrics.get('mean_error', 'N/A'):.2f}°"
        
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
    
    for method_name, method_results in results.items():
        metrics = method_results['metrics']
        methods.append(method_name)
        mean_errors.append(metrics.get('mean_error', 0))
        median_errors.append(metrics.get('median_error', 0))
        max_errors.append(metrics.get('max_error', 0))
    
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
    plt.ylim(0, max(max_errors) * 1.1)  # Add some headroom
    plt.legend(title='Metric')
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=15)
    
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
    parser.add_argument('--gps', required=True, help='Path to GPS CSV file')
    parser.add_argument('--mag', required=True, help='Path to magnetometer CSV file')
    parser.add_argument('--rpm', required=True, help='Path to RPM CSV file')
    parser.add_argument('--output', default='./calibration_results', help='Output directory')
    parser.add_argument('--wheel-diameter', type=float, default=130, help='Wheel diameter in mm')
    parser.add_argument('--wheel-base', type=float, default=200, help='Wheel base in mm')
    
    args = parser.parse_args()
    
    results = compare_calibration_methods(
        args.gps, args.mag, args.rpm, args.output,
        args.wheel_diameter, args.wheel_base
    )