import numpy as np
import math
from scipy.optimize import least_squares
from filterpy.kalman import KalmanFilter

class CompassCalibrator:
    """
    A class to calibrate compass (magnetometer) data using GPS ground truth.
    
    This calibrator handles:
    - Time alignment between GPS and magnetometer data
    - Calculation of ground truth headings from GPS data
    - Detection and filtering of magnetic anomalies
    - Outlier removal
    - Calibration to compensate for hard and soft iron effects
    - Kalman filtering for smoothed outputs
    """
    
    def __init__(self, min_speed_threshold=1.0, max_heading_change_rate=45.0, 
                 outlier_threshold=3.0, time_alignment_threshold=0.5,
                 use_kalman_filter=True):
        """
        Initialize the compass calibrator
        
        Args:
            min_speed_threshold: Minimum speed (m/s) to consider GPS headings reliable
            max_heading_change_rate: Maximum heading change rate (deg/s) for outlier detection
            outlier_threshold: Threshold in standard deviations for outlier detection
            time_alignment_threshold: Maximum time difference (s) for timestamp alignment
            use_kalman_filter: Whether to apply Kalman filtering for smoothing
        """
        self.min_speed_threshold = min_speed_threshold
        self.max_heading_change_rate = max_heading_change_rate
        self.outlier_threshold = outlier_threshold
        self.time_alignment_threshold = time_alignment_threshold
        self.use_kalman_filter = use_kalman_filter
        
        self.calibration_params = None
        self.validation_metrics = None
    
    def calibrate(self, gps_data, mag_data):
        """
        Main calibration function
        
        Args:
            gps_data: List of dictionaries with keys 'latitude', 'longitude', 'timestamp'
            mag_data: List of dictionaries with keys 'x', 'y', 'z', 'timestamp'
            
        Returns:
            List of dictionaries with keys 'timestamp' and 'heading' (in degrees, NED frame)
        """
        # Step 1: Preprocess and align data
        aligned_data = self._align_timestamps(gps_data, mag_data)
        
        # Step 2: Calculate GPS headings and speeds
        aligned_data = self._calculate_gps_headings(aligned_data)
        
        # Step 3: Detect magnetic anomalies
        aligned_data = self._detect_magnetic_anomalies(aligned_data)
        
        # Step 4: Filter out data points where GPS heading is unreliable or magnetic anomalies exist
        filtered_data = [entry for entry in aligned_data 
                         if entry.get('speed', 0) >= self.min_speed_threshold and 
                         not entry.get('is_magnetic_anomaly', False)]
        
        # Step 5: Calculate raw magnetometer headings
        filtered_data = self._calculate_mag_headings(filtered_data)
        
        # Step 6: Remove outliers based on heading differences
        cleaned_data = self._remove_heading_outliers(filtered_data)
        
        # Step 7: Perform calibration if we have enough clean data
        if len(cleaned_data) < 10:
            raise ValueError(f"Not enough clean data points for calibration ({len(cleaned_data)} < 10)")
            
        self.calibration_params = self._perform_calibration(cleaned_data)
        
        # Step 8: Apply calibration to all magnetometer readings
        calibrated_headings = self._apply_calibration(aligned_data)
        
        # Step 9: Apply Kalman filter for smoothing (if enabled)
        if self.use_kalman_filter:
            calibrated_headings = self._apply_kalman_filter(calibrated_headings)
        
        # Step 10: Validate calibration
        self.validation_metrics = self._validate_calibration(aligned_data, calibrated_headings)
        
        return calibrated_headings
    
    def _align_timestamps(self, gps_data, mag_data):
        """
        Align GPS and magnetometer data based on timestamps using interpolation
        
        Returns:
            List of dictionaries with aligned data
        """
        # Sort data by timestamp
        gps_data = sorted(gps_data, key=lambda x: x['timestamp'])
        mag_data = sorted(mag_data, key=lambda x: x['timestamp'])
        
        # For each magnetometer reading, find or interpolate GPS data
        aligned_data = []
        
        for mag_entry in mag_data:
            mag_ts = mag_entry['timestamp']
            
            # Find surrounding GPS points
            lower_gps = None
            upper_gps = None
            
            for gps_entry in gps_data:
                if gps_entry['timestamp'] <= mag_ts:
                    lower_gps = gps_entry
                elif gps_entry['timestamp'] > mag_ts:
                    upper_gps = gps_entry
                    break
            
            # Skip if we don't have enough GPS data for interpolation
            if lower_gps is None or upper_gps is None:
                # Unless it's very close to an endpoint
                if lower_gps is not None and mag_ts - lower_gps['timestamp'] < self.time_alignment_threshold:
                    gps_lat = lower_gps['latitude']
                    gps_lon = lower_gps['longitude']
                    is_interpolated = True
                elif upper_gps is not None and upper_gps['timestamp'] - mag_ts < self.time_alignment_threshold:
                    gps_lat = upper_gps['latitude']
                    gps_lon = upper_gps['longitude']
                    is_interpolated = True
                else:
                    continue
            else:
                # Perform linear interpolation
                lower_ts = lower_gps['timestamp']
                upper_ts = upper_gps['timestamp']
                
                # Interpolation weight
                weight = (mag_ts - lower_ts) / (upper_ts - lower_ts) if upper_ts != lower_ts else 0
                
                # Interpolate latitude and longitude
                gps_lat = lower_gps['latitude'] + weight * (upper_gps['latitude'] - lower_gps['latitude'])
                gps_lon = lower_gps['longitude'] + weight * (upper_gps['longitude'] - lower_gps['longitude'])
                is_interpolated = weight != 0 and weight != 1
            
            aligned_entry = {
                'timestamp': mag_ts,
                'latitude': gps_lat,
                'longitude': gps_lon,
                'mag_x': mag_entry['x'],
                'mag_y': mag_entry['y'],
                'mag_z': mag_entry['z'],
                'is_interpolated_gps': is_interpolated
            }
            aligned_data.append(aligned_entry)
        
        return aligned_data
    
    def _calculate_gps_headings(self, aligned_data):
        """
        Calculate headings from GPS data
        
        Returns:
            Data with GPS headings and speeds added
        """
        for i in range(1, len(aligned_data)):
            prev_entry = aligned_data[i-1]
            curr_entry = aligned_data[i]
            
            # Skip if either entry has interpolated GPS data
            if prev_entry.get('is_interpolated_gps') or curr_entry.get('is_interpolated_gps'):
                continue
            
            # Calculate time difference
            time_diff = curr_entry['timestamp'] - prev_entry['timestamp']
            if time_diff <= 0:
                continue
                
            # Convert to radians
            lat1, lon1 = math.radians(prev_entry['latitude']), math.radians(prev_entry['longitude'])
            lat2, lon2 = math.radians(curr_entry['latitude']), math.radians(curr_entry['longitude'])
            
            # Calculate heading
            y = math.sin(lon2 - lon1) * math.cos(lat2)
            x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
            heading = math.atan2(y, x)
            
            # Convert to degrees and normalize to 0-360
            heading = math.degrees(heading)
            heading = (heading + 360) % 360
            
            # Calculate distance using haversine formula
            a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            distance = 6371000 * c  # Earth radius in meters
            
            # Calculate speed
            speed = distance / time_diff
            
            # Store heading and speed
            curr_entry['gps_heading'] = heading
            curr_entry['speed'] = speed
        
        return aligned_data
    
    def _detect_magnetic_anomalies(self, data):
        """
        Detect magnetic anomalies like proximity to metal structures
        
        Returns:
            Data with anomaly flags
        """
        # Extract magnetic field strengths
        field_strengths = []
        for entry in data:
            # Calculate total magnetic field strength
            strength = np.sqrt(entry['mag_x']**2 + entry['mag_y']**2 + entry['mag_z']**2)
            entry['field_strength'] = strength
            field_strengths.append(strength)
        
        if not field_strengths:
            return data
            
        # Calculate statistics
        mean_strength = np.mean(field_strengths)
        std_strength = np.std(field_strengths)
        
        # Flag anomalies
        for entry in data:
            # Check if field strength is anomalous
            is_anomaly = abs(entry['field_strength'] - mean_strength) > self.outlier_threshold * std_strength
            
            # Check if field strength is very low (near zero)
            is_weak = entry['field_strength'] < mean_strength * 0.2
            
            entry['is_magnetic_anomaly'] = is_anomaly or is_weak
        
        return data
    
    def _calculate_mag_headings(self, data):
        """
        Calculate headings from magnetometer data in the NED frame
        According to the provided schema:
        - Positive X = Vertical Up
        - Positive Y = West
        - Positive Z = South
        
        Converting to NED (North-East-Down) frame:
        - North = -Z (since Z points South)
        - East = -Y (since Y points West)
        - Down = -X (since X points Up)
        
        Returns:
            Data with magnetometer headings added
        """
        for entry in data:
            north = -entry['mag_z']
            east = -entry['mag_y']
            
            # Calculate heading (clockwise from North)
            heading = math.degrees(math.atan2(east, north))
            heading = (heading + 360) % 360
            
            entry['mag_heading'] = heading
        
        return data
    
    def _remove_heading_outliers(self, data):
        """
        Remove outliers based on heading differences and change rates
        
        Returns:
            Cleaned data without outliers
        """
        # Calculate heading differences and change rates
        heading_diffs = []
        change_rates = []
        
        for i in range(1, len(data)):
            if 'gps_heading' in data[i] and 'mag_heading' in data[i]:
                # Heading difference
                gps_heading = data[i]['gps_heading']
                mag_heading = data[i]['mag_heading']
                
                # Calculate smallest angle difference
                diff = (mag_heading - gps_heading + 180) % 360 - 180
                heading_diffs.append(abs(diff))
                data[i]['heading_diff'] = diff
                
                # Heading change rate
                if 'gps_heading' in data[i-1]:
                    time_diff = data[i]['timestamp'] - data[i-1]['timestamp']
                    if time_diff > 0:
                        prev_heading = data[i-1]['gps_heading']
                        curr_heading = data[i]['gps_heading']
                        
                        # Calculate smallest angle difference for change
                        heading_change = (curr_heading - prev_heading + 180) % 360 - 180
                        rate = abs(heading_change) / time_diff
                        
                        change_rates.append(rate)
                        data[i]['heading_change_rate'] = rate
        
        if not heading_diffs:
            return []
            
        # Calculate statistics for outlier detection
        mean_diff = np.mean(heading_diffs)
        std_diff = np.std(heading_diffs) if len(heading_diffs) > 1 else 1
        
        # Filter out outliers
        cleaned_data = []
        
        for entry in data:
            # Check heading difference outlier
            is_diff_outlier = False
            if 'heading_diff' in entry:
                is_diff_outlier = abs(entry['heading_diff']) > mean_diff + self.outlier_threshold * std_diff
            
            # Check heading change rate outlier
            is_rate_outlier = False
            if 'heading_change_rate' in entry:
                is_rate_outlier = entry['heading_change_rate'] > self.max_heading_change_rate
            
            # Include only non-outliers
            if not is_diff_outlier and not is_rate_outlier:
                cleaned_data.append(entry)
        
        return cleaned_data
    
    def _perform_calibration(self, data):
        """
        Perform calibration of the magnetometer using GPS headings as reference
        
        Returns:
            Calibration parameters [scale_x, scale_y, scale_z, offset_x, offset_y, offset_z]
        """
        # Extract data for calibration
        mag_x = np.array([entry['mag_x'] for entry in data])
        mag_y = np.array([entry['mag_y'] for entry in data])
        mag_z = np.array([entry['mag_z'] for entry in data])
        gps_headings = np.array([entry['gps_heading'] for entry in data])
        
        # Convert GPS headings to unit vectors in the NED frame
        gps_n = np.cos(np.radians(gps_headings))
        gps_e = np.sin(np.radians(gps_headings))
        
        # Initial parameters [scale_x, scale_y, scale_z, offset_x, offset_y, offset_z]
        initial_params = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        
        def residuals(params):
            scale_x, scale_y, scale_z, offset_x, offset_y, offset_z = params
            
            # Apply calibration to get corrected magnetometer values
            corrected_x = scale_x * mag_x + offset_x
            corrected_y = scale_y * mag_y + offset_y
            corrected_z = scale_z * mag_z + offset_z
            
            # Convert to NED frame
            ned_n = -corrected_z  # North = -Z
            ned_e = -corrected_y  # East = -Y
            
            # Calculate unit vectors
            magnitudes = np.sqrt(ned_n**2 + ned_e**2)
            unit_n = ned_n / magnitudes
            unit_e = ned_e / magnitudes
            
            # Difference between unit vectors
            n_diff = unit_n - gps_n
            e_diff = unit_e - gps_e
            
            return np.concatenate((n_diff, e_diff))
        
        # Perform optimization
        try:
            result = least_squares(residuals, initial_params)
            return result.x
        except Exception as e:
            print(f"Calibration optimization failed: {e}")
            # Return default parameters in case of failure
            return initial_params
    
    def _apply_calibration(self, data):
        """
        Apply calibration to all magnetometer readings
        
        Returns:
            List of dictionaries with keys 'timestamp' and 'heading'
        """
        if self.calibration_params is None:
            raise ValueError("Calibration must be performed before applying it")
        
        scale_x, scale_y, scale_z, offset_x, offset_y, offset_z = self.calibration_params
        
        calibrated_headings = []
        
        for entry in data:
            # Apply calibration
            corrected_x = scale_x * entry['mag_x'] + offset_x
            corrected_y = scale_y * entry['mag_y'] + offset_y
            corrected_z = scale_z * entry['mag_z'] + offset_z
            
            # Convert to NED frame
            north = -corrected_z  # North = -Z
            east = -corrected_y   # East = -Y
            
            # Calculate heading (clockwise from North)
            heading = math.degrees(math.atan2(east, north))
            heading = (heading + 360) % 360
            
            calibrated_headings.append({
                'timestamp': entry['timestamp'],
                'heading': heading
            })
        
        return calibrated_headings
    
    def _apply_kalman_filter(self, calibrated_headings):
        """
        Apply Kalman filtering to smooth the heading estimates
        
        Returns:
            Smoothed headings
        """
        if not calibrated_headings:
            return []
            
        # Initialize Kalman filter
        kf = KalmanFilter(dim_x=2, dim_z=1)  # State: [heading, heading_rate], Measurement: [heading]
        
        # State transition matrix
        dt = 1.0  # Average time step (could be refined)
        kf.F = np.array([[1, dt],
                         [0, 1]])
        
        # Measurement matrix
        kf.H = np.array([[1, 0]])
        
        # Measurement noise
        kf.R = 10  # Adjust based on expected noise level
        
        # Process noise
        kf.Q = np.array([[0.01, 0.01],
                         [0.01, 0.1]])
        
        # Initial state
        kf.x = np.array([[calibrated_headings[0]['heading']], [0]])
        
        # Initial covariance
        kf.P = np.array([[100, 0],
                         [0, 10]])
        
        smoothed_headings = []
        
        for i, entry in enumerate(calibrated_headings):
            # Calculate actual time step if timestamps are available
            if i > 0:
                dt = entry['timestamp'] - calibrated_headings[i-1]['timestamp']
                kf.F = np.array([[1, dt],
                                 [0, 1]])
            
            # Handle heading wraparound (e.g., 359° -> 1°)
            predicted_heading = kf.x[0, 0] % 360
            measured_heading = entry['heading']
            
            # Adjust measurement to minimize the difference (handling 0/360 wrap)
            diff = (measured_heading - predicted_heading + 180) % 360 - 180
            adjusted_measurement = predicted_heading + diff
            
            # Update filter
            kf.predict()
            kf.update(adjusted_measurement)
            
            # Store smoothed heading
            smoothed_heading = kf.x[0, 0] % 360
            
            smoothed_headings.append({
                'timestamp': entry['timestamp'],
                'heading': smoothed_heading
            })
        
        return smoothed_headings
    
    def _validate_calibration(self, original_data, calibrated_headings):
        """
        Validate the calibration by comparing with GPS headings
        
        Returns:
            Validation metrics
        """
        # Build dictionaries for quick lookups
        gps_dict = {}
        for entry in original_data:
            if 'gps_heading' in entry and entry.get('speed', 0) >= self.min_speed_threshold:
                gps_dict[entry['timestamp']] = entry['gps_heading']
        
        cal_dict = {entry['timestamp']: entry['heading'] for entry in calibrated_headings}
        
        # Calculate errors
        errors = []
        for ts in set(gps_dict.keys()).intersection(set(cal_dict.keys())):
            gps_heading = gps_dict[ts]
            cal_heading = cal_dict[ts]
            
            # Calculate smallest angle difference
            error = (cal_heading - gps_heading + 180) % 360 - 180
            errors.append(abs(error))
        
        if not errors:
            return {'error_count': 0, 'mean_error': float('nan'), 'max_error': float('nan')}
        
        return {
            'error_count': len(errors),
            'mean_error': np.mean(errors),
            'median_error': np.median(errors),
            'max_error': np.max(errors)
        }
    
    def get_calibration_params(self):
        """
        Get the calibration parameters
        
        Returns:
            Dictionary with calibration parameters
        """
        if self.calibration_params is None:
            return None
            
        return {
            'scale_x': self.calibration_params[0],
            'scale_y': self.calibration_params[1],
            'scale_z': self.calibration_params[2],
            'offset_x': self.calibration_params[3],
            'offset_y': self.calibration_params[4],
            'offset_z': self.calibration_params[5]
        }
    
    def get_validation_metrics(self):
        """
        Get the validation metrics
        
        Returns:
            Dictionary with validation metrics
        """
        return self.validation_metrics
    
    def apply_calibration_to_new_data(self, mag_data):
        """
        Apply the existing calibration to new magnetometer data
        
        Args:
            mag_data: List of dictionaries with keys 'x', 'y', 'z', 'timestamp'
            
        Returns:
            List of dictionaries with keys 'timestamp' and 'heading' (in degrees, NED frame)
        """
        if self.calibration_params is None:
            raise ValueError("Calibration must be performed before applying it to new data")
            
        # Apply calibration
        calibrated_headings = []
        
        scale_x, scale_y, scale_z, offset_x, offset_y, offset_z = self.calibration_params
        
        for entry in mag_data:
            # Apply calibration
            corrected_x = scale_x * entry['x'] + offset_x
            corrected_y = scale_y * entry['y'] + offset_y
            corrected_z = scale_z * entry['z'] + offset_z
            
            # Convert to NED frame
            north = -corrected_z  # North = -Z
            east = -corrected_y   # East = -Y
            
            # Calculate heading (clockwise from North)
            heading = math.degrees(math.atan2(east, north))
            heading = (heading + 360) % 360
            
            calibrated_headings.append({
                'timestamp': entry['timestamp'],
                'heading': heading
            })
            
        # Apply Kalman filter for smoothing (if enabled)
        if self.use_kalman_filter:
            calibrated_headings = self._apply_kalman_filter(calibrated_headings)
            
        return calibrated_headings


# Example usage
def example():
    # Sample data format
    gps_data = [
        {'latitude': 37.7749, 'longitude': -122.4194, 'timestamp': 1609459200},
        {'latitude': 37.7750, 'longitude': -122.4195, 'timestamp': 1609459201},
        # More GPS data...
    ]
    
    mag_data = [
        {'x': 123, 'y': 456, 'z': 789, 'timestamp': 1609459200},
        {'x': 124, 'y': 457, 'z': 788, 'timestamp': 1609459201},
        # More magnetometer data...
    ]
    
    # Initialize and run calibration
    calibrator = CompassCalibrator()
    calibrated_headings = calibrator.calibrate(gps_data, mag_data)
    
    # Get calibration parameters
    params = calibrator.get_calibration_params()
    print("Calibration parameters:", params)
    
    # Get validation metrics
    metrics = calibrator.get_validation_metrics()
    print("Validation metrics:", metrics)
    
    # Apply to new data
    new_mag_data = [
        {'x': 125, 'y': 458, 'z': 787, 'timestamp': 1609459202},
        # More new magnetometer data...
    ]
    
    new_headings = calibrator.apply_calibration_to_new_data(new_mag_data)
    
    return calibrated_headings, new_headings

if __name__ == "__main__":
    example()