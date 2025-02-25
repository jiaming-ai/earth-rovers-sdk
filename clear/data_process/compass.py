import numpy as np
import math
from scipy.optimize import least_squares
from scipy.linalg import eigh
from filterpy.kalman import KalmanFilter

class CompassCalibrator:
    """
    A class to calibrate compass (magnetometer) data using multiple methods:
    1. Ellipsoid fitting for hard/soft iron calibration
    2. GPS ground truth validation and calibration
    3. Wheel encoder (RPM) data integration
    
    This calibrator handles:
    - Time alignment between GPS, magnetometer, and RPM data
    - Calculation of ground truth headings from GPS data
    - Detection and filtering of magnetic anomalies
    - Outlier removal
    - Calibration to compensate for hard and soft iron effects
    - Kalman filtering with sensor fusion for smoothed outputs
    """
    
    def __init__(self, min_speed_threshold=1.0, max_heading_change_rate=45.0, 
                 outlier_threshold=3.0, time_alignment_threshold=0.5,
                 use_kalman_filter=True, wheel_diameter=130, wheel_base=200):
        """
        Initialize the compass calibrator
        
        Args:
            min_speed_threshold: Minimum speed (m/s) to consider GPS headings reliable
            max_heading_change_rate: Maximum heading change rate (deg/s) for outlier detection
            outlier_threshold: Threshold in standard deviations for outlier detection
            time_alignment_threshold: Maximum time difference (s) for timestamp alignment
            use_kalman_filter: Whether to apply Kalman filtering for smoothing
            wheel_diameter: Diameter of the wheels in mm
            wheel_base: Distance between left and right wheels in mm
        """
        self.min_speed_threshold = min_speed_threshold
        self.max_heading_change_rate = max_heading_change_rate
        self.outlier_threshold = outlier_threshold
        self.time_alignment_threshold = time_alignment_threshold
        self.use_kalman_filter = use_kalman_filter
        
        # Wheel parameters for RPM calculations (convert to meters)
        self.wheel_diameter = wheel_diameter / 1000.0  # mm to m
        self.wheel_base = wheel_base / 1000.0  # mm to m
        self.wheel_radius = self.wheel_diameter / 2.0
        
        # Calibration parameters
        self.ellipsoid_params = None  # From ellipsoid fitting
        self.gps_cal_params = None    # From GPS ground truth
        self.validation_metrics = None
    
    def calibrate(self, gps_data, mag_data, rpm_data=None):
        """
        Main calibration function
        
        Args:
            gps_data: List of dictionaries with keys 'latitude', 'longitude', 'timestamp'
            mag_data: List of dictionaries with keys 'x', 'y', 'z', 'timestamp'
            rpm_data: List of dictionaries with keys 'rpm_1', 'rpm_2', 'rpm_3', 'rpm_4', 'timestamp'
            
        Returns:
            List of dictionaries with keys 'timestamp' and 'heading' (in degrees, NED frame)
        """
        # Step 1: Perform ellipsoid fitting on magnetometer data
        self.ellipsoid_params = self._perform_ellipsoid_fitting(mag_data)
        
        # Apply initial ellipsoid calibration to magnetometer data
        mag_data_calibrated = self._apply_ellipsoid_calibration(mag_data)
        
        # Step 2: Preprocess and align data
        aligned_data = self._align_timestamps(gps_data, mag_data_calibrated, rpm_data)
        
        # Step 3: Calculate GPS headings and speeds
        aligned_data = self._calculate_gps_headings(aligned_data)
        
        # Step 4: Calculate RPM-based angular velocity (if available)
        if rpm_data:
            aligned_data = self._calculate_rpm_angular_velocity(aligned_data)
        
        # Step 5: Detect magnetic anomalies
        aligned_data = self._detect_magnetic_anomalies(aligned_data)
        
        # Step 6: Filter out data points where GPS heading is unreliable or magnetic anomalies exist
        filtered_data = [entry for entry in aligned_data 
                         if entry.get('speed', 0) >= self.min_speed_threshold and 
                         not entry.get('is_magnetic_anomaly', False)]
        
        # Step 7: Calculate raw magnetometer headings
        filtered_data = self._calculate_mag_headings(filtered_data)
        
        # Step 8: Remove outliers based on heading differences
        cleaned_data = self._remove_heading_outliers(filtered_data)
        
        # Step 9: Perform fine-tuning calibration with GPS if we have enough clean data
        if len(cleaned_data) >= 10:
            self.gps_cal_params = self._perform_gps_calibration(cleaned_data)
            # Use combined calibration parameters
            self.calibration_params = self._combine_calibration_params()
        else:
            print(f"Warning: Not enough clean data points for GPS calibration ({len(cleaned_data)} < 10)")
            # Use only ellipsoid calibration parameters
            self.calibration_params = self.ellipsoid_params
            
        # Step 10: Apply final calibration to all magnetometer readings
        calibrated_headings = self._apply_calibration(aligned_data)
        
        # Step 11: Apply Kalman filter for smoothing (if enabled)
        if self.use_kalman_filter:
            calibrated_headings = self._apply_kalman_filter(aligned_data, calibrated_headings)
        
        # Step 12: Validate calibration
        self.validation_metrics = self._validate_calibration(aligned_data, calibrated_headings)
        
        return calibrated_headings
    
    def _perform_ellipsoid_fitting(self, mag_data):
        """
        Perform ellipsoid fitting to calibrate magnetometer data
        
        Args:
            mag_data: List of magnetometer readings
            
        Returns:
            Dictionary with calibration parameters
        """
        # Extract magnetic field readings
        x = np.array([entry['x'] for entry in mag_data])
        y = np.array([entry['y'] for entry in mag_data])
        z = np.array([entry['z'] for entry in mag_data])
        
        # Stack data
        D = np.column_stack((x**2, y**2, z**2, 2*y*z, 2*x*z, 2*x*y, 2*x, 2*y, 2*z, np.ones_like(x)))
        
        # Solve least squares for ellipsoid parameters
        try:
            # S is the solution to the least squares problem
            S, residuals, rank, singular_values = np.linalg.lstsq(D, np.ones_like(x), rcond=None)
            
            # Extract the ellipsoid parameters
            A = np.array([
                [S[0], S[5], S[4]],
                [S[5], S[1], S[3]],
                [S[4], S[3], S[2]]
            ])
            
            # Center (hard iron offset)
            v = np.array([S[6], S[7], S[8]])
            offset = -np.linalg.inv(A).dot(v) / 2
            
            # Get the radii and transformation matrix
            T, radii = self._get_ellipsoid_params(A, offset)
            
            # Normalize to get a sphere
            R = np.diag(1/np.sqrt(radii))
            
            # Combined transformation
            W = R.dot(T)
            
            return {
                'offset_x': float(offset[0]),
                'offset_y': float(offset[1]),
                'offset_z': float(offset[2]),
                'transform': W.tolist(),
                'method': 'ellipsoid'
            }
        except Exception as e:
            print(f"Ellipsoid fitting failed: {e}")
            # Return identity parameters if fitting fails
            return {
                'offset_x': 0.0,
                'offset_y': 0.0,
                'offset_z': 0.0,
                'transform': np.eye(3).tolist(),
                'method': 'identity'
            }
    
    def _get_ellipsoid_params(self, A, offset):
        """
        Extract ellipsoid parameters from matrix form
        
        Args:
            A: 3x3 quadratic form matrix
            offset: center of ellipsoid
            
        Returns:
            T: transformation matrix, radii: squared radii of ellipsoid
        """
        # Calculate the matrix containing offset
        C = np.eye(3)
        
        # Get eigenvalues and eigenvectors
        eigvals, eigvecs = eigh(A)
        
        # Radii are inversely proportional to the square root of eigenvalues
        radii = 1.0 / eigvals
        
        # Transform matrix from ellipsoid to sphere
        T = eigvecs
        
        return T, radii
    
    def _apply_ellipsoid_calibration(self, mag_data):
        """
        Apply ellipsoid calibration to magnetometer data
        
        Args:
            mag_data: List of magnetometer readings
            
        Returns:
            Calibrated magnetometer data
        """
        if self.ellipsoid_params is None:
            return mag_data
            
        offset_x = self.ellipsoid_params['offset_x']
        offset_y = self.ellipsoid_params['offset_y']
        offset_z = self.ellipsoid_params['offset_z']
        transform = np.array(self.ellipsoid_params['transform'])
        
        calibrated_data = []
        
        for entry in mag_data:
            # Center the data (remove hard iron)
            x_centered = entry['x'] - offset_x
            y_centered = entry['y'] - offset_y
            z_centered = entry['z'] - offset_z
            
            # Apply transformation (correct soft iron)
            v = np.array([x_centered, y_centered, z_centered])
            calibrated_v = transform.dot(v)
            
            # Create new entry with calibrated values
            new_entry = entry.copy()
            new_entry['x'] = float(calibrated_v[0])
            new_entry['y'] = float(calibrated_v[1])
            new_entry['z'] = float(calibrated_v[2])
            
            calibrated_data.append(new_entry)
            
        return calibrated_data
    
    def _align_timestamps(self, gps_data, mag_data, rpm_data=None):
        """
        Align GPS, magnetometer, and RPM data based on timestamps using interpolation
        
        Returns:
            List of dictionaries with aligned data
        """
        # Sort data by timestamp
        gps_data = sorted(gps_data, key=lambda x: x['timestamp'])
        mag_data = sorted(mag_data, key=lambda x: x['timestamp'])
        
        if rpm_data:
            rpm_data = sorted(rpm_data, key=lambda x: x['timestamp'])
        
        # For each magnetometer reading, find or interpolate GPS and RPM data
        aligned_data = []
        
        for mag_entry in mag_data:
            mag_ts = mag_entry['timestamp']
            
            # Find or interpolate GPS data
            gps_lat, gps_lon, is_gps_interpolated = self._interpolate_data_point(
                gps_data, mag_ts, 'latitude', 'longitude')
            
            # Prepare aligned entry
            aligned_entry = {
                'timestamp': mag_ts,
                'latitude': gps_lat,
                'longitude': gps_lon,
                'mag_x': mag_entry['x'],
                'mag_y': mag_entry['y'],
                'mag_z': mag_entry['z'],
                'is_interpolated_gps': is_gps_interpolated
            }
            
            # Add RPM data if available
            if rpm_data:
                rpm_values, is_rpm_interpolated = self._interpolate_rpm_data(rpm_data, mag_ts)
                aligned_entry.update(rpm_values)
                aligned_entry['is_interpolated_rpm'] = is_rpm_interpolated
                
            aligned_data.append(aligned_entry)
        
        return aligned_data
    
    def _interpolate_data_point(self, data_list, target_ts, *fields):
        """
        Interpolate data fields at a target timestamp
        
        Args:
            data_list: List of data points
            target_ts: Target timestamp
            fields: Fields to interpolate
            
        Returns:
            Tuple of interpolated values and boolean indicating if interpolation was used
        """
        # Find surrounding data points
        lower_point = None
        upper_point = None
        
        for point in data_list:
            if point['timestamp'] <= target_ts:
                lower_point = point
            elif point['timestamp'] > target_ts:
                upper_point = point
                break
        
        # Check if we have enough data for interpolation
        is_interpolated = True
        
        if lower_point is None and upper_point is None:
            # No data points available
            return tuple([None] * len(fields) + [True])
            
        elif lower_point is None:
            # Only have data after target time
            if upper_point['timestamp'] - target_ts < self.time_alignment_threshold:
                # Close enough to use the upper point
                return tuple([upper_point[field] for field in fields] + [True])
            else:
                # Too far to use
                return tuple([None] * len(fields) + [True])
                
        elif upper_point is None:
            # Only have data before target time
            if target_ts - lower_point['timestamp'] < self.time_alignment_threshold:
                # Close enough to use the lower point
                return tuple([lower_point[field] for field in fields] + [True])
            else:
                # Too far to use
                return tuple([None] * len(fields) + [True])
                
        elif lower_point['timestamp'] == target_ts:
            # Exact match with lower point
            return tuple([lower_point[field] for field in fields] + [False])
            
        elif upper_point['timestamp'] == target_ts:
            # Exact match with upper point
            return tuple([upper_point[field] for field in fields] + [False])
            
        else:
            # Perform linear interpolation
            lower_ts = lower_point['timestamp']
            upper_ts = upper_point['timestamp']
            
            # Check if time difference is too large
            if upper_ts - lower_ts > self.time_alignment_threshold * 10:
                # Too sparse data
                if target_ts - lower_ts < self.time_alignment_threshold:
                    return tuple([lower_point[field] for field in fields] + [True])
                elif upper_ts - target_ts < self.time_alignment_threshold:
                    return tuple([upper_point[field] for field in fields] + [True])
                else:
                    return tuple([None] * len(fields) + [True])
            
            # Calculate interpolation weight
            weight = (target_ts - lower_ts) / (upper_ts - lower_ts)
            
            # Interpolate values
            values = []
            for field in fields:
                interpolated = lower_point[field] + weight * (upper_point[field] - lower_point[field])
                values.append(interpolated)
                
            return tuple(values + [True])
    
    def _interpolate_rpm_data(self, rpm_data, target_ts):
        """
        Interpolate RPM data at a target timestamp
        
        Args:
            rpm_data: List of RPM readings
            target_ts: Target timestamp
            
        Returns:
            Dictionary with RPM values and boolean indicating if interpolation was used
        """
        fields = ['rpm_1', 'rpm_2', 'rpm_3', 'rpm_4']
        results = self._interpolate_data_point(rpm_data, target_ts, *fields)
        
        values = results[:-1]  # All but the last element (is_interpolated)
        is_interpolated = results[-1]
        
        # Return as dictionary
        if all(v is not None for v in values):
            return {field: value for field, value in zip(fields, values)}, is_interpolated
        else:
            return {}, True
    
    def _calculate_gps_headings(self, aligned_data):
        """
        Calculate headings from GPS data
        
        Returns:
            Data with GPS headings and speeds added
        """
        for i in range(1, len(aligned_data)):
            prev_entry = aligned_data[i-1]
            curr_entry = aligned_data[i]
            
            # Skip if either entry has interpolated GPS data or missing GPS data
            if (prev_entry.get('is_interpolated_gps', True) or curr_entry.get('is_interpolated_gps', True) or
                'latitude' not in prev_entry or 'longitude' not in prev_entry or
                'latitude' not in curr_entry or 'longitude' not in curr_entry or
                prev_entry['latitude'] is None or prev_entry['longitude'] is None or
                curr_entry['latitude'] is None or curr_entry['longitude'] is None):
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
    
    def _calculate_rpm_angular_velocity(self, aligned_data):
        """
        Calculate angular velocity from RPM data
        
        Returns:
            Data with RPM-based angular velocity added
        """
        for entry in aligned_data:
            # Skip if RPM data is missing
            if ('rpm_1' not in entry or 'rpm_2' not in entry or 
                'rpm_3' not in entry or 'rpm_4' not in entry):
                continue
                
            # Convert RPM to rad/s
            # For a differential drive, we use the average of left and right wheels
            rpm_left = (entry['rpm_1'] + entry['rpm_3']) / 2  # Front-Left and Rear-Left
            rpm_right = (entry['rpm_2'] + entry['rpm_4']) / 2  # Front-Right and Rear-Right
            
            # Convert RPM to linear velocity (m/s)
            v_left = (rpm_left / 60) * 2 * math.pi * self.wheel_radius
            v_right = (rpm_right / 60) * 2 * math.pi * self.wheel_radius
            
            # Calculate angular velocity (rad/s)
            angular_velocity = (v_right - v_left) / self.wheel_base
            
            # Convert to deg/s
            angular_velocity_deg = math.degrees(angular_velocity)
            
            entry['rpm_angular_velocity'] = angular_velocity_deg
            
            # Calculate linear velocity (m/s)
            linear_velocity = (v_right + v_left) / 2
            entry['rpm_linear_velocity'] = linear_velocity
        
        # Calculate heading change based on angular velocity
        for i in range(1, len(aligned_data)):
            prev_entry = aligned_data[i-1]
            curr_entry = aligned_data[i]
            
            # Skip if angular velocity not available
            if ('rpm_angular_velocity' not in prev_entry or 
                'rpm_angular_velocity' not in curr_entry):
                continue
                
            # Calculate time difference
            time_diff = curr_entry['timestamp'] - prev_entry['timestamp']
            if time_diff <= 0:
                continue
            
            # Integrate angular velocity to get heading change
            avg_angular_vel = (prev_entry['rpm_angular_velocity'] + curr_entry['rpm_angular_velocity']) / 2
            heading_change = avg_angular_vel * time_diff
            
            # If previous entry has a heading, calculate current heading
            if 'rpm_heading' in prev_entry:
                curr_entry['rpm_heading'] = (prev_entry['rpm_heading'] + heading_change) % 360
            elif 'gps_heading' in prev_entry:
                # Initialize RPM heading with GPS heading if available
                curr_entry['rpm_heading'] = (prev_entry['gps_heading'] + heading_change) % 360
        
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
    
    def _perform_gps_calibration(self, data):
        """
        Perform final calibration using GPS headings as reference
        
        Returns:
            Calibration parameters
        """
        # Extract data for calibration
        mag_headings = np.array([entry['mag_heading'] for entry in data])
        gps_headings = np.array([entry['gps_heading'] for entry in data])
        
        # Initial parameters [scale, offset]
        initial_params = [1.0, 0.0]
        
        def residuals(params):
            scale, offset = params
            
            # Apply calibration
            calibrated_headings = (scale * mag_headings + offset) % 360
            
            # Calculate smallest angle differences
            errors = np.array([((c - g + 180) % 360 - 180) for c, g in zip(calibrated_headings, gps_headings)])
            
            return errors
        
        # Perform optimization
        try:
            result = least_squares(residuals, initial_params, loss='soft_l1')
            return {
                'scale': float(result.x[0]),
                'offset': float(result.x[1]),
                'method': 'gps'
            }
        except Exception as e:
            print(f"GPS calibration optimization failed: {e}")
            # Return default parameters in case of failure
            return {
                'scale': 1.0,
                'offset': 0.0,
                'method': 'default'
            }
    
    def _combine_calibration_params(self):
        """
        Combine ellipsoid and GPS calibration parameters
        
        Returns:
            Combined calibration parameters
        """
        if self.ellipsoid_params is None:
            return self.gps_cal_params
            
        if self.gps_cal_params is None:
            return self.ellipsoid_params
            
        # For now, we'll just use both separately - ellipsoid for hard/soft iron,
        # and GPS for final heading adjustment
        combined = {
            'ellipsoid': self.ellipsoid_params,
            'gps': self.gps_cal_params,
            'method': 'combined'
        }
        
        return combined
    
    def _apply_calibration(self, data):
        """
        Apply calibration to all magnetometer readings
        
        Returns:
            List of dictionaries with keys 'timestamp' and 'heading'
        """
        if self.calibration_params is None:
            raise ValueError("Calibration must be performed before applying it")
        
        calibrated_headings = []
        
        for entry in data:
            # Apply ellipsoid calibration first
            if 'ellipsoid' in self.calibration_params:
                offset_x = self.calibration_params['ellipsoid']['offset_x']
                offset_y = self.calibration_params['ellipsoid']['offset_y']
                offset_z = self.calibration_params['ellipsoid']['offset_z']
                transform = np.array(self.calibration_params['ellipsoid']['transform'])
                
                # Apply hard iron correction
                x_centered = entry['mag_x'] - offset_x
                y_centered = entry['mag_y'] - offset_y
                z_centered = entry['mag_z'] - offset_z
                
                # Apply soft iron correction
                v = np.array([x_centered, y_centered, z_centered])
                calibrated_v = transform.dot(v)
                
                # Convert to NED frame
                north = -calibrated_v[2]  # North = -Z
                east = -calibrated_v[1]   # East = -Y
            else:
                # Convert directly to NED frame
                north = -entry['mag_z']
                east = -entry['mag_y']
            
            # Calculate heading (clockwise from North)
            heading = math.degrees(math.atan2(east, north))
            heading = (heading + 360) % 360
            
            # Apply GPS scale/offset calibration if available
            if 'gps' in self.calibration_params:
                scale = self.calibration_params['gps']['scale']
                offset = self.calibration_params['gps']['offset']
                heading = (scale * heading + offset) % 360
            
            calibrated_headings.append({
                'timestamp': entry['timestamp'],
                'heading': heading
            })
        
        return calibrated_headings
    
    def _apply_kalman_filter(self, aligned_data, calibrated_headings):
        """
        Apply Kalman filtering to smooth the heading estimates
        
        Args:
            aligned_data: Original aligned data with RPM and GPS info
            calibrated_headings: Calibrated magnetometer headings
            
        Returns:
            Smoothed headings
        """
        if not calibrated_headings:
            return []
            
        # Convert to dict for easier lookup
        heading_dict = {entry['timestamp']: entry['heading'] for entry in calibrated_headings}
        
        # State vector: [heading, heading_rate]
        n_states = 2
        # Measurement vector: [mag_heading, gps_heading (optional), rpm_heading (optional)]
        n_measurements = 3  # Maximum number of measurements
        
        # Initialize Kalman filter
        kf = KalmanFilter(dim_x=n_states, dim_z=n_measurements)
        
        # Initial state - use first heading
        first_ts = calibrated_headings[0]['timestamp']
        first_heading = calibrated_headings[0]['heading']
        kf.x = np.array([[first_heading], [0]])  # [heading, heading_rate]
        
        # Initial covariance
        kf.P = np.diag([100, 10])  # High uncertainty initially
        
        # Process noise (uncertainty in how heading evolves)
        kf.Q = np.diag([0.1, 1])  # Adjust based on expected system dynamics
        
        # Measurement noise (uncertainties in each sensor)
        # Order: [mag_heading, gps_heading, rpm_heading]
        kf.R = np.diag([10, 5, 20])  # GPS more reliable than magnetometer, RPM less reliable
        
        smoothed_headings = []
        prev_ts = first_ts
        
        for i, entry in enumerate(aligned_data):
            current_ts = entry['timestamp']
            
            # Skip first entry as it was used for initialization
            if i == 0:
                smoothed_headings.append({
                    'timestamp': current_ts,
                    'heading': first_heading
                })
                continue
                
            # Time update
            dt = current_ts - prev_ts
            if dt <= 0:
                # Skip entries with non-positive time difference
                continue
                
            # Update state transition matrix for current dt
            kf.F = np.array([
                [1, dt],
                [0, 1]
            ])
            
            # Process noise increases with time
            kf.Q = np.array([
                [0.1*dt, 0.05*dt],
                [0.05*dt, 0.1*dt]
            ])
            
            # Predict
            kf.predict()
            
            # Prepare measurement and measurement matrix
            measurements = []
            H_rows = []
            
            # 1. Magnetometer heading (always available after calibration)
            if current_ts in heading_dict:
                mag_heading = heading_dict[current_ts]
                # Handle heading wraparound
                predicted_heading = kf.x[0, 0] % 360
                # Adjust measurement to minimize difference
                diff = (mag_heading - predicted_heading + 180) % 360 - 180
                adjusted_mag_heading = predicted_heading + diff
                measurements.append(adjusted_mag_heading)
                H_rows.append([1, 0])  # Measurement maps to heading state
            
            # 2. GPS heading (if available and reliable)
            if ('gps_heading' in entry and 
                not entry.get('is_interpolated_gps', True) and 
                entry.get('speed', 0) >= self.min_speed_threshold):
                gps_heading = entry['gps_heading']
                # Handle heading wraparound
                predicted_heading = kf.x[0, 0] % 360
                # Adjust measurement to minimize difference
                diff = (gps_heading - predicted_heading + 180) % 360 - 180
                adjusted_gps_heading = predicted_heading + diff
                measurements.append(adjusted_gps_heading)
                H_rows.append([1, 0])  # Measurement maps to heading state
            
            # 3. RPM heading (if available)
            if 'rpm_heading' in entry:
                rpm_heading = entry['rpm_heading']
                # Handle heading wraparound
                predicted_heading = kf.x[0, 0] % 360
                # Adjust measurement to minimize difference
                diff = (rpm_heading - predicted_heading + 180) % 360 - 180
                adjusted_rpm_heading = predicted_heading + diff
                measurements.append(adjusted_rpm_heading)
                H_rows.append([1, 0])  # Measurement maps to heading state
            
            # 4. RPM angular velocity (if available)
            if 'rpm_angular_velocity' in entry:
                rpm_angular_vel = entry['rpm_angular_velocity']
                measurements.append(rpm_angular_vel)
                H_rows.append([0, 1])  # Measurement maps to heading rate state
            
            # Update if we have measurements
            if measurements:
                # Create measurement vector
                z = np.array(measurements).reshape(-1, 1)
                
                # Create measurement matrix
                H = np.array(H_rows)
                
                # Update measurement noise matrix dimensions
                R_dim = len(measurements)
                R = np.eye(R_dim)
                
                # Set measurement noise based on sensor type
                for j in range(R_dim):
                    if j < len(H_rows):
                        if H_rows[j][0] == 1:  # Heading measurement
                            if j == 0:  # Magnetometer
                                R[j, j] = 10
                            elif j == 1:  # GPS
                                R[j, j] = 5
                            elif j == 2:  # RPM
                                R[j, j] = 20
                        elif H_rows[j][1] == 1:  # Angular velocity measurement
                            R[j, j] = 15
                
                # Update Kalman filter parameters
                kf.H = H
                kf.R = R
                
                # Update state
                kf.update(z)
            
            # Get smoothed heading (normalized to 0-360)
            smoothed_heading = kf.x[0, 0] % 360
            
            # Save result
            smoothed_headings.append({
                'timestamp': current_ts,
                'heading': smoothed_heading
            })
            
            # Update previous timestamp
            prev_ts = current_ts
        
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
        return {
            'ellipsoid': self.ellipsoid_params,
            'gps': self.gps_cal_params,
            'combined': self.calibration_params
        }
    
    def get_validation_metrics(self):
        """
        Get the validation metrics
        
        Returns:
            Dictionary with validation metrics
        """
        return self.validation_metrics
    
    def apply_calibration_to_new_data(self, mag_data, rpm_data=None):
        """
        Apply the existing calibration to new magnetometer data
        
        Args:
            mag_data: List of dictionaries with keys 'x', 'y', 'z', 'timestamp'
            rpm_data: Optional list of dictionaries with RPM data
            
        Returns:
            List of dictionaries with keys 'timestamp' and 'heading' (in degrees, NED frame)
        """
        if self.calibration_params is None:
            raise ValueError("Calibration must be performed before applying it to new data")
        
        # Apply ellipsoid calibration first
        mag_data_calibrated = self._apply_ellipsoid_calibration(mag_data)
        
        # Align with RPM data if provided
        if rpm_data:
            aligned_data = self._align_timestamps([], mag_data_calibrated, rpm_data)
            aligned_data = self._calculate_rpm_angular_velocity(aligned_data)
        else:
            # Create minimal aligned data with just mag readings
            aligned_data = []
            for entry in mag_data_calibrated:
                aligned_entry = {
                    'timestamp': entry['timestamp'],
                    'mag_x': entry['x'],
                    'mag_y': entry['y'],
                    'mag_z': entry['z']
                }
                aligned_data.append(aligned_entry)
        
        # Apply calibration
        calibrated_headings = self._apply_calibration(aligned_data)
        
        # Apply Kalman filter for smoothing (if enabled)
        if self.use_kalman_filter:
            calibrated_headings = self._apply_kalman_filter(aligned_data, calibrated_headings)
            
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
    
    rpm_data = [
        {'rpm_1': 50, 'rpm_2': 50, 'rpm_3': 50, 'rpm_4': 50, 'timestamp': 1609459200},
        {'rpm_1': 55, 'rpm_2': 45, 'rpm_3': 55, 'rpm_4': 45, 'timestamp': 1609459201},
        # More RPM data...
    ]
    
    # Initialize calibrator with robot parameters
    calibrator = CompassCalibrator(wheel_diameter=130, wheel_base=200)
    
    # Perform calibration with all available data
    calibrated_headings = calibrator.calibrate(gps_data, mag_data, rpm_data)
    
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
    
    new_rpm_data = [
        {'rpm_1': 60, 'rpm_2': 40, 'rpm_3': 60, 'rpm_4': 40, 'timestamp': 1609459202},
        # More new RPM data...
    ]
    
    new_headings = calibrator.apply_calibration_to_new_data(new_mag_data, new_rpm_data)
    
    return calibrated_headings, new_headings

if __name__ == "__main__":
    example()