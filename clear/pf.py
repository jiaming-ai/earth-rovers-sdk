import threading
import time
import numpy as np
import math
from collections import deque
import copy
import pyproj
import requests
import logging
from typing import Dict, List, Tuple, Any, Optional

class StateEstimationPF:
    """
    Particle Filter for State Estimation
    The estimated state is in the NED frame (North, East, Down)
    """
    def __init__(self, num_particles=100, wheel_diameter=130, wheel_base=200, log_level=logging.INFO):
        # Set up logging
        self.logger = logging.getLogger("StateEstimationPF")
        self.logger.setLevel(log_level)
        
        # Create console handler if no handlers exist
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        max_queue_size = 1000
        # Robot physical parameters
        self.wheel_diameter = wheel_diameter
        self.wheel_base = wheel_base
        self.wheel_radius = wheel_diameter / 2
        
        # Particle filter parameters
        self.num_particles = num_particles
        self.particles = None  # Will be initialized in init_particles
        self.weights = np.ones(num_particles) / num_particles
        
        # Data storage
        self.sensor_data = {
            "accels": deque(maxlen=max_queue_size),
            "gyros": deque(maxlen=max_queue_size),
            "mags": deque(maxlen=max_queue_size),
            "rpms": deque(maxlen=max_queue_size),
            "gps": deque(maxlen=max_queue_size)
        }
        
        # Last processed timestamps
        self.last_processed = {
            "accels": 0,
            "gyros": 0,
            "mags": 0,
            "rpms": 0,
            "gps": 0
        }
        
        # Current state estimate
        self.current_state = {
            "x": None,
            "y": None,
            "theta": None,
            "timestamp": None
        }
        
        # Noise parameters
        self.motion_noise = {
            "x": 0.1,
            "y": 0.1,
            "theta": 0.05
        }
        
        self.gps_noise = 3.0  # meters
        self.compass_noise = 0.1  # radians
        
        # Threading control
        self.running = False
        self.data_thread = None
        self.pf_thread = None
        self.data_lock = threading.Lock()
        
        # Don't initialize particles yet - wait for first readings
        self.particles = None
        
        # Add reference GPS coordinates (first valid reading will set these)
        self.ref_lat = None
        self.ref_lon = None
        
        # Earth's radius in meters
        self.EARTH_RADIUS = 6371000
        
        # Set up projection for GPS conversions (will be initialized later)
        self.proj = None
        
        # Store update timestamps for frequency calculation
        self.update_timestamps = deque(maxlen=100)
    
    def setup_projection(self):
        """Set up the projection system for GPS conversions"""
        if self.ref_lat is not None and self.ref_lon is not None:
            # Set up a transverse Mercator projection centered on the reference point
            proj_string = f"+proj=tmerc +lat_0={self.ref_lat} +lon_0={self.ref_lon} +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
            self.proj = pyproj.Proj(proj_string)
    
    def init_particles(self):
        """Initialize particles using first GPS and compass readings"""
        # Try to get initial GPS and magnetometer readings
        with self.data_lock:
            gps_reading = self.get_latest_sensor_data("gps")
            mag_reading = self.get_latest_sensor_data("mags")
        
        # Check if we have valid GPS readings
        if not gps_reading:
            self.logger.info("Waiting for valid GPS signal...")
            return False
            
        if not mag_reading:
            self.logger.info("Waiting for valid magnetometer readings...")
            return False
        
        # Initialize position from GPS
        x, y, timestamp = gps_reading
        
        # Calculate initial heading from magnetometer
        mx, my, mz, _ = mag_reading
        initial_theta = math.atan2(-my, -mz)  # Convert magnetometer reading to heading
        
        # Initialize particles around these measurements with vectorized operations
        self.particles = np.zeros((self.num_particles, 3))
        
        # Create random noise vectors all at once
        pos_noise = np.random.normal(0, self.gps_noise, (self.num_particles, 2))
        theta_noise = np.random.normal(0, self.compass_noise, self.num_particles)
        
        # Set positions with noise
        self.particles[:, 0] = x + pos_noise[:, 0]
        self.particles[:, 1] = y + pos_noise[:, 1]
        
        # Set orientation with noise
        self.particles[:, 2] = initial_theta + theta_noise
        
        # Normalize angles (vectorized)
        self.particles[:, 2] = self.normalize_angle_vector(self.particles[:, 2])
        
        # Initialize weights uniformly
        self.weights = np.ones(self.num_particles) / self.num_particles
        
        # Set initial state
        self.current_state["x"] = x
        self.current_state["y"] = y
        self.current_state["theta"] = initial_theta
        self.current_state["timestamp"] = timestamp
        
        # Set up the projection for GPS conversion
        if self.ref_lat is None and self.ref_lon is None:
            self.ref_lat, self.ref_lon = self.reverse_ned_to_latlon(x, y)
            # Initialize projection with reference point
            self.setup_projection()
        
        self.logger.info(f"Particle filter initialized at position ({x:.2f}, {y:.2f}) with heading {math.degrees(initial_theta):.2f}°")
        return True
    
    def start(self):
        """Start the data collection and particle filter threads"""
        self.running = True
        
        # Start data collection thread first
        self.data_thread = threading.Thread(target=self.data_collection_thread)
        # self.data_thread.daemon = True
        self.data_thread.start()
        
        # Wait for initial sensor readings and particle initialization
        while self.particles is None:
            if self.init_particles():
                break
            time.sleep(0.1)
        
        # Start particle filter thread
        self.pf_thread = threading.Thread(target=self.particle_filter_thread)
        # self.pf_thread.daemon = True
        self.pf_thread.start()
        
        self.logger.info("State estimation started")
    
    def stop(self):
        """Stop all threads"""
        self.running = False
        if self.data_thread:
            self.data_thread.join(timeout=1.0)
        if self.pf_thread:
            self.pf_thread.join(timeout=1.0)
        self.logger.info("State estimation stopped")
    
    def data_collection_thread(self):
        """Non-blocking thread to fetch and store sensor data"""
        while self.running:
            try:
                # Get data from robot API
                new_data = self.get_data()
                self.logger.debug(f"Data collection thread: {new_data}")
                if not new_data:
                    time.sleep(0.01)
                    continue
                
                # Process and store the data
                with self.data_lock:
                    # Process accelerometer readings
                    if "accels" in new_data and new_data["accels"]:
                        # Apply outlier detection before storing
                        filtered_accels = self.handle_outliers(new_data["accels"], threshold=3.0)
                        for reading in filtered_accels:
                            timestamp = reading[3]
                            if timestamp > self.last_processed["accels"]:
                                self.sensor_data["accels"].append(reading)
                                self.last_processed["accels"] = max(self.last_processed["accels"], timestamp)
                    
                    # Process gyroscope readings
                    if "gyros" in new_data and new_data["gyros"]:
                        filtered_gyros = self.handle_outliers(new_data["gyros"], threshold=3.0)
                        for reading in filtered_gyros:
                            timestamp = reading[3]
                            if timestamp > self.last_processed["gyros"]:
                                self.sensor_data["gyros"].append(reading)
                                self.last_processed["gyros"] = max(self.last_processed["gyros"], timestamp)
                    
                    # Process magnetometer readings
                    if "mags" in new_data and new_data["mags"]:
                        filtered_mags = self.handle_outliers(new_data["mags"], threshold=3.0)
                        for reading in filtered_mags:
                            timestamp = reading[3]
                            if timestamp > self.last_processed["mags"]:
                                self.sensor_data["mags"].append(reading)
                                self.last_processed["mags"] = max(self.last_processed["mags"], timestamp)
                    
                    # Process wheel encoder (RPM) readings
                    if "rpms" in new_data and new_data["rpms"]:
                        filtered_rpms = self.handle_outliers(new_data["rpms"], threshold=3.0)
                        for reading in filtered_rpms:
                            timestamp = reading[4]
                            if timestamp > self.last_processed["rpms"]:
                                self.sensor_data["rpms"].append(reading)
                                self.last_processed["rpms"] = max(self.last_processed["rpms"], timestamp)
                    
                    # Process GPS readings with outlier detection
                    if "latitude" in new_data and "longitude" in new_data:
                        lat = float(new_data["latitude"])
                        lon = float(new_data["longitude"])
                        timestamp = float(new_data["timestamp"])
                        
                        # Skip invalid GPS readings (1000, 1000 indicates no signal)
                        if lat == 1000 and lon == 1000:
                            continue
                        
                        # Convert GPS to meters from reference point
                        x, y = self.gps_to_meters(lat, lon)
                        
                        # Add to temporary list for outlier detection
                        gps_reading = [x, y, timestamp]
                        if len(self.sensor_data["gps"]) > 0:
                            # Create list with recent GPS readings plus new reading
                            recent_gps = list(self.sensor_data["gps"])[-4:] + [gps_reading]
                            filtered_gps = self.handle_outliers(recent_gps, threshold=3.0)
                            
                            # If the new reading survived outlier detection, add it
                            if any(reading[2] == timestamp for reading in filtered_gps):
                                if timestamp > self.last_processed["gps"]:
                                    self.sensor_data["gps"].append(gps_reading)
                                    self.last_processed["gps"] = timestamp
                        else:
                            # First GPS reading, add without filtering
                            self.sensor_data["gps"].append(gps_reading)
                            self.last_processed["gps"] = timestamp
                
                # Small sleep to avoid overwhelming the API
                time.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Error in data collection: {e}")
                time.sleep(0.1)  # Sleep on error to prevent tight loop
    
    def get_data(self):
        """Fetch data from the robot API"""
        try:
            # Call the API to get sensor data
            response = requests.get("http://localhost:8000/data")
            data = response.json()
            
            # Validate data format
            if not isinstance(data, dict):
                self.logger.warning("Invalid data format from API")
                return None
            
            return data
        except Exception as e:
            self.logger.error(f"Error fetching data: {e}")
            return None
    
    def particle_filter_thread(self):
        """Thread to run the particle filter algorithm periodically"""
        update_threshold = 0.05  # 50ms threshold for updates
        
        while self.running:
            try:
                self.logger.debug("Trying to get data lock")
                with self.data_lock:
                    self.logger.debug("Got data lock")
                    # Get latest timestamps from sensor data
                    latest_timestamps = {
                        sensor: max([reading[-1] for reading in data]) if data else 0
                        for sensor, data in self.sensor_data.items()
                    }
                    
                    # Use the most recent sensor timestamp
                    current_time = max(latest_timestamps.values()) if latest_timestamps else 0
                    
                    # Skip if not enough time has passed
                    if (self.current_state["timestamp"] is not None and 
                        (current_time - self.current_state["timestamp"]) < update_threshold):
                        time.sleep(0.01)  # Small sleep to prevent tight loop
                        continue
                    
                    # Make a copy of the current state to use for prediction
                    prev_state = copy.deepcopy(self.current_state)
                
                # Run prediction step using sensor timestamps
                self.predict(current_time, prev_state)
                
                # Run update step
                self.update()
                
                # Resample particles if needed
                self.resample()
                
                # Update current state estimate from particles
                self.update_state_estimate()
                
            except Exception as e:
                self.logger.error(f"Error in particle filter thread: {e}")
                time.sleep(0.1)  # Sleep on error to prevent tight loop
    
    def predict(self, current_time, prev_state):
        """
        Vectorized prediction step of the particle filter using motion model in NED frame
        Args:
            current_time: Current timestamp
            prev_state: Previous state estimate (x, y, theta, timestamp)
        """
        if prev_state["timestamp"] is None or prev_state["timestamp"] == 0:
            self.current_state["timestamp"] = current_time
            return
        
        prev_time = prev_state["timestamp"]
        time_diff = current_time - prev_time
        
        if time_diff <= 0:
            return  # No time has passed, skip prediction
        
        # Calculate time-dependent noise scaling (more noise for longer intervals)
        noise_scale = min(5.0, max(1.0, time_diff))
        
        with self.data_lock:
            # Get relevant sensor data between prev_time and current_time
            accel_data = [a for a in self.sensor_data["accels"] if prev_time < a[3] <= current_time]
            gyro_data = [g for g in self.sensor_data["gyros"] if prev_time < g[3] <= current_time]
            rpm_data = [r for r in self.sensor_data["rpms"] if prev_time < r[4] <= current_time]
        
        # Extract current particle states
        theta = self.particles[:, 2]
        
        # Initialize deltas with zeros
        delta_x = np.zeros(self.num_particles)
        delta_y = np.zeros(self.num_particles)
        delta_theta = np.zeros(self.num_particles)
        
        # Process wheel encoder data if available
        if rpm_data:
            # Sort RPM data by timestamp
            rpm_data.sort(key=lambda x: x[4])
            
            for j in range(len(rpm_data)):
                # Get wheel RPMs
                left_rpm, right_rpm = rpm_data[j][0], rpm_data[j][1]
                reading_time = rpm_data[j][4]
                
                # Calculate time difference
                if j > 0:
                    dt = reading_time - rpm_data[j-1][4]
                else:
                    dt = reading_time - prev_time
                
                # Skip if dt is invalid
                if dt <= 0:
                    continue
                
                # Convert RPM to linear velocity (meters per second)
                wheel_circumference = 2 * math.pi * self.wheel_radius / 1000  # Convert to meters
                vl = left_rpm * wheel_circumference / 60
                vr = right_rpm * wheel_circumference / 60
                
                # Differential drive kinematics
                v = (vl + vr) / 2  # Linear velocity
                omega = (vr - vl) / (self.wheel_base / 1000)  # Angular velocity
                
                # Vectorized update for all particles
                # Handle straight line vs. curved motion
                straight_motion = np.abs(omega) < 1e-6
                
                # Update straight motion particles (vectorized)
                if np.any(straight_motion):
                    delta_x[straight_motion] += v * dt * np.cos(theta[straight_motion])
                    delta_y[straight_motion] += v * dt * np.sin(theta[straight_motion])
                
                # Update curved motion particles (vectorized)
                if np.any(~straight_motion):
                    # Radius of curvature for each particle
                    r = v / omega
                    
                    # Update for curved motion
                    delta_x[~straight_motion] += r * (np.sin(theta[~straight_motion] + omega * dt) - 
                                                     np.sin(theta[~straight_motion]))
                    delta_y[~straight_motion] += r * (np.cos(theta[~straight_motion]) - 
                                                     np.cos(theta[~straight_motion] + omega * dt))
                    delta_theta[~straight_motion] += omega * dt
        
        # Use gyro data for orientation update when wheel encoders are insufficient
        if not rpm_data or len(rpm_data) < 2:
            if gyro_data:
                # Process all gyro readings at once for more efficient computation
                gyro_times = np.array([g[3] for g in gyro_data])
                gyro_yaw_rates = np.array([math.radians(g[0]) for g in gyro_data])  # Convert to radians/s
                
                if len(gyro_data) > 1:
                    # Calculate time differences between consecutive readings
                    dt_array = np.diff(gyro_times, prepend=prev_time)
                    # Apply orientation updates
                    delta_theta_gyro = np.sum(gyro_yaw_rates * dt_array)
                    delta_theta += delta_theta_gyro
                else:
                    # Only one reading, use time_diff
                    delta_theta += gyro_yaw_rates[0] * time_diff
            
            # Instead of double integration with accelerometer data,
            # use a simplified motion model based on acceleration magnitude
            if accel_data and len(accel_data) >= 3:
                # Convert accelerometer data to numpy array for vectorized operations
                accel_array = np.array(accel_data)
                
                # Compute mean acceleration over the interval
                mean_accel = np.mean(accel_array[:, :3], axis=0)
                
                # Convert from g to m/s²
                g = 9.81
                mean_accel *= g
                
                # In the robot frame: x=up, y=left, z=backward
                # Convert to NED: forward=-z, right=-y
                ax_ned = -mean_accel[2]  # Forward acceleration (North)
                ay_ned = -mean_accel[1]  # Right acceleration (East)
                
                # Compute approximate velocity change instead of position
                # This avoids double integration problems
                accel_magnitude = np.sqrt(ax_ned**2 + ay_ned**2)
                
                # Only apply if acceleration is significant (filter noise)
                if accel_magnitude > 0.2:  # Threshold to filter out noise
                    # Direction of acceleration in global frame
                    accel_heading = math.atan2(ay_ned, ax_ned)
                    
                    # Apply a simplified model (v = a*t) with scale factor to reduce drift
                    v_estimate = accel_magnitude * time_diff * 0.5  # Scale factor
                    
                    # Update all particles with this velocity estimate
                    # But in each particle's individual direction
                    delta_x += v_estimate * np.cos(theta)
                    delta_y += v_estimate * np.sin(theta)
        
        # Generate noise vectors for all particles at once
        noise = np.random.normal(
            scale=[
                self.motion_noise["x"] * noise_scale,
                self.motion_noise["y"] * noise_scale,
                self.motion_noise["theta"] * noise_scale
            ],
            size=(self.num_particles, 3)
        )
        
        # Apply motion updates with noise in a vectorized manner
        self.particles[:, 0] += delta_x + noise[:, 0]
        self.particles[:, 1] += delta_y + noise[:, 1]
        self.particles[:, 2] = self.normalize_angle_vector(self.particles[:, 2] + delta_theta + noise[:, 2])
        
        # Update timestamp
        self.current_state["timestamp"] = current_time
    
    def update(self):
        """Vectorized measurement update step using GPS and compass data"""
        with self.data_lock:
            # Get the latest GPS and magnetometer readings
            latest_gps = self.get_latest_sensor_data("gps")
            latest_mag = self.get_latest_sensor_data("mags")
        
        # Skip update if no measurements are available
        if not latest_gps and not latest_mag:
            return
        
        # Initialize log-weights
        log_weights = np.zeros(self.num_particles)
        
        # Update weights based on GPS measurement
        if latest_gps:
            gps_x, gps_y, _ = latest_gps
            
            # Calculate distances for all particles at once
            particle_x, particle_y = self.particles[:, 0], self.particles[:, 1]
            distances = np.sqrt((particle_x - gps_x)**2 + (particle_y - gps_y)**2)
            
            # Update log-weights based on GPS measurement likelihood
            log_weights += self.gaussian_log_likelihood_vector(distances, 0, self.gps_noise)
        
        # Update weights based on compass measurement
        if latest_mag:
            # Extract magnetometer readings
            mag_x, mag_y, mag_z, _ = latest_mag
            
            # Calculate heading from magnetometer
            heading = math.atan2(-mag_y, -mag_z)
            
            # Calculate angle differences for all particles
            particle_theta = self.particles[:, 2]
            angle_diff = self.normalize_angle_vector(particle_theta - heading)
            
            # Update log-weights based on compass measurement likelihood
            log_weights += self.gaussian_log_likelihood_vector(angle_diff, 0, self.compass_noise)
        
        # Convert log-weights to weights with numerical stability
        max_log_weight = np.max(log_weights)
        weights = np.exp(log_weights - max_log_weight)
        
        # Normalize weights
        sum_weights = np.sum(weights)
        if sum_weights > 0:
            self.weights = weights / sum_weights
        else:
            # If all weights are zero, reinitialize with uniform weights
            self.weights = np.ones(self.num_particles) / self.num_particles
    
    def resample(self):
        """Vectorized resampling of particles based on weights"""
        # Calculate effective number of particles
        n_eff = 1.0 / np.sum(np.square(self.weights))
        
        # Resample if effective number of particles is too low
        if n_eff < self.num_particles / 2:
            # Systematic resampling
            indices = self.systematic_resample()
            
            # Create new particles by indexing
            self.particles = self.particles[indices]
            
            # Reset weights
            self.weights = np.ones(self.num_particles) / self.num_particles
    
    def systematic_resample(self):
        """Vectorized systematic resampling algorithm"""
        # Create evenly spaced points with a random offset
        positions = (np.arange(self.num_particles) + np.random.random()) / self.num_particles
        
        # Calculate cumulative sum of weights
        cumulative_sum = np.cumsum(self.weights)
        
        # Find indices using searchsorted for better performance
        indices = np.searchsorted(cumulative_sum, positions)
        
        # Handle edge case where searchsorted might return an index = len(cumulative_sum)
        indices = np.clip(indices, 0, len(cumulative_sum) - 1)
        
        return indices
    
    def update_state_estimate(self):
        """Update the current state estimate based on weighted particles"""
        # Weighted average for position (vectorized)
        x_mean = np.sum(self.particles[:, 0] * self.weights)
        y_mean = np.sum(self.particles[:, 1] * self.weights)
        
        # For orientation, we need to handle the circular nature of angles
        # Convert to unit vectors, take weighted average, then convert back to angle
        cos_sum = np.sum(np.cos(self.particles[:, 2]) * self.weights)
        sin_sum = np.sum(np.sin(self.particles[:, 2]) * self.weights)
        theta_mean = math.atan2(sin_sum, cos_sum)
        
        # Update current state
        self.current_state["x"] = x_mean
        self.current_state["y"] = y_mean
        self.current_state["theta"] = theta_mean
        
        # Record timestamp for frequency calculation
        self.update_timestamps.append(time.time())
    
    def get_latest_sensor_data(self, sensor_type):
        """Get the latest reading for a specific sensor type"""
        if not self.sensor_data[sensor_type]:
            return None
        return self.sensor_data[sensor_type][-1]
    
    def get_state(self):
        """Return the current state estimate"""
        return copy.deepcopy(self.current_state)
    
    def normalize_angle(self, angle):
        """Normalize angle to be between -pi and pi"""
        return (angle + math.pi) % (2 * math.pi) - math.pi
    
    def normalize_angle_vector(self, angles):
        """Vectorized version of normalize_angle for numpy arrays"""
        return (angles + np.pi) % (2 * np.pi) - np.pi
    
    def gaussian_log_likelihood(self, x, mean, std_dev):
        """Calculate log-likelihood of x under a Gaussian distribution"""
        return -0.5 * ((x - mean) / std_dev)**2 - math.log(std_dev) - 0.5 * math.log(2 * math.pi)
    
    def gaussian_log_likelihood_vector(self, x, mean, std_dev):
        """Vectorized version of gaussian_log_likelihood"""
        return -0.5 * ((x - mean) / std_dev)**2 - np.log(std_dev) - 0.5 * np.log(2 * np.pi)
    
    def handle_outliers(self, data_list, threshold=3.0):
        """
        Vectorized outlier detection using z-score method
        Returns data with outliers removed
        """
        if not data_list or len(data_list) < 4:
            return data_list
        
        # Convert to numpy array for easier calculations
        data_array = np.array(data_list)
        
        # Calculate z-scores in a vectorized way
        mean = np.mean(data_array, axis=0)
        std = np.std(data_array, axis=0)
        
        # Avoid division by zero
        std = np.where(std < 1e-10, 1e-10, std)
        
        # Calculate z-scores for each dimension
        z_scores = np.abs((data_array - mean) / std)
        
        # Find indices of non-outliers (all dimensions must pass the test)
        non_outliers = np.all(z_scores < threshold, axis=1)
        
        # Return non-outlier data
        return data_array[non_outliers].tolist()
    
    def gps_to_meters(self, lat, lon):
        """Convert GPS coordinates to meters in NED frame using pyproj"""
        if self.ref_lat is None or self.ref_lon is None:
            self.ref_lat = lat
            self.ref_lon = lon
            self.setup_projection()
            return 0, 0
        
        if self.proj is None:
            self.setup_projection()
        
        # Convert from lat/lon to projected coordinates
        east, north = self.proj(lon, lat)
        
        # NED frame convention has north as x and east as y
        return north, east
    
    def reverse_ned_to_latlon(self, north, east):
        """Convert NED coordinates back to lat/lon using pyproj"""
        if self.proj is None:
            # If projection isn't set up yet, return a default location
            return 0, 0
        
        # Convert from NED to lat/lon using inverse projection
        lon, lat = self.proj(east, north, inverse=True)
        
        return lat, lon
    
    def get_uncertainty(self):
        """
        Compute the uncertainty of the particle filter state estimate.
        Returns:
            position_std: Standard deviation of particle positions (meters)
            heading_std: Standard deviation of particle headings (radians)
        """
        if self.particles is None:
            return float('inf'), float('inf')
        
        # Calculate position uncertainty as the average std dev in x and y
        pos_x_std = np.std(self.particles[:, 0])
        pos_y_std = np.std(self.particles[:, 1])
        position_std = (pos_x_std + pos_y_std) / 2.0
        
        # For heading uncertainty, we need to handle circular statistics
        # Convert angles to unit vectors, compute resultant length
        cos_theta = np.cos(self.particles[:, 2])
        sin_theta = np.sin(self.particles[:, 2])
        
        # Calculate circular standard deviation
        resultant_length = np.sqrt(np.mean(cos_theta)**2 + np.mean(sin_theta)**2)
        heading_std = np.sqrt(-2 * np.log(resultant_length))
        
        return position_std, heading_std
        
    def get_data_fetching_frequency(self, window_size=10):
        """
        Compute the recent update frequency of the particle filter in Hz.
        
        Args:
            window_size: Number of recent updates to consider for frequency calculation
            
        Returns:
            frequency: Update frequency in Hz, or 0 if insufficient data
        """
        # Get timestamps from recent sensor data
        timestamps = []
        
        # Collect timestamps from all sensor types
        for sensor_type in self.sensor_data:
            data = self.sensor_data[sensor_type]
            if data:
                # Extract timestamps from the last elements of each reading
                sensor_timestamps = [reading[-1] for reading in data]
                timestamps.extend(sensor_timestamps)
        
        # Sort timestamps and take the most recent ones
        timestamps.sort(reverse=True)
        recent_timestamps = timestamps[:window_size*5]  # Get more than needed to filter
        
        if len(recent_timestamps) < 2:
            return 0.0  # Not enough data to calculate frequency
        
        # Remove duplicates and sort again
        unique_timestamps = sorted(set(recent_timestamps), reverse=True)
        
        # Take only the window_size most recent unique timestamps
        recent_unique = unique_timestamps[:window_size]
        
        if len(recent_unique) < 2:
            return 0.0  # Not enough unique timestamps
        
        # Calculate time differences between consecutive timestamps
        time_diffs = [recent_unique[i] - recent_unique[i+1] for i in range(len(recent_unique)-1)]
        
        # Calculate average time difference
        avg_time_diff = sum(time_diffs) / len(time_diffs)
        
        # Calculate frequency (Hz)
        if avg_time_diff > 0:
            return 1.0 / avg_time_diff
        else:
            return 0.0  # Avoid division by zero
    
    def get_particle_filter_frequency(self, window_size=10):
        """
        Compute the recent particle filter update frequency in Hz.
        This measures how often the particle filter state is being updated.
        
        Args:
            window_size: Number of recent updates to consider for frequency calculation
            
        Returns:
            frequency: Particle filter update frequency in Hz, or 0 if insufficient data
        """
        if len(self.update_timestamps) < 2:
            return 0.0  # Not enough data to calculate frequency
        
        # Get the most recent timestamps
        recent_timestamps = list(self.update_timestamps)[-window_size:]
        
        if len(recent_timestamps) < 2:
            return 0.0  # Not enough timestamps in the window
        
        # Calculate time differences between consecutive timestamps
        time_diffs = [recent_timestamps[i] - recent_timestamps[i-1] for i in range(1, len(recent_timestamps))]
        
        # Calculate average time difference
        avg_time_diff = sum(time_diffs) / len(time_diffs)
        
        # Calculate frequency (Hz)
        if avg_time_diff > 0:
            return 1.0 / avg_time_diff
        else:
            return 0.0  # Avoid division by zero

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start the particle filter
    pf = StateEstimationPF(log_level=logging.INFO)
    pf.start()
    
    try:
        # Main loop
        while True:
            # Get current state estimate
            state = pf.get_state()
            print(f"Position: ({state['x']:.2f}, {state['y']:.2f}), Heading: {math.degrees(state['theta']):.2f}°")
            time.sleep(1.0)
    except KeyboardInterrupt:
        # Stop the particle filter on Ctrl+C
        pf.stop()
        print("Program terminated")