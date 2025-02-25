import threading
import time
import numpy as np
import math
from collections import deque
import copy
from scipy.stats import norm
import json
import requests
from typing import Dict, List, Tuple, Any, Optional

class StateEstimationPF:
    """
    Particle Filter for State Estimation
    The estimated state is in the NED frame (North, East, Down)
    """
    def __init__(self, num_particles=100, wheel_diameter=130, wheel_base=200):

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
        
        # Initialize with None to indicate we need first readings
        self.current_state = {
            "x": None,
            "y": None,
            "theta": None,
            "timestamp": None
        }
        
        # Don't initialize particles yet - wait for first readings
        self.particles = None
        
        # Add reference GPS coordinates (first valid reading will set these)
        self.ref_lat = None
        self.ref_lon = None
        
        # Earth's radius in meters
        self.EARTH_RADIUS = 6371000
    
    def init_particles(self):
        """Initialize particles using first GPS and compass readings"""
        # Try to get initial GPS and magnetometer readings
        with self.data_lock:
            gps_reading = self.get_latest_sensor_data("gps")
            mag_reading = self.get_latest_sensor_data("mags")
        
        if not gps_reading or not mag_reading:
            print("Waiting for initial sensor readings...")
            return False
        
        # Initialize position from GPS
        x, y, timestamp = gps_reading
        
        # Calculate initial heading from magnetometer
        mx, my, mz, _ = mag_reading
        initial_theta = math.atan2(-my, -mz)  # Convert magnetometer reading to heading
        
        # Initialize particles around these measurements
        self.particles = np.zeros((self.num_particles, 3))
        
        # Add noise based on sensor uncertainties
        self.particles[:, 0] = x + np.random.normal(0, self.gps_noise, self.num_particles)
        self.particles[:, 1] = y + np.random.normal(0, self.gps_noise, self.num_particles)
        self.particles[:, 2] = initial_theta + np.random.normal(0, self.compass_noise, self.num_particles)
        
        # Normalize angles
        self.particles[:, 2] = self.normalize_angle(self.particles[:, 2])
        
        # Initialize weights uniformly
        self.weights = np.ones(self.num_particles) / self.num_particles
        
        # Set initial state
        self.current_state["x"] = x
        self.current_state["y"] = y
        self.current_state["theta"] = initial_theta
        self.current_state["timestamp"] = timestamp
        
        print(f"Particle filter initialized at position ({x:.2f}, {y:.2f}) with heading {math.degrees(initial_theta):.2f}°")
        return True
    
    def start(self):
        """Start the data collection and particle filter threads"""
        self.running = True
        
        # Start data collection thread first
        self.data_thread = threading.Thread(target=self.data_collection_thread)
        self.data_thread.daemon = True
        self.data_thread.start()
        
        # Wait for initial sensor readings and particle initialization
        while self.particles is None:
            if self.init_particles():
                break
            time.sleep(0.1)
        
        # Start particle filter thread
        self.pf_thread = threading.Thread(target=self.particle_filter_thread)
        self.pf_thread.daemon = True
        self.pf_thread.start()
        
        print("State estimation started")
    
    def stop(self):
        """Stop all threads"""
        self.running = False
        if self.data_thread:
            self.data_thread.join(timeout=1.0)
        if self.pf_thread:
            self.pf_thread.join(timeout=1.0)
        print("State estimation stopped")
    
    def data_collection_thread(self):
        """Non-blocking thread to fetch and store sensor data"""
        while self.running:
            try:
                # Get data from robot API
                new_data = self.get_data()
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
                print(f"Error in data collection: {e}")
                time.sleep(0.1)  # Sleep on error to prevent tight loop
    
    def get_data(self):
        """Fetch data from the robot API
        Sensor frames:
        - Accel: 
            x	g	Positive X = Upward
            y	g	Positive Y = Left
            z	g	Positive Z = Backwards
        - Gyro: 
            x	Degrees / second	Yaw
            y	Degrees / second	Pitch
            z	Degrees / second	Roll
        - Mag: 
            x	LSB Raw Data	Positive X = Vertical Up
            y	LSB Raw Data	Positive Y = West
            z	LSB Raw Data	Positive Z = South
        - GPS:
            latitude	Float	Latitude of the robot
            longitude	Float	Longitude of the robot
        - Control:
            linear	% of max control movement (-1 to 1)	Linear control from the gamer input (1 = Forward ; -1 = Backwards)
            angular	% of max control movement (-1 to 1)	Angular control from the gamer input (1 = Left ; -1 = Right )
            rpm_1	Revolutions Per Minute	RPM reading of the Front-Left wheel
            rpm_2	Revolutions Per Minute	RPM reading of the Front-Right wheel
            rpm_3	Revolutions Per Minute	RPM reading of the Rear-Left wheel
            rpm_4	Revolutions Per Minute	RPM reading of the Rear-Right wheel
            timestamp	Unix Timestamp Epoch (UTC+0)	Timestamp of the data
        """
        try:
            # Call the API to get sensor data
            # In a real implementation, this would be replaced with the actual API call
            response = requests.get("http://localhost:8000/data")
            data = response.json()
            
            # Validate data format
            if not isinstance(data, dict):
                print("Invalid data format from API")
                return None
            
            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def particle_filter_thread(self):
        """Thread to run the particle filter algorithm periodically"""
        update_threshold = 0.05  # 50ms threshold for updates
        
        while self.running:
            try:
                with self.data_lock:
                    # Get latest timestamps from sensor data
                    latest_timestamps = {
                        sensor: max([reading[-1] for reading in data]) if data else 0
                        for sensor, data in self.sensor_data.items()
                    }
                    
                    # Use the most recent sensor timestamp
                    current_time = max(latest_timestamps.values()) if latest_timestamps else 0
                    
                    # Skip if not enough time has passed
                    if (current_time - self.current_state["timestamp"]) < update_threshold:
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
                print(f"Error in particle filter thread: {e}")
                time.sleep(0.1)  # Sleep on error to prevent tight loop
    
    def predict(self, current_time, prev_state):
        """
        Prediction step of the particle filter using motion model in NED frame
        Args:
            current_time: Current timestamp
            prev_state: Previous state estimate (x, y, theta, timestamp)
            x: North (meters)
            y: East (meters)
            theta: heading in radians (0 = North, pi/2 = East)
        """
        if prev_state["timestamp"] == 0:
            self.current_state["timestamp"] = current_time
            return
        
        prev_time = prev_state["timestamp"]
        time_diff = current_time - prev_time
        
        if time_diff <= 0:
            return  # No time has passed, skip prediction
        
        # Calculate time-dependent noise scaling (more noise for longer intervals)
        noise_scale = min(5.0, max(1.0, time_diff))
        
        with self.data_lock:
            # Get relevant IMU and wheel encoder data between prev_time and current_time
            accel_data = [a for a in self.sensor_data["accels"] if prev_time < a[3] <= current_time]
            gyro_data = [g for g in self.sensor_data["gyros"] if prev_time < g[3] <= current_time]
            rpm_data = [r for r in self.sensor_data["rpms"] if prev_time < r[4] <= current_time]
        
        # Apply motion model to each particle
        for i in range(self.num_particles):
            x, y, theta = self.particles[i]
            
            delta_x = 0  # Change in North
            delta_y = 0  # Change in East
            delta_theta = 0
            
            # Use wheel encoder data if available
            if rpm_data:
                # Sort RPM data by timestamp to ensure chronological order
                rpm_data.sort(key=lambda x: x[4])
                
                for j in range(len(rpm_data)):
                    # Front-Left and Front-Right wheel RPMs
                    left_rpm, right_rpm = rpm_data[j][0], rpm_data[j][1]
                    reading_time = rpm_data[j][4]
                    
                    # Calculate time difference between consecutive RPM readings
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
                    
                    # Update position and orientation in NED frame
                    if abs(omega) < 1e-6:  # Straight line motion
                        delta_x += v * dt * math.cos(theta)  # North component
                        delta_y += v * dt * math.sin(theta)  # East component
                    else:  # Curved motion
                        r = v / omega  # Radius of curvature
                        delta_x += r * (math.sin(theta + omega * dt) - math.sin(theta))
                        delta_y += r * (math.cos(theta) - math.cos(theta + omega * dt))
                        delta_theta += omega * dt
            
            # Use IMU data if wheel encoder data is insufficient
            if not rpm_data or len(rpm_data) < 2:
                # Convert gyroscope data (degrees/s to rad/s) and handle frame conversion
                if gyro_data:
                    for gyro in gyro_data:
                        # Convert from degrees/s to rad/s and handle frame conversion
                        # Note: gyro.x is yaw rate, but we need to convert to NED frame
                        yaw_rate = math.radians(gyro[0])  # Convert to radians/s
                        dt = time_diff if len(gyro_data) <= 1 else gyro[3] - gyro_data[0][3]
                        delta_theta += yaw_rate * dt
                
                # Handle accelerometer data (convert from g to m/s²)
                if len(accel_data) >= 3:
                    g = 9.81  # m/s²
                    for i in range(1, len(accel_data)):
                        # Convert from g to m/s² and transform to NED frame
                        # accel.x is up (+), accel.y is left (+), accel.z is backward (+)
                        ax = -accel_data[i][2] * g  # Forward = -z_accel (North)
                        ay = -accel_data[i][1] * g  # Right = -y_accel (East)
                        
                        dt = accel_data[i][3] - accel_data[i-1][3]
                        
                        # Rotate acceleration to NED frame using current heading
                        ax_ned = ax * math.cos(theta) - ay * math.sin(theta)
                        ay_ned = ax * math.sin(theta) + ay * math.cos(theta)
                        
                        # Double integration for position
                        dvx = ax_ned * dt
                        dvy = ay_ned * dt
                        
                        delta_x += dvx * dt
                        delta_y += dvy * dt

            # Apply motion updates with noise
            self.particles[i, 0] += delta_x + np.random.normal(0, self.motion_noise["x"] * noise_scale)
            self.particles[i, 1] += delta_y + np.random.normal(0, self.motion_noise["y"] * noise_scale)
            self.particles[i, 2] = self.normalize_angle(self.particles[i, 2] + delta_theta + 
                                  np.random.normal(0, self.motion_noise["theta"] * noise_scale))
        
        # Update timestamp
        self.current_state["timestamp"] = current_time
    
    def update(self):
        """Measurement update step of the particle filter using GPS and compass data"""
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
            lat, lon, _ = latest_gps
            
            # In a real system, you would convert lat/lon to your coordinate system
            # Here I'll assume lat/lon are already in meters from origin for simplicity
            gps_x, gps_y = lat, lon
            
            # Calculate likelihood for each particle
            for i in range(self.num_particles):
                particle_x, particle_y = self.particles[i, 0], self.particles[i, 1]
                
                # Mahalanobis distance to GPS measurement
                distance = np.sqrt((particle_x - gps_x)**2 + (particle_y - gps_y)**2)
                
                # Update log-weight based on GPS measurement likelihood
                log_weights[i] += self.gaussian_log_likelihood(distance, 0, self.gps_noise)
        
        # Update weights based on compass measurement
        if latest_mag:
            # Extract magnetometer readings
            mag_x, mag_y, mag_z, _ = latest_mag
            
            # Calculate heading from magnetometer
            # The compass frame is (Vertical Up, West, South)
            # So we need to convert to get a proper heading
            # Assuming the magnetometer is roughly level
            heading = math.atan2(-mag_y, -mag_z)  # Convert from (West, South) to a heading
            
            # Calculate likelihood for each particle
            for i in range(self.num_particles):
                particle_theta = self.particles[i, 2]
                
                # Calculate angle difference
                angle_diff = self.normalize_angle(particle_theta - heading)
                
                # Update log-weight based on compass measurement likelihood
                log_weights[i] += self.gaussian_log_likelihood(angle_diff, 0, self.compass_noise)
        
        # Convert log-weights to weights
        # Shifting max log-weight to avoid numerical underflow
        max_log_weight = np.max(log_weights)
        weights = np.exp(log_weights - max_log_weight)
        
        # Normalize weights
        if np.sum(weights) > 0:
            self.weights = weights / np.sum(weights)
        else:
            # If all weights are zero, reinitialize with uniform weights
            self.weights = np.ones(self.num_particles) / self.num_particles
    
    def resample(self):
        """Resample particles based on their weights"""
        # Calculate effective number of particles
        n_eff = 1.0 / np.sum(np.square(self.weights))
        
        # Resample if effective number of particles is too low
        # (typically when n_eff < num_particles/2)
        if n_eff < self.num_particles / 2:
            # Systematic resampling
            indices = self.systematic_resample()
            
            # Create new particles array
            new_particles = np.zeros((self.num_particles, 3))
            
            # Copy selected particles
            for i in range(self.num_particles):
                new_particles[i] = self.particles[indices[i]]
            
            # Update particles
            self.particles = new_particles
            
            # Reset weights
            self.weights = np.ones(self.num_particles) / self.num_particles
    
    def systematic_resample(self):
        """Systematic resampling algorithm"""
        positions = (np.arange(self.num_particles) + np.random.random()) / self.num_particles
        
        indices = np.zeros(self.num_particles, dtype=int)
        cumulative_sum = np.cumsum(self.weights)
        i, j = 0, 0
        
        while i < self.num_particles:
            if positions[i] < cumulative_sum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1
        
        return indices
    
    def update_state_estimate(self):
        """Update the current state estimate based on particle weights"""
        # Weighted average for position
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
    
    def gaussian_log_likelihood(self, x, mean, std_dev):
        """Calculate log-likelihood of x under a Gaussian distribution"""
        return -0.5 * ((x - mean) / std_dev)**2 - math.log(std_dev) - 0.5 * math.log(2 * math.pi)
    
    def handle_outliers(self, data_list, threshold=3.0):
        """
        Simple outlier detection using z-score method
        Returns data with outliers removed
        """
        if not data_list or len(data_list) < 4:
            return data_list
        
        # Convert to numpy array for easier calculations
        data_array = np.array(data_list)
        
        # Calculate z-scores
        mean = np.mean(data_array, axis=0)
        std = np.std(data_array, axis=0)
        z_scores = np.abs((data_array - mean) / (std + 1e-10))  # Add small epsilon to avoid division by zero
        
        # Find indices of non-outliers
        non_outliers = np.all(z_scores < threshold, axis=1)
        
        # Return non-outlier data
        return data_array[non_outliers].tolist()

    def gps_to_meters(self, lat, lon):
        """Convert GPS coordinates to meters from reference point using flat Earth approximation.
        Returns (North, East) coordinates in meters to match NED frame convention."""
        if self.ref_lat is None or self.ref_lon is None:
            self.ref_lat = lat
            self.ref_lon = lon
            return 0, 0
        
        # Convert latitude/longitude from degrees to radians
        lat1, lon1 = math.radians(self.ref_lat), math.radians(self.ref_lon)
        lat2, lon2 = math.radians(lat), math.radians(lon)
        
        # Calculate differences
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        # Calculate North-South distance
        north = dlat * self.EARTH_RADIUS
        
        # Calculate East-West distance
        east = dlon * self.EARTH_RADIUS * math.cos(lat1)
        
        return north, east  # Returns (North, East) in meters to match NED frame

# Example usage
if __name__ == "__main__":
    # Create and start the particle filter
    pf = StateEstimationPF()
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