import cv2
import numpy as np
import threading
import time
from clear.pf import StateEstimationPF
import math
import requests
from io import BytesIO
from PIL import Image
from collections import deque
import os

# Force OpenCV to use a specific backend
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "1"

class ParticleFilterVisualizer:
    def __init__(self, particle_filter: StateEstimationPF, update_interval: float = 0.1):
        self.pf = particle_filter
        self.update_interval = update_interval
        
        # Visualization parameters
        self.arrow_size = 2.0  # meters
        self.min_zoom_width = 50  # meters
        self.map_size = (800, 800)  # pixels
        self.zoom_level = 19
        
        # Initialize the canvas
        self.canvas = np.zeros((self.map_size[0], self.map_size[1], 3), dtype=np.uint8)
        
        # Map cache
        self.map_cache = {}
        self.current_center = None
        
        # Control flags
        self.running = False
        self.viz_thread = None
        
        # Mouse interaction variables
        self.dragging = False
        self.drag_start = None
        self.drag_offset = (0, 0)
        
        # Performance tracking
        self.last_update_time = time.time()
        self.update_times = deque(maxlen=30)  # Store last 30 update times
        self.update_frequency = 0

        # Initialize OpenCV window early
        print("Creating OpenCV window...")
        cv2.namedWindow("Particle Filter Visualization", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Particle Filter Visualization", self.map_size[0], self.map_size[1])
        
        # Show a blank image to ensure window is created
        blank_img = np.ones((self.map_size[0], self.map_size[1], 3), dtype=np.uint8) * 240
        cv2.putText(blank_img, "Initializing...", (50, self.map_size[1]//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Particle Filter Visualization", blank_img)
        cv2.waitKey(1)  # Process the window
        print("OpenCV window created")

    def start(self):
        """Start the visualization thread"""
        self.running = True
        
        # Register window and mouse callback
        cv2.namedWindow("Particle Filter Visualization")
        cv2.setMouseCallback("Particle Filter Visualization", self._mouse_callback)
        
        self.viz_thread = threading.Thread(target=self._update_loop)
        self.viz_thread.daemon = True
        self.viz_thread.start()

    def stop(self):
        """Stop the visualization thread"""
        self.running = False
        if self.viz_thread:
            self.viz_thread.join()
        cv2.destroyAllWindows()

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for dragging and zooming"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.drag_start = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            # Update the map with the final drag position
            if self.current_center:
                # Calculate meters offset
                dx = self._pixels_to_meters(self.drag_offset[0])
                dy = self._pixels_to_meters(self.drag_offset[1])
                
                # Update center (in NED frame, y is north, x is east)
                center_lat, center_lon = self.current_center
                new_lat, new_lon = self._offset_latlon(center_lat, center_lon, -dy, -dx)
                
                self._update_map_image(new_lat, new_lon)
                self.current_center = (new_lat, new_lon)
                
                # Reset drag offset
                self.drag_offset = (0, 0)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                # Calculate drag offset
                dx = x - self.drag_start[0]
                dy = y - self.drag_start[1]
                self.drag_offset = (dx, dy)
        
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Check if control key is pressed
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                # Positive for zoom in, negative for zoom out
                if flags > 0:  # Zoom in
                    self.zoom_level = min(21, self.zoom_level + 1)
                else:  # Zoom out
                    self.zoom_level = max(14, self.zoom_level - 1)
                
                # Update map with new zoom level
                if self.current_center:
                    self._update_map_image(self.current_center[0], self.current_center[1])

    def _pixels_to_meters(self, pixels):
        """Convert pixels to meters based on current zoom level"""
        if self.current_center is None:
            return 0
        meters_per_pixel = 156543.03392 * math.cos(math.radians(self.current_center[0])) / (2 ** self.zoom_level)
        return pixels * meters_per_pixel

    def _offset_latlon(self, lat, lon, north_meters, east_meters):
        """Offset a lat/lon point by the given distances in meters"""
        # Earth's radius in meters
        earth_radius = 6371000
        
        # Convert meters to radians
        dlat = north_meters / earth_radius
        dlon = east_meters / (earth_radius * math.cos(math.radians(lat)))
        
        # Apply offset
        new_lat = lat + math.degrees(dlat)
        new_lon = lon + math.degrees(dlon)
        
        return new_lat, new_lon

    def _update_loop(self):
        """Main update loop for visualization"""
        print("Starting update loop...")
        while self.running:
            try:
                print("Loop iteration starting...")
                # Calculate update frequency
                current_time = time.time()
                self.update_times.append(current_time)
                
                if len(self.update_times) >= 2:
                    # Calculate updates per second over the tracked period
                    time_span = self.update_times[-1] - self.update_times[0]
                    if time_span > 0:
                        self.update_frequency = (len(self.update_times) - 1) / time_span
                
                # Update display
                print("Updating display...")
                self._update_display()
                print("Display updated")
                
                # Process key events with a short timeout
                print("Waiting for key...")
                key = cv2.waitKey(10) & 0xFF
                print(f"Key processed: {key}")
                if key == 27:  # ESC key
                    print("ESC pressed, exiting...")
                    self.running = False
                    break
                
                # Add a small sleep to prevent CPU hogging
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Error in visualization update: {e}")
                import traceback
                traceback.print_exc()  # Print full stack trace
                time.sleep(0.1)
        
        print("Update loop ended")

    def _draw_text_with_background(self, img, text, position, font_scale=0.7, thickness=2, 
                                  text_color=(255, 255, 255), bg_color=(0, 0, 0, 180)):
        """Draw text with a semi-transparent background"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Create background rectangle with padding
        padding = 10
        bg_rect = (position[0], position[1] - text_size[1] - padding, 
                  text_size[0] + padding * 2, text_size[1] + padding * 2)
        
        # Create overlay for semi-transparent background
        overlay = img.copy()
        cv2.rectangle(overlay, (bg_rect[0], bg_rect[1]), 
                     (bg_rect[0] + bg_rect[2], bg_rect[1] + bg_rect[3]), 
                     bg_color[:3], -1)
        
        # Apply transparency
        alpha = bg_color[3] / 255.0
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        # Draw text
        text_position = (position[0] + padding, position[1] - padding)
        cv2.putText(img, text, text_position, font, font_scale, text_color, thickness)
        
        return text_size[1] + padding * 2

    def _update_display(self):
        """Update the visualization with current state"""
        print("Starting _update_display...")
        state = self.pf.get_state()
        print(f"Got state: {state}")
        
        # Check if we have the required state variables
        if not all(v is not None for v in [state.get('x'), state.get('y'), state.get('theta')]):
            print("Waiting for valid state from particle filter...")
            # Create a simple display while waiting for valid state
            display_img = np.ones((self.map_size[0], self.map_size[1], 3), dtype=np.uint8) * 240
            cv2.putText(display_img, "Waiting for valid state...", (50, self.map_size[1]//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Particle Filter Visualization", display_img)
            return
        
        # Convert NED coordinates to lat/lon using the particle filter's conversion method
        try:
            print("Converting NED coordinates to lat/lon...")
            lat, lon = self.pf.reverse_ned_to_latlon(state['x'], state['y'])
            print(f"Converted coordinates: lat={lat}, lon={lon}")
            
            # Add lat/lon to the state dictionary for use in visualization
            state['lat'] = lat
            state['lon'] = lon
            
            print("Getting map tile...")
            # Get map tile for current position
            map_img = self._get_map_tile(lat, lon, self.zoom_level)
            if map_img is None:
                print("Failed to get map tile, using blank image")
                map_img = np.ones((self.map_size[0], self.map_size[1], 3), dtype=np.uint8) * 240
        except Exception as e:
            print(f"Error converting coordinates or getting map tile: {e}")
            import traceback
            traceback.print_exc()
            # Use a blank map with grid
            print("Using blank map due to conversion error")
            map_img = np.ones((self.map_size[0], self.map_size[1], 3), dtype=np.uint8) * 240
            
            # Add a grid to the blank map
            grid_spacing = 50
            for i in range(0, self.map_size[0], grid_spacing):
                cv2.line(map_img, (i, 0), (i, self.map_size[1]), (200, 200, 200), 1)
            for i in range(0, self.map_size[1], grid_spacing):
                cv2.line(map_img, (0, i), (self.map_size[0], i), (200, 200, 200), 1)
        
        print("Creating display image...")
        # Create a copy of the map image to draw on
        display_img = map_img.copy()
        
        # Draw particles only when uncertainty is high
        # We can access particles directly from the pf object
        try:
            print("Checking if we should draw particles...")
            # Access particles directly from the particle filter
            if hasattr(self.pf, 'particles') and self.pf.particles is not None:
                # Calculate effective number of particles to estimate uncertainty
                if hasattr(self.pf, 'weights'):
                    n_eff = 1.0 / np.sum(np.square(self.pf.weights))
                    print(f"Effective number of particles: {n_eff}")
                    
                    # Only draw particles if uncertainty is high (n_eff is low)
                    if n_eff < self.pf.num_particles / 2:
                        print("Drawing particles due to high uncertainty...")
                        for i in range(min(100, len(self.pf.particles))):  # Limit to 100 particles for performance
                            try:
                                # Get particle position in NED coordinates
                                p_north, p_east = self.pf.particles[i, 0], self.pf.particles[i, 1]
                                
                                # Convert to lat/lon
                                p_lat, p_lon = self.pf.reverse_ned_to_latlon(p_north, p_east)
                                
                                # Convert to pixel coordinates
                                x_pixel, y_pixel = self._latlon_to_pixel(p_lat, p_lon)
                                
                                # Draw particle
                                cv2.circle(display_img, (x_pixel, y_pixel), 1, (0, 0, 255), -1)
                            except Exception as e:
                                # Skip this particle if there's an error
                                continue
        except Exception as e:
            print(f"Error accessing or drawing particles: {e}")
            import traceback
            traceback.print_exc()
        
        print("Drawing robot position...")
        # Draw robot position
        try:
            x_pixel, y_pixel = self._latlon_to_pixel(state['lat'], state['lon'])
            
            # Draw robot position
            cv2.circle(display_img, (x_pixel, y_pixel), 5, (0, 255, 0), -1)
            
            # Draw robot heading
            heading_len = 20
            end_x = int(x_pixel + heading_len * math.cos(state['theta']))
            end_y = int(y_pixel + heading_len * math.sin(state['theta']))
            cv2.line(display_img, (x_pixel, y_pixel), (end_x, end_y), (0, 255, 0), 2)
        except Exception as e:
            print(f"Error drawing robot position: {e}")
        
        print("Drawing text info...")
        # Draw text information
        info_text = [
            f"Position: ({state['x']:.2f}, {state['y']:.2f})",
            f"Heading: {math.degrees(state['theta']):.2f}°",
        ]
        
        # Add GPS info if available
        if 'lat' in state and 'lon' in state:
            info_text.append(f"GPS: ({state['lat']:.6f}, {state['lon']:.6f})")
        
        info_text.append(f"Updates: {self.update_frequency:.1f} Hz")
        
        for i, text in enumerate(info_text):
            cv2.putText(display_img, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(display_img, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        print("Displaying image...")
        # Display the image
        cv2.imshow("Particle Filter Visualization", display_img)
        print("Image displayed")

    def _update_map_image(self, lat, lon):
        """Update the map image centered at the given lat/lon"""
        try:
            # Get map tile from OpenStreetMap
            map_img = self._get_map_tile(lat, lon, self.zoom_level)
            if map_img is not None:
                self.canvas = map_img
            else:
                # If map tile retrieval fails, create a blank canvas with grid
                self.canvas = np.ones((self.map_size[0], self.map_size[1], 3), dtype=np.uint8) * 240
                # Draw grid
                for i in range(0, self.map_size[0], 50):
                    cv2.line(self.canvas, (i, 0), (i, self.map_size[1]), (200, 200, 200), 1)
                for i in range(0, self.map_size[1], 50):
                    cv2.line(self.canvas, (0, i), (self.map_size[0], i), (200, 200, 200), 1)
        except Exception as e:
            print(f"Error updating map: {e}")
            # Create a blank canvas
            self.canvas = np.ones((self.map_size[0], self.map_size[1], 3), dtype=np.uint8) * 240

    def _get_map_tile(self, lat, lon, zoom):
        """Get a map tile from OpenStreetMap"""
        try:
            # Calculate tile coordinates
            n = 2.0 ** zoom
            xtile = int((lon + 180.0) / 360.0 * n)
            ytile = int((1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n)
            
            # Check cache
            cache_key = f"{xtile}_{ytile}_{zoom}"
            if cache_key in self.map_cache:
                return self.map_cache[cache_key]
            
            # Construct URL
            url = f"https://tile.openstreetmap.org/{zoom}/{xtile}/{ytile}.png"
            
            print(f"Fetching map tile from {url}...")
            
            # Add timeout to prevent hanging indefinitely
            response = requests.get(url, 
                                   headers={'User-Agent': 'ParticleFilterVisualizer/1.0'},
                                   timeout=3.0)  # 3 second timeout
            
            if response.status_code == 200:
                print("Map tile fetched successfully")
                img = Image.open(BytesIO(response.content))
                img = img.resize(self.map_size)
                map_img = np.array(img)
                # Convert RGB to BGR (OpenCV format)
                map_img = cv2.cvtColor(map_img, cv2.COLOR_RGB2BGR)
                
                # Cache the result
                self.map_cache[cache_key] = map_img
                return map_img
            else:
                print(f"Failed to get map tile: {response.status_code}")
                return None
        except requests.exceptions.Timeout:
            print("Map tile request timed out")
            return None
        except Exception as e:
            print(f"Error getting map tile: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _ned_to_latlon(self, north, east):
        """Convert NED coordinates back to lat/lon using the particle filter's projection"""
        # Use the particle filter's projection if available
        if self.pf.proj is not None:
            # Use the same projection method as in StateEstimationPF
            lon, lat = self.pf.proj(east, north, inverse=True)
            return lat, lon
        
        # Fallback to approximate calculation if projection isn't available
        if self.pf.ref_lat is None or self.pf.ref_lon is None:
            return 0, 0
        
        # Convert meters to degrees (approximate)
        dlat = north / self.pf.EARTH_RADIUS
        dlon = east / (self.pf.EARTH_RADIUS * math.cos(math.radians(self.pf.ref_lat)))
        
        lat = self.pf.ref_lat + math.degrees(dlat)
        lon = self.pf.ref_lon + math.degrees(dlon)
        
        return lat, lon

    def _latlon_to_pixel(self, lat, lon):
        """Convert lat/lon to pixel coordinates on the current map"""
        if self.current_center is None:
            return self.map_size[0] // 2, self.map_size[1] // 2
        
        # Calculate meters from center
        center_lat, center_lon = self.current_center
        y_meters = self._haversine(center_lat, center_lon, lat, center_lon) * (1 if lat > center_lat else -1)
        x_meters = self._haversine(center_lat, center_lon, center_lat, lon) * (1 if lon > center_lon else -1)
        
        # Convert to pixels
        center_x, center_y = self.map_size[0] // 2, self.map_size[1] // 2
        pixel_x = center_x + self._meters_to_pixels(x_meters)
        pixel_y = center_y - self._meters_to_pixels(y_meters)  # Negative because y-axis is inverted
        
        return pixel_x, pixel_y

    def _meters_to_pixels(self, meters):
        """Convert meters to pixels based on current zoom level"""
        # Approximate conversion based on zoom level
        meters_per_pixel = 156543.03392 * math.cos(math.radians(self.current_center[0])) / (2 ** self.zoom_level)
        return meters / meters_per_pixel

    def _haversine(self, lat1, lon1, lat2, lon2):
        """Calculate the great circle distance between two points in meters"""
        R = 6371000  # Earth radius in meters
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c

    def _distance(self, point1, point2):
        """Calculate distance between two lat/lon points in meters"""
        return self._haversine(point1[0], point1[1], point2[0], point2[1])

# Example usage
if __name__ == "__main__":
    # Create and start particle filter
    pf = StateEstimationPF()
    pf.start()
    
    # Create and start visualizer
    viz = ParticleFilterVisualizer(pf)
    
    try:
        viz.start()
        # Keep main thread alive
        while viz.running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        viz.stop()
        pf.stop()
        print("Program terminated")
