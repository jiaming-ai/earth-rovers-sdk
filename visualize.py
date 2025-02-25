import cv2
import numpy as np
import threading
import time
import logging
from clear.pf import StateEstimationPF
import math
import requests
from io import BytesIO
from PIL import Image
from collections import deque
import os
import argparse
import traceback
import sys
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

# Force OpenCV to use a specific backend
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

# Setup logging
logger = logging.getLogger("PFVisualizer")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)  # Default to INFO level

class ParticleFilterVisualizer:
    def __init__(self, particle_filter: StateEstimationPF, update_interval: float = 0.1):
        self.pf: StateEstimationPF = particle_filter
        self.update_interval = update_interval
        
        # Visualization parameters
        self.map_size = (800, 800)  # pixels
        self.zoom_level = 19
        self.max_particles_to_show = 50  # Max particles to display
        
        # Initialize the canvas
        self.canvas = np.zeros((self.map_size[0], self.map_size[1], 3), dtype=np.uint8)
        
        # Map cache with a size limit to prevent memory issues
        self.map_cache = {}
        self.max_cache_size = 10  # Maximum number of map tiles to cache
        self.current_center = None
        
        # Control flags
        self.running = False
        self.viz_thread = None
        
        # Mouse interaction variables
        self.dragging = False
        self.drag_start = None
        self.drag_offset = (0, 0)
        
        # Performance tracking
        self.update_times = deque(maxlen=30)  # Store last 30 update times
        self.update_frequency = 0
        self.last_update_time = time.time()
        
        # Thread communication and safety
        self.task_queue = Queue()
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        self.map_fetch_future = None
        
        # Initialize OpenCV window in main thread
        logger.info("Creating OpenCV window...")
        cv2.namedWindow("Particle Filter Visualization", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Particle Filter Visualization", self.map_size[0], self.map_size[1])
        
        # Show a waiting image
        self._show_waiting_image("Initializing visualization...")

    def _show_waiting_image(self, message):
        """Display a waiting message while initializing or loading"""
        try:
            blank_img = np.ones((self.map_size[0], self.map_size[1], 3), dtype=np.uint8) * 240
            cv2.putText(blank_img, message, (50, self.map_size[1]//2), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(blank_img, "Press ESC to exit", (50, self.map_size[1]//2 + 40), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
            cv2.imshow("Particle Filter Visualization", blank_img)
            cv2.waitKey(1)  # Process the window
        except Exception as e:
            logger.error(f"Error showing waiting image: {e}")
            traceback.print_exc()

    def start(self):
        """Start the visualization"""
        logger.info("Starting visualization...")
        self.running = True
        
        # Register mouse callback
        try:
            cv2.setMouseCallback("Particle Filter Visualization", self._mouse_callback)
        except Exception as e:
            logger.error(f"Failed to set mouse callback: {e}")
        
        logger.info("Visualization started")

    def stop(self):
        """Stop the visualization"""
        logger.info("Stopping visualization...")
        self.running = False
        cv2.destroyAllWindows()
        logger.info("Visualization stopped")

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for dragging and zooming"""
        try:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.dragging = True
                self.drag_start = (x, y)
                logger.debug("Mouse drag started at (%d, %d)", x, y)
            
            elif event == cv2.EVENT_LBUTTONUP:
                if not self.dragging:
                    return
                    
                self.dragging = False
                logger.debug("Mouse drag ended")
                
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
                # Handle zoom with ctrl key
                if flags & cv2.EVENT_FLAG_CTRLKEY:
                    wheel_direction = np.sign(flags >> 16)  # Extract wheel direction
                    
                    if wheel_direction < 0:  # Zoom in
                        self.zoom_level = min(21, self.zoom_level + 1)
                        logger.debug("Zooming in to level %d", self.zoom_level)
                    else:  # Zoom out
                        self.zoom_level = max(14, self.zoom_level - 1)
                        logger.debug("Zooming out to level %d", self.zoom_level)
                    
                    # Update map with new zoom level if we have a center
                    if self.current_center:
                        self._update_map_image(self.current_center[0], self.current_center[1])
        
        except Exception as e:
            logger.error(f"Error in mouse callback: {e}", exc_info=logger.isEnabledFor(logging.DEBUG))

    def _pixels_to_meters(self, pixels):
        """Convert pixels to meters based on current zoom level"""
        if self.current_center is None:
            return 0
        # Earth's circumference at the equator divided by 2^zoom level pixels
        meters_per_pixel = 156543.03392 * math.cos(math.radians(self.current_center[0])) / (2 ** self.zoom_level)
        return pixels * meters_per_pixel

    def _meters_to_pixels(self, meters):
        """Convert meters to pixels based on current zoom level"""
        if self.current_center is None:
            return 0
        meters_per_pixel = 156543.03392 * math.cos(math.radians(self.current_center[0])) / (2 ** self.zoom_level)
        return int(meters / meters_per_pixel)

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

    # def _update_loop(self):
    #     """Main update loop for visualization"""
    #     logger.info("Starting visualization update loop")
    #     last_frame_time = time.time()
    #     error_count = 0
        
    #     while self.running:
    #         try:
    #             # Emergency exit if too many errors
    #             if error_count > 10:
    #                 logger.error("Too many errors, shutting down visualization")
    #                 self.running = False
    #                 break
                
    #             # Limit frame rate to prevent excessive CPU usage
    #             current_time = time.time()
    #             elapsed = current_time - last_frame_time
                
    #             if elapsed < self.update_interval:
    #                 # Wait for the remaining time to achieve target frame rate
    #                 time.sleep(max(0, self.update_interval - elapsed))
    #                 continue
                
    #             last_frame_time = current_time
    #             logger.debug(f"Update loop iteration at {current_time:.3f}")
                
    #             # Calculate update frequency
    #             self.update_times.append(current_time)
                
    #             if len(self.update_times) >= 2:
    #                 # Calculate updates per second over the tracked period
    #                 time_span = self.update_times[-1] - self.update_times[0]
    #                 if time_span > 0:
    #                     self.update_frequency = (len(self.update_times) - 1) / time_span
                
    #             # Get the state from particle filter (with timeout protection)
    #             try:
    #                 state = self.pf.get_state()
    #                 logger.debug(f"Get state from PF: {state}")
    #                 if state is None or not all(k in state for k in ['x', 'y', 'theta']):
    #                     logger.debug("Invalid state from particle filter")
    #                     self._show_waiting_image("Waiting for valid state...")
    #                     time.sleep(0.1)
    #                     continue
    #             except Exception as e:
    #                 logger.error(f"Error getting state: {e}")
    #                 self._show_waiting_image("Error getting state")
    #                 time.sleep(0.1)
    #                 continue
                
    #             # Safe update display
    #             try:
    #                 self._update_display_safe(state)
    #             except Exception as e:
    #                 logger.error(f"Error updating display: {e}")
    #                 error_count += 1
    #                 traceback.print_exc()
    #                 time.sleep(0.1)
    #                 continue
                
    #             # Process key events with a very short timeout
    #             try:
    #                 key = cv2.waitKey(1) & 0xFF
    #                 if key == 27:  # ESC key
    #                     logger.info("ESC key pressed, exiting...")
    #                     self.running = False
    #                     break
    #             except Exception as e:
    #                 logger.error(f"Error processing key events: {e}")
    #                 error_count += 1
                
    #             # Reset error count on successful iteration
    #             error_count = 0
                
    #         except Exception as e:
    #             logger.error(f"Critical error in visualization update loop: {e}")
    #             traceback.print_exc()
    #             error_count += 1
    #             time.sleep(0.5)  # Longer sleep on critical error
                
    #     logger.info("Visualization update loop terminated")

    def _draw_text_with_background(self, img, text, position, font_scale=0.7, thickness=1, 
                                  text_color=(255, 255, 255), bg_color=(0, 0, 0, 180)):
        """Draw text with a semi-transparent background"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Create background rectangle with padding
        padding = 5
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

    def _draw_robot_triangle(self, img, position, heading, size=15, color=(0, 255, 0), thickness=2):
        """Draw the robot as a triangle pointing in the heading direction"""
        x, y = position
        
        # Calculate triangle points
        angle = heading
        pt1 = (int(x + size * math.cos(angle)), int(y + size * math.sin(angle)))
        pt2 = (int(x + size/2 * math.cos(angle + 2.5)), int(y + size/2 * math.sin(angle + 2.5)))
        pt3 = (int(x + size/2 * math.cos(angle - 2.5)), int(y + size/2 * math.sin(angle - 2.5)))
        
        # Draw triangle
        cv2.line(img, pt1, pt2, color, thickness)
        cv2.line(img, pt1, pt3, color, thickness)
        cv2.line(img, pt2, pt3, color, thickness)
        
        # Draw a small circle at the robot's position
        cv2.circle(img, (x, y), 3, color, -1)

    def _draw_particle_arrow(self, img, position, heading, weight=1.0, color=(0, 0, 255)):
        """Draw a particle as an arrow showing its position and heading"""
        x, y = position
        
        # Scale arrow length based on weight
        arrow_length = max(5, int(10 * weight))
        
        # Calculate arrow endpoint
        end_x = int(x + arrow_length * math.cos(heading))
        end_y = int(y + arrow_length * math.sin(heading))
        
        # Draw arrow
        cv2.arrowedLine(img, (x, y), (end_x, end_y), color, 1, tipLength=0.3)

    def _update_display_safe(self, state):
        """Update the visualization with current state - safer version"""
        # Step 1: Get base map image
        try:
            # Convert NED coordinates to lat/lon
            lat, lon = self.pf.reverse_ned_to_latlon(state['x'], state['y'])
            state['lat'] = lat
            state['lon'] = lon
            
            # Initialize map on first valid position
            if self.current_center is None:
                logger.debug("Initializing map center")
                self.current_center = (lat, lon)
                # Create a blank map initially to avoid blocking
                map_img = self._create_blank_map()
                # Queue map update for next frame
                self.thread_pool.submit(self._update_map_image, lat, lon)
            else:
                # Get map image - use cached or create blank
                if self.dragging:
                    map_img = self.canvas.copy()
                else:
                    map_img = self._get_map_tile_cached(lat, lon)
        except Exception as e:
            logger.error(f"Error preparing map: {e}")
            map_img = self._create_blank_map()
            state['lat'] = 0
            state['lon'] = 0
            
        # Create a safe copy of the map for drawing
        display_img = map_img.copy()
        
        # Step 2: Draw particles, robot position, and info overlay if needed
        self._draw_particles_safe(display_img, state)
        x_pixel, y_pixel = self._latlon_to_pixel(state['lat'], state['lon'])
        self._draw_robot_triangle(display_img, (x_pixel, y_pixel), state['theta'])
        self._draw_info_overlay(display_img, state)
        logger.debug("Drew particles, robot position, and info overlay")
        
        # Step 3: Display the image
        cv2.imshow("Particle Filter Visualization", display_img)

    
    def _draw_particles_safe(self, display_img, state):
        """Draw particles on the display image - safer version"""
        uncertainty = self._calculate_uncertainty()
        
        if uncertainty <= 0.5 or not hasattr(self.pf, 'particles') or self.pf.particles is None or not hasattr(self.pf, 'weights'):
            return
        
        try:
            # Sort particles by weight to show the most significant ones
            indices = np.argsort(-self.pf.weights)[:min(self.max_particles_to_show, len(self.pf.weights))]
            
            # Normalize weights for display
            max_weight = np.max(self.pf.weights)
            min_weight = np.min(self.pf.weights)
            weight_range = max(1e-10, max_weight - min_weight)  # Avoid division by zero
            
            for i in indices:
                try:
                    # Get particle position and orientation
                    p_north, p_east = self.pf.particles[i, 0], self.pf.particles[i, 1]
                    p_theta = self.pf.particles[i, 2]
                    
                    # Convert to lat/lon
                    p_lat, p_lon = self.pf.reverse_ned_to_latlon(p_north, p_east)
                    
                    # Convert to pixel coordinates
                    p_x, p_y = self._latlon_to_pixel(p_lat, p_lon)
                    
                    # Skip if out of bounds
                    if not (0 <= p_x < self.map_size[0] and 0 <= p_y < self.map_size[1]):
                        continue
                    
                    # Calculate normalized weight
                    normalized_weight = (self.pf.weights[i] - min_weight) / weight_range
                    
                    # Draw particle arrow
                    self._draw_particle_arrow(display_img, (p_x, p_y), p_theta, normalized_weight)
                except Exception as e:
                    # Skip this particle on error
                    continue
        except Exception as e:
            logger.error(f"Error processing particles: {e}")
    
    def _draw_info_overlay(self, display_img, state):
        """Draw information overlay - safer version"""
        # Calculate uncertainty
        uncertainty = self._calculate_uncertainty()
        
        # Create info text
        info_text = [
            f"Position: ({state['x']:.2f}m, {state['y']:.2f}m)",
            f"Heading: {math.degrees(state['theta']):.1f}°"
        ]
        
        # Add GPS info if available
        if 'lat' in state and 'lon' in state:
            info_text.append(f"GPS: ({state['lat']:.6f}, {state['lon']:.6f})")
        
        info_text.append(f"Update rate: {self.update_frequency:.1f} Hz")
        info_text.append(f"Uncertainty: {uncertainty:.2f}")
        
        # Draw semi-transparent background
        info_box_width = 250
        info_box_padding = 15
        info_box_height = len(info_text) * 25 + info_box_padding * 2
        
        try:
            # Create a simple solid background instead of using addWeighted
            cv2.rectangle(display_img, 
                         (self.map_size[0] - info_box_width - info_box_padding, info_box_padding), 
                         (self.map_size[0] - info_box_padding, info_box_height), 
                         (0, 0, 0), -1)
            
            # Draw text
            for i, text in enumerate(info_text):
                cv2.putText(display_img, text, 
                           (self.map_size[0] - info_box_width + 5, info_box_padding + 25 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        except Exception as e:
            logger.error(f"Error drawing info box: {e}")
        
        # Draw drag indicator if dragging
        if self.dragging:
            cv2.putText(display_img, "Dragging...", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def _calculate_uncertainty(self):
        """Calculate the uncertainty of the particle filter"""
        ps, hs = self.pf.get_uncertainty()
        return ps
    def _create_blank_map(self):
        """Create a blank map with grid lines"""
        blank_map = np.ones((self.map_size[0], self.map_size[1], 3), dtype=np.uint8) * 240
        
        # Draw grid lines
        grid_spacing = 50
        for i in range(0, self.map_size[0], grid_spacing):
            cv2.line(blank_map, (i, 0), (i, self.map_size[1]), (200, 200, 200), 1)
        for i in range(0, self.map_size[1], grid_spacing):
            cv2.line(blank_map, (0, i), (self.map_size[0], i), (200, 200, 200), 1)
        
        return blank_map

    def _update_map_image(self, lat, lon):
        """Update the map image centered at the given lat/lon (called in background thread)"""
        try:
            # Calculate tile coordinates
            n = 2.0 ** self.zoom_level
            xtile = int((lon + 180.0) / 360.0 * n)
            ytile = int((1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n)
            
            # Update current center
            self.current_center = (lat, lon)
            cache_key = f"{xtile}_{ytile}_{self.zoom_level}"
            
            # Try to fetch the map tile
            success = self._fetch_map_tile(lat, lon, xtile, ytile, cache_key)
            
            if not success:
                logger.debug("Using blank map as fallback")
                # Use a blank map if fetch failed
                self.canvas = self._create_blank_map()
                
        except Exception as e:
            logger.error(f"Error updating map: {e}")
            # Create a blank canvas
            self.canvas = self._create_blank_map()

    def _get_map_tile_cached(self, lat, lon):
        """Get a map tile, with proper caching and fallback"""
        # Always update the current center
        self.current_center = (lat, lon)
        
        # Calculate tile coordinates
        try:
            n = 2.0 ** self.zoom_level
            xtile = int((lon + 180.0) / 360.0 * n)
            ytile = int((1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n)
            cache_key = f"{xtile}_{ytile}_{self.zoom_level}"
            
            # First check the cache
            if cache_key in self.map_cache:
                logger.debug("Using cached map tile")
                return self.map_cache[cache_key]
            
            # If no cached tile, use the current canvas or create blank map
            if self.canvas is not None and self.canvas.shape == (self.map_size[0], self.map_size[1], 3):
                logger.debug("Using existing canvas as fallback")
                # Queue a background fetch for next time
                if self.map_fetch_future is None or self.map_fetch_future.done():
                    self.map_fetch_future = self.thread_pool.submit(self._fetch_map_tile, lat, lon, xtile, ytile, cache_key)
                return self.canvas
            else:
                logger.debug("Creating blank map")
                blank_map = self._create_blank_map()
                # Queue a background fetch
                if self.map_fetch_future is None or self.map_fetch_future.done():
                    self.map_fetch_future = self.thread_pool.submit(self._fetch_map_tile, lat, lon, xtile, ytile, cache_key)
                return blank_map
                
        except Exception as e:
            logger.error(f"Error in map tile handling: {e}")
            return self._create_blank_map()
    
    def _fetch_map_tile(self, lat, lon, xtile, ytile, cache_key):
        """Fetch a map tile in background thread"""
        try:
            # Construct URL
            url = f"https://tile.openstreetmap.org/{self.zoom_level}/{xtile}/{ytile}.png"
            logger.debug(f"Background fetch of map tile from {url}")
            
            # Add timeout to prevent hanging indefinitely
            response = requests.get(url, 
                                  headers={'User-Agent': 'ParticleFilterVisualizer/1.0'},
                                  timeout=3.0)  # 3 second timeout
            
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                img = img.resize(self.map_size)
                map_img = np.array(img)
                # Convert RGB to BGR (OpenCV format)
                map_img = cv2.cvtColor(map_img, cv2.COLOR_RGB2BGR)
                
                # Cache the result (with size limit)
                if len(self.map_cache) >= self.max_cache_size:
                    # Remove oldest item (first key)
                    oldest_key = next(iter(self.map_cache))
                    del self.map_cache[oldest_key]
                
                self.map_cache[cache_key] = map_img
                # Update canvas for next rendering
                self.canvas = map_img
                return True
            else:
                logger.warning(f"Failed to get map tile: HTTP {response.status_code}")
                return False
        except requests.exceptions.Timeout:
            logger.warning("Map tile request timed out")
            return False
        except Exception as e:
            logger.error(f"Error getting map tile: {e}")
            return False

    def _latlon_to_pixel(self, lat, lon):
        """Convert lat/lon to pixel coordinates on the current map"""
        if self.current_center is None:
            return self.map_size[0] // 2, self.map_size[1] // 2
        
        # Calculate meters from center
        center_lat, center_lon = self.current_center
        y_meters = self._haversine(center_lat, center_lon, lat, center_lon) * (1 if lat > center_lat else -1)
        x_meters = self._haversine(center_lat, center_lon, center_lat, lon) * (1 if lon > center_lon else -1)
        
        # Apply drag offset if dragging
        if self.dragging:
            x_meters += self._pixels_to_meters(self.drag_offset[0])
            y_meters += self._pixels_to_meters(self.drag_offset[1])
        
        # Convert to pixels
        center_x, center_y = self.map_size[0] // 2, self.map_size[1] // 2
        pixel_x = center_x + self._meters_to_pixels(x_meters)
        pixel_y = center_y - self._meters_to_pixels(y_meters)  # Negative because y-axis is inverted
        
        return int(pixel_x), int(pixel_y)

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


def main():
    # Setup command-line arguments
    parser = argparse.ArgumentParser(description='Particle Filter Visualization')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                        default='INFO', help='Set logging level')
    parser.add_argument('--update-interval', type=float, default=0.1,
                        help='Update interval in seconds (default: 0.1)')
    parser.add_argument('--no-map', action='store_true',
                        help='Disable map fetching (use blank map only)')
    
    args = parser.parse_args()
    
    # Set logging level
    logger.setLevel(getattr(logging, args.log_level))
    logger.info(f"Starting with log level: {args.log_level}")
    
    # Create and start particle filter
    logger.info("Initializing particle filter...")
    try:
        pf = StateEstimationPF(log_level=args.log_level)
        pf.start()
    except Exception as e:
        logger.error(f"Failed to initialize particle filter: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Create and start visualizer
    logger.info("Initializing visualizer...")
    viz = ParticleFilterVisualizer(pf, update_interval=args.update_interval)
    viz.start()
    
    # Main visualization loop (moved from _update_loop to main thread)
    logger.info("Visualization running. Press ESC to exit.")
    last_frame_time = time.time()
    error_count = 0
    
    try:
        while viz.running:
            # Emergency exit if too many errors
            if error_count > 10:
                logger.error("Too many errors, shutting down visualization")
                viz.running = False
                break
            
            # Limit frame rate to prevent excessive CPU usage
            current_time = time.time()
            elapsed = current_time - last_frame_time
            
            if elapsed < viz.update_interval:
                # Process key events while waiting for next frame
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    logger.info("ESC key pressed, exiting...")
                    viz.running = False
                    break
                time.sleep(0.01)  # Short sleep to avoid CPU hogging
                continue
            
            last_frame_time = current_time
            logger.debug(f"Update iteration at {current_time:.3f}")
            
            # Calculate update frequency
            viz.update_times.append(current_time)
            
            if len(viz.update_times) >= 2:
                # Calculate updates per second over the tracked period
                time_span = viz.update_times[-1] - viz.update_times[0]
                if time_span > 0:
                    viz.update_frequency = (len(viz.update_times) - 1) / time_span
            
            # Get the state from particle filter (with timeout protection)
            try:
                state = viz.pf.get_state()
                logger.debug(f"Get state from PF: {state}")
                if state is None or not all(k in state for k in ['x', 'y', 'theta']):
                    logger.debug("Invalid state from particle filter")
                    viz._show_waiting_image("Waiting for valid state...")
                    time.sleep(0.1)
                    continue
            except Exception as e:
                logger.error(f"Error getting state: {e}")
                viz._show_waiting_image("Error getting state")
                time.sleep(0.1)
                continue
            
            # Safe update display
            try:
                viz._update_display_safe(state)
            except Exception as e:
                logger.error(f"Error updating display: {e}")
                error_count += 1
                traceback.print_exc()
                time.sleep(0.1)
                continue
            
            # Process key events
            try:
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    logger.info("ESC key pressed, exiting...")
                    viz.running = False
                    break
            except Exception as e:
                logger.error(f"Error processing key events: {e}")
                error_count += 1
            
            # Reset error count on successful iteration
            error_count = 0
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error in main thread: {e}")
        traceback.print_exc()
    finally:
        # Clean shutdown
        logger.info("Shutting down visualization...")
        viz.stop()
        
        logger.info("Shutting down particle filter...")
        pf.stop()
        
        # Force-close all OpenCV windows
        try:
            cv2.destroyAllWindows()
            # Explicitly wait for windows to close
            for i in range(5):
                cv2.waitKey(1)
        except Exception:
            pass
            
        logger.info("Program terminated")


if __name__ == "__main__":
    main()