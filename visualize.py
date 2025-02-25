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
        self.free_navigation_mode = False  # Track if we're in free navigation mode
        
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
                # Check if the recenter button was clicked
                if self.free_navigation_mode and 10 <= x <= 130 and 50 <= y <= 90:
                    logger.debug("Recenter button clicked")
                    self.free_navigation_mode = False
                    self.drag_offset = (0, 0)
                    return
                    
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
                    # Reverse the direction to make dragging more intuitive
                    # When dragging left, we want to see what's to the right
                    center_lat, center_lon = self.current_center
                    new_lat, new_lon = self._offset_latlon(center_lat, center_lon, dy, dx)
                    
                    self._update_map_image(new_lat, new_lon)
                    self.current_center = (new_lat, new_lon)
                    
                    # Set free navigation mode
                    self.free_navigation_mode = True
                    
                    # Reset drag offset
                    self.drag_offset = (0, 0)
            
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.dragging:
                    # Calculate drag offset
                    dx = x - self.drag_start[0]
                    dy = y - self.drag_start[1]
                    self.drag_offset = (dx, dy)
            
            # Fix for mouse wheel zoom
            elif event == cv2.EVENT_MOUSEWHEEL:
                # Extract wheel direction from flags
                wheel_direction = np.sign(flags)
                logger.debug(f"Mouse wheel event detected, direction: {wheel_direction}, flags: {flags}")
                
                # Check if Ctrl key is pressed
                if flags & cv2.EVENT_FLAG_CTRLKEY:
                    # Store the current center before changing zoom
                    old_zoom = self.zoom_level
                    
                    if wheel_direction < 0:  # Zoom out
                        self.zoom_level = max(10, self.zoom_level - 1)  # Lower minimum to 10 or another value
                        logger.info(f"Zooming out to level {self.zoom_level}")
                    else:  # Zoom in
                        self.zoom_level = min(21, self.zoom_level + 1)
                        logger.info(f"Zooming in to level {self.zoom_level}")
                    
                    # Update map with new zoom level if we have a center
                    if self.current_center and old_zoom != self.zoom_level:
                        # Clear the map cache when zoom level changes
                        self.map_cache.clear()
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

    def _draw_robot_triangle(self, img, position, heading, size=15, color=(0, 255, 0), thickness=2):
        """Draw the robot as a triangle pointing in the heading direction"""
        x, y = position
        
        # Check if heading is None and provide a default value
        if heading is None:
            logger.warning("Received None heading, using 0 as default")
            angle = 0.0
        else:
            angle = heading
        
        # Calculate triangle points
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
        
        base_length = 20
        # Scale arrow length based on weight
        arrow_length = max(10, int(base_length * weight))
        
        # Calculate arrow endpoint
        end_x = int(x + arrow_length * math.cos(heading))
        end_y = int(y + arrow_length * math.sin(heading))
        
        # Draw arrow
        cv2.arrowedLine(img, (x, y), (end_x, end_y), color, 1, tipLength=0.3)

    def _update_display_safe(self, state):
        """Update the visualization with current state - safer version"""
        # Step 1: Get base map image
        try:
            # Ensure state has valid values
            for key in ['x', 'y', 'theta']:
                if key not in state or state[key] is None:
                    logger.warning(f"Missing or None value for {key} in state, using 0 as default")
                    state[key] = 0.0
                
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
                    # Only update the map center if not in free navigation mode
                    if not self.free_navigation_mode:
                        self.current_center = (lat, lon)
                        self._update_map_image(lat, lon)
                    map_img = self._get_map_tile_cached(self.current_center[0], self.current_center[1])
        except Exception as e:
            logger.error(f"Error preparing map: {e}")
            map_img = self._create_blank_map()
            state['lat'] = 0
            state['lon'] = 0
            
        # Create a safe copy of the map for drawing
        display_img = map_img.copy()
        
        # Step 2: Draw particles, robot position, and info overlay if needed
        self._draw_particles_safe(display_img, state)
        
        # Draw the robot position
        x_pixel, y_pixel = self._latlon_to_pixel(state['lat'], state['lon'])
        self._draw_robot_triangle(display_img, (x_pixel, y_pixel), state['theta'])
        
        # Draw info overlay
        self._draw_info_overlay(display_img, state)
        logger.debug("Drew particles, robot position, and info overlay")
        
        # Step 3: Display the image
        cv2.imshow("Particle Filter Visualization", display_img)

    
    def _draw_particles_safe(self, display_img, state):
        """Draw particles on the display image - safer version"""
        ps, hs = self.pf.get_uncertainty()
        total_uncertainty = ps + hs
        
        if total_uncertainty <= 0.5 or not hasattr(self.pf, 'particles') or self.pf.particles is None or not hasattr(self.pf, 'weights'):
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
        ps, hs = self.pf.get_uncertainty()
        hs_deg = math.degrees(hs)
        
        # Create info text
        info_text = [
            f"Position: ({state['x']:.2f}m, {state['y']:.2f}m)",
            f"Heading: {math.degrees(state['theta']):.1f} deg"
        ]
        
        # Add GPS info if available
        if 'lat' in state and 'lon' in state:
            info_text.append(f"GPS: ({state['lat']:.6f}, {state['lon']:.6f})")
        
        # Add data rate info
        data_rate = self.pf.get_data_fetching_frequency() or 0
        pf_rate = self.pf.get_particle_filter_frequency() or 0
        info_text.append(f"Data rate: {data_rate:.1f} Hz")
        info_text.append(f"PF update rate: {pf_rate:.1f} Hz")
        info_text.append(f"Std: {ps:.2f}m, {hs_deg:.2f}deg")
        
        # Draw semi-transparent background
        info_box_width = 300
        info_box_padding = 10
        line_space = 25
        info_box_height = len(info_text) * line_space + info_box_padding * 2
        try:
            # Create a simple solid background instead of using addWeighted
            cv2.rectangle(display_img, 
                         (self.map_size[0] - info_box_width - info_box_padding, info_box_padding), 
                         (self.map_size[0] - info_box_padding, info_box_height), 
                         (0, 0, 0), -1)
            
            # Draw text
            font_scale = 0.5
            font_thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_color = (255, 255, 255)
            for i, text in enumerate(info_text):
                position = (self.map_size[0] - info_box_width + 5, info_box_padding + line_space + i * line_space)
                cv2.putText(display_img, text, 
                           position, 
                           font, 
                           font_scale, 
                           font_color, 
                           font_thickness)
        except Exception as e:
            logger.error(f"Error drawing info box: {e}")
        
        # Draw drag indicator and recenter button if in free navigation mode
        if self.free_navigation_mode:
            # Show different messages depending on whether actively dragging
            if self.dragging:
                cv2.putText(display_img, "Dragging Map...", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(display_img, "Free Navigation Mode", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw recenter button
            button_width = 120
            button_height = 40
            button_x = 10
            button_y = 50
            cv2.rectangle(display_img, 
                         (button_x, button_y), 
                         (button_x + button_width, button_y + button_height), 
                         (0, 120, 255), -1)
            cv2.putText(display_img, "Recenter", 
                       (button_x + 10, button_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # def _calculate_uncertainty(self):
    #     """Calculate the uncertainty of the particle filter"""
    #     ps, hs = self.pf.get_uncertainty()
    #     return ps
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
            # Construct URL - try alternative tile servers
            servers = [
                f"https://a.tile.openstreetmap.org/{self.zoom_level}/{xtile}/{ytile}.png",
                f"https://b.tile.openstreetmap.org/{self.zoom_level}/{xtile}/{ytile}.png",
                f"https://c.tile.openstreetmap.org/{self.zoom_level}/{xtile}/{ytile}.png"
            ]
            
            # Try each server until one works
            for url in servers:
                try:
                    logger.debug(f"Fetching map tile from {url}")
                    
                    # Add timeout to prevent hanging indefinitely
                    response = requests.get(url, 
                                          headers={'User-Agent': 'ParticleFilterVisualizer/1.0 (https://yourwebsite.com)'},
                                          timeout=5.0)  # Increased timeout
                    
                    if response.status_code == 200:
                        img = Image.open(BytesIO(response.content))
                        img = img.resize(self.map_size)
                        map_img = np.array(img)
                        # Convert RGB to BGR (OpenCV format)
                        map_img = cv2.cvtColor(map_img, cv2.COLOR_RGB2BGR)
                        
                        # Cache the result (with size limit)
                        if len(self.map_cache) >= self.max_cache_size:
                            # Remove oldest key (first key)
                            oldest_key = next(iter(self.map_cache))
                            del self.map_cache[oldest_key]
                        
                        self.map_cache[cache_key] = map_img
                        # Update canvas for next rendering
                        self.canvas = map_img
                        return True
                    else:
                        logger.warning(f"Failed to get map tile from {url}: HTTP {response.status_code}")
                        # Continue to next server
                except requests.exceptions.Timeout:
                    logger.warning(f"Map tile request timed out for {url}")
                    # Continue to next server
                except Exception as e:
                    logger.warning(f"Error getting map tile from {url}: {e}")
                    # Continue to next server
            
            # If we get here, all servers failed
            logger.error("All map tile servers failed")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in map tile fetching: {e}")
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

    # Add a new method to pre-fetch surrounding tiles
    def _prefetch_surrounding_tiles(self, center_lat, center_lon):
        """Pre-fetch surrounding tiles to improve map navigation experience"""
        try:
            # Calculate center tile
            n = 2.0 ** self.zoom_level
            center_xtile = int((center_lon + 180.0) / 360.0 * n)
            center_ytile = int((1.0 - math.log(math.tan(math.radians(center_lat)) + 1.0 / math.cos(math.radians(center_lat))) / math.pi) / 2.0 * n)
            
            # Queue fetching of surrounding tiles (3x3 grid)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    # Skip center tile as it's already being fetched
                    if dx == 0 and dy == 0:
                        continue
                        
                    xtile = center_xtile + dx
                    ytile = center_ytile + dy
                    
                    # Calculate lat/lon for this tile
                    lon_deg = xtile / n * 360.0 - 180.0
                    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
                    lat_deg = math.degrees(lat_rad)
                    
                    cache_key = f"{xtile}_{ytile}_{self.zoom_level}"
                    
                    # Only fetch if not already in cache
                    if cache_key not in self.map_cache:
                        # Use low priority thread to fetch
                        self.thread_pool.submit(self._fetch_map_tile, lat_deg, lon_deg, xtile, ytile, cache_key)
        except Exception as e:
            logger.error(f"Error in prefetch: {e}")


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

            # first wait the pf to be initialized
            
            try:
                if not viz.pf.is_initialized:
                    logger.debug("Particle filter is not initialized")
                    viz._show_waiting_image("Waiting for particle filter to be initialized...")
                    time.sleep(0.1)
                    continue
            except Exception as e:
                logger.error(f"Error getting state: {e}")
                viz._show_waiting_image("Error getting state")
                time.sleep(0.1)
                continue
            
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