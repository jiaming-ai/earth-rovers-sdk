import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import threading
import time
from clear.pf import StateEstimationPF
import math

class ParticleFilterVisualizer:
    def __init__(self, particle_filter: StateEstimationPF, update_interval: float = 0.1):
        self.pf = particle_filter
        self.update_interval = update_interval
        
        # Create figure and axes with cartopy projection
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
        
        # Set up the map tiles
        self.tiles = cimgt.OSM()
        self.ax.add_image(self.tiles, 19)  # Zoom level 19 (very detailed)
        
        # Initialize plot elements
        self.position_marker = None
        self.uncertainty_circle = None
        self.particles_scatter = None
        
        # Visualization parameters
        self.arrow_size = 2.0  # meters
        self.min_zoom_width = 50  # meters
        
        # Control flags
        self.running = False
        self.viz_thread = None

    def start(self):
        """Start the visualization thread"""
        self.running = True
        self.viz_thread = threading.Thread(target=self._update_loop)
        self.viz_thread.daemon = True
        self.viz_thread.start()
        plt.show()

    def stop(self):
        """Stop the visualization thread"""
        self.running = False
        if self.viz_thread:
            self.viz_thread.join()

    def _update_loop(self):
        """Main update loop for visualization"""
        while self.running:
            try:
                self._update_plot()
                plt.pause(self.update_interval)
            except Exception as e:
                print(f"Error in visualization update: {e}")
                time.sleep(0.1)

    def _update_plot(self):
        """Update the visualization with current state"""
        state = self.pf.get_state()
        if not all(v is not None for v in [state['x'], state['y'], state['theta']]):
            return

        # Get current position and heading
        x, y = state['x'], state['y']
        theta = state['theta']

        # Convert NED coordinates back to lat/lon
        lat, lon = self._ned_to_latlon(x, y)
        
        # Calculate arrow endpoints for heading indicator
        dx = self.arrow_size * math.cos(theta)
        dy = self.arrow_size * math.sin(theta)
        
        # Convert to map projection coordinates
        proj_coords = ccrs.Mercator().transform_point(lon, lat, ccrs.PlateCarree())
        
        # Clear previous plots
        if self.position_marker:
            self.position_marker.remove()
        if self.uncertainty_circle:
            self.uncertainty_circle.remove()
        if self.particles_scatter:
            self.particles_scatter.remove()

        # Plot particles
        particle_lats, particle_lons = [], []
        for particle in self.pf.particles:
            plat, plon = self._ned_to_latlon(particle[0], particle[1])
            particle_lats.append(plat)
            particle_lons.append(plon)
        
        # Plot particles with low opacity
        self.particles_scatter = self.ax.scatter(
            particle_lons, particle_lats,
            transform=ccrs.PlateCarree(),
            color='blue', alpha=0.1, s=1
        )

        # Plot position and heading
        self.position_marker = self.ax.arrow(
            lon, lat,
            dx * 1e-4, dy * 1e-4,  # Scale arrow to approximate meters
            transform=ccrs.PlateCarree(),
            color='red',
            width=0.5e-4,
            head_width=1e-4
        )

        # Calculate uncertainty from particle spread
        if len(self.pf.particles) > 0:
            std_dev = np.std(self.pf.particles[:, :2], axis=0)
            uncertainty_radius = np.mean(std_dev)
            
            if uncertainty_radius > 1.0:  # Only show uncertainty circle if significant
                # Convert radius to degrees (approximate)
                radius_deg = uncertainty_radius * 1e-4
                circle = plt.Circle(
                    (lon, lat),
                    radius_deg,
                    color='red',
                    fill=False,
                    alpha=0.5,
                    transform=ccrs.PlateCarree()
                )
                self.uncertainty_circle = self.ax.add_patch(circle)

        # Update map extent to keep position centered
        self._update_map_extent(lon, lat)

    def _ned_to_latlon(self, north, east):
        """Convert NED coordinates back to lat/lon"""
        if self.pf.ref_lat is None or self.pf.ref_lon is None:
            return 0, 0
        
        # Convert meters to degrees (approximate)
        dlat = north / self.pf.EARTH_RADIUS
        dlon = east / (self.pf.EARTH_RADIUS * math.cos(math.radians(self.pf.ref_lat)))
        
        lat = self.pf.ref_lat + math.degrees(dlat)
        lon = self.pf.ref_lon + math.degrees(dlon)
        
        return lat, lon

    def _update_map_extent(self, center_lon, center_lat):
        """Update the map extent to keep the position centered"""
        # Calculate width in degrees (approximate)
        width_deg = self.min_zoom_width * 1e-4
        
        # Set map extent
        self.ax.set_extent([
            center_lon - width_deg,
            center_lon + width_deg,
            center_lat - width_deg,
            center_lat + width_deg
        ])

# Example usage
if __name__ == "__main__":
    # Create and start particle filter
    pf = StateEstimationPF()
    pf.start()
    
    # Create and start visualizer
    viz = ParticleFilterVisualizer(pf)
    
    try:
        viz.start()
    except KeyboardInterrupt:
        viz.stop()
        pf.stop()
        print("Program terminated")
