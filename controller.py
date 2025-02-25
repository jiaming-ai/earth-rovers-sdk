import cv2
import numpy as np
import requests
import json
import time
import base64

class RobotController:
    def __init__(self):
        # API endpoints
        self.control_url = "http://localhost:8000/control"
        self.screenshot_url = "http://localhost:8000/v2/front"
        
        # Control parameters
        self.linear_vel = 0
        self.angular_vel = 0
        
        # Display parameters
        self.window_name = "Robot Camera Feed"
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_thickness = 2
        self.text_color = (0, 255, 0)  # Green
        self.warning_color = (0, 0, 255)  # Red
        
        # Performance tracking
        self.frame_timestamps = []
        self.max_timestamps = 30  # Store last 30 timestamps for FPS calculation
        self.last_timestamp = None
        self.fps = 0
        
        # Timestamp threshold for warning (seconds)
        self.timestamp_threshold = 1.0
        
        # Initialize OpenCV window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1024, 576)  # Front camera resolution
        
        # Print instructions
        print("Robot Control Interface")
        print("----------------------")
        print("W: Move forward")
        print("S: Move backward")
        print("A: Turn left")
        print("D: Turn right")
        print("Space: Stop")
        print("Q: Quit")
        print("----------------------")

    def send_control_command(self):
        """Send control command to the robot API"""
        try:
            payload = {
                "command": {
                    "linear": self.linear_vel,
                    "angular": self.angular_vel
                }
            }
            
            print(f"Sending control command: {payload}")
            response = requests.post(
                self.control_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=1.0
            )
            
            if response.status_code == 200:
                return True
            else:
                print(f"Error sending command: {response.status_code}")
                return False
        except Exception as e:
            print(f"Exception sending command: {e}")
            return False

    def get_camera_feed(self):
        """Get camera frames from the robot API"""
        try:
            response = requests.get(self.screenshot_url, timeout=1.0)
            
            if response.status_code == 200:
                data = response.json()
                front_frame_b64 = data["front_frame"]
                # rear_frame_b64 = data["rear_frame"]
                timestamp = data["timestamp"]
                
                # Decode front frame
                front_frame_bytes = base64.b64decode(front_frame_b64)
                front_frame_np = np.frombuffer(front_frame_bytes, np.uint8)
                front_frame = cv2.imdecode(front_frame_np, cv2.IMREAD_COLOR)
                
                # Decode rear frame
                # rear_frame_bytes = base64.b64decode(rear_frame_b64)
                # rear_frame_np = np.frombuffer(rear_frame_bytes, np.uint8)
                # rear_frame = cv2.imdecode(rear_frame_np, cv2.IMREAD_COLOR)
                
                # Store timestamp for FPS calculation
                self.update_fps(timestamp)
                
                return front_frame, timestamp
            else:
                print(f"Error fetching camera feed: {response.status_code}")
                return None, None
        except Exception as e:
            print(f"Exception fetching camera feed: {e}")
            return None, None

    def update_fps(self, current_timestamp):
        """Update FPS calculation based on timestamps"""
        # Add current timestamp to the list
        self.frame_timestamps.append(current_timestamp)
        
        # Keep only the last N timestamps
        if len(self.frame_timestamps) > self.max_timestamps:
            self.frame_timestamps = self.frame_timestamps[-self.max_timestamps:]
        
        # Calculate FPS if we have enough timestamps
        if len(self.frame_timestamps) > 1:
            time_diff = self.frame_timestamps[-1] - self.frame_timestamps[0]
            if time_diff > 0:
                self.fps = (len(self.frame_timestamps) - 1) / time_diff
        
        # Update last timestamp
        self.last_timestamp = current_timestamp

    def process_keyboard_input(self, key):
        """Process keyboard input and update control values"""
        # Reset velocities when no key is pressed
        if key == 255:  # No key pressed
            if self.linear_vel != 0 or self.angular_vel != 0:
                self.linear_vel = 0
                self.angular_vel = 0
                self.send_control_command()
            return True
            
        # Process key presses
        if key == ord('w'):  # Forward
            self.linear_vel = 1
        elif key == ord('s'):  # Backward
            self.linear_vel = -1
        elif key == ord('a'):  # Left
            self.angular_vel = 1
        elif key == ord('d'):  # Right
            self.angular_vel = -1
        elif key == ord(' '):  # Space - Stop
            self.linear_vel = 0
            self.angular_vel = 0
        elif key == ord('q'):  # Quit
            # Stop the robot before quitting
            self.linear_vel = 0
            self.angular_vel = 0
            self.send_control_command()
            return False
        
        # Send the control command
        self.send_control_command()
        return True

    def draw_text_with_background(self, image, text, position, color):
        """Draw text with a black background for better visibility"""
        # Get text size
        text_size = cv2.getTextSize(text, self.font, self.font_scale, self.font_thickness)[0]
        
        # Draw background rectangle
        cv2.rectangle(
            image,
            (position[0], position[1] - text_size[1] - 5),
            (position[0] + text_size[0], position[1] + 5),
            (0, 0, 0),  # Black background
            -1  # Filled rectangle
        )
        
        # Draw text
        cv2.putText(
            image,
            text,
            position,
            self.font,
            self.font_scale,
            color,
            self.font_thickness
        )

    def add_info_overlay(self, frame, timestamp):
        """Add information overlay to the frame"""
        # Current time for timestamp diff calculation
        current_time = time.time()
        
        # Check if timestamp is too old
        timestamp_diff = current_time - timestamp
        timestamp_alert = timestamp_diff > self.timestamp_threshold
        
        # Draw FPS
        fps_text = f"FPS: {self.fps:.1f}"
        self.draw_text_with_background(frame, fps_text, (10, 30), self.text_color)
        
        # Draw current control values
        control_text = f"Linear: {self.linear_vel}, Angular: {self.angular_vel}"
        self.draw_text_with_background(frame, control_text, (10, 70), self.text_color)
        
        # Draw timestamp information
        if timestamp_alert:
            time_text = f"WARNING: Timestamp diff: {timestamp_diff:.2f}s"
            self.draw_text_with_background(frame, time_text, (10, 110), self.warning_color)
        else:
            time_text = f"Timestamp diff: {timestamp_diff:.2f}s"
            self.draw_text_with_background(frame, time_text, (10, 110), self.text_color)
            
        return frame

    def create_empty_frame(self, width=1024, height=576):
        """Create an empty frame with an error message"""
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            "No camera feed available",
            (width // 4, height // 2),
            self.font,
            1,
            (0, 0, 255),
            2
        )
        return frame

    def run(self):
        """Main control loop"""
        running = True
        while running:
            # Get camera feed
            front_frame, timestamp = self.get_camera_feed()
            
            if front_frame is not None and timestamp is not None:
                # Add information overlay
                display_frame = self.add_info_overlay(front_frame.copy(), timestamp)
                
                # Show frame
                cv2.imshow(self.window_name, display_frame)
            else:
                # Show empty frame with error message
                empty_frame = self.create_empty_frame()
                cv2.imshow(self.window_name, empty_frame)
            
            # Wait for keyboard input with a small timeout (30ms)
            # This allows us to check for key releases frequently
            key = cv2.waitKey(30) & 0xFF
            
            # Process keyboard input (both presses and no key)
            running = self.process_keyboard_input(key)
        
        # Clean up
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = RobotController()
    controller.run()