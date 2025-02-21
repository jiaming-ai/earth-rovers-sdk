class RobotController {
    constructor() {
        // Track pressed keys
        this.pressedKeys = new Set();
        // Control values
        this.linear = 0;
        this.angular = 0;
        // Control loop interval (100ms = 10Hz)
        this.controlInterval = null;
        
        // Bind event listeners
        this.setupKeyboardListeners();
        
        // Constants for control values
        this.FORWARD_SPEED = 1;
        this.BACKWARD_SPEED = -1;
        this.TURN_LEFT_SPEED = 1;
        this.TURN_RIGHT_SPEED = -1;
    }

    setupKeyboardListeners() {
        // Add keydown listener
        document.addEventListener('keydown', (event) => {
            // Prevent default behavior and stop propagation for WASD keys
            if (['w', 'a', 's', 'd'].includes(event.key.toLowerCase())) {
                event.preventDefault();
                event.stopPropagation();
            }
            
            // Only handle if key wasn't already pressed
            if (!this.pressedKeys.has(event.key)) {
                this.pressedKeys.add(event.key.toLowerCase());
                this.updateControlValues();
                
                // Start control loop if it's not running
                if (!this.controlInterval) {
                    this.startControlLoop();
                }
            }
        });

        // Add keyup listener
        document.addEventListener('keyup', (event) => {
            // Prevent default behavior and stop propagation for WASD keys
            if (['w', 'a', 's', 'd'].includes(event.key.toLowerCase())) {
                event.preventDefault();
                event.stopPropagation();
            }
            
            this.pressedKeys.delete(event.key.toLowerCase());
            this.updateControlValues();
            
            // If no keys are pressed, stop the control loop
            if (this.pressedKeys.size === 0) {
                this.stopControlLoop();
            }
        });
    }

    updateControlValues() {
        // Reset values
        this.linear = 0;
        this.angular = 0;

        // Update based on pressed keys
        if (this.pressedKeys.has('w')) this.linear = this.FORWARD_SPEED;
        if (this.pressedKeys.has('s')) this.linear = this.BACKWARD_SPEED;
        if (this.pressedKeys.has('a')) this.angular = this.TURN_LEFT_SPEED;
        if (this.pressedKeys.has('d')) this.angular = this.TURN_RIGHT_SPEED;
    }

    startControlLoop() {
        this.controlInterval = setInterval(() => {
            this.sendControlCommand();
        }, 100); // 100ms = 10Hz
    }

    stopControlLoop() {
        if (this.controlInterval) {
            clearInterval(this.controlInterval);
            this.controlInterval = null;
            // Send a stop command
            this.sendControlCommand();
        }
    }

    async sendControlCommand() {
        try {
            const response = await fetch('http://localhost:8000/control', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    command: {
                        linear: this.linear,
                        angular: this.angular
                    }
                })
            });

            if (!response.ok) {
                console.error('Failed to send control command:', response.statusText);
            }
        } catch (error) {
            console.error('Error sending control command:', error);
        }
    }
}

// Initialize the controller when the page loads
document.addEventListener('DOMContentLoaded', () => {
    const robotController = new RobotController();
}); 