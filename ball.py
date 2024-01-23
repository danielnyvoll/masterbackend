from agent import Agent
from math import sin, cos, fabs, acos, pi, inf
class Ball(Agent):
    """
    Represents a ball.
    """
    def __init__(self, pose, max_linear_speed, max_angular_speed, radius, behavior):
        Agent.__init__(self, pose, max_linear_speed, max_angular_speed, radius)
        self.behavior = behavior
        self.cont_friction = 0

    def set_rotation(self, increase):
            self.pose.rotation += increase
    
    def set_cont_friction(self, initial, increase):
        if initial:
            self.cont_friction = 0
        else: 
            self.cont_friction += increase

    def update(self):
        self.behavior.update(self)
        self.move()

    def move(self):
        # Movement logic based on the ball's velocity and direction
        dt = 1/60
        damping_factor = 0.98  # Adjust this value as needed
        self.linear_speed *= damping_factor

        # Increment friction counter
        self.set_cont_friction(initial=False, increase=dt)

        # Update position based on speed and direction
        self.pose.position.x += self.linear_speed * cos(self.pose.rotation) * dt
        self.pose.position.y += self.linear_speed * sin(self.pose.rotation) * dt

        # Stop the ball if its speed is below a certain threshold to prevent endless slow movement
        if abs(self.linear_speed) < 0.01:
            self.linear_speed = 0.0