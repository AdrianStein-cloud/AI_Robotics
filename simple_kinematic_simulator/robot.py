import pygame
from numpy import cos, sin, pi

from sensor import SingleRayDistanceAndColorSensor

class DifferentialDriveRobot:
    def __init__(self, env, x, y, theta, axel_length=40, wheel_radius=10, max_motor_speed=2*pi, kinematic_timestep=0.01):
        self.env = env
        self.x = x
        self.y = y
        self.theta = theta  # Orientation in radians
        self.axel_length = axel_length # in cm
        self.wheel_radius = wheel_radius # in cm

        self.kinematic_timestep = kinematic_timestep

        self.collided = False

        self.left_motor_speed  = 0 #rad/s
        self.right_motor_speed = 0 #rad/s
        #self.theta_noise_level = 0.01

        self.sensor = SingleRayDistanceAndColorSensor(100, 0)

    def move(self, robot_timestep): # run the control algorithm here
        # simulate kinematics during one execution cycle of the robot
        self._step_kinematics(robot_timestep)

        # check for collision
        self.collided = self.env.check_collision(self.get_robot_pose(), self.get_robot_radius())

        # update sensors
        self.sense()

        # run the control algorithm and update motor speeds
        # ...



    def _step_kinematics(self, robot_timestep):
        for _ in range(int(robot_timestep / self.kinematic_timestep)): # the kinematic model is updated in every step for robot_timestep/self.kinematic_timestep times
            # odometry is used to calculate where we approximately end up after each step
            pos = self._odometer(self.kinematic_timestep)
            self.x = pos.x
            self.y = pos.y
            self.theta = pos.theta
            # Add a small amount of noise to the orientation and/or position
            # noise = random.gauss(0, self.theta_noise_level)
            # self.theta += noise

    def sense(self):
        obstacles = self.env.get_obstacles()
        robot_pose = self.get_robot_pose()
        self.sensor.generate_beam_and_measure(robot_pose, obstacles)

    # this is in fact what a robot can predict about its own future position
    def _odometer(self, delta_time):
        left_angular_velocity = self.left_motor_speed
        right_angular_velocity = self.right_motor_speed

        v_x = cos(self.theta) * (self.wheel_radius * (left_angular_velocity + right_angular_velocity) / 2)
        v_y = sin(self.theta) * (self.wheel_radius * (left_angular_velocity + right_angular_velocity) / 2)
        omega = (self.wheel_radius * (left_angular_velocity - right_angular_velocity)) / self.axel_length

        next_x = self.x + (v_x * delta_time)
        next_y = self.y + (v_y * delta_time)
        next_theta = self.theta + (omega * delta_time)

        # Ensure the orientation stays within the range [0, 2*pi)
        next_theta = next_theta % (2 * pi)

        return RobotPose(next_x, next_y, next_theta)


    def get_robot_pose(self):
        return RobotPose(self.x, self.y, self.theta)

    def get_robot_radius(self):
        return self.axel_length/2

    def draw(self, surface):
        pygame.draw.circle(surface, (0,255,0), center=(self.x, self.y), radius=self.axel_length/2, width = 1)

        # Calculate the left and right wheel positions
        half_axl = self.axel_length/2
        left_wheel_x = self.x - half_axl * sin(self.theta)
        left_wheel_y = self.y + half_axl * cos(self.theta)
        right_wheel_x = self.x + half_axl * sin(self.theta)
        right_wheel_y = self.y - half_axl * cos(self.theta)

        # Calculate the heading line end point
        heading_length = half_axl + 2
        heading_x = self.x + heading_length * cos(self.theta)
        heading_y = self.y + heading_length * sin(self.theta)

        # Draw the axle line
        pygame.draw.line(surface, (0, 255, 0), (left_wheel_x, left_wheel_y), (right_wheel_x, right_wheel_y), 3)

        # Draw the heading line
        pygame.draw.line(surface, (255, 0, 0), (self.x, self.y), (heading_x, heading_y), 5)

        # Draw sensor beams
        self.sensor.draw(self.get_robot_pose(),surface)


class RobotPose:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    # this is for pretty printing
    def __repr__(self) -> str:
        return f"x:{self.x},y:{self.y},theta:{self.theta}"
