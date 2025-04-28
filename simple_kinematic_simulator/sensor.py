from shapely.geometry import Point, LineString
from numpy import cos, sin, pi
import pygame

class SingleRayDistanceAndColorSensor:
    def __init__(self, max_distance_cm, angle_rad):
        self.max_distance_cm = max_distance_cm
        # angle of the sensor relative to the robot's heading
        self.angle = angle_rad
        # the latest sensory inputs
        self.latest_reading = None

    def generate_beam_and_measure(self, robot_pose, obstacles):
        x = robot_pose.x
        y = robot_pose.y

        # Calculate the angle of the beam (in the worlds frame of reference)
        ray_angle = robot_pose.theta + self.angle

        # Ensure the ray angle is within the valid range (0 to 2 pi radians)
        ray_angle %= (2 * pi)

        # Calculate the end point of the beam
        x2, y2 = (
        x + self.max_distance_cm * cos(ray_angle), y + self.max_distance_cm * sin(ray_angle))

        end_point = Point(x2, y2)

        # Create a LineString representing the ray
        ray = LineString([(x, y), end_point])

        # Check for intersection with obstacles
        intersection = self._check_intersections(ray, obstacles)

        # Calculate distance based on intersection or return max distance
        if intersection:
            point, color = intersection
            distance = Point(x, y).distance(point)
            intersect_point = point
        else:
            distance = self.max_distance_cm
            color = None
            intersect_point = end_point

        self.latest_reading = (distance, color, intersect_point)

    def _check_intersections(self, ray, obstacles):
        """
        Check for intersections between the beam and obstacles.
        Parameters:
            - beam (LineString): LineString representing the beam.
            - obstacles (list of LineString and their color): List of LineString objects representing obstacle walls.
        Returns:
            - Point or None: The closest intersection point if there is one, otherwise None.
        """
        intersection_points = [(ray.intersection(obstacle),color) for obstacle,color in obstacles]
        # Filter valid points and ensure they are of type Point
        valid_intersections = [(point,color) for point,color in intersection_points if
                               not point.is_empty and isinstance(point, Point)]
        if valid_intersections:
            # find the closest intersection point along the ray from its starting point
            closest_intersection = min(valid_intersections, key=lambda pc: ray.project(pc[0]))
            return closest_intersection
        else:
            return None

    def draw(self, robot_pose, screen):
        x = robot_pose.x
        y = robot_pose.y
        if self.latest_reading is not None:
            distance, color, intersect_point = self.latest_reading
            pygame.draw.line(screen, (255, 255, 0), (x, y), (intersect_point.x, intersect_point.y), 1)
