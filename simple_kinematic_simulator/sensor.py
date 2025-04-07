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

    def get_distance(self):
        return self.latest_reading[0] if self.latest_reading is not None else None

class LidarSensor:
    def __init__(self, max_distance_cm=100):  # 8m = 800cm
        self.max_distance_cm = max_distance_cm
        self.angles = [i * pi/180 for i in range(360)]  # 360 measurements (1° intervals)
        self.measurements = [(max_distance_cm, None, None)] * 360  # (distance, color, point)

    def generate_beam_and_measure(self, robot_pose, obstacles):
        for i, angle in enumerate(self.angles):
            # Calculate absolute angle in world frame
            ray_angle = (robot_pose.theta + angle) % (2 * pi)
            
            # Calculate ray endpoint
            x2 = robot_pose.x + self.max_distance_cm * cos(ray_angle)
            y2 = robot_pose.y + self.max_distance_cm * sin(ray_angle)
            ray = LineString([(robot_pose.x, robot_pose.y), (x2, y2)])
            
            # Check intersections
            intersection = self._check_intersections(ray, obstacles)
            
            if intersection:
                point, color = intersection
                distance = Point(robot_pose.x, robot_pose.y).distance(point)
                self.measurements[i] = (distance, color, point)
            else:
                self.measurements[i] = (self.max_distance_cm, None, Point(x2, y2))

    def _check_intersections(self, ray, obstacles):
        intersection_points = [(ray.intersection(obstacle), color) 
                             for obstacle, color in obstacles]
        valid_intersections = [(point, color) 
                             for point, color in intersection_points 
                             if not point.is_empty and isinstance(point, Point)]
        
        if valid_intersections:
            return min(valid_intersections, key=lambda pc: ray.project(pc[0]))
        return None

    def draw(self, robot_pose, screen):
        for measurement in self.measurements:
            distance, color, point = measurement
            if distance < self.max_distance_cm:  # Only draw if there's an obstacle detected
                beam_color = (255, 0, 0)  # Red for detected obstacles
            else:
                beam_color = (0, 255, 0)  # Green for max range
            pygame.draw.line(screen, beam_color, 
                           (robot_pose.x, robot_pose.y), 
                           (point.x, point.y), 1)

    def get_distances(self):
        return [m[0] for m in self.measurements]