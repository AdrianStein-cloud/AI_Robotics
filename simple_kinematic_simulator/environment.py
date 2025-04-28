
import pygame
from shapely.geometry import LineString, Polygon, Point


class Environment:
    def __init__(self, width, height) -> None:
        self.width = width
        self.height = height
        self.obstacle_walls = []
        self.goal_wall = None    # will hold the goal boundary if set
        self.create_floorplan()

    def get_dimensions(self):
        return (self.width, self.height)

    def create_floorplan(self):
        # Create LineString objects for the walls
        left_wall = (LineString([(0, 0), (0, self.height)]), (255, 0, 255))  # purple
        bottom_wall = (LineString([(0, self.height), (self.width, self.height)]), (255, 0, 255))
        right_wall = (LineString([(self.width, self.height), (self.width, 0)]), (255, 0, 255))
        top_wall = (LineString([(self.width, 0), (0, 0)]), (255, 0, 255))

        # kitchen
        kitchen_wall1 = (LineString([(300, 0), (300, 150)]), (255, 0, 255))
        kitchen_wall2 = (LineString([(0, 500), (300, 500)]), (255, 0, 255))
        kitchen_wall3 = (LineString([(300, 400), (300, 500)]), (255, 0, 255))

        # room
        room_right_wall2 = (LineString([(800, 250), (800, 800)]), (255, 0, 255))

        # obstacle box
        box_1 = (LineString([(100, 380), (150, 380)]), (255, 255, 0))  # yellow
        box_2 = (LineString([(100, 380), (100, 420)]), (255, 255, 0))
        box_3 = (LineString([(100, 420), (150, 420)]), (255, 255, 0))
        box_4 = (LineString([(150, 380), (150, 420)]), (255, 255, 0))

        self.obstacle_walls = [
            left_wall, bottom_wall, right_wall, top_wall,
            kitchen_wall1, kitchen_wall2, kitchen_wall3,
            room_right_wall2,
            box_1, box_2, box_3, box_4
        ]

    def set_goal(self, x, y, radius, color=(255, 0, 0)):
        """
        Define the goal as a circular boundary that sensors can detect.
        x, y: center of goal
        radius: radius in cm
        color: RGB tuple used to mark goal
        """
        # Create a circular polygon and extract its outer boundary
        circle = Point(x, y).buffer(radius, resolution=16)
        boundary = circle.boundary
        self.goal_wall = (boundary, color)

    def get_obstacles(self):
        """
        Return all obstacles plus the goal boundary (if set),
        so that sensors will detect the goal as a red “wall.”
        """
        obs = list(self.obstacle_walls)
        if self.goal_wall:
            obs.append(self.goal_wall)
        return obs

    def check_collision(self, pos, radius):
        # collision only with true obstacles (not the goal boundary)
        for line, color in self.obstacle_walls:
            if line.distance(Point(pos.x, pos.y)) <= radius:
                return True
        return False

    def draw(self, screen):
        # Draw the walls
        for wall, color in self.obstacle_walls:
            pygame.draw.line(
                screen, color,
                (int(wall.xy[0][0]), int(wall.xy[1][0])),
                (int(wall.xy[0][1]), int(wall.xy[1][1])),
                4
            )
        # Draw goal if present
        if self.goal_wall:
            boundary, color = self.goal_wall
            coords = list(boundary.coords)
            pygame.draw.polygon(screen, color, coords, width=0)