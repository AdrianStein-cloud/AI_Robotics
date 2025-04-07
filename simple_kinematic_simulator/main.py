import pygame
from pygame.locals import QUIT, KEYDOWN
from environment import Environment
from robot import DifferentialDriveRobot

#for potential visualization
USE_VISUALIZATION = True

# to pause the execution
PAUSE = False

# Initialize Pygame
pygame.init()

# Set up environment
width, height = 1200, 800 # cm
env = Environment(width, height)

# (simulated) time taken for one cycle of the robot executing its algorithm
robot_timestep = 0.1 # in seconds (simulated time)

robot = DifferentialDriveRobot(env,width/2-100,height/2-100,0)

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Robot Kinematic Simulator")

def main():
    global USE_VISUALIZATION, PAUSE
    start_time = pygame.time.get_ticks()
    # Game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            if event.type == KEYDOWN:
                if event.key == pygame.K_h: # use space key to toggle between visualization and headless
                    USE_VISUALIZATION = not USE_VISUALIZATION
                    print("Visualization is", "on" if USE_VISUALIZATION else "off")
                if event.key == pygame.K_SPACE:
                    PAUSE = not PAUSE

        if not PAUSE:
            # simulate one execution cycle of the robot
            robot_pose = robot.move(robot_timestep)


        if USE_VISUALIZATION:
            screen.fill((0, 0, 0))
            # draw environment
            env.draw(screen)
            # draw robot
            robot.draw(screen)

            # warn the user if collision happened
            if robot.collided:
                print("Collision!")
                # Draw the animation
                drawBoom()

            pygame.display.flip()
            pygame.display.update()


    print("total execution time:", (pygame.time.get_ticks() - start_time) / 1000, "seconds")  # runtime in seconds

    # Quit Pygame
    pygame.quit()

def drawBoom():
    font = pygame.font.SysFont("comicsansms", 172)  # pygame.font.Font(self.font, size)
    text_surface = font.render('BOOM', True, (255, 0, 0))
    text_rect = text_surface.get_rect(center=(width/2, height/2))
    screen.blit(text_surface, text_rect)

if __name__ == "__main__":
    main()
  

