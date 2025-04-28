import random
import math
import pickle
import os
import pygame
import numpy as np
from multiprocessing import Pool
from q_learning_agent import QLearningAgent
from environment import Environment
from robot import DifferentialDriveRobot, RobotPose


def run_episodes(
    episodes,
    max_steps,
    dt,
    max_distance,
    state_bins,
    alpha,
    gamma,
    epsilon,
    action_weights,
    save_interval,
    show_visual,
    speed_multiplier,
    resume,
    resume_path='q_table.pkl',
):
    """
    Core training loop: runs `episodes` episodes, returns trained QLearningAgent.
    If show_visual=True, renders one window. Does NOT save unless save_interval is set.
    If resume=True and resume_path exists, loads existing Q-table before training.
    """
    # Setup environment and agent
    width, height = 1200, 800
    env = Environment(width, height)
    actions = ['left', 'right', 'forward']
    agent = QLearningAgent(actions, state_bins, alpha, gamma, epsilon)

    # Resume from existing Q-table
    if resume and os.path.isfile(resume_path):
        agent.load(resume_path)
        print(f"Loaded Q-table from {resume_path} (resume training)")

    # Normalize action weights if provided
    if action_weights:
        total_w = sum(action_weights)
        action_weights = [w/total_w for w in action_weights]

    # Visualization initialization
    if show_visual:
        pygame.init()
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Q-Learning Training")
        clock = pygame.time.Clock()
        base_fps = 1 / dt
        target_fps = max(1, int(base_fps * speed_multiplier))

    for ep in range(episodes):
        # Spawn robot at random collision-free location
        while True:
            x = random.uniform(0.1*width, 0.9*width)
            y = random.uniform(0.1*height, 0.9*height)
            theta = random.random() * 2*math.pi
            if not env.check_collision(RobotPose(x, y, theta), 20):
                break
        robot = DifferentialDriveRobot(env, x, y, theta)

        # Spawn goal at random non-colliding spot away from robot
        while True:
            gx = random.uniform(0.1*width, 0.9*width)
            gy = random.uniform(0.1*height, 0.9*height)
            if math.hypot(gx - x, gy - y) > 50 and not env.check_collision(RobotPose(gx, gy, 0), 20):
                break

        # Episode loop
        for step in range(max_steps):
            # Read sensors
            robot.sense()
            L = robot.sensorLeft.latest_reading[0]
            F = robot.sensorStraight.latest_reading[0]
            R = robot.sensorRight.latest_reading[0]
            state = agent.discretize([L, F, R], max_distance)

            # ε-greedy with optional action_weights
            if random.random() < agent.epsilon:
                if action_weights:
                    action = random.choices(actions, weights=action_weights, k=1)[0]
                else:
                    action = random.choice(actions)
            else:
                qs = agent._get_qs(state)
                max_q = np.max(qs)
                candidates = [a for a, q in zip(actions, qs) if q == max_q]
                action = random.choice(candidates)

            # Execute action
            if action == 'left':
                robot.theta = (robot.theta + math.pi/2) % (2*math.pi)
            elif action == 'right':
                robot.theta = (robot.theta - math.pi/2) % (2*math.pi)
            else:  # forward
                robot.left_motor_speed = 1
                robot.right_motor_speed = 1
                robot._step_kinematics(dt)

            # Compute reward
            robot.sense()
            collided = env.check_collision(robot.get_robot_pose(), robot.get_robot_radius())
            dist_goal = math.hypot(robot.x - gx, robot.y - gy)
            if collided:
                reward, done = -100, True
            elif dist_goal < robot.get_robot_radius():
                reward, done = +100, True
            else:
                reward, done = -1, False

            # Learning update
            robot.sense()
            L2 = robot.sensorLeft.latest_reading[0]
            F2 = robot.sensorStraight.latest_reading[0]
            R2 = robot.sensorRight.latest_reading[0]
            next_state = agent.discretize([L2, F2, R2], max_distance)
            agent.learn(state, action, reward, next_state)

            # Visualization (single process)
            if show_visual:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        show_visual = False
                screen.fill((0, 0, 0))
                env.draw(screen)
                robot.draw(screen)
                pygame.draw.circle(screen, (255, 0, 0), (int(gx), int(gy)), int(robot.get_robot_radius()))
                pygame.display.flip()
                clock.tick(target_fps)

            if done:
                break

        # Epsilon decay
        if ep and ep % 100 == 0:
            agent.epsilon *= 0.99

        # Periodic save (overwrite same file)
        if save_interval and (ep + 1) % save_interval == 0:
            agent.save('q_table.pkl')
            print(f"Episode {ep+1}: Q-table overwritten to q_table.pkl")

    if show_visual:
        pygame.quit()
    return agent


def merge_q_tables(q_tables):
    """
    Average multiple Q-tables (list of dicts state->np.array), ignoring unexplored values (1.0).
    """
    merged = {}
    counts = {}

    for qt in q_tables:
        for state, qs in qt.items():
            if state not in merged:
                merged[state] = np.zeros_like(qs, dtype=float)
                counts[state] = np.zeros_like(qs, dtype=int)
            for i, q in enumerate(qs):
                if q != 1.0:  # Skip unexplored values
                    merged[state][i] += q
                    counts[state][i] += 1

    for state in merged:
        for i in range(len(merged[state])):
            if counts[state][i] > 0:  # Avoid division by zero
                merged[state][i] /= counts[state][i]
            else:
                merged[state][i] = 1.0  # Retain unexplored value

    return merged


def train(
    episodes=1000,
    workers=1,
    max_steps=200,
    dt=0.5,
    max_distance=100,
    state_bins=[5,5,5],
    alpha=0.1,
    gamma=0.9,
    epsilon=0.1,
    action_weights=None,
    save_interval=None,
    show_visual=False,
    speed_multiplier=1,
    resume=False,
    resume_path='q_table.pkl'
):
    """
    Entry point: trains either single-thread (workers=1) or parallel (workers>1).

    If show_visual=True, forces workers=1 to render only one window.
    resume=True loads existing Q-table from resume_path before training.
    """
    if show_visual and workers > 1:
        print("Visualization requires a single worker; setting workers=1.")
        workers = 1

    if workers < 2:
        agent = run_episodes(
            episodes,
            max_steps,
            dt,
            max_distance,
            state_bins,
            alpha,
            gamma,
            epsilon,
            action_weights,
            save_interval,
            show_visual,
            speed_multiplier,
            resume,
            resume_path,
        )
        agent.save('q_table.pkl')
        print(f"Training done. Q table saved.")
        return

    # Parallel execution: visualization off, no resume per worker
    per_worker = episodes // workers
    params = []
    for _ in range(workers):
        params.append(
            (
                per_worker,
                max_steps,
                dt,
                max_distance,
                state_bins,
                alpha,
                gamma,
                epsilon,
                action_weights,
                None,       # child save_interval disabled
                False,      # no visual in workers
                speed_multiplier,
                False,      # no resume in workers
                resume_path
            )
        )
    with Pool(workers) as pool:
        agents = pool.starmap(run_episodes, params)

    q_tables = [agent.q_table for agent in agents]
    merged = merge_q_tables(q_tables)
    with open(resume_path, 'wb') as f:
        pickle.dump(merged, f)
    print(f"Parallel training done ({workers} workers). Saved Q table.pkl")

if __name__ == '__main__':
    # Example usage:
    train(
        episodes=100,
        workers=12,
        max_steps=500,
        dt=0.5,
        max_distance=100,
        state_bins=[5,5,5],
        alpha=0.1,
        gamma=0.95,
        epsilon=0.1,
        action_weights=[0.1,0.1,0.8],
        save_interval=None,
        show_visual=True,
        speed_multiplier=5000,
        resume=True,
        resume_path='q_table.pkl'
    )
