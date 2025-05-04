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
    Tracks successes and failures and includes a "closer/further" feature in the state.
    """
    # Setup environment and agent
    width, height = 1200, 800
    env = Environment(width, height)
    actions = ['left', 'right', 'forward']
    agent = QLearningAgent(actions, state_bins, alpha, gamma, epsilon)

    # Resume from existing Q-table if requested
    if resume and os.path.isfile(resume_path):
        agent.load(resume_path)
        print(f"Loaded Q-table from {resume_path} (resuming)")

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

    # Stats counters
    success_count = 0
    failure_count = 0

    for ep in range(episodes):
        # Spawn robot at a random, truly collision-free location
        while True:
            x     = random.uniform(0.1*width, 0.9*width)
            y     = random.uniform(0.1*height, 0.9*height)
            theta = random.random() * 2*math.pi
            tmp_robot = DifferentialDriveRobot(env, x, y, theta)
            if not env.check_collision(tmp_robot.get_robot_pose(), tmp_robot.get_robot_radius()):
                robot = tmp_robot
                break

        # Spawn goal at a random, collision-free location away from robot
        while True:
            gx = random.uniform(0.1*width, 0.9*width)
            gy = random.uniform(0.1*height, 0.9*height)
            if (math.hypot(gx - robot.x, gy - robot.y) > 50
                and not env.check_collision(RobotPose(gx, gy, 0), robot.get_robot_radius())):
                break

        # Set initial previous distance for the "closer/further" feature
        prev_dist = math.hypot(robot.x - gx, robot.y - gy)

        for step in range(max_steps):
            # Record previous position (for rotation penalty)
            prev_x, prev_y = robot.x, robot.y

            # Sense
            robot.sense()
            L = robot.sensorLeft.latest_reading[0]
            F = robot.sensorStraight.latest_reading[0]
            R = robot.sensorRight.latest_reading[0]

            # Compute current distance and closeness feature
            curr_dist = math.hypot(robot.x - gx, robot.y - gy)
            if curr_dist < prev_dist:
                closeness = 1
            elif curr_dist > prev_dist:
                closeness = -1
            else:
                closeness = 0

            # Form state: sensor buckets + closeness feature
            sens_state = agent.discretize([L, F, R], max_distance)
            state = sens_state + (closeness,)

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

            # Execute
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
            new_dist = math.hypot(robot.x - gx, robot.y - gy)
            if collided:
                reward, done = -100, True
            elif new_dist < robot.get_robot_radius():
                reward, done = +10000, True
            else:
                reward, done = -1, False

            # Penalty for in-place rotation
            if not done and (robot.x == prev_x and robot.y == prev_y):
                reward -= 1

            # Prepare next_state with updated closeness
            if new_dist < curr_dist:
                next_closeness = 1
                reward += 1  # Encourage moving closer
            elif new_dist > curr_dist:
                next_closeness = -1
            else:
                next_closeness = 0
            L2 = robot.sensorLeft.latest_reading[0]
            F2 = robot.sensorStraight.latest_reading[0]
            R2 = robot.sensorRight.latest_reading[0]
            sens_next = agent.discretize([L2, F2, R2], max_distance)
            next_state = sens_next + (next_closeness,)

            # Learn
            agent.learn(state, action, reward, next_state)

            # Update prev_dist
            prev_dist = curr_dist

            # Visualization
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

        # Record episode outcome
        if done and reward > 0:
            success_count += 1
        else:
            failure_count += 1

        # Epsilon decay
        if ep and ep % 100 == 0:
            agent.epsilon *= 0.99

        # Periodic save
        if save_interval and (ep + 1) % save_interval == 0:
            agent.save(resume_path)
            print(f"Episode {ep+1}: Q-table overwritten to {resume_path}")

    # Stats summary
    total = success_count + failure_count
    success_rate = (success_count / total * 100) if total else 0.0
    print(f"Training completed: {success_count} successes, {failure_count} failures ({success_rate:.2f}% success rate)")

    if show_visual:
        pygame.quit()
    return agent


def merge_q_tables(q_tables):
    """
    Average multiple Q-tables, ignoring unexplored (Q=1.0).
    """
    merged = {}
    counts = {}
    for qt in q_tables:
        for state, qs in qt.items():
            if state not in merged:
                merged[state] = np.zeros_like(qs, dtype=float)
                counts[state] = np.zeros_like(qs, dtype=int)
            for i, q in enumerate(qs):
                if q != 1.0:
                    merged[state][i] += q
                    counts[state][i] += 1
    for state in merged:
        for i in range(len(merged[state])):
            merged[state][i] = merged[state][i] / counts[state][i] if counts[state][i] > 0 else 1.0
    return merged


def train(
    episodes=1000,
    workers=1,
    max_steps=200,
    dt=0.5,
    max_distance=100,
    state_bins=[5,5,5,3],
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
    Entry point: runs single or parallel training.
    Prints overall success rate for single-worker mode.
    """

    pygame.mixer.init()
    beep_sound = pygame.mixer.Sound('beep.mp3')

    # Ensure visualization uses single worker
    if show_visual and workers > 1:
        print("Visualization requires a single worker; forcing workers=1.")
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
        agent.save(resume_path)
        print(f"Final Q-table saved to {resume_path}")
        return

    # Parallel execution
    per_worker = episodes // workers
    params = []
    for _ in range(workers):
        params.append((
            per_worker,
            max_steps,
            dt,
            max_distance,
            state_bins,
            alpha,
            gamma,
            epsilon,
            action_weights,
            None,      # no periodic save in workers
            False,     # no visual
            speed_multiplier,
            resume,
            resume_path
        ))
    with Pool(workers) as pool:
        agents = pool.starmap(run_episodes, params)

    # Merge Q-tables
    q_tables = [agent.q_table for agent in agents]
    merged = merge_q_tables(q_tables)
    with open(resume_path, 'wb') as f:
        pickle.dump(merged, f)
    print(f"Parallel training done ({workers} workers). Q-table saved to {resume_path}")
    beep_sound.play()
    pygame.time.delay(int(beep_sound.get_length() * 1000))

if __name__ == '__main__':
    train(
        episodes=1000,
        workers=12,
        max_steps=1500,
        dt=0.5,
        max_distance=100,
        state_bins=[5,5,5,3],  # sensor bins + closeness bins
        alpha=0.2,
        gamma=0.95,
        epsilon=0.0,
        action_weights=[0.1,0.1,0.8],
        save_interval=None,
        show_visual=False,
        speed_multiplier=1000,
        resume=True,
        resume_path='q_table_closeness.pkl'
    )
