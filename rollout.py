import time
import numpy as np
from collections import deque

from robot.gripper import Gripper
from robot.ur import URRobot
from robot.fsr_sensor import FSRSensor


class RealWorldRobotEnv:
    def __init__(self):
        self.gripper = Gripper()
        self.robot = URRobot()
        self.sensor = FSRSensor()

        self.action_range = (0, 550)
        self.state_stack = deque(maxlen=1)  # optional: increase to stack states

    def normalize_action(self, a):
        """Map [-1, 1] to [0, 550]"""
        a = np.clip(a, -1, 1)
        a = ((a + 1) / 2.0) * (self.action_range[1] - self.action_range[0]) + self.action_range[0]
        return int(a)

    def denormalize_action(self, a):
        """Map [0, 550] to [-1, 1]"""
        return ((a - self.action_range[0]) / (self.action_range[1] - self.action_range[0])) * 2 - 1

    def get_state(self):
        tcp_pose = self.robot.get_tcp_pose()
        gripper_pos = self.gripper.get_position()
        a0, a1 = self.sensor.read()
        state = np.array([tcp_pose[2], gripper_pos, a0, a1], dtype=np.float32)
        return state

    def step(self, action):
        pos = self.normalize_action(action)
        self.gripper.set_position(pos)
        time.sleep(0.05)  # 20 Hz
        next_state = self.get_state()
        reward = self.compute_reward(next_state)
        done = False  # For now, we assume infinite horizon
        return next_state, reward, done, {}

    def compute_reward(self, state):
        # Example: reward the robot for minimizing force difference between two sensors
        a0, a1 = state[2], state[3]
        force_balance = -abs(a0 - a1)  # encourage balanced force
        return force_balance

    def reset(self):
        # Reset robot to initial state
        init_pos = self.normalize_action(0.0)
        self.gripper.set_position(init_pos)
        time.sleep(1.0)
        return self.get_state()

    def close(self):
        self.gripper.close()
        self.robot.close()


def rollout(agent, length=80, train=False, random=False):
    env = RealWorldRobotEnv()
    initial_state = env.reset()
    initial_z = env.robot.get_tcp_pose()[2]
    final_z = initial_z + 0.15

    episode_return = 0
    state = initial_state

    for t in range(length):
        # Step 1: Move in Z direction manually
        if t < 40:
            z_cmd = initial_z  # hold for 3s
        else:
            # Linear move over 40 steps
            progress = (t - 40) / 40.0
            z_cmd = initial_z + progress * 0.15

        env.robot.set_tcp_z(z_cmd)  # You need to implement this in URRobot class

        # Step 2: Choose action
        if random:
            action = np.random.uniform(-1, 1)
        else:
            action = agent.act(state, train=train)

        # Step 3: Execute and store
        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.append([state, action, [reward], next_state, [not done]])
        episode_return += reward
        state = next_state

        time.sleep(0.05)  # 20Hz

    # After episode
    env.robot.move_to_initial_pose()  # Move back to initial z
    time.sleep(5.0)  # Allow user to rearrange
    env.close()

    return episode_return

