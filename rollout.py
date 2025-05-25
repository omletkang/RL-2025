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


def rollout(agent, length=200, train=False, random=False):
    env = RealWorldRobotEnv()
    state = env.reset()
    episode_return = 0
    for t in range(length):
        if random:
            action = np.random.uniform(-1, 1)
        else:
            action = agent.act(state, train=train)

        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.append([state, action, [reward], next_state, [not done]])
        episode_return += reward
        state = next_state
        if done:
            break
    env.close()
    return episode_return
