import time
import numpy as np
from collections import deque
import threading  # For threading support
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

        # Track the last few sensor values (for penalty calculation)
        self.past_a0 = deque(maxlen=10)  # Past a0 sensor values (10 timesteps)
        self.past_a1 = deque(maxlen=10)  # Past a1 sensor values (10 timesteps)

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

        # Store the latest sensor readings
        self.past_a0.append(a0)
        self.past_a1.append(a1)

        state = np.array([tcp_pose[2], gripper_pos, a0, a1], dtype=np.float32)
        return state

    def step(self, action):
        pos = self.normalize_action(action)
        self.gripper.set_position(pos)
        time.sleep(0.01)  # 20 Hz # 100 Hz
        next_state = self.get_state()
        reward = self.compute_reward(next_state)
        done = False  # For now, we assume infinite horizon
        return next_state, reward, done, {}

    def compute_reward(self, state):
        a0, a1 = state[2], state[3]

        # Reward for making the gripper grasp with adequate force (200 - 300 range)
        gripper_force_reward = self.compute_gripper_force_reward(a0, a1)

        # Reward for sensor value drop (e.g., average derivative over 5 timesteps)
        sensor_drop_penalty = self.compute_sensor_drop_penalty()

        total_reward = gripper_force_reward + sensor_drop_penalty

        return total_reward
    
    def compute_gripper_force_reward(self, a0, a1):
        """Reward the robot for grasping with adequate force (around 200 - 300)."""
        adequate_force_min = 200
        adequate_force_max = 300

        average_force = (a0 + a1) / 2

        # If the force is within the adequate range, give a smaller penalty
        if adequate_force_min <= average_force <= adequate_force_max:
            return 1  # Positive reward for good grasp
        else:
            return -0.5  # Smaller penalty for non-adequate grasp

    def compute_sensor_drop_penalty(self):
        """Compute the penalty if sensor values drop significantly over 5 or 10 timesteps."""
        if len(self.past_a0) < 10 or len(self.past_a1) < 10:
            return 0 
        
        # Calculate the average rate of change over the last 5 timesteps
        a0_change = (self.past_a0[-1] - self.past_a0[0]) / len(self.past_a0)
        a1_change = (self.past_a1[-1] - self.past_a1[0]) / len(self.past_a1)

        # If there's a large drop in either sensor value, give a negative reward
        drop_threshold = -30  # If the sensor drops by more than 50 over 5 timesteps, penalize
        penalty_factor = 0.01  # Scale the penalty

        penalty = 0
        if a0_change < drop_threshold:
            penalty += abs(a0_change) * penalty_factor
        if a1_change < drop_threshold:
            penalty += abs(a1_change) * penalty_factor

        return -penalty  # Negative penalty for significant drops in sensor readings

    def reset(self):
        # Reset robot to initial state
        init_pos = self.normalize_action(-1.0)
        self.gripper.set_position(init_pos)
        time.sleep(1.0)
        return self.get_state()

    def close(self):
        self.gripper.close()
        self.robot.close()

def move_robot_up_at_t80(env, initial_z, final_z):
    """Thread function to move the robot up at t=80."""
    # Move robot up once at t = 80
    z_cmd = initial_z + 0.15
    env.robot.set_tcp_z(z_cmd)  # Command to move the robot


def rollout(agent, length=140, train=False, random=False):
    env = RealWorldRobotEnv()
    initial_state = env.reset()
    env.robot.move_to_initial_pose()  # Move back to initial z
    initial_z = env.robot.get_tcp_pose()[2]
    final_z = initial_z + 0.15

    episode_return = 0
    state = initial_state

    # Flag to check if the robot has started moving
    robot_moving = False

    for t in range(length):
        if t == 80 and not robot_moving:
            # Start robot movement in a separate thread
            move_thread = threading.Thread(target=move_robot_up_at_t80, args=(env, initial_z, final_z))
            move_thread.start()
            robot_moving = True

        # Step 2: Choose action
        if random:
            # action = np.random.uniform(-1, 1)
            # action = np.random.normal(0, 0.3)
            action = np.random.uniform(0.2, 1)
        else:
            action = agent.act(state, train=train)

        # Step 3: Execute and store
        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.append([state, action, [reward], next_state, [not done]])
        episode_return += reward
        state = next_state

        time.sleep(0.04)  # 20Hz # 100 Hz

    time.sleep(5.0)  # Hold on for 5 sec
    # After episode
    _ = env.reset()
    env.robot.move_to_initial_pose()  # Move back to initial z
    time.sleep(1.0)  # Allow user to rearrange
    env.close()

    return episode_return
