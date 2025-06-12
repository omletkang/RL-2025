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
        self.past_a0 = deque(maxlen=5)  # Past a0 sensor values (10 timesteps)
        self.past_a1 = deque(maxlen=5)  # Past a1 sensor values (10 timesteps)

        # # time penalty
        self.max_steps = 140
        self.timestep = 0

        self.reward_scale = 0.007
        # self.step_count = 0
        # self.time_limit = 140 

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

        # state = np.array([tcp_pose[2], gripper_pos, a0, a1], dtype=np.float32)
        
        # time penalty
        timestep_norm = self.timestep / self.max_steps
        state = np.array([tcp_pose[2], gripper_pos, a0, a1, timestep_norm], dtype=np.float32)
        # print(f"[{time.time():.3f}], 5")
        return state

    def step(self, action):
        pos = self.normalize_action(action)
        self.gripper.set_position(pos)
        # time.sleep(0.01)  # 20 Hz # 100 Hz
        next_state = self.get_state()
        # print(next_state)
        f_reward, d_reward, t_reward = self.compute_reward(next_state)
        done = False  # For now, we assume infinite horizon
        ## time penalty
        self.timestep += 1
        return next_state, t_reward, done, {}, f_reward, d_reward

    def compute_reward(self, state):
        a0, a1 = state[2], state[3]

        # Reward for making the gripper grasp with adequate force (200 - 300 range)
        gripper_force_reward = self.compute_gripper_force_reward(a0, a1)

        # Reward for sensor value drop (e.g., average derivative over 5 timesteps)
        sensor_drop_penalty = self.compute_sensor_drop_penalty()

        total_reward = gripper_force_reward + sensor_drop_penalty
        
        # # time penalty
        # time_penalty = self.compute_time_penalty(avg_force)
        # total_reward = gripper_force_reward + sensor_drop_penalty + time_penalty

        return gripper_force_reward, sensor_drop_penalty, total_reward
    
    def compute_gripper_force_reward(self, a0, a1):
        """Reward the robot for maintaining 5% deformation after first contact."""
        force_threshold = 50
        desired_ratio = 0.01
        full_deformation_max = 735

        gripper_pos = self.gripper.get_position()
        average_force = (a0 + a1) / 2

        # print("average force : %f" % average_force)
        # Step 1: Detect initial contact
        if average_force >= force_threshold and not hasattr(self, 'init_contact'):
            self.init_contact = gripper_pos  # Set init_contact once
            print('Contact is triggered!')
            print("pos: %f" % gripper_pos)
        
        # Step 2: If contact hasn't been made yet
        if not hasattr(self, 'init_contact'):
            return -30* self.reward_scale  # 0.2  # Small penalty to encourage exploration
        

        ### 여기 추가했음 (재현) 250609 11PM
        if average_force < force_threshold:
        # Small penalty for losing contact after initial contact
            return -15 * self.reward_scale

        # Step 3: Compute ratio
        deformation_range = full_deformation_max - self.init_contact
        if deformation_range <= 0:
            return -1* self.reward_scale   # Avoid division by zero or inverted contact

        current_ratio = (gripper_pos - self.init_contact) / deformation_range

        # Step 4: Reward is highest when current_ratio ≈ 0.05
        error = abs(desired_ratio-current_ratio)
        reward = -error * 5  # penalize deviation

        # Optional: clip to avoid too negative reward
        return reward * self.reward_scale # np.clip(reward, -1.0, 1.0)


    def compute_sensor_drop_penalty(self):
        """Compute the penalty if sensor values drop significantly over 5 or 10 timesteps."""
        if len(self.past_a0) < 5 or len(self.past_a1) < 5:
            return 0 
        
        # Calculate the average rate of change over the last 5 timesteps
        a0_change = (self.past_a0[-1] - self.past_a0[0]) / len(self.past_a0)
        a1_change = (self.past_a1[-1] - self.past_a1[0]) / len(self.past_a1)

        # If there's a large drop in either sensor value, give a negative reward
        drop_threshold = -6 # 30 # If the sensor drops by more than 50 over 5 timesteps, penalize
        penalty_factor = 1.2 # 0.01  # Scale the penalty

        penalty = 0
        if a0_change < drop_threshold:
            penalty += abs(a0_change) * penalty_factor
        if a1_change < drop_threshold:
            penalty += abs(a1_change) * penalty_factor

        return -penalty * self.reward_scale  # Negative penalty for significant drops in sensor readings

    # # time penalty
    # def compute_time_penalty(self, avg_force):
    #     late_start   = 50        # Check after 50 steps
    #     force_threshold  = 50
    #     time_penalty = -50.0
    
    #     if self.step_count >= late_start and avg_force < force_floor:
    #         return time_penalty
    #     return 0.0

    
    def reset(self):
        # Reset robot to initial state
        init_pos = self.normalize_action(-1.0)
        self.gripper.set_position(init_pos)
        
        # time penalty
        self.timestep = 0
        
        if hasattr(self, 'init_contact'):
            del self.init_contact  # Forget contact info every episode
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
    f_return = 0
    d_return = 0
    state = initial_state

    history_dict = {"f_reward": [],
                    "d_reward": [],
                    "state": [],
                    }

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
            action = np.random.uniform(0.2, 0.8)
        else:
            action = agent.act(state, train=train)

        print(f"[{time.time():.3f}], {t}")

        # Step 3: Execute and store
        next_state, reward, done, _, f_reward, d_reward = env.step(action) # Additional reward

        # Record
        history_dict["f_reward"].append(f_reward)
        history_dict["d_reward"].append(d_reward)
        history_dict["state"].append(state.tolist())


        print(f"[{time.time():.3f}], {t}")

        # ---- THIS IS THE IMPORTANT PART ----
        if isinstance(action, np.ndarray):
            action_to_store = action.astype(np.float32).flatten()
        else:
            action_to_store = np.array([float(action)], dtype=np.float32)
        agent.replay_buffer.append([state, action_to_store, float(reward), next_state, float(not done)]) # action
        episode_return += reward
        f_return += f_reward
        d_return += d_reward
        state = next_state
        

        # time.sleep(0.04)  # 20Hz # 100 Hz

    time.sleep(5.0)  # Hold on for 5 sec
    # After episode
    print(f"return: {episode_return:.2f} || force return : {f_return:.2f} || drop_return : {d_return:.2f}")
    _ = env.reset()
    env.robot.move_to_initial_pose()  # Move back to initial z
    time.sleep(1.0)  # Allow user to rearrange
    env.close()

    return episode_return, history_dict
