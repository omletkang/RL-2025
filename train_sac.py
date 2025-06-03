import os
import argparse
import pickle
import glob
import math
import numpy as np
from tqdm import tqdm 
from datetime import datetime

import torch
from sac import SAC
from rollout import rollout, RealWorldRobotEnv


def main():
    parser = argparse.ArgumentParser(description='Train SAC on Real Robot')
    parser.add_argument('--n_episodes', default=20, type=int)
    parser.add_argument('--resume', default='', type=str)
    args = parser.parse_args()

    obs_size = 5  # TCP Z, Gripper Pos, FSR A0, A1
    act_size = 1  # Gripper action

    EPISODES_PER_TRAINING = 10  # Collect 10 episodes before each training phase

    epsilon_start = 1.0 # Epsilon
    epsilon_final = 0.02
    epsilon_decay = 150

    os.makedirs('run', exist_ok=True)

    if args.resume == '':
        now = datetime.now()
        project_dir = os.path.join('run', now.strftime('%Y_%m_%d_%H_%M_%S'))
        os.makedirs(project_dir, exist_ok=True)
        start_episode = 0
    else:
        project_dir = args.resume
        list_of_files = glob.glob(f'{args.resume}/sac_*')
        latest_file = max(list_of_files, key=os.path.getctime)
        start_episode = int(latest_file.split('.')[-2].split('_')[-1]) + 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.resume != '':
        with open(latest_file, 'rb') as f:
            sac = pickle.load(f)
    else:
        sac = SAC(device, obs_size, act_size)



    for episode in range(start_episode, start_episode + args.n_episodes):
        now = datetime.now().isoformat()
        print(f'\nEpisode {episode} {now}')

        epsilon = epsilon_final + (epsilon_start - epsilon_final) * math.exp(-episode / epsilon_decay)
        is_random = np.random.rand() < epsilon
        
        if is_random:
            print(f'Collecting random episode (epsilon={epsilon:.3f})')
            episode_return = rollout(sac, random=True)
        else:
            print(f'Collecting with policy (epsilon={epsilon:.3f})')
            sac.actor.to('cpu')
            episode_return = rollout(sac, random=False)
            sac.actor.to(device)

        # print(f'Return: {episode_return:.2f}')

        with open(f'{project_dir}/episode_{episode}.pickle', 'wb') as f:
            pickle.dump((list(sac.replay_buffer), []), f)

        if episode >= 9 and (episode + 1) % EPISODES_PER_TRAINING == 0:
            print(f'Training... (Episodes {episode-EPISODES_PER_TRAINING+1} to {episode})')
            n_updates = len(sac.replay_buffer) * 2 # 10
            for _ in tqdm(range(n_updates), desc=f"Training after ep {episode}"):
                sac.update_parameters()

        with open(f'{project_dir}/sac_{episode}.pickle', 'wb') as f:
            pickle.dump(sac, f)


if __name__ == '__main__':
    main()
