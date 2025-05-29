import os
import argparse
import pickle
import glob
from datetime import datetime

import torch
from sac import SAC
from rollout import rollout, RealWorldRobotEnv


def main():
    parser = argparse.ArgumentParser(description='Train SAC on Real Robot')
    parser.add_argument('--n_episodes', default=100, type=int)
    parser.add_argument('--resume', default='', type=str)
    args = parser.parse_args()

    obs_size = 4  # TCP Z, Gripper Pos, FSR A0, A1
    act_size = 1  # Gripper action

    os.makedirs('run', exist_ok=True)

    if args.resume == '':
        now = datetime.now()
        project_dir = os.path.join('run', now.strftime('%Y_%m_%d_%H_%M_%S'))
        os.makedirs(project_dir, exist_ok=True)
        start_episode = 0
    else:
        project_dir = args.resume
        list_of_files = glob.glob(f'{args.resume}/td3_*')
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
        
        if episode < 10:
            print('Collecting random episode')
            episode_return = rollout(sac, random=True)
        else:
            print('Collecting with policy')
            sac.actor.to('cpu')
            episode_return = rollout(sac, random=False)
            sac.actor.to(device)

        print(f'Return: {episode_return:.2f}')

        with open(f'{project_dir}/episode_{episode}.pickle', 'wb') as f:
            pickle.dump((list(sac.replay_buffer), []), f)

        if episode >= 9:
            print('Training...')
            for _ in range(len(sac.replay_buffer) * 10):
                sac.update_parameters()

        with open(f'{project_dir}/td3_{episode}.pickle', 'wb') as f:
            pickle.dump(sac, f)


if __name__ == '__main__':
    main()
