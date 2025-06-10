import os
import csv
import argparse
import pickle
from datetime import datetime
from collections import deque

import torch
import numpy as np

from sac import SAC 
from rollout import RealWorldRobotEnv

# ========== 테스트용 rollout ==========
def rollout_test(agent, length=140):
    env = RealWorldRobotEnv()
    state = env.reset()
    env.robot.move_to_initial_pose()
    init_z = env.robot.get_tcp_pose()[2]
    final_z = init_z + 0.15

    episode_return = 0.0
    force_return = 0.0
    drop_return = 0.0

    for t in range(length):
        if t == 80:                      # 물체를 들어 올리는 구간
            env.robot.set_tcp_z(final_z)

        action = agent.act(state, train=False)   # 순수 정책
        next_state, r, _, _, f_r, d_r = env.step(action)

        episode_return += r
        force_return += f_r
        drop_return  += d_r
        state = next_state

    env.reset()
    env.robot.move_to_initial_pose()
    env.close()
    return episode_return, force_return, drop_return
# ======================================

def main():
    parser = argparse.ArgumentParser(description="SAC controller test")
    parser.add_argument("--checkpoint", required=True, help="pickle file path")
    parser.add_argument("--n_tests", type=int, default=5)
    parser.add_argument("--csv", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 학습된 SAC 로드
    with open(args.checkpoint, "rb") as f:
        sac: SAC = pickle.load(f)
    sac.actor.to(device).eval()          # 평가 모드
    sac.critic = None                    # 메모리 절약
    sac.replay_buffer = deque(maxlen=1)  # 버퍼 비활성화

    # 결과 저장용 CSV 설정
    if args.csv == "":
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"test_result_{stamp}.csv"
    else:
        csv_path = args.csv
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f_csv:
        writer = csv.writer(f_csv)
        if write_header:
            writer.writerow(["episode", "return", "force_r", "drop_r"])

        returns = []
        for ep in range(args.n_tests):
            ep_ret, f_ret, d_ret = rollout_test(sac)
            returns.append(ep_ret)
            writer.writerow([ep, ep_ret, f_ret, d_ret])
            print(f"Ep {ep:02d}: total {ep_ret:.2f} | force {f_ret:.2f} | drop {d_ret:.2f}")

    mean_ret = np.mean(returns)
    std_ret  = np.std(returns)
    print(f"\n평균 리턴 {mean_ret:.2f} ± {std_ret:.2f}")
    print(f"세부 결과 CSV 저장 위치: {csv_path}")

if __name__ == "__main__":
    main()
