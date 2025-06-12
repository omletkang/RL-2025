import os
import json
import pickle
import argparse
import numpy as np
from datetime import datetime
from rollout import rollout
from sac import SAC
import torch


def main():
    parser = argparse.ArgumentParser(description='Evaluate SAC Policy on Real Robot')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved SAC pickle file (e.g., run/2025_06_04_17_41_36/sac_19.pickle)')
    parser.add_argument('--n_episodes', type=int, default=5,
                        help='Number of evaluation episodes (default: 5)')
    parser.add_argument('--log', action='store_true',
                        help='Log episode results to a JSON file (default: False)')
    parser.add_argument('-object', type=str, # required=True,
                            help='Name of the object being tested (e.g., "bottle", "can")')
    args = parser.parse_args()

    assert os.path.exists(args.model_path), f"Model file does not exist: {args.model_path}"

    # Set up JSON log
    log_path = None
    if args.log:
        # Create a descriptive log file name from the model path
        model_name = os.path.splitext(os.path.basename(args.model_path))[0]
        run_folder = os.path.basename(os.path.dirname(args.model_path))
        log_filename = f"{run_folder}-{model_name}.json"
        # log_filename = f"{run_folder}-{model_name}-{args.object}.json"

        # Define and create the output directory
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, log_filename)
        print(f"Logging results to: {log_path}")


    # Load SAC agent
    with open(args.model_path, 'rb') as f:
        sac = pickle.load(f)

    sac.actor.to('cpu')  # Ensure inference runs on CPU for real robot

    all_returns = []
    episode_logs = []
    history_logs = []
        
    # Define the mapping from status code to a descriptive string
    status_map = {
        0: "success",
        1: "fail-notouch",
        2: "fail-drop",
        3: "fail-break"
    }

    print(f"\nEvaluating {args.n_episodes} episodes using model: {args.model_path}\n")

    for i in range(args.n_episodes):
        print(f"Episode {i + 1}:")
        episode_return, history_dict = rollout(sac, train=False, random=False) # history_dict
        all_returns.append(episode_return)

        # Get user input for success/fail status
        if args.log:
            status_code = -1
            while status_code not in status_map:
                try:
                    prompt = "Enter status (0: success, 1: fail-notouch, 2: fail-drop, 3: fail-break): "
                    user_input = input(prompt)
                    status_code = int(user_input)
                    if status_code not in status_map:
                        print("Invalid input. Please enter 0, 1, 2, or 3.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            
            # Record the log for this episode
            episode_data = {
                "episode": i + 1,
                "return": episode_return,
                "status_code": status_code,
                "status_text": status_map[status_code]
            }
            episode_logs.append(episode_data)

            history_logs.append(history_dict)

        print("------------------------")

    mean_return = np.mean(all_returns)
    std_return = np.std(all_returns)

    print(f"\nEvaluation Summary:")
    print(f"Mean Return: {mean_return:.2f} | Std Dev: {std_return:.2f}")

    # Write all collected data to the JSON file at the end ---
    if args.log:
        log_data = {
            "model_path": args.model_path,
            "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "episodes": episode_logs,
            "history": history_logs,
            "summary": {
                "mean_return": round(mean_return, 4),
                "std_return": round(std_return, 4)
            }
        }
        
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=4)
        
        print(f"\nSuccessfully saved evaluation log to {log_path}")

if __name__ == "__main__":
    main()
