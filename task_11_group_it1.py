# ----------------------------------------------------------
## Year 2 Block B DataLab Task 11: Reinforcement Learning Group Iteration 1 ##
# Names: Alexi Kehayias, Daan Quaadvlied, Michon Goddijn, Jason van Hamond
# Student Numbers: 232230, 231146, 231849, 232567
# ----------------------------------------------------------

# Import packages
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.env_checker import check_env
from ot2_gym_wrapper import OT2Env
import gymnasium as gym
import argparse
from clearml import Task
import wandb
import typing_extensions as TypeIs
import tensorflow
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

# Load the API key for wandb
os.environ['WANDB_API_KEY'] = 'b5568f289e67777846d0dd9aa888f0d1701b32c1'
# Initiate the remote task.
task = Task.init(project_name="Mentor Group K/Group 1",
                    task_name="Group1_Iteration1")


# Setting docker image
task.set_base_docker('deanis/2023y2b-rl:latest')
# Setting task to run remotely
task.execute_remotely(queue_name="default")

#Define the model
env = OT2Env()

# Initialate wandb
run = wandb.init(project="task11_group",sync_tensorboard=True)
save_path = f"models/{run.id}"
os.makedirs(save_path, exist_ok=True)

# Set the amount of epochs for the model to learn
timesteps = 5000000
# Define the arguments
parser = argparse.ArgumentParser()
# Set the default parameters.
parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=50)
parser.add_argument("--gamma", type=float, default=1)
parser.add_argument("--policy", type=str, default="MlpPolicy")
parser.add_argument("--clip_range", type=float, default=0.1)
parser.add_argument("--value_coefficient", type=float, default=0.5)
args = parser.parse_args()

# Create the PPO Model based on the wrapper.
model = PPO(args.policy, env, verbose=1,
            learning_rate=args.learning_rate, 
            batch_size=args.batch_size, 
            n_steps=args.n_steps, 
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            clip_range=args.clip_range,
            vf_coef=args.value_coefficient,
            tensorboard_log=f"runs/{run.id}")

wandb_callback = WandbCallback(
    model_save_freq=100000,
    model_save_path=f"models/{run.id}",
    verbose=2,)

# Train the model
model.learn(total_timesteps=timesteps, callback=wandb_callback, progress_bar=True, reset_num_timesteps=False,tb_log_name=f"runs/{run.id}")
# Save the model.
model.save(f"models/{run.id}/{timesteps}_baseline")
wandb.save(f"models/{run.id}/{timesteps}_baseline")
