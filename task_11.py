# ----------------------------------------------------------
## Year 2 Block B DataLab Task 11: Reinforcement Learning ##
# Name: Jason van Hamond
# Student Number: 232567
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

# Load the API key for wandb
os.environ['WANDB_API_KEY'] = 'b5568f289e67777846d0dd9aa888f0d1701b32c1'
# Initiate the remote task.
task = Task.init(project_name="Mentor Group K/Group 1/JasonvanHamond",
                    task_name='Experiment1')



# Setting docker image
task.set_base_docker('deanis/2023y2b-rl:latest')
# Setting task to run remotely
task.execute_remotely(queue_name="default")

#Define the model
env = OT2Env()

# Initialate wandb
run = wandb.init(project="task11",sync_tensorboard=True)

# Set the amount of epochs for the model to learn
timesteps = 10000
# Define the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=10)
args = parser.parse_args()

# Create the PPO Model based on the wrapper.
model = PPO("MlpPolicy", env, verbose=1, 
            learning_rate=args.learning_rate, 
            batch_size=args.batch_size, 
            n_steps=args.n_steps, 
            n_epochs=args.n_epochs,)
# Set callbacks
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/",
    log_path="./logs/",
    eval_freq=1000,
    deterministic=True,
    render=False,
)
wandb_callback = WandbCallback(
    model_save_freq=1000,
    model_save_path=f"models/{run.id}",
    verbose=2,)
# Add the callbacks into a list
callbacks = CallbackList([eval_callback, wandb_callback])

timesteps = 1000
for i in range(10):
    # Train the model
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True, reset_num_timesteps=False,tb_log_name=f"runs/{run.id}")
    # Save the model.
    model.save(f"models/{run.id}/{timesteps*(i+1)}")