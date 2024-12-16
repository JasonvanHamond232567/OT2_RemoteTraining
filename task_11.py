# ----------------------------------------------------------
## Year 2 Block B DataLab Task 11: Reinforcement Learning ##
# Name: Jason van Hamond
# Student Number: 232567
# ----------------------------------------------------------

# Import packages
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from ot2_gym_wrapper import OT2Env
import gymnasium as gym
import argparse
from clearml import Task
import wandb

#Define the model
env = OT2Env()
task = Task.init(project_name="Mentor Group K/Group 1/JasonvanHamond",
                    task_name='Experiment1')

#copy these lines exactly as they are
#setting the base docker image
task.set_base_docker('deanis/2023y2b-rl:latest')
#setting the task to run remotely on the default queue
task.execute_remotely(queue_name="default")


# Ensure compatibility
run = wandb.init(project="local_task11",sync_tensorboard=True)

# Set the amount of epochs for the model to learn
timesteps = 1000
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
# Callback
wandb_callback = WandbCallback(model_save_freq=1000,
                                model_save_path=f"models/{run.id}",
                                verbose=2,)
timesteps = 1000
for i in range(10):
    # add the reset_num_timesteps=False argument to the learn function to prevent the model from resetting the timestep counter
    # add the tb_log_name argument to the learn function to log the tensorboard data to the correct folder
    model.learn(total_timesteps=timesteps, callback=wandb_callback, progress_bar=True, reset_num_timesteps=False,tb_log_name=f"runs/{run.id}")
    # save the model to the models folder with the run id and the current timestep
    model.save(f"models/{run.id}/{timesteps*(i+1)}")