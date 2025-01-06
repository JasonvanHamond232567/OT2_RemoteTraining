# Import required packages
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation

# Create the class
class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(num_agents=1, render=self.render)

        # Set the maximum values according to the working environment.
        self.x_min, self.x_max = -0.187, 0.2531
        self.y_min, self.y_max = -0.1705, 0.2195
        self.z_min, self.z_max = 0.1195, 0.2895
        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([self.x_min, self.y_min, self.z_min, -self.x_max, -self.y_max, -self.z_max], dtype=np.float32),
            high=np.array([self.x_max, self.y_max, self.z_max, self.x_max, self.y_max, self.z_max], dtype=np.float32),
            dtype=np.float32
        )

        # Keep track of the step amount
        self.steps = 0

    def reset(self, seed=None):
        # Set a seed if it was not set yet
        if seed is not None:
            np.random.seed(seed)

        # Randomise the goal position
        x = np.random.uniform(self.x_min, self.x_max)
        y = np.random.uniform(self.y_min, self.y_max)
        z = np.random.uniform(self.z_min, self.z_max)
        # Set a random goal position
        self.goal_position = np.array([x, y, z])
        # Call reset function
        observation = self.sim.reset(num_agents=1)
        # Set the observation.
        observation = np.concatenate(
            (
                self.sim.get_pipette_position(self.sim.robotIds[0]), 
                self.goal_position
            ), axis=0
        ).astype(np.float32)

        # Reset the number of steps
        self.steps = 0

        info = {}

        return observation, info

    def step(self, action):
        # set the actions
        action = np.append(np.array(action, dtype=np.float32), 0)
        # Call the step function
        observation = self.sim.run([action])
        pipette_position = self.sim.get_pipette_position(self.sim.robotIds[0])
        # Calculate the distance between the agent and goal
        distance = np.linalg.norm(pipette_position - self.goal_position)
        # Get the previous distance, set infinite if it doens't exist.
        if not hasattr(self, "previous_distance"):
            self.previous_distance = distance
            # Set the initial distance.
            self.initial_distance = distance
        # Reward the agent for getting closer to the goal.
        progress_reward = (self.previous_distance - distance) * 10
        self.previous_distance = distance
        # Give a reward for hitting a milestone.
        milestones = [0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999]
        milestone_reward = 0
        for milestone in milestones:
            # Calculate the milestone reward using the previous, initial and current distances.
            if self.previous_distance > milestone * self.initial_distance and distance <= milestone * self.initial_distance:
                # Calculate reward based on the milestone
                milestone_reward += 20 * milestone
                # Remove the milestone to ensure it does not receive it anymore.
                milestones.remove(milestone)
        # Calculate the reward
        reward = progress_reward + milestone_reward + -0.01

        # Check if the agent reaches within the threshold of the goal position
        if np.linalg.norm(pipette_position - self.goal_position) <= 0.001:
            # Add extra reward for reaching the goal
            reward += 100
            terminated = True
        else:
            terminated = False

        # Check if episode should be truncated
        if self.steps >= self.max_steps:
            truncated = True
        else:
            truncated = False
        observation = np.concatenate((pipette_position, self.goal_position), axis=0).astype(np.float32)
        info = {}

        # Update the amount of steps
        self.steps += 1

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass
    
    def close(self):
        self.sim.close()