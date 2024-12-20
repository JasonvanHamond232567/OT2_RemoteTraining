# Changes: 
# - New reward function
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
                self.goal_position - self.sim.get_pipette_position(self.sim.robotIds[0])
            ), axis=0
        ).astype(np.float32) 

        # Reset the number of steps
        self.steps = 0

        info = {}

        return observation, info

    def step(self, action):
        # set the actions
        scaled_action = np.array([
            self.x_min + (action[0] + 1.0) / 2.0 * (self.x_max - self.x_min),
            self.y_min + (action[1] + 1.0) / 2.0 * (self.y_max - self.y_min),
            self.z_min + (action[2] + 1.0) / 2.0 * (self.z_max - self.z_min)
        ], dtype=np.float32)

        # Call the step function in the simulation
        self.sim.run([np.append(scaled_action, 0)])
        # Get the current pipette position
        pipette_position = self.sim.get_pipette_position(self.sim.robotIds[0])
        # Calculate how far away the goal is 
        goal_distance = np.linalg.norm(pipette_position - self.goal_position)
        # Initialise reward
        reward = 0
        # Check if previous_distance has been set yet or not
        if hasattr(self, "prev_distance"):
            # Reward the agent on their progress towards the goal
            reward += (self.prev_distance - goal_distance) * 10
        self.prev_distance = goal_distance

        # Check if the agent reaches within the threshold of the goal position
        if np.linalg.norm(pipette_position - self.goal_position) <= 0.001:
            # Reward agent for reaching the goal.
            reward += 100
            terminated = True
        else:
            terminated = False
            reward -= 0.1
            
        # Check for stagnation after 10 steps to avoid exploitation.
        if np.abs(self.prev_distance - goal_distance) < 0.001:
                reward -= 5
        # Set a proximity reward to ensure better outcomes.
        reward += max(0, 10 - goal_distance * 1000) * 0.5  

        # Check if episode should be truncated
        if self.steps >= self.max_steps:
            truncated = True
        else:
            truncated = False
        # Set the observation.
        observation = np.concatenate((pipette_position, self.goal_position - pipette_position), axis=0).astype(np.float32) 
        info = {}


        # Update the amount of steps
        self.steps += 1

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass
    
    def close(self):
        self.sim.close()