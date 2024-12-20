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
        self.action_space = spaces.Box(
            low=np.array([self.x_min, self.y_min, self.z_min]),
            high=np.array([self.x_max, self.y_max, self.z_max]),
            shape=(3,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.array([self.x_min, self.y_min, self.z_min]),
            high=np.array([self.x_max, self.y_max, self.z_max]),
            shape=(3,), dtype=np.float32
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
        # Process observation
        observation = np.array(self.sim.get_pipette_position(self.sim.robotIds[0]), dtype=np.float32)

        # Reset the number of steps
        self.steps = 0

        info = {"goal_position": self.goal_position.tolist(),}

        return observation, info

    def step(self, action):
        # set the actions
        action = np.append(np.array(action, dtype=np.float32), 0)
        # Call the step function
        observation = self.sim.run([action])
        pipette_position = self.sim.get_pipette_position(self.sim.robotIds[0])
        # Calculate how far away the goal is 
        goal_distance = np.linalg.norm(pipette_position - self.goal_position)
        # Process observation
        observation = np.array(pipette_position, dtype=np.float32)
        # Initialise reward
        reward = 0
        # Check if previous_distance has been set yet or not
        if hasattr(self, "prev_distance"):
            # Reward the agent on their progress towards the goal
            reward += self.prev_distance - goal_distance
        self.prev_distance = goal_distance
        
        # Check if the agent reaches within the threshold of the goal position
        if np.linalg.norm(pipette_position - self.goal_position) <= 0.001:
            # Reward agent for reaching the goal.
            reward += 100
            terminated = True
            info = {"episode": {"reward": reward}}
        else:
            terminated = False
        reward -= 0.1

        # Check if episode should be truncated
        if self.steps >= self.max_steps:
            truncated = True
            info = {"episode": {"reward": reward}}
        else:
            truncated = False

        info = {}

        # Update the amount of steps
        self.steps += 1

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass
    
    def close(self):
        self.sim.close()