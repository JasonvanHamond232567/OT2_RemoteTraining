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

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)

        # Set the maximum values according to the working environment.
        self.x_min, self.x_max = -0.187, 0.2531
        self.y_min, self.y_max = -0.1705, 0.2195
        self.z_min, self.z_max = 0.1195, 0.2895

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
        # Process observation
        observation = np.array(self.sim.get_pipette_position(self.sim.robotIds[0]), dtype=np.float32)
        # Calculate distance to goal
        distance_to_goal = np.linalg.norm(pipette_position - self.goal_position)
        distance_reward = self.previous_distance - distance_to_goal if hasattr(self, 'previous_distance') else 0
        self.previous_distance = distance_to_goal
        # Reward function
        reward = 0
        # 1. Positive reward for reducing the distance to the goal
        reward += 2 * distance_reward  # Multiply for more noticeable impact 
        # 2. Bonus for being very close to the goal
        if distance_to_goal < 0.05:
            reward += 20  # Large positive reward for reaching the goal
            terminated = True
        else:
            terminated = False
        # 3. Small constant reward for making progress each step
        reward += 1.0  # Encourages movement toward the goal
        # 4. Penalize large or unnecessary actions
        action_magnitude = np.linalg.norm(action)
        reward -= 0.05 * action_magnitude  # Penalize large movements, reduced weight
        # 5. Add a small time penalty to encourage faster completion
        reward -= 0.01  # Small penalty per step
        # Check termination condition
        # Check if the agent reaches within the threshold of the goal position
        if np.linalg.norm(pipette_position - self.goal_position) <= 0.1:
            terminated = True
        else:
            terminated = False

        # Check if episode should be truncated
        if self.steps > self.max_steps:
            truncated = True
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