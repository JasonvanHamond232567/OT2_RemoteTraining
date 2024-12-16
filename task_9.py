from sim_class import Simulation
import numpy as np

# Initialize the simulation with one agent
sim = Simulation(num_agents=1)

# Define coordinates for each corner
coords = [(0.2531, 0.2195, 0.2895),
          (0.2531, 0.2195, 0.1195),
          (0.2531, -0.1705, 0.1195),
          (0.2531, -0.1705, 0.2895),
          (-0.187, -0.1705, 0.2895),
          (-0.1874, -0.1705, 0.1195),
          (-0.187, 0.2195, 0.1195),
          (-0.1874, 0.2195, 0.2895)]
# Initiate the index counter for the corners
corner_i = 0
# Run the simulation for a specified number of steps
for i in range(1100):
    # Get agent position
    agent_pos = sim.get_pipette_position(1)
    target = coords[corner_i]

    # Calculate the direction of the next corner.
    direction = np.array(target) - np.array(agent_pos)
    distance = np.linalg.norm(direction)

    # Normalize the direction value
    if distance > 0:
        direction = direction / distance

    # Set velocity for movement of the agent.
    velocity = direction * 0.5
    drop_command = 0
    # Create the next action.
    actions = [[velocity[0], velocity[1], velocity[2], drop_command]]
    # Ensure that the next corner will be found after the agent gets close enough to the currently selected corner
    # (selected value brings the best output of movements.)
    if distance < 0.05:
        corner_i = (corner_i + 1) % len(coords)\
    # Perform the actions
    state = sim.run(actions)
    # Show the observations of the agent.
    print(state)