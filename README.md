# Task 9: Simulation Environment

## Setting up
To set up the environment, the following github repository was copied: https://github.com/BredaUniversityADSAI/Y2B-2023-OT2_Twin.git.
Along with this, the Visual Studio C++ Toolkit also had to be installed to ensure the usability of pybullet. A file: task_9.py, was then created to ensure that the Simulation could run.

## Requirements:
The following packages and files are required for this project:
- pybullet==3.2.6
- numpy==1.26.4
- Y2B-2023-OT2_Twin files: https://github.com/BredaUniversityADSAI/Y2B-2023-OT2_Twin.git

## Finding the working Envelope
The working envelope was found by setting the velocities to different combinations of 0 and 1 and printing the coordinates of the pippet at any moment. This would return the coordinates of each corner step by step, which could then be copied and saved in the agent's path. Next, a loop was created to ensure that the agent visualised the corners by moving towards them. The coordinates of the working Envelope are the following:
1. Right-front-top:    (0.2531, 0.2195, 0.2895)
2. Right-front-bottom: (0.2531, 0.2195, 0.1195)
3. Left-front-bottom:  (0.2531, -0.1705, 0.1195)
4. Left-front-top:     (0.2531, -0.1705, 0.2895)
5. Left-back-top:      (-0.187, -0.1705, 0.2895)
6. Left-back-bottom:   (-0.1874, -0.1705, 0.1195)
7. Right-back-bottom:  (-0.187, 0.2195, 0.1195)
8. Right-back-top:     (-0.1874, 0.2195, 0.2895)