# Door Open Task Environment
This directory contains a custom MuJoCo environment for a door opening task. The goal of this task is to train a model to open a door by applying the correct forces and torques.

## Credits

This environment is derived from the [repository](https://github.com/Yujin1007/franka_door/tree/master/model). \
Special thanks to the contributors for their work.

## Contents

- `scene/`: Contains the mobile robot and door environment files (MJCF).
- `door_open_env_v[version].py`: Contains RL configuration for the door opening task.

## Task Description

In the door opening task, the trained model controls the forces and torques applied to the door to open it. The objective is to successfully open the door within a given time frame.

<img style="display: block; margin: 0 auto;" width="315" alt="Mujoco Screen" src="https://github.com/user-attachments/assets/d5234791-1dc7-4fce-8605-e19332c4c406" />
