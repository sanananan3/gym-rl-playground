# Ball Balance Task Environment

This directory contains a custom MuJoCo environment for a ball balance task. The goal of this task is to train a model to balance a plane in such a way that it prevents a ball from falling off.

## Credits

This environment is adapted from the [original repository](https://github.com/denisgriaznov/CustomMuJoCoEnviromentForRL/tree/master). \
Special thanks to the contributors for their work.

## Contents

- `ball_ballance.xml`: Contains the ball and plane environment (MJCF).
- `ball_balance_env.py`: Contains RL configuration for the ball balance task.

## Task Description

In the ball balance task, the trained model controls the plane's movements to keep the ball balanced on top of it. The objective is to prevent the ball from falling off the plane for as long as possible.

## Video

| Before Training | After Training |
|:---------------:|:--------------:|
| <video controls><source src="https://github.com/user-attachments/assets/e6993d93-80cb-498e-83d0-69ba59fa28ff" type="video/mp4"></video> | <video controls><source src="https://github.com/user-attachments/assets/e0c26fd1-1cd2-44b1-a737-cedfd02106be" type="video/mp4"></video> |