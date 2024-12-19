<p align="center">
    <img src="https://github.com/user-attachments/assets/926f6a81-834f-42ba-b854-a85ba2b501d1"/>
</p>

A playground for exploring and implementing reinforcement learning (RL) algorithms with **OpenAI Gym** environments. \
This repository contains modular implementations of RL algorithms, mujoco environments, and detailed experiments to learn and test RL ideas effectively.

## Directory Structure

```
gym-rl-playground/
├── algorithms/          # Implementations of RL algorithms
│   ├── ppo.py 
│   ├── (more to be added)      
│   └── skill_based/
├── experiments/         # Experiment scripts and configurations
│   ├── config/          
│   ├── log/      
│   ├── train.py        
│   └── test.py         
├── mjc_env/             # Mujoco custom enviroments
│   ├── ball/    
│   ├── door/   
│   └── ...           
└── README.md            # Project documentation
```

## Getting Started
### Dependencies
- Python 3.11
- [Gymnasium]((https://gymnasium.farama.org/index.html))
- NumPy
- PyTorch
- Mujoco 2.1.0 (for mujoco custom environments)

## Usage

You can install the necessary packages by running the following command:
```sh
pip install -r requirements.txt
```

### Training an Algorithm
To train a PPO on the Ball Balance environment:
```bash
python -m experiments.train ball_balance_ppo
```

### Testing an Algorithm
To test a trained model:
```bash
python -m experiments.test ball_balance_ppo
```

## Planned Features
- Add support for multi-task algorithms.
- Include pre-trained models for benchmarking.
- Add visualization for training metrics and rewards.