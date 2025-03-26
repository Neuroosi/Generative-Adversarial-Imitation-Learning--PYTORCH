# Generative Adversarial Imitation Learning (GAIL) - PyTorch Implementation

This repository contains a PyTorch implementation of the Generative Adversarial Imitation Learning (GAIL) algorithm, based on the paper [Generative Adversarial Imitation Learning](https://arxiv.org/pdf/1606.03476.pdf).

## Features

- Implementation of GAIL algorithm with PPO policy optimization
- Support for both discrete and continuous action spaces
- Integration with Weights & Biases for experiment tracking
- Configurable hyperparameters through a dedicated config file
- Support for various Gym environments (tested with BipedalWalker-v3 and LunarLander-v2)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Generative-Adversarial-Imitation-Learning--PYTORCH.git
cd Generative-Adversarial-Imitation-Learning--PYTORCH
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Configure your environment and hyperparameters in `config.py`

2. Run the training:
```bash
python gail.py
```

3. Monitor training progress using Weights & Biases:
```bash
wandb login
```

## Project Structure

- `gail.py`: Main implementation of the GAIL algorithm
- `config.py`: Configuration and hyperparameters
- `Generator.py`: Generator network implementation
- `Discriminator.py`: Discriminator network implementation
- `value_net.py`: Value network implementation
- `requirements.txt`: Project dependencies

## Configuration

You can modify the following parameters in `config.py`:

- Environment settings (env_name, action_space, obs_space)
- Training settings (max_steps, trajectory_len, etc.)
- Network architecture (hidden_dim, num_hidden_layers)
- Optimization parameters (learning rates, gamma, epsilon)
- Training iterations and batch sizes

## Results

The trained models will be saved in the following format:
- `{env_name}_generator.pth`
- `{env_name}_discriminator.pth`
- `{env_name}_generator_value.pth`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
