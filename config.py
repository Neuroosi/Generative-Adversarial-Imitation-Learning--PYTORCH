from dataclasses import dataclass
from typing import Optional

@dataclass
class GAILConfig:
    # Environment settings
    env_name: str = "BipedalWalker-v3"
    is_discrete: bool = False
    action_space: int = 4
    obs_space: int = 24
    
    # Training settings
    max_steps: int = 1000
    trajectory_len: int = 4096
    total_steps: int = 4096
    expert_steps: int = 3 * 10**4
    expert_reward_limit: float = 200
    load_expert: bool = True
    
    # Model loading settings
    load_models: bool = True
    model_path: Optional[str] = f"models/"  # Path to the directory containing the models
    expert_model_path: Optional[str] = f"models/expert{env_name}"  # Path to the pre-trained expert model
    
    # Network settings
    hidden_dim: int = 50
    num_hidden_layers: int = 3
    
    # Optimization settings
    learning_rate_disc: float = 0.003
    learning_rate_gen: float = 0.0003
    learning_rate_gen_value: float = 0.001
    gamma: float = 0.99
    epsilon: float = 0.2
    
    # Training iterations
    max_iters_gen: int = 10
    max_iters_gen_value: int = 10
    max_iters_disc: int = 1
    
    # Batch settings
    batch_size: int = 64
    
    # Device settings
    device: Optional[str] = None  # Will be set automatically
    
    # Wandb settings
    use_wandb: bool = True
    wandb_project: str = "GAIL"
    wandb_entity: Optional[str] = "Neuroori"
    wandb_name: Optional[str] = None
    
    def __post_init__(self):
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Set default model path if not provided
        if self.load_models and self.model_path is None:
            self.model_path = f"models/{self.env_name}"
            
        # Set default expert model path if not provided
        if self.expert_model_path is None:
            self.expert_model_path = f"models/{self.env_name}" 