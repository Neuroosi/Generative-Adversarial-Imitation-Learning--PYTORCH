import gym
import numpy as np
import Generator, Discriminator, value_net
from torch._C import device
from torch import ge, optim
import torch
from wandb import wandb
import random
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from config import GAILConfig
from torch.distributions import Categorical, MultivariateNormal
from typing import Tuple, List
from pathlib import Path


class GAIL:
    def __init__(self, config: GAILConfig):
        self.config = config
        self.device = config.device
        
        # Initialize networks
        self.generator_value = value_net.value_net(config.obs_space).to(self.device)
        self.generator = Generator.Generator(config.obs_space, config.action_space, config.is_discrete).to(self.device)
        self.discriminator = Discriminator.Discriminator(config.action_space + config.obs_space, config.is_discrete).to(self.device)
        
        # Initialize optimizers
        self.optimizer_generator = optim.Adam(self.generator.parameters(), config.learning_rate_gen)
        self.optimizer_generator_value = optim.Adam(self.generator_value.parameters(), config.learning_rate_gen_value)
        self.optimizer_discriminator = optim.Adam(self.discriminator.parameters(), lr=config.learning_rate_disc)
        
        # Load pre-trained models if specified
        if config.load_models:
            self.load_models()
        
        # Initialize wandb if enabled
        if config.use_wandb:
            import wandb
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.wandb_name or f"GAIL_{config.env_name}",
                config=vars(config)
            )
        self.use_wandb = config.use_wandb
        
    def generator_predict(self, state: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate action and value predictions from the generator."""
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(self.device)
            if self.config.is_discrete:
                logprob = self.generator(state)
                values = self.generator_value(state)
                prob = torch.exp(logprob)
                prob = prob.cpu().detach().numpy()
                prob = np.squeeze(prob)
                return np.random.choice(self.config.action_space, p=prob), values, logprob
            else:
                mu = self.generator(state)
                values = self.generator_value(state)
                sigma = torch.exp(self.generator.log_std)
                dist = torch.distributions.MultivariateNormal(mu, torch.eye(self.config.action_space).to(self.device) * sigma**2)
                action = dist.sample()
                log_probs = dist.log_prob(action)
                return action, values, log_probs
    
    def update_discriminator(self, expert_data: List[torch.Tensor], sample_data: torch.Tensor) -> float:
        """Update the discriminator network."""
        total_loss = 0
        batch = random.sample(expert_data, len(sample_data))
        batch = torch.stack(batch).to(self.device)
        
        for _ in range(self.config.max_iters_disc):
            # Expert data forward pass
            output_expert = self.discriminator(batch)
            output_expert_p = self.discriminator.sigmoid(output_expert)
            output_expert_p = torch.log(1 - output_expert_p)
            
            # Sample data forward pass
            output_sample = self.discriminator(sample_data)
            output_sample_p = self.discriminator.sigmoid(output_sample)
            output_sample_p = torch.log(output_sample_p)
            
            # Calculate loss
            loss = -torch.mean(output_expert_p) - torch.mean(output_sample_p)
            
            # Backward pass
            self.optimizer_discriminator.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
            self.optimizer_discriminator.step()
            
            total_loss += loss.item()
            
        return total_loss / self.config.max_iters_disc
    
    def discounted_reward(self, rewards: np.ndarray, terminal: np.ndarray) -> np.ndarray:
        """Calculate discounted rewards."""
        G = np.zeros(len(rewards))
        cache = 0
        for t in reversed(range(len(rewards))):
            if terminal[t]:
                cache = 0
            cache = cache * self.config.gamma + rewards[t]
            G[t] = cache
        return (G - np.mean(G)) / (np.std(G) + 1e-8)
    
    def train(self):
        """Main training loop."""
        # Generate expert data
        expert_data = self.generate_expert_trajectories()
        
        for i in range(self.config.max_steps):
            # Generate sample trajectories
            sample_data, states, actions, values, probs, terminals, reward = self.generate_sample_trajectories()
            
            # Update networks
            loss_disc = self.update_discriminator(expert_data, sample_data)
            entropy_gen, test_acc, policy_loss = self.update_generator_ppo_policy(
                sample_data, states, actions, values, probs, terminals
            )
            value_loss = self.update_generator_ppo_values(
                sample_data, states, actions, values, terminals
            )
            
            # Log metrics
            self.log_metrics(i, loss_disc, reward, test_acc, policy_loss, entropy_gen, value_loss)
            
            # Update learning rates
            self.update_learning_rates(i)
        
        # Save models
        self.save_models()
        
    def log_metrics(self, iteration: int, loss_disc: float, reward: float, 
                   test_acc: float, policy_loss: float, entropy: float, value_loss: float):
        """Log training metrics to wandb if enabled."""
        metrics = {
            "iteration": iteration,
            "discriminator_loss": loss_disc,
            "episode_mean_reward": reward,
            "test_accuracy": test_acc,
            "policy_loss": policy_loss,
            "entropy": entropy,
            "value_loss": value_loss
        }
        
        if self.use_wandb:
            import wandb
            wandb.log(metrics)
        else:
            # Print metrics to console if wandb is not enabled
            print(f"Iteration {iteration}:")
            print(f"  Discriminator Loss: {loss_disc:.4f}")
            print(f"  Episode Mean Reward: {reward:.2f}")
            print(f"  Test Accuracy: {test_acc:.4f}")
            print(f"  Policy Loss: {policy_loss:.4f}")
            print(f"  Entropy: {entropy:.4f}")
            print(f"  Value Loss: {value_loss:.4f}")
            print("-" * 50)
    
    def update_learning_rates(self, epoch: int):
        """Update learning rates using linear schedule."""
        for optimizer, initial_lr in [
            (self.optimizer_generator, self.config.learning_rate_gen),
            (self.optimizer_discriminator, self.config.learning_rate_disc),
            (self.optimizer_generator_value, self.config.learning_rate_gen_value)
        ]:
            lr = initial_lr - (initial_lr * (epoch / float(self.config.max_steps)))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    def save_models(self):
        """Save trained models."""
        # Create models directory if it doesn't exist
        model_path = Path(self.config.model_path)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save models
        torch.save(self.generator.state_dict(), model_path / f"{self.config.env_name}_generator.pth")
        torch.save(self.discriminator.state_dict(), model_path / f"{self.config.env_name}_discriminator.pth")
        torch.save(self.generator_value.state_dict(), model_path / f"{self.config.env_name}_generator_value.pth")
        print(f"Saved models to {model_path}")

    def update_generator_ppo_policy(self, sample_data: torch.Tensor, states: torch.Tensor, 
                                  actions: torch.Tensor, old_values: torch.Tensor, 
                                  old_probs: torch.Tensor, terminals: torch.Tensor) -> Tuple[float, float, float]:
        """Update the generator policy using PPO."""
        total_entropy = 0
        total_test_acc = 0
        total_policy_loss = 0
        updates = 0
        indices = np.arange(len(states))
        
        with torch.no_grad():
            r = self.discriminator.sigmoid(self.discriminator(sample_data))
            test_accuracy = (r < 0.5).float()
            test_accuracy = torch.sum(test_accuracy)/len(test_accuracy)
            total_test_acc += test_accuracy
            r = -torch.log(r + 1e-08)
            r = torch.squeeze(r)
            
        Q = self.discounted_reward(r, terminals)
        Q = torch.from_numpy(Q).to(self.device).float()

        for _ in range(self.config.max_iters_gen):
            np.random.shuffle(indices)
            lower_M = 0
            upper_M = self.config.batch_size
            
            for _ in range(self.config.batch_size):
                index = indices[lower_M:upper_M]
                actions_ = actions[index].to(self.device)
                old_value = old_values[index]
                old_prob = old_probs[index]
                Q_ = Q[index]
                
                with torch.no_grad():
                    values = self.generator_value(states[index])
                adv = (Q_ - values.squeeze()).to(self.device)

                if self.config.is_discrete:
                    log_probs = self.generator(states[index])
                    dist = Categorical(torch.exp(log_probs))
                    entropies = dist.entropy()
                    entropy_loss = torch.mean(entropies)
                    pred_ratio = torch.exp(log_probs - old_prob) * actions_
                    pred_ratio = torch.sum(pred_ratio, dim=1)
                else:
                    mu = self.generator(states[index])
                    sigma = torch.exp(self.generator.log_std)
                    diag = [torch.eye(self.config.action_space).to(self.device) for _ in range(len(actions_))]
                    diag = torch.stack(diag).to(self.device)
                    for i in range(len(sigma)):
                        diag[i] *= sigma[i]**2
                    dist = MultivariateNormal(mu, diag.to(self.device))
                    log_probs = dist.log_prob(actions_)
                    entropies = dist.entropy()
                    entropy_loss = torch.mean(entropies)
                    pred_ratio = torch.exp(log_probs - old_prob)

                # Calculate policy loss
                clip = torch.clamp(pred_ratio, 1 - self.config.epsilon, 1 + self.config.epsilon)
                policy_loss = -torch.mean(torch.min(pred_ratio * adv, clip * adv))

                # Update policy
                self.optimizer_generator.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 0.5)
                self.optimizer_generator.step()

                total_policy_loss += policy_loss.item()
                total_entropy += entropy_loss.item()
                lower_M += self.config.batch_size
                upper_M += self.config.batch_size
                updates += 1

        return total_entropy/updates, total_test_acc, total_policy_loss/updates

    def update_generator_ppo_values(self, sample_data: torch.Tensor, states: torch.Tensor,
                                  actions: torch.Tensor, old_values: torch.Tensor,
                                  terminals: torch.Tensor) -> float:
        """Update the generator value function using PPO."""
        total_value_loss = 0
        updates = 0
        indices = np.arange(len(states))
        
        with torch.no_grad():
            r = self.discriminator.sigmoid(self.discriminator(sample_data))
            r = -torch.log(r + 1e-08)
            r = torch.squeeze(r)
            
        Q = self.discounted_reward(r, terminals)
        Q = torch.from_numpy(Q).to(self.device).float()

        for _ in range(self.config.max_iters_gen_value):
            np.random.shuffle(indices)
            lower_M = 0
            upper_M = self.config.batch_size
            
            for _ in range(self.config.batch_size):
                index = indices[lower_M:upper_M]
                old_value = old_values[index]
                Q_ = Q[index]
                values = self.generator_value(states[index])
                adv = (Q_ - values.squeeze()).to(self.device)

                # Calculate value loss with clipping
                clip = old_value + (values - old_value).clamp(-self.config.epsilon, self.config.epsilon)
                values_loss = (adv)**2
                clip_loss = (clip - values)**2
                values_loss = torch.max(values_loss, clip_loss)
                values_loss = torch.mean(values_loss)

                # Update value function
                self.optimizer_generator_value.zero_grad()
                values_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator_value.parameters(), 1.0)
                self.optimizer_generator_value.step()

                total_value_loss += values_loss.item()
                lower_M += self.config.batch_size
                upper_M += self.config.batch_size
                updates += 1

        return total_value_loss/updates

    def generate_sample_trajectories(self):
        """Generate sample trajectories using the current policy."""
        trajectories_states = []
        trajectories_actions = []
        trajectories_values = []
        trajectories_oldprobs = []
        trajectories_terminal = []
        trajectories_state_action = []
        env = gym.make(self.config.env_name)
        steps = 0
        episode = 0
        trajs_mean_reward = 0
        while steps < self.config.total_steps:
            obs, _ = env.reset()  # Handle the new Gym API reset return
            total_reward = 0
            for step in range(self.config.trajectory_len):
                action, values, logprobs = self.generator_predict(obs)
                prev_obs = obs
                if self.config.is_discrete:
                    obs, reward, terminated, truncated, info = env.step(action)  # Handle the new Gym API step return
                    done = terminated or truncated
                else:
                    obs, reward, terminated, truncated, info = env.step(action.cpu().detach().numpy())  # Handle the new Gym API step return
                    done = terminated or truncated
                state = torch.tensor(prev_obs).to(self.device)
                trajectories_states.append(state)
                trajectories_terminal.append(torch.tensor(done))
                trajectories_values.append(values)
                trajectories_oldprobs.append(logprobs)
                if self.config.is_discrete:
                    cache = np.zeros(self.config.action_space)
                    cache[action] = 1
                    cache = torch.tensor(cache).to(self.device)
                    trajectories_actions.append(cache)
                    trajectories_state_action.append(torch.cat((state, cache.float()), 0))
                else:
                    trajectories_state_action.append(torch.cat((state, action), 0))
                    trajectories_actions.append(action)
                total_reward += reward
                steps += 1  
                if done:
                    episode += 1
                    trajs_mean_reward += total_reward
                    break

        print("Step:", steps, " Mean reward:", trajs_mean_reward / max(1, episode))
        env.close()
        trajectories_states = torch.stack(trajectories_states)
        trajectories_actions = torch.stack(trajectories_actions)
        trajectories_values = torch.stack(trajectories_values)
        trajectories_oldprobs = torch.stack(trajectories_oldprobs)
        trajectories_terminal = torch.stack(trajectories_terminal)
        trajectories_state_action = torch.stack(trajectories_state_action).to(self.device)
        return trajectories_state_action, trajectories_states, trajectories_actions, trajectories_values, trajectories_oldprobs, trajectories_terminal, trajs_mean_reward / max(1, episode)

    def generate_expert_trajectories(self):
        """Generate expert trajectories using a pre-trained PPO model."""
        if not self.config.load_expert:
            env = make_vec_env(self.config.env_name, n_envs=4)
            callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=self.config.expert_reward_limit, verbose=1)
            eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)
            model = PPO("MlpPolicy", env, verbose=1)
            model.learn(total_timesteps=10*10**6, callback=eval_callback)
            model.save(self.config.expert_model_path)
        else:
            try:
                model = PPO.load(self.config.expert_model_path)
                print(f"Loaded expert model from {self.config.expert_model_path}")
            except Exception as e:
                print(f"Warning: Could not load expert model from {self.config.expert_model_path}")
                print(f"Error: {str(e)}")
                print("Training new expert model...")
                env = make_vec_env(self.config.env_name, n_envs=4)
                callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=self.config.expert_reward_limit, verbose=1)
                eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)
                model = PPO("MlpPolicy", env, verbose=1)
                model.learn(total_timesteps=10*10**6, callback=eval_callback)
                model.save(self.config.expert_model_path)
        
        # Create a single environment for expert data collection
        env = gym.make(self.config.env_name)
        obs, _ = env.reset()  # Handle the new Gym API reset return
        state_action_pairs = []
        print("GENERATING EXPERT_DATA")
        total_reward = 0
        
        while True:
            action, _states = model.predict(obs, deterministic=True)  # Add deterministic=True for expert behavior
            state = torch.tensor(obs)
            if self.config.is_discrete:
                cache = np.zeros(self.config.action_space)
                cache[action] = 1
                cache = torch.tensor(cache)
                state_action_pairs.append(torch.cat((state, cache.float()), 0))
            else:
                state_action_pairs.append(torch.cat((state, torch.from_numpy(action).float()), 0))
            
            obs, reward, terminated, truncated, info = env.step(action)  # Handle the new Gym API step return
            done = terminated or truncated
            total_reward += reward
            
            if done:
                print("EPISODE_REWARD", total_reward)
                print("EXPERT_DATA GENERATED", len(state_action_pairs), "/", self.config.expert_steps)
                if len(state_action_pairs) >= self.config.expert_steps:
                    env.close()
                    break
                else:
                    total_reward = 0
                    obs, _ = env.reset()  # Handle the new Gym API reset return
        
        print("EXPERT_DATA GENERATED SUCCESFULLY!")
        return state_action_pairs

    def load_models(self):
        """Load pre-trained models from the specified path."""
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        # Load generator
        generator_path = model_path / f"{self.config.env_name}_generator.pth"
        if generator_path.exists():
            self.generator.load_state_dict(torch.load(generator_path, map_location=self.device))
            print(f"Loaded generator from {generator_path}")
        else:
            print(f"Warning: Generator model not found at {generator_path}")
        
        # Load discriminator
        discriminator_path = model_path / f"{self.config.env_name}_discriminator.pth"
        if discriminator_path.exists():
            self.discriminator.load_state_dict(torch.load(discriminator_path, map_location=self.device))
            print(f"Loaded discriminator from {discriminator_path}")
        else:
            print(f"Warning: Discriminator model not found at {discriminator_path}")
        
        # Load value network
        value_path = model_path / f"{self.config.env_name}_generator_value.pth"
        if value_path.exists():
            self.generator_value.load_state_dict(torch.load(value_path, map_location=self.device))
            print(f"Loaded value network from {value_path}")
        else:
            print(f"Warning: Value network not found at {value_path}")

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    config = GAILConfig()
    gail = GAIL(config)
    gail.train()

if __name__ == "__main__":
    main()