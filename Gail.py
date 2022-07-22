import gym
import numpy as np
import Generator, Discriminator
from torch._C import device
from torch import optim
import torch
from wandb import wandb
import random
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
TRAJECTORY_LEN = 500
MAX_STEPS = 500
learning_rate_disc = 0.001
learning_rate_gen = 0.0001
GAMMA = 0.99
ALPHA = 0.001
KL_LIMES = 0.0015
EXPERT_TRAJECTORIES = 500

def generator_predict(generator, state, action_space_size):
    with torch.no_grad():
        logprob = generator(torch.from_numpy(state).float())
        prob = torch.exp(logprob)
        prob = prob.cpu().detach().numpy()
        prob = np.squeeze(prob)
        return np.random.choice(action_space_size, p = prob), logprob

def update_discriminator(optimizer, discriminator, state_action_pairs_expert, state_action_pairs_sample):
    total_loss = 0
    batch = random.sample(state_action_pairs_expert, len(state_action_pairs_sample))
    for m in range(len(state_action_pairs_sample)):
            


        state_action_expert = batch[m]
        state_action_sample = state_action_pairs_sample[m]

        output_expert = discriminator(state_action_expert)
        output_expert_p = discriminator.sigmoid(output_expert)
        output_expert_p = torch.log(1- output_expert_p)
        expert_accuracy = (torch.exp(output_expert_p)> 0.5).float()
        expert_accuracy = torch.sum(expert_accuracy)/len(expert_accuracy)
    
        output_sample = discriminator(state_action_sample)
        output_sample_p = discriminator.sigmoid(output_sample)
        output_sample_p = torch.log(output_sample_p)
        sample_accuracy = (torch.exp(output_sample_p) > 0.5).float()
        sample_accuracy = torch.sum(sample_accuracy)/len(sample_accuracy)

        loss = -torch.mean(output_expert_p) - torch.mean(output_sample_p)
        optimizer.zero_grad()
        loss.backward()
        for param in discriminator.parameters():
            param.grad.data.clamp_(-1,1)
        optimizer.step()
        total_loss += loss.item()
    print("expert_accuracy", expert_accuracy.item(), "sample_accuracy", sample_accuracy.item())
    return total_loss/len(state_action_pairs_sample)
        
def discounted_reward(rewards):
    G = np.zeros(len(rewards))
    ##Calculate discounted reward
    cache = 0
    for t in reversed(range(0, len(rewards))):
    #for t in range(len(rewards) - 1, -1, -1):
        cache = cache*GAMMA + rewards[t]
        G[t] = cache
    ##Normalize
    G = (G-np.mean(G))/(np.std(G)+1e-8)
    return G


def update_generator(optimizer, generator, discriminator, state_action_pairs_sample, states, actions):
    total_loss = 0
    total_entropy = 0
    total_kl_app = 0
    for m in range(len(state_action_pairs_sample)):
        log_probs = generator(states[m])
        dist = torch.distributions.Categorical(torch.exp(log_probs))
        entropies = dist.entropy()
        actions_ = actions[m].to(device)
        with torch.no_grad():
            r = discriminator.sigmoid(discriminator(state_action_pairs_sample[m]))
            r = -torch.log(r)
            r = torch.squeeze(r)
        Q = discounted_reward(r)
        Q = torch.from_numpy(Q).to(device).float()
        adv = torch.sum(actions_*log_probs, dim = 1)
        entropy_loss = torch.mean(entropies)
        loss = -torch.mean(adv*Q)-ALPHA*entropy_loss
        optimizer.zero_grad()
        loss.backward()
        for param in generator.parameters():
            param.grad.data.clamp_(-1,1)
        optimizer.step()
        updated_log_probs = generator(states[m])
        kl_app = 0.5*(updated_log_probs - log_probs)**2
        kl_app = torch.mean(kl_app)
        total_kl_app += kl_app
        if total_kl_app > KL_LIMES:
            print("Breaking at step" , m)
            break
        total_loss += loss.item()
        total_entropy += entropy_loss.item()
    return total_loss/(m+1), total_entropy/(m+1)

def generate_sample_trajectories(generator, N, render, verbose, game):
    trajectories= []
    trajectories_states = []
    trajectories_actions = []
    env = gym.make(game)
    action_space_size = env.action_space.n
    trajs_mean_reward = 0
    for episode in range(N):
        observation = env.reset()
        sub_trajectories = []
        sub_trajectories_states = []
        sub_trajectories_actions = []
        total_reward = 0
        for step in range(TRAJECTORY_LEN):
            action, logprobs = generator_predict(generator, observation, action_space_size)
            observation, reward, done, info = env.step(action)
            state = torch.tensor(observation)
            cache = np.zeros(action_space_size)
            cache[action] = 1
            cache = torch.tensor(cache)
            action = torch.tensor(action)
            state_action_pair = torch.cat((state, action.unsqueeze(0)), 0)
            sub_trajectories.append(state_action_pair)
            sub_trajectories_states.append(state)
            sub_trajectories_actions.append(cache)
            if render:
                env.render()
            total_reward += reward
            if done:
                break
        if verbose:
            print("Reward", total_reward)
        sub_trajectories = torch.stack(sub_trajectories)
        trajectories.append(sub_trajectories)
        sub_trajectories_states = torch.stack(sub_trajectories_states)
        trajectories_states.append(sub_trajectories_states)
        sub_trajectories_actions = torch.stack(sub_trajectories_actions)
        trajectories_actions.append(sub_trajectories_actions)
        trajs_mean_reward += total_reward

    print("Mean reward",trajs_mean_reward / N)
    env.close()
    return trajectories, trajectories_states, trajectories_actions, trajs_mean_reward / N

def generate_exprert_trajectories(game, load):
    if load is False:
        env = make_vec_env(game, n_envs=4)

        model = A2C("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=5*10**5)
        model.save("expert" + game)
    else:
        model = A2C.load("expert" + game)
    env = gym.make(game)
    obs = env.reset()
    states = []
    actions = []
    state_action_pairs = []
    print("GENERATING EXPERT_DATA")
    total_reward = 0
    while True:
        action, _states = model.predict(obs)
        states.append(obs)
        actions.append(action)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            states = torch.tensor(states)
            actions = torch.tensor(actions)
            state_action_pair = torch.cat((states, actions.unsqueeze(1)), 1)
            state_action_pairs.append(state_action_pair)
            print("EPISODE_REWARD", total_reward)
            print("EXPERT_DATA GENERATED", len(state_action_pairs), "/", EXPERT_TRAJECTORIES)
            if len(state_action_pairs) == EXPERT_TRAJECTORIES:
                env.close()
                break
            else:
                states = []
                actions = []
                total_reward = 0
                obs = env.reset()
    print("EXPERT_DATA GENERATED SUCCESFULLY!")
    return state_action_pairs
        

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("LEARNING_RATE", lr)

def train():
    #game = "CartPole-v1"
    game = "LunarLander-v2"
    wandb.init(project="GAIL_" + game, entity="neuroori") 
    expert_data = generate_exprert_trajectories(game, True)
    generator = Generator.Generator(8, 4).to(device)
    discriminator = Discriminator.Discriminator(9).to(device)
    optimizer_generator = optim.Adam(generator.parameters(), lr = learning_rate_gen)
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr = learning_rate_disc)
    for i in range(MAX_STEPS):
        sample_trajectories, sample_states, sample_actions, reward = generate_sample_trajectories(generator, 200, False, False, game)
        loss_disc = update_discriminator(optimizer_discriminator, discriminator, expert_data, sample_trajectories)
        loss_gen, entropy_gen = update_generator(optimizer_generator, generator, discriminator, sample_trajectories, sample_states, sample_actions)
        print("Iteration:",i,"Generator_loss:", loss_gen, "Entropy_gen:", entropy_gen , "Discriminator_loss:", loss_disc,)
        wandb.log({"Generator_loss": loss_gen, "Discriminator_loss": loss_disc, "Episode_mean_reward": reward})
        update_linear_schedule(optimizer_generator,i , MAX_STEPS, learning_rate_gen)
        update_linear_schedule(optimizer_discriminator,i , MAX_STEPS, learning_rate_disc)
    torch.save(generator.state_dict(), game + "_geneator.pth")
    torch.save(discriminator.state_dict(), game + "_discriminator.pth")
    generate_sample_trajectories(generator, 200, True, True, game)
def main():
    train()

if __name__ == "__main__":
    main()