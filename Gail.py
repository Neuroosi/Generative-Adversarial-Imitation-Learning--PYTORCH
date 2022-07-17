import gym
import numpy as np
from d3rlpy.datasets import get_cartpole
from sklearn.model_selection import train_test_split
import Generator, Discriminator
from torch._C import device
from torch import optim
import torch
from wandb import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
TRAJECTORY_LEN = 200
MAX_STEPS = 10**6
learning_rate = 0.0001
GAMMA = 0.99

def generator_predict(generator, state, action_space_size):
    with torch.no_grad():
        logprob = generator(torch.from_numpy(state).float())
        prob = torch.exp(logprob)
        prob = prob.cpu().detach().numpy()
        prob = np.squeeze(prob)
        return np.random.choice(action_space_size, p = prob), logprob

def update_discriminator(optimizer, discriminator, state_action_pairs_expert, state_action_pairs_sample):
    total_loss = 0
    for i in range(len(state_action_pairs_expert)):
        state_action_pair_expert = state_action_pairs_expert[i]
        state_action_pair_sample = state_action_pairs_sample[i]

        output_expert = discriminator(state_action_pair_expert)
        output_expert_p = discriminator.logsigmoid(output_expert)

        output_sample = discriminator(state_action_pair_sample)
        output_sample_p = discriminator.logsigmoid(output_sample)
        output_sample_q = torch.log(1 - torch.exp(output_sample_p))

        loss = -(torch.mean(output_expert_p) + torch.mean(output_sample_q)) 
        optimizer.zero_grad()
        loss.backward()
        for param in discriminator.parameters():
            param.grad.data.clamp_(-1,1)
        optimizer.step()
        total_loss += loss.item()

    return total_loss/len(state_action_pair_expert)
        
def discounted_reward(rewards):
    G = np.zeros(len(rewards))
    ##Calculate discounted reward
    cache = 0
    for t in reversed(range(0, len(rewards))):
        if rewards[t] != 0: cache = 0
        cache = cache*GAMMA + rewards[t]
        G[t] = cache
    ##Normalize
    G = (G-np.mean(G))/(np.std(G)+1e-8)
    return G


def update_generator(optimizer, generator, discriminator, state_action_pairs_sample, states, actions):
    total_loss = 0
    for i in range(len(state_action_pairs_sample)):
        state_action_pair = state_action_pairs_sample[i]
        log_probs = generator(states[i])
        action = actions[i]
        action = action.to(device)
        output = discriminator(state_action_pair)
        output_p = torch.exp(discriminator.logsigmoid(output))
        Q = discounted_reward(output_p)
        Q = torch.from_numpy(Q).to(device).float()
        adv = torch.sum(action*log_probs, dim = 1)/len(state_action_pair)
        loss = -torch.mean(adv*Q)
        optimizer.zero_grad()
        loss.backward()
        for param in generator.parameters():
            param.grad.data.clamp_(-1,1)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(state_action_pairs_sample)

def generate_sample_trajectories(generator, N):
    trajectories= []
    trajectories_states = []
    trajectories_actions = []
    env = gym.make("CartPole-v1")
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
            #env.render()
            total_reward += reward
            if done:
                break

        sub_trajectories = torch.stack(sub_trajectories)
        trajectories.append(sub_trajectories)
        sub_trajectories_states = torch.stack(sub_trajectories_states)
        trajectories_states.append(sub_trajectories_states)
        sub_trajectories_actions = torch.stack(sub_trajectories_actions)
        trajectories_actions.append(sub_trajectories_actions)
        trajs_mean_reward += total_reward/TRAJECTORY_LEN

    print("Episode reward",trajs_mean_reward / N)
    env.close()
    return trajectories, trajectories_states, trajectories_actions

def generate_expert_data():
    dataset, _ = get_cartpole()
    state_actions_pairs = []

    for episode in dataset:
        states = torch.tensor(episode.observations)
        actions = torch.tensor(episode.actions)
        state_action_pair = torch.cat((states, actions.unsqueeze(1)), 1)
        state_actions_pairs.append(state_action_pair)
    return state_actions_pairs

def train():
    wandb.init(project="GAIL", entity="neuroori") 
    expert_data = generate_expert_data()
    generator = Generator.Generator(4, 2).to(device)
    discriminator = Discriminator.Discriminator(5).to(device)
    optimizer_generator = optim.Adam(generator.parameters(), lr = learning_rate)
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr = learning_rate)
    for i in range(MAX_STEPS):
        sample_trajectories, sample_states, sample_actions = generate_sample_trajectories(generator, 1582)
        loss_disc = update_discriminator(optimizer_discriminator, discriminator, expert_data, sample_trajectories)
        loss_gen = update_generator(optimizer_generator, generator, discriminator, sample_trajectories, sample_states, sample_actions)
        print("Iteration:",i,"Generator_loss:", loss_gen, "Discriminator_loss:", loss_disc,)
        wandb.log({"Generator_loss": loss_gen, "Discriminator_loss": loss_disc})
def main():
    train()

if __name__ == "__main__":
    main()