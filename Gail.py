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
import copy

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
TRAJECTORY_LEN = 4096
MAX_STEPS = 1000
learning_rate_disc = 0.0003
learning_rate_gen = 0.0001
learning_rate_gen_value = 0.001
GAMMA = 0.99
ALPHA = 0.001
EPSILON = 0.2
BETA = 1
KL_LIMES = 0.015
TOTAL_STEPS = 4096
EXPERT_STEPS = 2*10**6
IS_DISCRETE = False
EARLY_STOPPING = False
MAX_ITERS_GEN = 4
MAX_ITERS_DISC = 1

def generator_predict(generator, generator_value, state, action_space_size):
    with torch.no_grad():
        if IS_DISCRETE:
            logprob = generator(torch.from_numpy(state).float())
            prob = torch.exp(logprob)
            prob = prob.cpu().detach().numpy()
            prob = np.squeeze(prob)
            return np.random.choice(action_space_size, p = prob), logprob
        mu = generator(torch.from_numpy(state).float())
        values = generator_value(torch.from_numpy(state).float())
        sigma = torch.exp(generator.log_std)
        dist = torch.distributions.MultivariateNormal(mu, torch.eye(4).to(device)*sigma**2)
        action = dist.sample()
        log_probs = dist.log_prob(action)
        return action, values, log_probs

def update_discriminator(optimizer, discriminator, state_action_pairs_expert, state_action_pairs_sample):
    total_loss = 0
    batch = random.sample(state_action_pairs_expert, len(state_action_pairs_sample))
    batch = torch.stack(batch).to(device)
    for iter in range(MAX_ITERS_DISC):



        output_expert = discriminator(batch)
        output_expert_p = discriminator.sigmoid(output_expert)
        output_expert_p = torch.log(1- output_expert_p)
        expert_accuracy = (torch.exp(output_expert_p)> 0.5).float()
        expert_accuracy = torch.sum(expert_accuracy)/len(expert_accuracy)
    
        output_sample = discriminator(state_action_pairs_sample)
        output_sample_p = discriminator.sigmoid(output_sample)
        output_sample_p = torch.log(output_sample_p)
        sample_accuracy = (torch.exp(output_sample_p) > 0.5).float()
        sample_accuracy = torch.sum(sample_accuracy)/len(sample_accuracy)
        sample_accuracy2 = (torch.exp(torch.log(discriminator.sigmoid(discriminator(batch)))) < 0.5).float()
        sample_accuracy2 = torch.sum(sample_accuracy2)/len(sample_accuracy2)

        loss = -torch.mean(output_expert_p) - torch.mean(output_sample_p)
        optimizer.zero_grad()
        loss.backward()
        #for param in discriminator.parameters():
        #    param.grad.data.clamp_(-1,1)
        #torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1)
        optimizer.step()
        total_loss += loss.item()
    print("expert_accuracy", expert_accuracy.item(), "sample_accuracy", sample_accuracy.item(), "sample_accuracy2", sample_accuracy2.item())
    return total_loss/(iter + 1)
        
def discounted_reward(rewards, terminal):
    G = np.zeros(len(rewards))
    ##Calculate discounted reward
    cache = 0
    for t in reversed(range(0, len(rewards))):
        if terminal[t]:
            cache = 0
        cache = cache*GAMMA + rewards[t]
        G[t] = cache
    ##Normalize
    G = (G-np.mean(G))/(np.std(G)+1e-8)
    return G

def update_generator_ppo(optimizer, generator,  generator_value, discriminator, state_action_pairs_sample, states, actions, old_values, old_probs, terminal):
    total_loss = 0
    total_entropy = 0
    total_kl_app = 0
    total_test_acc = 0
    total_policy_loss = 0
    total_value_loss = 0
    updates = 0
    indexs = np.arange(len(states))
    #old_dists = []
    #for i in range(len(states)):
    #    mu = generator(states[i])
    #    sigma = torch.exp(generator.log_std)
        #diag = [torch.eye(4).to(device) for i in range(len(states))]
        #diag = torch.stack(diag).to(device)
        #for i in range(len(sigma)):
        #    diag[i] *= sigma[i]**2
    #    diag = torch.eye(4).to(device)*sigma**2
    #    dist = torch.distributions.MultivariateNormal(mu, diag.to(device))
     #   old_dists.append(torch.tensor(dist))
    #old_dists = torch.stack(old_dists)
    with torch.no_grad():
        r = discriminator.sigmoid(discriminator(state_action_pairs_sample))
        test_accuracy = (r < 0.5).float()
        test_accuracy = torch.sum(test_accuracy)/len(test_accuracy)
        total_test_acc += test_accuracy
        r = -torch.log(r+1e-08)
        r = torch.squeeze(r)
    Q = discounted_reward(r, terminal)
    Q = torch.from_numpy(Q).to(device).float()

    for iters in range(MAX_ITERS_GEN):
        np.random.shuffle(indexs)
        lower_M = 0
        upper_M = 64
        for i in range(64):
            index = indexs[lower_M: upper_M]
            actions_ = actions[index].to(device)
            old_value = old_values[index]
            old_prob = old_probs[index]
            #old_dist = old_dists[index]
            Q_ = Q[index]
            if IS_DISCRETE:
                log_probs = generator(states)
                dist = torch.distributions.Categorical(torch.exp(log_probs))
                entropies = dist.entropy()
            else:
                mu = generator(states[index])
                values = generator_value(states[index])
                sigma = torch.exp(generator.log_std)
                diag = [torch.eye(4).to(device) for i in range(len(actions_))]
                diag = torch.stack(diag).to(device)
                for i in range(len(sigma)):
                    diag[i] *= sigma[i]**2
                dist = torch.distributions.MultivariateNormal(mu, diag.to(device))
                log_probs = dist.log_prob(actions_)
                entropies = dist.entropy()
            if IS_DISCRETE:
                adv = torch.sum(actions_*log_probs, dim = 1)
                entropy_loss = torch.mean(entropies)
                loss = -torch.mean(adv*Q)-ALPHA*entropy_loss
            else:
                adv = (Q_ - values).to(device)
                ##policy_loss
                pred_ratio = torch.exp(log_probs - old_prob)
                clip = torch.clamp(pred_ratio, 1-EPSILON, 1+ EPSILON)
                policy_loss = -torch.mean(torch.min(pred_ratio*adv, clip*adv))
                ##values_loss
                clip = old_value + (values - old_value).clamp(-EPSILON, EPSILON)
                values_loss = adv**2
                clip_loss = (clip-values)**2
                values_loss = torch.max(values_loss, clip_loss)
                values_loss = torch.mean(values_loss)
                ##Entropu
                entropy_loss = torch.mean(entropies)
                loss = BETA*values_loss + policy_loss -ALPHA*entropy_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 0.5)
            optimizer.step()
            lower_M += 64
            upper_M += 64
            updates += 1
            total_loss += loss.item()
            total_entropy += entropy_loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += values_loss.item()
            if EARLY_STOPPING:
                if IS_DISCRETE:
                    updated_log_probs = generator(states[index])
                    kl_app = 0.5*(updated_log_probs - log_probs)**2
                    kl_app = torch.mean(kl_app)
                    total_kl_app += kl_app
                else:
                    kl_app = torch.distributions.kl_divergence(dist, old_dist)
                    kl_app = torch.mean(torch.tensor(kl_app))
                    total_kl_app += kl_app
                #if total_kl_app > KL_LIMES:
                #    #print("Breaking at step" , i , m)
                #    break
    
    return total_loss/(updates), total_entropy/(updates), total_test_acc/(updates), total_policy_loss/(updates), total_value_loss/(updates), total_kl_app/(updates)

def generate_sample_trajectories(generator, generator_value, render, verbose, game):
    trajectories_states = []
    trajectories_actions = []
    trajectories_values = []
    trajectories_oldprobs = []
    trajectories_terminal = []
    trajectories_state_action = []
    env = gym.make(game)
    steps = 0
    episode = 0
    if IS_DISCRETE:
        action_space_size = env.action_space.n
    else:
        action_space_size = -1
    trajs_mean_reward = 0
    while steps < TOTAL_STEPS:
        observation = env.reset()
        total_reward = 0
        for step in range(TRAJECTORY_LEN):
            action, values, logprobs = generator_predict(generator, generator_value, observation, action_space_size)
            if IS_DISCRETE:
                observation, reward, done, info = env.step(action)
            else:
                observation, reward, done, info = env.step(action.cpu().detach().numpy())
            state = torch.tensor(observation).to(device)
            trajectories_states.append(state)
            trajectories_terminal.append(torch.tensor(done))
            trajectories_values.append(values)
            trajectories_oldprobs.append(logprobs)
            if IS_DISCRETE:
                cache = np.zeros(action_space_size)
                cache[action] = 1
                cache = torch.tensor(cache)
                action = torch.tensor(action)
                trajectories_actions.append(action)
                trajectories_state_action.append(torch.cat((state, action.unsqueeze(0)), 0))
            else:
                trajectories_state_action.append(torch.cat((state, action), 0))
                trajectories_actions.append(action)
            if render:
                env.render()
            total_reward += reward
            steps += 1
            if done:
                episode += 1
                trajs_mean_reward += total_reward
                break
        if verbose:
            print("Reward", total_reward)
            print("Samples generated", episode,"/", max(1, episode))
            print("sample_len", step)


    print("Mean reward",trajs_mean_reward / max(1,episode))
    env.close()
    trajectories_states = torch.stack(trajectories_states)
    trajectories_actions = torch.stack(trajectories_actions)
    trajectories_values = torch.stack(trajectories_values)
    trajectories_oldprobs = torch.stack(trajectories_oldprobs)
    trajectories_terminal = torch.stack(trajectories_terminal)
    trajectories_state_action = torch.stack(trajectories_state_action).to(device)
    return trajectories_state_action, trajectories_states, trajectories_actions, trajectories_values, trajectories_oldprobs, trajectories_terminal,  trajs_mean_reward / max(1,episode)

def generate_exprert_trajectories(game, load):
    if load is False:
        env = make_vec_env(game, n_envs=4)

        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=5*10**6)
        model.save("expert" + game)
    else:
        model = PPO.load("expert" + game)
    env = gym.make(game)
    obs = env.reset()
    state_action_pairs = []
    print("GENERATING EXPERT_DATA")
    total_reward = 0
    while True:
        action, _states = model.predict(obs)
        state = torch.tensor(obs)
        state_action_pairs.append(torch.cat((state, torch.from_numpy(action).float()), 0))
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            print("EPISODE_REWARD", total_reward)
            print("EXPERT_DATA GENERATED", len(state_action_pairs), "/", EXPERT_STEPS)
            if len(state_action_pairs) >= EXPERT_STEPS:
                env.close()
                break
            else:
                total_reward = 0
                obs = env.reset()
    print("EXPERT_DATA GENERATED SUCCESFULLY!")
    return state_action_pairs
        

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train():
    game = "BipedalWalker-v3"
    #game = "LunarLander-v2"
    wandb.init(project="GAIL_" + game, entity="neuroori")
    generator_value = value_net.value_net(24).to(device)
    generator = Generator.Generator(24, 4, False).to(device)
    discriminator = Discriminator.Discriminator(28).to(device)
    optimizer_generator = optim.Adam(([
            {"params": generator.parameters(), "lr": learning_rate_gen},
            {"params": generator_value.parameters(), "lr": learning_rate_gen_value}
        ]))
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr = learning_rate_disc)
    expert_data = generate_exprert_trajectories(game, True)
    for i in range(MAX_STEPS):
        sample_state_actions, sample_states, sample_actions, sample_values, sample_probs, sample_terminals ,reward = generate_sample_trajectories(generator, generator_value,  False, False, game)
        loss_disc = update_discriminator(optimizer_discriminator, discriminator, expert_data, sample_state_actions)
        loss_gen, entropy_gen, total_test_acc, policy_loss, value_loss, kl = update_generator_ppo(optimizer_generator, generator,generator_value, discriminator, sample_state_actions, sample_states, sample_actions, sample_values, sample_probs, sample_terminals)
        print("Iteration:",i,"Generator_loss:", loss_gen, "Entropy_gen:", entropy_gen ,
         "Discriminator_loss:", loss_disc, "total_test_acc", total_test_acc)
        wandb.log({"Generator_loss": loss_gen, "Discriminator_loss": loss_disc, "Episode_mean_reward": reward, "total_test_acc":total_test_acc
        ,"policy_loss": policy_loss, "entropy": entropy_gen, "value_loss": value_loss, "kl_div":kl})
        update_linear_schedule(optimizer_generator,i , MAX_STEPS, learning_rate_gen)
        update_linear_schedule(optimizer_discriminator,i , MAX_STEPS, learning_rate_disc)
    torch.save(generator.state_dict(), game + "_geneator.pth")
    torch.save(discriminator.state_dict(), game + "_discriminator.pth")
    generate_sample_trajectories(generator, generator_value, True, True, game)
def main():
    train()

if __name__ == "__main__":
    main()