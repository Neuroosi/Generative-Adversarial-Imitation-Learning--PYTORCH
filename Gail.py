import gym
import numpy as np
import Generator, Discriminator
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
TRAJECTORY_LEN = 5000
MAX_STEPS = 1000
learning_rate_disc = 0.0003
learning_rate_gen = 0.0001
GAMMA = 0.99
ALPHA = 0.001
EPSILON = 0.2
BETA = 0.5
KL_LIMES = 0.015
TRAJECTORIES = 100
EXPERT_TRAJECTORIES = 2000
IS_DISCRETE = False
EARLY_STOPPING = True
MAX_ITERS = 8

def generator_predict(generator, state, action_space_size):
    with torch.no_grad():
        if IS_DISCRETE:
            logprob = generator(torch.from_numpy(state).float())
            prob = torch.exp(logprob)
            prob = prob.cpu().detach().numpy()
            prob = np.squeeze(prob)
            return np.random.choice(action_space_size, p = prob), logprob
        mu, values = generator(torch.from_numpy(state).float())
        sigma = torch.exp(generator.log_std)
        dist = torch.distributions.MultivariateNormal(mu, torch.eye(4).to(device)*sigma**2)
        action = dist.sample()
        log_probs = dist.log_prob(action)
        return action, values, log_probs, dist

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
        sample_accuracy2 = (torch.exp(torch.log(discriminator.sigmoid(discriminator(state_action_expert)))) < 0.5).float()
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
    total_test_acc = 0
    for m in range(len(state_action_pairs_sample)):
        actions_ = actions[m].to(device)
        if IS_DISCRETE:
            log_probs = generator(states[m])
            dist = torch.distributions.Categorical(torch.exp(log_probs))
            entropies = dist.entropy()
        else:
            mu = generator(states[m])
            sigma = torch.exp(generator.log_std)
            diag = [torch.eye(4).to(device) for i in range(len(actions_))]
            diag = torch.stack(diag).to(device)
            for i in range(len(sigma)):
                diag[i] *= sigma[i]**2
            dist = torch.distributions.MultivariateNormal(mu, diag.to(device))
            log_probs = dist.log_prob(actions_)
            entropies = dist.entropy()
        with torch.no_grad():
            r = discriminator.sigmoid(discriminator(state_action_pairs_sample[m]))
            test_accuracy = (r < 0.5).float()
            test_accuracy = torch.sum(test_accuracy)/len(test_accuracy)
            total_test_acc += test_accuracy
            r = -torch.log(r+1e-08)
            r = torch.squeeze(r)
        Q = discounted_reward(r)
        Q = torch.from_numpy(Q).to(device).float()
        if IS_DISCRETE:
            adv = torch.sum(actions_*log_probs, dim = 1)
            entropy_loss = torch.mean(entropies)
            loss = -torch.mean(adv*Q)-ALPHA*entropy_loss
        else:
            entropy_loss = torch.mean(entropies)
            loss = -torch.mean(log_probs*Q)-ALPHA*entropy_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1)
        optimizer.step()
        total_loss += loss.item()
        total_entropy += entropy_loss.item()
        if EARLY_STOPPING:
            if IS_DISCRETE:
                updated_log_probs = generator(states[m])
                kl_app = 0.5*(updated_log_probs - log_probs)**2
                kl_app = torch.mean(kl_app)
                total_kl_app += kl_app
            else:
                mu = generator(states[m])
                sigma = torch.exp(generator.log_std)
                diag = [torch.eye(4).to(device) for i in range(len(actions_))]
                diag = torch.stack(diag).to(device)
                for i in range(len(sigma)):
                    diag[i] *= sigma[i]**2
                new_dist = torch.distributions.MultivariateNormal(mu, diag.to(device))
                kl_app = torch.distributions.kl_divergence(new_dist, dist)
                kl_app = torch.mean(kl_app)
                total_kl_app += kl_app
            if total_kl_app > KL_LIMES:
                print("Breaking at step" , m)
                break
    
    return total_loss/(m+1), total_entropy/(m+1), total_test_acc/(m+1)

def update_generator_ppo(optimizer, generator, discriminator, state_action_pairs_sample, states, actions, old_values, old_probs):
    total_loss = 0
    total_entropy = 0
    total_kl_app = 0
    total_test_acc = 0
    total_policy_loss = 0
    total_value_loss = 0
    old_dists = []
    updates = 0
    indexs = np.arange(len(states))
    for m in range(len(states)):
        mu, values = generator(states[m])
        sigma = torch.exp(generator.log_std)
        diag = [torch.eye(4).to(device) for i in range(len(states[m]))]
        diag = torch.stack(diag).to(device)
        for i in range(len(sigma)):
            diag[i] *= sigma[i]**2
        dist = torch.distributions.MultivariateNormal(mu, diag.to(device))
        old_dists.append(dist)
    for iters in range(MAX_ITERS):
        np.random.shuffle(indexs)
        for i in range(len(indexs)):
            m = indexs[i]
            actions_ = actions[m].to(device)
            old_value = old_values[m]
            old_prob = old_probs[m]
            old_dist = old_dists[m]
            if IS_DISCRETE:
                log_probs = generator(states[m])
                dist = torch.distributions.Categorical(torch.exp(log_probs))
                entropies = dist.entropy()
            else:
                mu, values = generator(states[m])
                sigma = torch.exp(generator.log_std)
                diag = [torch.eye(4).to(device) for i in range(len(actions_))]
                diag = torch.stack(diag).to(device)
                for i in range(len(sigma)):
                    diag[i] *= sigma[i]**2
                dist = torch.distributions.MultivariateNormal(mu, diag.to(device))
                log_probs = dist.log_prob(actions_)
                entropies = dist.entropy()
            with torch.no_grad():
                r = discriminator.sigmoid(discriminator(state_action_pairs_sample[m]))
                test_accuracy = (r < 0.5).float()
                test_accuracy = torch.sum(test_accuracy)/len(test_accuracy)
                total_test_acc += test_accuracy
                r = -torch.log(r+1e-08)
                r = torch.squeeze(r)
            Q = discounted_reward(r)
            Q = torch.from_numpy(Q).to(device).float()
            if IS_DISCRETE:
                adv = torch.sum(actions_*log_probs, dim = 1)
                entropy_loss = torch.mean(entropies)
                loss = -torch.mean(adv*Q)-ALPHA*entropy_loss
            else:
                adv = (Q - values).to(device)
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
            updates += 1
            total_loss += loss.item()
            total_entropy += entropy_loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += values_loss.item()
            if EARLY_STOPPING:
                if IS_DISCRETE:
                    updated_log_probs = generator(states[m])
                    kl_app = 0.5*(updated_log_probs - log_probs)**2
                    kl_app = torch.mean(kl_app)
                    total_kl_app += kl_app
                else:
                    kl_app = torch.distributions.kl_divergence(dist, old_dist)
                    kl_app = torch.mean(torch.tensor(kl_app))
                    total_kl_app += kl_app
                if kl_app > KL_LIMES:
                    print("Breaking at step" , i , m)
                    break
    
    return total_loss/(updates), total_entropy/(updates), total_test_acc/(updates), total_policy_loss/(updates), total_value_loss/(updates), total_kl_app/(updates)

def generate_sample_trajectories(generator,  render, verbose, game):
    trajectories= []
    trajectories_states = []
    trajectories_actions = []
    trajectories_values = []
    trajectories_oldprobs = []
    env = gym.make(game)
    if IS_DISCRETE:
        action_space_size = env.action_space.n
    else:
        action_space_size = -1
    trajs_mean_reward = 0
    for episode in range(TRAJECTORIES):
        observation = env.reset()
        sub_trajectories = []
        sub_trajectories_states = []
        sub_trajectories_actions = []
        sub_trajectories_values = []
        sub_trajectories_oldprobs = []
        total_reward = 0
        for step in range(TRAJECTORY_LEN):
            action, values, logprobs, dist = generator_predict(generator, observation, action_space_size)
            if IS_DISCRETE:
                observation, reward, done, info = env.step(action)
            else:
                observation, reward, done, info = env.step(action.cpu().detach().numpy())
            state = torch.tensor(observation)
            sub_trajectories_values.append(values)
            sub_trajectories_oldprobs.append(logprobs)
            if IS_DISCRETE:
                cache = np.zeros(action_space_size)
                cache[action] = 1
                cache = torch.tensor(cache)
                action = torch.tensor(action)
                state_action_pair = torch.cat((state, action.unsqueeze(0)), 0)
            else:
                state_action_pair = torch.cat((state.to(device), action.to(device)), 0)
                cache = action
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
            print("Samples generated", episode,"/", TRAJECTORIES)
            print("sample_len", step)
        sub_trajectories = torch.stack(sub_trajectories)
        trajectories.append(sub_trajectories)
        sub_trajectories_states = torch.stack(sub_trajectories_states)
        trajectories_states.append(sub_trajectories_states)
        sub_trajectories_actions = torch.stack(sub_trajectories_actions)
        trajectories_actions.append(sub_trajectories_actions)
        sub_trajectories_values = torch.stack(sub_trajectories_values)
        trajectories_values.append(sub_trajectories_values)
        sub_trajectories_oldprobs = torch.stack(sub_trajectories_oldprobs)
        trajectories_oldprobs.append(sub_trajectories_oldprobs)
        trajs_mean_reward += total_reward

    print("Mean reward",trajs_mean_reward / TRAJECTORIES)
    env.close()
    return trajectories, trajectories_states, trajectories_actions, trajectories_values, trajectories_oldprobs,  trajs_mean_reward / TRAJECTORIES

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
            states = torch.tensor(np.array(states))
            actions = torch.tensor(np.array(actions))
            if IS_DISCRETE:
                state_action_pair = torch.cat((states, actions.unsqueeze(1)), 1)
            else:
                state_action_pair = torch.cat((states, actions), 1)
            if total_reward > 0:
                state_action_pairs.append(state_action_pair)
                print(len(states), len(actions))
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
    game = "BipedalWalker-v3"
    #game = "LunarLander-v2"
    wandb.init(project="GAIL_" + game, entity="neuroori") 
    generator = Generator.Generator(24, 4, False).to(device)
    discriminator = Discriminator.Discriminator(28).to(device)
    optimizer_generator = optim.Adam(generator.parameters(), lr = learning_rate_gen)
    optimizer_discriminator = optim.SGD(discriminator.parameters(), lr = learning_rate_disc)
    expert_data = generate_exprert_trajectories(game, True)
    for i in range(MAX_STEPS):
        sample_trajectories, sample_states, sample_actions, sample_values, sample_probs ,reward = generate_sample_trajectories(generator,  False, False, game)
        loss_disc = update_discriminator(optimizer_discriminator, discriminator, expert_data, sample_trajectories)
        loss_gen, entropy_gen, total_test_acc, policy_loss, value_loss, kl = update_generator_ppo(optimizer_generator, generator, discriminator, sample_trajectories, sample_states, sample_actions, sample_values, sample_probs)
        print("Iteration:",i,"Generator_loss:", loss_gen, "Entropy_gen:", entropy_gen ,
         "Discriminator_loss:", loss_disc, "total_test_acc", total_test_acc)
        wandb.log({"Generator_loss": loss_gen, "Discriminator_loss": loss_disc, "Episode_mean_reward": reward, "total_test_acc":total_test_acc
        ,"policy_loss": policy_loss, "entropy": entropy_gen, "value_loss": value_loss, "kl_div":kl})
        update_linear_schedule(optimizer_generator,i , MAX_STEPS, learning_rate_gen)
        update_linear_schedule(optimizer_discriminator,i , MAX_STEPS, learning_rate_disc)
    torch.save(generator.state_dict(), game + "_geneator.pth")
    torch.save(discriminator.state_dict(), game + "_discriminator.pth")
    generate_sample_trajectories(generator, 200, True, True, game)
def main():
    train()

if __name__ == "__main__":
    main()