import gym
from ac_lstm_model import Policy
from utils import preprocess_state

import sys
import argparse
import pickle
from collections import deque

import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np


def main():

    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('-g', action='store', dest='game')
    parser.add_argument('-w', action='store_true', dest='warm_start',
                        default=False)

    args = parser.parse_args()
    game = args.game
    warm_start = args.warm_start

    # Initialize environment
    env = gym.make(game)
    num_actions = env.action_space.n

    # Initialize constants
    num_frames = 4
    max_episodes = 1000000
    max_frames = 10000
    gamma = 0.95

    # Cold start
    if not warm_start:
        # Initialize model
        model = Policy(input_channels=num_frames, num_actions=num_actions)

        # Initialize statistics
        running_reward = None
        running_rewards = []
        prior_eps = 0

    # Warm start
    if warm_start:

        data_file = 'results/{}.p'.format(game)

        try:
            with open(data_file, 'rb') as f:
                running_rewards = pickle.load(f)
                running_reward = running_rewards[-1]

            prior_eps = len(running_rewards)

            model_file = 'saved_models/actor_critic_{}_ep_{}.p'.format(
                                                                    game,
                                                                    prior_eps)
            with open(model_file, 'rb') as f:
                model = pickle.load(f)

        except OSError:
            print('Saved file not found. Creating new cold start model.')
            model = Policy(input_channels=num_frames, num_actions=num_actions)
            running_reward = None
            running_rewards = []
            prior_eps = 0

    cuda = torch.cuda.is_available()

    if cuda:
        model = model.cuda()

    optimizer = optim.RMSprop(model.parameters(), lr=1e-3, weight_decay=0.1)

    for ep in range(max_episodes):
        # Anneal temperature from 2.0 down to 0.5 over 10000 episodes
        model.temperature = max(0.5, 2.0 - 1.5 * ((ep+prior_eps) / 1.0e4))

        state = env.reset()
        state = preprocess_state(state)
        state = np.stack([state]*num_frames)

        reward_sum = 0.0
        for frame in range(max_frames):

            # Select action
            action, log_prob, state_value = select_action(model, state, cuda)
            model.saved_actions.append((log_prob, state_value))

            # Perform step
            next_state, reward, done, info = env.step(action)

            # Add reward to reward buffer
            model.rewards.append(reward)
            reward_sum += reward

            # Compute latest state
            next_state = preprocess_state(next_state)

            # Evict oldest diff add new diff to state
            next_state = np.stack([next_state]*num_frames)
            next_state[1:, :, :] = state[:-1, :, :]
            state = next_state

            if done:
                break

        # Compute/display statistics
        if running_reward is None:
            running_reward = reward_sum
        else:
            running_reward = running_reward * 0.99 + reward_sum * 0.01

        running_rewards.append(running_reward)

        verbose_str = 'Episode {} complete'.format(ep+prior_eps+1)
        verbose_str += '\tReward total:{}'.format(reward_sum)
        verbose_str += '\tRunning mean: {:.4}'.format(running_reward)
        sys.stdout.write('\r' + verbose_str)
        sys.stdout.flush()

        # Update model
        finish_episode(model, optimizer, gamma, cuda)

        if (ep+prior_eps+1) % 500 == 0:
            model_file = 'saved_models/actor_critic_{}_ep_{}.p'.format(
                                                                game,
                                                                ep+prior_eps+1)
            data_file = 'results/{}.p'.format(game)
            with open(model_file, 'wb') as f:
                pickle.dump(model.cpu(), f)

            if cuda:
                model = model.cuda()

            with open(data_file, 'wb') as f:
                pickle.dump(running_rewards, f)


def select_action(model, state, cuda):
    num_frames, height, width = state.shape
    state = torch.FloatTensor(state.reshape(-1, num_frames, height, width))

    if cuda:
        state = state.cuda()

    probs, state_value = model(Variable(state))

    m = torch.distributions.Categorical(probs)
    action = m.sample()
    log_prob = m.log_prob(action)
    return action.data[0], log_prob, state_value


def finish_episode(model, optimizer, gamma, cuda):
    current_reward = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = deque()

    for r in model.rewards[::-1]:
        current_reward = r + gamma * current_reward
        rewards.appendleft(current_reward)
    rewards = list(rewards)
    rewards = torch.Tensor(rewards)
    if cuda:
        rewards = rewards.cuda()
    # z-score rewards
    rewards = (rewards - rewards.mean()) / (rewards.std() +
                                            np.finfo(np.float32).eps)

    for (log_prob, state_value), r in zip(saved_actions, rewards):
        reward = r - state_value.data[0]
        policy_losses.append(-log_prob * Variable(reward))
        r = torch.Tensor([r])
        if cuda:
            r = r.cuda()
        value_losses.append(torch.nn.functional.smooth_l1_loss(state_value,
                                                               Variable(r)))

    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]


if __name__ == '__main__':
    main()
