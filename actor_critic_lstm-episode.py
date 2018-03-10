import gym
from ac_lstm_model import Policy   # LSTM change: was from actor_critic import Policy
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
    max_episodes = 50000
    max_frames = 8000   # limit episode to 6000 game steps
    gamma = 0.95
    lr = 1e-4   # LSTM Update: Work well in 1st iteration
    target_score = 21.0  # Temperature Update: specific to Pong

    max_frames_ep = 0   # track highest number of frames an episode can last

    # Cold start
    if not warm_start:
        # Initialize model
        model = Policy(input_channels=num_frames, num_actions=num_actions)
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=0.1)  #LSTM Change: lr = 1e-4

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
                # Model Save and Load Update: Include both model and optim parameters
                saved_model = pickle.load(f)
                model, optimizer = saved_model

        except OSError:
            print('Saved file not found. Creating new cold start model.')
            model = Policy(input_channels=num_frames, num_actions=num_actions)
            optimizer = optim.RMSprop(model.parameters(), lr=lr,
                                      weight_decay=0.1)
            running_reward = None
            running_rewards = []
            prior_eps = 0

    cuda = torch.cuda.is_available()

    if cuda:
        model = model.cuda()


    for ep in range(max_episodes):

        # Temperature Update: specific to Pong
        # Anneal temperature from 2.0 down to 0.9 based on how far running reward is from 
        # target score
        if running_reward is None:
            model.temperature = 1.8   # Start with temp = 2.0 (Explore)
        else:
        # Anneal temperature from 1.8 down to 1.0 over 100000 episodes
            model.temperature = max(0.8, 1.8 - 0.8 * ((ep+prior_eps) / 5.0e4))

        state = env.reset()
        state = preprocess_state(state)
        state = np.stack([state]*num_frames)

        # LSTM change - reset LSTM hidden units when episode begins
        cx = Variable(torch.zeros(1, 256))
        hx = Variable(torch.zeros(1, 256))
        if cuda:
            cx = cx.cuda()
            hx = hx.cuda()

        reward_sum = 0.0
        for frame in range(max_frames):

            
            # Select action
            # LSTM Change: Need to cycle hx and cx thru select_action
            action, log_prob, state_value, (hx,cx)  = select_action(model, state, (hx,cx), cuda)
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

        if frame > max_frames_ep:
            max_frames_ep = frame

        # Compute/display statistics
        if running_reward is None:
            running_reward = reward_sum
        else:
            running_reward = running_reward * 0.99 + reward_sum * 0.01

        running_rewards.append(running_reward)

        verbose_str = 'Episode {} complete'.format(ep+prior_eps+1)
        verbose_str += '\tReward total:{}'.format(reward_sum)
        verbose_str += '\tRunning mean: {:.4}'.format(running_reward)
        # Temperature Update: Track temp
        if (ep+prior_eps+1) % 5 == 0: 
            verbose_str += '\tTemp = {:.4}'.format(model.temperature)
            verbose_str += '\tMax frames = {}'.format(max_frames_ep)    # Keep track of highest frames/episode

        # Update model
        total_norm = finish_episode(model, optimizer, gamma, cuda)

        if (ep+prior_eps+1) % 5 == 0: 
            verbose_str += '\tMax Norm = {}'.format(total_norm)    # Keep track of highest frames/episode

        sys.stdout.write('\r' + verbose_str)
        sys.stdout.flush()

        if (ep+prior_eps+1) % 500 == 0: 
            model_file = 'saved_models/actor_critic_{}_ep_{}.p'.format(
                                                                game,
                                                                ep+prior_eps+1)
            data_file = 'results/{}.p'.format(game)
            with open(model_file, 'wb') as f:
                # Model Save and Load Update: Include both model and optim parameters 
                pickle.dump((model.cpu(), optimizer), f)

            if cuda:
                model = model.cuda()

            with open(data_file, 'wb') as f:
                pickle.dump(running_rewards, f)


# LSTM Change: Need to cycle hx and cx thru function
def select_action(model, state, lstm_hc, cuda):
    hx , cx = lstm_hc 
    num_frames, height, width = state.shape
    state = torch.FloatTensor(state.reshape(-1, num_frames, height, width))

    if cuda:
        state = state.cuda()

    probs, state_value,(hx, cx) = model((Variable(state), (hx, cx)))

    m = torch.distributions.Categorical(probs)
    action = m.sample()
    log_prob = m.log_prob(action)
    # LSTM Change: Need to cycle hx and cx thru function
    return action.data[0], log_prob, state_value, (hx, cx)


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

    total_norm = torch.nn.utils.clip_grad_norm(model.parameters(), 8000)   # Gradient Clipping Update: prevent exploding gradient

    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]

    return total_norm


if __name__ == '__main__':
    main()
