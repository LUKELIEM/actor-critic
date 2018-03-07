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
    max_episodes = 1000000
    max_frames = 6000   # limit episode to 6000 game steps
    gamma = 0.95
    lr = 1e-4   # LSTM Update: Work well in 1st iteration
    target_score = 21.0  # Temperature Update: specific to Pong

    # Truncated Backprop(TBP) Update: 
    # Slide 41-44 CS231N_2017 Lecture 10 
    # Run forward and backward through chunks of sequence vs whole sequence. While hidden values hx and cx
    # are carried forward in time forever.
    chunk_size = 512  

    # Cold start
    if not warm_start:
        # Initialize model
        model = Policy(input_channels=num_frames, num_actions=num_actions)
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=0.1)  #LSTM Change: lr = 1e-4

        # Initialize statistics
        running_reward = -21  # Temperature Update: set running_reward to -21 to ensure temp = 2.0
        running_rewards = []
        prior_eps = 0

    # Warm start
    if warm_start:

        data_file = 'results/ac-lstm-batch_{}.p'.format(game)

        try:
            with open(data_file, 'rb') as f:
                running_rewards = pickle.load(f)
                running_reward = running_rewards[-1]

            prior_eps = len(running_rewards)

            model_file = 'saved_models/ac-lstm-batch_{}_ep_{}.p'.format(
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
            running_reward = -21
            running_rewards = []
            prior_eps = 0

    cuda = torch.cuda.is_available()

    if cuda:
        model = model.cuda()


    for ep in range(max_episodes):   # Truncated Backprop(TBP) Update: For every episode

        # Anneal temperature from 1.8 down to 1.0 over 10000 episodes
        model.temperature = max(0.5, 1.8 - 0.8 * ((ep+prior_eps) / 1.0e4))

        state = env.reset()
        state = preprocess_state(state)
        state = np.stack([state]*num_frames)

        done = False   # TBP Update: init done

        # LSTM change - reset LSTM hidden units when episode begins
        cx = Variable(torch.zeros(1, 256))
        hx = Variable(torch.zeros(1, 256))
        if cuda:
            cx = cx.cuda()
            hx = hx.cuda()

        reward_sum = 0.0
        grad_norm = 0.0  # Track grad norm for the episode

        while not done:  # TBP Update: if episode is not done

            # TBP Update: Forward a fixed number of game steps thru CNN-LSTM
            for frame in range(chunk_size):

                # env.render()    # For initial debugging
            
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
                    # Update model


            # TBP Update: Backprop the fixed number of game steps back thru CNN-LSTM, and perform
            # an update on the parameters of the Actor-Critic.
            if frame > chunk_size/4:   
                grad_norm = finish_chunk(model, optimizer, gamma, cuda)

                # print (grad_norm, frame)   # for debugging nan problem

                # TBP Update: hidden values are carried forward
                cx = Variable(cx.data)   
                hx = Variable(hx.data)

        # TBP Update: At this point, the episode is done. We need to do some bookkeeping
            
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
            verbose_str += '\tGrad norm:{}'.format(grad_norm)   
        sys.stdout.write('\r' + verbose_str)
        sys.stdout.flush()


        # Periodically save model and optimizer parameters, and statistics
        if (ep+prior_eps+1) % 100 == 0: 
            model_file = 'saved_models/ac-lstm-batch_{}_ep_{}.p'.format(
                                                                game,
                                                                ep+prior_eps+1)
            data_file = 'results/ac-lstm-batch_{}.p'.format(game)
            with open(model_file, 'wb') as f:
                # Model Save and Load Update: Include both model and optim parameters 
                pickle.dump((model, optimizer), f)

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


def finish_chunk(model, optimizer, gamma, cuda):   # TBP Update: Name change only
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

    grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), 1000)   # Gradient Clipping Update: prevent exploding gradient

    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]

    return grad_norm


if __name__ == '__main__':
    main()
