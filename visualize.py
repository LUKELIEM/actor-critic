import gym
from actor_critic_model import Policy
from utils import preprocess_state

import argparse
import pickle

import torch
from torch.autograd import Variable
import numpy as np


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('-g', action='store', dest='game')
    parser.add_argument('-e', action='store', dest='episodes', type=int,
                        default=0)

    args = parser.parse_args()
    game = args.game
    episodes = args.episodes

    # Initialize environment
    render = True
    env = gym.make(game)
    num_actions = env.action_space.n

    # Initialize constants
    num_frames = 4
    max_episodes = 10
    max_frames = 10000

    # Initialize model
    if episodes > 0:
        model_file = 'saved_models/actor_critic_{}_ep_{}.p'.format(game,
                                                                   episodes)
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)

        except OSError:
            print('Model file not found.')
            return

    else:
        model = Policy(input_channels=num_frames, num_actions=num_actions)

    model.temperature = max(0.5, 2.0 - 1.5 * ((episodes) / 1.0e4))

    for ep in range(max_episodes):

        state = env.reset()
        state = preprocess_state(state)
        state = np.stack([state]*num_frames)

        for frame in range(max_frames):

            if render:
                env.render()

            # Select action
            action, log_prob, state_value = select_action(model, state)

            # Perform step
            next_state, reward, done, info = env.step(action)

            # Compute latest state
            next_state = preprocess_state(next_state)

            # Evict oldest diff add new diff to state
            next_state = np.stack([next_state]*num_frames)
            next_state[1:, :, :] = state[:-1, :, :]
            state = next_state

            if done:
                break


def select_action(model, state):
    num_frames, height, width = state.shape
    state = torch.FloatTensor(state.reshape(-1, num_frames, height, width))

    probs, state_value = model(Variable(state))

    m = torch.distributions.Categorical(probs)
    action = m.sample()
    log_prob = m.log_prob(action)
    return action.data[0], log_prob, state_value


if __name__ == '__main__':
    main()
