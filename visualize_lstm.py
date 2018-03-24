import gym
from ac_lstm_model import Policy   # LSTM change: was from actor_critic import Policy
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
                # Model Save and Load Update: Include both model and optim parameters
                saved_model = pickle.load(f)
                model, _ = saved_model

        except OSError:
            print('Model file not found.')
            return

    else:
        model = Policy(input_channels=num_frames, num_actions=num_actions)


    model.temperature = 1.0   # When we play, we sample as usual.

    for ep in range(max_episodes):

        state = env.reset()
        state = preprocess_state(state)
        state = np.stack([state]*num_frames)

        # LSTM change - reset LSTM hidden units when episode begins
        cx = Variable(torch.zeros(1, 256))
        hx = Variable(torch.zeros(1, 256))


        for frame in range(max_frames):

            env.render()

            # Select action
            # LSTM Change: Need to cycle hx and cx thru select_action
            action, log_prob, state_value, (hx,cx)  = select_action(model, state, (hx,cx))

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


# LSTM Change: Need to cycle hx and cx thru function
def select_action(model, state, lstm_hc):
    hx , cx = lstm_hc 
    num_frames, height, width = state.shape
    state = torch.FloatTensor(state.reshape(-1, num_frames, height, width))

    probs, state_value,(hx, cx) = model((Variable(state), (hx, cx)))

    m = torch.distributions.Categorical(probs)
    action = m.sample()
    log_prob = m.log_prob(action)
    # LSTM Change: Need to cycle hx and cx thru function
    return action.data[0], log_prob, state_value, (hx, cx)


if __name__ == '__main__':
    main()
