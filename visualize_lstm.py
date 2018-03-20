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
    parser.add_argument('-f', action='store', dest='filename', default=None)

    args = parser.parse_args()
    game = args.game
    model_file = args.filename

    # Initialize environment
    render = True
    env = gym.make(game)
    num_actions = env.action_space.n

    # Initialize constants
    num_frames = 4
    max_episodes = 1  # Just render 1 episode
    max_frames = 10000

    # Initialize model
    try:
        with open(model_file, 'rb') as f:
            # Model Save and Load Update: Include both model and optim parameters
            # saved_model = torch.load(model_file,map_location=lambda storage, loc:storage)
            saved_model = pickle.load(f)

            if hasattr(saved_model, '__iter__'):
                model, _ = saved_model
            else:
                model = saved_model

    except OSError:
        print('Model file not found.')
        return


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
