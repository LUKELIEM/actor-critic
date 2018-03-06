import numpy as np


def preprocess_state(state):
    # preprocess 210x160x3 uint8 frame into 80x80 2D float array
    # https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
    state = state[35:195]  # crop to 160 x 160 x 3
    state = convert_rgb_to_grayscale(state)
    state = state[::2, ::2]  # downsample by factor of 2
    return state.astype(float).reshape(80, 80)


def convert_rgb_to_grayscale(rgb):
    rgb = np.array(rgb)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
