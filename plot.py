import matplotlib.pyplot as plt
import pickle
import argparse

human_scores = {'Asteroids-v0': 13157,
                'Breakout-v0': 31.8,
                'Pong-v0': 9.3,
                'SpaceInvaders-v0': 1652,
                }

# Parse arguments
parser = argparse.ArgumentParser()

parser.add_argument('-g', action='store', dest='game')

args = parser.parse_args()
game = args.game
data_file = 'results/{}.p'.format(game)

with open(data_file, 'rb') as f:
    data = pickle.load(f)

human = [human_scores[game] for _ in range(len(data))]
out_file = 'results/{}_plot.png'.format(game)

plt.plot(data, label='Actor-Critic')
plt.plot(human, label='Human Expert')
plt.title('{} Rewards versus Episodes'.format(game))
plt.xlabel('Episodes')
plt.ylabel('Average Rewards')
plt.legend()
plt.tight_layout()
plt.savefig(out_file)
plt.show()
