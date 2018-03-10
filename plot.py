import matplotlib.pyplot as plt
import pickle
import argparse

human_scores = {'Asteroids-v0': 13157,
                'Breakout-v0': 31.8,
                'Pong-v0': 9.3,
                'SpaceInvaders-v0': 1652,
                }

# Parse arguments
# python3 plot.py -f "acl-batch_Pong-v0_cs_512.p" -g "Pong-v0"
parser = argparse.ArgumentParser()

parser.add_argument('-g', action='store', dest='game')
parser.add_argument('-f', action='store', dest='filename', default=None)


args = parser.parse_args()
game = args.game
filename = args.filename

if filename is None:
	data_file = 'results/{}.p'.format(game)
else:
	data_file = 'results/{}'.format(filename)

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
