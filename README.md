# Single Agent Actor-Critic 

The starter code is written by Chris Lamb where the actor-critic policy is a 2-layer CNN connected to an Actor and a Critic head.

I changed the policy to be a 3-layer CNN followed by a 256-hidden unit LSTMCell. The output of the LSTMCell is then connected to the Actor and the Critic Head.

The agent learned to beat Pong by 10.0 in 4500 episodes, and to 16.0 by 9000 episodes. Its skill then crashed at around episode = 9200.
