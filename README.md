# Single Agent Actor-Critic 

The starter code is written by Chris Lamb where the actor-critic policy is a 2-layer CNN connected to an Actor and a Critic head.

I changed the policy to be a 3-layer CNN followed by a 256-hidden unit LSTMCell. The output of the LSTMCell is then connected to the Actor and the Critic Head.

![Running Reward Plot] (/results/Pong-v0_plot ( AC-lstm-fullepisode-ep=9500).png)
