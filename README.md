# Single Agent Actor-Critic 

We implement and train a single-agent actor-critic agent based on the CNN+LSTM+Actor/Critic architecture.

We perform hyperparameter optimization on 4 key parameters:

1.	Temperature
2.	Learning rate
3.	Gradient Clipping
4.	Backprop methodologies (TBPTT or BPTE)

In addition we benchmark our agent against A3C 1,4 and 16 agents implemented by ikostrikov:

https://github.com/ikostrikov/pytorch-a3c

An overview of our results:

** Pong ** 
Our agent achieves:
* human performance (9.3) after 4467 episodes  
* max performance (16.2) after 9327 episodes

** Breakout ** 
Our agent achieves:
* human performance (31.8) after 22201 episodes  
* max performance (154.8) after 109621 episodes

