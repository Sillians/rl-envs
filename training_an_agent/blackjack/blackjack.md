# Blackjack

Blackjack is a card game where the player tries to get as close to 21 as possible without going over. The player can choose to "hit" (take another card) or "stand" (keep their current hand). The dealer also has a hand and follows a fixed set of rules for hitting and standing. The goal is to have a higher hand than the dealer without going over 21. 

In the context of reinforcement learning, we can model Blackjack as an environment where the agent (player) learns to make optimal decisions (hit or stand) based on the current state of the game (the player's hand and the dealer's visible card). The agent receives rewards based on the outcome of the game (win, lose, or draw) and learns to maximize its expected reward over time.


## Key Rules and Features

- **Objective**: Get a hand value as close to 21 as possible without exceeding it, and beat the dealer's hand.

- **Card Values**: Number cards are worth their face value, face cards (Jack, Queen, King) are worth 10, and Aces can be worth 1 or 11.

- **Actions**: The player can choose to "hit" (take another card) or "stand" (keep their current hand).

- **Dealer's Play**: The dealer must hit until their hand value is 17 or higher.

- **Rewards**: The player receives a reward based on the outcome of the game: +1 for a win, -1 for a loss, and 0 for a draw.

- **State Representation**: The state can be represented as a tuple of the player's current hand value, the dealer's visible card, and whether the player has a usable Ace (an Ace that can be counted as 11 without busting).

- **Exploration vs. Exploitation**: The agent must balance exploring new strategies (hitting or standing in different situations) and exploiting known strategies that yield high rewards.

- **Learning Algorithm**: Q-Learning can be used to learn the optimal policy for playing Blackjack, where the agent updates its Q-values based on the rewards received from the environment after taking actions in different states.

- **Optimal Policy**: The optimal policy for Blackjack can be derived from the Q-values learned by the agent, which indicate the best action to take in each state to maximize expected rewards over time.



## Hand Values

- **Number Cards**: Worth their face value (2-10).

- **Face Cards**: Jack, Queen, King are worth 10.

- **Aces**: Can be worth 1 or 11, depending on which value keeps the hand from busting (exceeding 21).

