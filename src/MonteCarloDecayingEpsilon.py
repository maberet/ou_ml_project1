import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import BlackjackEnv

from gymnasium.envs.registration import register

register(
    id='BlackjackCustom-v0',
    entry_point='BlackjackEnv:BlackjackEnv',
    kwargs={'sab': True}
)

env = gym.make('BlackjackCustom-v0')

#env = gym.make('Blackjack-v1', sab=True)

#0 (stand), 1 (hit)
actions = 2
gamma = 0.9

# Initialize action-value function Q(s, a) and returns
Q = {}
returns = {}

# Initialize policy
policy = {}

for player in range(4, 22):
    # Dealer's card
    for dealer in range(1, 11):
        #Keep Track if the ace is useable or not
        for usable_ace in [True, False]:
            #if the player has less than 12 then they do not have a useable ace
            if player < 12:
                usable_ace = False

            state = (player, dealer, usable_ace)
            Q[state] = np.zeros(actions)
            returns[state] = [list(), list()]
            policy[state] = np.random.choice([0, 1])

def monte_carlo_policy(episodes):
    epsilon = 1.0
    switchdecay = 1000
    for i in range(episodes):
        #make an epsiode
        if i > switchdecay:
            epsilon = max(0.05, epsilon * 0.999)
        episode = generate_episode(policy, epsilon)

        G = 0
        visited_state_actions = {}

        # Loop through each step of the episode in reverse order
        for state, action, reward in episode[::-1]:
            #add the reward
            G += reward
            if (state, action) not in visited_state_actions:
                #mark state as visited
                visited_state_actions[(state, action)] = True

                #Using formula: Gt = Rt+1 + γRt+2 + γ^2Rt+3 + · · · + γ^(T−1)Rt+T
                G = reward + gamma * G

                returns[state][action].append(G)
                Q[state][action] = np.mean(returns[state][action])


            best_action = np.argmax(Q[state])
            policy[state] = best_action

def generate_episode(policy, epsilon=0.3):
    episode = []
    state, info = env.reset()
    done = False

    while not done:
        if np.random.rand() < epsilon:
            # choose a random action
            action = np.random.choice([0, 1])
        else:
            # choose the best action based on current policy
            action = policy.get(state, np.random.choice([0, 1]))

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode.append((state, action, reward))
        state = next_state

    return episode


def test_policy(env, policy, episodes):
    wins = 0
    losses = 0
    draws = 0

    for i in range(episodes):
        state, info = env.reset()
        done = False

        while not done:
            # Get action based on the learned policy
            action = policy.get(state, 0)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or terminated

            state = next_state


            if done:
                if reward == 1:
                    wins += 1
                elif reward == 0:
                    draws += 1
                else:
                    losses += 1
    print(f"Tested over {episodes} episodes:")
    print(f"Wins: {wins}, Draws: {draws}, Losses: {losses}")
    print(f"Win rate: {wins/episodes:.2f}, Draw rate: {draws/episodes:.2f}, Loss rate: {losses/episodes:.2f}")
    return wins, losses, draws

# Execute Monte Carlo policy evaluation
monte_carlo_policy(episodes=10000)

# Calculate rewards
wins, losses, draws = test_policy(env, policy, episodes=10000)
total_games = wins + losses + draws
total_points = wins - losses
print(total_points)

#
#Policy Map

#useableabe and nonuseable ace matrix
policy_matrix_usable_ace = np.zeros((10, 10))  # Usable ace (player's hand: 12-21)
policy_matrix_no_ace = np.zeros((18, 10))  # Non-usable ace (player's hand: 4-21)

for player in range(4, 22):
    for dealer in range(1, 11):
        # Non-usable ace
        no_ace = (player, dealer, False)
        action_no_ace = policy.get(no_ace, 0)
        if player <= 21:
            policy_matrix_no_ace[player - 4, dealer - 1] = action_no_ace

        # Usable ace
        if player >= 12:  # Usable ace only applies to totals 12 or more
            usable_ace = (player, dealer, True)
            action_usable_ace = policy.get(usable_ace, 0)
            policy_matrix_usable_ace[player - 12, dealer - 1] = action_usable_ace


hit_patch = mpatches.Patch(color='red', label='Hit (1)')
stand_patch = mpatches.Patch(color='blue', label='Stand (0)')

# Plotting strategy maps
fig, axs = plt.subplots(1, 2, figsize=(20, 8))
# Usable Ace
sns.heatmap(policy_matrix_usable_ace, cmap='coolwarm', annot=True, cbar=False, ax=axs[0],
            xticklabels=range(1, 11), yticklabels=range(12, 22))
axs[0].set_title('Usable Ace')
axs[0].set_xlabel('Dealer Showing')
axs[0].set_ylabel('Player Hand Value')

# Non-Usable Ace
sns.heatmap(policy_matrix_no_ace, cmap='coolwarm', annot=True, cbar=False, ax=axs[1],
            xticklabels=range(1, 11), yticklabels=range(4, 22))
axs[1].set_title('No Usable Ace')
axs[1].set_xlabel('Dealer Showing')
axs[1].set_ylabel('Player Hand Value')

fig.legend(handles=[hit_patch, stand_patch], loc='upper right', ncol=2, bbox_to_anchor=(1, 1))

plt.show()
