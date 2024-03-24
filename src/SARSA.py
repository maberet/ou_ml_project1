import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


#env = gym.make('Blackjack-v1', sab=True)

from gymnasium.envs.registration import register

register(
    id='BlackjackCustom-v0',
    entry_point='BlackjackEnv:BlackjackEnv',
    kwargs={'sab': True}
)

env = gym.make('BlackjackCustom-v0')

# 0 (stand), 1 (hit)
actions = 2

# Initialize action-value function Q(s, a) and returns
Q = {}
returns = {}

# Initialize a random policy
policy = {}
policyD = {}

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

#using epsilon greedy policy
def choose_action(state, Q, epsilon):
    if np.random.rand() < epsilon:
        # choose a random action: 0 for stand and 1 for hit
        return np.random.choice([0, 1])
    else:
        # choose the action with the highest Q-value for the current state
        return np.argmax(Q.get(state, np.zeros(actions)))

def sarsa_evaluation(env, episodes, alpha=0.1, gamma=0.9, epsilon=0.1):
    for episode in range(episodes):
        done = False
        state, i = env.reset()
        action = choose_action(state, Q, epsilon)

        while not done:
            next_state, reward, terminated, truncated, i = env.step(action)
            next_action = choose_action(next_state, Q, epsilon)
            done = terminated or truncated

            #Use the formula: Q(s,a)←Q(s,a)+α[r+γQ(s ′,a ′)−Q(s,a)
            if not done:
                next_action = choose_action(next_state, Q, epsilon)  # Next action choice for SARSA
                Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
                # update policy to always choose the best action according to Q
                policy[state] = np.argmax(Q[state])
                state, action = next_state, next_action
            else:
                # if done then the future reward is 0
                Q[state][action] += alpha * (reward - Q[state][action])
                break
def sarsa_evaluationD(env, episodes, alpha=0.1, gamma=0.9, epsilon=0.1):
    for episode in range(episodes):
        done = False
        state, info = env.reset()
        switchdecay = 1000
        if episode > switchdecay:
            epsilon = max(0.05, epsilon * 0.999)
        action = choose_action(state, Q, epsilon)

        while not done:
            next_state, reward, terminated, truncated, info = env.step(action)
            next_action = choose_action(next_state, Q, epsilon)
            done = terminated or truncated

            #Use the formula: Q(s,a)←Q(s,a)+α[r+γQ(s ′,a ′)−Q(s,a)
            if not done:
                next_action = choose_action(next_state, Q, epsilon)  # Next action choice for SARSA
                # Update rule for Q using SARSA
                Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
                # update policy to always choose the best action according to Q
                policyD[state] = np.argmax(Q[state])
                state, action = next_state, next_action
            else:
                # if done then the future reward is 0
                Q[state][action] += alpha * (reward - Q[state][action])
                break

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
            done = terminated or truncated

            state = next_state


            if done:
                if reward > 0:
                    wins += 1
                elif reward == 0:
                    draws += 1
                else:
                    losses += 1

    print(f"Tested over {episodes} episodes:")
    print(f"Wins: {wins}, Draws: {draws}, Losses: {losses}")
    print(f"Win rate: {wins/episodes:.2f}, Draw rate: {draws/episodes:.2f}, Loss rate: {losses/episodes:.2f}")
    return wins, losses, draws

def test_policyD(env, policyD, episodes):
    wins = 0
    losses = 0
    draws = 0

    for i in range(episodes):
        state, info = env.reset()
        done = False

        while not done:
            # Get action based on the learned policy
            action = policyD.get(state, 0)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            state = next_state

            if done:
                if reward > 0:
                    wins += 1
                elif reward == 0:
                    draws += 1
                else:
                    losses += 1

    print(f"Tested over {episodes} episodes:")
    print(f"WinsD: {wins}, DrawsD: {draws}, LossesD: {losses}")
    print(f"Win rate: {wins/episodes:.2f}, Draw rate: {draws/episodes:.2f}, Loss rate: {losses/episodes:.2f}")
    return wins, losses, draws


# run Monte Carlo policy evaluation
sarsa_evaluation(env, episodes=100000, alpha=0.1, gamma=0.9, epsilon=0.3)
sarsa_evaluationD(env, episodes=100000, alpha=0.1, gamma=0.9, epsilon=1.0)

# Calculate rewards
wins, losses, draws = test_policy(env, policy, episodes=10000)
winsD, lossesD, drawsD = test_policyD(env, policyD, episodes=10000)

total_games = wins + losses + draws
total_points = wins - losses
total_gamesD = winsD + lossesD + drawsD
total_pointsD = winsD - lossesD

print(total_points)
print(total_pointsD)


## GRAPHING
def test_policy_single_episode(env, policy):
    state, info = env.reset()
    done = False

    while not done:
        action = policy.get(state, 0)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        state = next_state
    # Return 1 for win, 0 otherwise
    return 1 if reward == 1 else 0



# Run test_policy for 10,000 episodes
win_rates = [test_policy_single_episode(env, policy) for i in range(10000)]
win_ratesD = [test_policy_single_episode(env, policyD) for i in range(10000)]

# Calculate the cumulative win rate
cumulative_win_rates = np.cumsum(win_rates) / (np.arange(10000) + 1)
cumulative_win_ratesD = np.cumsum(win_ratesD) / (np.arange(10000) + 1)

# Plotting the cumulative win rate
plt.figure(figsize=(10, 6))
plt.plot(cumulative_win_rates, label='Constant Epsilon', marker='o', markevery=500, color='red')
plt.plot(cumulative_win_ratesD, label='Decaying Epsilon', marker='o', markevery=500, color='green')
plt.xlabel('Episodes')
plt.ylabel('Win Rate')
plt.title('SARSA Win Rate over 10,000 Episodes')
plt.grid(True)
plt.legend()
plt.show()


#Policy Map

#useableabe and nonuseable ace matrix
policy_matrix_usable_ace = np.zeros((10, 10))  # Usable ace(player's hand: 12-21)
policy_matrix_no_ace = np.zeros((18, 10))  # Non-usable ace(player's hand: 4-21)

for player in range(4, 22):
    for dealer in range(1, 11):
        # Non-usable ace
        no_ace = (player, dealer, False)
        action_no_ace = policyD.get(no_ace, 0)
        if player <= 21:
            policy_matrix_no_ace[player - 4, dealer - 1] = action_no_ace

        # Usable ace
        if player >= 12:  # Usable ace only applies to totals 12 or more
            usable_ace = (player, dealer, True)
            action_usable_ace = policyD.get(usable_ace, 0)
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


