import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# Initialize a random policy
policy = {}
policyC = {}
policyD = {}

# Player's card
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


def generate_episode(policy):
    episode = []
    state, info = env.reset()
    done = False

    while not done:
        action = policy.get(state, np.random.choice([0, 1]))
        #next_state: what the agent will receive after taking an action
        #reward: reward (win, lose, draw)
        #terminated: environment terminated or not (done playing)
        #truncated: ended by truncation
        #info: additional info about environment

        next_state, reward, terminated, truncated, info = env.step(action)
        # Consider the episode done if either is True
        done = terminated or truncated
        episode.append((state, action, reward))
        state = next_state

    return episode

def generate_episodeEpsilon(policyC, epsilon=0.3):
    episode = []
    state, info = env.reset()
    done = False

    while not done:
        if np.random.rand() < epsilon:
            #choose a random action
            action = np.random.choice([0, 1])
        else:
            #choose the best action based on current policy
            action = policyC.get(state, np.random.choice([0, 1]))

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode.append((state, action, reward))
        state = next_state

    return episode

def generate_episodeDecay(policyD, epsilon=0.3):
    episode = []
    state, info = env.reset()
    done = False

    while not done:
        if np.random.rand() < epsilon:
            # choose a random action
            action = np.random.choice([0, 1])
        else:
            #choose the best action based on current policy
            action = policyD.get(state, np.random.choice([0, 1]))

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode.append((state, action, reward))
        state = next_state

    return episode



def monte_carlo_policy(episodes):
    for i in range(episodes):
        episode = generate_episode(policy)

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

def monte_carlo_policyC(episodes):
    epsilon = 0.1
    for i in range(episodes):
        episode = generate_episodeEpsilon(policy, epsilon)

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
            policyC[state] = best_action

def monte_carlo_policyD(episodes):
    epsilon = 1.0
    switchdecay = 1000
    for i in range(episodes):
        #make an epsiode
        if i > switchdecay:
            epsilon = max(0.05, epsilon * 0.999)
        episode = generate_episodeDecay(policy, epsilon)

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
            policyD[state] = best_action
def test_policy(env, policy, episodes):
    wins = 0
    losses = 0
    draws = 0

    for episode in range(episodes):
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

def test_policyC(env, policyC, episodes):
    wins = 0
    losses = 0
    draws = 0

    for episode in range(episodes):
        state, i = env.reset()
        done = False

        while not done:
            # Get action based on the learned policy
            action = policyC.get(state, 0)
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
    print(f"WinsC: {wins}, DrawsC: {draws}, LossesC: {losses}")
    print(f"WinC rate: {wins/episodes:.2f}, DrawC rate: {draws/episodes:.2f}, LossC rate: {losses/episodes:.2f}")
    return wins, losses, draws
def test_policyD(env, policyD, episodes):
    wins = 0
    losses = 0
    draws = 0

    for epsiode in range(episodes):
        state, info = env.reset()
        done = False

        while not done:
            # Get action based on the learned policy
            action = policyD.get(state, 0)
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
    print(f"WinsD: {wins}, DrawsD: {draws}, LossesD: {losses}")
    print(f"WinD rate: {wins/episodes:.2f}, DrawD rate: {draws/episodes:.2f}, LossD rate: {losses/episodes:.2f}")
    return wins, losses, draws

# run Monte Carlo policy evaluation
monte_carlo_policy(episodes=10000)
monte_carlo_policyC(episodes=10000)
monte_carlo_policyD(episodes=10000)
# Calculate rewards
wins, losses, draws = test_policy(env, policy, episodes=10000)
winsC, lossesC, drawsC = test_policyC(env, policyC, episodes=10000)
winsD, lossesD, drawsD = test_policyD(env, policyD, episodes=10000)
total_games = wins + losses + draws
total_points = wins - losses
total_gamesC = winsC + lossesC + drawsC
total_pointsC = winsC - lossesC
total_gamesD = winsD + lossesD + drawsD
total_pointsD = winsD - lossesD

print(total_points)
print(total_pointsC)
print(total_pointsD)

## GRAPHING
def test_policy_single_episode(env, policy):
    state, info = env.reset()
    done = False

    while not done:
        action = policy.get(state, 0)
        next_state, reward, terminated, truncated, i = env.step(action)
        done = terminated or truncated

        state = next_state
    # Return 1 for win, 0 otherwise
    return 1 if reward == 1 else 0

def test_policy_single_episodeC(env, policyC):
    state, info = env.reset()
    done = False

    while not done:
        action = policyC.get(state, 0)
        next_state, reward, terminated, truncated, i = env.step(action)
        done = terminated or truncated

        state = next_state
    # Return 1 for win, 0 otherwise
    return 1 if reward == 1 else 0

def test_policy_single_episodeD(env, policyD):
    state, info = env.reset()
    done = False

    while not done:
        action = policyD.get(state, 0)
        next_state, reward, terminated, truncated, i = env.step(action)
        done = terminated or truncated

        state = next_state

    # Return 1 for win, 0 otherwise
    return 1 if reward == 1 else 0
# Run test_policy for 10,000 episodes
win_rates = [test_policy_single_episode(env, policy) for i in range(10000)]
win_ratesC = [test_policy_single_episodeC(env, policyC) for i in range(10000)]
win_ratesD = [test_policy_single_episodeD(env, policyD) for i in range(10000)]

# Calculate the cumulative win rate
cumulative_win_rates = np.cumsum(win_rates) / (np.arange(10000) + 1)
cumulative_win_ratesC = np.cumsum(win_ratesC) / (np.arange(10000) + 1)
cumulative_win_ratesD = np.cumsum(win_ratesD) / (np.arange(10000) + 1)

# Plotting the cumulative win rate
plt.figure(figsize=(10, 6))
plt.plot(cumulative_win_rates, label='No Epsilon', marker='o', markevery=500, color = 'red')
plt.plot(cumulative_win_ratesC, label='Constant Epsilon', marker='o', markevery=500, color = 'blue')
plt.plot(cumulative_win_ratesD, label='Decaying Hybrid Epsilon', marker='o', markevery=500, color = 'green')
plt.xlabel('Episodes')
plt.ylabel('Win Rate')
plt.title('Monte Carlo Win Rate over 10,000 Episodes')
plt.grid(True)
plt.legend()
plt.show()




