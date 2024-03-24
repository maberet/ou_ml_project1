#This is the class for the eligibility traces Agent 

import random
import json
import plotFigure as pf
from tqdm import tqdm
import numpy as np
import gymnasium as gym


class EligibilityAgent :

    def __init__(self,env, epsilon=0.1, decay=False):
        self.env = env
        self.Q = {}
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = epsilon
        self.policy={}
        self.lambda_ = 0.6
        self.eligibility = {}
        self.winrate = []
        self.decay = decay
        

    # Initialize the Q table 

    def initialize_Q(self):
        for player_sum in range(4, 33): # 1 to 32 cards
            for dealer_card in range(1,11): # 1 to 11 cards, 11 means dealer has an ace 
                for usable_ace in [True,False]:
                    if player_sum < 12:
                        usable_ace = False
                    state = (player_sum,dealer_card,usable_ace)
                    self.Q[state] =np.zeros(2)#[random.randint(1, 100)/100,random.randint(1, 100)/100] # is the initial value of Q(s,a) for all s and a
                    self.eligibility[state] = 0.0
                    self.policy[state] = np.random.choice([0, 1])  
    def print_Q(self):  
        for key, value in self.Q.items():
            print(key, value)

    def initialize_E(self):
        for player_sum in range(4,33): # 1 to 32 cards
            for dealer_card in range(1,12):
                for action in range(0,2):
                    self.eligibility[(player_sum,dealer_card,action)] = 0.0
    
    def update_Trace (self, state):
        for key in self.eligibility:
            self.eligibility[key] = self.eligibility[key] * self.lambda_ * self.gamma
        self.eligibility[state]= self.eligibility[state]+ 1.0

    def getQ(self):
        return self.Q

    def update_Q(self, state, action, reward, next_state,done):
        future_q = (not done)*max(self.Q.get(next_state)[0],self.Q.get(next_state)[1])
        td_diff = reward + self.gamma*future_q - self.Q[state][action]
        self.Q[state][action] = self.Q[state][action] + self.alpha*(td_diff*self.eligibility[state])

    def get_new_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice([0, 1])
        else:
            maxi= max(self.Q[state][0],self.Q[state][1]) ## greedy action take the max of 
            if maxi == self.Q[state][0]:
                return 0
            else:
                return 1

    def train(self, num_episodes):
        self.initialize_Q()
        self.initialize_E()
        for episode in tqdm(range(num_episodes)):
            done = False
            state,info = self.env.reset()
            action = self.get_new_action(state)
            if self.decay:
                switchdecay = 1000
                if episode > switchdecay:
                    self.epsilon = max(0.05, self.epsilon * 0.999)
            while not done:
                next_state, reward, done, truncated,info = self.env.step(action)
                next_action = self.get_new_action(next_state)
                done = done or truncated
                if not done: 
                    self.update_Trace(state)
                    next_action = self.get_new_action(next_state)  # Next action choice for SARSA
                    self.Q[state][action] += self.alpha * (reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action])*self.eligibility[state]   ##Update the Q value
                    ##Update the policy of the agent
                    self.policy[state] = np.argmax(self.Q[state])
                    state ,action =next_state, next_action
                else:
                    self.Q[state][action] += self.alpha * (reward - self.Q[state][action])
                    break
        
    def test(self, num_episodes):
        wins=0
        losses=0
        draws=0
        winratebyepisode = []
        for episode in range (num_episodes):
            state,info = self.env.reset()
            done = False
            while not done:
                action = self.policy.get(state, 0)
                state, reward, done, truncated,info = self.env.step(action)    
                done = done or truncated
            if reward == 1:
                wins += 1
            elif reward == 0:
                draws += 1
            else:
                losses += 1
            winratebyepisode.append([episode, wins/(episode+1)])
        self.winrate=winratebyepisode
        # pf.plot_winrate(winratebyepisode)
        pf.plot_policy(pf.create_policy_table(self.Q),decaying=self.decay)
        if self.decay:
            print("Decaying epsilon Agent")
        else:
            print("Constant epsilon Agent")
        print("Wins: ", wins, " Draws: ", draws, " Losses: ", losses)   
        print("Win rate: ", wins/num_episodes, " Draw rate: ", draws/num_episodes, " Loss rate: ", losses/num_episodes)
        return wins, losses, draws
     

    # Save the Q table to a file 
    # This is useful if you want to save the Q table after training and use it later
    # self.Q is a dictionary, so we can use the json module to save it to a file
    def save_Q(self):                           
        with open('Q_table.json', 'w') as f:
            #change all keys to string
            self.Q = {str(k): v for k, v in self.Q.items()}

            json.dump(self.Q, f)

    # Load the Q table from a file
    # This is useful if you want to load a Q table that you have saved previously
    # self.Q is a dictionary, so we can use the json module to load it from a file
    def import_Q(self):
        with open('Q_table.json', 'r') as f:
            self.Q = json.load(f)
            #change all keys back to tuple
            self.Q = {tuple(map(int, k.split(','))): v for k, v in self.Q.items()}

if __name__ == "__main__":

    env = gym.make("Blackjack-v1", sab=True)
    agent = EligibilityAgent(env)
    agentD = EligibilityAgent(env,decay=True)
    print("Training of the agent with constant epsilon")
    agent.train(100000)
    print("Training of the agent with decaying epsilon")
    agentD.train(100000)
    agent.test(10000)
    agentD.test(10000)
    pf.plot_winrate_comparison(winratebyepisode=agent.winrate,winratebyepisodeD= agentD.winrate)