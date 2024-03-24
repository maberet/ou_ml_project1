#This is the class for the eligibility traces Agent 

import random
import json
import plotFigure as pf
from tqdm import tqdm


class EligibilityAgent :

    def __init__(self,env):
        self.env = env
        self.Q = {}
        self.alpha = 0.1
        self.gamma = 0.9
        self.start_epsilon = 0.5
        self.final_epsilon = 0.01   
        self.epsilon = self.start_epsilon
        self.TDdiff=[]
        self.lambda_ = 0.6
        self.eligibility = {}

    # Initialize the Q table 

    def initialize_Q(self):
        for player_sum in range(4,33): # 1 to 32 cards
            for dealer_card in range(1,12): # 1 to 11 cards, 11 means dealer has an ace 
                for action in range(0,2):
                    self.Q[(player_sum,dealer_card,action)] = [0.0,0.0]#[random.randint(1, 100)/100,random.randint(1, 100)/100] # is the initial value of Q(s,a) for all s and a
                    self.eligibility[(player_sum,dealer_card,action)] = 0.0
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
        self.TDdiff.append(td_diff)

    def get_new_action(self, state):
        if random.randint(1,100)/100 < self.epsilon:
            return self.env.action_space.sample()## random action retrieve a random action between 0 and 1
        else:
            maxi= max(self.Q.get(state)[0],self.Q.get(state)[1]) ## greedy action take the max of 
            if maxi == self.Q.get(state)[0]:
                return 0
            else:
                return 1

    def train(self, num_episodes):
        self.initialize_Q()
        numberwin = 0
        numberdraw = 0
        numberloose = 0
        winratebyepisode = []
        self.initialize_E()
        for episode in tqdm(range(num_episodes)):
            done = False
            state,info = self.env.reset()
            self.epsilon = self.final_epsilon + (self.start_epsilon - self.final_epsilon) * (num_episodes - episode) / num_episodes
            while not done:
                action = self.get_new_action(state)
                next_state, reward, done, truncated,info = self.env.step(action)
                self.update_Trace(state)
                self.update_Q(state, action, reward, next_state, done)
                done = done or truncated
                state = next_state
            if reward == 1:
                numberwin += 1
            elif reward == 0:
                numberdraw += 1
            else:
                numberloose += 1
            winratebyepisode.append([episode, numberwin/(episode+1)])
        pf.plot_winrate(winratebyepisode)
        pf.plot_game_stats(numberwin, numberdraw, numberloose)
        pf.plot_policy(pf.create_policy_table(self.Q))
        self.save_Q()# Save the Q table to a file after training
        
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
            