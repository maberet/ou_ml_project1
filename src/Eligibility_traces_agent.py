#This is the class for the eligibility traces Agent 

import random
import json
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

class EligibilityAgent :

    def __init__(self,env):
        self.env = env
        self.Q = {}
        self.alpha = 0.01
        self.gamma = 0.9
        self.start_epsilon = 0.1
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
        if random.randint(1, 100)/100 < self.epsilon:
            return self.env.action_space.sample()## random action retrieve a random action between 0 and 1
        else:
            maxi= max(self.Q.get(state)[0],self.Q.get(state)[1]) ## greedy action take the max of 
            if maxi == self.Q.get(state)[0]:
                return 0
            else:
                return 1

    def train(self, num_episodes, epsilon):
        self.initialize_Q()
        self.epsilon = epsilon
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
        self.plot_winrate(winratebyepisode)
        self.plot_game_stats(numberwin, numberdraw, numberloose)
        print("Training done")
        print("Win rate: ", numberwin/num_episodes)
        print("Draw rate: ", numberdraw/num_episodes)
        print("Loose rate: ", numberloose/num_episodes)
        plot_policy(create_policy_table(self.Q,ace=False),ace=False)
        plot_policy(create_policy_table(self.Q,ace=True),ace=True)
        self.save_Q()# Save the Q table to a file after training

    def plot_winrate(self,winratebyepisode):
        plt.plot([x[0] for x in winratebyepisode], [x[1] for x in winratebyepisode])
        plt.xlabel('Episode')
        plt.ylabel('Win rate')
        plt.title('Win rate by episode')
        plt.show()

    def pot_td_diff(self):
        plt.plot(self.TDdiff)
        plt.xlabel('Episode')
        plt.ylabel('TD difference')
        plt.title('TD difference by episode')
        plt.show()

    def plot_game_stats(self, numberwin, numberdraw, numberloose):
        labels = 'Win', 'Draw', 'Loose'
        sizes = [numberwin, numberdraw, numberloose]
        colors = ['gold', 'yellowgreen', 'lightcoral']
        explode = (0.1, 0, 0)   
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
        plt.axis('equal')
        plt.show()
        
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
            

def create_policy_table(Qtable, ace =False):
    policy = {}
    for key, value in Qtable.items():
        tempskey=key
        if (ace and key[2]==1):
            if tempskey[0] <= 21 and tempskey[0] >=11:
                if value[0] > value[1]:
                    policy[tempskey] = 0
                else:
                    policy[tempskey] = 1
        elif (not ace and key[2]==0):
            if tempskey[0] <= 21 and tempskey[0] >=12:
                if value[0] > value[1]:
                    policy[tempskey] = 0
                else:
                    policy[tempskey] = 1
        else:
            continue
    return policy

def plot_policy(policy,ace=False): 
    x = []
    y = []
    for key, value in policy.items():
        x.append(key)
        y.append(value)
    colors = ['yellow', 'purple']
    plt.scatter([x[0] for x in x], [x[1] for x in x], c=y, s=100, alpha=0.5)
    xlabel = 'Player sum'
    if ace:
        xlabel = 'Player sum with ace usable'

    legend_elements = [
        Patch(facecolor="yellow", edgecolor="black", label="Hit"),
        Patch(facecolor="purple", edgecolor="black", label="Stick")
    ]
    plt.legend(handles=legend_elements, loc="upper right")
    plt.xlabel(xlabel)
    plt.ylabel('Dealer card')
    
    plt.title('Policy')
    plt.show()

