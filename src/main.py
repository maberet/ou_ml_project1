import Q_learning_agent as agQ
import Eligibility_traces_agent as agE
import sys
import gymnasium as gym
import argparse
import plotFigure as pf

from gymnasium.envs.registration import register

register(
id='BlackjackCustom-v0',
entry_point='BlackjackEnv:BlackjackEnv',
kwargs={'sab': True}
)

def parser():
    parser = argparse.ArgumentParser(description="This program trains an agent to play Blackjack.\n\nThe agent can be of type 'Q' or 'E'.")
    parser.add_argument("-a", "--agent", help="Type of agent to use. The type of agent could be 'Q' for Q-learning, 'E' for eligibility traces", required=True)
    parser.add_argument("-t", "--twist", help="If twist or not, 'T' for the twist", required=False, default="F")
    args = parser.parse_args()
    agenttype = args.agent

    print ("Processing to the training of the agent: ", agenttype)
    if (agenttype != "Q" and agenttype != "E"):
        print("Error: Agent type not recognized")
        return 1
    if (args.twist != "T" and args.twist != "F"):
        print("Error: Twist parameter not recognized")
        return 1
    return agenttype, args.twist

number_of_episodes_training = 100000
number_of_episodes_testing = 10000

def main():
    agenttype, twist = parser()
    
    ## Create the environment with twist or not
    if (twist == "T"):
        print("Twist parameter is set to True")
        env = gym.make('BlackjackCustom-v0')
    else:
        print("Twist parameter is set to False")
        env = gym.make("Blackjack-v1", sab=True)

    if (env is None):
        print("Error: Environment not found")
        return 1

    # Create the agent
    if (agenttype == "Q"):
        agent = agQ.Agent(env,decay=False)
        agentD = agQ.Agent(env,decay=True)
    elif (agenttype == "E"):
        agent = agE.EligibilityAgent(env,decay=False)
        agentD = agE.EligibilityAgent(env,decay=True)
    else:
        print("Error: Agent type not recognized")
        return 1
    

    ##Train the agents
    print("Training of the agent with constant epsilon")
    agent.train(number_of_episodes_training)
    print("Training of the agent with decaying epsilon")
    agentD.train(number_of_episodes_training)
    ##Test the agents
    agent.test(number_of_episodes_testing)
    agentD.test(number_of_episodes_testing)

    pf.plot_winrate_comparison(winratebyepisode=agent.winrate,winratebyepisodeD= agentD.winrate)

    
    return 0 # Success

if __name__ == "__main__":
    main()