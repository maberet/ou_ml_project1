import Q_learning_agent as agQ
import Eligibility_traces_agent as agE
import sys
import gymnasium as gym
import argparse

def main():
    parser = argparse.ArgumentParser(description="This program trains an agent to play Blackjack.\n\nThe agent can be of type 'Q' or 'Eligibility'.")
    parser.add_argument("-a", "--agent", help="Type of agent to use", required=True)
    args = parser.parse_args()

    agenttype = args.agent

    print ("Processing to the training of the agent: ", agenttype)

    # Create the environment
    env = gym.make("Blackjack-v1", sab=True)
    if (env is None):
        print("Error: Environment not found")
        return 1

    # Create the agent
    if (agenttype == "Q"):
        agent = agQ.Agent(env)
    elif (agenttype == "Eligibility"):
        agent = agE.EligibilityAgent(env)
    else:
        print("Error: Agent type not recognized")
        return 1
    
    
    agent.train(10000)

    
    return 0 # Success

if __name__ == "__main__":
    main()