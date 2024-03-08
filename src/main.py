import Q_learning_agent as ag
import gymnasium as gym

def main():
    env = gym.make("Blackjack-v1", sab=True)

    # reset the environment to get the first observation
    agent = ag.Agent(env)
    agent2= ag.Agent(env)
    
    agent.initialize_Q()
    
    agent.train(10000000, 0.1)

    #agent.train(1000000, 0.1)
    #agent.playWithModel()
    
    return 0 # Success

if __name__ == "__main__":
    main()