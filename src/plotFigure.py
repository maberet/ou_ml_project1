from matplotlib import pyplot as plt
from matplotlib.patches import Patch

def create_policy_table(Qtable):
    policytab= []
    for ace in range (2):
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
        policytab.append(policy)
    return policytab

def plot_policy(policy): 
    #Plot the two graph next to each other 
    fig, axs = plt.subplots(1,2, figsize=(10, 5))
    fig.suptitle('Policy')
    for i in range(2):
        x = []
        y = []
        for key, value in policy[i].items():
            x.append(key)
            y.append(value)
        colors = ['yellow', 'purple']
        axs[i].scatter([x[0] for x in x], [x[1] for x in x], c=y, s=100, alpha=0.5)
        xlabel = 'Player sum'
        if i:
            xlabel = 'Player sum with ace usable'
        legend_elements = [
            Patch(facecolor="yellow", edgecolor="black", label="Hit"),
            Patch(facecolor="purple", edgecolor="black", label="Stick")
        ]
        axs[i].legend(handles=legend_elements, loc="upper right")
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel('Dealer card')
    plt.show()


def plot_winrate(winratebyepisode):
    plt.plot([x[0] for x in winratebyepisode], [x[1] for x in winratebyepisode])
    plt.xlabel('Episode')
    plt.ylabel('Win rate')
    plt.title('Win rate by episode')
    plt.show()


def plot_game_stats( numberwin, numberdraw, numberloose):
    labels = 'Win', 'Draw', 'Loose'
    sizes = [numberwin, numberdraw, numberloose]
    colors = ['gold', 'yellowgreen', 'lightcoral']
    explode = (0.1, 0, 0)   
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')
    plt.show()