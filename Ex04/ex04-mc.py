import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Blackjack-v0')


def single_run_20():
    """ run the policy that sticks for >= 20 """
    # This example shows how to perform a single run with the policy that hits for player_sum >= 20
    # It can be used for the subtasks
    # Use a comment for the print outputs to increase performance (only there as example)
    obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
    done = False
    states = []
    ret = 0.
    while not done:
        print("observation:", obs)
        states.append(obs)
        if obs[0] >= 20:
            print("stick")
            obs, reward, done, _ = env.step(0)  # step=0 for stick
        else:
            print("hit")
            obs, reward, done, _ = env.step(1)  # step=1 for hit
        print("reward:", reward, "\n")
        ret += reward  # Note that gamma = 1. in this exercise
    print("final observation:", obs)
    return states, ret


def policy_evaluation():
    """ Implementation of first-visit Monte Carlo prediction """
    # suggested dimensionality: player_sum (12-21), dealer card (1-10), useable ace (true/false)
    # possible variables to use:
    V = np.zeros((10, 10, 2))
    returns = np.zeros((10, 10, 2))
    visits = np.zeros((10, 10, 2))
    maxiter = 10000  # use whatever number of iterations you want
    for _ in range(maxiter):
        E_states, E_ret = single_run_20()
        first_visit = np.zeros((10, 10, 2))
        
        for i,state in enumerate(E_states):
            player_sum, dealer_card, useable_ace = state
            
            if first_visit[player_sum - 12, dealer_card - 1, int(useable_ace)] == 0: 
                #state not visited before, first visit!
                #update returns for this state 
                returns[player_sum - 12, dealer_card - 1, int(useable_ace)] += E_ret
                visits[player_sum - 12, dealer_card - 1, int(useable_ace)] += 1
                
                #state set as visited in this episode
                first_visit[player_sum - 12, dealer_card - 1, int(useable_ace)] = 1
                
                
    V = np.divide(returns, visits, out=0, where=visits != 0)
    
def plot_value_function(V):
    '''this is a helper function to plot the V as a result'''
    player_sums = np.arange(12, 22)
    dealer_cards = np.arange(1, 11)

    # Plot with Usable Ace
    plt.figure(figsize=(8, 6))
    plt.imshow(V[:, :, 1], origin='lower', cmap='viridis', extent=[1, 10, 12, 21])
    plt.xticks(dealer_cards)
    plt.yticks(player_sums)
    plt.xlabel('Dealer Card')
    plt.ylabel('Player Sum')
    plt.title('Value Function with Usable Ace')
    cbar = plt.colorbar()
    cbar.set_label('State Value')
    plt.tight_layout()
    plt.show()

    # Plot without Usable Ace
    plt.figure(figsize=(8, 6))
    plt.imshow(V[:, :, 0], origin='lower', cmap='viridis', extent=[1, 10, 12, 21])
    plt.xticks(dealer_cards)
    plt.yticks(player_sums)
    plt.xlabel('Dealer Card')
    plt.ylabel('Player Sum')
    plt.title('Value Function without Usable Ace')
    cbar = plt.colorbar()
    cbar.set_label('State Value')
    plt.tight_layout()
    plt.show()


def monte_carlo_es():
    """ Implementation of Monte Carlo ES """
    # suggested dimensionality: player_sum (12-21), dealer card (1-10), useable ace (true/false)
    # possible variables to use:
    pi = np.zeros((10, 10, 2))
    # Q = np.zeros((10, 10, 2, 2))
    Q = np.ones((10, 10, 2, 2)) * 100  # recommended: optimistic initialization of Q
    returns = np.zeros((10, 10, 2, 2))
    visits = np.zeros((10, 10, 2, 2))
    maxiter = 100000000  # use whatever number of iterations you want
    for i in range(maxiter):
        if i % 100000 == 0:
            print("Iteration: " + str(i))
            print(pi[:, :, 0])
            print(pi[:, :, 1])


def main():
    single_run_20()
    # policy_evaluation()
    # monte_carlo_es()


if __name__ == "__main__":
    main()
