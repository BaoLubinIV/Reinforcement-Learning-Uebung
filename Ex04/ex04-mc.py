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
    maxiter = 50000  # use whatever number of iterations you want
    for i in range(maxiter):
        states, ret = single_run_20()
        G=ret
        for t in range(len(states)):
            if states[-(t+1)] not in states[:-(t+2)]:
                returns[states[-(t+1)][0]-12, states[-(t+1)][1]-1, int(states[-(t+1)][2])] += G
                visits[states[-(t+1)][0]-12, states[-(t+1)][1]-1, int(states[-(t+1)][2])] += 1
                V[states[-(t+1)][0]-12, states[-(t+1)][1]-1, int(states[-(t+1)][2])] = returns[states[-(t+1)][0]-12, states[-(t+1)][1]-1, int(states[-(t+1)][2])] / visits[states[-(t+1)][0]-12, states[-(t+1)][1]-1, int(states[-(t+1)][2])]
    
    # print(V)
    fig = plt.figure()

    # Plot value function with useable ace
    ax1 = fig.add_subplot(121, projection='3d')
    y = np.arange(12, 22)
    x = np.arange(1, 11)
    x, y = np.meshgrid(x, y)
    ax1.plot_surface(x, y, V[:, :, 1], rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax1.set_title('Value function with useable ace')

    # Plot value function without useable ace
    ax2 = fig.add_subplot(122, projection='3d')
    y = np.arange(12, 22)
    x = np.arange(1, 11)
    x, y = np.meshgrid(x, y)
    ax2.plot_surface(x, y, V[:, :, 0], rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax2.set_title('Value function without useable ace')

    plt.show()
                
def single_run_es(pi):

    obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
    done = False
    states = []
    actions = []
    ret = 0.
    states.append(obs)
    # hit or stick with equal probability in the first step
    p = np.random.rand()
    if p < 0.5:
        obs, reward, done, _ = env.step(0)
        actions.append(0)  # step=0 for stick
    else:
        obs, reward, done, _ = env.step(1)  # step=1 for hit
        actions.append(1)
    ret += reward  # Note that gamma = 1. in this exercise
    while not done:
        states.append(obs)
        #search for action accrording to current policy
        player_sum, dealer_card, useable_ace = obs
        action = pi[player_sum - 12, dealer_card - 1, int(useable_ace)]
        actions.append(action)
        #take the action
        obs, reward, done, _ = env.step(action)
        ret += reward  # Note that gamma = 1. in this exercise
                       # reward is only given at very end of an episode

    return states, actions, ret

def monte_carlo_es():
    """ Implementation of Monte Carlo ES """
    # suggested dimensionality: player_sum (12-21), dealer card (1-10), useable ace (true/false)
    # possible variables to use:
    pi = np.zeros((10, 10, 2), dtype=int)
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
        states, actions, ret = single_run_es(pi)
        G=ret
        for t in range(len(states)):
            if states[-(t+1)] not in states[:-(t+2)]:
                returns[states[-(t+1)][0]-12, states[-(t+1)][1]-1, int(states[-(t+1)][2]), actions[-(t+1)]] += G
                visits[states[-(t+1)][0]-12, states[-(t+1)][1]-1, int(states[-(t+1)][2]), actions[-(t+1)]] += 1
                Q[states[-(t+1)][0]-12, states[-(t+1)][1]-1, int(states[-(t+1)][2]), actions[-(t+1)]] = returns[states[-(t+1)][0]-12, states[-(t+1)][1]-1, int(states[-(t+1)][2]), actions[-(t+1)]] / visits[states[-(t+1)][0]-12, states[-(t+1)][1]-1, int(states[-(t+1)][2]), actions[-(t+1)]]
                pi[states[-(t+1)][0]-12, states[-(t+1)][1]-1, int(states[-(t+1)][2])] = int(np.argmax(Q[states[-(t+1)][0]-12, states[-(t+1)][1]-1, int(states[-(t+1)][2])]))
        


def main():
    #single_run_20()
    policy_evaluation()
    # monte_carlo_es()


if __name__ == "__main__":
    main()
