import gym
import numpy as np
import matplotlib.pyplot as plt

#functions to discretize the states
def discretize_state(state, n_intervals):
    intervals = [
        np.linspace(-1.2, 0.6, n_intervals + 1),
        np.linspace(-0.07, 0.07, n_intervals + 1)
    ]
    discretized = []
    for i in range(len(state)):
        discretized.append(np.digitize(state[i], intervals[i]) - 1)
    return tuple(discretized)


def Q_learning(env, alpha = 0.1, gamma = 0.9, epsilon = 0.1, n_episodes = 100, n_intervals = 20):
    # initialize Q table
    q_table = np.zeros((n_intervals, n_intervals, env.action_space.n))
    cumulative_successes = np.zeros(n_episodes)
    steps_per_episode = np.zeros(n_episodes)
    
    # implement Q learning agent
    for episode in range(n_episodes):
        state = discretize_state(env.reset(), n_intervals)
        done = False
        steps = 0
        successes = 0
        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            
            next_state, reward, done, _ = env.step(action)
            next_state = discretize_state(next_state, n_intervals)
            
            best_next_action = np.argmax(q_table[next_state])
            td_target = reward + gamma * q_table[next_state][best_next_action]
            td_error = td_target - q_table[state][action]
            
            q_table[state][action] += alpha * td_error
            
            state = next_state
            steps += 1
            if done and env.state[0] >= 0.5:
                successes += 1
        cumulative_successes[episode] = successes
        steps_per_episode[episode] = steps
        
        
        if (episode+1) % 20 == 0:
            # calculate and observe current value function 
            value_function = np.max(q_table, axis=2)

            # plot the value function
            plt.figure(figsize=(10, 6))
            plt.title('Value Function at episode'+ str(episode))
            plt.xlabel('Position')
            plt.ylabel('Velocity')
            plt.imshow(value_function, cmap='viridis', origin='lower')
            plt.colorbar(label='Value')
            plt.xticks(range(n_intervals), np.round(np.linspace(-1.2, 0.6, n_intervals), 2), rotation=90)
            plt.yticks(range(n_intervals), np.round(np.linspace(-0.07, 0.07, n_intervals), 2))
            plt.show()
            
    return cumulative_successes, steps_per_episode


      
def Q_learning_evaluation(env, repeats = 10, n_episodes = 1500):
    '''function to implement Q learning for 10 times and observe learning curve'''
    
    all_successes = np.zeros((repeats, n_episodes))
    all_steps = np.zeros((repeats, n_episodes))
    
    for run in range(repeats):
        print(run,"run")
        cumulative_successes, steps_per_episode = Q_learning(env, n_episodes=n_episodes)
        all_successes[run,:] = cumulative_successes
        all_steps[run,:] = steps_per_episode
    
    # calculate average over 10 runs
    avg_successes = np.mean(all_successes, axis=0)
    avg_steps = np.mean(all_steps, axis=0)
    
    # plot learning curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(n_episodes), np.cumsum(avg_successes))
    plt.title('Averaged Cumulative Number of Successes')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Successes')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(n_episodes), avg_steps)
    plt.title('Averaged Number of Steps per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    
    plt.tight_layout()
    plt.show()
    
    






def main():
    env = gym.make('MountainCar-v0')
    env.reset()
    Q_learning(env)
    #Q_learning_evaluation(env)
    env.close()


if __name__ == "__main__":
    main()
