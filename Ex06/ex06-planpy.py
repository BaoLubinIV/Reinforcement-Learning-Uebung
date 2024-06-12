import gym
import copy
import random
import numpy as np
import matplotlib.pyplot as plt


class Node:
    def __init__(self, parent=None, action=None):
        self.parent = parent  # parent of this node
        self.action = action  # action leading from parent to this node
        self.children = []
        self.sum_value = 0.0  # sum of values observed for this node, use sum_value/visits for the mean
        self.visits = 0
        self.depth = 0 if parent is None else parent.depth + 1  # depth of the node


def rollout(env, maxsteps=100):
    """Random policy for rollouts"""
    G = 0
    for i in range(maxsteps):
        action = env.action_space.sample()
        _, reward, terminal, _ = env.step(action)
        G += reward
        if terminal:
            return G
    return G


def mcts(env, root, maxiter=500):
    """TODO: Use this function as a starting point for implementing Monte Carlo Tree Search"""

    # this is an example of how to add nodes to the root for all possible actions:
    root.children = [Node(root, a) for a in range(env.action_space.n)]
    epsilon = 0.1
    longest_path_length = []

    for i in range(maxiter):
        state = copy.deepcopy(env)
        G = 0.0

        # TODO: traverse the tree using an epsilon greedy tree policy
        # This is an example howto randomly choose a node and perform the action:

        node = root

        while node.children:
            if random.random() < epsilon:
                node = random.choice(root.children)
            else:
                # select best child node, if possible, otherwise choose first child (argmax will only return first index if only elem)
                values = [
                    (c.sum_value / c.visits) if c.visits > 0 else -np.inf
                    for c in node.children
                ]

                node = node.children[np.argmax(values)]

            _, reward, terminal, _ = state.step(node.action)
            G += reward
            if terminal:
                break

        # TODO: Expansion of tree
        # This performs a rollout (Simulation):
        if not terminal:
            node.children = [Node(node, a) for a in range(env.action_space.n)]
            G += rollout(state)

        # TODO: update all visited nodes in the tree
        # This updates values for the current node:
        while node is not None:
            node.visits += 1
            node.sum_value += G
            node = node.parent

        # Track the longest path length
        max_depth = max([child.depth for child in root.children] + [0])
        longest_path_length.append(max_depth)

    return longest_path_length


def main():
    env = gym.make("Taxi-v3")
    env.seed(0)  # use seed to make results better comparable
    # run the algorithm 10 times:
    rewards = []
    all_longest_path_lengths = []

    for i in range(10):
        env.reset()
        terminal = False
        root = Node()  # Initialize empty tree
        sum_reward = 0.0
        longest_path_length = []

        while not terminal:
            env.render()
            path_lengths = mcts(env, root)
            longest_path_length.extend(
                path_lengths
            )  # expand tree from root node using mcts
            values = values = [
                (c.sum_value / c.visits) if c.visits > 0 else -np.inf
                for c in root.children
            ]
            bestchild = root.children[np.argmax(values)]  # select the best child
            _, reward, terminal, _ = env.step(
                bestchild.action
            )  # perform action for child
            root = bestchild  # use the best child as next root
            root.parent = None
            sum_reward += reward

        rewards.append(sum_reward)
        all_longest_path_lengths.append(longest_path_length)
        print("finished run " + str(i + 1) + " with reward: " + str(sum_reward))

    print("mean reward: ", np.mean(rewards))

    # Plot the longest path lengths
    plt.figure(figsize=(12, 6))
    for i, path_lengths in enumerate(all_longest_path_lengths):
        plt.plot(path_lengths, label=f"Run {i+1}")
    plt.xlabel("Iteration")
    plt.ylabel("Longest Path Length")
    plt.title("Longest Path Length Over Iterations")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
