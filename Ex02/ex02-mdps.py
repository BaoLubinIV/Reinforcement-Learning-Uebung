import gym
import numpy as np
from itertools import product

print(gym.__version__)
# Init environment
# Lets use a smaller 3x3 custom map for faster computations
custom_map3x3 = [
    "SFF",
    "FFF",
    "FHG",
]
env = gym.make("FrozenLake-v0", desc=custom_map3x3)

# TODO: Uncomment the following line to try the default map (4x4):
# env = gym.make("FrozenLake-v1")

# Uncomment the following lines for even larger maps:
# random_map = generate_random_map(size=5, p=0.8)
# env = gym.make("FrozenLake-v0", desc=random_map)

# Init some useful variables:
n_states = env.observation_space.n
n_actions = env.action_space.n

r = np.zeros(n_states)
# the r vector is zero everywhere except for the goal state (last state)
r[-1] = 1.0

gamma = 0.8


""" This is a helper function that returns the transition probability matrix P for a policy """


def trans_matrix_for_policy(policy):
    transitions = np.zeros((n_states, n_states))
    for s in range(n_states):
        probs = env.P[s][policy[s]]
        for el in probs:
            transitions[s, el[1]] += el[0]
    return transitions


""" This is a helper function that returns terminal states """


def terminals():
    terms = []
    for s in range(n_states):
        # terminal is when we end with probability 1 in terminal:
        if env.P[s][0][0][0] == 1.0 and env.P[s][0][0][3] == True:
            terms.append(s)
    return terms


def value_policy(policy):
    P = trans_matrix_for_policy(policy)
    # TODO: calculate and return v
    # (P, r and gamma already given)

    E = np.eye(n_states)
    v = np.dot(np.linalg.inv(E - gamma * P), r)
    return v


def bruteforce_policies():
    optimal_policies = []
    optimal_value = None

    # Set 0 action for terminal states
    term_states = terminals()
    action_options = [
        range(n_actions) if i not in term_states else [0] for i in range(n_states)
    ]

    # each policy is a tuple where indices represent states and values represent the chosen action
    # cartesian product of n_actions repeated n_states times
    all_policies = product(*action_options)

    for policy_tuple in all_policies:
        policy = np.array(policy_tuple, dtype=int)
        v = value_policy(policy)

        # calc value of policy
        current_value = np.sum(v)

        # update optimal value and policies
        if optimal_value is None or current_value > optimal_value:
            optimal_value = current_value
            optimal_policies = [policy.copy()]
        elif current_value == optimal_value:
            optimal_policies.append(policy.copy())

    print("Optimal value function total:", optimal_value)
    print("Number of optimal policies:", len(optimal_policies))
    print("Optimal policies:")
    for p in optimal_policies:
        print(p)

    return optimal_policies


def main():
    # print the environment
    env.reset()
    print("current environment: ")
    env.render()
    print("")

    # Here a policy is just an array with the action for a state as element
    policy_left = np.zeros(n_states, dtype=np.int64)  # 0 for all states
    policy_right = np.ones(n_states, dtype=np.int64) * 2  # 2 for all states

    # Value functions:
    print("Value function for policy_left (always going left):")
    print(value_policy(policy_left))
    print("Value function for policy_right (always going right):")
    print(value_policy(policy_right))

    optimalpolicies = bruteforce_policies()

    # This code can be used to "rollout" a policy in the environment:

    print("rollout policy:")
    maxiter = 100
    state = env.reset()
    print(state)
    for i in range(maxiter):
        new_state, reward, done, info = env.step(optimalpolicies[0][state])
        env.render()
        state = new_state
        if done:
            print("Finished episode")
            break


if __name__ == "__main__":
    main()
