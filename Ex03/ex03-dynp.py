import gym
import numpy as np

custom_map3x3 = [
    "SFF",
    "FFF",
    "FHG",
]
# env = gym.make("FrozenLake-v0", desc=custom_map3x3)

# Init environment
env = gym.make("FrozenLake-v0")

# you can set it to deterministic with:
# env = gym.make("FrozenLake-v0", is_slippery=False)

# If you want to try larger maps you can do this using:
# random_map = gym.envs.toy_text.frozen_lake.generate_random_map(size=5, p=0.8)
# env = gym.make("FrozenLake-v0", desc=random_map)
# Or:
# env = gym.make("FrozenLake-v0", map_name="8x8")


# Init some useful variables:
n_states = env.observation_space.n
n_actions = env.action_space.n


def print_policy(policy, env):
    """This is a helper function to print a nice policy representation from the policy"""
    moves = ["←", "↓", "→", "↑"]
    if not hasattr(env, "desc"):
        env = env.env
    dims = env.desc.shape
    pol = np.chararray(dims, unicode=True)
    pol[:] = " "
    for s in range(len(policy)):
        idx = np.unravel_index(s, dims)
        pol[idx] = moves[policy[s]]
        if env.desc[idx] in [b"H", b"G"]:
            pol[idx] = env.desc[idx]
    print("\n".join(["".join(["{:2}".format(item) for item in row]) for row in pol]))


def value_iteration():
    V_states = np.zeros(n_states)  # init values as zero
    theta = 1e-8
    gamma = 0.8
    # TODO: implement the value iteration algorithm
    # Hint: env.P[state][action] gives you tuples (p, n_state, r, is_terminal), which tell you the probability p that you end up in the next state n_state and receive reward r

    # TODO: After value iteration algorithm, obtain policy and return it
    # we update the policy in the loop directly (more efficient than updating it after the loop is finished)
    policy = np.zeros(n_states, dtype=int)

    iters = 0

    while True:
        delta = 0
        for s in range(n_states):
            v = V_states[s]
            # initialize with negative infinity to ensure that the first value is always higher
            max_v = -np.inf
            # loop for finding max_v over all actions
            for a in range(n_actions):
                q = 0
                # loop for calculating action value q over all possible next states
                for p, n_state, r, _ in env.P[s][a]:
                    q += p * (r + gamma * V_states[n_state])
                if q > max_v:
                    max_v = q
                    policy[s] = a  # update policy in the loop directly
            V_states[s] = max_v
            delta = max(delta, abs(v - V_states[s]))

        iters += 1
        if delta < theta:
            break

    print(
        "Value iteration converged after",
        iters,
        "iterations for theta =",
        theta,
        "and gamma =",
        gamma,
    )

    return policy


def main():
    # print the environment
    print("current environment: ")
    env.render()
    dims = env.desc.shape
    print()

    # run the value iteration
    policy = value_iteration()
    print("Computed policy: ")
    print(policy.reshape(dims))
    # if you computed a (working) policy, you can print it nicely with the following command:
    print_policy(policy, env)

    # This code can be used to "rollout" a policy in the environment:
    """print ("rollout policy:")
    maxiter = 100
    state = env.reset()
    for i in range(maxiter):
        new_state, reward, done, info = env.step(policy[state])
        env.render()
        state=new_state
        if done:
            print("Finished episode")
            break"""


if __name__ == "__main__":
    main()
