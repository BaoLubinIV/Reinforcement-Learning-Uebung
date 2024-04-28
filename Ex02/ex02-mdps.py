import gym
import numpy as np
print(gym.__version__)
# Init environment
# Lets use a smaller 3x3 custom map for faster computations
custom_map3x3 = [
    'SFF',
    'FFF',
    'FHG',
]
env = gym.make("FrozenLake-v1", desc=custom_map3x3)

# TODO: Uncomment the following line to try the default map (4x4):
#env = gym.make("FrozenLake-v1")

# Uncomment the following lines for even larger maps:
#random_map = generate_random_map(size=5, p=0.8)
#env = gym.make("FrozenLake-v0", desc=random_map)

# Init some useful variables:
n_states = env.observation_space.n
n_actions = env.action_space.n

r = np.zeros(n_states) # the r vector is zero everywhere except for the goal state (last state)
r[-1] = 1.

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
    v = np.dot(np.linalg.inv(E-gamma*P),r)
    return v

#function to generate all the policies
def decimal_to_base4(decimal):
    if decimal == 0:
        return '0'
    
    result = ''
    while decimal > 0:
        remainder = decimal % 4
        result = str(remainder) + result
        decimal //= 4
    
    return result

def bruteforce_policies():
    terms = terminals()
    print(terms)
    optimalpolicies = []

    policy = np.zeros(n_states, dtype=np.int64)  # in the discrete case a policy is just an array with action = policy[state]
    optimalvalue = np.zeros(n_states)
    n_non_term_s = n_states-len(terms)
    # TODO: implement code that tries all possible policies, calculates the values using def value_policy().
    for i in range(0, 4**n_non_term_s):      
        generator = decimal_to_base4(i).zfill(n_non_term_s)        #generate all policies
        
        index = 0
        policy = np.zeros(n_states, dtype=np.int64)
        for s in range (n_states):
            if s not in terms:
            #all states besides terms
                policy[s]=generator[index] 
                index += 1
            else:
                policy[s]=0
        print("Policy:",policy)
        
        v = value_policy(policy)
        print("Value",v)
        
        #Find the optimal values and the optimal policies to answer the exercise questions.             
        if np.sum(v) > np.sum(optimalvalue):
            optimalvalue = v
            optimalpolicies = []
            optimalpolicies.append(policy)
        elif np.sum(v) == np.sum(optimalvalue):
            optimalpolicies.append(policy)
                

    print("Optimal value function:")
    print(optimalvalue)
    print("number optimal policies:")
    print(len(optimalpolicies))
    print("optimal policies:")
    print(np.array(optimalpolicies))
    return optimalpolicies


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
'''
    print("rollout policy:")
    maxiter = 100
    state = env.reset()
    print(state)
    for i in range(maxiter):
        new_state, reward, done, info = env.step(optimalpolicies[0][state])
        env.render()
        state=new_state
        if done:
            print("Finished episode")
            break
   '''     

if __name__ == "__main__":
    main()

