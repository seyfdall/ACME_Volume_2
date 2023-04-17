# policy_iteration.py
"""Volume 2: Policy Function Iteration.
<Name> Dallin Seyfried
<Class> 001
<Date> 04/13/2023
"""

import numpy as np
import gym
from gym import wrappers

# Intialize P for test example
#Left =0
#Down = 1
#Right = 2
#Up= 3

P = {s : {a: [] for a in range(4)} for s in range(4)}
P[0][0] = [(0, 0, 0, False)]
P[0][1] = [(1, 2, -1, False)]
P[0][2] = [(1, 1, 0, False)]
P[0][3] = [(0, 0, 0, False)]
P[1][0] = [(1, 0, -1, False)]
P[1][1] = [(1, 3, 1, True)]
P[1][2] = [(0, 0, 0, False)]
P[1][3] = [(0, 0, 0, False)]
P[2][0] = [(0, 0, 0, False)]
P[2][1] = [(0, 0, 0, False)]
P[2][2] = [(1, 3, 1, True)]
P[2][3] = [(1, 0, 0, False)]
P[3][0] = [(0, 0, 0, True)]
P[3][1] = [(0, 0, 0, True)]
P[3][2] = [(0, 0, 0, True)]
P[3][3] = [(0, 0, 0, True)]


# Problem 1
def value_iteration(P, nS ,nA, beta = 1, tol=1e-8, maxiter=3000):
    """Perform Value Iteration according to the Bellman optimality principle.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.
        maxiter (int): The maximum number of iterations.

    Returns:
       v (ndarray): The discrete values for the true value function.
       n (int): number of iterations
    """
    # Initialize value function vectors
    V_olds = np.zeros(nS)
    V_new = np.zeros(nS)
    k = 0

    for k in range(maxiter):
        for s in range(nS):
            sa_vector = np.zeros(nA)
            for a in range(nA):
                for tuple_info in P[s][a]:
                    # tuple_info is a tuple of (probability, next state, reward, done)
                    p, s_, u, _ = tuple_info
                    # sums up the possible end states and rewards with given action
                    sa_vector[a] += (p * (u + beta * V_olds[s_]))
                # Add the max value to the value function
                V_new[s] = np.max(sa_vector)
        # Check to see if vectors match, update and repeat if not
        if np.linalg.norm(V_new - V_olds) < tol:
            break
        V_olds = V_new.copy()

    return V_new, k + 1


# Test Problem 1
def test_value_iteration():
    print('\n')
    print(value_iteration(P=P, nS=4, nA=4, beta=1))


# Problem 2
def extract_policy(P, nS, nA, v, beta = 1.0):
    """Returns the optimal policy vector for value function v

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        v (ndarray): The value function values - optimal V* from value iteration.
        beta (float): The discount rate (between 0 and 1).

    Returns:
        policy (ndarray): which direction to move in from each square.
    """
    # Initialize policy vector
    policy = np.zeros(nS)
    for s in range(nS):
        sa_vector = np.zeros(nA)
        for a in range(nA):
            for tuple_info in P[s][a]:
                # tuple_info is a tuple of (probability, next state, reward, done)
                p, s_, u, _ = tuple_info
                # sums up the possible end states and rewards with given action
                sa_vector[a] += (p * (u + beta * v[s_]))
            # Add the max value to the value function
            policy[s] = np.argmax(sa_vector)

    return policy


# Test Problem 2
def test_extract_policy():
    print('\n')
    v = value_iteration(P=P, nS=4, nA=4)[0]
    print(extract_policy(P=P, nS=4, nA=4, v=v))


# Problem 3
def compute_policy_v(P, nS, nA, policy, beta=1.0, tol=1e-8):
    """Computes the value function for a policy using policy evaluation.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        policy (ndarray): The policy to estimate the value function.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.

    Returns:
        v (ndarray): The discrete values for the true value function.
    """
    # Define values
    V_olds = np.zeros(nS)
    V_new = np.ones(nS)

    # Iterate until Values found
    while np.linalg.norm(V_new - V_olds) > tol:
        V_olds = V_new.copy()
        for s in range(nS):
            # Optimal action already known so
            # maximizing for loop not needed
            tuple_info = P[s][policy[s]][0]
            p, s_, u, _ = tuple_info
            V_new[s] = (p * (u + beta * V_olds[s_]))

    return V_new


# Test Problem 3
def test_compute_policy():
    print('\n')
    v = value_iteration(P=P, nS=4, nA=4)[0]
    print(f'Original Value: {v}\n')
    policy = extract_policy(P=P, nS=4, nA=4, v=v)
    print(f'Recomputed Value: {compute_policy_v(P=P, nS=4, nA=4, policy=policy, beta=1.0, tol=1e-8)}')


# Problem 4
def policy_iteration(P, nS, nA, beta=1, tol=1e-8, maxiter=200):
    """Perform Policy Iteration according to the Bellman optimality principle.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.
        maxiter (int): The maximum number of iterations.

    Returns:
    	v (ndarray): The discrete values for the true value function
        policy (ndarray): which direction to move in each square.
        n (int): number of iterations
    """
    # Initialize policy with common choice
    p_new = np.ones(nS)
    v_new = np.ones(nS)
    k = 0

    # Iterate up to maxiter times to find optimal policy
    for k in range(maxiter):
        # Evaluate policy and improve upon it
        v_k = compute_policy_v(P, nS, nA, p_new, beta, tol)
        p_k = extract_policy(P, nS, nA, v_k, beta)

        # If policy is close enough return
        if np.linalg.norm(p_k - p_new) < tol:
            p_new = p_k
            v_new = v_k
            break

        p_new = p_k.copy()

    return v_new, p_new, k


# Test Problem 4
def test_policy_iteration():
    print('\n')
    print(f'Value Iteration: {value_iteration(P=P, nS=4, nA=4, beta=1)}')
    print('\n')
    v = value_iteration(P=P, nS=4, nA=4)[0]
    print(f'Extract Policy: {extract_policy(P=P, nS=4, nA=4, v=v)}')
    v_new, p_new = policy_iteration(P=P, nS=4, nA=4)
    print('\n')
    print(f'v_new {v_new}\n')
    print(f'p_new {p_new}\n')


# Problem 5 and 6
def frozen_lake(basic_case=True, M=1000, render=False):
    """ Finds the optimal policy to solve the FrozenLake problem

    Parameters:
    basic_case (boolean): True for 4x4 and False for 8x8 environemtns.
    M (int): The number of times to run the simulation using problem 6.
    render (boolean): Whether to draw the environment.

    Returns:
    vi_policy (ndarray): The optimal policy for value iteration.
    vi_total_rewards (float): The mean expected value for following the value iteration optimal policy.
    pi_value_func (ndarray): The maximum value function for the optimal policy from policy iteration.
    pi_policy (ndarray): The optimal policy for policy iteration.
    pi_total_rewards (float): The mean expected value for following the policy iteration optimal policy.
    """
    # Make environment for 4x4 scenario or 8x8 as appropriate
    env_name = 'FrozenLake8x8-v1'
    if basic_case:
        env_name = 'FrozenLake-v1'
    env = gym.make(env_name).env

    # Put environment in starting state
    obs = env.reset()

    # Find number of states and actions
    number_of_states = env.observation_space.n
    number_of_actions = env.action_space.n

    # Get the dictionary with all the states and actions
    dictionary_P = env.env.env.P

    # Compute policy and rewards via value iteration
    vi_policy = extract_policy(dictionary_P, number_of_states, number_of_actions,
                               value_iteration(dictionary_P, number_of_states, number_of_actions)[0])
    vi_total_rewards = np.mean(compute_policy_v(dictionary_P, number_of_states, number_of_actions, vi_policy))

    # Compute value, policy, and rewards via policy iteration
    pi_value_func, pi_policy, k = policy_iteration(dictionary_P, number_of_states, number_of_actions)
    pi_total_rewards = np.mean(pi_value_func)

    env.close()

    return vi_policy, vi_total_rewards, pi_value_func, pi_policy, pi_total_rewards


# Test Problem 5
def test_frozen_lake():
    print(frozen_lake(True, 1))


# Problem 6
def run_simulation(env, policy, render=True, beta = 1.0):
    """ Evaluates policy by using it to run a simulation and calculate the reward.

    Parameters:
    env (gym environment): The gym environment.
    policy (ndarray): The policy used to simulate.
    beta float: The discount factor.
    render (boolean): Whether to draw the environment.

    Returns:
    total reward (float): Value of the total reward received under policy.
    """
    raise NotImplementedError("Problem 6 Incomplete")


# Test Problem 6
def test_run_simulation():
    pass
