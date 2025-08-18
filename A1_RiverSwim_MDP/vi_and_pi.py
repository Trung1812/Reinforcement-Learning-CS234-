### MDP Value Iteration and Policy Iteration

import numpy as np
from riverswim import RiverSwim

np.set_printoptions(precision=3)

def bellman_backup(state, action, R, T, gamma, V:np.array):
    """
    Perform a single Bellman backup. So the policy for this is deterministic,
    it just given the action, not the probability distribution of the action

    Parameters
    ----------
    state: int
    action: int
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    V: np.array (num_states)

    Returns
    -------
    backup_val: float
    """
    backup_val = R[state][action] + gamma * np.matmul(T[state], V)[action]
    V[state] = backup_val
    return backup_val

def policy_evaluation(policy, R, T, gamma, tol=1e-3):
    """
    Compute the value function induced by a given policy for the input MDP.
    The policy is determined by the state and is deterministic. My assumption
    is verified.
    Parameters
    ----------
    policy: np.array (num_states)
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    tol: float

    Returns
    -------
    value_function: np.array (num_states)
    """
    num_states, num_actions = R.shape
    next_value_function = value_function = np.zeros(num_states)
    while True:
        next_value_function = value_function
        value_function = (R + gamma * np.matmul(T, value_function))[np.arange(num_states),policy]
        if np.max(np.abs(next_value_function-value_function)) < tol:
            break
    return value_function


def policy_improvement(policy, R, T, V_policy, gamma):
    """
    Given the value function induced by a given policy, perform policy improvement
    Parameters
    ----------
    policy: np.array (num_states)
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    V_policy: np.array (num_states)
    gamma: float

    Returns
    -------
    new_policy: np.array (num_states)
    """
    print(f"Policy Improvement: {policy}")
    new_policy = np.argmax(R + gamma*np.matmul(T, V_policy), axis=-1)
    
    return new_policy, (new_policy == policy).all()


def policy_iteration(R, T, gamma, tol=1e-3):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)

    Returns
    -------
    V_policy: np.array (num_states)
    policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    V_policy = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    
    
    stop = False
    while not stop:
        V_policy = policy_evaluation(policy, R, T, gamma, tol)
        policy, stop = policy_improvement(policy, R, T, V_policy, gamma)
    return V_policy, policy


def value_iteration(R, T, gamma, tol=1e-3):
    """Runs value iteration.
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)

    Returns
    -------
    value_function: np.array (num_states)
    policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    next_value_function = value_function = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    
    while True:
        next_value_function = value_function
        ret = R + gamma * np.matmul(T, value_function)
        value_function = np.max(ret, axis=-1)
        if np.max(np.abs(next_value_function-value_function)) < tol:
            policy = np.argmax(ret, axis=-1)
            break

    return value_function, policy


# Edit below to run policy and value iteration on different configurations
# You may change the parameters in the functions below
if __name__ == "__main__":
    SEED = 1234

    RIVER_CURRENT = 'WEAK'
    # RIVER_CURRENT = 'STRONG'
    assert RIVER_CURRENT in ['WEAK', 'MEDIUM', 'STRONG']
    env = RiverSwim(RIVER_CURRENT, SEED)

    R, T = env.get_model()
    discount_factor = 0.99
    # discount_factor = 0.67
    
    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)

    V_pi, policy_pi = policy_iteration(R, T, gamma=discount_factor, tol=1e-3)
    print(V_pi)
    print([['L', 'R'][a] for a in policy_pi])
    
    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)
    
    V_vi, policy_vi = value_iteration(R, T, gamma=discount_factor, tol=1e-3)
    print(V_vi)
    print([['L', 'R'][a] for a in policy_vi])
 
    V = bellman_backup(1, 1, R, T, discount_factor, V_pi)
    print(f'V={V}')
