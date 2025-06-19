from rl.algorithms.bandits.epsilon_greedy import EpsilonGreedy
from rl.environment.bandits.k_armed_bandit import KArmedTestbed

import numpy as np

environment = KArmedTestbed(num_runs=1000, k=10)
epsilon_greedy = EpsilonGreedy(env=environment, epsilon=0)
epsilon_greedy.q_values = np.array([3, 3, 2, 100, 100, 30, 40, 100, 44, 100])
max_value_indices = [3, 4, 7, 9]
non_max_value_indices = [1, 2, 5, 6, 8]


def test_should_always_exploit_with_epsilon_of_zero():
    epsilon_greedy.epsilon = 0
    first_pick = epsilon_greedy.act()
    assert first_pick in max_value_indices

    found = False
    for _ in range(100):
        second_pick = epsilon_greedy.act()
        assert second_pick in max_value_indices
        if first_pick != second_pick:
            found = True

    assert found


def test_should_aways_explore_when_epsilon_is_non_zero():
    epsilon_greedy.epsilon = 1
    found = False
    for _ in range(epsilon_greedy.num_actions * 2):
        found = epsilon_greedy.act() in non_max_value_indices
        if found:
            break

    assert found
