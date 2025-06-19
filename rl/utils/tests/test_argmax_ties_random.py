import numpy as np
import pytest
from rl.utils.general import argmax_ties_random

def test_should_throw_if_q_values_has_more_than_1_dimension():
    q_values = np.array([[1]])
    assert(q_values.ndim > 1)

    with pytest.raises(ValueError):
        argmax_ties_random(q_values=q_values)

def test_should_throw_if_q_values_is_empty():
    with pytest.raises(ValueError):
        argmax_ties_random(q_values=np.array([]))

def test_should_return_index_of_max_value_of_single_value_array():
    assert(0 == argmax_ties_random(q_values=np.array([42])))

def test_should_return_index_of_max_value_of_multi_value_array():
    assert(1 == argmax_ties_random(q_values=np.array([1,2])))

def test_should_return_index_of_max_value_of_multi_value_array_with_repeated_max_values():
    q_values = np.array([1,1,2,10,3,4,3,3,4,5,2,10,9,10,10,10,1,10,9,9,9,8,10])
    correct_answers = [3,11,13,14,15,17,22]
    first_answer = argmax_ties_random(q_values=q_values)
    assert(first_answer in correct_answers)

    found = False
    for _ in range(10):
        second_answer = argmax_ties_random(q_values=q_values)
        assert(second_answer in correct_answers)
        found = first_answer != second_answer
        if found:
            break

    assert(found)


