#!/usr/bin/env python3

import pickle
# NO ADDITIONAL IMPORTS ALLOWED!

# Note that part of your checkoff grade for this lab will be based on the
# style/clarity of your code.  As you are working through the lab, be on the
# lookout for things that would be made clearer by comments/docstrings, and for
# opportunities to rearrange aspects of your code to avoid repetition (for
# example, by introducing helper functions).


def transform_data(raw_data):
    return raw_data


def acted_together(data, actor_id_1, actor_id_2):
    raise NotImplementedError("Implement me!")


def actors_with_bacon_number(data, n):
    raise NotImplementedError("Implement me!")


def bacon_path(data, actor_id):
    raise NotImplementedError("Implement me!")


def actor_to_actor_path(data, actor_id_1, actor_id_2):
    raise NotImplementedError("Implement me!")


def actor_path(data, actor_id_1, goal_test_function):
    raise NotImplementedError("Implement me!")


def actors_connecting_films(data, film1, film2):
    raise NotImplementedError("Implement me!")


if __name__ == '__main__':
    with open('resources/small.pickle', 'rb') as f:
        smalldb = pickle.load(f)

    # additional code here will be run only when lab.py is invoked directly
    # (not when imported from test.py), so this is a good place to put code
    # used, for example, to generate the results for the online questions.
    pass
