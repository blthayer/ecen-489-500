########################################################################
# IMPORTS
########################################################################
from constraint import Problem, AllDifferentConstraint
import json

########################################################################
# FUNCTIONS
########################################################################

########################################################################
# PROBLEM 1
########################################################################


def check_constraint_problem_1(a, b):
    """Ensure that transitioning from state a to b is valid.

    Returns True if valid, false otherwise.
    """
    # Cast the integers to binary.
    a_binary = '{0:03b}'.format(a)
    b_binary = '{0:03b}'.format(b)

    # Loop over bits and count the differences.
    # Note that rather than a looping implementation, an implementation
    # using binary operators may be more efficient.
    c = 0
    for idx in range(len(a_binary)):
        # If this bit is different
        if a_binary[idx] != b_binary[idx]:
            c += 1

        # Note we could break the loop here if c > 1...

    # Determine if the assignment is valid.
    if c > 1:
        return False
    else:
        return True


def test_check_constraint_problem_1():
    """Light-weight tests for check_constraint_problem"""
    assert check_constraint_problem_1(0, 1)
    assert not check_constraint_problem_1(0, 7)
    assert check_constraint_problem_1(0, 2)
    assert check_constraint_problem_1(7, 6)


def problem_1():
    """Function for performing work of problem 1."""
    # Initialize the problem.
    problem = Problem()

    # Add the variables (which all have the same domain)
    problem.addVariables(variables=["NM", "AR", "RR", "AL", "RL", "RA"],
                         domain=[0, 1, 2, 3, 4, 5, 6, 7])

    # All variables must be different.
    problem.addConstraint(AllDifferentConstraint())

    # Add individual constraints.
    # NM --> AR
    problem.addConstraint(check_constraint_problem_1, ["NM", "AR"])
    # AR --> RR
    problem.addConstraint(check_constraint_problem_1, ["AR", "RR"])
    # NM --> AL
    problem.addConstraint(check_constraint_problem_1, ["NM", "AL"])
    # AL --> RL
    problem.addConstraint(check_constraint_problem_1, ["AL", "RL"])
    # RR --> RA
    problem.addConstraint(check_constraint_problem_1, ["RR", "RA"])
    # RL --> RA
    problem.addConstraint(check_constraint_problem_1, ["RL", "RA"])
    # RA --> NM
    problem.addConstraint(check_constraint_problem_1, ["RA", "NM"])

    return problem

########################################################################
# MAIN
########################################################################


if __name__ == '__main__':
    pdf = 'homework_1.pdf'
    # Problem 1
    print('Problem 1:')
    print('See {} for part (a).'.format(pdf))
    test_check_constraint_problem_1()
    p1 = problem_1()
    # b)
    sol1 = p1.getSolution()
    print('(b):')
    print(json.dumps(sol1, indent=0))
    # c)
    all_sol1 = p1.getSolutions()
    print('(c):')
    print('There are {} valid solutions.'.format((len(all_sol1))))
    print('')
