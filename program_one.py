########################################################################
# IMPORTS
########################################################################
import numpy as np
from heapq import heappush, heappop

########################################################################
# CONSTANTS
########################################################################
# Mapping of moves to symbols (North, East, South, West)
MOVE_SYMBOLS = ['^', '>', 'v', '<']

# Define arrays, which when added to a position move us to a new
# position in the direction of the variable name.
NORTH = np.array([-1, 0])
EAST = np.array([0, 1])
SOUTH = np.array([1, 0])
WEST = np.array([0, -1])

# Put the arrays together in order
MOVE_ARRAYS = (NORTH, EAST, SOUTH, WEST)

# Define the starting location - always start at origin.
START = np.array((0, 0))

########################################################################
# LINKED LIST CLASS
########################################################################


class LinkedListNode:
    """Simple doubly linked list implementation for use in A*"""

    def __init__(self, cost, position, back, direction):
        """

        :param cost: Cumulative cost to get to this position.
        :param position: Position as a numpy array. E.g.,
            np.array([1, 1]). Note this is zero-indexed.
        :param back: Pointer to node before.
        :param direction: Character from MOVE_SYMBOLS indicating the
            move it took to get from 'back' to 'position'
        """
        # Simply assign.
        self.cost = cost
        self.position = position
        self.back = back
        self.direction = direction

    def __repr__(self):
        s = 'Position: {}, '.format(self.position)
        if self.back is None:
            s += 'Back: None, '
        else:
            s += 'Back: {}, '.format(self.back.position)

        s += 'Cost: {} '.format(self.cost)

        return s

########################################################################
# FUNCTIONS
########################################################################


def _check_move(world, position, previous_position):
    """Helper function to check if a move to 'position' is valid.

    For information on inputs, check get_available_moves function.
    """
    # No going backward. NOTE: This shouldn't harm our expansion array -
    # going backward will always land you at the high-cost end of the
    # priority queue.
    if np.array_equal(position, previous_position):
        return False

    # Our world doesn't wrap, so no negative indices allowed.
    if any(position < 0):
        return False

    # Check the value of the position in the world
    try:
        position_value = world[position[0]][position[1]]
    except IndexError:
        # The position is out of bounds.
        return False

    # The position is valid if the value is 0, but not if it's a 1.
    if position_value == 1:
        return False
    elif position_value == 0:
        return True
    else:
        raise ValueError(
            'position_value evaluated to {}'.format(position_value))


def get_available_moves(world, position, previous_position):
    """Determine the set of all viable next moves.

    :param world: numpy array containing the world. 0 is a valid
        location to occupy, 1 is not.
    :param position: two element tuple indicating current position in
        the world.
    :param previous_position: two element tuple indicating previous
        position in the world.
    """
    # Evaluate possible moves. None will indicate the move is not valid.
    # ORDER: N, E, S, W
    moves = []
    for m in MOVE_ARRAYS:
        new_position = position + m
        if _check_move(world=world, position=new_position,
                       previous_position=previous_position):
            # Move is valid.
            moves.append(new_position)
        else:
            # Move is not valid.
            moves.append(None)

    # Done.
    return moves


def heuristic(position, goal):
    """Heuristic is Manhattan Distance (sum of number of rows + columns
    between location and goal)"""
    return np.sum(goal - position)


def a_star(world, expansion, path):
    """Run the A* algorithm."""
    # Track nodes that have been evaluated.
    visited_nodes = {tuple(START), }

    # Initialize loop:
    # We'll be using a Python's heapq as a priority queue, hence hq.
    hq = []
    root = LinkedListNode(cost=0, position=START, back=None, direction=None)
    # Get the initial possible moves.
    moves = get_available_moves(world=world, position=START,
                                previous_position=np.array((-1, -1)))
    # Initialize counter for hq (for tie-breaking)
    q_count = 0
    # Push all the possibilities onto the queue.
    for idx, m in enumerate(moves):
        if m is not None:
            # Create a linked list entry. We're giving the cost of 1,
            # because the first move always costs 1.
            linked_list = LinkedListNode(cost=1, position=m, back=root,
                                         direction=MOVE_SYMBOLS[idx])
            total_cost = (linked_list.cost
                          + heuristic(linked_list.position, GOAL))
            heappush(hq, (total_cost, q_count, linked_list))
            q_count += 1

    # Update the expansion array - we're expanding our starting point.
    expansion[START[0]][START[1]] = 0

    # t for time (also our iteration counter)
    t = 1

    # Loop.
    max_iter = 10000
    while (not np.array_equal(hq[0][2].position, GOAL)) and (t <= max_iter):
        # Pop the lowest cost list from the queue.
        # TODO: if hq is empty, we failed to find a path. In other
        #   words, a valid path doesn't exist. We likely don't need to
        #   worry about that for this simple homework assignment.
        queue_entry = heappop(hq)
        node = queue_entry[2]

        # If we've already visited this node, move along.
        # NOTE: It would probably be more efficient to use some sort of
        # unique identifier for each node, rather than a tuple of
        # position, but oh well.
        position_tuple = tuple(node.position)
        if position_tuple in visited_nodes:
            # Move to the next iteration of the loop.
            continue
        else:
            # Update our set.
            visited_nodes.add(position_tuple)

        # Update our expansion array.
        expansion[node.position[0]][node.position[1]] = t

        # Get possible moves from here.
        moves = get_available_moves(world=world, position=node.position,
                                    previous_position=node.back.position)

        # Loop over the moves and evaluate them.
        for idx, m in enumerate(moves):
            if m is not None:
                # Create linked list entry. Increment cost by 1.
                linked_list = LinkedListNode(cost=node.cost + 1, position=m,
                                             back=node,
                                             direction=MOVE_SYMBOLS[idx])
                total_cost = (linked_list.cost
                              + heuristic(linked_list.position, GOAL))
                heappush(hq, (total_cost, q_count, linked_list))
                q_count += 1

        t += 1

    if t >= max_iter:
        raise UserWarning('Algorithm terminated after {} '
                          'iterations.'.format(max_iter))

    # Map out the best path.
    best_entry = heappop(hq)
    # Start at the second to last node (we aren't moving from the goal).
    node = best_entry[2]

    while node.back is not None:
        # Update the path entry.
        path[node.back.position[0]][node.back.position[1]] = node.direction

        # Move to the next node.
        node = node.back

    # Done.
    return expansion, path


def main(world):
    """Function to avoid namespace conflicts. Simple wrapper to call
    a_star.
    """
    # Initialize expansion array.
    # TODO: This will certainly change.
    expansion_array = np.zeros_like(world_array) - 1

    # Initialize path array.
    path_array = world_array.astype('str')
    path_array[GOAL[0]][GOAL[1]] = '*'

    e, p = a_star(world=world, expansion=expansion_array,
                  path=path_array)

    return e, p

########################################################################
# Main program
########################################################################


if __name__ == '__main__':

    # Example world from the exercise prompt.
    given_world = [
        [0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0]]

    world_array = np.array(given_world)
    GOAL = np.array(world_array.shape) - 1

    e_array, p_array = main(world=world_array)

    print('Expansion array:')
    print(e_array)

    print('Path array:')
    print(p_array)

    # print('World array:')
    # print(world_array.astype(str))
