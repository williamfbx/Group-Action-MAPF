import time as timer
import heapq
import random
import copy
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost


def detect_first_collision_for_path_pair(path1, path2):
    ##############################
    # Task 2.1: Return the first collision that occurs between two robot paths (or None if there is no collision)
    #           There are two types of collisions: vertex collision and edge collision.
    #           A vertex collision occurs if both robots occupy the same location at the same timestep
    #           An edge collision occurs if the robots swap their location at the same timestep.
    #           You should use "get_location(path, t)" to get the location of a robot at time t.

    T = max(len(path1), len(path2))
    
    for t in range(T):
        p1_t = get_location(path1, t)
        p2_t = get_location(path2, t)
        
        # Vertex collision
        if p1_t == p2_t:
            return {'loc': [p1_t], 'timestep': t}
    
        # Edge collision
        if t > 0:
            p1_prev = get_location(path1, t - 1)
            p2_prev = get_location(path2, t - 1)

            if p1_prev == p2_t and p2_prev == p1_t:
                return {'loc': [p1_prev, p1_t], 'timestep': t}
        
    return None


def detect_collisions_among_all_paths(paths):
    ##############################
    # Task 2.1: Return a list of first collisions between all robot pairs.
    #           A collision can be represented as dictionary that contains the id of the two robots, the vertex or edge
    #           causing the collision, and the timestep at which the collision occurred.
    #           You should use your detect_collision function to find a collision between two robots.

    collisions = []
    num_agents = len(paths)
    
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            collision = detect_first_collision_for_path_pair(paths[i], paths[j])
            if collision is not None:
                collisions.append({'a1': i,
                                   'a2': j,
                                   'loc': collision['loc'],
                                   'timestep': collision['timestep']})
    
    return collisions


def standard_splitting(collision):
    ##############################
    # Task 2.2: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint prevents the first agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the second agent to be at the
    #                            specified location at the specified timestep.
    #           Edge collision: the first constraint prevents the first agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the second agent to traverse the
    #                          specified edge at the specified timestep

    a1 = collision['a1']
    a2 = collision['a2']
    t = collision['timestep']
    loc = collision['loc']
    
    # Vertex collision
    if len(loc) == 1:
        return [
            {'agent': a1, 'loc': [loc[0]], 'timestep': t},
            {'agent': a2, 'loc': [loc[0]], 'timestep': t}]
        
    # Edge collision
    if len(loc) == 2:
        return [
            {'agent': a1, 'loc': [loc[0], loc[1]], 'timestep': t},
            {'agent': a2, 'loc': [loc[1], loc[0]], 'timestep': t}]
        
    raise ValueError(f"Invalid collision: {collision}")


class CBSSolver(object):
    """The high-level search of CBS."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        print("Expand node {}".format(id))
        self.num_of_expanded += 1
        return node

    def find_solution(self):
        """ Finds paths for all agents from their start locations to their goal locations

        """

        self.start_time = timer.time()

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths
        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}
        for i in range(self.num_of_agents):  # Find initial path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions_among_all_paths(root['paths'])
        self.push_node(root)

        # Task 2.1: Testing
        print(root['collisions'])

        # Task 2.2: Testing
        for collision in root['collisions']:
            print(standard_splitting(collision))

        ##############################
        # Task 2.3: High-Level Search
        #           Repeat the following as long as the open list is not empty:
        #             1. Get the next node from the open list (you can use self.pop_node()
        #             2. If this node has no collision, return solution
        #             3. Otherwise, choose the first collision and convert to a list of constraints (using your
        #                standard_splitting function). Add a new child node to your open list for each constraint
        #           Ensure to create a copy of any objects that your child nodes might inherit
        
        while len(self.open_list) > 0:
            P = self.pop_node()
            
            if len(P['collisions']) == 0:
                self.print_results(P)
                return P['paths']
            
            collision = P['collisions'][0]
            constraints = standard_splitting(collision)
            
            for constraint in constraints:
                Q = copy.deepcopy(P)
                Q['constraints'].append(constraint)
                ai = constraint['agent']
                
                new_path = a_star(self.my_map,
                                  self.starts[ai],
                                  self.goals[ai],
                                  self.heuristics[ai],
                                  ai,
                                  Q['constraints'])
                
                if new_path is None:
                    continue
                
                Q['paths'][ai] = new_path
                Q['collisions'] = detect_collisions_among_all_paths(Q['paths'])
                Q['cost'] = get_sum_of_cost(Q['paths'])
                
                self.push_node(Q)
                
        raise BaseException('No solutions')

        # These are just to print debug output - can be modified once you implement the high-level search
        # self.print_results(root)
        # return root['paths']

    def print_results(self, node):
        print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))
