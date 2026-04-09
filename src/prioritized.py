import time as timer
from single_agent_planner import compute_heuristics, a_star, get_sum_of_cost


class PrioritizedPlanningSolver(object):
    """A planner that plans for each robot sequentially."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.CPU_time = 0

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def find_solution(self):
        """ Finds paths for all agents from their start locations to their goal locations."""

        start_time = timer.time()
        result = []
        constraints = []

        # for i in range(self.num_of_agents):  # Find path for each agent
        for i in [0, 4, 5, 9, 10, 19, 18, 3, 13, 14, 8, 17, 6, 11, 7, 1, 15, 16]:
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, constraints)
            if path is None:
                print("No solution")
                raise BaseException('No solutions')
                # return None
            result.append(path)

            ##############################
            # Task 1.3/1.4/2: Add constraints here
            #         Useful variables:
            #            * path contains the solution path of the current (i'th) agent, e.g., [(1,1),(1,2),(1,3)]
            #            * self.num_of_agents has the number of total agents
            #            * constraints: array of constraints to consider for future A* searches
            
            map_upper = len(self.my_map) * len(self.my_map[0])
            path_length_upper = 0
            # for k in range(i + 1):
            #     path_length_upper += len(result[k])
            # upper_bound = map_upper + path_length_upper
            # print("(DEBUG) Upper bound on time steps for agent {}: {}".format(i, upper_bound))
            upper_bound = map_upper
            
            goal_loc = path[-1]
            arrival_time = len(path) - 1
            
            for j in range(i + 1, self.num_of_agents):
                for t in range(len(path)):
                    constraints.append({'agent': j,
                                        'loc': [path[t]],
                                        'timestep': t})
                    
                for t in range(1, len(path)):
                    constraints.append({'agent': j,
                                        'loc': [path[t], path[t - 1]],
                                        'timestep': t})
                    
                for t in range(arrival_time, upper_bound + 1):
                    constraints.append({'agent': j,
                                        'loc': [goal_loc],
                                        'timestep': t})

            # print("(DEBUG) Constraints after planning for agent {}: {}".format(i, constraints))
            ##############################

        self.CPU_time = timer.time() - start_time

        print("\n Found a solution! \n")
        print("CPU time (s):    {:.2f}".format(self.CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(result)))
        print(result)
        return result
