import time as timer
import heapq
from collections import deque


def move(loc, dir):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]


def compute_heuristics(my_map, goal):
    # Use Dijkstra to build a shortest-path tree rooted at the goal location
    open_list = []
    closed_list = dict()
    root = {'loc': goal, 'cost': 0}
    heapq.heappush(open_list, (root['cost'], goal, root))
    closed_list[goal] = root
    while len(open_list) > 0:
        (cost, loc, curr) = heapq.heappop(open_list)
        for dir in range(4):
            child_loc = move(loc, dir)
            child_cost = cost + 1
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
                    or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
                continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            child = {'loc': child_loc, 'cost': child_cost}
            if child_loc in closed_list:
                existing_node = closed_list[child_loc]
                if existing_node['cost'] > child_cost:
                    closed_list[child_loc] = child
                    heapq.heappush(open_list, (child_cost, child_loc, child))
            else:
                closed_list[child_loc] = child
                heapq.heappush(open_list, (child_cost, child_loc, child))

    # Build the heuristics table
    h_values = dict()
    for loc, node in closed_list.items():
        h_values[loc] = node['cost']
    return h_values


def get_sum_of_cost(paths):
    rst = 0
    if paths is None:
        return -1
    for path in paths:
        rst += len(path) - 1
    return rst


def convert_to_path(transposition_deque, num_of_agents, starts):
    
    # Split up sequence of transpositions into timesteps
    timestep_deque = deque()
    agent_set = set()
    end_location_set = set()
    single_timestep = []
    
    while transposition_deque:
        agent_id, (end_location, start_location) = transposition_deque.pop()

        # Flush current timestep if duplicate agent
        if agent_id in agent_set:
            timestep_deque.appendleft(single_timestep)
            single_timestep = []
            agent_set = set()
            end_location_set = set()
        
        # Flush current timestep if duplicate end location
        if end_location in end_location_set:
            timestep_deque.appendleft(single_timestep)
            single_timestep = []
            agent_set = set()
            end_location_set = set()
        
        single_timestep.append((agent_id, end_location, start_location))
        agent_set.add(agent_id)
        end_location_set.add(end_location)
        
    if single_timestep:
        timestep_deque.appendleft(single_timestep)
        
    # DEBUG 
    for i, timestep in enumerate(reversed(timestep_deque)):
        print("Timestep {}:".format(i))
        for agent_id, end_location, start_location in timestep:
            print("Agent {}: {} -> {}".format(agent_id, start_location, end_location))
    
    # Convert sequence of transpositions into paths
    # TODO: Handle case where agent does not move at a timestep
    
    result = [[start] for start in starts]
    
    for i, timestep in enumerate(reversed(timestep_deque)):
        for agent_id, end_location, start_location in timestep:
            result[agent_id].append(end_location)
    
    # DEBUG
    for path in result:
        print("Path: {}".format(path))
        
    return result


def agents_at_goal(curr_agent_locations, goals):
    for agent_location, goal in zip(curr_agent_locations, goals):
        if agent_location != goal:
            return False
    return True


def find_legal_moves(agent_location, my_map):
    legal_moves = []
    for dir in range(5):
        next_location = move(agent_location, dir)
        if next_location[0] < 0 or next_location[0] >= len(my_map) \
                or next_location[1] < 0 or next_location[1] >= len(my_map[0]):
            continue
        if my_map[next_location[0]][next_location[1]]:
            continue
        legal_moves.append(next_location)
    return legal_moves


class GroupActionSolver(object):
    """A planner that plans for each robot via group action."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = list(starts)
        self.goals = goals
        self.num_of_agents = len(goals)
        
        self.transposition_deque = deque()
        self.curr_agent_locations = list(starts)
        self.next_expansion_agent_deque = deque([i for i in range(self.num_of_agents)])

        self.CPU_time = 0
        
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))
        # print("Heuristics: {}".format(self.heuristics))


    def find_solution(self):
        """ Finds paths for all agents from their start locations to their goal locations."""

        start_time = timer.time()
        
        while not agents_at_goal(self.curr_agent_locations, self.goals):

            next_agent_id = self.next_expansion_agent_deque.popleft()
            
            legal_moves = find_legal_moves(self.curr_agent_locations[next_agent_id], self.my_map)
            best_move = None
            best_h_value = float('inf')
            
            for legal_move in legal_moves:
                h_value = self.heuristics[next_agent_id][legal_move]
                if h_value < best_h_value:
                    best_h_value = h_value
                    best_move = legal_move

            next_move = (next_agent_id, (best_move, self.curr_agent_locations[next_agent_id]))
            # print("Expanded agent {}: {} -> {}".format(next_agent_id, next_move[1][1], next_move[1][0]))
            
            self.transposition_deque.appendleft(next_move)
            self.curr_agent_locations[next_agent_id] = next_move[1][0]
            self.next_expansion_agent_deque.append(next_agent_id)
            
            # DEBUG
            # Break out when hitting max time
            if timer.time() - start_time > 1:
                print("Exceeded max time. No solution found.")
                return None
        
        print(self.transposition_deque)
        result = convert_to_path(self.transposition_deque, self.num_of_agents, self.starts)
        self.CPU_time = timer.time() - start_time

        print("\n Found a solution! \n")
        print("CPU time (s):    {:.2f}".format(self.CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(result)))

        return result
