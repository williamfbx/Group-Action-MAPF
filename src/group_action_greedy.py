import time as timer
import heapq
from collections import deque


def move(loc, dir):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
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
            
        # TODO: Flush current timestep if agents swapping locations

        
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
    result = [[start] for start in starts]
    
    for timestep in reversed(timestep_deque):
        moved_agents = set()
        for agent_id, end_location, start_location in timestep:
            result[agent_id].append(end_location)
            moved_agents.add(agent_id)

        for agent_id in range(num_of_agents):
            if agent_id not in moved_agents:
                result[agent_id].append(result[agent_id][-1])
    
    # DEBUG
    for agent, path in enumerate(result):
        print("Agent {}: Path: {}".format(agent, path))
        
    return result


def agents_at_goal(curr_agent_locations, goals):
    for agent_location, goal in zip(curr_agent_locations, goals):
        if agent_location != goal:
            return False
    return True


def find_legal_moves(agent_location, my_map):
    legal_moves = []
    for dir in range(4):
        next_location = move(agent_location, dir)
        if next_location[0] < 0 or next_location[0] >= len(my_map) \
                or next_location[1] < 0 or next_location[1] >= len(my_map[0]):
            continue
        if my_map[next_location[0]][next_location[1]]:
            continue
        legal_moves.append(next_location)
    return legal_moves


def rank_legal_moves(legal_moves, heuristic):
    ranked_legal_moves = sorted(legal_moves, key=lambda loc: heuristic[loc])
    return ranked_legal_moves


def check_collision_on_move(ranked_legal_moves, curr_agent_locations):
    collision_on_move = []
    for move in ranked_legal_moves:
        if move in curr_agent_locations:
            collision_on_move.append(True)
        else:
            collision_on_move.append(False)
    return collision_on_move


def find_clashing_agent(move, curr_agent_locations):
    return curr_agent_locations.index(move) if move in curr_agent_locations else None


def find_non_backtracking_move(agent_id, legal_moves, transposition_deque, curr_agent_locations):
    most_recent_start = None

    for moved_agent_id, (end_location, start_location) in transposition_deque:
        if moved_agent_id == agent_id:
            most_recent_start = start_location
            break

    if most_recent_start is None:
        non_backtracking_moves = legal_moves
    else:
        non_backtracking_moves = [loc for loc in legal_moves if loc != most_recent_start]

    collision_on_move = check_collision_on_move(non_backtracking_moves, curr_agent_locations)
    return [move for move, collision in zip(non_backtracking_moves, collision_on_move) if not collision]


def backtrack_most_recent_move(agent_id, transposition_deque, curr_agent_locations=None):
    target_index = None
    target_move = None

    for i, move_record in enumerate(transposition_deque):
        if move_record[0] == agent_id:
            target_index = i
            target_move = move_record
            break

    if target_move is None:
        return None

    _, (target_end, target_start) = target_move
    del transposition_deque[target_index]

    if curr_agent_locations is not None:
        curr_agent_locations[agent_id] = target_start

    occupied_by_reverts = {target_start}

    i = target_index - 1
    while i >= 0:
        dep_agent_id, (dep_end, dep_start) = transposition_deque[i]

        if dep_end in occupied_by_reverts:
            del transposition_deque[i]
            occupied_by_reverts.add(dep_start)

            if curr_agent_locations is not None:
                curr_agent_locations[dep_agent_id] = dep_start

        i -= 1

    return target_move


class GroupActionGreedySolver(object):
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
            
            # Break out when hitting max time
            if timer.time() - start_time > 5:
                print("Exceeded max time. Terminating search.")
                break

            agent_id = self.next_expansion_agent_deque.popleft()
            
            # Agent already at goal, move only if pushed by another agent
            if self.curr_agent_locations[agent_id] == self.goals[agent_id]:
                self.next_expansion_agent_deque.append(agent_id)
                continue
            
            # Agent not at goal, rank legal moves by heuristic
            legal_moves = find_legal_moves(self.curr_agent_locations[agent_id], self.my_map)
            ranked_legal_moves = rank_legal_moves(legal_moves, self.heuristics[agent_id])
            collision_on_move = check_collision_on_move(ranked_legal_moves, self.curr_agent_locations)
            
            # DEBUG
            print("Agent: {}, Current Location: {}, Legal Moves: {}, Ranked Legal Moves: {}, Collision on Move: {}".format(
                agent_id, self.curr_agent_locations[agent_id], legal_moves, ranked_legal_moves, collision_on_move))
            
            # Priority move is non-clashing
            if ranked_legal_moves and not collision_on_move[0]:
                self.transposition_deque.appendleft((agent_id, (ranked_legal_moves[0], self.curr_agent_locations[agent_id])))
                self.curr_agent_locations[agent_id] = ranked_legal_moves[0]
                self.next_expansion_agent_deque.append(agent_id)
                continue

            # Priority move is clashing
            clashing_agent_id = find_clashing_agent(ranked_legal_moves[0], self.curr_agent_locations)
            print("Agent {} collides with Agent {} at {}".format(agent_id, clashing_agent_id, ranked_legal_moves[0]))
            
            # Move clashing agent to non-backtracking cell
            clash_legal_moves = find_legal_moves(self.curr_agent_locations[clashing_agent_id], self.my_map)
            clash_legal_moves = [loc for loc in clash_legal_moves if loc != self.curr_agent_locations[agent_id]]
            clash_ranked_legal_moves = rank_legal_moves(clash_legal_moves, self.heuristics[clashing_agent_id])
            clash_move = find_non_backtracking_move(clashing_agent_id, clash_ranked_legal_moves, self.transposition_deque, self.curr_agent_locations)
            print("Clashing agent {} non-backtracking moves: {}".format(clashing_agent_id, clash_move))
            
            if clash_move:
                self.transposition_deque.appendleft((clashing_agent_id, (clash_move[0], self.curr_agent_locations[clashing_agent_id])))
                self.curr_agent_locations[clashing_agent_id] = clash_move[0]
                self.next_expansion_agent_deque.appendleft(agent_id)
                continue
            
            # Move own agent to non-backtracking cell
            self_moves = find_non_backtracking_move(agent_id, ranked_legal_moves[1:], self.transposition_deque, self.curr_agent_locations)
            print("Agent {} non-backtracking moves: {}".format(agent_id, self_moves))
            
            if self_moves:
                self.transposition_deque.appendleft((agent_id, (self_moves[0], self.curr_agent_locations[agent_id])))
                self.curr_agent_locations[agent_id] = self_moves[0]
                self.next_expansion_agent_deque.append(agent_id)
                continue
            
            # Backtrack own agent
            self_backtrack = backtrack_most_recent_move(agent_id, self.transposition_deque, self.curr_agent_locations)
            if self_backtrack:
                self.next_expansion_agent_deque.append(agent_id)
                print("Backtracking agent {} from {} to {}".format(agent_id, self_backtrack[1][0], self_backtrack[1][1]))
                continue
            
            # Backtrack clashing agent
            clash_backtrack = backtrack_most_recent_move(clashing_agent_id, self.transposition_deque, self.curr_agent_locations)
            if clash_backtrack:
                self.next_expansion_agent_deque.appendleft(agent_id)
                print("Backtracking clashing agent {} from {} to {}".format(clashing_agent_id, clash_backtrack[1][0], clash_backtrack[1][1]))
                continue
            
            # No possible moves, skip expansion
            print("Agent {} skipping expansion.".format(agent_id))
            self.next_expansion_agent_deque.append(agent_id)
        
        print(self.transposition_deque)
        result = convert_to_path(self.transposition_deque, self.num_of_agents, self.starts)
        self.CPU_time = timer.time() - start_time

        if agents_at_goal(self.curr_agent_locations, self.goals):
            print("\n Found a solution! \n")
        else:
            print("\n No solution found. \n")
            return None
        
        print("CPU time (s):    {:.2f}".format(self.CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(result)))

        return result
