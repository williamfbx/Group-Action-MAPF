import time as timer
import numpy as np
from collections import deque
from treelib import Tree
from poisson_solver import solve_poisson
from plotter import plot_solution


def move(loc, dir):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]


def get_sum_of_cost(paths):
    rst = 0
    if paths is None:
        return -1
    for path in paths:
        rst += len(path) - 1
    return rst


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


def is_free(r, c, my_map, max_row, max_col):
    return 0 <= r < max_row and 0 <= c < max_col and not my_map[r][c]


def compute_downhill_gradient_at_cell(cell, potential_field, my_map):
    row, col = cell
    max_row, max_col = potential_field.shape

    up_free = is_free(row - 1, col, my_map, max_row, max_col)
    down_free = is_free(row + 1, col, my_map, max_row, max_col)
    left_free = is_free(row, col - 1, my_map, max_row, max_col)
    right_free = is_free(row, col + 1, my_map, max_row, max_col)

    if up_free and down_free:
        grad_row = (potential_field[row + 1, col] - potential_field[row - 1, col]) / 2.0
    elif down_free:
        grad_row = potential_field[row + 1, col] - potential_field[row, col]
    elif up_free:
        grad_row = potential_field[row, col] - potential_field[row - 1, col]
    else:
        grad_row = 0.0

    if left_free and right_free:
        grad_col = (potential_field[row, col + 1] - potential_field[row, col - 1]) / 2.0
    elif right_free:
        grad_col = potential_field[row, col + 1] - potential_field[row, col]
    elif left_free:
        grad_col = potential_field[row, col] - potential_field[row, col - 1]
    else:
        grad_col = 0.0

    return np.array([-grad_row, -grad_col])


def rank_legal_moves(agent_location, legal_moves, potential_field, my_map):
    row, col = agent_location
    current_phi = float(potential_field[row, col])

    ranked_legal_moves = sorted(
        legal_moves,
        key=lambda loc: -float(current_phi - potential_field[loc]),
    )
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


def agents_at_goal(curr_agent_locations, goals):
    return set(curr_agent_locations) == set(goals)


def calculate_tree_value(curr_agent_locations, active_agents, phi, num_of_agents, my_map):
    num_agents_completed = num_of_agents - len(active_agents)
    source_strength = 100.0
    
    grad_magnitude_sum = 0.0
    for agent_id in active_agents:
        agent_loc = curr_agent_locations[agent_id]
        grad = compute_downhill_gradient_at_cell(agent_loc, phi, my_map)
        grad_magnitude = np.linalg.norm(grad)
        grad_magnitude_sum += grad_magnitude
    
    return (num_agents_completed * 2 * source_strength) + grad_magnitude_sum


def calculate_ucb(node, parent_visits, exploration_constant=1.41):

    visits = node.data.get('visits', 1)
    value = node.data.get('value', 0)

    exploitation = value / visits
    exploration = exploration_constant * np.sqrt(np.log(parent_visits + 1) / visits)
    ucb = exploitation + exploration
    
    return ucb


def get_transposition_history(tree, node_id):
    # Collect node ids from root down to node_id
    path_to_node = []
    current_id = node_id
    while current_id is not None:
        path_to_node.append(current_id)
        parent = tree.parent(current_id)
        current_id = parent.identifier if parent is not None else None
    path_to_node.reverse()

    # Index 0 is newest due to appendleft. Reverse each list for chronological order
    history = []
    for nid in path_to_node:
        node = tree.get_node(nid)
        transpositions = node.data['transpositions']
        if transpositions is not None:
            history.extend(reversed(transpositions))

    return history


def convert_to_path(tree, node_id, num_of_agents, starts, verbose=False):
    
    # Split up sequence of transpositions into timesteps
    timesteps = []
    agent_set = set()
    end_location_set = set()
    single_timestep = []
    
    transposition_history = get_transposition_history(tree, node_id)
    for agent_id, (end_location, start_location) in transposition_history:

        # Flush current timestep if duplicate agent
        if agent_id in agent_set:
            timesteps.append(single_timestep)
            single_timestep = []
            agent_set = set()
            end_location_set = set()
        
        # Flush current timestep if duplicate end location
        if end_location in end_location_set:
            timesteps.append(single_timestep)
            single_timestep = []
            agent_set = set()
            end_location_set = set()
        
        single_timestep.append((agent_id, end_location, start_location))
        agent_set.add(agent_id)
        end_location_set.add(end_location)
        
    if single_timestep:
        timesteps.append(single_timestep)
        
    if verbose:
        for i, timestep in enumerate(timesteps):
            print("Timestep {}:".format(i))
            for agent_id, end_location, start_location in timestep:
                print("Agent {}: {} -> {}".format(agent_id, start_location, end_location))
    
    # Convert sequence of transpositions into paths
    result = [[start] for start in starts]
    
    for timestep in timesteps:
        moved_agents = set()
        for agent_id, end_location, start_location in timestep:
            result[agent_id].append(end_location)
            moved_agents.add(agent_id)

        for agent_id in range(num_of_agents):
            if agent_id not in moved_agents:
                result[agent_id].append(result[agent_id][-1])
    
    if verbose:
        print("\nFinal Paths:")
        for agent, path in enumerate(result):
            print("Agent {}: Path: {}".format(agent, path))
        
    return result


class GroupActionTAPFSolver(object):
    """A Target Assignment and Path Finding planner that plans for each robot via group action."""

    def __init__(self, my_map, starts, goals, graph=False):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map    
        self.starts = list(starts)
        self.goals = goals
        self.num_of_agents = len(goals)
        self.graph = graph
        self.phi = solve_poisson(my_map, starts, goals)
        
        self.active_agents = set(range(self.num_of_agents))
        self.active_goals = set(goals)
        self.num_of_active_agents = self.num_of_agents
        
        self.curr_agent_locations = list(starts)
        self.next_expansion_agent_deque = deque(range(self.num_of_agents))
        self.transposition_deque = deque()
        self.backtrack_tally = {agent_id: 0 for agent_id in range(self.num_of_agents)}
        self.pick_iteration = 0
        self.last_picked_iteration = {agent_id: -1 for agent_id in range(self.num_of_agents)}
        
        self.tree = Tree()
        self.current_tree_node = "root"
        self.next_tree_node_id = 1

        root_value = calculate_tree_value(self.curr_agent_locations, self.active_agents, self.phi, self.num_of_agents, self.my_map)
        self.tree.create_node(
            tag="root",
            identifier="root",
            parent=None,
            data={
                'current_agent_positions': list(self.curr_agent_locations),
                'transpositions': None,
                'phi': self.phi,
                'active_agents': set(self.active_agents),
                'active_goals': set(self.active_goals),
                'value': root_value,
                'visits': 1,
            }
        )

        self.CPU_time = 0
        
        if self.graph:
            try:
                plot_solution(
                    self.my_map,
                    self.starts,
                    self.goals,
                    self.phi,
                    "group_action_tapf_fields.png",
                )
                print("Saved visualization.")
            except Exception as e:
                print("Graph initialization plot failed: {}".format(e))
                self.graph = False

    def _handle_goal_reached(self, agent_id):
        goal = self.curr_agent_locations[agent_id]
        if goal not in self.active_goals:
            return
        
        self.active_goals.discard(goal)
        self.active_agents.discard(agent_id)
        
        if self.active_agents and self.active_goals:
            active_starts = [self.curr_agent_locations[i] for i in sorted(self.active_agents)]
            self.phi = solve_poisson(self.my_map, active_starts, list(self.active_goals))
        
        node_value = calculate_tree_value(self.curr_agent_locations, self.active_agents, self.phi, self.num_of_agents, self.my_map)
        node_id = str(self.next_tree_node_id)
        self.next_tree_node_id += 1
        self.tree.create_node(
            tag=node_id,
            identifier=node_id,
            parent=self.current_tree_node,
            data={
                'current_agent_positions': list(self.curr_agent_locations),
                'transpositions': list(self.transposition_deque),
                'phi': self.phi,
                'active_agents': set(self.active_agents),
                'active_goals': set(self.active_goals),
                'value': node_value,
                'visits': 1,
            }
        )

        self.current_tree_node = node_id
        self.transposition_deque.clear()
        print("Agent {} reached goal at {}. Remaining agents: {}, Remaining goals: {}".format(
            agent_id, goal, self.active_agents, self.active_goals))

    def _choose_agent(self):
        queue = self.next_expansion_agent_deque

        # Remove inactive agents from the existing queue.
        queue = deque(agent_id for agent_id in queue if agent_id in self.active_agents)

        # Add any active agents that are not yet in the queue.
        seen = set(queue)
        for agent_id in sorted(self.active_agents):
            if agent_id not in seen:
                queue.append(agent_id)

        self.next_expansion_agent_deque = queue
        agent_id = queue.popleft()
        queue.append(agent_id)
        return agent_id

    def _choose_agent_backtrack(self, backtrack_ratio=0.05):
        queue = self.next_expansion_agent_deque

        # Remove inactive agents from the existing queue.
        queue = deque(agent_id for agent_id in queue if agent_id in self.active_agents)

        # Add any active agents that are not yet in the queue.
        seen = set(queue)
        for agent_id in sorted(self.active_agents):
            if agent_id not in seen:
                queue.append(agent_id)

        self.next_expansion_agent_deque = queue

        best_agent = None
        best_score = float('-inf')
        best_wait = -1

        for aid in queue:
            last_iter = self.last_picked_iteration.get(aid, -1)
            wait_iters = self.pick_iteration - last_iter
            score = float(wait_iters) * (1.0 + backtrack_ratio * float(self.backtrack_tally.get(aid, 0)))

            if score > best_score or (score == best_score and wait_iters > best_wait):
                best_score = score
                best_wait = wait_iters
                best_agent = aid

        chosen = best_agent

        queue.remove(chosen)
        queue.append(chosen)

        self.pick_iteration += 1
        self.last_picked_iteration[chosen] = self.pick_iteration

        return chosen

    def _backtrack_with_ucb(self):
        all_nodes = self.tree.all_nodes()
        if not all_nodes:
            return False
        
        root_node = self.tree.get_node("root")
        root_visits = root_node.data.get('visits', 1)
        
        best_node = None
        best_ucb = float('-inf')
        
        for node in all_nodes:
            # Skip the current node we're backtracking from
            if node.identifier == self.current_tree_node:
                continue
            
            parent = self.tree.parent(node.identifier)
            parent_visits = parent.data.get('visits', 1) if parent else root_visits
            
            ucb = calculate_ucb(node, max(parent_visits, 1), exploration_constant=1.41)
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_node = node
        
        if best_node is None:
            print("No valid backtrack node found.")
            return False
        
        # Restore solver state from best node
        node_data = best_node.data
        self.curr_agent_locations = list(node_data['current_agent_positions'])
        self.active_agents = set(node_data['active_agents'])
        self.active_goals = set(node_data['active_goals'])
        self.phi = node_data['phi']
        self.transposition_deque = deque(node_data['transpositions']) if node_data['transpositions'] else deque()
        
        # Update tree position and backpropagate visit counts
        self.current_tree_node = best_node.identifier
        walk_node = best_node
        while walk_node is not None:
            walk_node.data['visits'] = walk_node.data.get('visits', 0) + 1
            parent = self.tree.parent(walk_node.identifier)
            walk_node = parent if parent is not None else None
        
        # Re-initialize agent queue with active agents
        self.next_expansion_agent_deque = deque(sorted(self.active_agents))
        
        print("Backtracked to node {} with UCB value {:.2f}. Visits: {}".format(
            best_node.identifier, best_ucb, best_node.data['visits']))
        
        return True

    def _try_push_chain(self, agent_to_push, forbidden_cells, chain_agents):
        if agent_to_push in chain_agents:
            return False

        new_chain = chain_agents | {agent_to_push}
        current_pos = self.curr_agent_locations[agent_to_push]
        new_forbidden = forbidden_cells | {current_pos}

        legal_moves = find_legal_moves(current_pos, self.my_map)
        legal_moves = [loc for loc in legal_moves if loc not in forbidden_cells]
        ranked_moves = rank_legal_moves(current_pos, legal_moves, self.phi, self.my_map)

        # Non-backtracking filter
        most_recent_start = None
        for moved_id, (_end, start_loc) in self.transposition_deque:
            if moved_id == agent_to_push:
                most_recent_start = start_loc
                break
        non_bt = [loc for loc in ranked_moves if loc != most_recent_start]
        candidates = non_bt if non_bt else ranked_moves

        for candidate in candidates:
            occupant = find_clashing_agent(candidate, self.curr_agent_locations)
            
            # Cell is free
            if occupant is None:
                if agent_to_push not in self.active_agents:
                    self.active_agents.add(agent_to_push)
                    self.active_goals.add(current_pos)
                    print("Reactivated agent {} due to collision".format(agent_to_push))
                self.transposition_deque.appendleft((agent_to_push, (candidate, current_pos)))
                self.curr_agent_locations[agent_to_push] = candidate
                self._handle_goal_reached(agent_to_push)
                return True
            
            # Cell is occupied by an agent not already in the chain
            elif occupant not in new_chain:
                if self._try_push_chain(occupant, new_forbidden, new_chain):
                    if agent_to_push not in self.active_agents:
                        self.active_agents.add(agent_to_push)
                        self.active_goals.add(current_pos)
                        print("Reactivated agent {} due to collision".format(agent_to_push))
                    self.transposition_deque.appendleft((agent_to_push, (candidate, current_pos)))
                    self.curr_agent_locations[agent_to_push] = candidate
                    self._handle_goal_reached(agent_to_push)
                    return True

        return False

    def find_solution(self):
        """ Finds paths for all agents from their start locations to their goal locations."""

        start_time = timer.time()
        
        while not agents_at_goal(self.curr_agent_locations, self.goals):
            
            # Break out when hitting max time
            if timer.time() - start_time > 30.0:
                break

            # Choose queued agent
            agent_id = self._choose_agent_backtrack()
            
            # Agent already at goal
            if self.curr_agent_locations[agent_id] in self.active_goals:
                self._handle_goal_reached(agent_id)
                continue
            
            # Agent not at goal, rank legal moves by potential field
            legal_moves = find_legal_moves(self.curr_agent_locations[agent_id], self.my_map)
            ranked_legal_moves = rank_legal_moves(self.curr_agent_locations[agent_id], legal_moves, self.phi, self.my_map)
            collision_on_move = check_collision_on_move(ranked_legal_moves, self.curr_agent_locations)
            
            # # DEBUG
            # print("Agent: {}, Current Location: {}, Legal Moves: {}, Ranked Legal Moves: {}, Collision on Move: {}".format(
            #     agent_id, self.curr_agent_locations[agent_id], legal_moves, ranked_legal_moves, collision_on_move))
            
            # Priority move is non-clashing
            if ranked_legal_moves and not collision_on_move[0]:
                print("Agent {} moves from {} to {}".format(agent_id, self.curr_agent_locations[agent_id], ranked_legal_moves[0]))
                self.transposition_deque.appendleft((agent_id, (ranked_legal_moves[0], self.curr_agent_locations[agent_id])))
                self.curr_agent_locations[agent_id] = ranked_legal_moves[0]
                self._handle_goal_reached(agent_id)
                continue
            
            # Priority move is clashing
            clashing_agent_id = find_clashing_agent(ranked_legal_moves[0], self.curr_agent_locations)
            print("Agent {} collides with Agent {} at {}".format(agent_id, clashing_agent_id, ranked_legal_moves[0]))

            # Recursively push the chain of blocking agents to free the target cell.
            target_cell = ranked_legal_moves[0]
            if self._try_push_chain(clashing_agent_id, {self.curr_agent_locations[agent_id]}, {agent_id}):
                self.transposition_deque.appendleft((agent_id, (target_cell, self.curr_agent_locations[agent_id])))
                self.curr_agent_locations[agent_id] = target_cell
                self._handle_goal_reached(agent_id)
                continue
            
            # Move own agent to non-backtracking cell
            self_moves = find_non_backtracking_move(agent_id, ranked_legal_moves[1:], self.transposition_deque, self.curr_agent_locations)
            print("Agent {} non-backtracking moves: {}".format(agent_id, self_moves))
            
            if self_moves:
                self.transposition_deque.appendleft((agent_id, (self_moves[0], self.curr_agent_locations[agent_id])))
                self.curr_agent_locations[agent_id] = self_moves[0]
                self._handle_goal_reached(agent_id)
                continue
            
            # Backtrack to best node using UCB
            self.backtrack_tally[agent_id] += 1
            if not self._backtrack_with_ucb():
                print("Backtracking failed. No valid nodes to backtrack to.")
                break
            

        if agents_at_goal(self.curr_agent_locations, self.goals):
            print("\n Found a solution! \n")
        else:
            print("\n No solution found. \n")
            return None

        # Flush any remaining transpositions not yet committed to the tree
        if self.transposition_deque:
            final_node_value = calculate_tree_value(self.curr_agent_locations, self.active_agents, self.phi, self.num_of_agents, self.my_map)
            node_id = str(self.next_tree_node_id)
            self.next_tree_node_id += 1
            self.tree.create_node(
                tag=node_id,
                identifier=node_id,
                parent=self.current_tree_node,
                data={
                    'current_agent_positions': list(self.curr_agent_locations),
                    'transpositions': list(self.transposition_deque),
                    'phi': self.phi,
                    'active_agents': set(self.active_agents),
                    'active_goals': set(self.active_goals),
                    'value': final_node_value,
                    'visits': 1,
                }
            )
            self.current_tree_node = node_id
            self.transposition_deque.clear()

        result = convert_to_path(self.tree, self.current_tree_node, self.num_of_agents, self.starts)
        self.CPU_time = timer.time() - start_time

        print("CPU time (s):    {:.2f}".format(self.CPU_time))
        print("Num of agents:   {}".format(self.num_of_agents))
        print("Node count:      {}".format(len(self.tree.all_nodes())))
        print("Sum of costs:    {}".format(get_sum_of_cost(result)))

        return result
