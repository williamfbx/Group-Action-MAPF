import time as timer
import heapq
from treelib import Tree
from collections import deque
from poisson_solver import solve_poisson
from plotter import plot_solution
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Circle, Rectangle

COLORS = ['green', 'blue', 'orange']


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


def get_transposition_history(transposition_tree, node_id):
    history = []
    current_id = node_id

    while current_id is not None:
        node = transposition_tree.get_node(current_id)
        if node is None:
            break

        node_data = node.data
        if node_data['transposition'] is not None and node_data['agent_id'] is not None:
            history.append((node_data['agent_id'], node_data['transposition']))

        current_id = node_data['parent_node']

    history.reverse()
    return history


def convert_to_path(transposition_tree, node_id, num_of_agents, starts, verbose=True):
    
    # Split up sequence of transpositions into timesteps
    timestep_deque = deque()
    agent_set = set()
    end_location_set = set()
    single_timestep = []
    
    transposition_history = get_transposition_history(transposition_tree, node_id)
    for agent_id, (end_location, start_location) in transposition_history:

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
        
    if verbose:
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
    
    if verbose:
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


def find_non_backtracking_move(agent_id, legal_moves, transposition_tree, node_id, curr_agent_locations):
    most_recent_start = None

    current_id = node_id
    while current_id is not None:
        node = transposition_tree.get_node(current_id)
        if node is None:
            break

        node_data = node.data
        if node_data['agent_id'] == agent_id and node_data['transposition'] is not None:
            _, start_location = node_data['transposition']
            most_recent_start = start_location
            break

        current_id = node_data['parent_node']

    if most_recent_start is None:
        non_backtracking_moves = legal_moves
    else:
        non_backtracking_moves = [loc for loc in legal_moves if loc != most_recent_start]

    collision_on_move = check_collision_on_move(non_backtracking_moves, curr_agent_locations)
    return [move for move, collision in zip(non_backtracking_moves, collision_on_move) if not collision]


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
        
        self.transposition_tree = Tree()
        self.curr_agent_locations = list(starts)
        self.next_expansion_agent_deque = deque([i for i in range(self.num_of_agents)])
        self.current_transposition_node = "root"
        self.next_transposition_node_id = 1

        self.transposition_tree.create_node(
            tag="root",
            identifier="root",
            data={
                'agent_id': None,
                'current_agent_positions': list(self.curr_agent_locations),
                'transposition': None,
                'next_expansion_agent_deque': list(self.next_expansion_agent_deque),
                'parent_node': None,
                'restrictions': []
            }
        )

        self.CPU_time = 0
        
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))
        # print("Heuristics: {}".format(self.heuristics))

        self.graph_step = 0
        
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

    def _graph_current_state_and_wait(self):
        paths = convert_to_path(
            self.transposition_tree,
            self.current_transposition_node,
            self.num_of_agents,
            self.starts,
            verbose=False
        )

        current_locations = [path[-1] for path in paths]
        rows = len(self.my_map)
        cols = len(self.my_map[0])

        fig, ax = plt.subplots(figsize=(6, 6))
        fig.subplots_adjust(bottom=0.12)
        ax.set_aspect('equal')
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(-0.5, rows - 0.5)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.add_patch(Rectangle((-0.5, -0.5), cols, rows, facecolor='none', edgecolor='gray'))

        for r in range(rows):
            for c in range(cols):
                if self.my_map[r][c]:
                    y = rows - 1 - r
                    ax.add_patch(Rectangle((c - 0.5, y - 0.5), 1, 1, facecolor='gray', edgecolor='gray'))

        for i, goal in enumerate(self.goals):
            gx, gy = goal[1], rows - 1 - goal[0]
            ax.add_patch(Rectangle((gx - 0.25, gy - 0.25), 0.5, 0.5, facecolor=COLORS[i % len(COLORS)], edgecolor='black', alpha=0.5))
            ax.text(gx, gy - 0.35, str(i), fontsize=7, ha='center', va='center')

        for i, loc in enumerate(current_locations):
            x, y = loc[1], rows - 1 - loc[0]
            ax.add_patch(Circle((x, y), 0.3, facecolor=COLORS[i % len(COLORS)], edgecolor='black', alpha=1.0))
            ax.text(x, y + 0.42, str(i), fontsize=8, ha='center', va='center')

        fig.suptitle("GroupAction step {} (static state)".format(self.graph_step), fontsize=10)
        proceed = {'clicked': False}

        button_ax = fig.add_axes([0.86, 0.02, 0.12, 0.06])
        next_button = Button(button_ax, 'Next')

        def _on_next(_event):
            proceed['clicked'] = True
            plt.close(fig)

        next_button.on_clicked(_on_next)
        plt.show(block=True)

        if not proceed['clicked']:
            raise RuntimeError("Visualization closed before pressing Next.")

        self.graph_step += 1

    def _add_transposition_node(self, agent_id, transposition):
        parent_node = self.transposition_tree.get_node(self.current_transposition_node)
        if parent_node is not None:
            if 'restrictions' not in parent_node.data or parent_node.data['restrictions'] is None:
                parent_node.data['restrictions'] = []
            parent_node.data['restrictions'].append(transposition)

        node_id = "n{}".format(self.next_transposition_node_id)
        self.next_transposition_node_id += 1

        self.transposition_tree.create_node(
            tag=node_id,
            identifier=node_id,
            parent=self.current_transposition_node,
            data={
                'agent_id': agent_id,
                'current_agent_positions': list(self.curr_agent_locations),
                'transposition': transposition,
                'next_expansion_agent_deque': list(self.next_expansion_agent_deque),
                'parent_node': self.current_transposition_node,
                'restrictions': []
            }
        )
        self.current_transposition_node = node_id

    def _restore_from_node(self, node_id):
        node = self.transposition_tree.get_node(node_id)
        if node is None:
            return False

        node_data = node.data
        self.current_transposition_node = node_id
        self.curr_agent_locations = list(node_data['current_agent_positions'])
        self.next_expansion_agent_deque = deque(node_data['next_expansion_agent_deque'])
        return True

    def _branch_at_previous_agent_move(self, agent_id):
        original_node_id = self.current_transposition_node
        original_positions = list(self.curr_agent_locations)
        original_queue = deque(self.next_expansion_agent_deque)

        search_node_id = self.current_transposition_node

        while search_node_id is not None:
            node = self.transposition_tree.get_node(search_node_id)
            if node is None:
                break

            node_data = node.data
            if node_data['agent_id'] == agent_id and node_data['transposition'] is not None:
                branch_parent_id = node_data['parent_node']
                if branch_parent_id is not None and self._restore_from_node(branch_parent_id):
                    parent_node = self.transposition_tree.get_node(self.current_transposition_node)
                    restrictions = parent_node.data.get('restrictions', [])

                    legal_moves = find_legal_moves(self.curr_agent_locations[agent_id], self.my_map)
                    ranked_legal_moves = rank_legal_moves(legal_moves, self.heuristics[agent_id])
                    candidate_moves = find_non_backtracking_move(agent_id, ranked_legal_moves, self.transposition_tree, self.current_transposition_node, self.curr_agent_locations)

                    start_location = self.curr_agent_locations[agent_id]
                    for move in candidate_moves:
                        transposition = (move, start_location)
                        if transposition in restrictions:
                            continue

                        self.curr_agent_locations[agent_id] = move
                        
                        # Remove duplicates from queue and re-append agent to end of queue
                        for queued_agent_id in list(self.next_expansion_agent_deque):
                            if queued_agent_id == agent_id:
                                self.next_expansion_agent_deque.remove(queued_agent_id)
                        self.next_expansion_agent_deque.append(agent_id)
                        
                        self._add_transposition_node(agent_id, transposition)
                        print("Branched Agent {} at {} and moved to {}".format(agent_id, start_location, move))
                        return True

            search_node_id = node_data['parent_node']

        self.current_transposition_node = original_node_id
        self.curr_agent_locations = original_positions
        self.next_expansion_agent_deque = original_queue
        return False


    def find_solution(self):
        """ Finds paths for all agents from their start locations to their goal locations."""

        start_time = timer.time()
        
        while not agents_at_goal(self.curr_agent_locations, self.goals):
            if self.graph:
                try:
                    self._graph_current_state_and_wait()
                except Exception as e:
                    print("Graph mode disabled due to visualization error: {}".format(e))
                    self.graph = False
            
            # Break out when hitting max time
            if not self.graph and timer.time() - start_time > 5:
                print("Exceeded max time. Terminating search.")
                break

            agent_id = self.next_expansion_agent_deque.popleft()
            
            # Agent already at goal, move only if pushed by another agent
            if self.curr_agent_locations[agent_id] == self.goals[agent_id]:
                print("Agent {} is already at the goal. Re-adding to queue.".format(agent_id))
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
                print("Agent {} moves from {} to {}".format(agent_id, self.curr_agent_locations[agent_id], ranked_legal_moves[0]))
                start_location = self.curr_agent_locations[agent_id]
                self.curr_agent_locations[agent_id] = ranked_legal_moves[0]
                self.next_expansion_agent_deque.append(agent_id)
                self._add_transposition_node(agent_id, (ranked_legal_moves[0], start_location))
                continue

            # Priority move is clashing
            clashing_agent_id = find_clashing_agent(ranked_legal_moves[0], self.curr_agent_locations)
            print("Agent {} collides with Agent {} at {}".format(agent_id, clashing_agent_id, ranked_legal_moves[0]))
            
            # Move clashing agent to non-backtracking cell
            clash_legal_moves = find_legal_moves(self.curr_agent_locations[clashing_agent_id], self.my_map)
            clash_legal_moves = [loc for loc in clash_legal_moves if loc != self.curr_agent_locations[agent_id]]
            clash_ranked_legal_moves = rank_legal_moves(clash_legal_moves, self.heuristics[clashing_agent_id])
            clash_move = find_non_backtracking_move(clashing_agent_id, clash_ranked_legal_moves, self.transposition_tree, self.current_transposition_node, self.curr_agent_locations)
            print("Clashing agent {} non-backtracking moves: {}".format(clashing_agent_id, clash_move))
            
            if clash_move:
                print("Moving clashing agent {} from {} to {} to resolve collision".format(clashing_agent_id, self.curr_agent_locations[clashing_agent_id], clash_move[0]))
                clash_start_location = self.curr_agent_locations[clashing_agent_id]
                self.curr_agent_locations[clashing_agent_id] = clash_move[0]
                self.next_expansion_agent_deque.appendleft(agent_id)
                self._add_transposition_node(clashing_agent_id, (clash_move[0], clash_start_location))
                continue
            
            # Move own agent to non-backtracking cell
            self_moves = find_non_backtracking_move(agent_id, ranked_legal_moves[1:], self.transposition_tree, self.current_transposition_node, self.curr_agent_locations)
            print("Agent {} non-backtracking moves: {}".format(agent_id, self_moves))
            
            if self_moves:
                print("Moving agent {} from {} to {} to resolve collision".format(agent_id, self.curr_agent_locations[agent_id], self_moves[0]))
                self_start_location = self.curr_agent_locations[agent_id]
                self.curr_agent_locations[agent_id] = self_moves[0]
                self.next_expansion_agent_deque.append(agent_id)
                self._add_transposition_node(agent_id, (self_moves[0], self_start_location))
                continue
            
            # Recursively branch at agent's most recent avaliable move
            if self._branch_at_previous_agent_move(agent_id):
                continue

            # No moves available, re-append agent to deque
            print("No moves available for Agent {}. Re-adding to queue.".format(agent_id))
            self.next_expansion_agent_deque.append(agent_id)
        
        print("Final transposition history:\n{}".format(get_transposition_history(self.transposition_tree, self.current_transposition_node)))
        result = convert_to_path(self.transposition_tree, self.current_transposition_node, self.num_of_agents, self.starts)
        self.CPU_time = timer.time() - start_time

        if agents_at_goal(self.curr_agent_locations, self.goals):
            print("\n Found a solution! \n")
        else:
            print("\n No solution found. \n")
            return None
        
        print("CPU time (s):    {:.2f}".format(self.CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(result)))

        return result
