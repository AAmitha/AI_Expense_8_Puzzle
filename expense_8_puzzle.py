# Importing the required libraries
import sys
import queue
from datetime import datetime
from queue import Queue

# BFS algorithm
def bfs(start, goal):
    start = tuple(map(tuple, start))
    goal = tuple(map(tuple, goal))
    queue = Queue()
    queue.put(start)
    visited = set()
    parent = {}
    depth = {start: 0}
    cost = {start: 0}
    max_fringe_size = 1
    result = []
    
    #Checking whether the below queue is empty or not
    while not queue.empty():
        node = queue.get()
        visited.add(node) # adding the node into the visited set 
        
        # comparing and checking the condition that the goal state is equal to the start state
        if node == goal:
            path = []
            while node in parent:
                path.append(node)
                node = parent[node]
            path.append(start)
            path.reverse()
            print(f"Nodes Popped: {len(visited)}")
            print(f"Nodes Expanded: {len(visited)-1}")
            print(f"Nodes Generated: {len(parent)}")
            print(f"Max Fringe Size: {max_fringe_size}")
            print((f"Solution Found at depth {depth[goal]} with cost of {cost[goal]}."))
            print('Steps:')
            for step in result:
                move_cost, move_name = step[1], step[2]
                print(f"\tMove {move_cost} {move_name}")
            print("Result Found Successfully!")
            file_name_bfs = f'bfs_trace_file_{datetime.now().strftime("%d_%m_%Y_%H_%M")}.txt'
            file_bfs = open(file_name_bfs, "a")
            file_bfs.write(f"\tNodes Popped: {len(visited)}\n")
            file_bfs.write(f"\tNodes Expanded: {len(visited)-1}\n")
            file_bfs.write(f"\tNodes Generated: {len(parent)}\n")
            file_bfs.write(f"\tMax Fringe Size: {max_fringe_size}\n")
            file_bfs.write((f"\tSolution Found at depth {depth[goal]} with cost of {cost[goal]}."))
            with open(file_name_bfs, 'r+') as file:
                content = file.read()
                file.seek(0, 0)
                file.write("Method_selected: BFS \nRunning BFS Algorithm \n" + content)
            return None
        
        # adding each move into the visited set if already the move not present in the visted set
        for move in moves_bfs(node, visited):
            if move[0] not in visited:
                queue.put(move[0])
                visited.add(move[0])
                parent[move[0]] = node
                depth[move[0]] = depth[node] + 1
                cost[move[0]] = cost[node] + move[1]
                max_fringe_size = max(max_fringe_size, queue.qsize())
                result.append(move)
    print("No solution found.")
    
    
def moves_bfs(node, visited):
    moves = []
    with open(f'bfs_trace_file_{datetime.now().strftime("%d_%m_%Y_%H_%M")}.txt', 'a') as file:
        blank_row, blank_col = get_blank_pos(node)
        # Based on the blank row and cloumn the below for loop will idenify the whether to move 0 to up or down or left or right
        file.write(f"Closed Set: {visited}\n")
        for dr, dc, move_name in [(-1, 0, "Up"), (1, 0, "Down"), (0, -1, "Left"), (0, 1, "Right")]:
            new_row, new_col = blank_row + dr, blank_col + dc
            if 0 <= new_row < len(node) and 0 <= new_col < len(node[0]):
                new_node = swap_positions(node, blank_row, blank_col, new_row, new_col)
                move_cost = node[new_row][new_col]
                moves.append((new_node, move_cost, move_name))
                file.write(f"Fringe_steps: {moves}\n")
    return moves

# This below functin will identify the 0 position in the puzzle and store it's row and column 
def get_blank_pos(node):
    for r, row in enumerate(node):
        for c, val in enumerate(row):
            if val == 0:
                return r, c
    raise ValueError("node is not valid because there is no blank tile(0) found")

# This function will swap the '0' to respective position value.
def swap_positions(node, r1, c1, r2, c2):
    node = [list(row) for row in node]
    node[r1][c1], node[r2][c2] = node[r2][c2], node[r1][c1]
    return tuple(map(tuple, node))
# Getting all the successors of the current node
def get_successors_ucs(state, visited):
    successors = []
    blank_pos = state.index(0)
    blank_row = blank_pos // 3
    blank_col = blank_pos % 3
    with open(f'ucs_trace_file_{datetime.now().strftime("%d_%m_%Y_%H_%M")}.txt', 'a+') as file:
        file.write(f"Closed Set: {visited}\n")
        for move_row, move_col, action in [(-1, 0, 'Up'), (1, 0, 'Down'), (0, -1, 'Left'), (0, 1, 'Right')]:
            new_row = blank_row + move_row
            new_col = blank_col + move_col
            
            if new_row < 0 or new_row >= 3 or new_col < 0 or new_col >= 3:
                continue
            
            new_pos = new_row * 3 + new_col
            new_state = state[:]
            new_state[blank_pos], new_state[new_pos] = new_state[new_pos], new_state[blank_pos]
            
            cost = new_state[blank_row]
            successors.append((action, new_state, cost, blank_pos))
            file.write(f"Successors : {successors}\n")
    return successors

#Uniform Cost Search algorithm
def ucs(start_state, goal_state):
    nodes_popped = 0
    nodes_expanded = 0
    nodes_generated = 0
    max_fringe_size = 0
    
    visited = set()
    frontier = queue.PriorityQueue() 
    frontier.put((0, [start_state, [], 0, []]))
    
    # Checking if the priority_queue is empty or not
    while not frontier.empty():
        current_node = frontier.get()[1]
        current_state, current_path, current_cost, current_pos = current_node
        nodes_popped += 1
        
        # Checking the goal state is equal to the start state
        if current_state == goal_state:
            print(f'Nodes Popped: {nodes_popped}')
            print(f'Nodes Expanded: {nodes_expanded}')
            print(f'Nodes Generated: {nodes_generated}')
            print(f'Max Fringe Size: {max_fringe_size}')
            print(f'Solution Found at depth {len(current_path)} with cost of {current_cost}.')
            print('Steps:')
            file_name_ucs = f'ucs_trace_file_{datetime.now().strftime("%d_%m_%Y_%H_%M")}.txt'
            file_ucs = open(file_name_ucs, "a+")
            file_ucs.write(f"\n\tNodes Popped: {nodes_popped}\n")
            file_ucs.write(f"\tNodes Expanded: {nodes_expanded}\n")
            file_ucs.write(f"\tNodes Generated: {nodes_generated}\n")
            file_ucs.write(f"\tMax Fringe Size: {max_fringe_size}\n")
            file_ucs.write((f"\tSolution Found at depth {len(current_path)} with cost of {current_cost}."))
            with open(file_name_ucs, 'r+') as file:
                content = file.read()
                file.seek(0, 0)
                file.write("Method_selected: UCS \nRunning UCS Algorithm \n" + content)
            for step, pos in zip(current_path, current_pos):
                print(f'\tMove {pos} {step}')
            print("result founded sucessfully!")
            return current_path

        visited.add(tuple(current_state))
        nodes_expanded += 1
        
        for action, new_state, cost, pos in get_successors_ucs(current_state, visited):
            if tuple(new_state) not in visited:
                nodes_generated += 1
                new_path = current_path + [action]
                new_cost = current_cost + cost
                new_pos = current_pos + [pos]
                frontier.put((new_cost, [new_state, new_path, new_cost, new_pos]))
        
        if frontier.qsize() > max_fringe_size:
            max_fringe_size = frontier.qsize() 
    return None
# a_star algorithm
def a_star(initial_state, goal_state):

    # Calculating the heuristic valye using the manhattan distance
    def distance(state):
        dist = 0
        for i in range(len(state)):
            if state[i] != 0:
                dist += abs(i // 3 - (state[i]-1) // 3) + abs(i % 3 - (state[i]-1) % 3)
        return dist

    class Node:
        def __init__(self, state, parent=None, move=None, cost=0):
            self.state = state
            self.parent = parent
            self.move = move
            self.cost = cost
            # Coumputing the heuristic value
            self.heuristic = distance(state) 
            if self.parent:
                self.depth = parent.depth + 1
            else:
                self.depth = 0

        def __lt__(self, other):
            return (self.cost + self.heuristic) < (other.cost + other.heuristic)

    start_node = Node(initial_state) # storing the start state, parent state, move and cost in the class Node

    nodes_popped = 0
    nodes_expanded = 0
    nodes_generated = 0
    cost = 0
    max_fringe_size = 0
    frontier = queue.PriorityQueue()
    frontier.put(start_node)
    visited = set()

   # Checking whether the priorityQueue is empty or not
    while not frontier.empty():
        if frontier.qsize() > max_fringe_size:
            max_fringe_size = frontier.qsize()

        node = frontier.get()
        nodes_popped += 1

        # Checking the goal state is equal to the start state
        if node.state == goal_state:
            path = []
            while node.parent:
                path.append(node.move)
                node = node.parent
                cost += node.cost
            path.reverse()
            depth = len(path)
            print(f"Nodes Popped: {nodes_popped}")
            print(f"Nodes Expanded: {nodes_expanded}")
            print(f"Nodes Generated: {nodes_generated}")
            print(f"Max Fringe Size: {max_fringe_size}")
            print(f"Solution Found at depth {depth} with cost of {cost}.")
            print("Steps:")
            file_name_astar = f'astar_trace_file_{datetime.now().strftime("%d_%m_%Y_%H_%M")}.txt'
            file_astar = open(file_name_astar, "a")
            file_astar.write(f"\tNodes Popped: {nodes_popped}\n")
            file_astar.write(f"\tNodes Expanded: {nodes_expanded}\n")
            file_astar.write(f"\tNodes Generated: {nodes_generated}\n")
            file_astar.write(f"\tMax Fringe Size: {max_fringe_size}\n")
            file_astar.write((f"\tSolution Found at depth {depth} with cost of {cost}."))
            with open(file_name_astar, 'r+') as file:
                content = file.read()
                file.seek(0, 0)
                file.write("Method_selected: ASTAR \nRunning ASTAR Algorithm \n" + content)
            for step in path:
                print(f"\t{step}")
            print("result found successfully!")
            return None
        
        with open(f'astar_trace_file_{datetime.now().strftime("%d_%m_%Y_%H_%M")}.txt', 'a+') as file:
            file.write(f'Closed Set: {visited}\n')
            visited.add(node.state)
            nodes_expanded += 1
            for move, state in get_successors_astar(node.state, file):
                if state not in visited:
                    child = Node(state, parent=node, move=move, cost=node.cost+1)
                    frontier.put(child)
                    nodes_generated += 1
    return None

# This function will return all successors of the node
def get_successors_astar(state, file): 
    successors = []
    i = state.index(0)
    if i not in [0, 1, 2]:
        new_state = list(state)
        new_state[i], new_state[i-3] = new_state[i-3], new_state[i]
        successors.append(("Move {} Down".format(state[i-3]), tuple(new_state)))
    if i not in [6, 7, 8]:
        new_state = list(state)
        new_state[i], new_state[i+3] = new_state[i+3], new_state[i]
        successors.append(("Move {} Up".format(state[i+3]), tuple(new_state)))
    if i not in [0, 3, 6]:
        new_state = list(state)
        new_state[i], new_state[i-1] = new_state[i-1], new_state[i]
        successors.append(("Move {} Right".format(state[i-1]), tuple(new_state)))
    if i not in [2, 5, 8]:
        new_state = list(state)
        new_state[i], new_state[i+1] = new_state[i+1], new_state[i]
        successors.append(("Move {} Left".format(state[i+1]), tuple(new_state)))
    file.write(f'Fringe steps : \n{successors}\n')
    return successors


#Greedy Search Algorithm
def greedy(start, goal):
    visited = set()
    fringe = queue.PriorityQueue()
    fringe.put((heuristic_value(start, goal), start, []))
    max_fringe_size = 1
    
    nodes_popped = 0
    nodes_expanded = 0
    nodes_generated = 1

    # Checking whether fring is not empty or not
    while not fringe.empty():
        _, current, path = fringe.get()
        nodes_popped += 1
        
        # Checking whether goal state is equal to initial state
        if current == goal:
            print(f'Nodes Popped: {nodes_popped}')
            print(f'Nodes Expanded: {nodes_expanded}')
            print(f'Nodes Generated: {nodes_generated}')
            print(f'Max Fringe Size: {max_fringe_size}')
            print(f'Solution Found at depth {len(path)} with cost of {int(len(path)*4.6)}.')
            print('Steps:')
            for step in path:
                print(step)
            print(" Result found Successfully!")
            file_name_greddy = f'greedy_trace_file_{datetime.now().strftime("%d_%m_%Y_%H_%M")}.txt'
            file_greddy = open(file_name_greddy, "a+")
            file_greddy.write(f"\n\tNodes Popped: {nodes_popped}\n")
            file_greddy.write(f"\tNodes Expanded: {nodes_expanded}\n")
            file_greddy.write(f"\tNodes Generated: {nodes_generated}\n")
            file_greddy.write(f"\tMax Fringe Size: {max_fringe_size}\n")
            file_greddy.write((f"\tSolution Found at depth {len(path)} with cost of {int(len(path)*4.6)}."))
            with open(file_name_greddy, 'r+') as file:
                content = file.read()
                file.seek(0, 0)
                file.write("Method_selected: Greedy \nRunning Greddy Algorithm \n" + content)
            return
            
        with open(f'greedy_trace_file_{datetime.now().strftime("%d_%m_%Y_%H_%M")}.txt', 'a+') as file:
            visited.add(current)  # adding the explored node to visited set
            nodes_expanded += 1
            for neighbor in get_successors_greddy(current, file):
                if neighbor not in visited:
                    cost = heuristic_value(neighbor, goal)
                    fringe.put((cost, neighbor, path + [moves_greddy(current, neighbor)]))
                    nodes_generated += 1
                    file.write(f'\nvsited:{visited}, move: {moves_greddy(current, neighbor)}\n')
            max_fringe_size = max(max_fringe_size, fringe.qsize())
    
    return None, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size


def get_successors_greddy(state, file):
    neighbors = []
    x, y = divmod(state.index('0'), 3)
    for dx, dy in [(0, -1), (-1, 0), (0, 1), (1, 0)]:
        newx, newy = x + dx, y + dy
        if 0 <= newx < 3 and 0 <= newy < 3:
            neighbor = list(state)
            neighbor[x*3+y], neighbor[newx*3+newy] = neighbor[newx*3+newy], neighbor[x*3+y]
            neighbors.append(''.join(neighbor))
        file.write(f'Successors : \n{neighbors}\n')
    return neighbors

# This function will return the heuristic value based on manhattan distance
def heuristic_value(state, goal): 
    distance = 0
    for i in range(len(state)):
        x1, y1 = divmod(state.index(str(i)), 3)
        x2, y2 = divmod(goal.index(str(i)), 3)
        distance += abs(x1 - x2) + abs(y1 - y2)
    return distance

# To get all the moves
def moves_greddy(state1, state2):
    index1, index2 = state1.index('0'), state2.index('0')
    x1, y1 = divmod(index1, 3)
    x2, y2 = divmod(index2, 3)
    if y1 > y2:
        return f"\tMove {state1[index2]} Left"
    elif y1 < y2:
        return f"\tMove {state1[index2]} Right"
    elif x1 > x2:
        return f"\tMove {state1[index2]} Up"
    elif x1 < x2:
        return f"\tMove {state1[index2]} Down"


def read_input_txt_to_list_format(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        return [[int(i) for i in line.split()] for line in lines[:3]]

def read_input_txt_to_list(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        return [int(i) for line in lines[:3] for i in line.split()]

def read_input_txt_to_tuple_format(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        return tuple([int(i) for line in lines[:3] for i in line.split()])

def read_input_txt_to_string_format(file_name):
    with open(file_name, 'r') as f:
        return "".join(f.readlines()[:3])

if __name__ == '__main__':
    """ Need to give 5 arguments in the input file and 2 arguments are optional
        1) Python execution file
        2) Start file
        3) Goal file
        4) Method ( This argument is optional by default it is astar)
        5) Dump flag (This argument is optional by default it is false)
    """
    if len(sys.argv) < 3:
        print("Need to give 3 arguments: expense_8_puzzle.py(python execution file name) <start-file> <goal-file> <method> [<dump-flag>]") #If we enter less than 3 arguments it will print this statement and program execution completed.
        sys.exit()

    # Storing the input arguments to the variables
    start_file = sys.argv[1]
    goal_file = sys.argv[2]
    method = sys.argv[3].lower() if len(sys.argv) > 3 else 'astar'
    dump_flag = True if len(sys.argv) > 4 and sys.argv[4].lower() == 'true' else False

    print(f"{'>>' * 30} Executing {method} algorithm{'<<' * 30}")

    # Based on the input given by the user it will execute particular if method
    if method == 'bfs':
        start_state = read_input_txt_to_list_format(start_file)
        goal_state = read_input_txt_to_list_format(goal_file)
        bfs(start_state, goal_state)

    if method ==  'ucs':
        start_state = read_input_txt_to_list(start_file)
        goal_state = read_input_txt_to_list(goal_file)
        ucs(start_state, goal_state)

    if method == 'astar':
        start_state = read_input_txt_to_tuple_format(start_file)
        goal_state = read_input_txt_to_tuple_format(goal_file)
        a_star(start_state, goal_state)

    if method == 'greedy':
        start_state = "".join(map(str, read_input_txt_to_list(start_file)))
        goal_state = "".join(map(str, read_input_txt_to_list(goal_file)))
        greedy(start_state, goal_state)