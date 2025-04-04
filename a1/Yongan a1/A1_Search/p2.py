import sys, grader, parse
from collections import deque

def bfs_search(problem):
    # find the start node
    # find its neightbors then sort them a list based on the edge size
    # dequeue the list and enqueue its neighbors
    # iteratively repeat the steps until find the goal node
    
    start = problem['start_state']
    goal = problem['goal_states']
    state_space_graph = problem['state_space_graph']

    queue = deque([(start, [start])])  
    explored = []  # Exploration order
    visited = set()  # Visted states

    while queue:
        current_state, path = queue.popleft()
        if current_state in visited:
            continue
        visited.add(current_state)
        
        if current_state not in goal:
            explored.append(current_state)
        # print(explored)

        if current_state in goal:
            return ' '.join(explored) + '\n' + ' '.join(path)
        
        next = state_space_graph.get(current_state, [])
        nexts = sorted(next, key=lambda x: x[0])
        
        for cost, neighbor in nexts:
            if neighbor not in visited:
                new_path = path + [neighbor]
                queue.append((neighbor, new_path))
    
    return ":("
    
if __name__ == "__main__":
    test_case_id = int(sys.argv[1])
    problem_id = 2
    grader.grade(problem_id, test_case_id, bfs_search, parse.read_graph_search_problem)