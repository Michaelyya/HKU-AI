import sys, grader, parse

def dfs_search(problem):
    # find the start node
    # find its neightbors then sort them a list based on the edge
    # reverse the node lists in order to use stack
    # pop the stack, and add the pop'd node's neighbors
    # iteratively repeat the steps until find the goal node
    
    start = problem['start_state']
    goal = problem['goal_states']
    state_space_graph = problem['state_space_graph']
    
    stack = [(start, [start])]  
    explored = []  # Exploration order
    visited = set()  # Visted states
    
    while stack:
        current_state, path = stack.pop()
        
        if current_state in visited:
            continue
        visited.add(current_state)
        
        if current_state not in goal:
            explored.append(current_state)
        # print(explored)
        
        if current_state in goal:
            # Output
            return ' '.join(explored) + '\n' + ' '.join(path)
        
        next = state_space_graph.get(current_state, [])
        nexts = sorted(next, key=lambda x: x[0])
        
        for cost, neighbor in reversed(nexts):
            if neighbor not in visited:
                new_path = path + [neighbor]
                stack.append((neighbor, new_path))
    
    return "Oops"

if __name__ == "__main__":
    test_case_id = int(sys.argv[1])
    problem_id = 1
    grader.grade(problem_id, test_case_id, dfs_search, parse.read_graph_search_problem)