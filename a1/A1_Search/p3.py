import sys, parse, grader

def greedy_search(problem):
    # greedy is primarily based on the heuristic number within the node
    # still find the start node
    # sort the node list from the start node but based on the heuristic num
    # dequeue node each time and add its neighbor (but sort the list each time)
    # iteratively repeat through the priority queue until find the goal node
    
    start = problem['start_state']
    goal = problem['goal_states']
    state_space_graph = problem['state_space_graph']
    heuristics = problem['heuristics']

    frontier = [(heuristics[start], start, [start])]
    explored = []
    visited = set()
    
    while frontier:
        frontier.sort(key=lambda x: (x[0], ''.join(x[2])))
        h_value, current_state, path = frontier.pop(0)
        
        if current_state in visited:
            continue
            
        visited.add(current_state)
        # print(visited)
        
        if current_state not in goal:
            explored.append(current_state)
            
        if current_state in goal:
            return ' '.join(explored) + '\n' + ' '.join(path)

        for _, neighbor in state_space_graph.get(current_state, []):
            if neighbor not in visited:
                new_path = path + [neighbor]
                frontier.append((heuristics[neighbor], neighbor, new_path))
    return ":,)"

if __name__ == "__main__":
    test_case_id = int(sys.argv[1])
    problem_id = 3
    grader.grade(problem_id, test_case_id, greedy_search, parse.read_graph_search_problem)