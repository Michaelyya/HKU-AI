import sys, parse, grader

def astar_search(problem):
    # astar search also uses priority queue
    # steps are similar to greedy, but it adds on the heuristic num with the edge num
    # iteratively dequeue the lowest sum from the list until find the goal node
    
    start = problem['start_state']
    goal = problem['goal_states']
    state_space_graph = problem['state_space_graph']
    heuristics = problem['heuristics']

    frontier = [(heuristics[start], start, [start], 0)]
    explored = []
    best = {start: 0}
    
    visited = set()
    
    while frontier:
        frontier.sort(key=lambda x: (x[0], x[1]))
        f_value, current_state, path, g_value = frontier.pop(0)
        if current_state in visited and g_value >= best[current_state]:
            continue
        
        if current_state not in visited:
            visited.add(current_state)
            # print("**********")
            if current_state not in goal:
                explored.append(current_state)
        # print(explored)

        best[current_state] = g_value
        # print(best)
        if current_state in goal:
            return ' '.join(explored) + '\n' + ' '.join(path)
        
        for edge_cost, next in state_space_graph.get(current_state, []):
            new_g = g_value + edge_cost
            
            if next not in best or new_g < best[next]:
                new_f = new_g + heuristics[next]
                new_path = path + [next]
                frontier.append((new_f, next, new_path, new_g))
    
    return ":)"

if __name__ == "__main__":
    test_case_id = int(sys.argv[1])
    problem_id = 4
    grader.grade(problem_id, test_case_id, astar_search, parse.read_graph_search_problem)