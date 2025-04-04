import os, sys
def read_graph_search_problem(file_path):
    output = {}
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        output['start_state'] = lines[0].strip().split(': ')[1]
        output['goal_states'] = lines[1].strip().split(': ')[1].split()
        state_space_graph = {} 
        heuristics = {}
        line_index = 2 #SKIP first two

        while line_index < len(lines) and len(lines[line_index].strip().split()) == 2:
            state, num = lines[line_index].strip().split()
            state_space_graph[state] = []  
            heuristics[state] = float(num)
            line_index += 1
        # print(state_space_graph)

        while line_index < len(lines):
            line = lines[line_index].strip()
            if line:
                parts = line.split()
                if len(parts) == 3:
                    start, end, cost = parts
                    cost = float(cost)
                    
                    if start in state_space_graph:
                        state_space_graph[start].append((cost, end))
                    else:
                        state_space_graph[start] = [(cost, end)]
            
            line_index += 1
        # print(state_space_graph)

        for state in state_space_graph:
            state_space_graph[state] = sorted(state_space_graph[state], key=lambda x: x[0]) #sorted for DFS
            
        output['state_space_graph'] = state_space_graph
        output['heuristics'] = heuristics
    # print(state_space_graph)
    return output

def read_8queens_search_problem(file_path):
    #Your p6 code here
    problem = ''
    return problem

if __name__ == "__main__":
    if len(sys.argv) == 3:
        problem_id, test_case_id = sys.argv[1], sys.argv[2]
        if int(problem_id) <= 5:
            problem = read_graph_search_problem(os.path.join('test_cases','p'+problem_id, test_case_id+'.prob'))
        else:
            problem = read_8queens_search_problem(os.path.join('test_cases','p'+problem_id, test_case_id+'.prob'))
        print(problem)
    else:
        print('Error: I need exactly 2 arguments!')