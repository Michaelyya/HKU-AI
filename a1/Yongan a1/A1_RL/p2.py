import sys, random, grader, parse
from collections import defaultdict
import math
from p1 import GameState, get_position, run_game, BasicPacman
import time, os, copy

"""
Q-Learning Pacman Assignment Part B

Objective: Implement core Q-learning components to help Pacman navigate mazes while avoiding ghosts.

Part 1: Feature Engineering (get_features)
- Implement state feature extraction using Manhattan distance
- Suggested features:
  1. closest_food: (distance to nearest food)
  2. ghost_distance: distance to ghost
  3. ghost_nearby: 1 if ghost within 2 units else 0

Part 2: Q-Value Calculation (get_q_value)
- Compute Q(s,a) = Σ(feature * weight)

Part 3: Q-Value Update (update)
- Implement Q-learning update rule using TD error

Part 4: Exploration vs Exploitation (get_action)
- Implement ε-greedy strategy

Part 5: Hyperparameter Tuning (Optional)
- Adjust feature weights
- Experiment with ε, α, γ values

Testing:
- Run: python p2.py 1 40 0 (problem_id, num_runs, if verbose information)
- Aim for >70% win rate on p1, p2, p3, p4

"""


class QLearningPacman(BasicPacman):
    def __init__(self, pos, epsilon=0.2, alpha=0.0002, gamma=0.9):
        """
        Initialize Q-learning parameters
        - self.epsilon: exploration rate
        - self.alpha: learning rate
        - self.gamma: discount factor
        - self.q_values: dictionary storing feature weights
        """
        super().__init__(pos=pos)
        self.pos = pos
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        # init learnable weights
        self.weights = {
            'closest_food': 0.0,
            'ghost_distance': 0.0,
            'ghost_nearby': 0.0
        }


    def get_features(self, state, action):
        """
        Example: Extract state-action features
        Return: Dictionary of {feature_name: feature_value}
        Features to implement:
        1. Manhattan Distance to closest food (scaled by self.feature_weights['food_distance'])
        2. Manhattan Distance to ghost (scaled by self.feature_weights['ghost_distance'])
        3. Binary indicator if ghost is within 2 units (self.feature_weights['ghost_nearby'])
        """
        features = {}
        self.feature_weights = {
            'food_distance': -1.5,
            'ghost_distance': 2.0,
            'ghost_nearby': -10.0,
        }
        next_pos = get_position(self.pos, action)
        food_distances = [abs(next_pos[0] - food[0]) + abs(next_pos[1] - food[1]) for food in state.food]
        if food_distances:
            features['closest_food'] = min(food_distances)
        else:
            features['closest_food'] = 0

        ghost_distance = abs(next_pos[0] - state.ghost_pos[0]) + abs(next_pos[1] - state.ghost_pos[1])
        features['ghost_distance'] = ghost_distance
        features['ghost_nearby'] = 1 if ghost_distance <= 2 else 0
    
        return features

    def get_q_value(self, state, action):
        """
        TODO: Compute Q(s,a) = Σ(feature * weight)
        """
        features = self.get_features(state, action)
        q= 0
        
        for feature, value in features.items():
            q += value * self.weights[feature]
        
        return q

    def update(self, state, action, next_state, reward):
        """
        TODO: Implement Q-learning update rule:
        Q(s,a) += α * [r + γ * max_a' Q(s',a') - Q(s,a)]
        """
        q = self.get_q_value(state, action)
        next_q = float('-inf')
        next_actions = self.get_legal_actions(next_state)
        
        if next_actions:
            next_q_values = [self.get_q_value(next_state, next_action) for next_action in next_actions]
            next_q = max(next_q_values)
        else:
            next_q = 0
        td_error = reward + self.gamma * next_q - q
    
        features = self.get_features(state, action)
        for feature, value in features.items():
            self.weights[feature] += self.alpha * td_error * value


    def get_action(self, state):
        """
        TODO: With prob ε: random action
        Otherwise: action with highest Q-value
        """
        legal_actions = self.get_legal_actions(state)
        
        if random.random() < self.epsilon:
            return random.choice(legal_actions)

        legal_actions = self.get_legal_actions(state)
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        
        q_values = {action: self.get_q_value(state, action) for action in legal_actions}
        max_q = max(q_values.values())
        best_actions = [action for action, q_value in q_values.items() if q_value == max_q]
        
        return random.choice(best_actions)
    
    def stop_train(self):
        self.epsilon = 0.0    
        self.alpha = 0.0



def q_train(problem, episodes=500):
    """Train a Q-learning Pacman agent through specified number of episodes.
    
    Args:
        problem: Dictionary containing game layout (walls, food, ghost, pacman)
        episodes: Number of training iterations (default: 500)
        
    Returns:
        Trained Pacman agent with learned Q-values
    """
    pacman = QLearningPacman(problem['pacman'])
    stats = {'wins': 0, 'steps': []}
    
    for episode in range(episodes):
        walls = problem['walls']
        food = set(problem['food'])
        ghost_pos = problem['ghost']
        pacman.pos = problem['pacman']
        
        step_count = 0
        episode_active = True
        while episode_active and step_count < 1000:
            current_state = GameState(walls, food, ghost_pos)
            
            # 1. Agent Decision
            action = pacman.get_action(current_state)
            new_pacman_pos = pacman.set_new_position(action)
            
            # 2. Reward Calculation
            reward = 0
            if new_pacman_pos == ghost_pos:
                reward = -100  # Strong penalty for ghost collision
            elif new_pacman_pos in food:
                reward = 10    # Positive reward for eating food
                food.remove(new_pacman_pos)
            else:
                reward = -1    # Small penalty to encourage efficiency
                
            # 3. Q-value Update
            next_state = GameState(walls, food, ghost_pos)
            pacman.update(current_state, action, next_state, reward)
            
            # 4. Ghost Movement 
            ghost_actions = [d for d in ['N','S','W','E'] 
                           if get_position(ghost_pos, d) not in walls]
            ghost_move = random.choice(ghost_actions)
            ghost_pos = (
                ghost_pos[0] + {'N': -1, 'S': 1}.get(ghost_move, 0),
                ghost_pos[1] + {'W': -1, 'E': 1}.get(ghost_move, 0)
            )
            
            # 5. Episode Termination Check
            episode_active = not (new_pacman_pos == ghost_pos or not food)
            
            if not episode_active:
                stats['wins'] += int(not food)  # Track successful episodes
                stats['steps'].append(step_count)
                
            step_count += 1
        
        # Exploration Rate Decay (Gradual shift from exploration to exploitation)
        pacman.epsilon = 0.1 + 0.4 * math.exp(-episode/1000)
    
    # Training Summary
    print(f"[Training Complete]")
    print(f"Episodes: {episodes}")
    print(f"Win Rate: {stats['wins']/episodes:.1%}")  # Formatted percentage
    print(f"Avg Steps per Episode: {sum(stats['steps'])/episodes:.1f}")
    
    # Freeze learning parameters for evaluation
    pacman.stop_train()
    return pacman


def better_play_single_ghosts(problem, pacman=None):
    # random.seed(problem['seed'])
    walls = problem['walls']
    food = set(problem['food'])
    ghost_pos = problem['ghost']
    seed = problem['seed']
    winner, solution = run_game(walls, food, ghost_pos, pacman, seed)
    return solution, winner


if __name__ == "__main__":
    test_case_id = int(sys.argv[1])    
    problem_id = 2
    file_name_problem = str(test_case_id)+'.prob' 
    file_name_sol = str(test_case_id)+'.sol'
    path = os.path.join('test_cases','p'+str(problem_id)) 
    problem = parse.read_layout_problem(os.path.join(path,file_name_problem))
    num_trials = int(sys.argv[2])
    verbose = bool(int(sys.argv[3]))
    print('test_case_id:',test_case_id)
    print('num_trials:',num_trials)
    print('verbose:',verbose)
    # global pacman
    random.seed(problem['seed'])

    pacman = q_train(problem=problem, episodes=500)

    start = time.time()
    win_count = 0
    for i in range(num_trials):
        solution, winner = better_play_single_ghosts(copy.deepcopy(problem), pacman=pacman)
        if winner == 'Pacman':
            win_count += 1
        if verbose:
            print(solution)
    win_p = win_count/num_trials * 100
    end = time.time()
    print('time: ',end - start)
    print('win %',win_p)
