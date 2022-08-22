#Implementation of the Nash Q Learning algorithm for simple games with two agents

import numpy as np
import random
from collections import defaultdict
import nashpy as nash
from tqdm import tqdm


class Player:
    def __init__(self, 
                 position = [0,0],
                 movements = ['left','right','up','down','stay']
                ):
            '''
            This class is a representation of a player.
            
            position (list) : list of two integers giving the starting position coordinates of the player
            movements (list) : list of strings containing the possible movements. 
            '''
        self.movements = movements
        self.position = position

        
    def move(self, movement):
        '''
        Compute the new position of a player after performing a movement.
        movement (string) : the movement to perform. Invalid string values are interpreted as the 'stay' movement.     
        '''
        if movement == 'left'  and 'left' in self.movements:
            new_position = [self.position[0] - 1, self.position[1]]
        elif movement == 'right' and 'right' in self.movements:
            new_position = [self.position[0] + 1, self.position[1]]
        elif movement == 'up' and 'up' in self.movements:
            new_position = [self.position[0], self.position[1] + 1]
        elif movement == 'down' and 'down' in self.movements:
            new_position = [self.position[0], self.position[1] - 1]
        else:  
            new_position = self.position
        
        return new_position
   

class Grid:
    def __init__(self,
                 length = 2, 
                 width = 2, 
                 players = [Player(),Player()],
                 reward_coordinates = [1,1],
                 reward_value = 20,
                 obstacle_coordinates = [],
                 collision_allowed = False,
                 collision_penalty = 0):
        '''
        This class is a representation of the game grid.

        length (int) : horizontal dimension of the grid
        width (int)  : vertical dimension of the grid
        players (list) : list of 2 Player objects.
        reward_coordinates (list) : list of 2 integers giving the coordinates of the reward
        reward_value (int) : value obtained by reaching the reward coordinates
        obstacle_coordinates (list) : list of obstacle coordinates. Each obstacle coordinate is a list of 2 integers giving their coordinates
        collision_allowed (bool) : whether agents are allowed to be at the same time on the same cell or not
        collision_penalty (int)  : negative reward obtained for hitting a wall or colliding with another player
        joint_player_coordinates (list) : list containing the starting positions of the two players
        '''
        self.length = length
        self.width = width
        self.players =  players
        self.reward_coordinates = reward_coordinates
        self.reward_value = reward_value
        self.obstacle_coordinates = obstacle_coordinates
        self.collision_allowed = collision_allowed
        self.collision_penalty =  collision_penalty
        self.joint_player_coordinates = [players[0].position, players[1].position]
 

    def get_player_0(self):
        return self.players[0]

    
    def get_player_1(self):
        return self.players[1]
  

    def joint_states(self):
        '''
        Returns a list of all possible joint states in the game.
        '''
        if not self.collision_allowed:
            #Agents are only allowed to collide on the reward cell, whether they arrive there at the same time or not
            joint_states = [[[i,j],
                             [k,l]] for i in range(self.length) for j in range(self.width) 
                             for k in range(self.length) for l in range(self.width)
                             if [i,j] != [k,l]  and [i,j] not in self.obstacle_coordinates and [k,l] not in self.obstacle_coordinates
            ]
            joint_states.append([self.reward_coordinates,self.reward_coordinates]) #Add the reward state as joint state

        else:  #Agents can collide on any cell, but they can't move to an obstacle
            joint_states = [[[i,j],
                             [k,l]] for i in range(self.length) for j in range(self.width) 
                             for k in range(self.length) for l in range(self.width)
                             if  [i,j] not in self.obstacle_coordinates and [k,l] not in self.obstacle_coordinates
            ]

        return joint_states
 

    def identify_walls(self):
        '''
        Identify all impossible transitions due to the grid walls and the obstacles
        '''
        walls = []
        for i in range(self.length):
            for j in range(self.width):
                if [i,j] not in self.obstacle_coordinates:
                    fictious_player = Player(position = [i,j]) #Used to explore the grid in search of walls
                    if fictious_player.move('left')[0] not in range(self.length) or fictious_player.move('left')in self.obstacle_coordinates:
                        walls.append(['left',fictious_player.position])
                    if fictious_player.move('right')[0] not in range(self.length) or fictious_player.move('right') in self.obstacle_coordinates:
                        walls.append(['right',fictious_player.position])
                    if fictious_player.move('up')[1] not in range(self.width) or fictious_player.move('up') in self.obstacle_coordinates:
                        walls.append(['up',fictious_player.position])
                    if fictious_player.move('down')[1] not in range(self.width) or fictious_player.move('down') in self.obstacle_coordinates:
                        walls.append(['down',fictious_player.position])

        return walls
 

    def compute_reward(self,
                       old_state,
                       new_state,
                       movement,
                       collision_detected = False):
        '''
        Compute the reward obtained by a player for transitioning from its old state to its new state
        '''
        if  old_state == self.reward_coordinates:  #Stop receiving rewards once the goal is reached
            reward = 0
        elif new_state == self.reward_coordinates:  #The goal state is reached for the first time
            reward = self.reward_value
        elif new_state == old_state and movement != 'stay': #The player moved and bumped in a player or an obstacle
            reward = self.collision_penalty
        elif movement == 'stay' and collision_detected:  #The player stayed but was percuted by another player
            reward = self.collision_penalty       
        else: # The player made a regular valid movement
            reward = 0
        return reward

    
    def create_transition_table(self):
        '''
        Creates a dictionary where each pair of joint state and joint movement is mapped to a new resulting joint state
        '''
        recursivedict = lambda : defaultdict(recursivedict)
        transitions = recursivedict()
        joint_states = self.joint_states()
        walls = self.identify_walls()
        player0_movements =  self.players[0].movements
        player1_movements =  self.players[1].movements

        for state in joint_states:
            for m0 in player0_movements:
                for m1 in player1_movements:
                    if [m1,state[1]] in walls or state[1] == self.reward_coordinates:
                        if [m0,state[0]]  in walls or state[0] == self.reward_coordinates:
                            new_state = state
                        else : 
                            new_state = [Player(state[0]).move(m0),state[1]]                            
                    else:
                        if [m0,state[0]] in walls or state[0] == self.reward_coordinates:
                            new_state = [state[0],Player(state[1]).move(m1)]
                        else:
                            new_state = [Player(state[0]).move(m0),Player(state[1]).move(m1)]                   
                    if (new_state[0] == state[1]  and new_state[1] == state[0] ) or new_state not in joint_states: 
                        # There is a collision or a swap of positions
                         new_state = state #Return to previous state  
                    transitions[joint_states.index(state)][m0][m1] = joint_states.index(new_state)
                    
        return transitions

    
    def create_stage_games(self):
        '''
        Creates the stage game tables which contains the reward obtained by the players for each pair of joint states and joint movements.
        The stage game tables are represented as 3-dimensional tensors. 
        '''
        joint_states = self.joint_states()
        walls = self.identify_walls()
        player0_movements =  self.players[0].movements
        player1_movements =  self.players[1].movements

        stage_games0  = np.zeros((len(joint_states),
                                 len(player0_movements),
                                 len(player1_movements),
                                 ))
        
        stage_games1  = np.zeros((len(joint_states),
                                 len(player0_movements),
                                 len(player1_movements),
                                 ))
        for state in tqdm(joint_states):
            for m0 in player0_movements:
                for m1 in player1_movements:
                    if [m1,state[1]] in walls:
                        if [m0,state[0]] not in walls:
                            new_state = [Player(state[0]).move(m0),state[1]]
                        else : 
                            new_state = state
                    else:
                        if  [m0,state[0]] in walls:
                            new_state = [state[0],Player(state[1]).move(m1)]
                        else:
                            new_state = [Player(state[0]).move(m0),Player(state[1]).move(m1)]                    
                    collision_detected = False
                    if (new_state[0] == state[1]  and new_state[1] == state[0] ) or new_state not in joint_states: 
                         # There is a collision
                         new_state = state #Return to previous state
                         collision_detected = True
                
                    reward0 = self.compute_reward(state[0],new_state[0],m0,collision_detected)
                    reward1 = self.compute_reward(state[1],new_state[1],m1,collision_detected)                     
                    stage_games0[joint_states.index(state)] [player0_movements.index(m0)][player1_movements.index(m1)]= reward0
                    stage_games1[joint_states.index(state)][player0_movements.index(m0)][player1_movements.index(m1)] = reward1
                  
        return stage_games0, stage_games1

    
    def create_q_tables(self): 
        '''
        Creates the q tables which contains the Q-values used by the the nash Q Learning algorithm.
        The q tables are represented as 3-dimensional tensors and are initialized with null values. 
        '''
        joint_states = self.joint_states()
        player0_movements =  self.players[0].movements
        player1_movements =  self.players[1].movements
        q_tables0 = np.zeros((len(joint_states),
                              len(player0_movements),
                              len(player1_movements)
                                    ))    
        q_tables1 = np.zeros((len(joint_states),
                              len(player0_movements),
                              len(player1_movements)
                                    ))
        
        return q_tables0, q_tables1

  
class  NashQLearning:

    def __init__(self,
                 grid = Grid(),
                 learning_rate = 0.5,
                 max_iter = 100,
                 discount_factor = 0.7,
                 decision_strategy = 'random',
                 epsilon = 0.5,
                 random_state = 42):       
        '''
        This class represents an instance of the Nash Q-Learning algorithm
        
        grid (Grid) : the game grid 
        learning rate (int) : the weighted importance given to the update of the Q-values compared to their current value
        max_iter (int) : max number of iterations of the algorithm
        discount_factor (int) : discount factor applied to the nash equilibria value in the Q-values update formula
        decision_strategy (str) : decision strategy applied to select the next movement, possible values are 'random','greedy','epsilon-greedy'
        epsilon (int) : only if decision_strategy is 'epsilon_greedy', threshold to decide between a greedy or random movement
        random_state (int) : seed for results reproducibility
        '''
        self.grid = grid
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.discount_factor = discount_factor
        self.decision_strategy = decision_strategy
        self.epsilon = epsilon
        random.seed(random_state)
        

    def fit(self,
            return_history = False):
        """
        Fit the Nash Q Learning algorithm on the grid and return one Q table per player. 
        return_history (bool) : if True, print all the changing positions of the players on the grid during the learning cycle.
        """
        current_state = [self.grid.players[0].position,self.grid.players[1].position]
        joint_states = self.grid.joint_states()
        player0_movements =  self.grid.players[0].movements
        player1_movements =  self.grid.players[1].movements
        stage_games0, stage_games1 = self.grid.create_stage_games()
        Q0, Q1 = self.grid.create_q_tables()
        transition_table = self.grid.create_transition_table()    
        state_tracker = [current_state]
        
        for i in tqdm(range(self.max_iter)):
            if current_state == joint_states[-1]:  #Both players reached the reward, return to original position
                current_state = [self.grid.players[0].position,self.grid.players[1].position]
                
            if self.decision_strategy == 'random':
                m0 = player0_movements[random.randrange(len(player0_movements))]
                m1 = player1_movements[random.randrange(len(player1_movements))]
            if self.decision_strategy == 'greedy':
                greedy_matrix0 = Q0[joint_states.index(current_state)]
                greedy_matrix1 = Q1[joint_states.index(current_state)]
                greedy_game =  nash.Game(greedy_matrix0,greedy_matrix1)
                equilibriums = list(greedy_game.support_enumeration())
                greedy_equilibrium = equilibriums[random.randrange(len(equilibriums))] #One random equilibrium
                if len(np.where(greedy_equilibrium[0] == 1)[0]) ==0: #No strict equilibrium found
                    m0 = player0_movements[random.randrange(len(player0_movements))] #Random move
                    m1 = player1_movements[random.randrange(len(player1_movements))]
                else:  #Select the movements corresponding to the nash equilibrium
                    m0 = player0_movements[np.where(greedy_equilibrium[0] == 1)[0][0]]
                    m1 = player1_movements[np.where(greedy_equilibrium[1] == 1)[0][0]]
            if self.decision_strategy == 'epsilon-greedy':
                random_number = random.uniform(0,1)
                if random_number >= self.epsilon: #greedy
                    greedy_matrix0 = Q0[joint_states.index(current_state)]
                    greedy_matrix1 = Q1[joint_states.index(current_state)]
                    greedy_game =  nash.Game(greedy_matrix0,greedy_matrix1)
                    equilibriums = list(greedy_game.support_enumeration())
                    greedy_equilibrium = equilibriums[random.randrange(len(equilibriums))] #One random equilibrium
                    if len(np.where(greedy_equilibrium[0] == 1)[0]) == 0: #No strict equilibrium found
                        m0 = player0_movements[random.randrange(len(player0_movements))] #Random move
                        m1 = player1_movements[random.randrange(len(player1_movements))]
                    else:  #Select the movements corresponding to the nash equilibrium
                        m0 = player0_movements[np.where(greedy_equilibrium[0] == 1)[0][0]]
                        m1 = player1_movements[np.where(greedy_equilibrium[1] == 1)[0][0]]          
                else: #random
                    m0 = player0_movements[random.randrange(len(player0_movements))]
                    m1 = player1_movements[random.randrange(len(player1_movements))]
              
            #Update state
            new_state = joint_states[transition_table[joint_states.index(current_state)][m0][m1]]
            #Solve Nash equilibrium problem in new state
            nash_eq_matrix0 = Q0[joint_states.index(new_state)]
            nash_eq_matrix1 = Q1[joint_states.index(new_state)]
            game = nash.Game(nash_eq_matrix0,nash_eq_matrix1)
            equilibriums = list(game.support_enumeration())
            best_payoff = -np.Inf
            equilibrium_values = []
            for eq in equilibriums:
                payoff =  game[eq][0] + game[eq][1]
                if payoff >= best_payoff:
                    best_payoff = payoff
                    equilibrium_values = game[eq]
                          
            #Q Tables update formula
            Q0[joint_states.index(current_state)][player0_movements.index(m0)][player1_movements.index(m1)] = (
                (1 - self.learning_rate) * Q0[joint_states.index(current_state)][player0_movements.index(m0)][player1_movements.index(m1)]
                + self.learning_rate * (stage_games0[joint_states.index(current_state)][player0_movements.index(m0)][player1_movements.index(m1)]
                                        + self.discount_factor * equilibrium_values[0])
            )

            Q1[joint_states.index(current_state)][player0_movements.index(m0)][player1_movements.index(m1)] = (
                (1 - self.learning_rate) * Q1[joint_states.index(current_state)][player0_movements.index(m0)][player1_movements.index(m1)]
                + self.learning_rate * (stage_games1[joint_states.index(current_state)][player0_movements.index(m0)][player1_movements.index(m1)]
                                        + self.discount_factor * equilibrium_values[1])
            )             
            
            current_state = new_state
            state_tracker.append(current_state)   
            
        if return_history:
            print(state_tracker)
        return Q0, Q1

    
    def get_best_policy(self,
                           Q0,
                           Q1):
        """
        Given two Q tables, one for each agent, return their best available path on the grid.
        """
        current_state = [self.grid.players[0].position,self.grid.players[1].position]
        joint_states = self.grid.joint_states()
        transition_table = self.grid.create_transition_table()
        player0_movements =  self.grid.players[0].movements
        player1_movements =  self.grid.players[1].movements
        policy0 = []
        policy1 = []
        while current_state != joint_states[-1]: #while the reward state is not reached for both agents
            print(current_state)
            q_state0 = Q0[joint_states.index(current_state)]
            q_state1 = Q1[joint_states.index(current_state)]
            game = nash.Game(q_state0,q_state1)
            equilibriums = list(game.support_enumeration())
            best_payoff = -np.Inf
            m0 = 'stay'
            m1 = 'stay'
            for eq in equilibriums:
                 if len(np.where(eq[0] == 1)[0]) != 0: #The equilibrium needs to be a strict nash equilibrium (no mixed-strategy)
                     total_payoff = q_state0[np.where(eq[0]==1)[0][0]][np.where(eq[1]==1)[0][0]] + q_state1[np.where(eq[0]==1)[0][0]][np.where(eq[1]==1)[0][0]]
                     if total_payoff >= best_payoff and (player0_movements[np.where(eq[0]==1)[0][0]] != 'stay'
                                                        or player1_movements[np.where(eq[1]==1)[0][0]] != 'stay'):
                                                        #payoff is better and at least one agent is moving
                         best_payoff = total_payoff
                         m0 = player0_movements[np.where(eq[0]==1)[0][0]]
                         m1 = player1_movements[np.where(eq[1]==1)[0][0]]           
            if current_state[0] != joint_states[-1][0]:
                policy0.append(m0)
            else : #target reached for player 0
                policy0.append('stay')
            if current_state[1] != joint_states[-1][1]:
                policy1.append(m1)
            else: #target reached for player 1
                policy1.append('stay')
            if current_state != joint_states[transition_table[joint_states.index(current_state)][m0][m1]]: #there was a movement
                current_state = joint_states[transition_table[joint_states.index(current_state)][m0][m1]]
            else :  #No movement, the model did not converge
                policy0 = 'model failed to converge to a policy'
                policy1 = 'model failed to converge to a policy'
                break        
        print(current_state)            
        return policy0, policy1
