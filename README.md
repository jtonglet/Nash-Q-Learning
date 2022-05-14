# Nash Q Learning 

Simple implementation of the Nash Q Learning  algorithm to solve games with two agents, as seen in the course Multiagent Systems at Politecnico di Milano. 
The Nash Q Learning algorithm was introduced in the paper [**Nash q-learning for general-sum stochastic games**](https://dl.acm.org/doi/10.5555/945365.964288) (Hu, J., Wellman, M.P., 2003).

![](img/img1.PNG)


## Solve the game with Nash Q Learning 

Prepare the grid 

```python
#Initialize the two players
player1 = Player([0,0])
player2 = Player([2,0])
#Initialize the grid
grid = Grid(length = 3,
            width = 3,
            players = [player1,player2],
           obstacle_coordinates = [[1,1]], #A single obstacle in the middle of the grid
           reward_coordinates = [1,2],
           reward_value = 20,
           collision_penalty = -1)
```

Train the Nash Q Learning algo 

```python
nashQ = NashQLearning(grid, 
                      max_iter = 2000,
                      discount_factor = 0.7,
                      learning_rate = 0.7,
                      epsilon = 0.5,
                     decision_strategy = 'epsilon-greedy')
#Retrieve the updated Q tables after fitting the algorithm
Q0, Q1 = nashQ.fit(return_history = False)
#Best path followed by each player given the values in the Q tables
p0, p1 = nashQ.get_best_policy(Q0,Q1)
```

```python
print('Player 0 follows the  policy : %s of length %s. Player 1 follows the  policy : %s of length %s.'%('-'.join(p0),len(p0),'-'.join(p1),len(p1)))
>>> Player 0 follows the  policy : up-up-right of length 3. Player 1 follows the  policy : up-up-left of length 3. 
```

## Installation

<p align="justify">
This repository requires Python 3.8. Requirements and dependencies can be installed using the following command.
  
 ```
 git clone https://github.com/jtonglet/Nash_Q_Learning.git
 cd Nash_Q_Learning
 pip install -r requirements.txt
 ```
  </p>
  



