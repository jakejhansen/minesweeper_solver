# Minesweeper Solver - Using Deep Reinforcement Learning

This repository contains:
* The minesweeper game: Packed into a class that can be imported and actions can be taken on the game object which will return a new game state and a reward. 
* Deep Reinforcement Learning Agent: An agent capable of solving minesweeper through actions determined by its policy. By playing a lot of games it will learn a policy through policy gradient descent.

## How to run
Install `pygame and numpy`


### Base
Run the script `minesweeper.py`, you can then input actions in the console as (row,col) e.g (3,2)


### With an agent
Import **minesweeper.py** and create a object where your agent lives.
If you want to display the output using pygame, set display=True.

Minimal example:
```python
game = Minesweeper(display=True)
state = game.get_state()

  while True:
      act = agent_get_action(state) #Get action from your agent
      res = game.action(row, col)
      state = res["s"]
      reward = res["r"]
```

### Play interactively with the mouse
Run the script `minesweeper_old.py`. This is an older version but it alows you to play classic minesweeper using the mouse. 
