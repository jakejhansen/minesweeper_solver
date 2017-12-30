# Minesweeper Solver - Using Deep Reinforcement Learning

This repository contains:
* The minesweeper game: Packed into a class that can be imported and actions can be taken on the game object which will return a new game state and a reward. 
* Different agents: We have agents using both policy gradients, DQN and evolutionary strategies

## How to run
### Create Anaconda virtual environment + required packages
In order to run the contents of this notebook, it is recommended to download anaocnda for your platform here
https://www.anaconda.com/download/
and run the following in your terminal to setup everything you need. See file `requirements.txt` for full package list with version numbering.

`conda create --name deep python=3.5.4`

`source activate deep`

`conda install jupyter`

`python -m ipykernel install --user --name deep --display-name "Python (deep)"`

`conda install -c conda-forge keras`

`conda install -c conda-forge opencv`

`conda install matplotlib=2.0.2`

`conda install pandas`

`pip install gym`

`pip install mss`

Run jupyter notebook using `jupyter notebook`.

Manually change kernel to `Python (deep)` within the jupyter notebook if not already running.



### Base
`minesweeper_pygame.py`: older implementation using pygame, here you can input the choise in the console as (row,col) which an agent can also use to interact with the program.

`minesweeper_tk.py`: Up to date version which can display the game using the Tk library which proved to work best multi-platform. Again you can input an action as (row,col) in the console.

### With an agent
Import **minesweeper_tk.py** and create a object where your agent lives.
If you want to display the output using pygame, set display=True.

Minimal example:
```python
env = Minesweeper(display=True)
state = env.get_state()

  while True:
      action = agent_get_action(state) #Get action from your agent
      state, reward, done, _ = env.step(action) #Return the state and reward for the given action
      if done:
          env.reset()
          state = env.get_state()
```

### Play interactively with the mouse
Run the script `minesweeper_old.py`. This is an older version but it alows you to play classic minesweeper using the mouse. 

### Input Arguments
The game takes the following input arguments:
* `ROWS`: Number of rows in the game (default 10)
* `COLS`: Number of columns in the game (default 10)
* `OUT`: Which state representation is returned, can either be "FULL", "CONDENSED" or "IMAGE" (see desp below) (default "FULL")
* `SIZEOFSQ`: The size of squares in pixels. The game is scaled according to your screen resolution but you can adjust this to increase or decrease the size. (default 100)
* `MINES`: Number of mines on the field (default 13)
* `interval`: If you want to generate games with varying number of mines, you can use this to sample uniformly around #MINES, i.e if MINES = 5 and interval = 2, your games will contain 3-7 mines chosen uniformly. (defualt 0)
* `display`: True to display the game, false to just play it. (default False, leave false under training for much faster games)
* 'rewards': The reward structure as dictionary. (default: rewards = {"win" : 1, "loss" : -1, "progress" : 0.9, "noprogress" : -0.3, "YOLO" : -0.3})

### State Representationi
The game can return 3 different representation of the current board state:
#### FULL
Has the dimensions ROWS * COLS * 10 and is a one-hot encoded representation of the state. First 8 channels cointains the ont-hot encoded representation of the integers on the board, i.e channel 3 has a 1, if the field = 3, zero otherwise.

The 9. channel has 1 if the field is unknown, 0 otherwise.
The 10. channel has 1 if the field is empty (i.e neither unknown or a number), 0 otherwise.

#### CONDENSED
Has the deimensions ROWS * COLS * 2 and is a condensed versioin of the FULL representation with the first 8. channels merged. 

The 1. channel has 0 if the field is unknown or empty, and value / 4 otherwise. 
The 2. channel has 1 if unknown, 0 otherwise.

Code used to generate CONDENSED representation:
```python
res = np.zeros((rows, cols, 2))
filtr = ~np.logical_or(state == "U", state == "E") #Not U or E
res[filtr,0] = state[filtr] / 4
res[state == "U", 1] = 1
```

#### IMAGE
Is the image representation of the board and has the dimension ROWS * COLS * 1
Unknown is -1
Empty is 0
Integer is integer / 8

```python
res = np.zeros((rows, cols,1))
res[state == "U", 0] = -1
res[state == "E", 0] = 0
filtr = ~np.logical_or(state == "U", state == "E") #Not U or E
res[filtr, 0] = state[filtr] / 8
```

### Reward Structure:
The program has the following default reward structure, that proved to work for our methods:
`rewards = {"win" : 1, "loss" : -1, "progress" : 0.9, "noprogress" : -0.3, "YOLO" : -0.3}`

The rewards are given by the conditions:
* win: win the game
* loss: loss the game
* progress: clear an unknown field, thus making progression towards solving the board
* noprogress: clicks an already discovered field, takes and action but doesn't make progress
* YOLO: clicks an unknown field where all the neighbors are unknown, i.e it has no information if the field is a bomb or not. This loss was needed because when the mine density is low, the agent can figure out that it can just click on random fiels and collect the progress reward, with little chance to actually hit a bomb. 
