import math
import numpy as np
import random
import time
import mss
from tkinter import *
from tkinter import font
import IPython


class Minesweeper(object):
    def __init__(self, ROWS = 10, FULL = True, COLS = 10, SIZEOFSQ = 100, MINES = 13, display = False, rewards = {"win" : 10, "loss" : -10, "progress" : 1, "noprogress" : -1, "YOLO" : -0.5}):
        """ Initialize Minesweeper
            Rows, Cols: int  - Number of rows and cols on the board
            SIZEOFSQ: pixels -  Determines the size of the window, reduce to get smaller window
            Mines: integer - Number of mines generated on the board
            display: bool - chooses weather to display the game with pygame
        """

        self.ROWS = ROWS
        self.COLS = COLS
        self.FULL = FULL
        self.MINES = MINES
        self.display = display
        self.rewards = rewards

        self.grid = np.zeros((self.ROWS, self.COLS), dtype=object)
        self.state = np.zeros((self.ROWS, self.COLS), dtype=object)
        self.state_last = np.copy(self.state)
        self.nbMatrix = np.zeros((ROWS, COLS), dtype=object)

        self.computeNeighbors() #Fills out the nbMatrix

        self.won = 0
        self.lost = 0
        if display: #Load pygame stuff

            #Scale to resolutions
            with mss.mss() as sct:
                img = np.array(sct.grab(sct.monitors[1]))
                self.SIZEOFSQ = int(SIZEOFSQ * img.shape[1] / 3840)
                SIZEOFSQ = self.SIZEOFSQ

            self.root = Tk()
            self.C = Canvas(self.root, bg="white", height= COLS * SIZEOFSQ - 1, width = ROWS * SIZEOFSQ - 1)



        self.initGame()

        if display:
            self.drawState()



    def drawState(self):
        c1 = "#0285DF"
        c2 = "#0491DF" 
        #Draw checked pattern
        for row in range(self.ROWS):
            for col in range(self.COLS):
                if self.checkpattern(col, row):
                    c = c1
                else:
                    c = c2

                self.C.create_rectangle(col*self.SIZEOFSQ,row*self.SIZEOFSQ, col*self.SIZEOFSQ + self.SIZEOFSQ ,row*self.SIZEOFSQ + self.SIZEOFSQ, fill=c, width=0)

        #Draw state
        for row in range(self.ROWS):
            for col in range(self.COLS):
                cell = self.state[row][col]
                if cell == 'E' or cell != 'U':
                    if self.checkpattern(col,row):
                        c = "#F2F4F7"
                    else:
                        c = "#F7F9FC"

                    self.C.create_rectangle(col*self.SIZEOFSQ,row*self.SIZEOFSQ, col*self.SIZEOFSQ + self.SIZEOFSQ ,row*self.SIZEOFSQ + self.SIZEOFSQ, fill=c, width=0)
                
                if cell != 'U' and cell !='E':    
                    if cell == 1:
                        c2 = "#00CC00"
                    elif cell == 2:
                        c2 = "#FFCC00"
                    elif cell == 3:
                        c2 = "#CC0000"
                    elif cell == 4:
                        c2 = "#003399"
                    elif cell == 5:
                        c2 = "#FF6600"
                    elif cell == 6:
                        c2 = "#FF6600"
                    elif cell == 'flag':
                        c2 = "#FF0000"


                    #num = np.random.randint(len(font.families()))
                    #print("({},{}) : {}".format(row,col,num))
                    f = self.C.create_text(col*self.SIZEOFSQ + int(0.5*self.SIZEOFSQ),row*self.SIZEOFSQ + int(0.5*self.SIZEOFSQ), \
                        font=('Nimbus Sans L', 24, "bold"), fill = c2) #101 pretty good font
                    self.C.itemconfigure(f, text=str(cell))

        self.C.pack()
        #self.root.mainloop()    


    def initGame(self):
        self.grid = self.initBoard(startcol = 2, startrow = 2)
        self.state = np.ones((self.ROWS, self.COLS), dtype=object) * 'U'
        self.state_last = np.copy(self.state)


        self.action((2,2)) #Hack alert, to start off with non empty board. Can be removed but then agent has to learn
                         #what to do when the board starts out empty. 

    def initBoard(self, startcol, startrow):
        """ Initializes the board """

        #random.seed(a=np.random.randint(0,3))
        COLS = self.COLS
        ROWS = self.ROWS
        grid = np.zeros((self.ROWS, self.COLS), dtype=object)
        mines = self.MINES

        #Randomly place bombs
        while mines > 0:
            (row, col) = (random.randint(0, ROWS-1), random.randint(0, COLS-1))
            #if (col,row) not in findNeighbors(startcol, startrow, grid) and grid[col][row] != 'B' and (col, row) not in (startcol, startrow):
            if (row,col) not in self.nbMatrix[startrow, startcol] and (row,col) != (startrow, startcol) and grid[row][col] != 'B':
                grid[row][col] = 'B'
                mines = mines - 1


        #Get rest of board when bombs have been placed
        for col in range(COLS):
            for row in range(ROWS):
                if grid[row][col] != 'B':
                    totMines = self.sumMines(col, row, grid)
                    if totMines > 0:
                        grid[row][col] = totMines
                    else:
                        grid[row][col] = 'E'


        return grid

    def computeNeighbors(self):
        """ Computes the neighbor matrix for quick lookups"""

        for row in range(self.ROWS):
            for col in range(self.COLS):
                self.nbMatrix[row][col] = self.findNeighbors(row, col)



    def findNeighbors(self, rowin, colin):
        """ Takes col, row and grid as input and returns as list of neighbors
        """
        COLS = self.grid.shape[1]
        ROWS = self.grid.shape[0]
        neighbors = []
        for col in range(colin-1, colin+2):
            for row in range(rowin-1, rowin+2):
                if (-1 < rowin < ROWS and 
                    -1 < colin < COLS and 
                    (rowin != row or colin != col) and
                    (0 <= col < COLS) and
                    (0 <= row < ROWS)):
                    neighbors.append((row,col))

        return neighbors


    def sumMines(self, col, row, grid):
        """ Finds amount of mines adjacent to a field.
        """
        mines = 0
        neighbors = self.nbMatrix[row, col]
        for n in neighbors:
            if grid[n[0],n[1]] == 'B':
                mines = mines + 1
        return mines


    def printState(self):
        """Prints the current state"""
        grid = self.state
        COLS = grid.shape[1]
        ROWS = grid.shape[0]
        for row in range(0,ROWS):
            print(' ')
            for col in range(0,COLS):
                print(grid[row][col], end=' ')


    def printBoard(self):
        """Prints the board """
        grid = self.grid
        COLS = grid.shape[1]
        ROWS = grid.shape[0]
        for row in range(0,ROWS):
            print(' ')
            for col in range(0,COLS):
                print(grid[row][col], end=' ')


    def reveal(self, col, row, checked, press = "LM"):
        """Finds out which values to show in the state when a square is pressed
           Checked : np.array((row,col)) to check which squares has already been checked
                     If the field is not a bomb we want to reveal it, if the field is empty
                     we want to find it's neighbors and reveal them too if they are not a bomb.   
        """
        if press == "LM":
            if checked[row][col] != 0:
                return
            checked[row][col] = checked[row][col] + 1
            if self.grid[row][col] != 'B':

                #Reveal to state space
                self.state[row][col] = self.grid[row][col]

                if self.grid[row][col] == 'E':
                    neighbors = self.nbMatrix[row, col]
                    for n in neighbors:
                        if not checked[n[0],n[1]]: 
                            self.reveal(n[1], n[0], checked)
        
        elif press == "RM":
            #Draw flag, not used for agent
            pass



    def action(self, a):
        """ External action, taken by human or agent
            row,col: integer - where the agent want to press
        """

        
        #If press a bomb game over, start new game and return bad reward, -10 in this case
        row, col = a[0], a[1]
        if self.grid[row][col] == "B":
            self.lost += 1
            #self.initGame()
            return({"s" : np.copy(self.state), "r" : self.rewards['loss'], "d" : True})

        #Take action and reveal new state
        self.reveal(col, row , np.zeros_like(self.grid))
        if self.display == True:
            self.drawState()

        #Winning condition
        if np.sum(self.state == "U") == self.MINES:
            self.won += 1
            #self.initGame()
            return({"s" : np.copy(self.state), "r" :  self.rewards['win'], "d" : True})

        #Get the reward for the given action
        reward = self.compute_reward(a)

        #if reward == self.rewards['noprogress']:
        #    self.lost += 1
        #    return({"s" : np.copy(self.state), "r" : self.rewards['loss'], "d" : True})

        #return the state and the reward
        return({"s" : np.copy(self.state), "r" : reward, "d" : False})


    def compute_reward(self, a):
        """Computes the reward for a given action"""

        #Reward = 1 if we get less unknowns, 0 otherwise 
        if (np.sum(self.state_last == 'U') - np.sum(self.state == 'U')) > 0:
            reward =  self.rewards['progress']
        else:
            reward =  self.rewards['noprogress']

        #YOLO -> it it clicks on a random field with unknown neighbors
        tot = 0
        for n in self.nbMatrix[a[0],a[1]]:
            if self.state_last[n[0],n[1]] == 'U':
                tot += 1
        if tot == len(self.nbMatrix[a[0],a[1]]):
            reward = self.rewards['YOLO']

        self.state_last = np.copy(self.state)
        return(reward)
            


    def checkpattern(self, col, row):
        #Function to construct the checked pattern in pygame
        if row % 2:
            if col % 2: #If unequal
                return True
            else: #if equal
                return False
        else: 
            if col % 2: #If unequal
                return False
            else: #if equal
                return True

    def initPattern(self):
        #Initialize pattern:

        c1 = "#0285DF"
        c2 = "#0491DF"          
        rects = []
        for row in range(self.ROWS):
            for col in range(self.COLS):
                if self.checkpattern(col, row):
                    c = c1
                else:
                    c = c2

                self.C.create_rectangle(col*self.SIZEOFSQ,row*self.SIZEOFSQ, col*self.SIZEOFSQ + self.SIZEOFSQ ,row*self.SIZEOFSQ + self.SIZEOFSQ, fill=c)
                self.C.pack()
                self.root.mainloop()


    def get_state(self):
        #Returns the internal representation of the state
        return np.copy(self.state)


    def stateConverter(self, state):
        """ Converts 2d state to one-hot encoded 3d state
            input: state (rows x cols)
            output: state3d (row x cols x 10) (if full)
                            (row x cols x 2) (if not full)
        """
        rows, cols = state.shape
        if self.FULL:
            res = np.zeros((rows,cols,10), dtype = int)
            for i in range(0,8):
                res[:,:,i] = state == i+1 #1-7
            res[:,:,8] = state == 'U'
            res[:,:,9] = state == 'E'
           
            return(res)
        else:
            res = np.zeros((rows, cols, 2), dtype = int)
            filtr = ~np.logical_or(state == "U", state == "E") #Not U or E
            res[filtr,0] = state[filtr]
            res[state == "U", 1] = 1
            
            return(res)

    def get_validMoves(self):
        return(self.state == "U") #All unknowns are valid moves


    # Wrap to openai gym API
    def step(self, a):
        a = np.unravel_index(a, (self.ROWS,self.COLS))
        d2 = self.get_state()
        d = self.action(a)
        d["s"] = np.reshape(self.stateConverter(d["s"]),(self.ROWS*self.COLS*2))
        return d["s"], d["r"], d["d"], None

    def reset(self):
        self.initGame()
        return np.reshape(self.stateConverter(self.state),(self.ROWS*self.COLS*2))


if __name__ == "__main__":
    import time
    game = Minesweeper(display=True, ROWS = 6, COLS = 6, MINES = 7)
    game.printState()

    i = 0
    #start = time.time()
    while True:
        inp = input("Enter input (ROW,COL)")
        if inp == 'bb':
            import IPython
            IPython.embed()
        row = int(inp[1])
        col = int(inp[3])
        v = game.action((row, col))
        game.printState()
        print("\nReward = {}".format(v["r"]))
        if v["d"]:
        	game.reset()

        """
        #Test how fast it can run:
        i += 1
        print(i)
        act = [np.random.randint(0,10), np.random.randint(0,10)]
        env = game.action(act[0],act[1])
        state = stateConverter(env['state'])
        reward = env['reward']
        if i >= 1000:
            break
        """

    #print("Took: " + str(time.time()-start))

