import pygame
import math
import numpy as np
import random
import time

def checkpattern(col, row):
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

def drawSpirit(screen, col, row, type, myfont):
    """
    Draws a spirit at pos col, row of type = [E (empty), B (bomb), 1, 2, 3, 4, 5, 6]
    """
    if type == 'E':
        if checkpattern(col,row):
            c = (242, 244, 247)
        else:
            c = (247, 249, 252)
        pygame.draw.rect(screen, c, pygame.Rect(col*SIZEOFSQ, row*SIZEOFSQ, SIZEOFSQ, SIZEOFSQ))

    else:
        drawSpirit(screen, col, row, 'E', myfont)
        if type == 1:
            text = myfont.render("1", 1, (0, 204, 0))
        elif type == 2:
            text = myfont.render("2", 1, (255, 204, 0))
        elif type == 3:
            text = myfont.render("3", 1, (204, 0, 0))
        elif type == 4:
            text = myfont.render("4", 1, (0, 51, 153))
        elif type == 5:
            text = myfont.render("5", 1, (255, 102, 0))
        elif type == 6:
            text = myfont.render("6", 1, (255, 102, 0))
        elif type == 'flag':
            text = myfont.render("F", 1, (255, 0, 0))

        #Get the text rectangle and center it inside the rectangles
        textRect = text.get_rect()
        textRect.center = (col*SIZEOFSQ + int(0.5*SIZEOFSQ)),(row*SIZEOFSQ + int(0.5*SIZEOFSQ))
        screen.blit(text, textRect)

def findNeighbors2(y, x, grid): #Taken online, y = col x = row, return [(row,col),(row,col)]
    COLS = grid.shape[1]
    ROWS = grid.shape[0]
    neighbors = [(y2, x2) for x2 in range(x-1, x+2)
                               for y2 in range(y-1, y+2)
                               if (-1 < x < COLS and
                                  -1 < y < ROWS and
                                   (x != x2 or y != y2) and
                                   (0 <= x2 < COLS) and
                                   (0 <= y2 < ROWS))]
    return neighbors

def findNeighbors(rowin, colin, grid):
    """ Takes col, row and grid as input and returns as list of neighbors
    """
    COLS = grid.shape[1]
    ROWS = grid.shape[0]
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

def sumMines(grid, col, row):
    """ Finds amount of mines adjacent to a field.
    """
    mines = 0
    neighbors = findNeighbors(row, col, grid)
    for n in neighbors:
        if grid[n[0],n[1]] == 'B':
            mines = mines + 1
    return mines

def initBoard(screen, grid, startcol, startrow, mines):
    """ Initializes the board
    """
    #Randomly place bombs
    COLS = grid.shape[1]
    ROWS = grid.shape[0]

    while mines > 0:
        (row, col) = (random.randint(0, ROWS-1), random.randint(0, COLS-1))
        #if (col,row) not in findNeighbors(startcol, startrow, grid) and grid[col][row] != 'B' and (col, row) not in (startcol, startrow):
        if (row,col) not in findNeighbors(startrow, startcol, grid) and (row,col) != (startrow, startcol) and grid[row][col] != 'B':
            grid[row][col] = 'B'
            mines = mines - 1


    #Get rest of board when bombs have been placed
    for col in range(COLS):
        for row in range(ROWS):
            if grid[row][col] != 'B':
                totMines = sumMines(grid, col, row)
                if totMines > 0:
                    grid[row][col] = totMines
                else:
                    grid[row][col] = 'E'

    return grid


def printBoard(grid):
    COLS = grid.shape[1]
    ROWS = grid.shape[0]
    for row in range(0,ROWS):
        print(' ')
        for col in range(0,COLS):
            print(grid[row][col], end=' ')


def reveal(screen, grid, col, row, myfont, checked, press = "LM"):
    if press == "LM":
        if checked[row][col] != 0:
            return
        checked[row][col] = checked[row][col] + 1
        if grid[row][col] != 'B':
            #print(grid[row][col])
            drawSpirit(screen, col, row, grid[row][col], myfont)
            #pygame.display.flip()
            #time.sleep(0.2)
            #print(checked)
            #time.sleep(5)

            if grid[row][col] == 'E':
                neighbors = findNeighbors(row, col, grid)
                for n in neighbors:
                    if not checked[n[0],n[1]]: 
                        reveal(screen, grid, n[1], n[0], myfont, checked)
    elif press == "RM":
        drawSpirit(screen, col, row, "flag", myfont)


if __name__ == "__main__":

    ROWS = 6
    COLS = 6
    SIZEOFSQ = 100
    MINES = 6

    grid = np.zeros((ROWS,COLS), dtype=object)
    #color of squares
    c1 = (4, 133, 223)
    c2 = (4, 145, 223)


    pygame.init()
    pygame.font.init()
    myfont = pygame.font.SysFont("monospace-bold", 100)
    screen = pygame.display.set_mode((COLS * SIZEOFSQ, ROWS * SIZEOFSQ))

    rects = []

    #Initialize Game:
    for row in range(ROWS):
        for col in range(COLS):
            if checkpattern(col, row):
                c = c1
            else:
                c = c2

            rects.append(pygame.draw.rect(screen, c, pygame.Rect(col*SIZEOFSQ, row*SIZEOFSQ, SIZEOFSQ, SIZEOFSQ)))
    

    done = False
    firstClick = True
    while not done:
            for event in pygame.event.get(): #If someone clicks or does something
                #pygame.draw.rect(screen, (0, 128, 255), pygame.Rect(30, 30, 60, 60))
                if event.type == pygame.QUIT:
                        done = True
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    for i, rect in enumerate(rects):
                        if rect.collidepoint(pos):
                            #print(i)
                            col = i % COLS
                            row = math.floor(i/COLS)
                            print(row, col)

                            if firstClick:
                                grid = initBoard(screen, grid, col, row, MINES)
                                firstClick = False
                                printBoard(grid)


                            if pygame.mouse.get_pressed() == (1, 0, 0):
                                reveal(screen, grid, col, row, myfont, np.zeros_like(grid))

                            elif pygame.mouse.get_pressed() == (0, 0, 1):
                                reveal(screen, grid, col, row, myfont, np.zeros_like(grid), press = "RM")
                            """
                            neighbors = findNeighbors(col,row, grid)
                            for n in neighbors:
                                drawSpirit(screen, n[0], n[1], 'one', myfont)
                            """
            

            pygame.display.flip()