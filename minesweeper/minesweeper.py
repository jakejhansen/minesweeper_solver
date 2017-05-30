import pygame
import math

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
    Draws a spirit at pos col, row of type = [blank, bomb, one, two, three, four, five, six]
    """
    if type == 'blank':
        if checkpattern(col,row):
            c = (242, 244, 247)
        else:
            c = (247, 249, 252)

        pygame.draw.rect(screen, c, pygame.Rect(col*SIZEOFSQ, row*SIZEOFSQ, SIZEOFSQ, SIZEOFSQ))

    else:
        drawSpirit(screen, col, row, 'blank', myfont)
        if type == 'one':
            text = myfont.render("1", 1, (0, 204, 0))
        elif type == 'two':
            text = myfont.render("2", 1, (255, 204, 0))
        elif type == 'three':
            text = myfont.render("3", 1, (204, 0, 0))
        elif type == 'four':
            text = myfont.render("4", 1, (0, 51, 153))
        elif type == 'five':
            text = myfont.render("5", 1, (255, 102, 0))
        elif type == 'six':
            text = myfont.render("6", 1, (255, 102, 0))
        elif type == 'flag':
            text = myfont.render("F", 1, (255, 0, 0))

        #Get the text rectangle and center it inside the rectangles
        textRect = text.get_rect()
        textRect.center = (col*SIZEOFSQ + int(0.5*SIZEOFSQ)),(row*SIZEOFSQ + int(0.5*SIZEOFSQ))
        screen.blit(text, textRect)

if __name__ == "__main__":

    ROWS = 10
    COLS = 10
    SIZEOFSQ = 100

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
    while not done:
            for event in pygame.event.get():
                #pygame.draw.rect(screen, (0, 128, 255), pygame.Rect(30, 30, 60, 60))
                if event.type == pygame.QUIT:
                        done = True
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    for i, rect in enumerate(rects):
                        if rect.collidepoint(pos):
                            print(i)
                            col = i % COLS
                            row = math.floor(i/ROWS)
                            drawSpirit(screen, col, row, 'flag', myfont)
            
            pygame.display.flip()