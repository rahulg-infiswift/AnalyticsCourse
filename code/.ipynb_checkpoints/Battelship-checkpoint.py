# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 16:00:28 2017

@author: rahul.garg
"""

from random import randint
import pandas as pd

board = []

#board.append(2)

for x in range(0, 5):
#    print(x)
    board.append(["O"] * 5)

def print_board(board):
  for row in board:
    print (" ".join(row))

print_board(board)

def random_row(board):
  return randint(0, len(board) - 1)

def random_col(board):
  return randint(0, len(board[0]) - 1)

ship_row = random_row(board)
ship_col = random_col(board)

# print(ship_row,ship_col)

# Everything from here on should be in your for loop
# don't forget to properly indent!
for turn in range(4):  
    print ("Turn", turn + 1)
    guess_row = int(input("Guess Row: "))
    guess_col = int(input("Guess Col: "))
    
    if guess_row == ship_row and guess_col == ship_col:
        print ("Congratulations! You sank the battleship!")
        break
    else:      
        if guess_row not in range(5) or guess_col not in range(5):    
            print ("Oops, that's not even in the ocean.")
        elif board[guess_row][guess_col] == "X":
            print( "You guessed that one already." )
        else:
            print ("You missed the battleship!")
            board[guess_row][guess_col] = "X"
            print_board(board)
            if turn == 3:    
                print ("Game Over")