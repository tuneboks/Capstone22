
import logging
from enum import Enum, IntEnum

import matplotlib.pyplot as plt
import numpy as np


class Cell(IntEnum):
    EMPTY = 0  # indicates empty cell where the agent can move to
    OCCUPIED = 1  # indicates cell which contains a wall and cannot be entered
    CURRENT = 2  # indicates current cell of the agent


class Action(IntEnum):
    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    MOVE_UP = 2
    MOVE_DOWN = 3
    MOVE_FORWARD = 4
    MOVE_BACK = 5


class Render(Enum):
    NOTHING = 0
    TRAINING = 1
    MOVES = 2


class Status(Enum):
    WIN = 0
    LOSE = 1
    PLAYING = 2

class Maze:
    actions = [Action.MOVE_LEFT, Action.MOVE_RIGHT, Action.MOVE_UP, Action.MOVE_DOWN, Action.FORWARD, Action.BACK]  # all possible actions
    reward_exit = 10.0  # reward for reaching the exit cell
    penalty_move = -0.05  # penalty for a move which did not result in finding the exit cell
    penalty_visited = -0.25  # penalty for returning to a cell which was visited earlier
    penalty_impossible_move = -0.75  # penalty for trying to enter an occupied cell or moving out of the maze

    def __init__(self, maze, start_cell=(0, 0, 0), exit_cell=None):
        self.maze = maze

        self.__minimum_reward = -0.5 * self.maze.size  # stop game if accumulated reward is below this threshold

        nrows, ncols = self.maze.shape
        self.cells = [(col, row) for col in range(ncols) for row in range(nrows)]
        self.empty = [(col, row) for col in range(ncols) for row in range(nrows) if self.maze[row, col] == Cell.EMPTY]
        self.__exit_cell = (ncols - 1, nrows - 1) if exit_cell is None else exit_cell
        self.empty.remove(self.__exit_cell)

        # Check for impossible maze layout
        if self.__exit_cell not in self.cells:
            raise Exception("Error: exit cell at {} is not inside maze".format(self.__exit_cell))
        if self.maze[self.__exit_cell[::-1]] == Cell.OCCUPIED:
            raise Exception("Error: exit cell at {} is not free".format(self.__exit_cell))

        # Variables for rendering using Matplotlib
        self.__render = Render.NOTHING  # what to render
        self.__ax1 = None  # axes for rendering the moves
        self.__ax2 = None  # axes for rendering the best action per cell
        self.__ax3 = None

        self.reset(start_cell)
        
         self.__ax1.set_xticks(np.arange(0.5, nrows, step=1))
            self.__ax1.set_xticklabels([])
    def reset(self, start_cell=(0, 0, 0)):
        """ Reset the maze to its initial state and place the agent at start_cell.
            :param tuple start_cell: here the agent starts its journey through the maze (optional, else upper left)
            :return: new state after reset
        """
        if start_cell not in self.cells:
            raise Exception("Error: start cell at {} is not inside maze".format(start_cell))
        if self.maze[start_cell[::-1]] == Cell.OCCUPIED:
            raise Exception("Error: start cell at {} is not free".format(start_cell))
        if start_cell == self.__exit_cell:
            raise Exception("Error: start- and exit cell cannot be the same {}".format(start_cell))

        self.__previous_cell = self.__current_cell = start_cell
        self.__total_reward = 0.0  # accumulated reward
        self.__visited = set()  # a set() only stores unique values

        if self.__render in (Render.TRAINING, Render.MOVES):
            # render the maze
            nrows, ncols = self.maze.shape
            self.__ax1.clear()
            self.__ax1.set_xticks(np.arange(0.5, nrows, step=1))
            self.__ax1.set_xticklabels([])
            self.__ax1.set_yticks(np.arange(0.5, ncols, step=1))
            self.__ax1.set_yticklabels([])
            self.__ax1.set_zticks(np.arange(0.5, nrows, step=1))
            self.__ax1.set_zticklabels([])
            self.__ax1.grid(True)
            self.__ax1.plot(*self.__current_cell, "rs", markersize=30)  # start is a big red square
            self.__ax1.text(*self.__current_cell, "Start", ha="center", va="center", color="white")
            self.__ax1.plot(*self.__exit_cell, "gs", markersize=30)  # exit is a big green square
            self.__ax1.text(*self.__exit_cell, "Exit", ha="center", va="center", color="white")
            self.__ax1.imshow(self.maze, cmap="binary")
            self.__ax1.get_figure().canvas.draw()
            self.__ax1.get_figure().canvas.flush_events()

        return self.__observe()
