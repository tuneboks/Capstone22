import logging
from enum import Enum, IntEnum

import matplotlib.pyplot as plt
import numpy as np



size = 5
axes = [size, size, size]
data = np.random.choice(2, size=axes, p=[0.7, 0.3])
data2 =  np.random.choice(2, size=axes, p=[.9, .1])
#print(data2)

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
    actions = [Action.MOVE_LEFT, Action.MOVE_RIGHT, Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_FORWARD, Action.MOVE_BACK]  # all possible actions
    reward_exit = 10.0  # reward for reaching the exit cell
    penalty_move = -0.05  # penalty for a move which did not result in finding the exit cell
    penalty_visited = -0.25  # penalty for returning to a cell which was visited earlier
    penalty_impossible_move = -0.75  # penalty for trying to enter an occupied cell or moving out of the maze

    def __init__(self, maze, start_cell=(0, 0, 0), exit_cell=None):
        self.maze = maze

        self.__minimum_reward = -0.5 * self.maze.size  # stop game if accumulated reward is below this threshold

        nX, nY, nZ  = self.maze.shape
        self.cells = [(x, y, z) for x in range(nX) for y in range(nY) for z in range(nZ)]
        self.empty = [(x, y, z) for x in range(nX) for y in range(nY) for z in range(nZ) if self.maze[x, y, z] == Cell.EMPTY]
        self.__exit_cell = (nX - 1, nY - 1, nZ - 1) if exit_cell is None else exit_cell
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

        self.reset(start_cell)

    def reset(self, start_cell=(0, 0, 0)):
        """ Reset the maze to its initial state and place the agent at start_cell.
            :param tuple start_cell: here the agent starts its journey through the maze (optional, else upper left)
            :return: new state after reset
        """
        print("RESET CALLED")
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
            nrows, ncols, nchannels = self.maze.shape
            print("AX_1")
            print(self.__ax1)
            self.__ax1.clear()
            self.__ax1.set_xticks(np.arange(0.5, nrows, step=1))
            self.__ax1.set_xticklabels([])
            self.__ax1.set_yticks(np.arange(0.5, ncols, step=1))
            self.__ax1.set_yticklabels([])
            self.__ax1.set_zticks(np.arange(0.5, nchannels, step=1))
            self.__ax1.set_zticklabels([])
            #self.__ax1.axes(projection="3d")
            self.__ax1.plot3D(*self.__current_cell, "rs", markersize=30)  # start is a big red square
            self.__ax1.text(*self.__current_cell, "Start", ha="center", va="center", color="white")
            self.__ax1.plot3D(*self.__exit_cell, "gs", markersize=30)  # exit is a big green square
            self.__ax1.text(*self.__exit_cell, "Exit", ha="center", va="center", color="white")
            #self.__ax1.imshow(self.maze, cmap="binary")
            self.__ax1.get_figure().canvas.draw()
            self.__ax1.get_figure().canvas.flush_events()

        return self.__observe()

    def __observe(self):

            """ Return the state of the maze - in this game the agents current location.
                :return numpy.array [1][2]: agents current location
            """
            return np.array([[*self.__current_cell]])

    def render(self, content=Render.NOTHING, location = None):
        """ Record what will be rendered during play and/or training.
            :param Render content: NOTHING, TRAINING, MOVES
        """
        self.__render = content

        if self.__render == Render.NOTHING:
            if self.__ax1:
                self.__ax1.get_figure().close()
                self.__ax1 = None
            if self.__ax2:
                self.__ax2.get_figure().close()
                self.__ax2 = None
        if self.__render == Render.TRAINING:
            if self.__ax2 is None:
                fig, self.__ax2 = plt.subplots(1, 1, tight_layout=True)
                fig.canvas.set_window_title("Best move")
                self.__ax2.set_axis_off()
                self.render_q(None)
        if self.__render in (Render.MOVES, Render.TRAINING):
            if self.__ax1 is None:
                self.__ax1 = plt.figure().add_subplot(111, projection='3d')
                #fig, self.__ax1 = plt.subplots(1, 1, tight_layout=True)
                #fig.canvas.set_window_title("Maze")
                colors = plt.cm.plasma(self.maze)
                self.__ax1.voxels(self.maze, facecolors =colors ,edgecolors='black')
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # self.maze2 = self.maze.copy()
        # if location is not None:
        #     self.maze2[location[0],location[1],location[2]] = 100


        plt.show()

    def step(self, action):
        """ Move the agent according to 'action' and return the new state, reward and game status.
            :param Action action: the agent will move in this direction
            :return: state, reward, status
        """
        reward = self.__execute(action)
        self.__total_reward += reward
        status = self.__status()
        state = self.__observe()
        logging.debug("action: {:10s} | reward: {: .2f} | status: {}".format(Action(action).name, reward, status))
        return state, reward, status

    def __status(self):
        """ Return the game status.
            :return Status: current game status (WIN, LOSE, PLAYING)
        """
        if self.__current_cell == self.__exit_cell:
            return Status.WIN

        if self.__total_reward < self.__minimum_reward:  # force end of game after too much loss
            return Status.LOSE

        return Status.PLAYING

    def __possible_actions(self, cell=None):
        """ Create a list with all possible actions from 'cell', avoiding the maze's edges and walls.
            :param tuple cell: location of the agent (optional, else use current cell)
            :return list: all possible actions
        """
        if cell is None:
            col, row, channel = self.__current_cell
        else:
            print("cell: ",cell)
            col, row, channel = cell

        possible_actions = Maze.actions.copy()  # initially allow all

        # now restrict the initial list by removing impossible actions
        nrows, ncols, nchannel = self.maze.shape
        if row == 0 or (row > 0 and self.maze[row - 1, col, channel] == Cell.OCCUPIED):
            possible_actions.remove(Action.MOVE_UP)
        if row == nrows - 1 or (row < nrows - 1 and self.maze[row + 1, col, channel] == Cell.OCCUPIED):
            possible_actions.remove(Action.MOVE_DOWN)

        if col == 0 or (col > 0 and self.maze[row, col - 1, channel] == Cell.OCCUPIED):
            possible_actions.remove(Action.MOVE_LEFT)
        if col == ncols - 1 or (col < ncols - 1 and self.maze[row, col + 1, channel] == Cell.OCCUPIED):
            possible_actions.remove(Action.MOVE_RIGHT)

        if channel == 0 or (channel > 0 and self.maze[row, col, channel-1] == Cell.OCCUPIED):
            possible_actions.remove(Action.MOVE_FORWARD)
        if channel == nchannel - 1 or (channel < nchannel - 1 and self.maze[row, col, channel+1] == Cell.OCCUPIED):
            possible_actions.remove(Action.MOVE_BACK)

        return possible_actions

    def __execute(self, action):
        """ Execute action and collect the reward or penalty.
            :param Action action: direction in which the agent will move
            :return float: reward or penalty which results from the action
        """
        possible_actions = self.__possible_actions(self.__current_cell)
        print("V:",possible_actions )

        if not possible_actions:
            reward = self.__minimum_reward - 1  # cannot move anywhere, force end of game
        elif action in possible_actions:
            col, row, channel = self.__current_cell
            if action == Action.MOVE_LEFT:
                col -= 1
            elif action == Action.MOVE_UP:
                row -= 1
            if action == Action.MOVE_RIGHT:
                col += 1
            elif action == Action.MOVE_DOWN:
                row += 1
            elif action == Action.MOVE_FORWARD:
                channel -= 1
            elif action == Action.MOVE_BACK:
                channel += 1

            self.__previous_cell = self.__current_cell
            self.__current_cell = (col, row, channel)

            if self.__render != Render.NOTHING:
                self.__draw()

            if self.__current_cell == self.__exit_cell:
                reward = Maze.reward_exit  # maximum reward when reaching the exit cell
            elif self.__current_cell in self.__visited:
                reward = Maze.penalty_visited  # penalty when returning to a cell which was visited earlier
            else:
                reward = Maze.penalty_move  # penalty for a move which did not result in finding the exit cell

            self.__visited.add(self.__current_cell)
        else:
            reward = Maze.penalty_impossible_move  # penalty for trying to enter an occupied cell or move out of the maze

        return reward

    def __draw(self):
        """ Draw a line from the agents previous cell to its current cell. """
        self.__ax1.plot(*zip(*[self.__previous_cell, self.__current_cell]), "bo-")  # previous cells are blue dots
        self.__ax1.plot(*self.__current_cell, "ro")  # current cell is a red dot
        self.__ax1.get_figure().canvas.draw()
        self.__ax1.get_figure().canvas.flush_events()

    def play(self, model, start_cell=(0, 0, 0)):
        """ Play a single game, choosing the next move based a prediction from 'model'.
            :param class AbstractModel model: the prediction model to use
            :param tuple start_cell: agents initial cell (optional, else upper left)
            :return Status: WIN, LOSE
        """
        self.reset(start_cell)

        state = self.__observe()
        print("Play called")
        print(state)
        while True:
        #for x in range(20):
            action = model.predict(state=state)
            print("action: ",action)
            state, reward, status = self.step(action)
            if status in (Status.WIN, Status.LOSE):
                return state, reward, status
        #return state, reward, status #temporary


#ax = fig.add_subplot(111, projection='3d')

# (1) start drawing the maize

#maze.step()

# only show the maze
#game.render(Render.MOVES)
#game.reset()

'''
# play using random model
if test == Test.RANDOM_MODEL:
    game.render(Render.MOVES)
    model = models.RandomModel(game)
    game.play(model, start_cell=(0, 0))
'''
import random
# (2) random sample a step to Execute
#actions = [Action.MOVE_LEFT, Action.MOVE_RIGHT, Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_FORWARD, Action.MOVE_BACK]
#action = random.choice(actions)


import numpy as np
import models
from models import AbstractModel


class RandomModel(AbstractModel):
    """ Prediction model which randomly chooses the next action. """

    def __init__(self, game, **kwargs):
        super().__init__(game, name="RandomModel", **kwargs)

    def q(self, state):
        """ Return Q value for all actions for a certain state.
            :return np.ndarray: Q values
        """
        return np.array([0, 0, 0, 0])

    def predict(self, **kwargs):
        """ Randomly choose the next action.
            :return int: selected action
        """
        return random.choice(self.actions)


#print(action)
game = Maze(data2)
game.render(location=(2,1,1))
game.reset()

game.render(Render.MOVES)
model = models.RandomModel(game)
 # print('model: ', model)
# print(game.actions)
state, reward, status = game.play(model, start_cell=(0, 0, 0))
print("final: ",state, reward, status)
