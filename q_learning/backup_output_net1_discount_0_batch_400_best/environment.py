"""
This module contains class definitions for open ai gym environments.
"""

import random
import gym

from minesweeper_env import MinesweeperEnv
import numpy as np


class BaseEnvironment(object):
    """A base environment class for open ai gym.

    Attributes
    ----------
    _episode_step : int 
        The number of steps in the current episode (game).
    _episode_number : int
        The number of games played.
    _episode_reward : int
        The score in the current episode.
    _max_reward_episode : float
        The maximum reward per episode.
    _global_step : int
        The total number of steps globally.
    _global_reward : float
        The total score over all games.

    """


    def __init__(self):

        self._episode_step       = 0
        self._episode_number     = 0
        self._episode_reward     = 0.0
        self._max_reward_episode = 0.0
        self._global_step        = 0
        self._global_reward      = 0.0
        self._recent_reward      = 0.0
        self._recent_episode_number = 0.0


    def avg_reward_per_episode(self):
        """Computes the average reward per episode.

        Returns
        -------
        float
            The average reward per episode.

        """
        return self.global_reward/(self._episode_number+1)


    def avg_reward_per_episode_recent(self):
        """Computes the average reward per most recent episodes/games

        Returns
        -------
        float
            The average reward per episode.

        """

        return self._recent_reward/(self._recent_episode_number+1)


    def avg_steps_per_episode(self):
        """Computes the average number of steps per episode.

        Returns
        -------
        float
            The average number of steps per episode.

        """
        return self.global_step/(self._episode_number+1)


    @property
    def episode_number(self):
        """int: The current episode number."""
        return self._episode_number


    @property
    def episode_step(self):
        """int: The number of steps in the current episode."""
        return self._episode_step


    @property
    def global_step(self):
        """int: The total number of steps globally."""
        return self._global_step


    @property
    def episode_reward(self):
        """float: The score in the current episode."""
        return self._episode_reward


    @property
    def global_reward(self):
        """float: The total score over all episodes."""
        return self._global_reward


    @property
    def max_reward_per_episode(self):
        """float: The maximum reward per episode."""
        return self._max_reward_episode


class MinesweeperEnvironment(BaseEnvironment):

    """An environment class for open ai gym atari games using the screen.

    Parameters
    ----------
    random_start : int
        The maximum random number of steps to take in an environment initially.
    display : Bool
        If true, show the game being played.
    game : str
        The selected game.

    Attributes
    ----------
    _display : bool
        If true, show the game being played.
    _screen_rgb : :obj: 'ndarray' of :obj: 'float'
        The screen output (rgb)
    _reward : float
        The amount of reward achieved by the previous action. 
    _done : bool
        Whether it is time to reset the environment again. 
        Most (but not all) tasks are divided up into well-defined episodes, 
        and done being True indicates the episode has terminated.
    _random_start : int
        How long we let the agent take random actions in a new game.
    screen_height : int
        The height of the screen after resizing.
    screen_width : int
        The width of the screen after resizing.
    _action_repeat : int
        The number of time-steps an action is repeated.
    env : :obj:
        The open ai gym environment object.

    """


    def __init__(
        self, rows = 10, cols = 10, mines_min = 13, mines_max=15, display = False, reward_recent_update = 1000
    ):

        # Constructor of base class
        super(MinesweeperEnvironment, self).__init__() # No arguments to init in
                                                    # BaseEnvironment

        self._mines_min = mines_min
        self._mines_max = mines_max

        self._screen    = None
        self._display       = display
        self._reward        = 0.0
        self._done          = False
        self._random_start  = 10
        self.reward_recent_update = reward_recent_update

        self._games_won = 0
        self._recent_games_won = 0

        self.screen_height  = rows
        self.screen_width   = cols

        self._action_repeat = 1

        # Initialize Gym Environment with selected game
        self.env = MinesweeperEnv(self.screen_height, self.screen_width, 100, self._mines_min, self._mines_max, self._display)

    def new_game(self):
        """Creates a new game.

        Returns
        -------
        int
            0
        :obj: 'ndarray' of :obj: 'float'
            The current screen in grayscale and resized
        bool
            True for game is over, False otherwise

        """
        self._max_reward_episode = max(self._max_reward_episode, 
                                        self._episode_reward)

        self._episode_step   = 0
        self._episode_reward = 0.0

        # The number of games played
        self._episode_number += 1 # Since we may have ended a previous episode
                                  # and started a new game

        self._recent_episode_number += 1


        if not (self._recent_episode_number + 1) % self.reward_recent_update: # We count the most recent reward
            print("Resetting recent reward counter")
            self._recent_reward = 0
            self._recent_episode_number = 0
            self._recent_games_won = 0

        self._screen = self.env.reset()
        self.render()
        return self.state


    def new_random_game(self):
        """Creates a new game where we initially perform some random actions
           to get a different start to the game.

        Returns
        -------
        int
            0.
        :obj: 'ndarray' of :obj: 'float'
            The current screen in grayscale and resized.
        bool
            True for game is over, False otherwise.

        """

        return self.new_game()


    def _step(self, action):
        """Takes a step in the environment by giving an action and updating
        all related attributes.

        Returns
        -------
        None

        """

        res = self.env.step(action)
        self._screen = res[0]
        self._reward = res[1]
        self._done = res[2]
        self._episode_step += 1
        self._global_step  += 1
        self._episode_reward += self._reward
        self._global_reward  += self._reward

        self._recent_reward += self._reward

        if self._reward == self.env.rewards["win"]:
            self._games_won += 1
            self._recent_games_won += 1

    def _random_step(self):
        action = self.act_random()
        self._step(action)



    def after_act(self, action):
        """Actions to perform after an environment action.

        Returns
        -------
        None

        """

        self.render()


    def act(self, action, is_training=True):
        """Perform an action in an environment with both a step and whatever
        extra is defined in after_act().

        Returns
        -------
        :obj: 'ndarray' of :obj: 'float'
            state after action has been performed

        """

        # https://github.com/openai/gym/blob/master/gym/envs/atari/atari_env.py
        # The Gym environment for Breakout performs the following _step():
        #   num_steps = self.np_random.randint(2, 5)
        #   for _ in range(num_steps):
        #      reward += self.ale.act(action)

        self._step(action)
        self.after_act(action)
        return self.state


    def act_random(self):
        return np.random.randint(0, self.screen_width*self.screen_height-1, size=1)[0]


    def render(self):
        """Render the environment.

        Returns
        -------
        None

        """
        if self._display:
            self.env.drawState()


    @property
    def state(self):
        """:obj: 'list' of 
                :obj: 'ndarray' of :obj: float, 
                :obj: float, 
                :obj: bool: 

        Returns the state of the environment.
        """
        return self.screen, self._reward, self._done


    @property
    def screen(self):
        return self._screen

    @property
    def win_rate(self):
        return self._games_won / (self._episode_number+1)


    @property
    def win_rate_recent(self):
        return self._recent_games_won / (self._recent_episode_number+1)


    @property
    def num_actions(self):
        """int: The number of actions available in the selected game."""
        return int(self.screen_width*self.screen_height)


    @property
    def legal_actions(self):
        """int: Available legal actions."""
        return int(self.screen_width*self.screen_height)

    def action_to_string(self, action_id):
        """str: The meanings of actions."""
        return 'Nope'
