# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional, Tuple

import numpy as np

from cholerama import Positions, helpers

AUTHOR = "YeastieBoys"  # This is your team name
SEED = None  # Set this to a value to make runs reproducible


class Bot:
    """
    This is the bot that will be instantiated for the competition.

    The pattern can be either a numpy array or a path to an image (white means 0,
    black means 1).
    """

    def __init__(
        self,
        number: int,
        name: str,
        patch_location: Tuple[int, int],
        patch_size: Tuple[int, int],
    ):
        """
        Parameters:
        ----------
        number: int
            The player number. Numbers on the board equal to this value mark your cells.
        name: str
            The player's name
        patch_location: tuple
            The i, j row and column indices of the patch in the grid
        patch_size: tuple
            The size of the patch
        """
        self.number = number  # Mandatory: this is your number on the board
        self.name = name  # Mandatory: player name
        self.color = None  # Optional
        self.patch_location = patch_location
        self.patch_size = patch_size

        self.rng = np.random.default_rng(SEED)

        # If we make the pattern too sparse, it just dies quickly
        xy = self.rng.integers(0, 12, size=(2, 100))
        self.pattern = Positions(
            x=xy[1] + patch_size[1] // 2, y=xy[0] + patch_size[0] // 2
        )
        # The pattern can also be just an image (0=white, 1=black)
        # self.pattern = "mypattern.png"

    def iterate(
        self, iteration: int, board: np.ndarray, patch: np.ndarray, tokens: int
    ) -> Optional[Positions]:
        """
        This method will be called by the game engine on each iteration.

        Parameters:
        ----------
        iteration : int
            The current iteration number.
        board : numpy array
            The current state of the entire board.
        patch : numpy array
            The current state of the player's own patch on the board.
        tokens : list
            The list of tokens on the board.

        Returns:
        -------
        An object containing the x and y coordinates of the new cells.
        """
        if tokens >= 5:
            # Pick a random empty region of size 3x3 inside my patch
            empty_regions = helpers.find_empty_regions(patch, (3, 3))
            nregions = len(empty_regions)
            if nregions == 0:
                return None
            # Make a glider
            ind = self.rng.integers(0, nregions)
            x = np.array([1, 2, 0, 1, 2]) + empty_regions[ind, 1]
            y = np.array([2, 1, 0, 0, 0]) + empty_regions[ind, 0]
            return Positions(x=x, y=y)
