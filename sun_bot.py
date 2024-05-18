# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional, Tuple

import numpy as np

from cholerama import Positions, helpers

AUTHOR = "SourDough"  # This is your team name
SEED = None  # Set this to a value to make runs reproducible

def _rotate_90(pattern: Positions) -> Positions:
    return Positions(
        x=pattern.y[::-1], y=pattern.x[::-1]
    )


def rotate_pattern(pattern: Positions, n_90: int) -> Positions:
    """
    Rotate the pattern 90 degrees n_90 times.
    """
    if n_90%4 > 0:
        return rotate_pattern(_rotate_90(pattern), n_90-1)
    return _rotate_90((pattern))


_FENTOMIO = Positions(  # fentomino pattern 
    x = np.array([1, 0, 1, 1, 2]),
    y = np.array([0, 1, 1, 2, 2])
)

def place_pattern(pattern: Positions, empty_space: np.ndarray) -> Positions:
    return Positions(
        x = pattern.x + empty_space[1],
        y = pattern.y + empty_space[0],
    )


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
        self.pattern = "start_pattern.png"

    def count_my_cells(self, board: np.ndarray) -> int:
        return sum(sum(board == self.number))

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
        empty_regions = helpers.find_empty_regions(patch, (3, 3))
        nregions = len(empty_regions)
        ind = self.rng.integers(0, nregions)
        if nregions == 0:
            return None

        if tokens >= 5 and (self.count_my_cells(board) < 200): # I'm dying!
            return place_pattern(
                rotate_pattern(_FENTOMIO, self.rng.integers(0, 4)), empty_regions[ind]
            )
        elif tokens >= 5 and (self.rng.integers(0, 10) <= int(iteration/400)):
            return place_pattern(
                rotate_pattern(_FENTOMIO, self.rng.integers(0, 4)), empty_regions[ind]
            )
