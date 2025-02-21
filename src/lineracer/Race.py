"""This files contains classes that implement the race back-end."""

# external imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import warnings
import math
from typing import List

# internal imports
from lineracer.RaceTrack import RaceTrack
from lineracer.Vehicles import Vehicle

class Race:
    """A class to represent a race with multiple vehicles.

    Attributes:
        track (RaceTrack): The race track.
        vehicles (List[Vehicle]): The list of vehicles.
        cv_idx (int): The index of the current controllable vehicle.
    """
    def __init__(self, **kwargs):
        """Initialize a Race instance.

        Args:
            **track (RaceTrack): The race track. Defaults to a randomly generated track.
            **n_player (int): The number of players. Defaults to 1.
            **n_npc (int): The number of non-player vehicles. Defaults to 1.
            **vehicles (List[Vehicle]): The list of vehicles.
        """
        self.track: RaceTrack = kwargs.get('track', RaceTrack.generate_random_track(y_var=1))

        self.vehicles: List[Vehicle] = kwargs.get('vehicles', [])
        if len(self.vehicles) == 0:
            self.n_player = kwargs.get('n_player', 1)
            self.n_npc = kwargs.get('n_npc', 1)
            for i in range(self.n_player):
                self.vehicles.append(Vehicle(track=self.track, starting_grid_index=i, is_player=True))
            for i in range(self.n_npc):
                idx = i + self.n_player
                self.vehicles.append(Vehicle(track=self.track, starting_grid_index=idx, is_player=False))
        self.n_vehicles = len(self.vehicles)
        self.cv_idx = 0

    def set_track(self, track: RaceTrack):
        """Assign a track instance.

        Args:
            track (RaceTrack): The track instance to assign.
        """
        self.track = track
        for v in self.vehicles:
            v.set_track(track)

    def get_cv(self):
        """Get the current vehicle."""
        return self.vehicles[self.cv_idx]

    def next_cv(self):
        """Get the next vehicle."""
        self.cv_idx = (self.cv_idx + 1) % len(self.vehicles)
        return self.get_cv()
