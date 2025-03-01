
# external imports
import numpy as np
import warnings
import matplotlib.pyplot as plt

# internal imports
from lineracer.Race import RaceTrack

class Grid:
    """A class to represent a discrete grid.

    Attributes:
        dx (float): The x-spacing of the grid.
        dy (float): The y-spacing of the grid.
    """
    def __init__(self, dx: float = 0.25, dy: float = 0.25):
        """Initialize a Grid instance.

        Args:
            dx (float): The x-spacing of the grid. Defaults to 0.25.
            dy (float): The y-spacing of the grid. Defaults to 0.25.
        """
        self.dx = dx
        self.dy = dy

class Controller:
    """An abstract class representing a controller of a vehicle."""
    def __init__(self, **kwargs):
        """Instantiate the controller class."""
        self.u = None
        self.track: RaceTrack = kwargs.get('track', None)

    def set_track(self, track: RaceTrack):
        """Assign a track instance.

        Args:
            track (RaceTrack): The track instance to assign.
        """
        self.track = track

    def get_feasible_controls(self):
        """Get the feasible controls."""

    def is_feasible(self, u):
        """Check if a control is feasible."""

    def set_control(self, u):
        """Set the control for the vehicle."""
        if self.is_feasible(u):
            self.u = u
        else:
            warnings.warn(f"Control {u} is not feasible.")

    def compute_control(self, pos: np.ndarray, vel: np.ndarray):
        """Get the control for the vehicle based on the current state of the vehicle.

        Args:
            pos (np.ndarray): The current position of the vehicle.
            vel (np.ndarray): The current velocity of the vehicle.
        """

    def get_control(self):
        return self.u

class DiscreteController(Controller):
    """Implements a discrete version of a Controller class."""
    def __init__(self, **kwargs):
        """Instantiate the DiscreteController class.

        Args:
            **grid: The grid to be used for the feasible controls. Defaults to a 3x3 grid.
        """
        super().__init__(**kwargs)
        self.grid = kwargs.get('grid', Grid())
        self.horizon = kwargs.get('horizon', 4)
        self.controls = []
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                self.controls.append([i*self.grid.dx, j*self.grid.dy])

    def get_feasible_controls(self):
        return self.controls

    def is_feasible(self, u):
        return u in self.controls

    def compute_control(self, pos: np.ndarray, vel: np.ndarray):
        """Get the control for the vehicle based on the current state of the vehicle.

        A simple brute force implementation.
        Args:
            pos (np.ndarray): The current position of the vehicle.
            vel (np.ndarray): The current velocity of the vehicle.
        """

        best_progress = -np.inf
        best_state = None
        track = self.track

        states = [{'pos': pos,
                   'vel': vel,
                   'mp': track.project_to_middle_line(pos),
                   'u0': None,
                   'k': 0,
                   'progress': -np.inf,
                   'trajectory': [pos]}]

        print(f"Computing control for {pos} with velocity {vel}...")

        while len(states) > 0:
            while len(states) > 0:

                # get next state
                x = states.pop(0)

                if x['k'] == self.horizon or x['progress'] >= track.progress_map[track.get_finish_middle_point()]:
                    # check if current state is best
                    if x['progress'] > best_progress:
                        best_progress = x['progress']
                        best_state = x
                        print(f"... best progress: {best_progress}")
                    continue

                # if not at end of horizon add next states
                for uk in self.get_feasible_controls():
                    new_p = x['pos'] + x['vel'] + uk
                    new_mp = track.project_to_middle_line(new_p)
                    new_progress = track.progress_map[tuple(new_mp)]

                    # ignore states that don't make positive progress
                    if new_progress <= x['progress']:
                        continue

                    # ignore infeasible states
                    on_track, new_mp = self.track.line_on_track(
                        point1=x['pos'],
                        point2=new_p,
                        mp=x['mp']
                    )
                    if not on_track:
                        continue

                    traj = [p for p in x['trajectory']]
                    traj.append(new_p)
                    states.append({
                        'pos': new_p,
                        'vel': x['vel'] + uk,
                        'mp': new_mp,
                        'u0': uk if x['k'] == 0 else x['u0'],
                        'k': x['k'] + 1,
                        'progress': new_progress,
                        'trajectory': traj
                    })
        self.u = best_state['u0']
