
# external imports
import numpy as np
import warnings
from typing import Tuple

# internal imports
from lineracer.Race import RaceTrack
from lineracer.Controllers import Controller, DiscreteController

vehicle_colors = [
    "#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628",
    "#984ea3", "#999999", "#e41a1c", "#dede00"
]

class Vehicle:
    """A class to represent a vehicle.

    Attributes:
        track (RaceTrack): The race track the vehicle is on.
        position (np.array): The current position of the vehicle.
        velocity (np.array): The current velocity of the vehicle.
        u (np.array): The current control action.
        color: The color of the vehicle.
        marker: The marker style for plotting.
        controller (Controller): The controller for the vehicle. Defaults to DiscreteController.
        starting_grid_index (int): The starting grid of the vehicle
    """
    def __init__(self, track = None, position=None, velocity=(0., 0.), marker='o', **kwargs):
        """Initialize a Vehicle instance.

        Args:
            *track (RaceTrack): The race track.
            *position (np.array): The initial position of the vehicle.
            *velocity (np.array): The initial velocity of the vehicle.
            *marker: The marker style for plotting. Defaults to 'o'.
            **controller (Controller): The controller for the vehicle. Defaults to DiscreteController.
            **starting_grid_index (int): The starting grid of the vehicle. Defaults to 0.
            **color: The color of the vehicle. Defaults to a color based on the starting grid.
            **is_player (bool): Whether the vehicle is controlled by the player. Defaults to True.
        """
        self.track: RaceTrack = track
        self.starting_grid_index = kwargs.get('starting_grid_index', 0)
        self.is_player = kwargs.get('is_player', False)
        if position is not None:
            self.position = np.array(position)
        else:
            self.position = self.get_start_point()
        self.velocity = np.array(velocity)

        if track is not None:
            self.mid_line_point = self.track.get_start_middle_point()

        self.color: str = kwargs.get('color', vehicle_colors[self.starting_grid_index])
        self.marker = marker
        self.controller: Controller = kwargs.get('controller', None)
        if self.controller is None:
            self.controller = DiscreteController(track=self.track)

        self.trajectory = np.array(self.position).reshape(2,1)

    def set_track(self, track):
        """Assign a track instance.

        Args:
            track (RaceTrack): The track instance to assign.
        """
        self.track = track
        self.controller.set_track(track)

    def check_collision(self, previous_position = None) -> Tuple[bool, tuple]:
        """Check if the vehicle has collided.

        If no previous position is passed, then we check if the current position is on the track.
        Otherwise, we check if the vehicle has collided while moving from the previous position to
        the current position by checking points along the path.
        Args:
            previous_position: The previous position of the vehicle. Defaults to None.
            grid_test_size: The size of the grid to test along the path. Defaults to 0.1.
        Returns:
            bool: True if the vehicle has collided, False otherwise.
            tuple: The mid-line point verifying that the last point is on the track. None, if the
                test fails.
        """

        if self.track is None:
            return False, None

        if previous_position is None:
            on_track, mp = self.track.on_track_after(
                self.position,
                middle_line_point=self.mid_line_point
            )
            return not on_track, mp

        # project the previous position to the middle line
        on_track, mp = self.track.line_on_track(
            point1=previous_position,
            point2=self.position,
            mp=self.mid_line_point
        )
        return not on_track, mp

    def get_feasible_controls(self):
        """Get the feasible control actions for the vehicle.

        Gets the feasible control actions from the controller (return type depends on controller).
        """
        return self.controller.get_feasible_controls()

    def get_start_point(self):
        """Get the starting point of the vehicle.

        If no track is assigned, the starting point is the origin. Otherwise, we get the starting
        point from the track based on our starting grid index.
        """
        if self.track is None:
            return np.zeros(2)
        return self.track.get_start_point(self.starting_grid_index)

    def reset(self):
        """Reset the vehicle.

        Velocity is set to zero and position is reset to the start position of the track. If no
        track is assigned, the position is set to the origin. The trajectory is reset as well.
        """
        self.velocity = np.array([0., 0.])
        self.position = self.get_start_point()
        self.trajectory = np.array(self.position).reshape(2, 1)

        if self.track is not None:
            self.mid_line_point = self.track.get_start_middle_point()

    def update(self):
        """Update the vehicle position based on the current control action.

        Update the position and velocity of the vehicle:
            p = p + v
            v = v + u
        If no control action is provided, the vehicle will continue with its current velocity.
        The trajectory is updated with the new position. If the vehicle has collided during the
        update, we reset the vehicle.
        """
        if self.controller.u is not None:
            self.velocity += np.array(self.controller.u)
        else:
            warnings.warn("No control action provided. Vehicle will continue with current velocity.")

        self.position += self.velocity
        self.trajectory = np.hstack([self.trajectory, self.position.reshape(2,1)])

        # check if we have collided and reset if we have
        collided, mp = self.check_collision(previous_position=self.trajectory[:,-2])
        self.mid_line_point = mp
        if collided:
            self.reset()
