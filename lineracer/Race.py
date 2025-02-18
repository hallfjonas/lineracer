"""This files contains classes that implement the race back-end."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

def smooth_line(points):
    """Smooth a line using interpolation.

    Args:
        points: List of (x, y) tuples defining the line.
    """

class RaceTrack:
    """A class representing a race track.

    Attributes:
        middle_line: The middle line of the track.
        left_boundary: The left boundary of the track.
        right_boundary: The right boundary of the track.
        start_middle_point: The mid-line point on the start line.
        width: The width of the track.
        directions: The directions of the track segments.
        progress_map: A map of middle line points to progress.
        i_map: A map of middle line points to index
    """
    def __init__(self, middle_line, left_boundary, right_boundary, start_middle_point, width):
        """Initializes the race track.

        Args:
            middle_line: List of (x, y) tuples defining the middle line.
            left_boundary: List of (x, y) tuples defining the left boundary.
            right_boundary: List of (x, y) tuples defining the right boundary.
            start_middle_point: (x, y) tuple marking the mid-line point on the start line.
            width: Width of the track.
        """
        self.middle_line = middle_line
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary
        self.directions = [middle_line[i+1,:] - middle_line[i,:] for i in range(len(middle_line) - 1)]
        for i in range(len(self.directions)):
            self.directions[i] /= np.linalg.norm(self.directions[i])
        self.start_middle_point = start_middle_point
        self.width = width
        self.progress_map = {tuple(point): i / len(middle_line) for i, point in enumerate(middle_line)}
        self.i_map = {tuple(point): i for i, point in enumerate(middle_line)}

    def is_on_track(self, point) -> bool:
        """Check if a given point lies within the track boundaries.

        Args:
            point: The point to check.
        """
        return self.distance_to_middle_line(point) <= self.width / 2
    
    def project_to_middle_line(self, point):
        """Find the closest point on the middle line to the given point.

        Args:
            point: The point to project.
        """
        x, y = point
        closest_point = min(self.middle_line, key=lambda p: (p[0] - x)**2 + (p[1] - y)**2)
        return closest_point
    
    def project_to_boundary(self, point, fraction=0.5):
        """Project the given point to the track boundary.

        Args:
            point: The point to project.
            *fraction: The fraction of the track width to project to. Defaults to 0.5.
        """
        mp = self.project_to_middle_line(point)
        direction = self.directions[self.i_map[tuple(mp)]]
        normal = np.array([direction[1], -direction[0]])
        if np.dot(np.array(point) - np.array(mp), normal) > 0:
            return mp + fraction * self.width/2 * normal
        else:
            return mp - fraction * self.width/2 * normal

    def distance_to_middle_line(self, point) -> float:
        """Calculate the distance from the given point to the middle line.

        Args:
            point: The point to calculate the distance from.
        """
        closest_point = self.project_to_middle_line(point)
        return np.linalg.norm(closest_point - np.array(point))

    def get_limits(self) -> tuple:
        """Get the x-y-limits of the track.

        Returns:
            A tuple of the x-limits and y-limits of the track.
        """
        m_x, m_y = zip(*self.middle_line)
        l_x, l_y = zip(*self.left_boundary)
        r_x, r_y = zip(*self.right_boundary)
        x = m_x + l_x + r_x
        y = m_y + l_y + r_y
        return [min(x), max(x)], [min(y), max(y)]

    def lap_progress(self, point) -> float:
        """Calculate progress along the lap based on closest middle line segment.

        Args:
            point: The point to calculate progress from.
        Returns:
            The progress along the lap as a fraction between 0 and 1.
        """
        mp = self.project_to_middle_line(point)
        return self.progress_map[tuple(mp)]
    
    def plot_track(self, ax: plt.Axes = None, color='black'):
        """Plot the race track using matplotlib.

        Args:
            *ax: The matplotlib axes to plot on. Defaults to the current axes.
            *color: The color of the track. Defaults to black.
        """
            ax = plt.gca()
        
        # fill between boundaries
        return ax.fill(np.concatenate([self.left_boundary[:,0], self.right_boundary[::-1,0]]),
                np.concatenate([self.left_boundary[:,1], self.right_boundary[::-1,1]]),
                color=color)

    @staticmethod
    def generate_random_track(num_points=10, width=1, y_var=1.):
        """Generate a random race track with smooth curves.

        Args:
            *num_points: The number of points to generate. Defaults to 10.
            *width: The width of the track. Defaults to 1.
            *y_var: The variation in the y-direction. Defaults to 1.
        """
        x_vals = np.linspace(0, num_points, num_points)
        y_vals = np.cumsum(np.cumsum(np.random.uniform(-y_var, y_var, num_points)))  # Smooth variation
        
        middle_line = list(zip(x_vals, y_vals))

        # smoothen middle line
        middle_x, middle_y = smooth_line(middle_line)
        middle_y = middle_y - middle_y[0]
        middle_line = np.array(list(zip(middle_x, middle_y)))
        start_middle_point = middle_line[0]

        directions = [middle_line[i+1,:] - middle_line[i,:] for i in range(len(middle_line) - 1)]
        for i in range(len(directions)):
            directions[i] /= np.linalg.norm(directions[i])
        left_boundary = [middle_line[i,:] + width/2 * np.array([directions[i][1], -directions[i][0]]) for i in range(len(directions))]
        right_boundary = [middle_line[i,:] - width/2 * np.array([directions[i][1], -directions[i][0]]) for i in range(len(directions))]
        return RaceTrack(middle_line, np.array(left_boundary), np.array(right_boundary), start_middle_point, width)

vehicle_colors = [
    '#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', 
    '#984ea3', '#999999', '#e41a1c', '#dede00'
]

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
    def __init__(self):
        """Instantiate the controller class."""

    def get_feasible_controls(self):
        """Get the feasible controls."""

class DiscreteController(Controller):
    """Implements a discrete version of a Controller class."""
    def __init__(self, **kwargs):
        """Instantiate the DiscreteController class.

        Args:
            **grid: The grid to be used for the feasible controls. Defaults to a 3x3 grid.
        """
        super().__init__(**kwargs)
        self.grid = kwargs.get('grid', Grid())
        self.controls = []
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                self.controls.append([i*self.grid.dx, j*self.grid.dy])

    def get_feasible_controls(self):
        return self.controls

class Vehicle:
    """A class to represent a vehicle.

    Attributes:
        track (RaceTrack): The race track the vehicle is on.
        position (np.array): The current position of the vehicle.
        velocity (np.array): The current velocity of the vehicle.
        u (np.array): The current control action.
        color: The color of the vehicle.
        marker: The marker style for plotting.
        **controller (Controller): The controller for the vehicle. Defaults to DiscreteController.
    """
    def __init__(self, track=None, position=None, velocity=(0., 0.), color='black', marker='o', **kwargs):
        """Initialize a Vehicle instance.

        Args:
            *track (RaceTrack): The race track.
            *position (np.array): The initial position of the vehicle.
            *velocity (np.array): The initial velocity of the vehicle.
            *color: The color of the vehicle. Defaults to black.
            *marker: The marker style for plotting. Defaults to 'o'.
            **controller (Controller): The controller for the vehicle. Defaults to DiscreteController.
        """
        self.track: RaceTrack = track
        if position is not None:
            self.position = np.array(position)
        elif self.track is not None:
            self.position = np.array(self.track.start_middle_point)
        else:
            self.position = np.zeros(2)
        self.velocity = np.array(velocity)
        self.u = None
        self.color = color
        self.marker = marker
        self.controller: Controller = kwargs.get('controller', DiscreteController())

    def check_collision(self):
        """Check if the vehicle has collided.

        Currently, only collision with track boundaries is checked.
        If a collision is detected, the vehicle is reset.
        """
        if self.track is None:
            return
        if not self.track.is_on_track(self.position):
            self.reset()

    def get_feasible_controls(self):
        """Get the feasible control actions for the vehicle.

        Gets the feasible control actions from the controller (return type depends on controller).
        """
        return self.controller.get_feasible_controls()

    def reset(self):
        """Reset the vehicle.

        Velocity is set to zero and position is reset to the start position of the track. If no
        track is assigned, the position is set to the origin. The trajectory is reset as well.
        """
        self.velocity = np.array([0., 0.])
        if self.track is None:
            self.position = np.zeros(2)
        else:
            self.position = np.array(self.track.start_middle_point)

    def update(self):
        """Update the vehicle position based on the current control action.

        Update the position and velocity of the vehicle:
            p = p + v
            v = v + u
        If no control action is provided, the vehicle will continue with its current velocity.
        The trajectory is updated with the new position and the vehicle is checked for collisions.
        """
        if self.u is not None:
            self.velocity += np.array(self.u)
        self.position += self.velocity
        self.check_collision()

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
            **n_vehicles (int): The number of vehicles. Defaults to 1. Will be ignored if vehicles
                                is provided.
            **vehicles (List[Vehicle]): The list of vehicles. Defaults to a single vehicle.
        """
        self.track: RaceTrack = kwargs.get('track', RaceTrack.generate_random_track(y_var=1))
        self.grid = kwargs.get('grid', Grid())
        

        self.vehicles = kwargs.get('vehicles', None)
        if self.vehicles is None:
            self.n_vehicles = kwargs.get('n_vehicles', 1)
            self.vehicles = [Vehicle(self.track, color=vehicle_colors[i]) for i in range(self.n_vehicles)]
        else:
            self.n_vehicles = len(self.vehicles)
        self.cv_idx = 0

    def set_track(self, track: RaceTrack):
        """Assign a track instance.

        Args:
            track (RaceTrack): The track instance to assign.
        """
        self.track = track
        for v in self.vehicles:
            v.track = track

    def get_cv(self):
        """Get the current controllable vehicle."""
        return self.vehicles[self.cv_idx]
    
    def next_cv(self):
        """Get the next controllable vehicle."""
        self.cv_idx = (self.cv_idx + 1) % len(self.vehicles)
        return self.get_cv()