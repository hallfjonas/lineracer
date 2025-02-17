import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import warnings

def smooth_line(points):
        x, y = zip(*points)
        x_smooth = np.linspace(min(x), max(x), 300)
        y_spline = make_interp_spline(x, y)(x_smooth)
        return x_smooth, y_spline

class RaceTrack:
    def __init__(self, middle_line, left_boundary, right_boundary, width):
        """
        Initializes the race track.
        :param middle_line: List of (x, y) tuples defining the middle line.
        :param left_boundary: List of (x, y) tuples defining the left boundary.
        :param right_boundary: List of (x, y) tuples defining the right boundary.
        :param width: Width of the track.
        """
        self.middle_line = middle_line
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary
        self.directions = [middle_line[i+1,:] - middle_line[i,:] for i in range(len(middle_line) - 1)]
        for i in range(len(self.directions)):
            self.directions[i] /= np.linalg.norm(self.directions[i])
        self.width = width
        self.progress_map = {tuple(point): i / len(middle_line) for i, point in enumerate(middle_line)}
        self.i_map = {tuple(point): i for i, point in enumerate(middle_line)}

    def is_on_track(self, point):
        """Check if a given point lies within the track boundaries."""
        return self.distance_to_middle_line(point) <= self.width / 2
    
    def project_to_middle_line(self, point):
        """Find the closest point on the middle line to the given point."""
        x, y = point
        closest_point = min(self.middle_line, key=lambda p: (p[0] - x)**2 + (p[1] - y)**2)
        return closest_point
    
    def project_to_boundary(self, point, fraction=0.5):
        mp = self.project_to_middle_line(point)
        direction = self.directions[self.i_map[tuple(mp)]]
        normal = np.array([direction[1], -direction[0]])
        if np.dot(np.array(point) - np.array(mp), normal) > 0:
            return mp + fraction * self.width/2 * normal
        else:
            return mp - fraction * self.width/2 * normal

    def distance_to_middle_line(self, point):
        """Calculate the distance from the given point to the middle line."""
        closest_point = self.project_to_middle_line(point)
        return np.linalg.norm(closest_point - np.array(point))

    def get_limits(self):
        """Get the limits of the track."""
        m_x, m_y = zip(*self.middle_line)
        l_x, l_y = zip(*self.left_boundary)
        r_x, r_y = zip(*self.right_boundary)
        x = m_x + l_x + r_x
        y = m_y + l_y + r_y
        return [min(x), max(x)], [min(y), max(y)]

    def get_start_middle_point(self) -> tuple:
        """Get the intersection point between start line and mid-line."""
        idx = int(len(self.middle_line) / 10)
        return tuple(self.middle_line[idx,:])

    def get_finish_middle_point(self) -> tuple:
        """Get the intersection point between goal line and mid-line."""
        idx = int(9 * len(self.middle_line) / 10)
        return tuple(self.middle_line[idx,:])

    def plot_line_at_middle_point(self, mp, ax: plt.Axes = None, **kwargs):
        """Add the starting line using matplotlib."""
        if ax == None:
            ax = plt.gca()
        try:
            s_dir = self.directions[self.i_map[tuple(mp)]]
            normal = np.array([s_dir[1], -s_dir[0]])
            start_left = mp - self.width/2 * normal
            start_right = mp + self.width/2 * normal
            return ax.plot([start_left[0], start_right[0]], [start_left[1], start_right[1]], **kwargs)
        except:
            warnings.warn("Middle point not found (skipping plot.)")

    def plot_start_line(self, ax: plt.Axes = None, **kwargs):
        smp = self.get_start_middle_point()
        return self.plot_line_at_middle_point(smp, ax, **kwargs)

    def plot_finish_line(self, ax: plt.Axes = None, **kwargs):
        smp = self.get_finish_middle_point()
        return self.plot_line_at_middle_point(smp, ax, **kwargs)

    def lap_progress(self, point):
        """Calculate progress along the lap based on closest middle line segment."""
        mp = self.project_to_middle_line(point)
        progress_start = self.progress_map[self.get_start_middle_point()]
        progress_end = self.progress_map[self.get_finish_middle_point()]
        progress_point = self.progress_map[tuple(mp)]
        return (progress_point - progress_start) / (progress_end - progress_start)
    
    def plot_track(self, ax: plt.Axes = None, color='black'):
        """Plot the race track using matplotlib."""
        if ax == None:
            ax = plt.gca()

        # plot start and finish lines
        self.plot_start_line(ax, color='white')
        self.plot_finish_line(ax, color='white')

        # fill between boundaries
        return ax.fill(np.concatenate([self.left_boundary[:,0], self.right_boundary[::-1,0]]),
                np.concatenate([self.left_boundary[:,1], self.right_boundary[::-1,1]]),
                color=color)

    @staticmethod
    def generate_random_track(num_points=10, width=1, y_var=1.):
        """Generate a random race track with smooth curves."""
        x_vals = np.linspace(0, num_points, num_points)
        y_vals = np.cumsum(np.cumsum(np.random.uniform(-y_var, y_var, num_points)))  # Smooth variation
        
        middle_line = list(zip(x_vals, y_vals))

        # smoothen middle line
        middle_x, middle_y = smooth_line(middle_line)
        middle_y = middle_y - middle_y[0]
        middle_line = np.array(list(zip(middle_x, middle_y)))

        directions = [middle_line[i+1,:] - middle_line[i,:] for i in range(len(middle_line) - 1)]
        for i in range(len(directions)):
            directions[i] /= np.linalg.norm(directions[i])
        left_boundary = [middle_line[i,:] + width/2 * np.array([directions[i][1], -directions[i][0]]) for i in range(len(directions))]
        right_boundary = [middle_line[i,:] - width/2 * np.array([directions[i][1], -directions[i][0]]) for i in range(len(directions))]
        return RaceTrack(middle_line, np.array(left_boundary), np.array(right_boundary), width)

vehicle_colors = [
    '#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', 
    '#984ea3', '#999999', '#e41a1c', '#dede00'
]

class Grid:
    def __init__(self, dx = 0.25, dy = 0.25):
        self.dx = dx
        self.dy = dy

class Controller:
    def __init__(self):
        pass

    def get_feasible_controls(self):
        pass

class DiscreteController(Controller):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grid = kwargs.get('grid', Grid())
        self.controls = []
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                self.controls.append([i*self.grid.dx, j*self.grid.dy])

    def get_feasible_controls(self):
        return self.controls

class Vehicle:
    def __init__(self, track=None, position=None, velocity=[0.,0.], color='black', marker='o', **kwargs):
        self.track: RaceTrack = track
        if position is not None:
            self.position = np.array(position)
        elif self.track is not None:
            self.position = np.array(self.track.get_start_middle_point())
        else:
            self.position = np.zeros(2)
        self.velocity = np.array(velocity)
        self.u = None
        self.color = color
        self.marker = marker
        self.controller: Controller = kwargs.get('controller', DiscreteController())

    def check_collision(self):
        if self.track is None:
            return
        if not self.track.is_on_track(self.position):
            self.reset()

    def get_feasible_controls(self):
        return self.controller.get_feasible_controls()

    def reset(self):
        self.velocity = np.array([0., 0.])
        if self.track is None:
            self.position = np.zeros(2)
        else:
            self.position = np.array(self.track.get_start_middle_point())

    def update(self):
        if self.u is not None:
            self.velocity += np.array(self.u)
        self.position += self.velocity
        self.check_collision()

class Race:
    def __init__(self, **kwargs):
        self.track: RaceTrack = kwargs.get('track', RaceTrack.generate_random_track(y_var=1))
        self.grid = kwargs.get('grid', Grid())
        

        self.vehicles = kwargs.get('vehicles', None)
        if self.vehicles is None:
            self.n_vehicles = kwargs.get('n_vehicles', 1)
            self.vehicles = [Vehicle(self.track, color=vehicle_colors[i]) for i in range(self.n_vehicles)]
        else:
            self.n_vehicles = len(self.vehicles)
        self.cv_idx = 0

    def set_track(self, track):
        self.track = track
        for v in self.vehicles:
            v.track = track

    def get_cv(self):
        return self.vehicles[self.cv_idx]
    
    def next_cv(self):
        self.cv_idx = (self.cv_idx + 1) % len(self.vehicles)
        return self.get_cv()