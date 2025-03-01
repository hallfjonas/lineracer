"""This files contains classes that implement the race back-end."""

# external imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import warnings
import math
from typing import Tuple

# internal imports
from lineracer.PlotObjects import PlotObject


def interpolate_2D(x, y, num_eval=100, degree=3) -> tuple:
    """Interpolate a 2D curve using splines.

    Given a set of x and y coordinates, this function interpolates a 2D curve using splines.
    If the initial point is reached multiple times, we extract a middle loop to ensure smooth
    transitions. Otherwise, the entire curve is interpolated.

    Args:
        x: The x-coordinates of the curve.
        y: The y-coordinates of the curve.
        *num_eval: The number of points to evaluate the curve at. Defaults to 100.
        *degree: The degree of the spline. Defaults to 3.

    Returns:
        A tuple of the interpolated x-coordinates and y-coordinates.

    Raises:
        ValueError: If the input vectors x and y do not have the same length.
    """
    n = len(x)
    if len(y) != n:
        raise ValueError("Input vectors x and y must have the same length")


    # keep track of indices during which the starting point is reached
    i_loop = [0]
    p0 = np.array([x[0],y[0]])

    # Calculate the distances between points
    norm_vals = [0.0]
    for i in range(1,n):
        old_p = np.array([x[i-1],y[i-1]])
        new_p = np.array([x[i],y[i]])
        norm_vals.append(np.linalg.norm(new_p-old_p))

        # check if we have reached starting point again
        if np.linalg.norm(new_p-p0) == 0:
            i_loop.append(i)

    # create a parameter t based on the relative distance of along the path
    accum_dist = np.cumsum(norm_vals)
    t = [a / accum_dist[-1] for a in accum_dist]
    t = np.array(t)

    # fit splines for x and y independently
    spl_x = make_interp_spline(t, x, k=min(degree, n-1))
    spl_y = make_interp_spline(t, y, k=min(degree, n-1))


    # if multiple loops, extract a middle loop (otherwise interpolate the entire curve)
    if len(i_loop) >= 3:
        i_start_idx = int(np.floor((len(i_loop)-0.5)/2))
        i_end_idx = i_start_idx + 1
        start_idx = i_loop[i_start_idx]
        end_idx = i_loop[i_end_idx]
    else:
        start_idx = 0
        end_idx = -1
    t_fine = np.linspace(t[start_idx], t[end_idx], num_eval)

    # Evaluate and return the smooth curve
    return spl_x(t_fine), spl_y(t_fine)

class RaceTrack:
    """A class representing a race track.

    Attributes:
        middle_line: The middle line of the track.
        left_boundary: The left boundary of the track.
        right_boundary: The right boundary of the track.
        width: The width of the track.
        directions: The directions of the track segments.
        progress_map: A map of middle line points to progress.
        i_map: A map of middle line points to index
    """
    def __init__(self, middle_line, left_boundary, right_boundary, width):
        """Initializes the race track.

        Args:
            middle_line: List of (x, y) tuples defining the middle line.
            left_boundary: List of (x, y) tuples defining the left boundary.
            right_boundary: List of (x, y) tuples defining the right boundary.
            width: Width of the track.
        """
        self.middle_line = middle_line
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary
        self.directions = [middle_line[i+1,:] - middle_line[i,:] for i in range(len(middle_line) - 1)]
        for i in range(len(self.directions)):
            self.directions[i] /= np.linalg.norm(self.directions[i])

    def assign_boundaries(self) -> bool:
        """Assign the left and right boundaries of the track.

        Returns:
            True iff the race track does not intersect itself.
        """
        lb = self.middle_line.copy()
        rb = self.middle_line.copy()
        not_intersects = True
        for i, _ in enumerate(self.middle_line):
            d = self.directions[i]
            n = np.array([d[1], -d[0]])
            lb[i,:] += 0.5 * self.width * n
            rb[i,:] -= 0.5 * self.width * n

            if self.on_track(lb[i,:], tol=-self.width*0.01):
                not_intersects = False

            if self.on_track(rb[i,:], tol=-self.width*0.01):
                not_intersects = False
        self.left_boundary = lb
        self.right_boundary = rb
        return not_intersects

    def ensure_closed_path(self, middle_line: np.ndarray, tol=1e-2):
        """Ensure that the middle line is a closed path.

        Args:
            middle_line: The middle line to check.
            *tol: The tolerance to consider the start and end point being closed. Defaults to 1e-2.

        Raises:
            ValueError: If the middle line is not a closed path.
        """
        if np.linalg.norm(middle_line[0] - middle_line[-1]) > tol:
            raise ValueError("The middle line must be a closed path.")

    def get_curvature(self, i: int) -> float:
        """Calculate the curvature at a given middle line point.

        Args:
            i: Index of the mid-line point at which to calculate the curvature.
        """
        omp = self.middle_line[i-1]
        mp = self.middle_line[i]
        nmp = self.middle_line[(i+1)%self.n]
        normalize = np.linalg.norm(nmp - mp) * np.linalg.norm(mp - omp)
        return (1-np.dot(nmp - mp, mp - omp)) / normalize

    def check_curvature_constraints(self, **kwargs) -> bool:
        """Check if the middle line satisfies curvature constraints.

        Args:
            **curve_max: The maximal allowed curvature. Defaults to 1e-1.
        Returns:
            True iff the middle line satisfies the curvature constraints
        """
        for i in range(self.n):
            curve = self.get_curvature(i)
            if curve < 0 or curve > kwargs.get('curve_max', 1e-1):
                return False
        return True

    def shift_middle_line_to_start(self):
        """Shift the middle line to start from the starting point.

        The starting point is calculated as the last point of the longest straight.
        """
        curve = [self.get_curvature(i) for i in range(self.n)]
        ten_percentile = np.percentile(curve, 10)
        start, stop = self.find_longest_straight(curve_tol=ten_percentile)
        self.middle_line = np.roll(self.middle_line, -stop, axis=0)

    def on_track(self, point, tol=0.0) -> bool:
        """Check if a given point lies within the track boundaries.

        Args:
            point: The point to check.
            *tol: The tolerance to consider a point on the track. Defaults to 0.
        """
        return self.distance_to_middle_line(point) <= self.width / 2 + tol

    def on_track_after(self, point, middle_line_point: tuple) -> Tuple[bool, tuple]:
        """Check if a point is on the track after a given mid-line point.

        The advantage of this method is that it provides a quick check if we roughly know where it
        is on the track. This is particularly useful if we want to check if a vehicle remains on the
        track when moving from one point to the next.
        Args:
            point: The point to check.
            middle_line_point: The first mid-line point to check from.
        Returns:
            tuple:
            - bool: True iff the point is on the track after the given mid-line point
            - tuple: The first mid-line point verifying that the point was on track.
        """
        idx = self.i_map[tuple(middle_line_point)]
        for i in range(idx, len(self.middle_line) - 1):
            if np.linalg.norm(self.middle_line[i] - np.array(point)) <= 0.5 * self.width:
                return True, self.middle_line[i]
        return False, None

    def line_on_track(self, point1, point2, mp = None, grid_test_size=0.1) -> Tuple[bool, tuple]:
        """Check if a line segment lies within the track boundaries.

        Args:
            point1: The starting point of the line segment (assumed to be of lower progress).
            point2: The ending point of the line segment  (assumed to be of larger progress).
            mp: The middle line point to check from. Defaults to None.

        Returns:
            tuple:
            - bool: True iff the line segment is on the track.
            - tuple: None if test failed, otherwise the first mid-line point verifying point2.
        """

        if mp is None:
            mp = self.project_to_middle_line(point1)

        # generate a test grid based of desired size
        nrm = np.linalg.norm(point2 - point1)
        test_grid = np.linspace(point1, point2, int(nrm / grid_test_size))

        # for each point in the grid, check if it is on the track after the previous mid-line point
        for p in test_grid:
            on_track, mp = self.on_track_after(p, mp)
            if not on_track:
                return False, None
        on_track, mp = self.on_track_after(point2, mp)
        return on_track, mp

    def get_start_point(self, starting_grid_index: int) -> tuple:
        """Get the starting point of the track.

        Similar to F1 starting grid: two vehicles per row, but on same progress line.

        Args:
            position: The position along the track to get the starting point from.
        """
        mlp = self.get_start_middle_point()
        mlp_idx = self.i_map[tuple(mlp)]
        dir_normalized = self.directions[mlp_idx] / np.linalg.norm(self.directions[mlp_idx])
        normal = np.array([dir_normalized[1], -dir_normalized[0]])
        dx = 0.25 * self.width
        sign = 1 if starting_grid_index % 2 == 0 else -1
        dy = dx * math.floor(starting_grid_index/2)
        return mlp + sign * dx * normal - dy * dir_normalized

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

    def find_longest_straight(self, curv_tol = 1e-6) -> tuple:
        """Find the longest straight segment in the track.

        Args:
            curv_tol: The curvature tolerance to consider a segment straight. Defaults to 1e-1.
        Returns:
            The start and end indices (from mid-line points) of the longest straight segment.
        """
        longest_straight = (0, 0)
        cur_straight = (0, 0)
        for i in range(self.n):
            curve = self.get_curvature(i)
            if curve < curv_tol:
                cur_straight = (cur_straight[0], i)
            else:
                if cur_straight[1] - cur_straight[0] > longest_straight[1] - longest_straight[0]:
                    longest_straight = cur_straight
                cur_straight = (i, i)
        return longest_straight

    def get_start_middle_point(self) -> tuple:
        """Get the intersection point between start line and mid-line."""
        idx = int(len(self.middle_line) / 10)
        return tuple(self.middle_line[idx,:])

    def get_finish_middle_point(self) -> tuple:
        """Get the intersection point between goal line and mid-line."""
        idx = int(9 * len(self.middle_line) / 10)
        return tuple(self.middle_line[idx,:])

    def plot_line_at_middle_point(self, mp, ax: plt.Axes = None, **kwargs) -> PlotObject:
        """Add the starting line using matplotlib.

        Args:
            mp: The middle point to plot the line at (expected to be in the list of mid-line points)
            *ax: The matplotlib axes to plot on. Defaults to the current axes.
            **kwargs: Additional keyword arguments to pass to the plot function.
        """
        if ax == None:
            ax = plt.gca()
        try:
            s_dir = self.directions[self.i_map[tuple(mp)]]
            normal = np.array([s_dir[1], -s_dir[0]])
            start_left = mp - self.width/2 * normal
            start_right = mp + self.width/2 * normal
            return PlotObject(ax.plot([start_left[0], start_right[0]], [start_left[1], start_right[1]], **kwargs))
        except:
            warnings.warn("Middle point not found (skipping plot.)")

    def plot_start_line(self, ax: plt.Axes = None, **kwargs) -> PlotObject:
        """Add the starting line using matplotlib.

        Args:
            *ax: The matplotlib axes to plot on. Defaults to the current axes.
            **kwargs: Additional keyword arguments to pass to the plot function.
        """
        smp = self.get_start_middle_point()
        return self.plot_line_at_middle_point(smp, ax, **kwargs)

    def plot_finish_line(self, ax: plt.Axes = None, **kwargs) -> PlotObject:
        """Add the finish line using matplotlib.

        Args:
            *ax: The matplotlib axes to plot on. Defaults to the current axes.
            **kwargs: Additional keyword arguments to pass to the plot function.
        """
        smp = self.get_finish_middle_point()
        return self.plot_line_at_middle_point(smp, ax, **kwargs)

    def lap_progress(self, point) -> float:
        """Calculate progress along the lap based on closest middle line segment.

        Args:
            point: The point to calculate progress from.
        Returns:
            The progress along the lap as a fraction between 0 and 1.
        """
        mp = self.project_to_middle_line(point)

    def plot_directions(self, ax: plt.Axes = None, **kwargs) -> PlotObject:
        """Plot the directions of the track segments using matplotlib.

        Args:
            *ax: The matplotlib axes to plot on. Defaults to the current axes.
            **kwargs: Additional keyword arguments to pass to the plot function.
        """
        if ax == None:
            ax = plt.gca()
        po = PlotObject()
        for i in self.i_map.values():
            p = self.middle_line[i]
            if i >= len(self.directions):
                continue
            po.add(ax.arrow(p[0], p[1], self.directions[i][0], self.directions[i][1], **kwargs))
        return po

    def plot_curvature(self, ax: plt.Axes = None, **kwargs) -> PlotObject:
        """Plot the curvature of the track segments using matplotlib.

        Args:
            *ax: The matplotlib axes to plot on. Defaults to the current axes.
            **kwargs: Additional keyword arguments to pass to the plot function.
        """
        if ax == None:
            ax = plt.gca()
        po = PlotObject()
        curve = [self.get_curvature(i) for i in range(self.n)]
        t = np.linspace(0, 1, len(curve))
        po.add(ax.plot(t, curve, **kwargs))
        return po

    def plot_track(self, ax: plt.Axes = None, color='black') -> PlotObject:
        """Plot the race track using matplotlib.

        Args:
            *ax: The matplotlib axes to plot on. Defaults to the current axes.
            *color: The color of the track. Defaults to black.
        """
        if ax is None:
            ax = plt.gca()


        # plot object container
        po = PlotObject()

        # plot start and finish lines
        po.add(self.plot_start_line(ax, color='white'))
        po.add(self.plot_finish_line(ax, color='white'))

        # fill between boundaries
        po.add(ax.fill(np.concatenate([self.left_boundary[:,0], self.right_boundary[::-1,0]]),
                np.concatenate([self.left_boundary[:,1], self.right_boundary[::-1,1]]),
                color=color))

        return po

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

        directions = [middle_line[i+1,:] - middle_line[i,:] for i in range(len(middle_line) - 1)]
        for i in range(len(directions)):
            directions[i] /= np.linalg.norm(directions[i])
        left_boundary = [middle_line[i,:] + width/2 * np.array([directions[i][1], -directions[i][0]]) for i in range(len(directions))]
        right_boundary = [middle_line[i,:] - width/2 * np.array([directions[i][1], -directions[i][0]]) for i in range(len(directions))]
        return RaceTrack(middle_line, np.array(left_boundary), np.array(right_boundary), width)
