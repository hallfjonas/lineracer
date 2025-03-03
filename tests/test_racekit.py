
import pytest
import numpy as np
import matplotlib.pyplot as plt

# internal imports
from lineracer.Race import Race
from lineracer.RaceTrack import RaceTrack
from lineracer.Vehicles import Vehicle
from lineracer.Controllers import DiscreteController

def test_track():
    track = RaceTrack.generate_random_track()
    assert isinstance(track, RaceTrack)

    sp = track.get_start_middle_point()
    assert isinstance(sp, tuple)
    assert round(track.lap_progress(sp)) == 0
    assert track.lap_progress(track.middle_line[-1,:]) > 0.95

def test_race():
    race = Race(n_player=1, n_npc=2)
    assert isinstance(race.track, RaceTrack)
    assert len(race.vehicles) == 3
    assert race.n_vehicles == 3
    assert all([isinstance(v, Vehicle) for v in race.vehicles])
    for v1 in race.vehicles:
        for v2 in race.vehicles:
            if v1 != v2:
                assert v1.color != v2.color
                assert v1.track == v2.track

    new_track = RaceTrack.generate_random_track()
    race.set_track(new_track)
    assert isinstance(race.track, RaceTrack)
    for v1 in race.vehicles:
        for v2 in race.vehicles:
            if v1 != v2:
                assert v1.color != v2.color
                assert v1.track == v2.track
    assert v1.track == new_track

def test_dynamics():
    v = Vehicle()
    assert np.allclose(v.position, np.array([0., 0.]))
    assert np.allclose(v.velocity, np.array([0., 0.]))
    v.controller.u = np.array([1., 1.])
    v.update()
    assert np.allclose(v.position, np.array([1., 1.]))
    assert np.allclose(v.velocity, np.array([1., 1.]))
    v.controller.u = np.array([-1., 1.])
    v.update()
    assert np.allclose(v.position, np.array([1., 3.]))
    assert np.allclose(v.velocity, np.array([0., 2.]))

def test_reset():
    track = RaceTrack.generate_random_track()
    idx = 0
    v = Vehicle(track, starting_grid_index=idx)
    start_pos = track.get_start_point(idx)
    v.velocity = np.array([0.01, -0.011])
    v.update()
    assert not np.allclose(v.position, start_pos)
    v.reset()
    assert np.allclose(v.position, start_pos)
    assert np.allclose(v.velocity, np.array([0., 0.]))

def test_manuel_controller():
    c = DiscreteController()
    assert len(c.get_feasible_controls()) > 0

    v = Vehicle()
    assert len(v.get_feasible_controls()) > 0

    v2 = Vehicle(controller=c)
    assert v2.get_feasible_controls() == c.get_feasible_controls()

def test_brute_force_controller():
    v = Vehicle(track=RaceTrack.generate_random_track())
    v.controller = DiscreteController(track=v.track)
    c = v.controller
    for _ in range(3):
        prog = v.track.lap_progress(v.position)
        pos = np.array(v.position)
        vel = np.array(v.velocity)
        c.compute_control(pos, vel)
        assert c.u is not None
        assert c.is_feasible(c.u)
        v.update()
        assert np.allclose(v.position, pos + vel + c.u)
        assert np.allclose(v.velocity, vel + c.u)
        assert v.track.lap_progress(v.position) >= prog

def test_collisions():
    track = RaceTrack.generate_random_track()
    p1 = track.get_start_middle_point()
    assert track.on_track(p1)
    on_track, mp1 = track.on_track_after(p1, p1)
    assert on_track
    assert np.allclose(mp1, p1)

    i1 = track.i_map[p1]
    p2 = track.middle_line[i1 + 5]
    on_track, mp2 = track.on_track_after(p2, p1)
    assert on_track

    on_track, mp_line = track.line_on_track(p1, p2, mp1)
    assert on_track
    assert np.allclose(mp_line, mp2)

    mp_end = track.middle_line[-1]
    on_track, mp_end = track.on_track_after(p1, mp_end)
    assert not on_track

    v = Vehicle(track=track)
    v.position = track.get_start_point(v.starting_grid_index)
    start_direction = track.directions[track.i_map[track.get_start_middle_point()]]
    start_normal = [-start_direction[1], start_direction[0]]
    v.velocity = 10 * np.array(start_normal) / np.linalg.norm(start_normal)
    v.position = p1 + v.velocity
    collided, mp = v.check_collision()
    assert collided
    assert not np.allclose(v.position, track.get_start_point(v.starting_grid_index))

def test_plot_racetrack():
    track = RaceTrack.generate_random_track(
        x_max = 20,
        y_max = 9,
    )

    fig, ax = plt.subplots()
    track.plot_curvature()
    ax.set_yscale('log')

    fig, ax = plt.subplots()
    track.plot_track(ax)
    ax.set_aspect('equal')

    fig, ax = plt.subplots()
    track.plot_directions(ax, alpha=0.1)
    plt.show()

