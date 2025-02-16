
import pytest

# internal imports
from lineracer import *

def test_track():
    track = RaceTrack.generate_random_track()
    assert isinstance(track, RaceTrack)

def test_race():
    race = Race(n_vehicles=3)
    assert isinstance(race.track, RaceTrack)
    assert len(race.vehicles) == 3
    assert race.n_vehicles == 3
    assert all([isinstance(v, Vehicle) for v in race.vehicles])
    for v1 in race.vehicles:
        for v2 in race.vehicles:
            if v1 != v2:
                assert v1.color != v2.color

def test_dynamics():
    v = Vehicle()
    assert np.allclose(v.position, np.array([0., 0.]))
    assert np.allclose(v.velocity, np.array([0., 0.]))
    v.u = np.array([1., 1.])
    v.update()
    assert np.allclose(v.position, np.array([1., 1.]))
    assert np.allclose(v.velocity, np.array([1., 1.]))
    v.u = np.array([-1., 1.])
    v.update()
    assert np.allclose(v.position, np.array([1., 3.]))
    assert np.allclose(v.velocity, np.array([0., 2.]))

def test_reset():
    v = Vehicle()
    v.position = np.array([0., 0.])
    v.velocity = np.array([1., 1.])
    v.update()
    assert np.allclose(v.position, np.array([1., 1.]))
    v.reset()
    assert np.allclose(v.position, np.array([0., 0.]))
    assert np.allclose(v.velocity, np.array([0., 0.]))
    