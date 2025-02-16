
import pytest

# internal imports
from LineRacer import *

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
