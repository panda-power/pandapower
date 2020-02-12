# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pandapower.plotting as plot
import pytest


def test_cmap_discrete():
    cmap_list = [((0, 10), "green"), ((10, 30), "yellow"), ((30, 100), "red")]
    cmap, norm = plot.cmap_discrete(cmap_list)
    assert cmap.N == 3
    assert norm.N == 4
    assert norm(0) == 0
    assert norm(9.999) == 0
    assert norm(10.01) == 1
    assert norm(29.999) == 1
    assert norm(30.01) == 2
    assert norm(99.999) == 2


def test_cmap_continuous():
    cmap_list = [(0.97, "blue"), (1.0, "green"), (1.03, "red")]
    cmap, norm = plot.cmap_continuous(cmap_list)

    assert np.allclose(cmap(0.99), (0.984313725490196, 0.007873894655901603, 0.0, 1.0))
    assert np.allclose(cmap(1.02), (1.0, 0.0, 0.0, 1.0))
    assert np.allclose([norm(0.97 + 0.01 * n) for n in range(7)],
                       [0.16666666666666666 * n for n in range(7)])


def test_cmap_logarithmic():
    min_value, max_value = 1.0, 1.03
    colors = ["blue", "green", "red"]
    cmap, norm = plot.cmap_logarithmic(min_value, max_value, colors)
    assert np.allclose(cmap(0.1), (0.0, 0.09770172300400634, 0.8053598487029561, 1.0))
    assert np.allclose(cmap(0.5), (0.0, 0.5002328217805124, 0.003442425359135415, 1.0))
    assert np.allclose(cmap(0.8), (0.5970222233016451, 0.20227904085250753, 0.0, 1.0))
    assert norm(1.) == 0
    assert norm(1.03) == 1
    assert np.isclose(norm(1.01), 0.3366283509006011)
    assert np.isclose(norm(1.02), 0.669940112402371)


if __name__ == "__main__":
    pytest.main(["test_create_colormaps.py"])
