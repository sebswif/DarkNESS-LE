import numpy as np
import pytest

from functions.conversion import conversion_probability_path, conversion_probability_uniform


def test_path_matches_uniform_in_vacuum():
    s = np.linspace(0, 1000, 101)
    p = conversion_probability_path(s, np.full(s.size, 3e-5), e=4.0, g=1e-10)
    assert p == pytest.approx(conversion_probability_uniform(3e-5, 1000, 1e-10))


def test_vector_field_matches_scalar():
    s = np.linspace(0, 1000, 101)
    vectors = np.tile([3e-5, 0, 0], (s.size, 1))
    scalar = conversion_probability_path(s, np.full(s.size, 3e-5), e=4.0, g=1e-10)
    vector = conversion_probability_path(s, vectors, e=4.0, g=1e-10)
    assert vector == pytest.approx(scalar)


def test_absorption_lowers_probability():
    s = np.linspace(0, 1000, 101)
    b = np.full(s.size, 3e-5)
    vacuum = conversion_probability_path(s, b, e=4.0, g=1e-10)
    lossy = conversion_probability_path(s, b, e=4.0, g=1e-10, attenuation=1e-3)
    assert lossy < vacuum


def test_uniform_scales_as_b_length_squared():
    base = conversion_probability_uniform(1.0, 1.0, 1e-10)
    assert conversion_probability_uniform(2.0, 3.0, 1e-10) == pytest.approx(36 * base)


@pytest.mark.reference
@pytest.mark.skip(reason="needs a published geomagnetic conversion probability (e.g. Davoudiasl & Huber 2006)")
def test_path_matches_published():
    pass
