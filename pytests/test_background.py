import numpy as np
import pytest

# functions.background imports transport.py, which needs xraylib and pymsis.
pytest.importorskip("xraylib")
pytest.importorskip("pymsis")
from functions.background import (  # noqa: E402
    background_budget, cxb, dark_earth, nxb, particle_background,
)


def test_cxb_positive_and_falling():
    sky = cxb([4, 10, 30])
    assert np.all(sky > 0) and np.all(np.diff(sky) < 0)


def test_cxb_rejects_soft_energies():
    with pytest.raises(ValueError):
        cxb(1.0)


def test_dark_earth_half_at_break():
    assert dark_earth(33.7, c=1.0) == pytest.approx(0.5)


def test_nxb_line_integrates_to_rate():
    e = np.linspace(5, 8, 3001)
    line = nxb(e, 0, [(6.4, 0.1, 1e-3)])
    assert np.trapezoid(line, e) == pytest.approx(1e-3, rel=1e-4)


def test_particle_background_weighted_mean():
    result = particle_background([[1, 1], [3, 3]], [1, 1])
    assert result == pytest.approx([2, 2])


def test_budget_total_is_sum():
    budget = background_budget([1, 2], 10, {"a": 1.0, "b": 2.0})
    assert budget["total"] == pytest.approx(budget["a"] + budget["b"])
    assert budget["total"] == pytest.approx(30)


@pytest.mark.reference
@pytest.mark.skip(reason="needs published CXB intensity values (Gruber et al. 1999)")
def test_cxb_matches_published():
    pass
