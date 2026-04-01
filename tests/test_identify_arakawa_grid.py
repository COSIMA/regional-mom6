import pytest
from regional_mom6.regional_mom6 import identify_arakawa_grid


def _make_var_map(u_x, v_x, tracer_x, u_y="lat_u", v_y="lat_v", tracer_y="lat_t"):
    return {
        "u_x_coord": u_x,
        "v_x_coord": v_x,
        "tracer_x_coord": tracer_x,
        "u_y_coord": u_y,
        "v_y_coord": v_y,
        "tracer_y_coord": tracer_y,
    }


def test_arakawa_a_grid():
    """All velocity and tracer points share the same x coordinate -> A grid."""
    var_map = _make_var_map(u_x="lon", v_x="lon", tracer_x="lon")
    assert identify_arakawa_grid(var_map) == "A"


def test_arakawa_b_grid():
    """u and v share x coordinate, but differ from tracer -> B grid."""
    var_map = _make_var_map(u_x="lon_uv", v_x="lon_uv", tracer_x="lon_t")
    assert identify_arakawa_grid(var_map) == "B"


def test_arakawa_c_grid():
    """u, v, and tracer all on different x coordinates, and v_y != tracer_y -> C grid."""
    var_map = _make_var_map(
        u_x="lon_u",
        v_x="lon_t",
        tracer_x="lon_t",
        v_y="lat_v",
        u_y="lat_t",
        tracer_y="lat_t",
    )
    assert identify_arakawa_grid(var_map) == "C"


def test_arakawa_raises_when_indeterminate():
    """v_x != u_x but u_x == tracer_x (and v_y == tracer_y) cannot be classified."""
    var_map = _make_var_map(
        u_x="lon_t",
        v_x="lon_v",
        tracer_x="lon_t",
        v_y="lat_t",
        tracer_y="lat_t",
    )
    with pytest.raises(ValueError, match="Could not determine Arakawa grid type"):
        identify_arakawa_grid(var_map)


def test_arakawa_c_grid_requires_distinct_v_y():
    """
    v_x != u_x and u_x != tracer_x, but v_y == tracer_y -> should not be C grid.
    The else branch raises ValueError.
    """
    var_map = _make_var_map(
        u_x="lon_u",
        v_x="lon_v",
        tracer_x="lon_t",
        v_y="lat_shared",
        tracer_y="lat_shared",
    )
    with pytest.raises(ValueError, match="Could not determine Arakawa grid type"):
        identify_arakawa_grid(var_map)
