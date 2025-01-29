import regional_mom6 as rmom6
import regional_mom6.rotation as rot
import regional_mom6.regridding as rgd
import pytest
import xarray as xr
import numpy as np
import os


def test_get_curvilinear_hgrid_fixture(get_curvilinear_hgrid):
    # If the fixture fails to find the file, the test will be skipped.
    assert get_curvilinear_hgrid is not None


def test_expanded_hgrid_generation(get_curvilinear_hgrid):
    hgrid = get_curvilinear_hgrid
    expanded_hgrid = rot.create_expanded_hgrid(hgrid)

    # Check Size
    assert len(expanded_hgrid.nxp) == (len(hgrid.nxp) + 2)
    assert len(expanded_hgrid.nyp) == (len(hgrid.nyp) + 2)

    # Check pseudo_hgrid keeps the same values
    assert (expanded_hgrid.x.values[1:-1, 1:-1] == hgrid.x.values).all()
    assert (expanded_hgrid.y.values[1:-1, 1:-1] == hgrid.y.values).all()

    # Check extra boundary has realistic values
    diff_check = 1
    assert (
        (
            expanded_hgrid.x.values[0, 1:-1]
            - (hgrid.x.values[0, :] - (hgrid.x.values[1, :] - hgrid.x.values[0, :]))
        )
        < diff_check
    ).all()
    assert (
        (
            expanded_hgrid.x.values[1:-1, 0]
            - (hgrid.x.values[:, 0] - (hgrid.x.values[:, 1] - hgrid.x.values[:, 0]))
        )
        < diff_check
    ).all()
    assert (
        (
            expanded_hgrid.x.values[-1, 1:-1]
            - (hgrid.x.values[-1, :] - (hgrid.x.values[-2, :] - hgrid.x.values[-1, :]))
        )
        < diff_check
    ).all()
    assert (
        (
            expanded_hgrid.x.values[1:-1, -1]
            - (hgrid.x.values[:, -1] - (hgrid.x.values[:, -2] - hgrid.x.values[:, -1]))
        )
        < diff_check
    ).all()

    # Check corners for the same...
    assert (
        expanded_hgrid.x.values[0, 0]
        - (hgrid.x.values[0, 0] - (hgrid.x.values[1, 1] - hgrid.x.values[0, 0]))
    ) < diff_check
    assert (
        expanded_hgrid.x.values[-1, 0]
        - (hgrid.x.values[-1, 0] - (hgrid.x.values[-2, 1] - hgrid.x.values[-1, 0]))
    ) < diff_check
    assert (
        expanded_hgrid.x.values[0, -1]
        - (hgrid.x.values[0, -1] - (hgrid.x.values[1, -2] - hgrid.x.values[0, -1]))
    ) < diff_check
    assert (
        expanded_hgrid.x.values[-1, -1]
        - (hgrid.x.values[-1, -1] - (hgrid.x.values[-2, -2] - hgrid.x.values[-1, -1]))
    ) < diff_check

    # Same for y
    assert (
        (
            expanded_hgrid.y.values[0, 1:-1]
            - (hgrid.y.values[0, :] - (hgrid.y.values[1, :] - hgrid.y.values[0, :]))
        )
        < diff_check
    ).all()
    assert (
        (
            expanded_hgrid.y.values[1:-1, 0]
            - (hgrid.y.values[:, 0] - (hgrid.y.values[:, 1] - hgrid.y.values[:, 0]))
        )
        < diff_check
    ).all()
    assert (
        (
            expanded_hgrid.y.values[-1, 1:-1]
            - (hgrid.y.values[-1, :] - (hgrid.y.values[-2, :] - hgrid.y.values[-1, :]))
        )
        < diff_check
    ).all()
    assert (
        (
            expanded_hgrid.y.values[1:-1, -1]
            - (hgrid.y.values[:, -1] - (hgrid.y.values[:, -2] - hgrid.y.values[:, -1]))
        )
        < diff_check
    ).all()

    assert (
        expanded_hgrid.y.values[0, 0]
        - (hgrid.y.values[0, 0] - (hgrid.y.values[1, 1] - hgrid.y.values[0, 0]))
    ) < diff_check
    assert (
        expanded_hgrid.y.values[-1, 0]
        - (hgrid.y.values[-1, 0] - (hgrid.y.values[-2, 1] - hgrid.y.values[-1, 0]))
    ) < diff_check
    assert (
        expanded_hgrid.y.values[0, -1]
        - (hgrid.y.values[0, -1] - (hgrid.y.values[1, -2] - hgrid.y.values[0, -1]))
    ) < diff_check
    assert (
        expanded_hgrid.y.values[-1, -1]
        - (hgrid.y.values[-1, -1] - (hgrid.y.values[-2, -2] - hgrid.y.values[-1, -1]))
    ) < diff_check

    return


def test_mom6_angle_calculation_method(get_curvilinear_hgrid):
    """
    Check no rotation, up tilt, down tilt.
    """

    # Check no rotation
    top_left = xr.Dataset(
        {
            "x": (("nyp", "nxp"), [[0]]),
            "y": (("nyp", "nxp"), [[1]]),
        }
    )
    top_right = xr.Dataset(
        {
            "x": (("nyp", "nxp"), [[1]]),
            "y": (("nyp", "nxp"), [[1]]),
        }
    )
    bottom_left = xr.Dataset(
        {
            "x": (("nyp", "nxp"), [[0]]),
            "y": (("nyp", "nxp"), [[0]]),
        }
    )
    bottom_right = xr.Dataset(
        {
            "x": (("nyp", "nxp"), [[1]]),
            "y": (("nyp", "nxp"), [[0]]),
        }
    )
    point = xr.Dataset(
        {
            "x": (("nyp", "nxp"), [[0.5]]),
            "y": (("nyp", "nxp"), [[0.5]]),
        }
    )

    assert (
        rot.mom6_angle_calculation_method(
            2, top_left, top_right, bottom_left, bottom_right, point
        )
        == 0
    )

    # Angled
    hgrid = get_curvilinear_hgrid
    ds_t = rgd.get_hgrid_arakawa_c_points(hgrid, "t")
    ds_q = rgd.get_hgrid_arakawa_c_points(hgrid, "q")

    # Reformat into x, y comps
    t_points = xr.Dataset(
        {
            "x": (("nyp", "nxp"), ds_t.tlon.data),
            "y": (("nyp", "nxp"), ds_t.tlat.data),
        }
    )
    q_points = xr.Dataset(
        {
            "x": (("nyp", "nxp"), ds_q.qlon.data),
            "y": (("nyp", "nxp"), ds_q.qlat.data),
        }
    )
    assert (
        (
            rot.mom6_angle_calculation_method(
                hgrid.x.max() - hgrid.x.min(),
                q_points.isel(nyp=slice(1, None), nxp=slice(0, -1)),
                q_points.isel(nyp=slice(1, None), nxp=slice(1, None)),
                q_points.isel(nyp=slice(0, -1), nxp=slice(0, -1)),
                q_points.isel(nyp=slice(0, -1), nxp=slice(1, None)),
                t_points,
            )
            - hgrid["angle_dx"].isel(nyp=ds_t.t_points_y, nxp=ds_t.t_points_x).values
        )
        < 1
    ).all()

    return


def test_initialize_grid_rotation_angle(get_curvilinear_hgrid):
    """
    Generate a curvilinear grid and test the grid rotation angle at t_points based on what we pass to generate_curvilinear_grid
    """
    hgrid = get_curvilinear_hgrid
    angle = rot.initialize_grid_rotation_angle(hgrid)
    ds_t = rgd.get_hgrid_arakawa_c_points(hgrid, "t")
    assert (
        (
            angle.values
            - hgrid["angle_dx"].isel(nyp=ds_t.t_points_y, nxp=ds_t.t_points_x).values
        )
        < 1
    ).all()  # Angle is correct
    assert angle.values.shape == ds_t.tlon.shape  # Shape is correct
    return


def test_initialize_grid_rotation_angle_using_expanded_hgrid(get_curvilinear_hgrid):
    """
    Generate a curvilinear grid and test the grid rotation angle at t_points based on what we pass to generate_curvilinear_grid
    """
    hgrid = get_curvilinear_hgrid
    angle = rot.initialize_grid_rotation_angles_using_expanded_hgrid(hgrid)

    assert (angle.values - hgrid.angle_dx < 1).all()
    assert angle.values.shape == hgrid.x.shape
    return


def test_get_rotation_angle(get_curvilinear_hgrid, get_rectilinear_hgrid):
    """
    Generate a curvilinear grid and test the grid rotation angle at t_points based on what we pass to generate to generate_curvilinear_grid
    """
    curved_hgrid = get_curvilinear_hgrid
    rect_hgrid = get_rectilinear_hgrid

    o = None
    rotational_method = rot.RotationMethod.NO_ROTATION
    angle = rot.get_rotation_angle(rotational_method, rect_hgrid, orientation=o)
    assert angle.shape == rect_hgrid.x.shape
    assert (angle.values == 0).all()

    rotational_method == rot.RotationMethod.NO_ROTATION
    with pytest.raises(
        ValueError, match="NO_ROTATION method only works with rectilinear grids"
    ):
        angle = rot.get_rotation_angle(rotational_method, curved_hgrid, orientation=o)

    rotational_method = rot.RotationMethod.GIVEN_ANGLE
    angle = rot.get_rotation_angle(rotational_method, curved_hgrid, orientation=o)
    assert angle.shape == curved_hgrid.x.shape
    assert (angle.values == curved_hgrid.angle_dx).all()
    angle = rot.get_rotation_angle(rotational_method, rect_hgrid, orientation=o)
    assert angle.shape == rect_hgrid.x.shape
    assert (angle.values == 0).all()

    rotational_method = rot.RotationMethod.EXPAND_GRID
    angle = rot.get_rotation_angle(rotational_method, curved_hgrid, orientation=o)
    assert angle.shape == curved_hgrid.x.shape
    assert (
        abs(angle.values - curved_hgrid.angle_dx) < 1
    ).all()  # There shouldn't be large differences
    angle = rot.get_rotation_angle(rotational_method, rect_hgrid, orientation=o)
    assert angle.shape == rect_hgrid.x.shape
    assert (angle.values == 0).all()

    # Check if o is boundary that the shape is of a boundary
    o = "north"
    rotational_method = rot.RotationMethod.NO_ROTATION
    angle = rot.get_rotation_angle(rotational_method, rect_hgrid, orientation=o)
    assert angle.shape == rect_hgrid.x[-1].shape
    assert (angle.values == 0).all()
    rotational_method = rot.RotationMethod.EXPAND_GRID
    angle = rot.get_rotation_angle(rotational_method, rect_hgrid, orientation=o)
    assert angle.shape == rect_hgrid.x[-1].shape
    assert (angle.values == 0).all()
    rotational_method = rot.RotationMethod.GIVEN_ANGLE
    angle = rot.get_rotation_angle(rotational_method, rect_hgrid, orientation=o)
    assert angle.shape == rect_hgrid.x[-1].shape
    assert (angle.values == 0).all()
