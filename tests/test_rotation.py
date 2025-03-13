import regional_mom6 as rmom6
import regional_mom6.rotation as rot
import regional_mom6.regridding as rgd
import math
import pytest
import xarray as xr
import numpy as np
import os

tol_angle = 5e-2  # tolerance for angles (in degrees)


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
    assert (
        (
            expanded_hgrid.x.values[0, 1:-1]
            - (hgrid.x.values[0, :] - (hgrid.x.values[1, :] - hgrid.x.values[0, :]))
        )
        < tol_angle
    ).all()
    assert (
        (
            expanded_hgrid.x.values[1:-1, 0]
            - (hgrid.x.values[:, 0] - (hgrid.x.values[:, 1] - hgrid.x.values[:, 0]))
        )
        < tol_angle
    ).all()
    assert (
        (
            expanded_hgrid.x.values[-1, 1:-1]
            - (hgrid.x.values[-1, :] - (hgrid.x.values[-2, :] - hgrid.x.values[-1, :]))
        )
        < tol_angle
    ).all()
    assert (
        (
            expanded_hgrid.x.values[1:-1, -1]
            - (hgrid.x.values[:, -1] - (hgrid.x.values[:, -2] - hgrid.x.values[:, -1]))
        )
        < tol_angle
    ).all()

    # Check corners for the same...
    assert (
        expanded_hgrid.x.values[0, 0]
        - (hgrid.x.values[0, 0] - (hgrid.x.values[1, 1] - hgrid.x.values[0, 0]))
    ) < tol_angle
    assert (
        expanded_hgrid.x.values[-1, 0]
        - (hgrid.x.values[-1, 0] - (hgrid.x.values[-2, 1] - hgrid.x.values[-1, 0]))
    ) < tol_angle
    assert (
        expanded_hgrid.x.values[0, -1]
        - (hgrid.x.values[0, -1] - (hgrid.x.values[1, -2] - hgrid.x.values[0, -1]))
    ) < tol_angle
    assert (
        expanded_hgrid.x.values[-1, -1]
        - (hgrid.x.values[-1, -1] - (hgrid.x.values[-2, -2] - hgrid.x.values[-1, -1]))
    ) < tol_angle

    # Same for y
    assert (
        (
            expanded_hgrid.y.values[0, 1:-1]
            - (hgrid.y.values[0, :] - (hgrid.y.values[1, :] - hgrid.y.values[0, :]))
        )
        < tol_angle
    ).all()
    assert (
        (
            expanded_hgrid.y.values[1:-1, 0]
            - (hgrid.y.values[:, 0] - (hgrid.y.values[:, 1] - hgrid.y.values[:, 0]))
        )
        < tol_angle
    ).all()
    assert (
        (
            expanded_hgrid.y.values[-1, 1:-1]
            - (hgrid.y.values[-1, :] - (hgrid.y.values[-2, :] - hgrid.y.values[-1, :]))
        )
        < tol_angle
    ).all()
    assert (
        (
            expanded_hgrid.y.values[1:-1, -1]
            - (hgrid.y.values[:, -1] - (hgrid.y.values[:, -2] - hgrid.y.values[:, -1]))
        )
        < tol_angle
    ).all()

    assert (
        expanded_hgrid.y.values[0, 0]
        - (hgrid.y.values[0, 0] - (hgrid.y.values[1, 1] - hgrid.y.values[0, 0]))
    ) < tol_angle
    assert (
        expanded_hgrid.y.values[-1, 0]
        - (hgrid.y.values[-1, 0] - (hgrid.y.values[-2, 1] - hgrid.y.values[-1, 0]))
    ) < tol_angle
    assert (
        expanded_hgrid.y.values[0, -1]
        - (hgrid.y.values[0, -1] - (hgrid.y.values[1, -2] - hgrid.y.values[0, -1]))
    ) < tol_angle
    assert (
        expanded_hgrid.y.values[-1, -1]
        - (hgrid.y.values[-1, -1] - (hgrid.y.values[-2, -2] - hgrid.y.values[-1, -1]))
    ) < tol_angle

    return


@pytest.mark.parametrize(("angle"), [0, 12.5, 65, -20])
def test_mom6_angle_calculation_method_simple_square_grids(angle):
    """
    Create a square of length 2. Rotate it by an `angle` and then compute
    the angle using rot.mom6_angle_calculation_method to ensure it gets
    the angle right.
    """

    # Rotation matrix
    θ = np.deg2rad(angle)  # radians
    R = np.array([[np.cos(θ), -np.sin(θ)], [np.sin(θ), np.cos(θ)]])

    # Define four point points on a square with side of
    # length 2 and centered at (0, 0)
    top_left = np.array([-1, +1])
    top_right = np.array([+1, +1])
    bottom_left = np.array([-1, -1])
    bottom_right = np.array([+1, -1])

    # Apply the rotation
    top_left = R @ top_left
    top_right = R @ top_right
    bottom_left = R @ bottom_left
    bottom_right = R @ bottom_right

    # translate the 4 rotated square points so that
    # the center of the square is at (center_x, center_y)
    center_x, center_y = 0, 0

    top_left[0] += center_x
    top_left[1] += center_y
    top_right[0] += center_x
    top_right[1] += center_y
    bottom_left[0] += center_x
    bottom_left[1] += center_y
    bottom_right[0] += center_x
    bottom_right[1] += center_y

    # create that dataset with the points
    top_left = xr.Dataset(
        {
            "x": (("nyp", "nxp"), [[top_left[0]]]),
            "y": (("nyp", "nxp"), [[top_left[1]]]),
        }
    )
    top_right = xr.Dataset(
        {
            "x": (("nyp", "nxp"), [[top_right[0]]]),
            "y": (("nyp", "nxp"), [[top_right[1]]]),
        }
    )
    bottom_left = xr.Dataset(
        {
            "x": (("nyp", "nxp"), [[bottom_left[0]]]),
            "y": (("nyp", "nxp"), [[bottom_left[1]]]),
        }
    )
    bottom_right = xr.Dataset(
        {
            "x": (("nyp", "nxp"), [[bottom_right[0]]]),
            "y": (("nyp", "nxp"), [[bottom_right[1]]]),
        }
    )
    point = xr.Dataset(
        {
            "x": (("nyp", "nxp"), [[center_x]]),
            "y": (("nyp", "nxp"), [[center_y]]),
        }
    )

    # Calculate len_lon
    top_left_bottom_right_diag = abs(top_left.x.item() - bottom_right.x.item())
    top_right_bottom_left_diag = abs(top_right.x.item() - bottom_left.x.item())
    len_lon = max(top_left_bottom_right_diag, top_right_bottom_left_diag)
    computed_angle = rot.mom6_angle_calculation_method(
        len_lon, top_left, top_right, bottom_left, bottom_right, point
    )

    assert math.isclose(computed_angle, angle)


def test_mom6_angle_calculation_method(get_curvilinear_hgrid):
    # Rotated grid
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
        < tol_angle
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
        < tol_angle
    ).all()  # Angle is correct
    assert angle.values.shape == ds_t.tlon.shape  # Shape is correct
    return


def test_initialize_grid_rotation_angle_using_expanded_hgrid(get_curvilinear_hgrid):
    """
    Generate a curvilinear grid and test the grid rotation angle at t_points based on what we pass to generate_curvilinear_grid
    """
    hgrid = get_curvilinear_hgrid
    angle = rot.initialize_grid_rotation_angles_using_expanded_hgrid(hgrid)

    assert (angle.values - hgrid.angle_dx < tol_angle).all()
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
        abs(angle.values - curved_hgrid.angle_dx) < tol_angle
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
