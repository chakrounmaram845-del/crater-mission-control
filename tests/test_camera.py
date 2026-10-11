import math

from common.camera import ground_to_pixel, intrinsics, pixel_to_ground

CX, CY, FOCAL = intrinsics(640, 480)


def test_ground_to_pixel_is_inverse_of_pixel_to_ground():
    for x, y in [(0, 80), (30, 120), (-45, 200), (60, 60)]:
        u, v = ground_to_pixel(x, y, CX, CY, FOCAL)
        gx, gy = pixel_to_ground(u, v, CX, CY, FOCAL)
        assert abs(gx - x) <= 1 and abs(gy - y) <= 1


def test_point_behind_camera_has_no_pixel():
    assert ground_to_pixel(0, -500, CX, CY, FOCAL) is None


def test_farther_point_is_higher_in_image():
    _, v_near = ground_to_pixel(0, 60, CX, CY, FOCAL)
    _, v_far = ground_to_pixel(0, 200, CX, CY, FOCAL)
    assert v_far < v_near


def test_focal_matches_field_of_view():
    from common.camera import HFOV
    assert abs(FOCAL - 320 / math.tan(HFOV / 2)) < 1e-6
