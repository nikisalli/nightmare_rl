import numpy as np
from numpy import sin, cos, arccos, arctan2, sqrt
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass

PI = np.pi
EPSILON = 1e-6

def no_zero(a):   # make a value always not zero
    if abs(a) < EPSILON:
        return EPSILON
    else:
        return a


def fmap(x: float, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def asymmetrical_sigmoid(val):
    # from 0, 0 to 1, 1
    return 1 / (1 + np.e**(-13 * (val - 0.5)))  # width set to 13


def vmult(matrix, vector):
    return (matrix.T * vector).T


def rotate(pose, rot, pivot=None, inverse=False):
    # check if the argument contains an euler rotation vector or a quaternion
    if len(rot) == 3:  # euler
        r = R.from_rotvec(rot)
    elif len(rot) == 4:  # quaternion
        r = R.from_quat(rot)

    if inverse:
        r = r.inv()

    if pivot is not None:  # check if we have a pivot to rotate around
        p = pose - pivot
        p = r.apply(p)
        return p + pivot
    else:
        return r.apply(pose)


def limit(val, maxv, minv):
    return min(max(val, minv), maxv)


def line_segment_circle_lintersection(pt1, pt2, circle_center, circle_radius, full_line=True, tangent_tol=1e-9):
    """ Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.

    :param circle_center: The (x, y) location of the circle center
    :param circle_radius: The radius of the circle
    :param pt1: The (x, y) location of the first point of the segment
    :param pt2: The (x, y) location of the second point of the segment
    :param full_line: True to find intersections along full line - not just in the segment.  False will just return intersections within the segment.
    :param tangent_tol: Numerical tolerance at which we decide the intersections are close enough to consider it a tangent
    :return Sequence[Tuple[float, float]]: A list of length 0, 1, or 2, where each element is a point at which the circle intercepts a line segment.

    Note: We follow: http://mathworld.wolfram.com/Circle-LineIntersection.html
    """

    (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx ** 2 + dy ** 2)**.5
    big_d = x1 * y2 - x2 * y1
    discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2

    if discriminant < 0:  # No intersection between circle and line
        return []
    else:  # There may be 0, 1, or 2 intersections with the segment
        intersections = [
            (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**.5) / dr ** 2,
             cy + (-big_d * dx + sign * abs(dy) * discriminant**.5) / dr ** 2)
            for sign in ((1, -1) if dy < 0 else (-1, 1))]  # This makes sure the order along the segment is correct
        if not full_line:  # If only considering the segment, filter out intersections that do not fall within the segment
            fraction_along_segment = [
                (xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in intersections]
            intersections = [pt for pt, frac in zip(
                intersections, fraction_along_segment) if 0 <= frac <= 1]
        # If line is tangent to circle, return just one point (as both intersections have same location)
        if len(intersections) == 2 and abs(discriminant) <= tangent_tol:
            return [intersections[0]]
        else:
            return intersections


def line_circle_intersection(pt1, pt2, circle_center, circle_radius, tangent_tol=1e-9):
    return line_segment_circle_lintersection(pt1, pt2, circle_center, circle_radius, True, tangent_tol)


def segment_circle_intersection(pt1, pt2, circle_center, circle_radius, tangent_tol=1e-9):
    return line_segment_circle_lintersection(pt1, pt2, circle_center, circle_radius, False, tangent_tol)


# Return true if line segments AB and CD intersect
def segments_intersect_2d(A, B, C, D):
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def shortest_distance_point_segment_2d(p1, p2, p):
    """ Find the shortest distance between a point and a line segment in 2D.

    :param p1: The (x, y) location of the first point of the segment
    :param p2: The (x, y) location of the second point of the segment
    :param p: The (x, y) location of the point
    :return float: The shortest distance between the point and the line segment

    Note: We follow: http://geomalgorithms.com/a02-_lines.html#Distance-to-Ray-or-Segment
    """
    (x1, y1), (x2, y2), (px, py) = p1, p2, p
    (dx, dy) = (x2 - x1), (y2 - y1)
    if dx == 0 and dy == 0:  # Special case for a point
        return ((px - x1) ** 2 + (py - y1) ** 2) ** .5
    else:
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        if t < 0:  # closest point is p1
            return ((px - x1) ** 2 + (py - y1) ** 2) ** .5
        elif t > 1:  # closest point is p2
            return ((px - x2) ** 2 + (py - y2) ** 2) ** .5
        else:  # closest point is p1 + t*(p2-p1)
            return ((px - (x1 + t * dx)) ** 2 + (py - (y1 + t * dy)) ** 2) ** .5


def shortest_distance_two_segments_2d(p1a, p1b, p2a, p2b):
    """ Find the shortest distance between two line segments.

    :param p1a: The first point of the first segment
    :param p1b: The second point of the first segment
    :param p2a: The first point of the second segment
    :param p2b: The second point of the second segment
    :return float: The shortest distance between the two segments
    """
    if segments_intersect_2d(p1a, p1b, p2a, p2b):
        return 0
    else:
        d1 = shortest_distance_point_segment_2d(p1a, p1b, p2a)
        d2 = shortest_distance_point_segment_2d(p1a, p1b, p2b)
        d3 = shortest_distance_point_segment_2d(p2a, p2b, p1a)
        d4 = shortest_distance_point_segment_2d(p2a, p2b, p1b)
        return min(d1, d2, d3, d4)
