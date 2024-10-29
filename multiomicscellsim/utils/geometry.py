import math
import numpy as np


def circle_rectangle_intersections(xc: float, yc:float, rc, rectangle_corners):
    """
        Finds all the intersections between a circle and a rectangle.

        Returns:
        Array of intesections
    """
    h, k = xc, yc
    r = rc
    x_min, y_min, x_max, y_max = rectangle_corners
    intersections = []

    def check_edge(x_edge=None, y_edge=None, is_vertical=True):
        """Check intersections for a given edge."""
        if is_vertical:
            x = x_edge
            discriminant = r**2 - (x - h)**2
            if discriminant >= 0:
                y1 = k + np.sqrt(discriminant)
                y2 = k - np.sqrt(discriminant)
                if y_min <= y1 <= y_max:
                    intersections.append((x, y1))
                if y_min <= y2 <= y_max:
                    intersections.append((x, y2))
        else:
            y = y_edge
            discriminant = r**2 - (y - k)**2
            if discriminant >= 0:
                x1 = h + np.sqrt(discriminant)
                x2 = h - np.sqrt(discriminant)
                if x_min <= x1 <= x_max:
                    intersections.append((x1, y))
                if x_min <= x2 <= x_max:
                    intersections.append((x2, y))

    # Check all edges of the rectangle
    check_edge(x_edge=x_min, is_vertical=True)  # Left edge
    check_edge(x_edge=x_max, is_vertical=True)  # Right edge
    check_edge(y_edge=y_min, is_vertical=False)  # Bottom edge
    check_edge(y_edge=y_max, is_vertical=False)  # Top edge

    return intersections

import math

def cartesian_to_angle(points, xc, yc):
    """
    Convert a list of Cartesian coordinates to angles (in radians) 
    based on a given circle's center.
    
    Parameters:
    points (list of tuple): A list of tuples where each tuple contains 
                            the x and y coordinates of a point (xi, yi).
    xc (float): The x-coordinate of the center of the circle.
    yc (float): The y-coordinate of the center of the circle.
    
    Returns:
    list of float: A list of angles in radians corresponding to each point.
    """
    angles = []
    for (xi, yi) in points:
        angle = math.atan2(yi - yc, xi - xc)  # in radians
        angles.append(angle)
    return angles

def angle_to_cartesian(theta, xc, yc, r):
    """
    Convert an angle (in radians) to Cartesian coordinates on a circle.
    
    Parameters:
    theta (float): The angle in radians.
    xc (float): The x-coordinate of the center of the circle.
    yc (float): The y-coordinate of the center of the circle.
    r (float): The radius of the circle.
    
    Returns:
    tuple: Cartesian coordinates (x, y) of the point on the circumference.
    """
    x = xc + r * math.cos(theta)
    y = yc + r * math.sin(theta)
    return (x, y)

def point_in_rectangle(px, py, x_min, y_min, x_max, y_max):
    """
    Check if a point lies within a rectangle.
    
    Parameters:
    px (float): x-coordinate of the point.
    py (float): y-coordinate of the point.
    x_min (float): x-coordinate of the bottom-left corner of the rectangle.
    y_min (float): y-coordinate of the bottom-left corner of the rectangle.
    x_max (float): x-coordinate of the top-right corner of the rectangle.
    y_max (float): y-coordinate of the top-right corner of the rectangle.
    
    Returns:
    bool: True if the point lies within the rectangle, False otherwise.
    """
    return x_min <= px <= x_max and y_min <= py <= y_max

def interpolate_angles(angle1, angle2):
    """
    Interpolate between two angles in radians to find the shorter 
    and longer midpoints.

    Parameters:
    angle1 (float): The first angle in radians.
    angle2 (float): The second angle in radians.

    Returns:
    tuple: A tuple containing the shorter midpoint and the longer 
           midpoint angles in radians.
    """
    # Normalize angles to the range [0, 2π)
    angle1 = angle1 % (2 * math.pi)
    angle2 = angle2 % (2 * math.pi)

    # Calculate the difference
    delta = angle2 - angle1

    # Ensure the shortest path
    if delta > math.pi:
        delta -= 2 * math.pi
    elif delta < -math.pi:
        delta += 2 * math.pi

    # Calculate midpoints
    midpoint_shorter = angle1 + delta / 2
    midpoint_longer = midpoint_shorter + math.pi if midpoint_shorter + math.pi < 2 * math.pi else midpoint_shorter - math.pi

    return midpoint_shorter, midpoint_longer

to_degrees = lambda x: x * (180 / math.pi)


def get_arcs_inside_rectangle(xc, yc, rc, xr_min, yr_min, xr_max, yr_max):
    """
        Given a circle and a rectangle, returns the radians intervals for each arc that lies within the rectangle.
    """
    intersections = circle_rectangle_intersections(xc, yc, rc, [xr_min, yr_min, xr_max, yr_max])

    arcs = []

    intersection_angles = cartesian_to_angle(points=intersections, xc=xc, yc=yc)
    # FIXME: Continue here

    print(intersection_angles)
    return arcs

