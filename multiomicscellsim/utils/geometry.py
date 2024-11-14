import math
import numpy as np
from typing import List


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

def cartesian_to_circle_angle(points, xc, yc):
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

def circle_polar_to_cartesian(theta, xc, yc, r):
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

def is_point_in_rectangle(px, py, x_min, y_min, x_max, y_max):
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

to_degrees = lambda x: x * (180 / math.pi)

normalize_angle = lambda x: x % (2*math.pi)

def angle_midpoint(start, end) -> float:
    """
        Get the angle midpoint between start and end considering a counterclockwise motion. 
        
        That is:
        angle_midpoint(0, math.pi) # Should return pi/2
        angle_midpoint(math.pi, 2*math.pi) # Should return 3/2pi
        angle_midpoint(math.pi/2, 3/2 * math.pi)  # Should return math.pi
        angle_midpoint(3/2 * math.pi, math.pi/2)  # Should return 0
        angle_midpoint(-math.pi/2, math.pi/2)  # Should return 0
        angle_midpoint(math.pi/2, -math.pi/2)  # Should return math.pi
    """
    if start > end:
        return abs(angle_midpoint(start=end, end=start) - math.pi)
    return start + (end - start) / 2

def get_arcs_inside_rectangle(xc, yc, rc, xr_min, yr_min, xr_max, yr_max):
    """
    Given a circle and a rectangle, returns the radians intervals for each arc that lies within the rectangle.
    If the circle is completely contained in the rectangle (no intersections), returns a single interval [0; 2π].

    Parameters
    ----------
    xc : float
        x-coordinate of the circle center.
    yc : float
        y-coordinate of the circle center.
    rc : float
        Radius of the circle.
    xr_min : float
        Minimum x-coordinate of the rectangle.
    yr_min : float
        Minimum y-coordinate of the rectangle.
    xr_max : float
        Maximum x-coordinate of the rectangle.
    yr_max : float
        Maximum y-coordinate of the rectangle.

    Returns
    -------
    list of [float, float]
        A list of [start, end] angle intervals (in radians) for each arc that lies within the rectangle.
        Each interval represents an arc segment of the circle within the rectangular boundary.
        If the circle is completely within the rectangle, a single interval [0, 2π] is returned.
    """

    arcs = []
    intersections = circle_rectangle_intersections(xc, yc, rc, [xr_min, yr_min, xr_max, yr_max])
    # Find the angles theta corresponding with intersection with borders
    intersection_angles = cartesian_to_circle_angle(points=intersections, xc=xc, yc=yc)

    # Iterate over all the arcs found, counterclockwise
    for start, end in zip(intersection_angles, np.roll(intersection_angles, -1)):
        # Get the middlepoint (counterclockwise, not on the shortest path)
        mid = angle_midpoint(start, end)
        x_mid, y_mid = circle_polar_to_cartesian(theta=mid, xc=xc, yc=yc, r=rc)
        
        if is_point_in_rectangle(x_mid, y_mid, xr_min, yr_min, xr_max, yr_max):
            if start > end:
                arcs.append([end, start])
            else:
                arcs.append([start, end])
    
    if len(arcs) == 0:
        arcs.append([0, 2*math.pi])

    return arcs


def map_samples_to_arcs(samples: List[float], arcs: List[List[float]]):
    """
        Given a set of arcs belonging to a circumference, maps some samples from [0, 1] to the arcs as they were a contiguous line.

        Parameters
        -------
        samples: A list of samples in [0, 1)
        arcs: A list of lists, [[start, end], [....]]
              where start and ends are the start and end angles for the arc (in radians)
    """
    # Mapping theta values from [0, 1] to visible arcs of the guideline
    total_arc_length = sum(end - start for start, end in arcs)

    sampled_thetas = []
    for sample in samples:
        # Scale sample to the range [0, total_arc_length]
        sample_position = sample * total_arc_length
        
        # Find which arc this sample falls into
        cumulative_length = 0
        for start, end in arcs:
            arc_length = end - start
            if cumulative_length + arc_length >= sample_position:
                # Sample is in this arc
                # Calculate the angle within this arc
                angle = start + (sample_position - cumulative_length)
                sampled_thetas.append(angle)
                break
            cumulative_length += arc_length

    return sampled_thetas


