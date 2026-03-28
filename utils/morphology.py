import cv2

import numpy as np



def classify_morphology(contour, rect):

    """

    Classifies a particle based on geometric heuristics.

    Returns: category (str), base_risk (int), solidity (float)

    """

    w, h = rect[1]

    if w == 0 or h == 0:

        return "Unknown", 0, 0



    aspect_ratio = max(w, h) / (min(w, h) if min(w, h) > 0 else 1)

    area = cv2.contourArea(contour)

    hull = cv2.convexHull(contour)

    hull_area = cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 1

   

    solidity = float(area) / hull_area

    perimeter = cv2.arcLength(contour, True)

    circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0



    # Decision Tree Logic

    if aspect_ratio > 3.0:

        category = "Fiber"

        base_risk = 90

    elif solidity < 0.88:

        category = "Fragment"

        base_risk = 70

    elif circularity < 0.75:

        category = "Film"

        base_risk = 50

    else:

        category = "Pellet"

        base_risk = 40



    return category, base_risk, solidity



def calculate_risk(category, size_um):

    """

    Calculates a risk score 0-100 based on shape and bioavailability (size).

    """

    # Smaller particles are more dangerous as they cross biological barriers

    size_factor = max(0, (1000 - size_um) / 10)

   

    category_weights = {

        "Fiber": 1.2,

        "Fragment": 1.0,

        "Film": 0.9,

        "Pellet": 0.8,

        "Unknown": 0.5

    }

   

    weight = category_weights.get(category, 1.0)

    risk_score = (50 + size_factor) * weight

   

    return min(100, max(0, risk_score))