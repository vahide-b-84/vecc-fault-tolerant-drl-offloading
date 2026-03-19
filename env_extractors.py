# env_extractors.py
# Utility functions for extracting network-related constants from SUMO XML files
# and estimating upper bounds used for normalization and delay modeling.

import xml.etree.ElementTree as ET
import numpy as np
from configuration import parameters


def extract_max_speed_from_rou(xml_file="SUMO/test.rou.xml", vehicle_type="Car"):
    """
    Extract the maximum vehicle speed from a SUMO .rou.xml file.

    Parameters:
        xml_file (str): Path to the SUMO route file.
        vehicle_type (str): vType ID to search for (default: "Car").

    Returns:
        float: maxSpeed value defined in the vType section.
               If not found, returns a fallback default (30.0).
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Search for the specified vehicle type
    for vtype in root.findall("vType"):
        if vtype.get("id") == vehicle_type:
            return float(vtype.get("maxSpeed"))

    # Fallback default if vehicle type is not found
    return 30.0


def extract_rsu_positions_from_additional(additional_file="SUMO/test.add.xml"):
    """
    Parse RSU (POI) positions from a SUMO .add.xml file.

    Parameters:
        additional_file (str): Path to the SUMO additional file.

    Returns:
        list[np.ndarray]: List of RSU positions as 2D numpy arrays [x, y].
    """
    tree = ET.parse(additional_file)
    root = tree.getroot()

    positions = []
    for poi in root.findall("poi"):
        x = float(poi.get("x"))
        y = float(poi.get("y"))
        positions.append(np.array([x, y]))

    return positions


def compute_max_rsu_distance(additional_file="SUMO/test.add.xml"):
    """
    Compute the maximum Euclidean distance between any two RSUs.

    This value is used as an upper bound for propagation delay estimation
    and for state normalization.

    Parameters:
        additional_file (str): Path to the SUMO additional file.

    Returns:
        float: Maximum pairwise RSU distance.
    """
    positions = extract_rsu_positions_from_additional(additional_file)

    max_dist = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            max_dist = max(max_dist, dist)

    return max_dist


def estimate_max_e2e_delay():
    """
    Estimate an upper bound on end-to-end (E2E) delay.

    This function provides a conservative maximum delay value that combines:
        - Transmission delay
        - Propagation delay
        - Queueing delay
        - Expected retransmission delay (based on link failure probability)

    The result is used mainly for normalization and scaling purposes
    in state representation and reward design.

    Returns:
        float: Estimated maximum E2E delay.
    """
    # Worst-case task size
    max_task_size = parameters.TASK_SIZE_RANGE[1]

    # Minimum possible link bandwidth (worst case)
    min_bandwidth = parameters.RSU_LINK_BANDWIDTH_RANGE[0]

    # Maximum RSU-to-RSU distance
    max_dist = compute_max_rsu_distance()

    # Network propagation speed
    speed = parameters.network_speed

    # Queueing model parameters
    alpha = parameters.Q_alpha
    beta = parameters.beta

    # Maximum normalized load ratio (worst case)
    max_L_ratio = 1

    # Upper-bound failure probability (clipped)
    P = min(parameters.link_failure_rate_range[1] + beta * max_L_ratio, 0.7)

    # Expected number of retransmissions
    E_N = (1 / (1 - P)) - 1

    # Transmission delay
    D_trans = max_task_size / min_bandwidth

    # Propagation delay
    D_prop = max_dist / speed

    # Queueing delay
    D_queue = alpha * max_L_ratio

    # Retransmission delay
    D_retrans = E_N * (D_trans + D_prop)

    # Total estimated E2E delay
    return D_trans + D_prop + D_queue + D_retrans
