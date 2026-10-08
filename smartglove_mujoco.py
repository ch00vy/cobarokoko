"""Compatibility exports for the former smartglove_mujoco module."""

from cli import main
from smartglove.config import (
    CALIBRATION_PACKETS,
    DISTAL_COUPLING,
    ENV_NAME,
    FILTER_CUTOFF_HZ,
    FINGER_JOINTS,
    IK_ITERATIONS,
)
from smartglove.mapping import HandQposMapper
from smartglove.quaternion import (
    QuaternionLowPassFilter,
    average_quaternions,
    normalize_quaternion,
    quaternion_conjugate,
    quaternion_error,
    quaternion_multiply,
    quaternion_relative,
    quaternion_slerp,
    sensor_quaternion_to_wxyz,
)
from smartglove.runtime import capture_sensor_quaternions, drain_udp_queue, run
from smartglove_realtime import calculate_packet_rate


if __name__ == "__main__":
    main()
