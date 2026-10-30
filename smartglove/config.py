ENV_NAME = "myoHandPoseFixed-v0"
CALIBRATION_PACKETS = 30
IK_ITERATIONS = 20
FILTER_CUTOFF_HZ = 12.0
# Arah putaran telapak, dikoreksi dari pengamatan glove terhadap viewer.
WRIST_PRONATION_SIGN = -1.0
# Couple the two flexion joints because one distal quaternion cannot resolve both.
DISTAL_COUPLING = 2.0 / 3.0

FINGER_JOINTS = {
    "THUMB": {
        "joints": (
            "cmc_abduction",
            "cmc_flexion",
            "mp_flexion",
            "ip_flexion",
        ),
        "flexion_slots": (1, 2, 3),
        "body": "distal_thumb",
        "flexion_closed_limit": "low",
        "invert_abduction": True,
    },
    "INDEX": {
        "joints": (
            "mcp2_flexion",
            "mcp2_abduction",
            "pm2_flexion",
            "md2_flexion",
        ),
        "flexion_slots": (0, 2, 3),
        "body": "distph2",
    },
    "MIDDLE": {
        "joints": (
            "mcp3_flexion",
            "mcp3_abduction",
            "pm3_flexion",
            "md3_flexion",
        ),
        "flexion_slots": (0, 2, 3),
        "body": "distph3",
    },
    "RING": {
        "joints": (
            "mcp4_flexion",
            "mcp4_abduction",
            "pm4_flexion",
            "md4_flexion",
        ),
        "flexion_slots": (0, 2, 3),
        "body": "distph4",
    },
    "PINKY": {
        "joints": (
            "mcp5_flexion",
            "mcp5_abduction",
            "pm5_flexion",
            "md5_flexion",
        ),
        "flexion_slots": (0, 2, 3),
        "body": "distph5",
    },
}
