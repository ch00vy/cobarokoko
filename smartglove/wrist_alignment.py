"""Penyelarasan sumbu sensor pergelangan terhadap sumbu sendi MyoHand.

Sensor IMU pada punggung tangan dan lengan bawah dipasang dengan orientasi
bebas, jadi sumbu lokal sensor TIDAK sama dengan sumbu fleksi/deviasi/pronasi
model. Modul ini mengestimasi sumbu anatomis di frame lokal sensor dari
gerakan kalibrasi (sapuan satu-sumbu), lalu menyusun rotasi tetap A
(frame sensor -> frame lunate model) dengan metode Kabsch.
"""

import json
from pathlib import Path

import numpy as np

from smartglove.quaternion import (
    normalize_quaternion,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_to_rotvec,
)

AXIS_NAMES = ("flexion", "deviation", "pro_sup")
DEFAULT_AXES_FILE = Path(__file__).with_name("wrist_axes.json")


def relative_hand_in_forearm(forearm: np.ndarray, hand: np.ndarray) -> np.ndarray:
    """Orientasi tangan pada frame lengan bawah: conj(F) * H."""
    return normalize_quaternion(
        quaternion_multiply(
            quaternion_conjugate(normalize_quaternion(forearm)),
            normalize_quaternion(hand),
        )
    )


def sweep_rotvecs(
    forearm_quaternions: list[np.ndarray],
    hand_quaternions: list[np.ndarray],
    neutral_frames: int = 10,
) -> np.ndarray:
    """Rotvec delta (frame tangan saat netral) untuk satu sapuan gerak."""
    relative = [
        relative_hand_in_forearm(f, h)
        for f, h in zip(forearm_quaternions, hand_quaternions)
    ]
    reference = relative[0]
    # Rata-rata kasar beberapa frame awal sebagai netral sapuan.
    aligned = [r if np.dot(reference, r) >= 0 else -r for r in relative[:neutral_frames]]
    neutral = normalize_quaternion(np.mean(aligned, axis=0))
    return np.array(
        [
            quaternion_to_rotvec(quaternion_multiply(quaternion_conjugate(neutral), r))
            for r in relative
        ]
    )


def principal_axis(rotvecs: np.ndarray) -> dict:
    """Sumbu dominan sapuan satu-sumbu.

    Arah sumbu diambil dari variasi data (SVD setelah centering) sehingga tetap
    benar walau rekaman tidak dimulai tepat di pose netral; simpangan dihitung
    terhadap pose awal rekaman.
    """
    _, singular, vt = np.linalg.svd(rotvecs - rotvecs.mean(axis=0), full_matrices=False)
    axis = vt[0] / np.linalg.norm(vt[0])
    projected = np.degrees(rotvecs @ axis)
    # Tanda sumbu: sisi dengan simpangan terbesar dibuat positif.
    if abs(projected.min()) > abs(projected.max()):
        axis, projected = -axis, -projected
    energy = singular**2
    return {
        "axis": axis,
        "explained": float(energy[0] / energy.sum()),
        "min_deg": float(projected.min()),
        "max_deg": float(projected.max()),
    }


def _orthonormal_basis(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """Basis ortonormal kanan: e1 = primary, e2 = secondary tegak lurus e1."""
    e1 = primary / np.linalg.norm(primary)
    residual = secondary - np.dot(secondary, e1) * e1
    norm = np.linalg.norm(residual)
    if norm < 1e-6:
        raise ValueError("Sumbu fleksi dan deviasi hampir sejajar; ulangi rekaman.")
    e2 = residual / norm
    return np.array((e1, e2, np.cross(e1, e2)))


def fit_alignment(
    sensor_axes: dict[str, np.ndarray],
    model_axes: dict[str, np.ndarray],
) -> np.ndarray:
    """Rotasi A (frame sensor -> frame lunate model).

    Sumbu fleksi (gerak paling bersih dan dominan) dipetakan EKSAK; sumbu
    deviasi hanya menentukan putaran di sekitar sumbu fleksi (komponen tegak
    lurusnya). Sumbu ketiga (pro_sup) mengikuti aturan tangan kanan, sehingga
    tanda pronasi tidak perlu ditentukan terpisah.
    """
    for name in ("flexion", "deviation"):
        if name not in sensor_axes or name not in model_axes:
            raise ValueError(f"Sumbu {name} dibutuhkan untuk penyelarasan.")
    sensor_basis = _orthonormal_basis(sensor_axes["flexion"], sensor_axes["deviation"])
    model_basis = _orthonormal_basis(model_axes["flexion"], model_axes["deviation"])
    return model_basis.T @ sensor_basis


def predicted_pro_sup_axis(sensor_axes: dict[str, np.ndarray]) -> np.ndarray:
    """Sumbu pronasi di frame sensor yang diprediksi dari fleksi + deviasi."""
    return _orthonormal_basis(sensor_axes["flexion"], sensor_axes["deviation"])[2]


def axes_quality(sensor_axes: dict[str, np.ndarray]) -> dict[str, float]:
    """Sudut antar sumbu terukur (idealnya 90 derajat)."""
    names = [n for n in AXIS_NAMES if n in sensor_axes]
    result = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            cosine = float(
                np.clip(
                    np.dot(sensor_axes[a], sensor_axes[b])
                    / (np.linalg.norm(sensor_axes[a]) * np.linalg.norm(sensor_axes[b])),
                    -1.0,
                    1.0,
                )
            )
            result[f"{a}-{b}"] = float(np.degrees(np.arccos(cosine)))
    return result


def save_axes(path: Path, order: str, axes: dict[str, dict]) -> None:
    path = Path(path)
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing[order] = {
        name: {
            "axis": [float(v) for v in info["axis"]],
            "explained": info.get("explained"),
            "min_deg": info.get("min_deg"),
            "max_deg": info.get("max_deg"),
        }
        for name, info in axes.items()
    }
    path.write_text(json.dumps(existing, indent=2))


def load_axes(path: Path, order: str) -> dict[str, np.ndarray] | None:
    path = Path(path)
    if not path.exists():
        return None
    stored = json.loads(path.read_text()).get(order)
    if not stored:
        return None
    return {name: np.array(info["axis"], dtype=float) for name, info in stored.items()}
