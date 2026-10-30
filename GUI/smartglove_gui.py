import csv
import importlib
import multiprocessing as mp
import re
import socket
import sys
import time
import tkinter as tk
from collections import deque
from datetime import datetime
from pathlib import Path
from queue import Empty, Full
from tkinter import messagebox, ttk
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import gymnasium as gym
import mujoco
import numpy as np

importlib.import_module("myosuite")

from GUI.full_viewer_worker import run_full_viewer
from control.smartglove_muscle_control import SmartGloveMuscleController
from fatigue import MyoSuiteFatigue
from smartglove.config import (
    CALIBRATION_PACKETS,
    ENV_NAME,
    FINGER_JOINTS,
)
from smartglove.mapping import HandQposMapper
from smartglove.quaternion import (
    QuaternionLowPassFilter,
    average_quaternions,
)
from smartglove.runtime import capture_sensor_quaternions
from smartglove.wrist_mapping import WristQposMapper
from smartglove_realtime import (
    RATE_WINDOW_INTERVALS,
    calculate_packet_rate,
    parse_packet_with_error,
)


RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"
FDS_NAMES = ("FDS2", "FDS3", "FDS4", "FDS5")


class SmartGloveGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Smart Glove + MuJoCo Live Preview")
        self.root.geometry("1120x760")
        self.root.minsize(900, 620)
        self.root.configure(background="#11161d")

        self.env: gym.Env | None = None
        self.model: Any = None
        self.data: Any = None
        self.mapper: HandQposMapper | None = None
        self.wrist_mapper: WristQposMapper | None = None
        self.muscle_controller: SmartGloveMuscleController | None = None
        self.fatigue_model: MyoSuiteFatigue | None = None
        self.quaternion_filter: QuaternionLowPassFilter | None = None
        self.full_viewer_process: Any = None
        self.full_viewer_state_queue: Any = None
        self.full_viewer_status_queue: Any = None
        self.full_viewer_ready = False
        self.last_full_viewer_update = 0.0
        self.udp_socket = None
        self.phase = "disconnected"
        self.frame_number = 0
        self.invalid_packets = 0
        self.packet_intervals: deque[float] = deque(maxlen=RATE_WINDOW_INTERVALS)
        self.last_packet_time = None
        self.last_datagram_time = None
        self.udp_receive_count = 0
        self.local_ipv4: list[str] = []
        self.calibration_samples: dict[str, list[np.ndarray]] = {}
        self.grip_calibration_samples: dict[str, list[np.ndarray]] = {}
        self.record_file = None
        self.csv_writer = None
        self.record_path: Path | None = None
        self.last_record_flush_time = 0.0
        self.fds_indices: dict[str, int] = {}
        self.preview_image = None
        self.latest_fds = dict.fromkeys(FDS_NAMES, 0.0)

        self.host_var = tk.StringVar(value="192.168.3.101")
        self.port_var = tk.StringVar(value="14041")
        self.subject_var = tk.StringVar(value="subject_smart_glove")
        self.connection_var = tk.StringVar(value="Disconnected")
        self.packet_var = tk.StringVar(value="Packets: 0")
        self.source_var = tk.StringVar(value="None")
        self.qpos_var = tk.StringVar(value="Waiting for glove frames")
        self.wrist_var = tk.StringVar(value="Waiting for wrist calibration")
        self.recording_var = tk.StringVar(value="Not recording")
        self.guidance_var = tk.StringVar(
            value="Connect to open the model and listen for glove data."
        )
        self.fatigue_enabled = False
        self.fds_vars = {name: tk.StringVar(value="0.000") for name in FDS_NAMES}

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self._tick()

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "TFrame",
            background="#19212b",
        )
        style.configure(
            "TLabel",
            background="#19212b",
            foreground="#dce7f5",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Header.TLabel",
            background="#11161d",
            foreground="#f3f6fb",
            font=("Segoe UI", 19, "bold"),
        )
        style.configure(
            "Section.TLabel",
            background="#19212b",
            foreground="#f3f6fb",
            font=("Segoe UI", 11, "bold"),
        )
        style.configure(
            "Hint.TLabel",
            background="#19212b",
            foreground="#92a8c2",
            font=("Segoe UI", 9),
        )
        style.configure(
            "Metric.TLabel",
            background="#19212b",
            foreground="#ffffff",
            font=("Segoe UI", 9, "bold"),
        )
        style.configure(
            "TEntry",
            fieldbackground="#141b23",
            foreground="#edf3fb",
            insertcolor="#edf3fb",
            bordercolor="#3b4654",
            padding=7,
        )
        style.configure(
            "Primary.TButton",
            background="#3275e8",
            foreground="#ffffff",
            padding=(10, 8),
            font=("Segoe UI", 9, "bold"),
        )
        style.map("Primary.TButton", background=[("active", "#4387fa")])
        style.configure(
            "TButton",
            background="#222c38",
            foreground="#dce7f5",
            padding=(8, 7),
        )

        container = ttk.Frame(self.root, padding=(16, 14))
        container.pack(fill="both", expand=True)
        ttk.Label(
            container,
            text="Smart Glove Realtime Control",
            style="Header.TLabel",
        ).pack(anchor="w", pady=(0, 14))

        columns = ttk.Frame(container)
        columns.pack(fill="both", expand=True)
        columns.columnconfigure(0, minsize=260, weight=0)
        columns.columnconfigure(1, weight=1)
        columns.rowconfigure(0, weight=1)

        self.left_panel = ttk.Frame(columns, padding=14, relief="solid")
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.right_panel = ttk.Frame(columns, padding=14, relief="solid")
        self.right_panel.grid(row=0, column=1, sticky="nsew")
        self.right_panel.columnconfigure(0, weight=1)
        self.right_panel.rowconfigure(3, weight=1)

        self._build_connection_panel()
        self._build_preview_panel()

    def _build_connection_panel(self) -> None:
        self.connection_canvas = tk.Canvas(
            self.left_panel,
            background="#19212b",
            highlightthickness=0,
            borderwidth=0,
        )
        scrollbar = ttk.Scrollbar(
            self.left_panel,
            orient="vertical",
            command=self.connection_canvas.yview,
        )
        self.connection_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.connection_canvas.pack(side="left", fill="both", expand=True)

        panel = ttk.Frame(self.connection_canvas, padding=14)
        self.connection_window = self.connection_canvas.create_window(
            (0, 0),
            window=panel,
            anchor="nw",
        )
        panel.bind("<Configure>", self._update_connection_scrollregion)
        self.connection_canvas.bind("<Configure>", self._resize_connection_content)
        self.root.bind_all(
            "<MouseWheel>",
            self._scroll_connection_panel,
        )

        ttk.Label(panel, text="Connection", style="Section.TLabel").pack(
            anchor="w"
        )
        ttk.Label(panel, text="UDP smart glove input", style="Hint.TLabel").pack(
            anchor="w", pady=(4, 14)
        )

        self._add_field(panel, "Host / bind IP", self.host_var)
        self._add_field(panel, "Port", self.port_var)
        self._add_field(panel, "Subject ID", self.subject_var)

        button_row = ttk.Frame(panel)
        button_row.pack(fill="x", pady=(8, 4))
        self.connect_button = ttk.Button(
            button_row,
            text="Connect",
            style="Primary.TButton",
            command=self._connect,
        )
        self.connect_button.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.disconnect_button = ttk.Button(
            button_row,
            text="Disconnect",
            command=self._disconnect,
            state="disabled",
        )
        self.disconnect_button.pack(side="left", fill="x", expand=True, padx=(4, 0))

        button_row = ttk.Frame(panel)
        button_row.pack(fill="x", pady=(0, 12))
        self.grip_button = ttk.Button(
            button_row,
            text="Calibrate grip",
            command=self._start_grip_calibration,
            state="disabled",
        )
        self.grip_button.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.record_button = ttk.Button(
            button_row,
            text="Record FDS",
            style="Primary.TButton",
            command=self._start_recording,
            state="disabled",
        )
        self.record_button.pack(side="left", fill="x", expand=True, padx=(4, 0))
        self.stop_record_button = ttk.Button(
            panel,
            text="Stop recording",
            command=self._stop_recording,
            state="disabled",
        )
        self.stop_record_button.pack(fill="x", pady=(0, 10))
        self.fatigue_button = ttk.Button(
            panel,
            text="Fatigue: OFF",
            command=self._toggle_fatigue,
            state="disabled",
        )
        self.fatigue_button.pack(fill="x", pady=(0, 10))

        ttk.Separator(panel).pack(fill="x", pady=4)
        self._status_row(panel, "Status", self.connection_var)
        self._status_row(panel, "Packets", self.packet_var)
        self._status_row(panel, "Glove source", self.source_var)
        self._status_row(panel, "Calibration", self.qpos_var)
        self._status_row(panel, "Wrist (deg)", self.wrist_var)
        self._status_row(panel, "Recording", self.recording_var)

        ttk.Separator(panel).pack(fill="x", pady=8)
        ttk.Label(panel, text="FDS activation", style="Section.TLabel").pack(
            anchor="w", pady=(0, 5)
        )
        for name in FDS_NAMES:
            self._status_row(panel, name, self.fds_vars[name])

        self.guidance_label = ttk.Label(
            panel,
            textvariable=self.guidance_var,
            style="Hint.TLabel",
            wraplength=228,
            justify="left",
        )
        self.guidance_label.pack(anchor="w", fill="x", pady=(12, 0))

    def _update_connection_scrollregion(self, _event: tk.Event) -> None:
        self.connection_canvas.configure(
            scrollregion=self.connection_canvas.bbox("all")
        )

    def _resize_connection_content(self, event: tk.Event) -> None:
        self.connection_canvas.itemconfigure(
            self.connection_window,
            width=event.width,
        )

    def _scroll_connection_panel(self, event: tk.Event) -> None:
        canvas_top = self.connection_canvas.winfo_rooty()
        canvas_bottom = canvas_top + self.connection_canvas.winfo_height()
        canvas_left = self.connection_canvas.winfo_rootx()
        canvas_right = canvas_left + self.connection_canvas.winfo_width()
        if not (
            canvas_left <= event.x_root < canvas_right
            and canvas_top <= event.y_root < canvas_bottom
        ):
            return
        self.connection_canvas.yview_scroll(
            -1 if event.delta > 0 else 1,
            "units",
        )

    @staticmethod
    def _add_field(parent: ttk.Frame, label: str, variable: tk.StringVar) -> None:
        ttk.Label(parent, text=label, style="Hint.TLabel").pack(
            anchor="w", pady=(0, 4)
        )
        ttk.Entry(parent, textvariable=variable).pack(fill="x", pady=(0, 9))

    @staticmethod
    def _status_row(
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, style="Hint.TLabel").pack(side="left")
        ttk.Label(row, textvariable=variable, style="Metric.TLabel").pack(
            side="right"
        )

    def _build_preview_panel(self) -> None:
        panel = self.right_panel
        ttk.Label(panel, text="MuJoCo Viewer", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            panel,
            text="The full viewer opens automatically after glove calibration.",
            style="Hint.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 10))
        self.preview_status = ttk.Label(
            panel,
            text="Waiting for connection",
            style="Hint.TLabel",
        )
        self.preview_status.grid(row=2, column=0, sticky="w", pady=(0, 8))
        controls = ttk.Frame(panel)
        controls.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        self.full_viewer_button = ttk.Button(
            controls,
            text="Open full viewer",
            command=self._toggle_full_viewer,
            state="disabled",
        )
        self.full_viewer_button.pack(side="left")
        ttk.Label(
            controls,
            text="Use the full viewer window to inspect and move the camera.",
            style="Hint.TLabel",
        ).pack(side="left", padx=(10, 0))
        self.viewer_message = tk.Label(
            panel,
            text=(
                "No embedded preview is rendered here.\n\n"
                "Connect the glove and complete the open-pose and grip "
                "calibration. MuJoCo will then open in its own full viewer "
                "window and follow the glove movement."
            ),
            background="#141b23",
            foreground="#92a8c2",
            font=("Segoe UI", 10),
            justify="center",
            relief="solid",
            borderwidth=1,
            wraplength=560,
        )
        self.viewer_message.grid(row=3, column=0, sticky="nsew")

    def _connect(self) -> None:
        if self.udp_socket is not None:
            return
        try:
            host = self.host_var.get().strip()
            port = int(self.port_var.get().strip())
            if not host:
                raise ValueError("Host/bind IP tidak boleh kosong.")
            if not 1 <= port <= 65535:
                raise ValueError("Port harus berada di antara 1 dan 65535.")

            subject = self.subject_var.get().strip()
            if not re.fullmatch(r"[\w.-]+", subject) or subject in {".", ".."}:
                raise ValueError(
                    "Subject ID hanya boleh berisi huruf, angka, titik, "
                    "garis bawah, atau tanda hubung."
                )

            env = gym.make(ENV_NAME)
            self.env = env
            unwrapped = env.unwrapped
            self.model = getattr(unwrapped, "mj_model")
            self.data = getattr(unwrapped, "mj_data")
            self.data.qpos[:] = getattr(unwrapped, "init_qpos")
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)  # type: ignore[attr-defined]
            self.mapper = HandQposMapper(
                self.model,
                self.data,
                invert_thumb_abduction=True,
            )
            self.wrist_mapper = WristQposMapper(self.model, self.data)
            self.muscle_controller = SmartGloveMuscleController(
                self.model,
                self.data,
                FINGER_JOINTS,
            )
            self.fatigue_model = MyoSuiteFatigue(
                self.model,
                time.perf_counter(),
            )
            self.fatigue_enabled = False
            self.fatigue_button.configure(text="Fatigue: OFF", state="disabled")
            self.fds_indices = {
                name: self.muscle_controller.actuator_names.index(name)
                for name in FDS_NAMES
            }
            self.quaternion_filter = QuaternionLowPassFilter()

            udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket = udp_socket
            udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
            udp_socket.bind((host, port))
            udp_socket.setblocking(False)
            self.local_ipv4 = sorted(
                {
                    str(address[4][0])
                    for address in socket.getaddrinfo(
                        socket.gethostname(),
                        None,
                        socket.AF_INET,
                    )
                    if not str(address[4][0]).startswith("127.")
                }
            )

            self.phase = "neutral"
            self.frame_number = 0
            self.invalid_packets = 0
            self.packet_intervals.clear()
            self.last_packet_time = None
            self.last_datagram_time = None
            self.udp_receive_count = 0
            self.calibration_samples = {finger: [] for finger in FINGER_JOINTS}
            self.wrist_calibration_samples = {
                sensor: [] for sensor in ("FOREARM", "HAND")
            }
            self.grip_calibration_samples = {
                finger: [] for finger in FINGER_JOINTS
            }
            self.latest_fds = dict.fromkeys(FDS_NAMES, 0.0)
            self.connection_var.set("Listening (no data)")
            self.packet_var.set("UDP: 0 | valid: 0")
            self.source_var.set("None")
            self.qpos_var.set("Open pose 0/30")
            self.recording_var.set("Not recording")
            destination = ", ".join(self.local_ipv4) or "PC Wi-Fi IPv4"
            self.guidance_var.set(
                f"Set glove UDP destination to {destination}:{port} "
                "on the same Wi-Fi. Waiting for glove packets."
            )
            self.preview_status.configure(text=f"Listening on {host}:{port}")
            self.full_viewer_button.configure(state="disabled")
            self.connect_button.configure(state="disabled")
            self.disconnect_button.configure(state="normal")
            self._update_fds_labels()
        except Exception as error:
            self._release_resources()
            messagebox.showerror("Connection failed", str(error), parent=self.root)
            self.connection_var.set("Disconnected")
            self.preview_status.configure(text="MuJoCo viewer closed")

    def _disconnect(self) -> None:
        self._stop_recording()
        self._release_resources()
        self.phase = "disconnected"
        self.connection_var.set("Disconnected")
        self.packet_var.set(
            f"UDP: {self.udp_receive_count} | valid: {self.frame_number}"
        )
        self.source_var.set("None")
        self.qpos_var.set("Waiting for glove frames")
        self.wrist_var.set("Waiting for wrist calibration")
        self.guidance_var.set(
            "Connect to open the model and listen for glove data."
        )
        self.preview_status.configure(text="MuJoCo viewer closed")
        self.connect_button.configure(state="normal")
        self.disconnect_button.configure(state="disabled")
        self.full_viewer_button.configure(state="disabled", text="Open full viewer")
        self.grip_button.configure(state="disabled")
        self.record_button.configure(state="disabled")
        self.fatigue_enabled = False
        self.fatigue_button.configure(text="Fatigue: OFF", state="disabled")
        self._update_fds_labels()

    def _release_resources(self) -> None:
        self._close_full_viewer()
        if self.udp_socket is not None:
            self.udp_socket.close()
            self.udp_socket = None
        if self.env is not None:
            self.env.close()
            self.env = None
        self.model = None
        self.data = None
        self.mapper = None
        self.wrist_mapper = None
        self.muscle_controller = None
        self.fatigue_model = None
        self.quaternion_filter = None

    def _toggle_fatigue(self) -> None:
        if self.fatigue_model is None or self.phase != "running":
            return
        self.fatigue_enabled = not self.fatigue_enabled
        self.fatigue_model.set_enabled(
            self.fatigue_enabled,
            time.perf_counter(),
        )
        state = "ON" if self.fatigue_enabled else "OFF"
        self.fatigue_button.configure(text=f"Fatigue: {state}")
        self.guidance_var.set(
            f"Model fatigue 3CC-r {state}. "
            "Efek fatigue mengubah aktivasi otot; pose saat ini tetap mengikuti glove."
        )

    def _start_grip_calibration(self) -> None:
        if self.phase != "await_grip":
            return
        self.phase = "grip"
        self.grip_calibration_samples = {finger: [] for finger in FINGER_JOINTS}
        self.qpos_var.set("Grip pose 0/30")
        self.guidance_var.set(
            "Tahan genggaman yang nyaman sampai 30 paket diterima."
        )
        self.grip_button.configure(state="disabled")

    def _tick(self) -> None:
        if self.udp_socket is not None:
            self._receive_packets()
            self._sync_full_viewer()
            self._update_glove_connection_status()
            self._flush_recording_if_due()
        self.root.after(10, self._tick)

    def _toggle_full_viewer(self) -> None:
        if self.full_viewer_process is not None:
            self._close_full_viewer()
            return
        if self.model is None or self.data is None:
            return
        try:
            context = mp.get_context("spawn")
            self.full_viewer_state_queue = context.Queue(maxsize=1)
            self.full_viewer_status_queue = context.Queue()
            process = context.Process(
                target=run_full_viewer,
                args=(
                    self.full_viewer_state_queue,
                    self.full_viewer_status_queue,
                ),
                name="MuJoCo full viewer",
            )
            process.start()
            self.full_viewer_process = process
            self.full_viewer_ready = False
            self.last_full_viewer_update = 0.0
            self.full_viewer_button.configure(text="Opening full viewer...")
            self.preview_status.configure(text="Opening full MuJoCo viewer")
        except Exception as error:
            self._close_full_viewer()
            messagebox.showerror(
                "MuJoCo viewer failed",
                str(error),
                parent=self.root,
            )

    def _sync_full_viewer(self) -> None:
        process = self.full_viewer_process
        if process is None:
            return

        while self.full_viewer_status_queue is not None:
            try:
                status, details = self.full_viewer_status_queue.get_nowait()
            except Empty:
                break
            if status == "ready":
                self.full_viewer_ready = True
                self.full_viewer_button.configure(text="Close full viewer")
                self.preview_status.configure(
                    text="Full MuJoCo viewer live"
                )
            elif status == "error":
                self._close_full_viewer()
                messagebox.showerror(
                    "MuJoCo viewer failed",
                    details,
                    parent=self.root,
                )
                return
            elif status == "closed":
                self._close_full_viewer()
                return

        if not process.is_alive():
            exit_code = process.exitcode
            self._close_full_viewer()
            if exit_code not in (None, 0):
                self.preview_status.configure(
                    text=f"Full MuJoCo viewer exited with code {exit_code}"
                )
            return

        now = time.perf_counter()
        if not self.full_viewer_ready or now - self.last_full_viewer_update < 1 / 60:
            return
        if self.data is None or self.full_viewer_state_queue is None:
            return
        state = (self.data.qpos.copy(), self.data.ctrl.copy())
        try:
            self.full_viewer_state_queue.put_nowait(state)
        except Full:
            try:
                self.full_viewer_state_queue.get_nowait()
            except Empty:
                pass
            try:
                self.full_viewer_state_queue.put_nowait(state)
            except Full:
                return
        self.last_full_viewer_update = now

    def _close_full_viewer(self) -> None:
        process = self.full_viewer_process
        state_queue = self.full_viewer_state_queue
        status_queue = self.full_viewer_status_queue
        self.full_viewer_process = None
        self.full_viewer_state_queue = None
        self.full_viewer_status_queue = None
        self.full_viewer_ready = False
        if process is not None:
            if process.is_alive() and state_queue is not None:
                try:
                    state_queue.put_nowait(None)
                except Full:
                    try:
                        state_queue.get_nowait()
                        state_queue.put_nowait(None)
                    except (Empty, Full):
                        pass
                process.join(timeout=3.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
            process.close()
        if state_queue is not None:
            state_queue.close()
            state_queue.join_thread()
        if status_queue is not None:
            status_queue.close()
            status_queue.join_thread()
        if hasattr(self, "full_viewer_button"):
            text = "Open full viewer"
            if self.udp_socket is not None:
                self.full_viewer_button.configure(text=text)

    def _receive_packets(self) -> None:
        if self.udp_socket is None:
            return
        latest_packet = None
        latest_address = None
        for _ in range(256):
            try:
                latest_packet, latest_address = self.udp_socket.recvfrom(65535)
            except BlockingIOError:
                break
            except OSError as error:
                self.guidance_var.set(f"UDP receive error: {error}")
                self._disconnect()
                return

            self.udp_receive_count += 1
            self.last_datagram_time = time.perf_counter()

        if latest_packet is None or latest_address is None:
            return

        self.source_var.set(f"{latest_address[0]}:{latest_address[1]}")
        parsed, parse_error = parse_packet_with_error(latest_packet)
        self.packet_var.set(
            f"UDP: {self.udp_receive_count} | valid: {self.frame_number}"
        )
        if parsed is None:
            self.invalid_packets += 1
            self.connection_var.set("Data received (invalid)")
            self.guidance_var.set(
                f"Latest packet from {latest_address[0]}:{latest_address[1]} "
                f"({len(latest_packet)} bytes) was rejected: {parse_error}. "
                "Check the glove UDP output format and port."
            )
            return

        try:
            self._process_frame(parsed)
        except (ValueError, RuntimeError) as error:
            self.guidance_var.set(f"Frame processing failed: {error}")
            self.connection_var.set("Error")
            self._disconnect()
            messagebox.showerror("Glove data error", str(error), parent=self.root)

    def _update_glove_connection_status(self) -> None:
        if self.last_datagram_time is None:
            return
        elapsed = time.perf_counter() - self.last_datagram_time
        if elapsed > 2.0:
            self.connection_var.set("No recent data")
            self.guidance_var.set(
                f"No UDP packets for {elapsed:.0f}s. Check that the glove "
                f"sends to {', '.join(self.local_ipv4) or 'the PC Wi-Fi IP'}:"
                f"{self.port_var.get().strip()}, is on the same Wi-Fi, and "
                "is allowed through Windows Firewall."
            )

    def _process_frame(self, parsed: dict) -> None:
        if (
            self.data is None
            or self.mapper is None
            or self.quaternion_filter is None
            or self.muscle_controller is None
        ):
            raise RuntimeError("Sesi glove belum siap menerima data.")
        now = datetime.now().astimezone()
        timestamp = time.perf_counter()
        if self.last_packet_time is not None:
            interval = timestamp - self.last_packet_time
            if interval > 0:
                self.packet_intervals.append(interval)
        self.last_packet_time = timestamp
        self.frame_number += 1
        self.packet_var.set(
            f"UDP: {self.udp_receive_count} | valid: {self.frame_number} "
            f"({calculate_packet_rate(self.packet_intervals):.1f} Hz)"
        )
        self.connection_var.set("Glove connected")

        sensor_quaternions = capture_sensor_quaternions(parsed, "xyzw")
        if self.phase == "neutral":
            for finger, quaternion in sensor_quaternions.items():
                if finger in self.calibration_samples:
                    self.calibration_samples[finger].append(quaternion)
                elif finger in self.wrist_calibration_samples:
                    self.wrist_calibration_samples[finger].append(quaternion)
            sample_count = len(self.calibration_samples["THUMB"])
            self.qpos_var.set(f"Open pose {sample_count}/{CALIBRATION_PACKETS}")
            if sample_count >= CALIBRATION_PACKETS:
                sensor_neutral = {
                    finger: average_quaternions(samples)
                    for finger, samples in self.calibration_samples.items()
                }
                sensor_wrist_neutral = {
                    sensor: average_quaternions(samples)
                    for sensor, samples in self.wrist_calibration_samples.items()
                }
                if self.wrist_mapper is None:
                    raise RuntimeError("Mapper pergelangan belum siap.")
                self.mapper.calibrate(sensor_neutral)
                self.wrist_mapper.calibrate(
                    sensor_wrist_neutral["FOREARM"],
                    sensor_wrist_neutral["HAND"],
                )
                self.quaternion_filter.reset(
                    {**sensor_neutral, **sensor_wrist_neutral},
                    timestamp,
                )
                self.phase = "await_grip"
                self.qpos_var.set("Open pose ready")
                self.guidance_var.set(
                    "Pose terbuka tersimpan. Genggam tangan, tahan, "
                    "lalu tekan Calibrate grip."
                )
                self.grip_button.configure(state="normal")
            return

        if self.phase == "grip":
            for finger, quaternion in sensor_quaternions.items():
                if finger in self.grip_calibration_samples:
                    self.grip_calibration_samples[finger].append(quaternion)
            sample_count = len(self.grip_calibration_samples["THUMB"])
            self.qpos_var.set(f"Grip pose {sample_count}/{CALIBRATION_PACKETS}")
            if sample_count < CALIBRATION_PACKETS:
                return
            sensor_grip = {
                finger: average_quaternions(samples)
                for finger, samples in self.grip_calibration_samples.items()
            }
            try:
                skipped_fingers = self.mapper.calibrate_grip(sensor_grip)
            except ValueError as error:
                self.phase = "await_grip"
                self.qpos_var.set("Grip calibration failed")
                self.guidance_var.set(
                    f"{error} Atur ulang genggaman, lalu coba lagi."
                )
                self.grip_button.configure(state="normal")
                return
            self.quaternion_filter.reset(sensor_quaternions, timestamp)
            self.phase = "running"
            skipped = (
                f" IK dipakai untuk jari: {', '.join(skipped_fingers)}."
                if skipped_fingers
                else ""
            )
            self.qpos_var.set("Calibrated")
            self.guidance_var.set(
                "Kalibrasi selesai. Gerakan tangan akan mengontrol model."
                + skipped
            )
            self.record_button.configure(state="normal")
            self.fatigue_button.configure(state="normal")
            self.full_viewer_button.configure(state="normal")
            self._toggle_full_viewer()
            return

        if self.phase != "running":
            return

        filtered = self.quaternion_filter.update(sensor_quaternions, timestamp)
        finger_quaternions = {
            finger: filtered[finger] for finger in FINGER_JOINTS
        }
        if self.wrist_mapper is None:
            raise RuntimeError("Mapper pergelangan belum siap.")
        # Wrist lebih dulu agar IK jari memakai pose wrist frame yang sama.
        self.wrist_mapper.map_frame(filtered)
        wrist_text = " / ".join(
            f"{name} {angle:+.0f}{'*' if saturated else ''}"
            for name, angle, saturated in zip(
                ("PS", "Dev", "Flex"),
                self.wrist_mapper.last_angles_deg,
                self.wrist_mapper.saturated,
            )
        )
        if self.wrist_mapper.twist_axis is None:
            wrist_text += " | PS uncalibrated"
        elif self.wrist_mapper.twist_source == "estimated":
            wrist_text += " | PS estimated"
        if self.wrist_mapper.saturated.any():
            wrist_text += " | * ROM limit"
        self.wrist_var.set(wrist_text)
        qpos = self.mapper.map_frame(finger_quaternions)
        target_activations = self.muscle_controller.update(qpos)
        if self.fatigue_model is None:
            raise RuntimeError("Model fatigue belum siap.")
        self.data.ctrl[:] = self.fatigue_model.update(
            target_activations,
            timestamp,
        )
        self.latest_fds = {
            name: float(self.data.ctrl[self.fds_indices[name]])
            for name in FDS_NAMES
        }
        self._update_fds_labels()
        if self.csv_writer is not None:
            if self.record_file is None:
                raise RuntimeError("File rekaman FDS tidak tersedia.")
            try:
                self.csv_writer.writerow(
                    [
                        now.isoformat(timespec="milliseconds"),
                        self.frame_number,
                        self.subject_var.get().strip(),
                        *(self.latest_fds[name] for name in FDS_NAMES),
                    ]
                )
            except OSError as error:
                self._stop_recording()
                messagebox.showerror(
                    "Recording failed",
                    str(error),
                    parent=self.root,
                )

    def _flush_recording_if_due(self) -> None:
        if self.record_file is None:
            return
        now = time.perf_counter()
        if now - self.last_record_flush_time < 1.0:
            return
        try:
            self.record_file.flush()
        except OSError as error:
            self._stop_recording()
            messagebox.showerror(
                "Recording failed",
                str(error),
                parent=self.root,
            )
            return
        self.last_record_flush_time = now

    def _start_recording(self) -> None:
        if self.phase != "running" or self.csv_writer is not None:
            return
        try:
            subject = self.subject_var.get().strip()
            if not re.fullmatch(r"[\w.-]+", subject) or subject in {".", ".."}:
                raise ValueError(
                    "Subject ID hanya boleh berisi huruf, angka, titik, "
                    "garis bawah, atau tanda hubung."
                )
            RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.record_path = RECORDINGS_DIR / f"{subject}_FDS_{timestamp}.csv"
            self.record_file = self.record_path.open(
                "w",
                newline="",
                encoding="utf-8",
            )
            self.csv_writer = csv.writer(self.record_file)
            self.csv_writer.writerow(
                ["timestamp", "packet", "subject", *FDS_NAMES]
            )
            self.record_file.flush()
            self.last_record_flush_time = time.perf_counter()
        except (OSError, ValueError) as error:
            self.record_file = None
            self.csv_writer = None
            self.record_path = None
            messagebox.showerror("Recording failed", str(error), parent=self.root)
            return
        self.recording_var.set("Recording")
        self.guidance_var.set(f"Recording FDS values to {self.record_path}")
        self.record_button.configure(state="disabled")
        self.stop_record_button.configure(state="normal")

    def _stop_recording(self) -> None:
        if self.record_file is None:
            return
        self.record_file.close()
        self.record_file = None
        self.csv_writer = None
        self.recording_var.set("Not recording")
        self.guidance_var.set(f"Recording saved: {self.record_path}")
        self.stop_record_button.configure(state="disabled")
        if self.phase == "running":
            self.record_button.configure(state="normal")

    def _update_fds_labels(self) -> None:
        for name, value in self.latest_fds.items():
            self.fds_vars[name].set(f"{value:.3f}")

    def _close(self) -> None:
        self._stop_recording()
        self._release_resources()
        self.root.unbind_all("<MouseWheel>")
        self.root.destroy()


def main() -> None:
    mp.freeze_support()
    root = tk.Tk()
    SmartGloveGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
