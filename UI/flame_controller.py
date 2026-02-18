import tkinter as tk
from tkinter import ttk
import torch
import numpy as np


class Flame_Controller:
    def __init__(self, root, flame_wrapper, flame_renderer):
        self.root = root
        self.root.title("FLAME Model Studio")
        self.root.geometry("450x800")

        # Style
        self.style = ttk.Style()
        self.root.configure(bg="#2b2b2b")
        self.style.theme_use("clam")
        self.configure_styles()

        self.flame_wrapper = flame_wrapper
        self.flame_renderer = flame_renderer

        # Parameters
        self.pose_params = np.zeros(6)
        self.shape_params = np.zeros(10)
        self.exp_params = np.zeros(10)
        self.render_joints = tk.BooleanVar(value=True)

        # Slider storage (IMPORTANT)
        self.shape_sliders = []
        self.exp_sliders = []
        self.pose_sliders = []

        self.setup_ui()

        vertices, joints, faces = self.flame_wrapper.generate_mesh()
        self.flame_renderer.start_view(vertices, joints, faces)

    # --------------------------------------------------
    # Styling
    # --------------------------------------------------
    def configure_styles(self):
        self.style.configure("TFrame", background="#2b2b2b")
        self.style.configure(
            "TLabelframe",
            background="#2b2b2b",
            foreground="#ffffff",
            bordercolor="#444444"
        )
        self.style.configure(
            "TLabelframe.Label",
            background="#2b2b2b",
            foreground="#3498db",
            font=("Segoe UI", 10, "bold")
        )
        self.style.configure(
            "TLabel",
            background="#2b2b2b",
            foreground="#cccccc",
            font=("Segoe UI", 9)
        )
        self.style.configure(
            "TCheckbutton",
            background="#2b2b2b",
            foreground="#ffffff"
        )

        self.style.configure(
            "Reset.TButton",
            background="#e74c3c",
            foreground="white",
            font=("Segoe UI", 9, "bold"),
            padding=6
        )
        self.style.map(
            "Reset.TButton",
            background=[("active", "#c0392b")]
        )

    # --------------------------------------------------
    # UI (Scrollable)
    # --------------------------------------------------
    def setup_ui(self):
        canvas = tk.Canvas(self.root, bg="#2b2b2b", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        main = ttk.Frame(canvas, padding=20)
        canvas.create_window((0, 0), window=main, anchor="nw")

        main.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.bind_all(
            "<MouseWheel>",
            lambda e: canvas.yview_scroll(-1 * int(e.delta / 120), "units")
        )

        ttk.Label(
            main,
            text="FLAME CONTROLLER",
            font=("Segoe UI", 16, "bold"),
            foreground="#3498db"
        ).pack(pady=(0, 20))

        # ---------------- Shape ----------------
        shape_frame = ttk.LabelFrame(main, text=" SHAPE (BETA) ", padding=10)
        shape_frame.pack(fill="x", pady=5)

        for i in range(5):
            s = self.create_slider(shape_frame, f"Shape {i}", -3, 3, i, "shape")
            self.shape_sliders.append(s)

        self.reset_button(shape_frame, self.reset_shape)

        # ---------------- Expression ----------------
        exp_frame = ttk.LabelFrame(main, text=" EXPRESSION (PSI) ", padding=10)
        exp_frame.pack(fill="x", pady=5)

        for i in range(5):
            s = self.create_slider(exp_frame, f"Expression {i}", -2, 2, i, "exp")
            self.exp_sliders.append(s)

        self.reset_button(exp_frame, self.reset_expression)

        # ---------------- Pose ----------------
        pose_frame = ttk.LabelFrame(main, text=" POSE ", padding=10)
        pose_frame.pack(fill="x", pady=5)

        self.reset_button(pose_frame, self.reset_pose)

        rot_frame = ttk.LabelFrame(pose_frame, text=" GLOBAL ROTATION ", padding=10)
        rot_frame.pack(fill="x", pady=5)

        yaw_frame = ttk.LabelFrame(pose_frame, text=" YAW ", padding=10)
        yaw_frame.pack(fill="x", pady=5)

        for i in range(3):
            s = self.create_slider(rot_frame, f"Rotation {i}", -180, 180, i, "pose")
            self.pose_sliders.append(s)

        for i in range(3, 5):
            s = self.create_slider(yaw_frame, f"Pose {i}", -2, 2, i, "pose")
            self.pose_sliders.append(s)

        # ---------------- Settings ----------------
        settings_frame = ttk.LabelFrame(main, text=" RENDER SETTINGS ", padding=10)
        settings_frame.pack(fill="x", pady=15)

        ttk.Checkbutton(
            settings_frame,
            text="Show Landmark Points",
            variable=self.render_joints,
            command=self.update_renderer
        ).pack(side="left", padx=5)

        # ---------------- Global Reset ----------------
        ttk.Button(
            main,
            text="Reset to Neutral",
            style="Reset.TButton",
            command=self.reset_all
        ).pack(fill="x", pady=20)

    # --------------------------------------------------
    # Widgets
    # --------------------------------------------------
    def reset_button(self, parent, command):
        ttk.Button(
            parent,
            text="Reset to Neutral",
            style="Reset.TButton",
            command=command
        ).pack(fill="x", pady=10)

    def create_slider(self, parent, label, mn, mx, index, ptype):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=2)

        ttk.Label(frame, text=label, width=12).pack(side="left")

        val_label = ttk.Label(frame, text="0.00", width=6, foreground="#3498db")
        val_label.pack(side="right")

        slider = ttk.Scale(
            frame,
            from_=mn,
            to=mx,
            orient="horizontal",
            command=lambda v, i=index, t=ptype, l=val_label:
                self.on_slider(v, i, t, l)
        )
        slider.pack(side="right", fill="x", expand=True, padx=5)
        slider.set(0)

        return slider

    # --------------------------------------------------
    # Logic
    # --------------------------------------------------
    def on_slider(self, value, index, ptype, label):
        val = float(value)
        label.config(text=f"{val:.2f}")

        if ptype == "shape":
            self.shape_params[index] = val
        elif ptype == "pose":
            self.pose_params[index] = val
        else:
            self.exp_params[index] = val

        self.update_renderer()

    # --------------------------------------------------
    # Reset
    # --------------------------------------------------
    def reset_shape(self):
        self.shape_params.fill(0)
        for s in self.shape_sliders:
            s.set(0)
        self.update_renderer()

    def reset_expression(self):
        self.exp_params.fill(0)
        for s in self.exp_sliders:
            s.set(0)
        self.update_renderer()

    def reset_pose(self):
        self.pose_params.fill(0)
        for s in self.pose_sliders:
            s.set(0)
        self.update_renderer()

    def reset_all(self):
        self.shape_params.fill(0)
        self.exp_params.fill(0)
        self.pose_params.fill(0)

        for s in (*self.shape_sliders, *self.exp_sliders, *self.pose_sliders):
            s.set(0)

        self.update_renderer()

    # --------------------------------------------------
    # Renderer
    # --------------------------------------------------
    def update_renderer(self):
        self.flame_wrapper.shape[0, :10] = torch.tensor(
            self.shape_params, device=self.flame_wrapper.device)

        self.flame_wrapper.expression[0, :10] = torch.tensor(
            self.exp_params, device=self.flame_wrapper.device)

        for i in range(self.pose_params.size):
            self.flame_wrapper.update_pose(i, self.pose_params[i], False)

        vertices, joints, faces = self.flame_wrapper.generate_mesh()
        self.flame_renderer.update_mesh_runtime(
            vertices, joints, faces, self.render_joints.get())
