"""Small Tk window: choose a shape, change its parameters (or load a glove JSON / card), preview, check, run."""
from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import mujoco
import numpy as np
from PIL import Image, ImageTk

from .card import load_card_mesh
from .pipeline import check, public
from .run import Spec, build_card, execute
from .scene import build_scene
from .shapes import DEFAULTS, SHAPES

FIELDS = {"box": [("x", "X (mm)"), ("y", "Y (mm)"), ("z", "Z / height (mm)")],
          "cylinder": [("radius", "radius (mm)"), ("height", "height (mm)")],
          "sphere": [("radius", "radius (mm)")]}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Robot hand: can it pick this up?")
        self.json_path = None
        self.card_path = None
        self.vars = {}
        f = ttk.Frame(self, padding=8)
        f.grid(row=0, column=0, sticky="nsew")
        r = 0
        ttk.Label(f, text="Shape").grid(row=r, column=0, sticky="w")
        self.shape = tk.StringVar(value="cylinder")
        cb = ttk.Combobox(f, textvariable=self.shape, values=list(SHAPES), state="readonly", width=12)
        cb.grid(row=r, column=1, sticky="w")
        cb.bind("<<ComboboxSelected>>", lambda e: self.fill_dims())
        r += 1
        self.dim_frame = ttk.Frame(f)
        self.dim_frame.grid(row=r, column=0, columnspan=3, sticky="w")
        self.dim_vars = {}
        r += 1
        for key, label, val in [("x", "object x (m)", "0.35"), ("y", "object y (m)", "-0.20"), ("yaw", "yaw (deg)", "0"),
                                ("mass", "mass (kg)", "0.35"), ("mu", "friction mu (pad)", "0.5"),
                                ("maxf", "max force (N)", "25")]:
            ttk.Label(f, text=label).grid(row=r, column=0, sticky="w")
            v = tk.StringVar(value=val)
            ttk.Entry(f, textvariable=v, width=10).grid(row=r, column=1, sticky="w")
            self.vars[key] = v
            r += 1
        self.hollow = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="hollow", variable=self.hollow).grid(row=r, column=0, sticky="w")
        self.use_fit = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="JSON: use fitted primitive (not the point hull)", variable=self.use_fit).grid(row=r, column=1, columnspan=2, sticky="w")
        r += 1
        ttk.Button(f, text="Load glove shape JSON...", command=self.pick_json).grid(row=r, column=0, sticky="w")
        ttk.Button(f, text="Load card JSON...", command=self.pick_card).grid(row=r, column=1, sticky="w")
        ttk.Button(f, text="Back to primitive", command=self.clear_src).grid(row=r, column=2, sticky="w")
        r += 1
        self.src = ttk.Label(f, text="source: primitive", foreground="#555")
        self.src.grid(row=r, column=0, columnspan=3, sticky="w")
        r += 1
        bf = ttk.Frame(f)
        bf.grid(row=r, column=0, columnspan=3, pady=6, sticky="w")
        ttk.Button(bf, text="Preview", command=self.preview).pack(side="left", padx=2)
        ttk.Button(bf, text="Check", command=lambda: self.go(check_only=True)).pack(side="left", padx=2)
        ttk.Button(bf, text="Run lift + video", command=lambda: self.go(check_only=False)).pack(side="left", padx=2)
        r += 1
        self.out = tk.Text(f, width=64, height=11, state="disabled", font=("Consolas", 9))
        self.out.grid(row=r, column=0, columnspan=3)
        self.img = ttk.Label(self)
        self.img.grid(row=0, column=1, padx=6)
        self.fill_dims()

    # ------------------------------------------------------------------
    def fill_dims(self):
        for w in self.dim_frame.winfo_children():
            w.destroy()
        self.dim_vars = {}
        for i, (k, label) in enumerate(FIELDS[self.shape.get()]):
            ttk.Label(self.dim_frame, text=label).grid(row=i, column=0, sticky="w")
            v = tk.StringVar(value=f"{DEFAULTS[self.shape.get()][k] * 1000:g}")
            ttk.Entry(self.dim_frame, textvariable=v, width=10).grid(row=i, column=1, padx=6)
            self.dim_vars[k] = v

    def pick_json(self):
        p = filedialog.askopenfilename(title="Glove shape JSON", filetypes=[("JSON", "*.json")])
        if p:
            self.json_path, self.card_path = p, None
            self.src.config(text=f"source: glove JSON  {os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}")

    def pick_card(self):
        p = filedialog.askopenfilename(title="Object card", filetypes=[("JSON", "*.json")])
        if p:
            self.card_path, self.json_path = p, None
            self.src.config(text=f"source: card  {os.path.basename(p)}")

    def clear_src(self):
        self.json_path = self.card_path = None
        self.src.config(text="source: primitive")

    def spec(self) -> Spec:
        f = lambda k: float(self.vars[k].get())
        dims = {k: float(v.get()) / 1000 for k, v in self.dim_vars.items()}
        return Spec(shape=self.shape.get(), dims=dims, json_path=self.json_path, use_fit=self.use_fit.get(),
                    card_path=self.card_path, pos=(f("x"), f("y")), yaw_deg=f("yaw"), mass_kg=f("mass"), mu_pad=f("mu"),
                    max_force_N=f("maxf"), hollow=self.hollow.get(), material="custom")

    def say(self, s):
        self.out.config(state="normal")
        self.out.insert("end", s + "\n")
        self.out.see("end")
        self.out.config(state="disabled")

    def preview(self):
        try:
            sp = self.spec()
            if self.json_path and not sp.pos:
                sp.pos = None
            card, shape, desc = build_card(sp)
            mesh = load_card_mesh(card)
            p = card.pos_m
            si = build_scene(mesh, [p[0], p[1], p[2]], [1, 0, 0, 0], card.mass_kg, card.friction or 0.5, card.hollow)
            r = mujoco.Renderer(si.model, 400, 520)
            cam = mujoco.MjvCamera()
            cam.distance, cam.azimuth, cam.elevation = 0.85, 118, -28
            cam.lookat[:] = [0.26, -0.14, 0.10]
            r.update_scene(si.data, cam)
            im = ImageTk.PhotoImage(Image.fromarray(r.render()))
            self.img.config(image=im)
            self.img.image = im
            e = (mesh.bounds[1] - mesh.bounds[0]) * 1000
            self.say(f"{desc}\nextent {e[0]:.0f} x {e[1]:.0f} x {e[2]:.0f} mm")
        except Exception as ex:
            messagebox.showerror("Preview", str(ex))

    def go(self, check_only):
        try:
            sp = self.spec()
        except ValueError:
            messagebox.showerror("Input", "Check the numbers.")
            return
        self.say("running..." if not check_only else "checking...")

        def work():
            try:
                res = execute(sp, check_only=check_only, log=lambda s: self.after(0, self.say, s))
                if not check_only and os.path.exists(os.path.join(res["output_dir"], "video.mp4")):
                    self.after(0, self.say, "video: " + os.path.join(res["output_dir"], "video.mp4"))
                    os.startfile(os.path.join(res["output_dir"], "video.mp4"))
            except Exception as ex:
                self.after(0, self.say, f"error: {ex}")
        threading.Thread(target=work, daemon=True).start()


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
