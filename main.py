import tkinter as tk
import torch

from UI.flame_controller import Flame_Controller
from UI.flame_renderer import FLAME_Renderer
from models.flame_pytorch.flame_wrapper import FLAME_Wrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA is not available")

flame_renderer = FLAME_Renderer()
flame_wrapper = FLAME_Wrapper(device)

if __name__ == "__main__":
    root = tk.Tk()
    app = Flame_Controller(root, flame_wrapper, flame_renderer)
    root.mainloop()