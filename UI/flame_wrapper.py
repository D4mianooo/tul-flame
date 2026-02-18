import torch
import numpy as np
from models.flame_pytorch import FLAME, get_config

class FLAME_Wrapper:
    def __init__(self, device=None):
        self.config = get_config()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.flamelayer = FLAME(self.config).to(self.device)

        self.batch_size = 1

        self.shape = torch.zeros(self.batch_size, 100).to(self.device)
        self.expression = torch.zeros(self.batch_size, 50).to(self.device)
        self.pose = torch.zeros(self.batch_size, 6).to(self.device)

        self.neck_pose = torch.zeros(self.batch_size, 3).to(self.device)
        self.eye_pose = torch.zeros(self.batch_size, 6).to(self.device)

    def update_shape(self, index, value):
        """Update a specific shape coefficient."""
        self.shape[0, index] = value

    def update_expression(self, index, value):
        """Update a specific expression coefficient."""
        self.expression[0, index] = value

    def update_pose(self, index, value, is_radian=True):
        """Update pose (Rotation). Index 0-2: Global, 3-5: Jaw."""
        val = value if is_radian else value * (np.pi / 180.0)
        self.pose[0, index] = val

    def generate_mesh(self):
        with torch.no_grad():
            vertices, landmarks = self.flamelayer(
                self.shape,
                self.expression,
                self.pose,
                self.neck_pose,
                self.eye_pose
            )

        return vertices[0].detach().cpu().numpy().squeeze(), landmarks[0].detach().cpu().numpy().squeeze(), self.flamelayer.faces