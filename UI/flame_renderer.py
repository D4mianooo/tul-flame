from typing import Optional

import pyrender
import numpy as np
import trimesh


class FLAME_Renderer:
    def __init__(self):
        self.transform_matrix = np.eye(4)
        self.scene = pyrender.Scene()
        self.viewer = None
        self.mesh_node = None
        self.joints_node = None
        self.joints_mesh = trimesh.creation.uv_sphere(radius=0.005)
        self.joints_mesh.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]

    def update(self, vertices, joints, faces, render_joint  : Optional[bool] = True):
        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]

        tri_mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors)

        mesh = pyrender.Mesh.from_trimesh(tri_mesh)
        self.scene.add(mesh, pose=self.transform_matrix)

        tfs = np.tile(np.eye(4), (len(joints), 1, 1))
        tfs[:, :3, 3] = joints
        joints_pcl = pyrender.Mesh.from_trimesh(self.joints_mesh, poses=tfs)

        if  render_joint:
            self.scene.add(joints_pcl, pose=self.transform_matrix)

    def start_view(self, vertices, joints, faces, render_joint  : Optional[bool] = True):
        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
        initial_mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors))

        sm = trimesh.creation.uv_sphere(radius=0.005)
        sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]

        tfs = np.tile(np.eye(4), (len(joints), 1, 1))
        tfs[:, :3, 3] = joints

        joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)

        if  render_joint:
            self.joints_node = self.scene.add(joints_pcl, pose=self.transform_matrix)

        self.mesh_node = self.scene.add(initial_mesh, pose=self.transform_matrix)

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        camera_pose = np.eye(4)
        camera_pose[2, 3] = 0.5
        camera_node = self.scene.add(camera, pose=camera_pose)

        self.viewer = pyrender.Viewer(self.scene, use_raymond_lighting=True, run_in_thread=True, camera_node=camera_node,
        viewer_flags={
            "rotate": False,
            "pan": False,
            "zoom": False
        })

    def update_mesh_runtime(self, vertices, joints, faces, render_joint  : Optional[bool] = True):
        if self.viewer is None or not self.viewer.is_active:
            return

        self.viewer.render_lock.acquire()

        new_trimesh = trimesh.Trimesh(vertices,faces)
        new_trimesh.visual.vertex_colors = [0.3, 0.3, 0.3, 0.8]
        new_mesh = pyrender.Mesh.from_trimesh(new_trimesh)

        self.scene.remove_node(self.mesh_node)
        self.mesh_node = self.scene.add(new_mesh)

        if  not render_joint:
            if self.joints_node is not None:
                self.scene.remove_node(self.joints_node)
            self.joints_node = None
        else:
            tfs = np.tile(np.eye(4), (len(joints), 1, 1))
            tfs[:, :3, 3] = joints

            joints_pcl = pyrender.Mesh.from_trimesh(self.joints_mesh, poses=tfs)

            if self.joints_node is not None:
                self.scene.remove_node(self.joints_node)
            self.joints_node = self.scene.add(joints_pcl, pose=self.transform_matrix)

        self.viewer.render_lock.release()