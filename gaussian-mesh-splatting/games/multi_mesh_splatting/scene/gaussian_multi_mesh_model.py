#
# Copyright (C) 2024, Gmum
# Group of Machine Learning Research. https://gmum.net/
# All rights reserved.
#
# The Gaussian-splatting software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr
#
# The Gaussian-mesh-splatting is software based on Gaussian-splatting, used on research.
# This Games software is free for non-commercial, research and evaluation use
#

import torch
import numpy as np

from torch import nn

from scene.gaussian_model import GaussianModel
from utils.general_utils import inverse_sigmoid, rot_to_quat_batch
from utils.sh_utils import RGB2SH
from games.multi_mesh_splatting.utils.graphics_utils import MultiMeshPointCloud


class GaussianMultiMeshModel(GaussianModel):

    def __init__(self, sh_degree: int):

        super().__init__(sh_degree)
        self.point_cloud = None
        self._alpha = torch.empty(0)
        self._scale = torch.empty(0)
        self.alpha = torch.empty(0)
        self.softmax = torch.nn.Softmax(dim=2)
        self.vertices = torch.empty(0)
        self.faces = torch.empty(0)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.update_alpha_func = self.softmax

    @property
    def get_xyz(self):
        return self._xyz

    def create_from_pcd(self, pcd: MultiMeshPointCloud, spatial_lr_scale: float):

        self.point_cloud = pcd
        self.spatial_lr_scale = spatial_lr_scale
        self._alpha = []
        self._features_dc = []
        self._features_rest = []
        self._scale = []
        self._opacity = []
        max_radii2D = 0
        for p in pcd:
            pcd_alpha_shape = p.alpha.shape

            print("Number of faces: ", pcd_alpha_shape[0])
            print("Number of points at initialisation in face: ", pcd_alpha_shape[1])

            alpha_point_cloud = p.alpha.float().cuda()
            scale = torch.ones((p.points.shape[0], 1)).float().cuda()

            print("Number of points at initialisation : ",
                alpha_point_cloud.shape[0] * alpha_point_cloud.shape[1])

            fused_color = RGB2SH(torch.tensor(np.asarray(p.colors)).float().cuda())
            features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0] = fused_color
            features[:, 3:, 1:] = 0.0

            opacities = inverse_sigmoid(0.1 * torch.ones((p.points.shape[0], 1), dtype=torch.float, device="cuda"))

            self._alpha.append(nn.Parameter(alpha_point_cloud.requires_grad_(True)))  # check update_alpha
            self._features_dc.append(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest.append(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
            self._scale.append(nn.Parameter(scale.requires_grad_(True)))
            self._opacity.append(opacities.requires_grad_(True))
            max_radii2D += p.points.shape[0]
            
        self._opacity = nn.Parameter(torch.cat(self._opacity))
        self._features_dc = nn.Parameter(torch.cat(self._features_dc))
        self._features_rest = nn.Parameter(torch.cat(self._features_rest))
        self.max_radii2D = torch.zeros((max_radii2D), device="cuda")
        self.verts_faces()
        self.update_alpha()
        self.prepare_scaling_rot()

    def verts_faces(self):
        self.vertices = []
        self.faces = []
        for p in self.point_cloud:
            vertices = torch.tensor(p.vertices, device="cuda").float()
            self.faces.append(torch.tensor(p.faces).long())
            self.vertices.append(nn.Parameter(vertices.requires_grad_(True)))

    def _calc_xyz(self):
        """
        calculate the 3d Gaussian center in the coordinates xyz.

        The alphas that are taken into account are the distances
        to the vertices and the coordinates of
        the triangles forming the mesh.

        """
        self._xyz = []
        for i, (alpha, vertices, faces) in enumerate(zip(self.alpha, self.vertices, self.faces)):
            _xyz = torch.matmul(
                alpha,
                vertices[faces]
            )
            self._xyz.append(
                _xyz.reshape(
                    _xyz.shape[0] * _xyz.shape[1], 3
                )
            )
        self._xyz = torch.cat(self._xyz)
        
    def prepare_scaling_rot(self, eps=1e-8):
        """
        approximate covariance matrix and calculate scaling/rotation tensors

        covariance matrix is [v0, v1, v2], where
        v0 is a normal vector to each face
        v1 is a vector from centroid of each face and 1st vertex
        v2 is obtained by orthogonal projection of a vector from centroid to 2nd vertex onto subspace spanned by v0 and v1
        """
        def dot(v, u):
            return (v * u).sum(dim=-1, keepdim=True)
        
        def proj(v, u):
            """
            projection of vector v onto subspace spanned by u

            vector u is assumed to be already normalized
            """
            coef = dot(v, u)
            return coef * u

        self._scaling = []
        self._rotation = []
        for i, (vertices, faces) in enumerate(zip(self.vertices, self.faces)):
            triangles = vertices[faces]
            normals = torch.linalg.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
                dim=1
            )
            v0 = normals / (torch.linalg.vector_norm(normals, dim=-1, keepdim=True) + eps)
            means = torch.mean(triangles, dim=1)
            v1 = triangles[:, 1] - means
            v1_norm = torch.linalg.vector_norm(v1, dim=-1, keepdim=True) + eps
            v1 = v1 / v1_norm
            v2_init = triangles[:, 2] - means
            v2 = v2_init - proj(v2_init, v0) - proj(v2_init, v1) # Gram-Schmidt
            v2 = v2 / (torch.linalg.vector_norm(v2, dim=-1, keepdim=True) + eps)

            s1 = v1_norm / 2.
            s2 = dot(v2_init, v2) / 2.
            s0 = eps * torch.ones_like(s1)
            scales = torch.concat((s0, s1, s2), dim=1).unsqueeze(dim=1)
            scales = scales.broadcast_to((*self.alpha[i].shape[:2], 3))
            self._scaling.append(torch.log(
                torch.nn.functional.relu(self._scale[i] * scales.flatten(start_dim=0, end_dim=1)) + eps
            ))
            rotation = torch.stack((v0, v1, v2), dim=1).unsqueeze(dim=1)
            rotation = rotation.broadcast_to((*self.alpha[i].shape[:2], 3, 3)).flatten(start_dim=0, end_dim=1)
            rotation = rotation.transpose(-2, -1)
            self._rotation.append(rot_to_quat_batch(rotation))

        self._scaling = torch.cat(self._scaling)
        self._rotation = torch.cat(self._rotation)

    def update_alpha(self):
        """
        Function to control the alpha value.

        Alpha is the distance of the center of the gauss
         from the vertex of the triangle of the mesh.
        Thus, for each center of the gauss, 3 alphas
        are determined: alpha1+ alpha2+ alpha3.
        For a point to be in the center of the vertex,
        the alphas must meet the assumptions:
        alpha1 + alpha2 + alpha3 = 1
        and alpha1 + alpha2 +alpha3 >= 0

        #TODO
        check:
        # self.alpha = torch.relu(self._alpha)
        # self.alpha = self.alpha / self.alpha.sum(dim=-1, keepdim=True)

        """
        self.alpha = []
        for alpha in self._alpha:
            alpha = torch.relu(alpha) + 1e-8
            self.alpha.append(alpha / alpha.sum(dim=-1, keepdim=True))
        self._calc_xyz()

    def training_setup(self, training_args):
        s = 0
        for xyz in self.get_xyz:
            s += xyz.shape[0]
        self.denom = torch.zeros((s, 1), device="cuda")

        l = [
            {'params': self._alpha, 'lr': training_args.alpha_lr, "name": "alpha"},
            {'params': self.vertices, 'lr': training_args.vertices_lr, "name": "vertices"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': self._scale, 'lr': training_args.scaling_lr, "name": "scaling"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        pass

    def save_ply(self, path):
        self._save_ply(path)

        attrs = self.__dict__
        additional_attrs = [
            '_alpha', 
            '_scale',
            'point_cloud',
            'vertices',
            'faces'
        ]

        save_dict = {}
        for attr_name in additional_attrs:
            save_dict[attr_name] = []
            for m in attrs[attr_name]:
                save_dict[attr_name].append(m)

        path_model = path.replace('point_cloud.ply', 'model_params.pt')
        torch.save(save_dict, path_model)

    def load_ply(self, path):
        self._load_ply(path)
        path_model = path.replace('point_cloud.ply', 'model_params.pt')
        params = torch.load(path_model)
        alpha = params['_alpha']
        scale = params['_scale']
        vertices = params['vertices']
        faces = params['faces']
        self._alpha = [nn.Parameter(a) for a in alpha]
        self._scale = [nn.Parameter(s) for s in scale]
        self.vertices = [nn.Parameter(v) for v in vertices]
        self.faces = faces

