import os
import numpy as np
import sys
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from utils.quaternion import qmul_np, qmul, qrot
from torch.utils.data import Dataset, DataLoader
from utils.LaFan import LaFan1
import yaml

class Skeleton:
    def __init__(self, offsets, parents, joints_left=None, joints_right=None):
        assert len(offsets) == len(parents)

        self._offsets = torch.FloatTensor(offsets)
        self._parents = np.array(parents)
        self._joints_left = joints_left
        self._joints_right = joints_right
        self._compute_metadata()
    
    def cuda(self):
        self._offsets = self._offsets.cuda()
        return self

    def to(self, device):
        self._offsets = self._offsets.to(device)
        return self
    
    def num_joints(self):
        return self._offsets.shape[0]
    
    def offsets(self):
        return self._offsets
    
    def parents(self):
        return self._parents
    
    def has_children(self):
        return self._has_children
    
    def children(self):
        return self._children
    
    def remove_joints(self, joints_to_remove):
        """
        Remove the joints specified in 'joints_to_remove', both from the
        skeleton definition and from the dataset (which is modified in place).
        The rotations of removed joints are propagated along the kinematic chain.
        """
        valid_joints = []
        for joint in range(len(self._parents)):
            if joint not in joints_to_remove:
                valid_joints.append(joint)

        index_offsets = np.zeros(len(self._parents), dtype=int)
        new_parents = []
        for i, parent in enumerate(self._parents):
            if i not in joints_to_remove:
                new_parents.append(parent - index_offsets[parent])
            else:
                index_offsets[i:] += 1
        self._parents = np.array(new_parents)

        self._offsets = self._offsets[valid_joints]
        self._compute_metadata()
        
    def forward_kinematics(self, rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where N = batch size, L = sequence length, J = number of joints):
         -- rotations: (N, L, J, 4) tensor of unit quaternions describing the local rotations of each joint.
         -- root_positions: (N, L, 3) tensor describing the root joint positions.
        """
        assert len(rotations.shape) == 4
        assert rotations.shape[-1] == 4

        positions_world = []
        rotations_world = []

        expanded_offsets = self._offsets.expand(rotations.shape[0], rotations.shape[1],
                                                   self._offsets.shape[0], self._offsets.shape[1]).detach().cpu()

        # Parallelize along the batch and time dimensions
        for i in range(self._offsets.shape[0]):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(rotations[:, :, 0])
            else:
                positions_world.append(qrot(rotations_world[self._parents[i]], expanded_offsets[:, :, i]) \
                                       + positions_world[self._parents[i]])
                if self._has_children[i]:
                    rotations_world.append(qmul(rotations_world[self._parents[i]], rotations[:, :, i]))
                else:
                    # This joint is a terminal node -> it would be useless to compute the transformation
                    rotations_world.append(None)


        return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)
    
    def joints_left(self):
        return self._joints_left
    
    def joints_right(self):
        return self._joints_right
        
    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._children = []
        for i, parent in enumerate(self._parents):
            self._children.append([])
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i)

    def get_bone_length_weight(self):
        bone_length = []
        for i, parent in enumerate(self._parents):
            if parent == -1:
                bone_length.append(1)
            else:
                bone_length.append(
                    torch.linalg.norm(self._offsets[i : i + 1], ord="fro").item()
                )
        return torch.Tensor(bone_length)

    def forward_kinematics_with_rotation(self, rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where N = batch size, L = sequence length, J = number of joints):
         -- rotations: (N, L, J, 4) tensor of unit quaternions describing the local rotations of each joint.
         -- root_positions: (N, L, 3) tensor describing the root joint positions.
        """
        assert len(rotations.shape) == 4
        assert rotations.shape[-1] == 4

        positions_world = []
        rotations_world = []

        expanded_offsets = self._offsets.expand(
            rotations.shape[0],
            rotations.shape[1],
            self._offsets.shape[0],
            self._offsets.shape[1],
        )

        # Parallelize along the batch and time dimensions
        for i in range(self._offsets.shape[0]):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(rotations[:, :, 0])
            else:
                positions_world.append(
                    qrot(rotations_world[self._parents[i]], expanded_offsets[:, :, i])
                    + positions_world[self._parents[i]]
                )
                if self._has_children[i]:
                    rotations_world.append(
                        qmul(rotations_world[self._parents[i]], rotations[:, :, i])
                    )
                else:
                    # This joint is a terminal node -> it would be useless to compute the transformation
                    rotations_world.append(
                        torch.Tensor([1, 0, 0, 0])
                        .expand(rotations.shape[0], rotations.shape[1], 4)
                        .to(rotations.device)
                    )

        return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2), torch.stack(
            rotations_world, dim=3
        ).permute(0, 1, 3, 2)


    def convert_to_global_pos(self, unit_vec_rerp):
        """
        Convert the unit offset matrix to global position.
        First row(root) will have absolute position value in global coordinates.
        """
        bone_length = self.get_bone_length_weight()
        batch_size = unit_vec_rerp.size(0)
        seq_len = unit_vec_rerp.size(1)
        unit_vec_table = unit_vec_rerp.reshape(batch_size, seq_len, 22, 3)
        global_position = torch.zeros_like(unit_vec_table, device=unit_vec_table.device)

        for i, parent in enumerate(self._parents):
            if parent == -1:  # if root
                global_position[:, :, i] = unit_vec_table[:, :, i]

            else:
                global_position[:, :, i] = global_position[:, :, parent] + (
                    nn.functional.normalize(unit_vec_table[:, :, i], p=2.0, dim=-1)
                    * bone_length[i]
                )

        return global_position

    def convert_to_unit_offset_mat(self, global_position):
        """
        Convert the global position of the skeleton to a unit offset matrix.
        First row(root) will have absolute position value in global coordinates.
        """

        bone_length = self.get_bone_length_weight()
        unit_offset_mat = torch.zeros_like(
            global_position, device=global_position.device
        )

        for i, parent in enumerate(self._parents):

            if parent == -1:  # if root
                unit_offset_mat[:, :, i] = global_position[:, :, i]
            else:
                unit_offset_mat[:, :, i] = (
                    global_position[:, :, i] - global_position[:, :, parent]
                ) / bone_length[i]

        return unit_offset_mat