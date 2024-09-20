###########################################################################################
# Utilities
# Authors: Ilyes Batatia, Gregor Simm and David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging
# from typing import List, Optional, Tuple

from typing import Any, Tuple, Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch
import torch.nn
import torch.utils.data
from scipy.constants import c, e

from mace.tools import to_numpy
from mace.tools.scatter import scatter_sum
from mace.tools.torch_geometric.batch import Batch

from .blocks import AtomicEnergiesBlock
from .radial_basis import RadialBasis


def compute_forces(
    energy: torch.Tensor, positions: torch.Tensor, training: bool = True
) -> torch.Tensor:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
    gradient = torch.autograd.grad(
        outputs=[energy],  # [n_graphs, ]
        inputs=[positions],  # [n_nodes, 3]
        grad_outputs=grad_outputs,
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        allow_unused=True,  # For complete dissociation turn to true
    )[
        0
    ]  # [n_nodes, 3]
    if gradient is None:
        return torch.zeros_like(positions)
    return -1 * gradient


def compute_forces_virials(
    energy: torch.Tensor,
    positions: torch.Tensor,
    displacement: torch.Tensor,
    cell: torch.Tensor,
    training: bool = True,
    compute_stress: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
    forces, virials = torch.autograd.grad(
        outputs=[energy],  # [n_graphs, ]
        inputs=[positions, displacement],  # [n_nodes, 3]
        grad_outputs=grad_outputs,
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        allow_unused=True,
    )
    stress = torch.zeros_like(displacement)
    if compute_stress and virials is not None:
        cell = cell.view(-1, 3, 3)
        volume = torch.einsum(
            "zi,zi->z",
            cell[:, 0, :],
            torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1),
        ).unsqueeze(-1)
        stress = virials / volume.view(-1, 1, 1)
    if forces is None:
        forces = torch.zeros_like(positions)
    if virials is None:
        virials = torch.zeros((1, 3, 3))

    return -1 * forces, -1 * virials, stress


def get_symmetric_displacement(
    positions: torch.Tensor,
    unit_shifts: torch.Tensor,
    cell: Optional[torch.Tensor],
    edge_index: torch.Tensor,
    num_graphs: int,
    batch: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cell is None:
        cell = torch.zeros(
            num_graphs * 3,
            3,
            dtype=positions.dtype,
            device=positions.device,
        )
    sender = edge_index[0]
    displacement = torch.zeros(
        (num_graphs, 3, 3),
        dtype=positions.dtype,
        device=positions.device,
    )
    displacement.requires_grad_(True)
    symmetric_displacement = 0.5 * (
        displacement + displacement.transpose(-1, -2)
    )  # From https://github.com/mir-group/nequip
    positions = positions + torch.einsum(
        "be,bec->bc", positions, symmetric_displacement[batch]
    )
    cell = cell.view(-1, 3, 3)
    cell = cell + torch.matmul(cell, symmetric_displacement)
    shifts = torch.einsum(
        "be,bec->bc",
        unit_shifts,
        cell[batch[sender]],
    )
    return positions, shifts, displacement


def get_outputs(
    energy: torch.Tensor,
    positions: torch.Tensor,
    displacement: Optional[torch.Tensor],
    cell: torch.Tensor,
    training: bool = False,
    compute_force: bool = True,
    compute_virials: bool = True,
    compute_stress: bool = True,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if (compute_virials or compute_stress) and displacement is not None:
        # forces come for free
        forces, virials, stress = compute_forces_virials(
            energy=energy,
            positions=positions,
            displacement=displacement,
            cell=cell,
            compute_stress=compute_stress,
            training=training,
        )
    elif compute_force:
        forces, virials, stress = (
            compute_forces(energy=energy, positions=positions, training=training),
            None,
            None,
        )
    else:
        forces, virials, stress = (None, None, None)
    return forces, virials, stress


def get_edge_vectors_and_lengths(
    positions: torch.Tensor,  # [n_nodes, 3]
    edge_index: torch.Tensor,  # [2, n_edges]
    shifts: torch.Tensor,  # [n_edges, 3]
    normalize: bool = False,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sender = edge_index[0]
    receiver = edge_index[1]
    vectors = positions[receiver] - positions[sender] + shifts  # [n_edges, 3]
    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
    if normalize:
        vectors_normed = vectors / (lengths + eps)
        return vectors_normed, lengths

    return vectors, lengths


def _check_non_zero(std):
    if std == 0.0:
        logging.warning(
            "Standard deviation of the scaling is zero, Changing to no scaling"
        )
        std = 1.0
    return std


def extract_invariant(x: torch.Tensor, num_layers: int, num_features: int, l_max: int):
    out = []
    for i in range(num_layers - 1):
        out.append(
            x[
                :,
                i
                * (l_max + 1) ** 2
                * num_features : (i * (l_max + 1) ** 2 + 1)
                * num_features,
            ]
        )
    out.append(x[:, -num_features:])
    return torch.cat(out, dim=-1)

def pos_svd_frame(data: Dict[str, torch.Tensor]):
    # print("pos_svd_data:", type(data), data["batch"])
    pos = data["positions"]
    batch = data["batch"]
    batch_size = int(batch.max()) + 1

    with torch.cuda.amp.autocast(False):
        rotated_pos_list = []
        for i in range(batch_size):
            # Center each structure around mean position
            pos_batch = pos[batch == i]
            pos_batch = pos_batch - pos_batch.mean(0)

            # Rotate each structure into its SVD frame
            # (only can do this if structure has at least 3 atoms,
            # i.e., the position matrix has full rank)
            if pos_batch.shape[0] > 2:
                U, S, V = torch.svd(pos_batch)
                rotated_pos_batch = torch.matmul(pos_batch, V)

            else:
                rotated_pos_batch = pos_batch

            rotated_pos_list.append(rotated_pos_batch)

        pos = torch.cat(rotated_pos_list)

    return pos

def x_to_k_cell(cells: torch.Tensor, batch_size: int):
    # print("cells:", cells.shape)
    cells = cells.reshape(batch_size, 3, 3)
    cross_a2a3 = torch.cross(cells[:, :, 1], cells[:, :, 2], dim=-1)
    # print("cross_a2a3:", cross_a2a3.shape)
    cross_a3a1 = torch.cross(cells[:, :, 2], cells[:, :, 0], dim=-1)
    # print("cross_a3a1:", cross_a3a1.shape)
    cross_a1a2 = torch.cross(cells[:, :, 0], cells[:, :, 1], dim=-1)
    # print("cross_a1a2:", cross_a1a2.shape)

    vol = torch.sum(cells[:, :, 0] * cross_a2a3, dim=-1, keepdim=True)

    b1 = 2 * np.pi * cross_a2a3 / vol
    b2 = 2 * np.pi * cross_a3a1 / vol
    b3 = 2 * np.pi * cross_a1a2 / vol

    bcells = torch.stack((b1, b2, b3), dim=1)
    # bcells = bcells.type(torch.float32)

    # print("bcells:", bcells.shape, bcells.dtype, "vol:", vol.shape)

    return (bcells, vol[:, 0]) #check: changed the shape of vol. original: vol[:, 0]

def get_k_index_product_set(num_k_x: float, num_k_y: float, num_k_z: float):
    # Get a box of k-lattice indices around (0,0,0)
    k_index_sets = (
        torch.arange(-num_k_x, num_k_x + 1),#, dtype=torch.float64),
        torch.arange(-num_k_y, num_k_y + 1),#, dtype=torch.float64),
        torch.arange(-num_k_z, num_k_z + 1),# dtype=torch.float64),
    )
    k_index_product_set = torch.cartesian_prod(*k_index_sets)
    # Cut the box in half (we will always assume point symmetry)
    k_index_product_set = k_index_product_set[
        k_index_product_set.shape[0] // 2 + 1 :
    ]
    print("k_index_product_set from utils:", k_index_product_set.dtype)
    # Amount of k-points
    num_k_degrees_of_freedom = k_index_product_set.shape[0]
    
    return k_index_product_set, num_k_degrees_of_freedom

def get_k_voxel_grid(k_cutoff: float, delta_k: float, num_k_rbf: int):

    # Get indices for a cube of k-lattice sites containing the cutoff sphere
    num_k = k_cutoff / delta_k
    k_index_product_set, _ = get_k_index_product_set(num_k, num_k, num_k)
    # print("k_index_product_set shape if NOT periodic:", k_index_product_set.shape, k_index_product_set.dtype)

    # Orthogonal k-space basis, norm delta_k
    k_cell = torch.tensor(
        [[delta_k, 0, 0], [0, delta_k, 0], [0, 0, delta_k]])#, dtype=torch.float64)
    # print("k_cell shape if NOT periodic:", k_cell.shape, k_cell.dtype)

    # Translate lattice indices into k-vectors
    k_grid = torch.matmul(k_index_product_set, k_cell)
    # print("k_grid shape if NOT periodic:", k_grid.shape)

    # Prune all k-vectors outside the cutoff sphere
    k_grid = k_grid[torch.sum(k_grid**2, dim=-1) <= k_cutoff**2]

    # Probably quite arbitrary, for backwards compatibility with scaling
    # yaml files produced with old Ewald Message Passing code
    k_offset = 0.1 if num_k_rbf <= 48 else 0.25

    # Evaluate a basis of Gaussian RBF on the k-vectors
    k_rbf_values = RadialBasis(
        num_radial=num_k_rbf,
        # Avoids zero or extremely small RBF values (there are k-points until
        # right at the cutoff, where all RBF would otherwise drop to 0)
        cutoff=k_cutoff + k_offset,
        rbf={"name": "gaussian"},
        envelope={"name": "polynomial", "exponent": 5},
    )(
        torch.linalg.norm(k_grid, dim=-1)
    )  # Tensor of shape (N_k, N_RBF)

    num_k_degrees_of_freedom = k_rbf_values.shape[-1]

    return k_grid, k_rbf_values, num_k_degrees_of_freedom


def compute_mean_std_atomic_inter_energy(
    data_loader: torch.utils.data.DataLoader,
    atomic_energies: np.ndarray,
) -> Tuple[float, float]:
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

    avg_atom_inter_es_list = []

    for batch in data_loader:
        node_e0 = atomic_energies_fn(batch.node_attrs)
        graph_e0s = scatter_sum(
            src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs
        )
        graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
        avg_atom_inter_es_list.append(
            (batch.energy - graph_e0s) / graph_sizes
        )  # {[n_graphs], }

    avg_atom_inter_es = torch.cat(avg_atom_inter_es_list)  # [total_n_graphs]
    mean = to_numpy(torch.mean(avg_atom_inter_es)).item()
    std = to_numpy(torch.std(avg_atom_inter_es)).item()
    std = _check_non_zero(std)

    return mean, std


def _compute_mean_std_atomic_inter_energy(
    batch: Batch,
    atomic_energies_fn: AtomicEnergiesBlock,
) -> Tuple[torch.Tensor, torch.Tensor]:
    node_e0 = atomic_energies_fn(batch.node_attrs)
    graph_e0s = scatter_sum(
        src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs
    )
    graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
    atom_energies = (batch.energy - graph_e0s) / graph_sizes
    return atom_energies


def compute_mean_rms_energy_forces(
    data_loader: torch.utils.data.DataLoader,
    atomic_energies: np.ndarray,
) -> Tuple[float, float]:
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

    atom_energy_list = []
    forces_list = []

    for batch in data_loader:
        node_e0 = atomic_energies_fn(batch.node_attrs)
        graph_e0s = scatter_sum(
            src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs
        )
        graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
        atom_energy_list.append(
            (batch.energy - graph_e0s) / graph_sizes
        )  # {[n_graphs], }
        forces_list.append(batch.forces)  # {[n_graphs*n_atoms,3], }

    atom_energies = torch.cat(atom_energy_list, dim=0)  # [total_n_graphs]
    forces = torch.cat(forces_list, dim=0)  # {[total_n_graphs*n_atoms,3], }

    mean = to_numpy(torch.mean(atom_energies)).item()
    rms = to_numpy(torch.sqrt(torch.mean(torch.square(forces)))).item()
    rms = _check_non_zero(rms)

    return mean, rms


def _compute_mean_rms_energy_forces(
    batch: Batch,
    atomic_energies_fn: AtomicEnergiesBlock,
) -> Tuple[torch.Tensor, torch.Tensor]:
    node_e0 = atomic_energies_fn(batch.node_attrs)
    graph_e0s = scatter_sum(
        src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs
    )
    graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
    atom_energies = (batch.energy - graph_e0s) / graph_sizes  # {[n_graphs], }
    forces = batch.forces  # {[n_graphs*n_atoms,3], }

    return atom_energies, forces


def compute_avg_num_neighbors(data_loader: torch.utils.data.DataLoader) -> float:
    num_neighbors = []

    for batch in data_loader:
        _, receivers = batch.edge_index
        _, counts = torch.unique(receivers, return_counts=True)
        num_neighbors.append(counts)

    avg_num_neighbors = torch.mean(
        torch.cat(num_neighbors, dim=0).type(torch.get_default_dtype())
    )
    return to_numpy(avg_num_neighbors).item()


def compute_statistics(
    data_loader: torch.utils.data.DataLoader,
    atomic_energies: np.ndarray,
) -> Tuple[float, float, float, float]:
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

    atom_energy_list = []
    forces_list = []
    num_neighbors = []

    for batch in data_loader:
        node_e0 = atomic_energies_fn(batch.node_attrs)
        graph_e0s = scatter_sum(
            src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs
        )
        graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
        atom_energy_list.append(
            (batch.energy - graph_e0s) / graph_sizes
        )  # {[n_graphs], }
        forces_list.append(batch.forces)  # {[n_graphs*n_atoms,3], }

        _, receivers = batch.edge_index
        _, counts = torch.unique(receivers, return_counts=True)
        num_neighbors.append(counts)

    atom_energies = torch.cat(atom_energy_list, dim=0)  # [total_n_graphs]
    forces = torch.cat(forces_list, dim=0)  # {[total_n_graphs*n_atoms,3], }

    mean = to_numpy(torch.mean(atom_energies)).item()
    rms = to_numpy(torch.sqrt(torch.mean(torch.square(forces)))).item()

    avg_num_neighbors = torch.mean(
        torch.cat(num_neighbors, dim=0).type(torch.get_default_dtype())
    )

    return to_numpy(avg_num_neighbors).item(), mean, rms


def compute_rms_dipoles(
    data_loader: torch.utils.data.DataLoader,
) -> Tuple[float, float]:
    dipoles_list = []
    for batch in data_loader:
        dipoles_list.append(batch.dipole)  # {[n_graphs,3], }

    dipoles = torch.cat(dipoles_list, dim=0)  # {[total_n_graphs,3], }
    rms = to_numpy(torch.sqrt(torch.mean(torch.square(dipoles)))).item()
    rms = _check_non_zero(rms)
    return rms


def compute_fixed_charge_dipole(
    charges: torch.Tensor,
    positions: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    mu = positions * charges.unsqueeze(-1) / (1e-11 / c / e)  # [N_atoms,3]
    return scatter_sum(
        src=mu, index=batch.unsqueeze(-1), dim=0, dim_size=num_graphs
    )  # [N_graphs,3]
