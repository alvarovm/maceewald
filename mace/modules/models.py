###########################################################################################
# Implementation of MACE models and other models based E(3)-Equivariant MPNNs
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch

# torch.set_default_dtype(torch.float64)

from e3nn import o3
from e3nn.util.jit import compile_mode
from .ewald_block import EwaldBlock
from .base_layers import *

from mace.data import AtomicData
from mace.modules.radial import ZBLBasis
from mace.tools.scatter import scatter_sum
from torch_geometric.nn.models.schnet import GaussianSmearing

from .blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearDipoleReadoutBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearDipoleReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    ScaleShiftBlock,
)
from .utils import (
    compute_fixed_charge_dipole,
    compute_forces,
    get_edge_vectors_and_lengths,
    get_outputs,
    get_symmetric_displacement,
    pos_svd_frame,
    get_k_index_product_set,
    get_k_voxel_grid,
    x_to_k_cell,
)

# pylint: disable=C0302


@compile_mode("script")
class MACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: Union[int, List[int]],
        gate: Optional[Callable],
        pair_repulsion: bool = False,
        distance_transform: str = "None",
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=distance_transform,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")
        if pair_repulsion:
            self.pair_repulsion_fn = ZBLBasis(r_max=r_max, p=num_polynomial_cutoff)
            self.pair_repulsion = True

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readout
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation[0],
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearReadoutBlock(hidden_irreps))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                hidden_irreps_out = str(
                    hidden_irreps[0]
                )  # Select only scalars for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i + 1],
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearReadoutBlock(hidden_irreps_out, MLP_irreps, gate)
                )
            else:
                self.readouts.append(LinearReadoutBlock(hidden_irreps))

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]
        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )
        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.pair_repulsion_fn(
                lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
            )
            pair_energy = scatter_sum(
                src=pair_node_energy, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
        else:
            pair_node_energy = torch.zeros_like(node_e0)
            pair_energy = torch.zeros_like(e0)

        # Interactions
        energies = [e0, pair_energy]
        node_energies_list = [node_e0, pair_node_energy]
        node_feats_list = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_feats_list.append(node_feats)
            node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            energy = scatter_sum(
                src=node_energies, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
            energies.append(energy)
            node_energies_list.append(node_energies)

        # Concatenate node features
        node_feats_out = torch.cat(node_feats_list, dim=-1)

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        node_energy_contributions = torch.stack(node_energies_list, dim=-1)
        node_energy = torch.sum(node_energy_contributions, dim=-1)  # [n_nodes, ]

        # Outputs
        forces, virials, stress = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )

        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "node_feats": node_feats_out,
        }


@compile_mode("script")
class ScaleShiftMACE(MACE):
    def __init__(
        self,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["positions"].requires_grad_(True)
        data["node_attrs"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )
        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.pair_repulsion_fn(
                lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
            )
        else:
            pair_node_energy = torch.zeros_like(node_e0)
        # Interactions
        node_es_list = [pair_node_energy]
        node_feats_list = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"]
            )
            node_feats_list.append(node_feats)
            node_es_list.append(readout(node_feats).squeeze(-1))  # {[n_nodes, ], }
        # Concatenate node features
        node_feats_out = torch.cat(node_feats_list, dim=-1)
        # print("node_es_list", node_es_list)
        # Sum over interactions
        node_inter_es = torch.sum(
            torch.stack(node_es_list, dim=0), dim=0
        )  # [n_nodes, ]
        node_inter_es = self.scale_shift(node_inter_es)

        # Sum over nodes in graph
        inter_e = scatter_sum(
            src=node_inter_es, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Add E_0 and (scaled) interaction energy
        total_energy = e0 + inter_e
        node_energy = node_e0 + node_inter_es
        forces, virials, stress = get_outputs(
            energy=inter_e,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )
        output = {
            "energy": total_energy,
            "node_energy": node_energy,
            "interaction_energy": inter_e,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "node_feats": node_feats_out,
        }

        return output


class BOTNet(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
        gate: Optional[Callable],
        avg_num_neighbors: float,
        atomic_numbers: List[int],
    ):
        super().__init__()
        self.r_max = r_max
        self.atomic_numbers = atomic_numbers
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        # Interactions and readouts
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        self.interactions = torch.nn.ModuleList()
        self.readouts = torch.nn.ModuleList()

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
        )
        self.interactions.append(inter)
        self.readouts.append(LinearReadoutBlock(inter.irreps_out))

        for i in range(num_interactions - 1):
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=inter.irreps_out,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=hidden_irreps,
                avg_num_neighbors=avg_num_neighbors,
            )
            self.interactions.append(inter)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearReadoutBlock(inter.irreps_out, MLP_irreps, gate)
                )
            else:
                self.readouts.append(LinearReadoutBlock(inter.irreps_out))

    def forward(self, data: AtomicData, training=False) -> Dict[str, Any]:
        # Setup
        data.positions.requires_grad = True

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data.node_attrs)
        e0 = scatter_sum(
            src=node_e0, index=data.batch, dim=-1, dim_size=data.num_graphs
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data.node_attrs)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data.positions, edge_index=data.edge_index, shifts=data.shifts
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )

        # Interactions
        energies = [e0]
        for interaction, readout in zip(self.interactions, self.readouts):
            node_feats = interaction(
                node_attrs=data.node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data.edge_index,
            )
            node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            energy = scatter_sum(
                src=node_energies, index=data.batch, dim=-1, dim_size=data.num_graphs
            )  # [n_graphs,]
            energies.append(energy)

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]

        output = {
            "energy": total_energy,
            "contributions": contributions,
            "forces": compute_forces(
                energy=total_energy, positions=data.positions, training=training
            ),
        }

        return output


class ScaleShiftBOTNet(BOTNet):
    def __init__(
        self,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )

    def forward(self, data: AtomicData, training=False) -> Dict[str, Any]:
        # Setup
        data.positions.requires_grad = True

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data.node_attrs)
        e0 = scatter_sum(
            src=node_e0, index=data.batch, dim=-1, dim_size=data.num_graphs
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data.node_attrs)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data.positions, edge_index=data.edge_index, shifts=data.shifts
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )

        # Interactions
        node_es_list = []
        for interaction, readout in zip(self.interactions, self.readouts):
            node_feats = interaction(
                node_attrs=data.node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data.edge_index,
            )

            node_es_list.append(readout(node_feats).squeeze(-1))  # {[n_nodes, ], }

        # Sum over interactions
        node_inter_es = torch.sum(
            torch.stack(node_es_list, dim=0), dim=0
        )  # [n_nodes, ]
        node_inter_es = self.scale_shift(node_inter_es)

        # Sum over nodes in graph
        inter_e = scatter_sum(
            src=node_inter_es, index=data.batch, dim=-1, dim_size=data.num_graphs
        )  # [n_graphs,]

        # Add E_0 and (scaled) interaction energy
        total_e = e0 + inter_e

        output = {
            "energy": total_e,
            "forces": compute_forces(
                energy=inter_e, positions=data.positions, training=training
            ),
        }

        return output


@compile_mode("script")
class AtomicDipolesMACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: int,
        gate: Optional[Callable],
        atomic_energies: Optional[
            None
        ],  # Just here to make it compatible with energy models, MUST be None
        radial_type: Optional[str] = "bessel",
        radial_MLP: Optional[List[int]] = None,
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float64))
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        assert atomic_energies is None

        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]

        # Interactions and readouts
        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearDipoleReadoutBlock(hidden_irreps, dipole_only=True))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                assert (
                    len(hidden_irreps) > 1
                ), "To predict dipoles use at least l=1 hidden_irreps"
                hidden_irreps_out = str(
                    hidden_irreps[1]
                )  # Select only l=1 vectors for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearDipoleReadoutBlock(
                        hidden_irreps_out, MLP_irreps, gate, dipole_only=True
                    )
                )
            else:
                self.readouts.append(
                    LinearDipoleReadoutBlock(hidden_irreps, dipole_only=True)
                )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,  # pylint: disable=W0613
        compute_force: bool = False,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        assert compute_force is False
        assert compute_virials is False
        assert compute_stress is False
        assert compute_displacement is False
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )

        # Interactions
        dipoles = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_dipoles = readout(node_feats).squeeze(-1)  # [n_nodes,3]
            dipoles.append(node_dipoles)

        # Compute the dipoles
        contributions_dipoles = torch.stack(
            dipoles, dim=-1
        )  # [n_nodes,3,n_contributions]
        atomic_dipoles = torch.sum(contributions_dipoles, dim=-1)  # [n_nodes,3]
        total_dipole = scatter_sum(
            src=atomic_dipoles,
            index=data["batch"],
            dim=0,
            dim_size=num_graphs,
        )  # [n_graphs,3]
        baseline = compute_fixed_charge_dipole(
            charges=data["charges"],
            positions=data["positions"],
            batch=data["batch"],
            num_graphs=num_graphs,
        )  # [n_graphs,3]
        total_dipole = total_dipole + baseline

        output = {
            "dipole": total_dipole,
            "atomic_dipoles": atomic_dipoles,
        }
        return output


@compile_mode("script")
class EnergyDipolesMACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: int,
        gate: Optional[Callable],
        atomic_energies: Optional[np.ndarray],
        radial_MLP: Optional[List[int]] = None,
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float64))
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readouts
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearDipoleReadoutBlock(hidden_irreps, dipole_only=False))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                assert (
                    len(hidden_irreps) > 1
                ), "To predict dipoles use at least l=1 hidden_irreps"
                hidden_irreps_out = str(
                    hidden_irreps[:2]
                )  # Select scalars and l=1 vectors for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearDipoleReadoutBlock(
                        hidden_irreps_out, MLP_irreps, gate, dipole_only=False
                    )
                )
            else:
                self.readouts.append(
                    LinearDipoleReadoutBlock(hidden_irreps, dipole_only=False)
                )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )

        # Interactions
        energies = [e0]
        node_energies_list = [node_e0]
        dipoles = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_out = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            # node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            node_energies = node_out[:, 0]
            energy = scatter_sum(
                src=node_energies, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
            energies.append(energy)
            node_dipoles = node_out[:, 1:]
            dipoles.append(node_dipoles)

        # Compute the energies and dipoles
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        node_energy_contributions = torch.stack(node_energies_list, dim=-1)
        node_energy = torch.sum(node_energy_contributions, dim=-1)  # [n_nodes, ]
        contributions_dipoles = torch.stack(
            dipoles, dim=-1
        )  # [n_nodes,3,n_contributions]
        atomic_dipoles = torch.sum(contributions_dipoles, dim=-1)  # [n_nodes,3]
        total_dipole = scatter_sum(
            src=atomic_dipoles,
            index=data["batch"].unsqueeze(-1),
            dim=0,
            dim_size=num_graphs,
        )  # [n_graphs,3]
        baseline = compute_fixed_charge_dipole(
            charges=data["charges"],
            positions=data["positions"],
            batch=data["batch"],
            num_graphs=num_graphs,
        )  # [n_graphs,3]
        total_dipole = total_dipole + baseline

        forces, virials, stress = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )

        output = {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "dipole": total_dipole,
            "atomic_dipoles": atomic_dipoles,
        }
        return output

@compile_mode("script")
class MACE_Ewald(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: Union[int, List[int]],
        gate: Optional[Callable],
        pair_repulsion: bool = False,
        distance_transform: str = "None",
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        ewald_hyperparams=None,
        atom_to_atom_cutoff=None,    
        use_pbc: bool = False,
    ):
        super().__init__()
        self.slice_indices = []
        self.use_pbc = use_pbc
        # self.k_grid_original = None
    ############################################ EwaldMP layers ######################################################               
        # Parse Ewald hyperparams
        self.use_ewald = ewald_hyperparams is not None
        self.atom_to_atom_cutoff = atom_to_atom_cutoff
        self.use_atom_to_atom_mp = atom_to_atom_cutoff is not None
        hidden_channels = hidden_irreps.count(o3.Irrep(0, 1)) # <------ Embedding size of MACE, based on scalar features 
        
        # print("use_pbc:", self.use_pbc, use_pbc)
        # print("use_ewald:", self.use_ewald)

        if self.use_ewald:
            print("ewald is activated")
            if self.use_pbc:
                print("pbc is activated")
                # Integer values to define box of k-lattice indices
                self.num_k_x = ewald_hyperparams["num_k_x"]
                self.num_k_y = ewald_hyperparams["num_k_y"]
                self.num_k_z = ewald_hyperparams["num_k_z"]
                print("nkx, nky, nkz:", self.num_k_x, self.num_k_y, self.num_k_z)
                self.delta_k = None
            else:
                self.k_cutoff = ewald_hyperparams["k_cutoff"]
                # Voxel grid resolution
                self.delta_k = ewald_hyperparams["delta_k"]
                # Radial k-filter basis size
                self.num_k_rbf = ewald_hyperparams["num_k_rbf"]
            self.downprojection_size = ewald_hyperparams["downprojection_size"]
            # Number of residuals in update function
            self.num_hidden = ewald_hyperparams["num_hidden"]

        # Initialize k-space structure
        if self.use_ewald:
            if self.use_pbc:
                # Get the reciprocal lattice indices of included k-vectors
                (
                    self.k_index_product_set,
                    self.num_k_degrees_of_freedom,
                ) = get_k_index_product_set(
                    self.num_k_x,
                    self.num_k_y,
                    self.num_k_z,
                )
                
                self.k_rbf_values = None
                self.delta_k = None

            else:
                # Get the k-space voxel and evaluate Gaussian RBF (can be done at
                # initialization time as voxel grid stays fixed for all structures)
                (
                    self.k_grid_original,
                    self.k_rbf_values,
                    self.num_k_degrees_of_freedom,
                ) = get_k_voxel_grid(
                    self.k_cutoff,
                    self.delta_k,
                    self.num_k_rbf,
                )

            # Downprojection layer, weights are shared among all interaction blocks
            self.down = Dense(
                self.num_k_degrees_of_freedom,
                self.downprojection_size,
                activation=None,
                bias=False,
            )

            self.ewald_blocks = torch.nn.ModuleList(
                [
                    EwaldBlock(
                        self.down,
                        hidden_channels,  # Embedding size of short-range GNN
                        self.downprojection_size,
                        self.num_hidden,  # Number of residuals in update function
                        activation="silu",
                        use_pbc=self.use_pbc,
                        delta_k=self.delta_k,
                        k_rbf_values=self.k_rbf_values,
                    )
                    for i in range(num_interactions)
                ]
            )

        if self.use_atom_to_atom_mp:
            if self.use_pbc:
                # Compute neighbor threshold from cutoff assuming uniform atom density
                self.max_neighbors_at = int(
                    (self.atom_to_atom_cutoff / 6.0) ** 3 * 50
                )
            else:
                self.max_neighbors_at = 100
            # SchNet interactions for atom-to-atom message passing
            self.interactions_at = torch.nn.ModuleList(
                [
                    InteractionBlock(
                        hidden_channels,
                        200,  # num Gaussians
                        256,  # num filters
                        self.atom_to_atom_cutoff,
                    )
                    for i in range(num_interactions)
                ]
            )
            self.distance_expansion_at = GaussianSmearing(
                0.0, self.atom_to_atom_cutoff, 200
            )

        self.skip_connection_factor = (
            2.0 + float(self.use_ewald) + float(self.use_atom_to_atom_mp)
        ) ** (-0.5)

    ############################################ MACE layers ######################################################       

        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=distance_transform,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")
        if pair_repulsion:
            self.pair_repulsion_fn = ZBLBasis(r_max=r_max, p=num_polynomial_cutoff)
            self.pair_repulsion = True

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readout
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        self.slice_indices.append(hidden_irreps.count((o3.Irrep(0, 1))))
        print("hidden_irreps:", hidden_irreps.simplify(), self.slice_indices)

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation[0],
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearReadoutBlock(hidden_irreps))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                hidden_irreps_out = str(
                    hidden_irreps[0]
                )  # Select only scalars for last layer
                self.slice_indices.append(o3.Irreps(hidden_irreps_out).count((o3.Irrep(0, 1))))
                print("interaction:", i, "hidden_irreps:", o3.Irreps(hidden_irreps_out).simplify(), self.slice_indices)
            else:
                hidden_irreps_out = hidden_irreps
                self.slice_indices.append(hidden_irreps_out.count((o3.Irrep(0, 1))))
                print("interaction:", i, "hidden_irreps:", hidden_irreps.simplify(), self.slice_indices)

            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i + 1],
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearReadoutBlock(hidden_irreps_out, MLP_irreps, gate)
                )
            else:
                self.readouts.append(LinearReadoutBlock(hidden_irreps))

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        
        ################# EwaldSetup ###################
        # print("self.k_grid:", self.k_grid)
        # print("data type:", type(data))
        pos = (
            pos_svd_frame(data)
            if (self.use_ewald and not self.use_pbc)
            else data["positions"]
        )

        batch = data["batch"]
        
        batch_size = int(batch.max()) + 1
        print("batch-size:", batch_size, data["cell"])

        if self.use_ewald:
            if self.use_pbc:
                # Compute reciprocal lattice basis of structure
                k_cell, _ = x_to_k_cell(data["cell"], batch_size)
                # Translate lattice indices to k-vectors
                # print("k_index_product_set shape if periodic:",  self.k_index_product_set.shape, self.k_index_product_set.dtype)
                # print("k_cell shape if periodic:", k_cell.shape, k_cell.dtype)
                if not hasattr(self, 'k_index_product_set'):
                    raise RuntimeError("k_index_product_set seems wrong.")
                k_grid = torch.matmul(
                    self.k_index_product_set.type(torch.float64).to(batch.device), k_cell
                )

                # k_grid = (
                #     k_grid.to(batch.device)
                #     .unsqueeze(0)
                #     .expand(batch_size, -1, -1)
                # )

                # print("k_grid shape if periodic:", k_grid.shape, k_grid.dtype)
            else:
                # print("self.k_grid shape if NOT periodic:", self.k_grid.shape)
                # if self.k_grid is None:
                   # raise ValueError("k_grid must be provided if use_pbc is False")
                if not hasattr(self, 'k_grid_original'):
                    raise RuntimeError("k_grid_original seems wrong.")
                if self.k_grid_original is None:
                    raise RuntimeError("k_grid_original seems None.")
                
                k_grid = (self.k_grid_original.to(batch.device).unsqueeze(0).expand(batch_size, -1, -1))
                # print("k_grid shape if NOT periodic:", k_grid.shape, k_grid.dtype)
        else:
            k_grid = torch.torch.empty(0)

        # if self.use_atom_to_atom_mp:
        #     # Use separate graph (larger cutoff) for atom-to-atom long-range block
        #     (
        #         edge_index_at,
        #         edge_weight_at,
        #         distance_vec_at,
        #         cell_offsets_at,
        #         _,  # cell offset distances
        #         neighbors_at,
        #     ) = self.generate_graph(
        #         data,
        #         cutoff=self.atom_to_atom_cutoff,
        #         max_neighbors=self.max_neighbors_at,
        #     )

        #     edge_attr_at = self.distance_expansion_at(edge_weight_at)
        
        ################# MACE Setup ###################
            
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )

        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )
        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.pair_repulsion_fn(
                lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
            )
            pair_energy = scatter_sum(
                src=pair_node_energy, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
        else:
            pair_node_energy = torch.zeros_like(node_e0)
            pair_energy = torch.zeros_like(e0)

        # Interactions
        energies = [e0, pair_energy]
        node_energies_list = [node_e0, pair_node_energy]
        node_feats_list = []
        
        print("self.slice_indices:", self.slice_indices)
        print("node_feats:", node_feats.shape)

        ############ Ewald Interaction blocks ###############################################
        enum_ind = 0
        if self.use_ewald or self.use_atom_to_atom_mp:
            dot = torch.torch.empty(0) # These will be computed in first Ewald block and then passed
            sinc_damping = torch.torch.empty(0) # on between later Ewald blocks (avoids redundant recomputation)
            for interaction, product, readout, ew_blocks in zip(
                self.interactions, self.products, self.readouts, self.ewald_blocks
                ):
                    print("interaction layer:", enum_ind)
                    if self.use_ewald:
                        node_feats_ewald, dot, sinc_damping = ew_blocks (
                            node_feats[:, :self.slice_indices[enum_ind]], pos, k_grid, batch_size, batch, dot, sinc_damping
                        )
                    else:
                        node_feats_ewald = torch.tensor(0, dtype=torch.float64)

                    # if self.use_atom_to_atom_mp:
                    #     dx_at = self.interactions_at[enum_ind](
                    #         node_feats[:, :self.slice_indices[enum_ind]], edge_index_at, edge_weight_at, edge_attr_at
                    #     )
                    # else:
                    #     dx_at = 0
                    
                    print("interaction layer:", enum_ind, "node feats after ewald:", node_feats_ewald.shape)

        ################################ MACE Interaction blocks ####################################################                         
                    
                    node_feats, sc = interaction(
                        node_attrs=data["node_attrs"],
                        node_feats=node_feats,
                        edge_attrs=edge_attrs,
                        edge_feats=edge_feats,
                        edge_index=data["edge_index"],
                    )
                    
                    node_feats = product(
                        node_feats=node_feats,
                        sc=sc,
                        node_attrs=data["node_attrs"],
                    )

                    print("interaction layer:", enum_ind, "node feats after MACE:", node_feats.shape)

        ################################ Mixing MACE node_feats with Ewald ###################################################                         
                    
                    node_feats[:, :self.slice_indices[enum_ind]] += node_feats_ewald            
                    node_feats[:, :self.slice_indices[enum_ind]] *= self.skip_connection_factor
                    print("interaction layer:", enum_ind, "node_feats inside interaction:", node_feats.shape)            
                    node_feats_list.append(node_feats)
                    node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
                    energy = scatter_sum(
                        src=node_energies, index=data["batch"], dim=-1, dim_size=num_graphs
                    )  # [n_graphs,]
                    energies.append(energy)
                    node_energies_list.append(node_energies)
                    enum_ind+=1
    
        else:
            for enum_ind, (interaction, product, readout) in enumerate(zip(
                self.interactions, self.products, self.readouts)):
                    
                    node_feats, sc = interaction(
                        node_attrs=data["node_attrs"],
                        node_feats=node_feats,
                        edge_attrs=edge_attrs,
                        edge_feats=edge_feats,
                        edge_index=data["edge_index"],
                    )
                    
                    node_feats = product(
                        node_feats=node_feats,
                        sc=sc,
                        node_attrs=data["node_attrs"],
                    )

                    print("interaction layer:", enum_ind, "node feats after MACE:", node_feats.shape)                    
                    print("interaction layer:", enum_ind)
                    
                    # node_feats[:, :self.slice_indices[enum_ind]] += node_feats_ewald 
                    # node_feats[:, :self.slice_indices[enum_ind]] *= self.skip_connection_factor           
                    print("interaction layer:", enum_ind, "node_feats inside interaction:", node_feats.shape)            
                    node_feats_list.append(node_feats)
                    node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
                    energy = scatter_sum(
                        src=node_energies, index=data["batch"], dim=-1, dim_size=num_graphs
                    )  # [n_graphs,]
                    energies.append(energy)
                    node_energies_list.append(node_energies)  

        # Concatenate node features
        node_feats_out = torch.cat(node_feats_list, dim=-1)

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        node_energy_contributions = torch.stack(node_energies_list, dim=-1)
        node_energy = torch.sum(node_energy_contributions, dim=-1)  # [n_nodes, ]

        # Outputs
        forces, virials, stress = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )

        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "node_feats": node_feats_out,
        }


@compile_mode("script")
class MACE_Phys(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: Union[int, List[int]],
        gate: Optional[Callable],
        pair_repulsion: bool = False,
        distance_transform: str = "None",
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=distance_transform,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")
        if pair_repulsion:
            self.pair_repulsion_fn = ZBLBasis(r_max=r_max, p=num_polynomial_cutoff)
            self.pair_repulsion = True

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readout
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation[0],
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.ewald_readouts = torch.nn.ModuleList()

        self.readouts.append(LinearReadoutBlock(hidden_irreps))
        

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                hidden_irreps_out = str(
                    hidden_irreps[0]
                )  # Select only scalars for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i + 1],
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearReadoutBlock(hidden_irreps_out, MLP_irreps, gate)
                )
            else:
                self.readouts.append(LinearReadoutBlock(hidden_irreps))

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )

        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]
        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )
        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.pair_repulsion_fn(
                lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
            )
            pair_energy = scatter_sum(
                src=pair_node_energy, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
        else:
            pair_node_energy = torch.zeros_like(node_e0)
            pair_energy = torch.zeros_like(e0)

        # Interactions
        energies = [e0, pair_energy]
        node_energies_list = [node_e0, pair_node_energy]
        node_feats_list = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_feats_list.append(node_feats)
            node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            energy = scatter_sum(
                src=node_energies, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
            energies.append(energy)
            node_energies_list.append(node_energies)

        # Concatenate node features
        node_feats_out = torch.cat(node_feats_list, dim=-1)

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        node_energy_contributions = torch.stack(node_energies_list, dim=-1)
        node_energy = torch.sum(node_energy_contributions, dim=-1)  # [n_nodes, ]

        # Outputs
        forces, virials, stress = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )

        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "node_feats": node_feats_out,
        }
