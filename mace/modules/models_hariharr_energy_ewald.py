from typing import Any, Callable, Dict, List, Optional, Type
from .base_layers import *
from .ewald_block import EwaldBlock
# from .base import BaseModel
import numpy as np
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
from torch_geometric.nn.models.schnet import GaussianSmearing, InteractionBlock

from mace.data import AtomicData
from mace.tools.scatter import scatter_sum

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
        correlation: int,
        gate: Optional[Callable],
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        ewald_hyperparams=None,
        atom_to_atom_cutoff=None,    
        use_pbc=None,
    ):
        super().__init__()
        self.slice_indices = []
        self.use_pbc = use_pbc

    ############################################ EwaldMP layers ######################################################               
        # Parse Ewald hyperparams
        self.use_ewald = ewald_hyperparams is not None
        self.atom_to_atom_cutoff = atom_to_atom_cutoff
        self.use_atom_to_atom_mp = atom_to_atom_cutoff is not None
        hidden_channels = hidden_irreps.count(o3.Irrep(0, 1)) # <------ Embedding size of MACE, based on scalar features 
        
        if self.use_ewald:
            if self.use_pbc:
                # Integer values to define box of k-lattice indices
                self.num_k_x = ewald_hyperparams["num_k_x"]
                self.num_k_y = ewald_hyperparams["num_k_y"]
                self.num_k_z = ewald_hyperparams["num_k_z"]
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
                    self.k_grid,
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
            correlation=correlation,
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
                correlation=correlation,
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
        print("data type:", type(data))
        pos = (
            pos_svd_frame(data)
            if (self.use_ewald and not self.use_pbc)
            else data.positions
        )

        batch = data.batch
        
        batch_size = int(batch.max()) + 1
        
        if self.use_ewald:
            if self.use_pbc:
                # Compute reciprocal lattice basis of structure
                k_cell, _ = x_to_k_cell(data.cell, batch_size)
                # Translate lattice indices to k-vectors
                k_grid = torch.matmul(
                    self.k_index_product_set.to(batch.device), k_cell
                )
                k_grid = (
                    k_grid.to(batch.device)
                    .unsqueeze(0)
                    .expand(batch_size, -1, -1)
                )
                print("k_grid shape if periodic:", k_grid.shape)
                print("k_index_product_set shape if periodic:",  self.k_index_product_set.shape)
                print("k_cell shape if periodic:", k_cell.shape)

            else:
                print("self.k_grid shape if NOT periodic:", self.k_grid.shape)
                k_grid = (
                    self.k_grid.to(batch.device)
                    .unsqueeze(0)
                    .expand(batch_size, -1, -1)
                )
                print("k_grid shape if NOT periodic:", k_grid.shape)

        if self.use_atom_to_atom_mp:
            # Use separate graph (larger cutoff) for atom-to-atom long-range block
            (
                edge_index_at,
                edge_weight_at,
                distance_vec_at,
                cell_offsets_at,
                _,  # cell offset distances
                neighbors_at,
            ) = self.generate_graph(
                data,
                cutoff=self.atom_to_atom_cutoff,
                max_neighbors=self.max_neighbors_at,
            )

            edge_attr_at = self.distance_expansion_at(edge_weight_at)
        
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
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        energies = [e0]
        node_energies_list = [node_e0]
        node_feats_list = []
        
        print("self.slice_indices:", self.slice_indices)
        print("node_feats:", node_feats.shape)

        ############ Ewald Interaction blocks ###############################################

        if self.use_ewald or self.use_atom_to_atom_mp:
            dot = None  # These will be computed in first Ewald block and then passed
            sinc_damping = None  # on between later Ewald blocks (avoids redundant recomputation)
            
            for enum_ind, (interaction, product, readout) in enumerate(zip(
                self.interactions, self.products, self.readouts)):
                    print("interaction layer:", enum_ind)
                    if self.use_ewald:
                        node_feats_ewald, dot, sinc_damping = self.ewald_blocks[enum_ind](
                            node_feats[:, :self.slice_indices[enum_ind]], pos, k_grid, batch_size, batch, dot, sinc_damping
                        )
                    else:
                        node_feats_ewald = 0

                    if self.use_atom_to_atom_mp:
                        dx_at = self.interactions_at[enum_ind](
                            node_feats[:, :self.slice_indices[enum_ind]], edge_index_at, edge_weight_at, edge_attr_at
                        )
                    else:
                        dx_at = 0
                    
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