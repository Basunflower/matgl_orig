from __future__ import annotations

import os

import torch as th

from matgl.layers import BondExpansion
from matgl.models import MEGNet


class TestMEGNetTest:
    def test_megnet(self, graph_MoS):
        structure, graph, state = graph_MoS
        model = MEGNet(
            dim_node_embedding=16,
            dim_edge_embedding=100,
            dim_state_embedding=2,
            nblocks=3,
            include_states=True,
            hidden_layer_sizes_input=(64, 32),
            hidden_layer_sizes_conv=(64, 64, 32),
            activation_type="swish",
            nlayers_set2set=4,
            niters_set2set=3,
            hidden_layer_sizes_output=(32, 16),
            is_classification=True,
        )
        bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=6.0, num_centers=100, width=0.5)
        graph.edata["edge_attr"] = bond_expansion(graph.edata["bond_dist"])
        state = th.tensor(state)
        output = model(graph, graph.edata["edge_attr"], graph.ndata["node_type"], state)
        assert [th.numel(output)] == [1]

    def test_save_load(self):
        model = MEGNet(
            dim_node_embedding=16,
            dim_edge_embedding=100,
            dim_state_embedding=2,
            nblocks=3,
            include_states=True,
            hidden_layer_sizes_input=(64, 32),
            hidden_layer_sizes_conv=(64, 64, 32),
            activation_type="swish",
            nlayers_set2set=4,
            niters_set2set=3,
            hidden_layer_sizes_output=(32, 16),
            is_classification=True,
        )
        model.save(".", metadata={"description": "forme model"})
        MEGNet.load(".")
        os.remove("model.pt")
        os.remove("model.json")
        os.remove("state.pt")
