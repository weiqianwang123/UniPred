"""Graph neural network core code.

A lot of this was copied and modified from code originally written by
Kelsey Allen.
"""

import abc
import logging
from typing import Dict, List

import numpy as np
import torch
from torch import nn

from predicators.gnn.gnn_utils import GraphTransC2DDataset, replace_graph


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert quaternions to rotation matrices.

    Quaternions are given real part first, i.e., (w, x, y, z), the same
    convention as pytorch3d.transforms.quaternion_to_matrix. They do not
    need to have unit norm: the result is the rotation of the normalized
    quaternion.

    Args:
        quaternions: tensor of shape (..., 4).

    Returns:
        Rotation matrices of shape (..., 3, 3).
    """
    w, x, y, z = torch.unbind(quaternions, dim=-1)
    s = 2.0 / (quaternions * quaternions).sum(dim=-1)
    entries = (
        1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w),
        s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w),
        s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y),
    )
    return torch.stack(entries, dim=-1).reshape(quaternions.shape[:-1] +
                                                (3, 3))


def array_in_list(array, list_of_arrays):
    return any(np.array_equal(array, x) for x in list_of_arrays)


def MLP(layers: List[int], input_dim: int, with_ln: bool) -> nn.Sequential:
    """Create MLP."""
    LinearLayer = nn.Linear
    mlp_layers: List[nn.Module] = [LinearLayer(input_dim, layers[0])]

    for layer_num in range(0, len(layers) - 1):
        mlp_layers.append(nn.GELU())
        mlp_layers.append(nn.BatchNorm1d(layers[layer_num]))
        # mlp_layers.append(nn.Dropout(p=0.5))
        mlp_layers.append(LinearLayer(layers[layer_num],
                                      layers[layer_num + 1]))

    # Add a final normalization layer
    if with_ln:
        mlp_layers.append(nn.LayerNorm(layers[-1]))  # type: ignore

    return nn.Sequential(*mlp_layers)


class EncodeDecodeMLP(nn.Module, abc.ABC):
    """Encode-decode architecture."""

    def __init__(self, arity: int, \
                 encoder: nn.Module, \
                 decoder: nn.Module, \
                 init_fn) -> None:
        super().__init__()
        self.arity = arity
        self.encoder = encoder
        self.decoder = decoder
        self.apply_init(init_fn)

    def apply_init(self, init_fn):
        """Apply custom initialization to the encoder and decoder."""
        for p in self.parameters():
            if p.dim() > 1:
                init_fn(p)
            else:
                # Bias use uniform initialization
                nn.init.uniform_(p, -1.0, 1.0)

    def forward(self, graph: Dict) -> List[Dict]:
        """Torch forward model."""
        if self.arity == 1:
            input_data = graph['nodes']
        elif self.arity == 2:
            input_data = graph['edges']

        encoded = self.encoder(input_data)
        decoded = self.decoder(encoded)

        output_graph = []
        if self.arity == 1:
            replacements = {
                'nodes': decoded
            }
        elif self.arity == 2:
            replacements = {
                'edges': decoded
            }
        output_graph.append(replace_graph(graph, replacements))

        return output_graph
    
class SelectiveRotEncodeDecodeMLP(nn.Module, abc.ABC):
    """Encode-decode architecture."""

    def __init__(self, arity: int, \
                 encoder: nn.Module, \
                 encoder_rot: nn.Module, \
                 decoder: nn.Module, \
                 non_rot_feat: List[int], \
                 rot_feat: Dict, \
                 init_fn) -> None:
        super().__init__()
        self.non_rot_feat_inx = non_rot_feat
        self.rot_feat_ind = []
        for i in range(1, arity+1):
            local_rot = [None] * 4
            for feat in rot_feat.keys():
                # note that real part is the first element
                if "ent{}".format(i) in feat and "_qw" in feat:
                    local_rot[0] = rot_feat[feat]
                elif "ent{}".format(i) in feat and "_qx" in feat:
                    local_rot[1] = rot_feat[feat]
                elif "ent{}".format(i) in feat and "_qy" in feat:
                    local_rot[2] = rot_feat[feat]
                elif "ent{}".format(i) in feat and "_qz" in feat:
                    local_rot[3] = rot_feat[feat]
            self.rot_feat_ind.append(local_rot)
        self.arity = arity
        self.encoder = encoder
        self.encoder_rot = encoder_rot
        self.decoder = decoder
        self.apply_init(init_fn)

    def apply_init(self, init_fn):
        """Apply custom initialization to the encoder and decoder."""
        for p in self.parameters():
            if p.dim() > 1:
                init_fn(p)
            else:
                # Bias use uniform initialization
                nn.init.uniform_(p, -1.0, 1.0)

    def forward(self, graph: Dict) -> List[Dict]:
        """Torch forward model."""
        if self.arity == 1:
            input_data = graph['nodes']
            input_rot_quant = input_data[:, self.rot_feat_ind[0]]
            input_rot = quaternion_to_matrix(input_rot_quant).view(-1, 9)
        elif self.arity == 2:
            input_data = graph['edges']
            input_rot_quant_1 = input_data[:, self.rot_feat_ind[0]]
            if not (torch.isclose(torch.norm(input_rot_quant_1, dim=1), \
                    torch.ones_like(torch.norm(input_rot_quant_1, dim=1))).all()):
                logging.warning("Input quaternion does not have unit norm")
            input_rot_1 = quaternion_to_matrix(input_rot_quant_1).view(-1, 9)
            input_rot_quant_2 = input_data[:, self.rot_feat_ind[1]]
            if not torch.isclose(torch.norm(input_rot_quant_2, dim=1), \
                    torch.ones_like(torch.norm(input_rot_quant_2, dim=1))).all():
                logging.warning("Input quaternion does not have unit norm")
            input_rot_2 = quaternion_to_matrix(input_rot_quant_2).view(-1, 9)
            input_rot = torch.cat([input_rot_1, input_rot_2], dim=1)

        input_feat = input_data[:, self.non_rot_feat_inx]
        encoded_non_rot = self.encoder(input_feat)
        encoded_rot = self.encoder_rot(input_rot)
        encoded = torch.cat([encoded_non_rot, encoded_rot], dim=1)
        decoded = self.decoder(encoded)

        output_graph = []
        if self.arity == 1:
            replacements = {
                'nodes': decoded
            }
        elif self.arity == 2:
            replacements = {
                'edges': decoded
            }
        output_graph.append(replace_graph(graph, replacements))

        return output_graph
    
class SelectiveEncodeDecodeMLP(nn.Module, abc.ABC):
    """Encode-decode architecture."""

    def __init__(self, arity: int, \
                 encoder: nn.Module, \
                 decoder: nn.Module, \
                 feat_ind: List[int], \
                 init_fn) -> None:
        super().__init__()
        self.feat_ind = feat_ind
        self.arity = arity
        assert self.arity == 1, "Only support arity == 1 for selective feature"
        self.encoder = encoder
        self.decoder = decoder
        self.apply_init(init_fn)

    def apply_init(self, init_fn):
        """Apply custom initialization to the encoder and decoder."""
        for p in self.parameters():
            if p.dim() > 1:
                init_fn(p)
            else:
                # Bias use uniform initialization
                nn.init.uniform_(p, -1.0, 1.0)

    def forward(self, graph: Dict) -> List[Dict]:
        """Torch forward model."""
        input_data = graph['nodes']
        input_feat = input_data[:, self.feat_ind]
        encoded = self.encoder(input_feat)
        decoded = self.decoder(encoded)

        output_graph = []
        if self.arity == 1:
            replacements = {
                'nodes': decoded
            }
        elif self.arity == 2:
            replacements = {
                'edges': decoded
            }
        output_graph.append(replace_graph(graph, replacements))

        return output_graph
    
class SelectiveRelRotEncodeDecodeMLP(nn.Module, abc.ABC):
    """Encode-decode architecture."""

    def __init__(self, arity: int, \
                 encoder: nn.Module, \
                 encoder_rot: nn.Module, \
                 decoder: nn.Module, \
                 non_rot_feat: Dict, \
                 rot_feat: Dict, \
                 init_fn) -> None:
        super().__init__()
        assert arity == 2, "Only support arity == 2 for relative feature"
        self.non_rot_feat_inx = []
        self.rot_feat_ind = []
        for i in range(1, arity+1):
            local_rot = [None] * 4
            for feat in rot_feat.keys():
                # note that real part is the first element
                if "ent{}".format(i) in feat and "_qw" in feat:
                    local_rot[0] = rot_feat[feat]
                elif "ent{}".format(i) in feat and "_qx" in feat:
                    local_rot[1] = rot_feat[feat]
                elif "ent{}".format(i) in feat and "_qy" in feat:
                    local_rot[2] = rot_feat[feat]
                elif "ent{}".format(i) in feat and "_qz" in feat:
                    local_rot[3] = rot_feat[feat]
            self.rot_feat_ind.append(local_rot)
            # non rot also compute relative
            local_non_root = []
            for feat in non_rot_feat:
                if "ent{}".format(i) in feat:
                    local_non_root.append(non_rot_feat[feat])
                elif "ent{}".format(i) in feat:
                    local_non_root.append(non_rot_feat[feat])
                elif "ent{}".format(i) in feat:
                    local_non_root.append(non_rot_feat[feat])
                elif "ent{}".format(i) in feat:
                    local_non_root.append(non_rot_feat[feat])
            self.non_rot_feat_inx.append(local_non_root)
        self.arity = arity
        self.encoder = encoder
        self.encoder_rot = encoder_rot
        self.decoder = decoder
        self.apply_init(init_fn)

    def apply_init(self, init_fn):
        """Apply custom initialization to the encoder and decoder."""
        for p in self.parameters():
            if p.dim() > 1:
                init_fn(p)
            else:
                # Bias use uniform initialization
                nn.init.uniform_(p, -1.0, 1.0)

    def forward(self, graph: Dict) -> List[Dict]:
        """Torch forward model."""
        input_data = graph['edges']
        input_rot_quant_1 = input_data[:, self.rot_feat_ind[0]]
        assert torch.isclose(torch.norm(input_rot_quant_1, dim=1), \
                torch.ones_like(torch.norm(input_rot_quant_1, dim=1))).all()
        input_rot_1 = quaternion_to_matrix(input_rot_quant_1).view(-1, 9)
        input_rot_quant_2 = input_data[:, self.rot_feat_ind[1]]
        assert torch.isclose(torch.norm(input_rot_quant_2, dim=1), \
                torch.ones_like(torch.norm(input_rot_quant_2, dim=1))).all()
        input_rot_2 = quaternion_to_matrix(input_rot_quant_2).view(-1, 9)
        # additionally compute the relative rotation
        rel_rot = torch.matmul(input_rot_1.view(-1, 3, 3), input_rot_2.view(-1, 3, 3).transpose(1, 2))
        rel_rot = rel_rot.view(-1, 9)
        input_rot = torch.cat([input_rot_1, input_rot_2, rel_rot], dim=1)

        input_non_rot_1 = input_data[:, self.non_rot_feat_inx[0]]
        input_non_rot_2 = input_data[:, self.non_rot_feat_inx[1]]
        rel_non_rot = input_non_rot_1 - input_non_rot_2
        input_feat = torch.cat([input_non_rot_1, input_non_rot_2, rel_non_rot], dim=1)
        encoded_non_rot = self.encoder(input_feat)
        encoded_rot = self.encoder_rot(input_rot)
        encoded = torch.cat([encoded_non_rot, encoded_rot], dim=1)
        decoded = self.decoder(encoded)

        output_graph = []
        if self.arity == 1:
            replacements = {
                'nodes': decoded
            }
        elif self.arity == 2:
            replacements = {
                'edges': decoded
            }
        output_graph.append(replace_graph(graph, replacements))

        return output_graph

class SelectiveRelOnlyEncodeDecodeMLP(nn.Module, abc.ABC):
    """Encode-decode architecture."""

    def __init__(self, arity: int, \
                 encoder: nn.Module, \
                 decoder: nn.Module, \
                 non_rot_feat: Dict, \
                 rot_feat: Dict, \
                 init_fn) -> None:
        super().__init__()
        assert arity == 2, "Only support arity == 2 for relative feature"
        self.non_rot_feat_inx = []
        self.rot_feat_ind = []
        for i in range(1, arity+1):
            local_rot = [None] * 4
            for feat in rot_feat.keys():
                # note that real part is the first element
                if "ent{}".format(i) in feat and "_qw" in feat:
                    local_rot[0] = rot_feat[feat]
                elif "ent{}".format(i) in feat and "_qx" in feat:
                    local_rot[1] = rot_feat[feat]
                elif "ent{}".format(i) in feat and "_qy" in feat:
                    local_rot[2] = rot_feat[feat]
                elif "ent{}".format(i) in feat and "_qz" in feat:
                    local_rot[3] = rot_feat[feat]
            self.rot_feat_ind.append(local_rot)
            # non rot also compute relative
            local_non_root = []
            for feat in non_rot_feat:
                if "ent{}".format(i) in feat:
                    local_non_root.append(non_rot_feat[feat])
                elif "ent{}".format(i) in feat:
                    local_non_root.append(non_rot_feat[feat])
                elif "ent{}".format(i) in feat:
                    local_non_root.append(non_rot_feat[feat])
                elif "ent{}".format(i) in feat:
                    local_non_root.append(non_rot_feat[feat])
            assert len(local_non_root) == 3, "Only support 3 non-rot features (x,y,z)"
            self.non_rot_feat_inx.append(local_non_root)
        self.arity = arity
        self.encoder = encoder
        self.decoder = decoder
        self.apply_init(init_fn)

    def apply_init(self, init_fn):
        """Apply custom initialization to the encoder and decoder."""
        for p in self.parameters():
            if p.dim() > 1:
                init_fn(p)
            else:
                # Bias use uniform initialization
                nn.init.uniform_(p, -1.0, 1.0)

    def forward(self, graph: Dict) -> List[Dict]:
        """Torch forward model."""
        input_data = graph['edges']
        input_rot_quant_1 = input_data[:, self.rot_feat_ind[0]]
        assert torch.isclose(torch.norm(input_rot_quant_1, dim=1), \
                torch.ones_like(torch.norm(input_rot_quant_1, dim=1))).all()
        input_rot_1 = quaternion_to_matrix(input_rot_quant_1).view(-1, 3, 3)
        input_rot_quant_2 = input_data[:, self.rot_feat_ind[1]]
        assert torch.isclose(torch.norm(input_rot_quant_2, dim=1), \
                torch.ones_like(torch.norm(input_rot_quant_2, dim=1))).all()
        input_rot_2 = quaternion_to_matrix(input_rot_quant_2).view(-1, 3, 3)
        # additionally compute the relative rotation

        input_trans_1 = input_data[:, self.non_rot_feat_inx[0]]
        input_trans_2 = input_data[:, self.non_rot_feat_inx[1]]
        
        input_pose_1 = torch.zeros((input_trans_1.shape[0], 4, 4)).to(input_trans_1.device)
        input_pose_1[:, :3, :3] = input_rot_1
        input_pose_1[:, :3, 3] = input_trans_1
        input_pose_1[:, 3, 3] = 1.0

        input_pose_2 = torch.zeros((input_trans_2.shape[0], 4, 4)).to(input_trans_2.device)
        input_pose_2[:, :3, :3] = input_rot_2
        input_pose_2[:, :3, 3] = input_trans_2
        input_pose_2[:, 3, 3] = 1.0

        # rel pose
        rel_pose = torch.matmul(input_pose_1, input_pose_2.inverse())
        rel_pose = rel_pose.view(-1, 16)

        encoded = self.encoder(rel_pose)
        decoded = self.decoder(encoded)

        output_graph = []
        if self.arity == 1:
            replacements = {
                'nodes': decoded
            }
        elif self.arity == 2:
            replacements = {
                'edges': decoded
            }
        output_graph.append(replace_graph(graph, replacements))

        return output_graph
    
    
def _setup_neupi_dims(dataset: GraphTransC2DDataset) -> List[int]:
    """Extract dimensions from dataset."""
    input_sample = dataset[0]['input']
    output_sample = dataset[0]['target']
    input_node_dim = input_sample['nodes'].shape[-1]
    input_edge_dim = input_sample['edges'].shape[-1]
    if 'globals' in input_sample:
        if input_sample['globals'] is not None:
            input_global_dim = input_sample['globals'].shape[-1]
        else:  
            input_global_dim = 0
    else:
        input_global_dim = 0
    output_node_dim = output_sample['nodes'].shape[-1]
    output_edge_dim = output_sample['edges'].shape[-1]
    if 'globals' in output_sample:
        if output_sample['globals'] is not None:
            output_global_dim = output_sample['globals'].shape[-1]
        else:
            output_global_dim = 0
    else:
        output_global_dim = 0
    return [
        input_node_dim, input_edge_dim, input_global_dim, output_node_dim,
        output_edge_dim, output_global_dim
    ]

def setup_neupi_mlp_net(graph_dataset: GraphTransC2DDataset, 
                    pred_arity: int,
                    archi: Dict,
                    node_feat2inx: Dict,
                    edge_feat2inx: Dict,
                    dims: List[int] = None
                    ) -> EncodeDecodeMLP:
    """Create an EncodeProcessDecode GNN using the dimensions found in the
    dataset."""
    if dims is None:
        dims = _setup_neupi_dims(graph_dataset)
    else:
        assert graph_dataset is None, "dims and graph_dataset cannot be both provided"
    if pred_arity == 1:
        input_dim = dims[0]
    elif pred_arity == 2:
        input_dim = dims[1]
    else:
        raise ValueError(f"Unsupported pred_arity: {pred_arity}")
    if archi["type"] == "MLP":
        assert "layer_size" in archi
        layer_size = archi["layer_size"]
        encoder_model = MLP([layer_size, layer_size*2], input_dim, True)
        decoder_model = MLP([layer_size*2, layer_size, 1], layer_size*2, False)
        # created a connect mlp with encoder and decoder names
        if 'initializer' in archi:
            init_fn = get_init_fn(archi['initializer'])
        else:
            logging.warning("No initializer specified, using default (kaiming).")
            init_fn = get_init_fn('kaiming')
        connected_mlp = EncodeDecodeMLP(pred_arity, encoder_model, decoder_model, init_fn)
        return connected_mlp
    elif archi["type"] == "Selective-PosRot-MLP":
        assert "feature_names" in archi
        assert "layer_size" in archi
        layer_size = archi["layer_size"]
        rot_ind = {}
        non_rot_ind = []
        if pred_arity == 1:
            for feat in node_feat2inx.keys():
                if feat in archi["feature_names"]:
                    if '_q' in feat:
                        rot_ind[feat] = node_feat2inx[feat]
                    else:
                        non_rot_ind.append(node_feat2inx[feat])
        elif pred_arity == 2:
            for feat in edge_feat2inx.keys():
                if feat in archi["feature_names"]:
                    if '_q' in feat:
                        rot_ind[feat] = edge_feat2inx[feat]
                    else:
                        non_rot_ind.append(edge_feat2inx[feat])
        input_dim = len(non_rot_ind)
        rot_input_dim = 9 if pred_arity == 1 else 18
        encoder_model = MLP([layer_size, layer_size*2], input_dim, True)
        encoder_model_rot = MLP([layer_size, layer_size*2], rot_input_dim, True)
        decoder_model = MLP([layer_size*2, layer_size, 1], layer_size*4, False)
        # created a connect mlp with encoder and decoder names
        if 'initializer' in archi:
            init_fn = get_init_fn(archi['initializer'])
        else:
            logging.warning("No initializer specified, using default (kaiming).")
            init_fn = get_init_fn('kaiming')
        connected_selective_mlp = SelectiveRotEncodeDecodeMLP(pred_arity, 
                                encoder_model, encoder_model_rot,
                                decoder_model, non_rot_ind, rot_ind, init_fn)
        return connected_selective_mlp
    elif archi["type"] == "Selective-PosRelRot-MLP":
        assert "feature_names" in archi
        assert "layer_size" in archi
        assert pred_arity == 2, "Only support pred_arity == 2 for relative feature"
        layer_size = archi["layer_size"]
        rot_ind = {}
        non_rot_ind = {}
        for feat in edge_feat2inx.keys():
            if feat in archi["feature_names"]:
                if '_q' in feat:
                    rot_ind[feat] = edge_feat2inx[feat]
                else:
                    non_rot_ind[feat] = edge_feat2inx[feat]
        assert len(non_rot_ind) % 2 == 0, "Only support even number of non-rot features"
        input_dim = len(non_rot_ind) * 3 // 2 # additional 0.5 for relative rot
        rot_input_dim = 27 # additional 9 for relative rot
        encoder_model = MLP([layer_size, layer_size*2], input_dim, True)
        encoder_model_rot = MLP([layer_size, layer_size*2], rot_input_dim, True)
        decoder_model = MLP([layer_size*2, layer_size, 1], layer_size*4, False)
        # created a connect mlp with encoder and decoder names
        if 'initializer' in archi:
            init_fn = get_init_fn(archi['initializer'])
        else:
            logging.warning("No initializer specified, using default (kaiming).")
            init_fn = get_init_fn('kaiming')
        connected_selective_mlp = SelectiveRelRotEncodeDecodeMLP(pred_arity, 
                                encoder_model, encoder_model_rot,
                                decoder_model, non_rot_ind, rot_ind, init_fn)
        return connected_selective_mlp
    elif archi["type"] == "Selective-RelOnly-MLP":
        assert "feature_names" in archi
        assert "layer_size" in archi
        assert pred_arity == 2, "Only support pred_arity == 2 for relative feature"
        layer_size = archi["layer_size"]
        rot_ind = {}
        non_rot_ind = {}
        for feat in edge_feat2inx.keys():
            if feat in archi["feature_names"]:
                if '_q' in feat:
                    rot_ind[feat] = edge_feat2inx[feat]
                else:
                    non_rot_ind[feat] = edge_feat2inx[feat]
        assert len(non_rot_ind) % 2 == 0, "Only support even number of non-rot features"
        input_dim = len(non_rot_ind) // 2 # 0.5 for relative pose
        rot_input_dim = 9 # 9 for relative rot
        encoder_model = MLP([layer_size, layer_size*2], input_dim, True)
        encoder_model_rot = MLP([layer_size, layer_size*2], rot_input_dim, True)
        decoder_model = MLP([layer_size*2, layer_size, 1], layer_size*4, False)
        # created a connect mlp with encoder and decoder names
        if 'initializer' in archi:
            init_fn = get_init_fn(archi['initializer'])
        else:
            logging.warning("No initializer specified, using default (kaiming).")
            init_fn = get_init_fn('kaiming')
        connected_selective_mlp = SelectiveRelOnlyEncodeDecodeMLP(pred_arity, 
                                encoder_model, encoder_model_rot,
                                decoder_model, non_rot_ind, rot_ind, init_fn)
        return connected_selective_mlp
    elif archi["type"] == "Selective-RelPoseOnly-MLP":
        assert "feature_names" in archi
        assert "layer_size" in archi
        assert pred_arity == 2, "Only support pred_arity == 2 for relative feature"
        layer_size = archi["layer_size"]
        rot_ind = {}
        non_rot_ind = {}
        for feat in edge_feat2inx.keys():
            if feat in archi["feature_names"]:
                if '_q' in feat:
                    rot_ind[feat] = edge_feat2inx[feat]
                else:
                    non_rot_ind[feat] = edge_feat2inx[feat]
        assert len(non_rot_ind) % 2 == 0, "Only support even number of non-rot features"
        input_dim = 16 # 0.5 for relative pose
        encoder_model = MLP([layer_size, layer_size*2], input_dim, True)
        decoder_model = MLP([layer_size*2, layer_size, 1], layer_size*2, False)
        # created a connect mlp with encoder and decoder names
        if 'initializer' in archi:
            init_fn = get_init_fn(archi['initializer'])
        else:
            logging.warning("No initializer specified, using default (kaiming).")
            init_fn = get_init_fn('kaiming')
        connected_selective_mlp = SelectiveRelOnlyEncodeDecodeMLP(pred_arity, 
                                encoder_model, decoder_model, non_rot_ind, rot_ind, init_fn)
        return connected_selective_mlp
    elif archi["type"] == "Selective-MLP":
        assert "feature_names" in archi
        assert "layer_size" in archi
        layer_size = archi["layer_size"]
        feat_ind = []
        assert pred_arity == 1, "Only support pred_arity == 1 for selectiveMLP"
        for feat in node_feat2inx.keys():
            if feat in archi["feature_names"]:
                feat_ind.append(node_feat2inx[feat])
        input_dim = len(feat_ind)
        encoder_model = MLP([layer_size, layer_size*2], input_dim, True)
        decoder_model = MLP([layer_size*2, layer_size, 1], layer_size*2, False)
        # created a connect mlp with encoder and decoder names
        if 'initializer' in archi:
            init_fn = get_init_fn(archi['initializer'])
        else:
            logging.warning("No initializer specified, using default (kaiming).")
            init_fn = get_init_fn('kaiming')
        connected_selective_mlp = SelectiveEncodeDecodeMLP(pred_arity, 
                                encoder_model, decoder_model,
                                feat_ind, init_fn)
        return connected_selective_mlp
    else:
        raise ValueError(f"Unsupported architecture type: {archi['type']}")

def get_init_fn(init_type):
    """Function to retrieve different initialization functions."""
    if init_type == 'xavier':
        return nn.init.xavier_uniform_
    elif init_type == 'kaiming':
        return nn.init.kaiming_uniform_
    elif init_type == 'uniform':
        return nn.init.uniform_
    else:
        raise ValueError(f"Unknown initializer: {init_type}")
    

def setup_neupi_optimizer(model: nn.Module, opti_config: Dict, lr_scheduler_config: Dict = None) \
    -> torch.optim.Optimizer:
    # Choose the optimizer
    if opti_config['type'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), \
                                     **opti_config['kwargs'])
    elif opti_config['type'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), \
                                      **opti_config['kwargs'])
    else:
        raise ValueError(f"Unsupported optimizer type: {opti_config['type']}")
    
    # Learning rate scheduler setup
    if lr_scheduler_config is not None:
        if lr_scheduler_config['type'] == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, \
                                **lr_scheduler_config['kwargs'])
        elif lr_scheduler_config['type'] == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, \
                                **lr_scheduler_config['kwargs'])
        elif lr_scheduler_config['type'] == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, \
                                **lr_scheduler_config['kwargs'])
        elif lr_scheduler_config['type'] == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,  \
                                **lr_scheduler_config['kwargs'])
        else:
            raise ValueError(f"Unsupported scheduler type: {lr_scheduler_config['type']}")
    else:
        scheduler = None
    
    return optimizer, scheduler

        

class MCTSNode:
    def __init__(self, state, level, parent=None):
        self.state = state          # The game state at this node
        self.parent = parent        # Parent node
        self.level = level          # Level in the tree
        self.visits = 0             # Number of times node has been visited
        self.guidance = np.zeros_like(state) - 1  # Guidance values init
        self.value = 0       # Total reward for this node
        self.next_states = set()  # Set of next states to avoid duplicates
        for i in range(len(self.state)):
            # next state only flip one "0" to "1 or "2"
            if self.state[i] == 0:
                for new_val in [1, 2]:  # Only allow 1 or 2
                    next_state = self.state.copy()
                    next_state[i] = new_val
                    self.next_states.add(tuple(next_state))  # Store as tuple for set

    def is_fully_expanded(self):
        return len(self.next_states) == 0
    
    def update_value(self, zero_loss, thresh):
        non_zero_indexes = np.where(self.state != 0)[0]
        non_zero_guidance = self.guidance[non_zero_indexes]
        # note that maximum value is 0.5
        # do not skip root
        if (non_zero_guidance.mean() < 0.4) or (non_zero_indexes.size == 0):
            local_zero_loss = zero_loss.copy()
            # non-zero indexes
            local_zero_loss[non_zero_indexes] = 0.0
            # count how large the loss of "zero" cells are, larger the more likely to be selected
            self.value = local_zero_loss.mean()
            if self.guidance.mean() > thresh:
                # It is far from the threshold, then large value
                self.value += self.guidance.mean() - thresh
        else:
            # non-zero loss is too large, wrong element wise supervision
            self.value = -np.inf

    def expand(self, zero_loss, rng: np.random.Generator):
        # Expand up to batch_size untried actions
        next_state = self.get_successor(zero_loss, rng)
        return MCTSNode(next_state, level=self.level+1, parent=self)

    def get_successor(self, zero_loss, rng: np.random.Generator):
        local_zero_loss = zero_loss.copy()
        # non-zero indexes
        non_zero_indexes = np.where(self.state != 0)[0]
        local_zero_loss[non_zero_indexes] = -1
        # try to find a max zero loss index that has not been tried
        while not (local_zero_loss==-1).all():
            assert not self.is_fully_expanded()
            max_zero_loss_idx = np.argmax(local_zero_loss)
            next_state = self.state.copy()
            next_state_ = self.state.copy()
            next_state[max_zero_loss_idx] = 1
            next_state_[max_zero_loss_idx] = 2
            if array_in_list(next_state, self.next_states):
                self.next_states.remove(tuple(next_state))
                return next_state
            elif array_in_list(next_state_, self.next_states):
                self.next_states.remove(tuple(next_state_))
                return next_state_
            else:
                # this cell has been tried, set it to -1
                local_zero_loss[max_zero_loss_idx] = -1
        if not self.is_fully_expanded():
            # randomly select one
            list_next_states = list(self.next_states)
            list_idx = rng.choice(len(list_next_states))
            next_state = np.array(list_next_states[list_idx])
            self.next_states.remove(tuple(next_state))
            return next_state
        else:
            raise ValueError("Node is fully expanded. Should not be considered")

class HierachicalMCTSearcher(abc.ABC):
    """Uniformally sample the vectors."""

    def __init__(self, 
                 dim_num: int, \
                 frontier_max_level: int, \
                 guidance_th: float,
                 rng: np.random.Generator) -> None:
        self.visits = 0  # Total visits for UCT calculation
        self._rng = rng
        self.guidance_th = min(guidance_th * dim_num / 2, 0.5)
        self.frontier_max_level = frontier_max_level
        root = MCTSNode(np.zeros(dim_num, dtype=np.int32), 0)
        root.guidance = np.zeros(root.state.shape)
        self.global_zero_loss = np.zeros(root.state.shape)
        root.update_value(self.global_zero_loss, self.guidance_th)
        self.frontier = [root]
        self.evaluated_values = {tuple(root.state)}

    def uct_selection(self, nodes, bs):
        # Use NumPy for efficient computation of UCT values
        visits = np.array([node.visits for node in nodes])
        values = np.array([node.value for node in nodes])
        
        # Avoid division by zero by setting a small value for nodes with zero visits
        visits = np.where(visits == 0, 1e-5, visits)
        
        total_visits = np.log(self.visits + 1)
        
        # UCT calculation
        uct_values = values / visits + 14.1 * self.guidance_th * np.sqrt(total_visits / visits)
        
        # Select top bs nodes by UCT values
        order = np.argsort(uct_values)[-bs:][::-1]
        
        # Return the top bs nodes in descending order of UCT values
        selected_nodes = [nodes[i] for i in order]
        
        return selected_nodes
    
    def update_value(self, state, guidance):
        # Update zero loss
        if (guidance != np.inf).all():
            # satisfiable
            zero_mask = state == 0
            self.global_zero_loss[zero_mask] += guidance[zero_mask]
            for node in self.frontier:
                if (node.state == state).all():
                    node.guidance = guidance.copy()
                node.update_value(self.global_zero_loss / self.visits, self.guidance_th)
        else:
            # unsatisfiable
            for node in self.frontier:
                if (node.state == state).all():
                    node.guidance = np.ones_like(node.guidance)
                    node.update_value(self.global_zero_loss / self.visits, self.guidance_th)
        
        self.update_front()

    def update_front(self):
        new_frontier = []
        for node in self.frontier:
            if (not node.is_fully_expanded()) and (node.value > -np.inf):
                new_frontier.append(node)
            else:
                logging.info(f"Node {node.state} is fully expanded or has -inf value.")
        if len(new_frontier) != len(self.frontier):
            logging.info(f"New Frontier: {[node.state for node in new_frontier]}")
        self.frontier = new_frontier
    
    def propose(self):
        # propose the next node to evaluate
        self.update_front()
        if len(self.frontier) == 0:
            return None
        else:
            # 1. Select a parent node using UCT
            while True:
                self.update_front()
                if self.frontier == []:
                    return None
                self.visits += 1
                selected_nodes = self.uct_selection(self.frontier, 1)
                # 2. Expand the selected nodes, add to frontier, and evaluate
                for node in selected_nodes:
                    assert not node.is_fully_expanded()
                    assert node.value > -np.inf
                    node.visits += 1
                    child = node.expand(self.global_zero_loss, self._rng)
                    if tuple(child.state) not in self.evaluated_values:
                        self.evaluated_values.add(tuple(child.state))
                        if child.level <= self.frontier_max_level:
                            self.frontier.append(child)
                        return child.state