# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp

from models.BaseModel import GeneralModel, CTRModel
from models.BaseContextModel import ContextCTRModel
from models.BaseImpressionModel import ImpressionModel

import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

class DynamicalGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=8):
        super().__init__()
        self.k = k
        self.conv = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        

    def knn_graph(self, x, k):
        # x: [batch, channels, num_nodes]
        batch_size, channels, num_nodes = x.shape
        x = x.transpose(2, 1).contiguous()  # [batch, num_nodes, channels]
        
        inner = -2 * torch.matmul(x, x.transpose(2, 1))  # [batch, num_nodes, num_nodes]
        xx = torch.sum(x**2, dim=2, keepdim=True)  # [batch, num_nodes, 1]
        pairwise_distance = -xx - inner - xx.transpose(2, 1)  # [batch, num_nodes, num_nodes]
        
        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # [batch, num_nodes, k]
        return idx
    
    def get_graph_feature(self, x, k, idx=None):
        batch_size, channels, num_nodes = x.shape
        
        if idx is None:
            idx = self.knn_graph(x, k)  # [batch, num_nodes, k]
        
        device = x.device
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_nodes
        idx = idx + idx_base
        idx = idx.view(-1)
        
        x = x.transpose(2, 1).contiguous()  # [batch, num_nodes, channels]
        feature = x.view(batch_size * num_nodes, -1)[idx, :]  # [batch*num_nodes*k, channels]
        feature = feature.view(batch_size, num_nodes, k, channels)  # [batch, num_nodes, k, channels]
        
        x = x.view(batch_size, num_nodes, 1, channels).repeat(1, 1, k, 1)  # [batch, num_nodes, k, channels]
        
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()  # [batch, 2*channels, num_nodes, k]
        
        return feature
        
    def forward(self, x):
        # x: [batch, channels, num_nodes]
        x = self.get_graph_feature(x, self.k)  # [batch, 2*channels, num_nodes, k]
        x = self.conv(x)  # [batch, out_channels, num_nodes, k]
        x = self.bn(x)
        x = F.relu(x)
        x = x.max(dim=-1, keepdim=False)[0]  # [batch, out_channels, num_nodes]
        return x

class EEGDGCNNEncoder(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=32, out_channels=16, k=8):
        super().__init__()
        self.k = k
        
        self.dgcnn1 = DynamicalGraphConv(in_channels, hidden_channels, k)
        self.dgcnn2 = DynamicalGraphConv(hidden_channels, out_channels, k)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # x: [batch, 62, 5]
        batch_size, num_nodes, in_channels = x.shape
        x = x.transpose(2, 1).contiguous().float()  # [batch, 5, 62]
        
        x = self.dgcnn1(x)  # [batch, hidden_channels, 62]
        x = self.dgcnn2(x)  # [batch, out_channels, 62]
        
        x = self.global_pool(x)  # [batch, out_channels, 1]
        x = x.squeeze(-1)  # [batch, out_channels]
        
        return x

class EEGGCNEncoder(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=32, out_channels=16, edge_index=None):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, out_channels)
        self.edge_index = edge_index  # [2, num_edges]

    def forward(self, x):
        # x: [batch, 62, 5]
        batch_size, num_nodes, in_channels = x.shape
        x = x.reshape(-1, in_channels).float()  # [batch*62, 5]
        edge_index = self.edge_index.to(x.device)
        edge_indices = []
        for i in range(batch_size):
            batch_edge_index = edge_index + i * num_nodes
            edge_indices.append(batch_edge_index)
        edge_index = torch.cat(edge_indices, dim=1)
        batch = torch.arange(batch_size, device=x.device).repeat_interleave(num_nodes)
        x = self.gcn1(x, edge_index)
        x = torch.relu(x)
        x = self.gcn2(x, edge_index)
        x = global_mean_pool(x, batch)  # [batch, out_channels]
        return x

def build_fully_connected_edge_index(num_nodes):
    row = []
    col = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                row.append(i)
                col.append(j)
    edge_index = torch.tensor([row, col], dtype=torch.long)
    return edge_index

def build_eeg_topology_edge_index():
    channels = [
        "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
        "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ",
        "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7",
        "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6",
        "PO8", "CB1", "O1", "OZ", "O2", "CB2"
    ]
    
    name_to_idx = {name: idx for idx, name in enumerate(channels)}
    
    connections = [
        ("FP1", "FPZ"), ("FPZ", "FP2"), ("FP1", "AF3"), ("FP2", "AF4"),
        ("AF3", "F3"), ("AF4", "F4"),
        
        ("F7", "F5"), ("F5", "F3"), ("F3", "F1"), ("F1", "FZ"), ("FZ", "F2"), 
        ("F2", "F4"), ("F4", "F6"), ("F6", "F8"),
        ("F7", "FT7"), ("F3", "FC3"), ("F1", "FC1"), ("FZ", "FCZ"), ("F2", "FC2"), 
        ("F4", "FC4"), ("F8", "FT8"),
        
        ("FT7", "FC5"), ("FC5", "FC3"), ("FC3", "FC1"), ("FC1", "FCZ"), ("FCZ", "FC2"), 
        ("FC2", "FC4"), ("FC4", "FC6"), ("FC6", "FT8"),
        ("FT7", "T7"), ("FC5", "C5"), ("FC3", "C3"), ("FC1", "C1"), ("FCZ", "CZ"), 
        ("FC2", "C2"), ("FC4", "C4"), ("FC6", "C6"), ("FT8", "T8"),
        
        ("T7", "C5"), ("C5", "C3"), ("C3", "C1"), ("C1", "CZ"), ("CZ", "C2"), 
        ("C2", "C4"), ("C4", "C6"), ("C6", "T8"),
        ("T7", "TP7"), ("C5", "CP5"), ("C3", "CP3"), ("C1", "CP1"), ("CZ", "CPZ"), 
        ("C2", "CP2"), ("C4", "CP4"), ("C6", "CP6"), ("T8", "TP8"),
        
        ("TP7", "CP5"), ("CP5", "CP3"), ("CP3", "CP1"), ("CP1", "CPZ"), ("CPZ", "CP2"), 
        ("CP2", "CP4"), ("CP4", "CP6"), ("CP6", "TP8"),
        ("TP7", "P7"), ("CP5", "P5"), ("CP3", "P3"), ("CP1", "P1"), ("CPZ", "PZ"), 
        ("CP2", "P2"), ("CP4", "P4"), ("CP6", "P6"), ("TP8", "P8"),
        
        ("P7", "P5"), ("P5", "P3"), ("P3", "P1"), ("P1", "PZ"), ("PZ", "P2"), 
        ("P2", "P4"), ("P4", "P6"), ("P6", "P8"),
        ("P7", "PO7"), ("P5", "PO5"), ("P3", "PO3"), ("PZ", "POZ"), ("P4", "PO4"), 
        ("P6", "PO6"), ("P8", "PO8"),
        
        ("PO7", "PO5"), ("PO5", "PO3"), ("PO3", "POZ"), ("POZ", "PO4"), ("PO4", "PO6"), ("PO6", "PO8"),
        ("PO7", "CB1"), ("PO3", "O1"), ("POZ", "OZ"), ("PO4", "O2"), ("PO8", "CB2"),
        
        ("CB1", "O1"), ("O1", "OZ"), ("OZ", "O2"), ("O2", "CB2"),
    ]
    
    row, col = [], []
    for conn in connections:
        if conn[0] in name_to_idx and conn[1] in name_to_idx:
            idx1, idx2 = name_to_idx[conn[0]], name_to_idx[conn[1]]
            row.extend([idx1, idx2])
            col.extend([idx2, idx1])
    
    edge_index = torch.tensor([row, col], dtype=torch.long)
    return edge_index


class LightGCNBase(nn.Module):
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--n_layers', type=int, default=3,
							help='Number of LightGCN layers.')
		return parser
	
	@staticmethod
	def build_adjmat(user_count, item_count, train_mat, selfloop_flag=False):
		R = sp.dok_matrix((user_count, item_count), dtype=np.float32)
		for user in train_mat:
			for item in train_mat[user]:
				R[user, item] = 1
		R = R.tolil()
        
		adj_mat = sp.dok_matrix((user_count + item_count, user_count + item_count), dtype=np.float32)
		adj_mat = adj_mat.tolil()
		adj_mat[:user_count, user_count:] = R
		adj_mat[user_count:, :user_count] = R.T
		adj_mat = adj_mat.todok()
       
		def normalized_adj_single(adj):
			rowsum = np.array(adj.sum(1)) + 1e-10

			d_inv_sqrt = np.power(rowsum, -0.5).flatten()
			d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
			d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

			bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
			return bi_lap.tocoo()
        
		if selfloop_flag:
			norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
		else:
			norm_adj_mat = normalized_adj_single(adj_mat)
        
		return norm_adj_mat.tocsr()

	def _base_init(self, args, corpus):
		super().__init__()
		self.emb_size = args.emb_size
		self.n_layers = args.n_layers
		self.norm_adj = self.build_adjmat(corpus.n_users, corpus.n_items, corpus.train_clicked_set)
		self._base_define_params()
		self.user_meta_data = corpus.user_features
		self.item_meta_data = corpus.item_features
		self.context_encoder = nn.Linear(1,16)
		self.fc = nn.Sequential(
            nn.Linear(64 + 64 + 16*7, 64),  # 64(user) + 64(item) + 16*6(context features) + 16(EEG)
            nn.ReLU(),
            nn.Linear(64, 1)
        )
		self.apply(self.init_weights)
	
	def _base_define_params(self):
		self.encoder = LGCNEncoder(self.user_num, self.item_num, self.emb_size, self.norm_adj, self.n_layers, self.device)
    
	def forward(self, feed_dict):
		self.check_list = []
		self.context_emb = []
		for key in feed_dict:
			if key[:2]=='c_' and key[2:5] != 'EEG' and key[2:9] != 'session' and key[2:6] != 'view' and key != 'c_video_order_f':
			# if key == 'c_interest_f' or key == 'c_immersion_f' or key == 'c_valence_f' or key == 'c_arousal_f':
			# if key == 'c_playrate_f' or key == 'c_video_type_c' or key == 'u_gender_c' or key == 'u_age_f':
				context2d = feed_dict[key].unsqueeze(1) 
				context16 = self.context_encoder(context2d.float())
				self.context_emb.append(context16)
		user, items = feed_dict['user_id'], feed_dict['item_id']
		u_embed, i_embed = self.encoder(user, items)
		i_embed = i_embed.squeeze(dim=1)
		eeg_raw = feed_dict['c_EEG_data_310_f']  # [batch, 310]
		eeg_feat = eeg_raw.view(-1, 62, 5)       # [batch, 62, 5]
		eeg_emb = self.eeg_dgcnn_encoder(eeg_feat) # [batch, 16]
		combined = torch.cat([u_embed, i_embed, eeg_emb], dim=-1)  # [batch, 64+64+16]
		# combined = torch.cat([u_embed, i_embed], dim=-1)  # [batch, 64+64+16]
		for feature in self.context_emb:
			combined = torch.cat([combined, feature], dim=-1)
		prediction = self.fc(combined).squeeze(-1) # [batch_size]
	

		# prediction = (u_embed[:, None, :] * i_embed).sum(dim=-1)  # [batch_size, -1]
		u_v = u_embed.repeat(1,items.shape[1]).view(items.shape[0],items.shape[1],-1)
		i_v = i_embed
		return {'prediction': prediction.view(feed_dict['batch_size'], -1), 'u_v': u_v, 'i_v':i_v}

class LightGCN(GeneralModel, LightGCNBase):
	reader = 'BaseReader'
	runner = 'BaseRunner'
	extra_log_args = ['emb_size', 'n_layers', 'batch_size']

	@staticmethod
	def parse_model_args(parser):
		parser = LightGCNBase.parse_model_args(parser)
		return GeneralModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		GeneralModel.__init__(self, args, corpus)
		self._base_init(args, corpus)
		# EEG GCN
		self.eeg_num_nodes = 62
		self.eeg_edge_index = build_fully_connected_edge_index(self.eeg_num_nodes)
		self.eeg_gcn_encoder = EEGGCNEncoder(in_channels=5, hidden_channels=32, out_channels=16, edge_index=self.eeg_edge_index)

	def forward(self, feed_dict):
		out_dict = LightGCNBase.forward(self, feed_dict)
		return {'prediction': out_dict['prediction']}

# EEG-DGCN
class CTRLightGCNCTR(ContextCTRModel, LightGCNBase):
	reader = 'ContextReader'
	runner = 'CTRRunner'
	extra_log_args = ['emb_size', 'n_layers', 'batch_size']

	@staticmethod
	def parse_model_args(parser):
		parser = LightGCNBase.parse_model_args(parser)
		return ContextCTRModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ContextCTRModel.__init__(self, args, corpus)
		self._base_init(args, corpus)
		self.loss_fn = nn.BCELoss()
		self.eeg_num_nodes = 62
		self.eeg_edge_index = build_fully_connected_edge_index(self.eeg_num_nodes)
		self.eeg_gcn_encoder = EEGGCNEncoder(in_channels=5, hidden_channels=32, out_channels=16, edge_index=self.eeg_edge_index)
		self.eeg_dgcnn_encoder = EEGDGCNNEncoder(in_channels=5, hidden_channels=32, out_channels=16, k=8)

	def forward(self, feed_dict):
		out_dict = LightGCNBase.forward(self, feed_dict)
		out_dict['prediction'] = out_dict['prediction'].view(-1).sigmoid()
		out_dict['label'] = feed_dict['label'].view(-1)
		return out_dict

class LightGCNImpression(ImpressionModel, LightGCNBase):
	reader = 'ImpressionReader'
	runner = 'ImpressionRunner'
	extra_log_args = ['emb_size', 'n_layers', 'batch_size']

	@staticmethod
	def parse_model_args(parser):
		parser = LightGCNBase.parse_model_args(parser)
		return ImpressionModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ImpressionModel.__init__(self, args, corpus)
		self._base_init(args, corpus)

	def forward(self, feed_dict):
		return LightGCNBase.forward(self, feed_dict)

class LGCNEncoder(nn.Module):
	def __init__(self, user_count, item_count, emb_size, norm_adj, n_layers=3, device=None):
		super(LGCNEncoder, self).__init__()
		self.user_count = user_count
		self.item_count = item_count
		self.emb_size = emb_size
		self.layers = [emb_size] * n_layers
		self.norm_adj = norm_adj
		self.device = device

		self.embedding_dict = self._init_model()
		self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)
    
	def _init_model(self):
		initializer = nn.init.xavier_uniform_
		embedding_dict = nn.ParameterDict({
			'user_emb': nn.Parameter(initializer(torch.empty(self.user_count, self.emb_size))),
			'item_emb': nn.Parameter(initializer(torch.empty(self.item_count, self.emb_size))),
		})
		return embedding_dict

	@staticmethod
	def _convert_sp_mat_to_sp_tensor(X):
		coo = X.tocoo()
		i = torch.LongTensor([coo.row, coo.col])
		v = torch.from_numpy(coo.data).float()
		return torch.sparse.FloatTensor(i, v, coo.shape)
    
	def forward(self, users, items):
		ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
		all_embeddings = [ego_embeddings]

		for k in range(len(self.layers)):
			ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
			all_embeddings += [ego_embeddings]
        
		all_embeddings = torch.stack(all_embeddings, dim=1)
		all_embeddings = torch.mean(all_embeddings, dim=1)
        
	
		user_all_embeddings = all_embeddings[:self.user_count, :]
		item_all_embeddings = all_embeddings[self.user_count:, :]
        
		user_embeddings = user_all_embeddings[users, :]
		item_embeddings = item_all_embeddings[items, :]

		return user_embeddings, item_embeddings
