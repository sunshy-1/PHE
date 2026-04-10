from .conv import *
from gensim.parsing.preprocessing import *
from collections import defaultdict
from .cython_util import *
from scipy import sparse
from sklearn.mixture import GaussianMixture

import random
import numpy as np
import time
import torch
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
import time


class PROPOSED_GNN(nn.Module):
    def __init__(self, gnn, rem_edge_list, attr_decoder, types, neg_samp_num, device, neg_queue_size = 0, hidden_dim = 400, sample_rate = [2,2,1], nei_num = 3, attn_drop = 0):
        super(PROPOSED_GNN, self).__init__()
        if gnn is None:
            return
        self.types = types
        self.gnn = gnn
        self.params = nn.ModuleList()
        self.neg_queue_size = neg_queue_size
        self.neg_queue1_size= neg_queue_size
        self.link_dec_dict = {}
        self.neg_queue = {}
        self.neg_queue1= {}
        for source_type in rem_edge_list:
            self.link_dec_dict[source_type] = {}
            self.neg_queue[source_type] = {}
            self.neg_queue1[source_type] = {}
            for relation_type in rem_edge_list[source_type]:
                print(source_type, relation_type)
                matcher = Matcher(gnn.n_hid, gnn.n_hid)
                self.neg_queue[source_type][relation_type] = torch.FloatTensor([]).to(device)
                self.neg_queue1[source_type][relation_type] = torch.FloatTensor([]).to(device)
                self.link_dec_dict[source_type][relation_type] = matcher
                self.params.append(matcher)
        
        self.attr_decoder = attr_decoder
        self.init_emb = nn.Parameter(torch.randn(gnn.in_dim))
        self.ce = nn.CrossEntropyLoss(reduction = 'none')
        self.neg_samp_num = neg_samp_num
        self.hidden_dim = hidden_dim
        self.sample_rate = sample_rate
        self.nei_num = nei_num
        self.attn_drop = attn_drop

        self.f_k = nn.Bilinear(gnn.n_hid, gnn.n_hid, 1)
        torch.nn.init.xavier_uniform_(self.f_k.weight.data)
        self.sigm = nn.Sigmoid()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.neg_structure_queue_size = 0
        self.neg_structure_queue = torch.randn(self.neg_structure_queue_size, gnn.n_hid).to(device)
        self.device = device
        self.structure_map_dict = {}
        self.structure_matcher = Matcher(gnn.n_hid, gnn.n_hid)
        
        for source_type in rem_edge_list:
            self.structure_map_dict[source_type] = {}
            for relation_type in rem_edge_list[source_type]:
                structure_map = StructureMapping(gnn.n_hid, gnn.n_hid)
                self.structure_map_dict[source_type][relation_type] = structure_map
                self.params.append(structure_map)
                
        sc_encoder = Sc_encoder(hidden_dim, sample_rate, nei_num, attn_drop, self.device)
        self.sc = sc_encoder
        self.params.append(sc_encoder)
        
    def neg_sample(self, souce_node_list, pos_node_list):
        np.random.shuffle(souce_node_list)
        neg_nodes = negative_sample(souce_node_list, pos_node_list, self.neg_samp_num)
        return neg_nodes
    
    def neg_sample_ori(self, souce_node_list, pos_node_list):
        np.random.shuffle(souce_node_list)
        neg_nodes = []
        keys = {key : True for key in pos_node_list}
        tot  = 0
        for node_id in souce_node_list:
            if node_id not in keys:
                neg_nodes += [node_id]
                tot += 1
            if tot == self.neg_samp_num:
                break     
        return neg_nodes   

    def forward(self, node_feature, node_type, edge_time, edge_index, edge_type):
        return self.gnn(node_feature, node_type, edge_time, edge_index, edge_type)
    
    # def semantic_loss(self, node_emb, rem_edge_list, ori_edge_list, node_dict, target_type):
    #     losses = 0
    #     ress   = []
        
    #     # Clustering Input: node_emb target_tmp_emb
    #     n_cluster = 5
    #     emb_need_clustering = np.array(node_emb.tolist())
    #     gmm = GaussianMixture(n_components=n_cluster)
    #     gmm.fit(emb_need_clustering)
    #     gmm_labels = gmm.predict(emb_need_clustering)
    #     all_labels = list(set(gmm_labels))
    #     all_ids = [i for i in range(gmm_labels.size)]
        
    #     dict_label_posid = {}
    #     for label in all_labels:
    #         dict_label_posid[label] = [i for i, gmm_label in enumerate(gmm_labels.tolist()) if gmm_label==label]

    #     dict_label_negid = {}
    #     for label in all_labels:
    #         dict_label_negid[label] = list(set(all_ids).difference(set(dict_label_posid[label])))

    #     target_tmp_emb = torch.Tensor([]).to(self.device)
    #     for source_type in rem_edge_list:
    #         if source_type not in self.link_dec_dict:
    #             continue
    #         for relation_type in rem_edge_list[source_type]:
    #             if relation_type not in self.link_dec_dict[source_type]:
    #                 continue
    #             rem_edges = rem_edge_list[source_type][relation_type]
    #             if len(rem_edges) <= 8:
    #                 continue

    #             matcher = self.link_dec_dict[source_type][relation_type]
                
    #             target_ids = rem_edges[:,0].reshape(-1,1)
    #             n_nodes = len(target_ids)
    #             n_neg_ids = 255
    #             n_pos_ids = 1
                
    #             negative_source_ids = np.array([\
    #                                     np.random.choice(dict_label_negid[gmm_labels[t_id[0] + node_dict[target_type][0]]], n_neg_ids, replace=True).tolist() \
    #                                         for t_id in target_ids])
    #             positive_source_ids = np.array([\
    #                                     np.random.choice(list(
    #                                         set(dict_label_posid[gmm_labels[t_id[0] + node_dict[target_type][0]]]).difference(\
    #                                         set([t_id[0] + node_dict[target_type][0]]))), n_pos_ids, replace=True).tolist() 
    #                                         for t_id in target_ids])
    #             source_ids = torch.LongTensor(np.concatenate((positive_source_ids, negative_source_ids), axis=-1))
    #             emb = node_emb[source_ids]
    #             rep_size = n_neg_ids + n_pos_ids
    #             source_emb = emb.reshape(source_ids.shape[0] * rep_size, -1)
                
    #             target_ids = target_ids.repeat(rep_size, 1) + node_dict[target_type][0]
    #             target_emb = node_emb[target_ids.reshape(-1)]
    #             res = matcher.forward(target_emb, source_emb)
    #             res = res.reshape(n_nodes, rep_size)
    #             ress += [res.detach()]
    #             losses += F.log_softmax(res, dim=-1)[:,0].mean()

    #     return -losses / len(ress), ress

    def link_loss(self, node_emb, node_emb_perturb, rem_edge_list, ori_edge_list, node_dict, target_type, use_queue = True, update_queue = False):
        losses = 0
        ress   = []
        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    
        for source_type in rem_edge_list:
            if source_type not in self.link_dec_dict:
                continue
            for relation_type in rem_edge_list[source_type]:
                if relation_type not in self.link_dec_dict[source_type]:
                    continue
                rem_edges = rem_edge_list[source_type][relation_type]
                if len(rem_edges) <= 8:
                    continue

                ori_edges = ori_edge_list[source_type][relation_type]
                matcher = self.link_dec_dict[source_type][relation_type]

                target_ids, positive_source_ids = rem_edges[:,0].reshape(-1, 1), rem_edges[:,1].reshape(-1, 1)
                n_nodes = len(target_ids)
                source_node_ids = np.unique(ori_edges[:, 1])

                negative_source_ids = [self.neg_sample(source_node_ids, \
                    ori_edges[ori_edges[:, 0] == t_id][:, 1]) for t_id in target_ids]
                sn = min([len(neg_ids) for neg_ids in negative_source_ids])
                
                negative_source_ids = [neg_ids[:sn] for neg_ids in negative_source_ids]

                source_ids = torch.LongTensor(np.concatenate((positive_source_ids, negative_source_ids), axis=-1) + node_dict[source_type][0])
                
                # used the perturb node embedding               
                emb = node_emb_perturb[source_ids]
                
                if use_queue and len(self.neg_queue[source_type][relation_type]) // n_nodes > 0:
                    tmp = self.neg_queue[source_type][relation_type]
                    stx = len(tmp) // n_nodes
                    tmp = tmp[: stx * n_nodes].reshape(n_nodes, stx, -1)
                    rep_size = sn + 1 + stx
                    source_emb = torch.cat([emb, tmp], dim=1)
                    source_emb = source_emb.reshape(n_nodes * rep_size, -1)
                else:
                    rep_size = sn + 1
                    source_emb = emb.reshape(source_ids.shape[0] * rep_size, -1)
                
                target_ids = target_ids.repeat(rep_size, 1) + node_dict[target_type][0]
                target_emb = node_emb[target_ids.reshape(-1)]
                res = matcher.forward(target_emb, source_emb)
                res = res.reshape(n_nodes, rep_size)
                ress += [res.detach()]
                losses += F.log_softmax(res, dim=-1)[:,0].mean()
                if update_queue and 'L1' not in relation_type and 'L2' not in relation_type:
                    tmp = self.neg_queue[source_type][relation_type]
                    self.neg_queue[source_type][relation_type] = \
                        torch.cat([node_emb_perturb[source_node_ids].detach(), tmp], dim=0)[:int(self.neg_queue_size * n_nodes)]
        return -losses / len(ress), ress
    
    def structure_loss(self, node_emb, node_emb_perturb, rem_edge_list, ori_edge_list, node_dict, target_type, use_queue = True, update_queue = False):
        losses = 0
        ress   = []
        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        
        # First, get the attention based schma embedding for each target id
        node_nbr_dict = defaultdict( # target_id
                            lambda: defaultdict( #source_type
                                lambda: defaultdict( #relation_type
                                    list #node list
                            )))
        target_node_ids = set()
        for source_type in rem_edge_list:
            if source_type not in self.link_dec_dict:
                continue
            for relation_type in rem_edge_list[source_type]:
                if relation_type not in self.link_dec_dict[source_type]:
                    continue
                rem_edges = rem_edge_list[source_type][relation_type].copy()
                if len(rem_edges) <= 8:
                    continue
                tmp_target_ids = rem_edges[:, 0] + node_dict[target_type][0]
                for _ in tmp_target_ids.tolist():
                    target_node_ids.add(_)

                for edge in rem_edges:
                    node_nbr_dict[edge[0] + node_dict[target_type][0]][source_type][relation_type].append(edge[1]+node_dict[source_type][0])
      
        # Attention
        dict_tarid_srctype_nbrlist = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for tarid in node_nbr_dict.keys():
              for srctype in node_nbr_dict[tarid].keys():
                    tmp_list = []
                    for reltype in node_nbr_dict[tarid][srctype].keys():
                          tmp_list = list(set(tmp_list + node_nbr_dict[tarid][srctype][reltype]))
                    dict_tarid_srctype_nbrlist[tarid][srctype] = tmp_list
        sc_encoder = self.sc
        
        for tarid in dict_tarid_srctype_nbrlist:
            if 'field' not in dict_tarid_srctype_nbrlist[tarid].keys():
                dict_tarid_srctype_nbrlist[tarid]['field'] = [tarid]
            if 'venue' not in dict_tarid_srctype_nbrlist[tarid].keys():
                dict_tarid_srctype_nbrlist[tarid]['venue'] = [tarid]
            if 'author' not in dict_tarid_srctype_nbrlist[tarid].keys():
                dict_tarid_srctype_nbrlist[tarid]['author'] = [tarid]
  
        target_node_ids = list(target_node_ids)
        schema_emb = torch.FloatTensor([]).to(self.device)
        schema_emb_perturb = torch.FloatTensor([]).to(self.device)
        
        # Get the schema_emb
        feats = [node_emb[torch.LongTensor(target_node_ids)], node_emb, node_emb, node_emb] # p a f v
        
        # target_id to schema_id
        tarid_scheid_remap = defaultdict(lambda: int)
        for i in range(len(target_node_ids)):
            tarid_scheid_remap[target_node_ids[i]]=i
        
        h_all = []
        for i in range(len(feats)):
            h_all.append(feats[i])
        
        nei_a = []
        nei_f = []
        nei_v = []
        for idx in target_node_ids:
            nei_a.append(np.array(dict_tarid_srctype_nbrlist[idx]['author']))
            nei_f.append(np.array(dict_tarid_srctype_nbrlist[idx]['field']))
            nei_v.append(np.array(dict_tarid_srctype_nbrlist[idx]['venue']))
        nei_a = np.array(nei_a)
        nei_f = np.array(nei_f)
        nei_v = np.array(nei_v)

        nei_a = [torch.LongTensor(i) for i in nei_a]
        nei_f = [torch.LongTensor(i) for i in nei_f]
        nei_v = [torch.LongTensor(i) for i in nei_v]
        nei_idx = [nei_a, nei_f, nei_v]

        schema_emb = sc_encoder.forward(h_all, nei_idx)
        
    
        for source_type in rem_edge_list:
            if source_type not in self.link_dec_dict:
                continue
            for relation_type in rem_edge_list[source_type]:
                if relation_type not in self.link_dec_dict[source_type]:
                    continue
                rem_edges = rem_edge_list[source_type][relation_type]
                if len(rem_edges) <= 8:
                    continue

                ori_edges = ori_edge_list[source_type][relation_type]
                matcher = self.link_dec_dict[source_type][relation_type]

                target_ids, positive_source_ids = rem_edges[:,0].reshape(-1, 1), rem_edges[:,1].reshape(-1, 1)
                
                n_nodes = len(target_ids)
                source_node_ids = np.unique(ori_edges[:, 1])

                negative_source_ids = [self.neg_sample(source_node_ids, \
                    ori_edges[ori_edges[:, 0] == t_id][:, 1]) for t_id in target_ids]
                sn = min([len(neg_ids) for neg_ids in negative_source_ids])
                
                negative_source_ids = [neg_ids[:sn] for neg_ids in negative_source_ids]

                source_ids = torch.LongTensor(np.concatenate((positive_source_ids, negative_source_ids), axis=-1) + node_dict[source_type][0])
                
                # used the perturb node embedding               
                emb = node_emb[source_ids]
                
                if use_queue and len(self.neg_queue1[source_type][relation_type]) // n_nodes > 0:
                    tmp = self.neg_queue1[source_type][relation_type]
                    stx = len(tmp) // n_nodes
                    tmp = tmp[: stx * n_nodes].reshape(n_nodes, stx, -1)
                    rep_size = sn + 1 + stx
                    source_emb = torch.cat([emb, tmp], dim=1)
                    source_emb = source_emb.reshape(n_nodes * rep_size, -1)
                else:
                    rep_size = sn + 1
                    source_emb = emb.reshape(source_ids.shape[0] * rep_size, -1)
                
                target_ids = target_ids.repeat(rep_size, 1) + node_dict[target_type][0]
                for i in range(len(target_ids)):
                    for j in range(len(target_ids[0])):
                        target_ids[i][j] = tarid_scheid_remap[target_ids[i][j]]
                        
                target_emb = schema_emb[target_ids.reshape(-1)]
                
                res = matcher.forward(target_emb, source_emb)
                res = res.reshape(n_nodes, rep_size)
                ress += [res.detach()]
                losses += F.log_softmax(res, dim=-1)[:,0].mean()
                if update_queue and 'L1' not in relation_type and 'L2' not in relation_type:
                    tmp = self.neg_queue1[source_type][relation_type]
                    self.neg_queue1[source_type][relation_type] = \
                        torch.cat([node_emb[source_node_ids].detach(), tmp], dim=0)[:int(self.neg_queue1_size * n_nodes)]
        return -losses / len(ress)
    
#     def structure_loss1(self, node_emb, node_emb_perturb, rem_edge_list, ori_edge_list, node_dict, target_type, use_queue = True, update_queue = False):
#         losses = 0
#         # ress = []

#         node_nbr_dict = defaultdict( # target_id
#                             lambda: defaultdict( #source_type
#                                 lambda: defaultdict( #relation_type
#                                     list #node list
#                             )))


#         target_node_ids = set()
#         for source_type in rem_edge_list:
#             if source_type not in self.link_dec_dict:
#                 continue
#             for relation_type in rem_edge_list[source_type]:
#                 if relation_type not in self.link_dec_dict[source_type]:
#                     continue
#                 rem_edges = rem_edge_list[source_type][relation_type].copy()
#                 if len(rem_edges) <= 8:
#                     continue
#                 tmp_target_ids = rem_edges[:, 0] + node_dict[target_type][0]
#                 for _ in tmp_target_ids.tolist():
#                     target_node_ids.add(_)

#                 for edge in rem_edges:
#                     node_nbr_dict[edge[0] + node_dict[target_type][0]][source_type][relation_type].append(edge[1]+node_dict[source_type][0])
        
#         # Attention Test
#         dict_tarid_srctype_nbrlist = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
#         for tarid in node_nbr_dict.keys():
#               for srctype in node_nbr_dict[tarid].keys():
#                     tmp_list = []
#                     for reltype in node_nbr_dict[tarid][srctype].keys():
#                           tmp_list = list(set(tmp_list + node_nbr_dict[tarid][srctype][reltype]))
#                     dict_tarid_srctype_nbrlist[tarid][srctype] = tmp_list
#         sc_encoder = self.sc

#         node_emb_size = node_emb.size()
#         for tarid in dict_tarid_srctype_nbrlist:
#             if 'field' not in dict_tarid_srctype_nbrlist[tarid].keys():
#                 dict_tarid_srctype_nbrlist[tarid]['field'] = [tarid]
#             if 'venue' not in dict_tarid_srctype_nbrlist[tarid].keys():
#                 dict_tarid_srctype_nbrlist[tarid]['venue'] = [tarid]
#             if 'author' not in dict_tarid_srctype_nbrlist[tarid].keys():
#                 dict_tarid_srctype_nbrlist[tarid]['author'] = [tarid]
  
#         target_node_ids = list(target_node_ids)
#         schema_emb = torch.FloatTensor([]).to(self.device)
#         schema_emb_perturb = torch.FloatTensor([]).to(self.device)
        
#         # Get the schema_emb
#         feats = [node_emb[torch.LongTensor(target_node_ids)], node_emb, node_emb, node_emb] # p a f v
        
#         h_all = []
#         for i in range(len(feats)):
#             h_all.append(feats[i])
        
#         nei_a = []
#         nei_f = []
#         nei_v = []
#         for idx in target_node_ids:
#             nei_a.append(np.array(dict_tarid_srctype_nbrlist[idx]['author']))
#             nei_f.append(np.array(dict_tarid_srctype_nbrlist[idx]['field']))
#             nei_v.append(np.array(dict_tarid_srctype_nbrlist[idx]['venue']))
#         nei_a = np.array(nei_a)
#         nei_f = np.array(nei_f)
#         nei_v = np.array(nei_v)

#         nei_a = [torch.LongTensor(i) for i in nei_a]
#         nei_f = [torch.LongTensor(i) for i in nei_f]
#         nei_v = [torch.LongTensor(i) for i in nei_v]
#         nei_idx = [nei_a, nei_f, nei_v]

#         schema_emb = sc_encoder.forward(h_all, nei_idx)
        
#         # Get the schema_emb_perturb
#         feats = [node_emb_perturb[torch.LongTensor(target_node_ids)], node_emb_perturb, node_emb_perturb, node_emb_perturb] # p a f v
#         h_all = []
#         for i in range(len(feats)):
#             h_all.append(feats[i])
        
#         nei_a = []
#         nei_f = []
#         nei_v = []
#         for idx in target_node_ids:
#             nei_a.append(np.array(dict_tarid_srctype_nbrlist[idx]['author']))
#             nei_f.append(np.array(dict_tarid_srctype_nbrlist[idx]['field']))
#             nei_v.append(np.array(dict_tarid_srctype_nbrlist[idx]['venue']))
#         nei_a = np.array(nei_a)
#         nei_f = np.array(nei_f)
#         nei_v = np.array(nei_v)

#         nei_a = [torch.LongTensor(i) for i in nei_a]
#         nei_f = [torch.LongTensor(i) for i in nei_f]
#         nei_v = [torch.LongTensor(i) for i in nei_v]
#         nei_idx = [nei_a, nei_f, nei_v]

#         schema_emb_perturb = sc_encoder.forward(h_all, nei_idx)
        
#         schema_idxs = list()
#         for idx in range(len(target_node_ids)):
#             schema_idxs.append([idx] + [_ for _ in range(idx)] + [_ for _ in range(idx + 1, len(target_node_ids))])
#         query_schema_emb = schema_emb[schema_idxs]
        
#         # tmp = torch.unsqueeze(self.neg_structure_queue, 0)
#         # tmp = tmp.repeat(query_schema_emb.shape[0], 1, 1)
#         # query_schema_emb = torch.cat([query_schema_emb, tmp], dim = 1)
#         # del tmp

#         # self.neg_structure_queue = torch.cat([schema_emb.detach(), self.neg_structure_queue], dim=0)[:self.neg_structure_queue_size]

#         rep_size = query_schema_emb.shape[1]

# #         query_emb = node_emb[torch.LongTensor(target_node_ids)]
# #         query_emb = query_emb.repeat(rep_size, 1, 1) 
        
#         my_query_idxs = list()
#         for idx in range(len(target_node_ids)):
#             my_query_idxs.append([idx] * rep_size)
#         query_emb = schema_emb_perturb[my_query_idxs]
        
#         # query_emb_tmp_my = node_emb[torch.LongTensor(target_node_ids)]
#         # my_query_idxs = list()
#         # for idx in range(len(target_node_ids)):
#         #     my_query_idxs.append([idx] * rep_size)
#         # query_emb = query_emb_tmp_my[my_query_idxs]

#         query_emb = query_emb.reshape(len(target_node_ids) * rep_size, -1)
#         query_schema_emb = query_schema_emb.reshape(len(target_node_ids) * rep_size, -1)
#         res = self.structure_matcher(query_emb, query_schema_emb)
#         res = res.reshape(len(target_node_ids), rep_size)

#         losses += F.log_softmax(res, dim=-1)[:,0].mean()

#         return -losses

    def text_loss(self, reps, texts, w2v_model, device):
        def parse_text(texts, w2v_model, device):
            idxs = []
            pad  = w2v_model.wv.vocab['eos'].index
            for text in texts:
                idx = []
                for word in ['bos'] + preprocess_string(text) + ['eos']:
                    if word in w2v_model.wv.vocab:
                        idx += [w2v_model.wv.vocab[word].index]
                idxs += [idx]
            mxl = np.max([len(s) for s in idxs]) + 1
            inp_idxs = []
            out_idxs = []
            masks    = []
            for i, idx in enumerate(idxs):
                inp_idxs += [idx + [pad for _ in range(mxl - len(idx) - 1)]]
                out_idxs += [idx[1:] + [pad for _ in range(mxl - len(idx))]]
                masks    += [[1 for _ in range(len(idx))] + [0 for _ in range(mxl - len(idx) - 1)]]
            return torch.LongTensor(inp_idxs).transpose(0, 1).to(device), \
                   torch.LongTensor(out_idxs).transpose(0, 1).to(device), torch.BoolTensor(masks).transpose(0, 1).to(device)
        inp_idxs, out_idxs, masks = parse_text(texts, w2v_model, device)
        pred_prob = self.attr_decoder(inp_idxs, reps.repeat(inp_idxs.shape[0], 1, 1))      
        return self.ce(pred_prob[masks], out_idxs[masks]).mean()

    def feat_loss(self, reps, out):
        return -self.attr_decoder(reps, out).mean()


class Classifier(nn.Module):
    def __init__(self, n_hid, n_out):
        super(Classifier, self).__init__()
        self.n_hid    = n_hid
        self.n_out    = n_out
        self.linear   = nn.Linear(n_hid,  n_out)
    def forward(self, x):
        tx = self.linear(x)
        return torch.log_softmax(tx.squeeze(), dim=-1)
    def __repr__(self):
        return '{}(n_hid={}, n_out={})'.format(
            self.__class__.__name__, self.n_hid, self.n_out)

    
class Matcher(nn.Module):
    '''
        Matching between a pair of nodes to conduct link prediction.
        Use multi-head attention as matching model.
    '''

    def __init__(self, n_hid, n_out, temperature = 0.1):
        super(Matcher, self).__init__()
        self.n_hid          = n_hid
        self.linear    = nn.Linear(n_hid,  n_out)
        self.sqrt_hd     = math.sqrt(n_out)
        self.drop        = nn.Dropout(0.2)
        self.cosine      = nn.CosineSimilarity(dim=1)
        self.cache       = None
        self.temperature = temperature
    def forward(self, x, ty, use_norm = True):
        tx = self.drop(self.linear(x))
        if use_norm:
            return self.cosine(tx, ty) / self.temperature
        else:
            return (tx * ty).sum(dim=-1) / self.sqrt_hd
    def __repr__(self):
        return '{}(n_hid={})'.format(
            self.__class__.__name__, self.n_hid)

class StructureMapping(nn.Module):

    def __init__(self, n_hid, n_out):
        super(StructureMapping, self).__init__()
        self.n_hid  = n_hid
        self.linear = nn.Linear(n_hid, n_out)
        self.drop   = nn.Dropout(0.2)
    
    def forward(self, x):
        return self.drop(self.linear(x))

    
class GNN(nn.Module):
    def __init__(self, in_dim, n_hid, num_types, num_relations, n_heads, n_layers, dropout = 0.2, conv_name = 'hgt', prev_norm = False, last_norm = False, use_RTE = True):
        super(GNN, self).__init__()
        self.gcs = nn.ModuleList()
        self.num_types = num_types
        self.in_dim    = in_dim
        self.n_hid     = n_hid
        self.adapt_ws  = nn.ModuleList()
        self.drop      = nn.Dropout(dropout)
        for t in range(num_types):
            self.adapt_ws.append(nn.Linear(in_dim, n_hid))
        for l in range(n_layers - 1):
            self.gcs.append(GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm = prev_norm, use_RTE = use_RTE))
        self.gcs.append(GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm = last_norm, use_RTE = use_RTE))

    def forward(self, node_feature, node_type, edge_time, edge_index, edge_type):
        res = torch.zeros(node_feature.size(0), self.n_hid).to(node_feature.device)
        for t_id in range(self.num_types):
            idx = (node_type == int(t_id))
            if idx.sum() == 0:
                continue
            res[idx] = torch.tanh(self.adapt_ws[t_id](node_feature[idx]))
        meta_xs = self.drop(res)
        del res
        for gc in self.gcs:
            meta_xs = gc(meta_xs, node_type, edge_index, edge_type, edge_time)
        return meta_xs   

    
class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
    def __init__(self, n_word, ninp, nhid, nlayers, dropout=0.2):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(nhid, nhid, nlayers)
        self.encoder = nn.Embedding(n_word, nhid)
        self.decoder = nn.Linear(nhid, n_word)
        self.adp     = nn.Linear(ninp + nhid, nhid)
    def forward(self, inp, hidden = None):
        emb = self.encoder(inp)
        if hidden is not None:
            emb = torch.cat((emb, hidden), dim=-1)
            emb = F.gelu(self.adp(emb))
        output, _ = self.rnn(emb)
        decoded = self.decoder(self.drop(output))
        return decoded
    def from_w2v(self, w2v):
        initrange = 0.1
        self.encoder.weight.data = w2v
        self.decoder.weight = self.encoder.weight
        
        self.encoder.weight.requires_grad = False
        self.decoder.weight.requires_grad = False

class inter_att(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(inter_att, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax()
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        # print("sc ", beta.data.cpu().numpy())  # type-level attention
        z_mc = 0
        for i in range(len(embeds)):
            z_mc += embeds[i] * beta[i]
        return z_mc


class intra_att(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(intra_att, self).__init__()
        self.att = nn.Parameter(torch.empty(size=(1, 2*hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

        self.softmax = nn.Softmax(dim=1)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, nei, h, h_refer):
        nei_emb = F.embedding(nei, h)
        h_refer = torch.unsqueeze(h_refer, 1)
        h_refer = h_refer.expand_as(nei_emb)
        all_emb = torch.cat([h_refer, nei_emb], dim=-1)
        attn_curr = self.attn_drop(self.att)
        att = self.leakyrelu(all_emb.matmul(attn_curr.t()))
        att = self.softmax(att)
        nei_emb = (att*nei_emb).sum(dim=1)
        return nei_emb


class Sc_encoder(nn.Module):
    def __init__(self, hidden_dim, sample_rate, nei_num, attn_drop, device):
        super(Sc_encoder, self).__init__()
        self.intra = nn.ModuleList([intra_att(hidden_dim, attn_drop) for _ in range(nei_num)])
        self.inter = inter_att(hidden_dim, attn_drop)
        self.sample_rate = sample_rate
        self.nei_num = nei_num
        self.device = device

    def forward(self, nei_h, nei_index):
        embeds = []
        for i in range(self.nei_num):
            sele_nei = []
            sample_num = self.sample_rate[i]
            for per_node_nei in nei_index[i]:
                if len(per_node_nei) >= sample_num:
                    select_one = torch.tensor(np.random.choice(per_node_nei, sample_num, replace=False))[np.newaxis]
                else:
                    select_one = torch.tensor(np.random.choice(per_node_nei, sample_num, replace=True ))[np.newaxis]
                sele_nei.append(select_one)
            
            sele_nei = torch.cat(sele_nei, dim=0).to(self.device)
            one_type_emb = F.elu(self.intra[i](sele_nei, nei_h[i + 1], nei_h[0]))
            embeds.append(one_type_emb)
        z_mc = self.inter(embeds)
        return z_mc

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()
       
def get_mixup_neg_pos_emb(ia, emb, target_emb_tmp, cos, device):
    tmp = emb[ia, :, :]
    pos_emb = tmp[0, :]
    neg_emb = tmp[1:,:]
    query_emb = pos_emb
    neg_emb_len = neg_emb.shape[0]
    emb_len = neg_emb.shape[1]

    ratio = 0.80
    ratio_neg_emb_len = int(ratio * neg_emb_len)
    cos_similarity = cos(neg_emb, query_emb)
    cos_topk_idx = torch.topk(cos_similarity, ratio_neg_emb_len).indices
    chosed_neg_emb = neg_emb[cos_topk_idx,:]
    still_need_neg_num = neg_emb_len - ratio_neg_emb_len
    emb_mixup = torch.zeros([still_need_neg_num, emb_len]).to(device)

    rand_alpha = torch.Tensor(np.random.beta(0.2, 0.2, still_need_neg_num)).reshape(still_need_neg_num,1).to(device)
    idx_tmp_rand = (rand_alpha<0.5).nonzero()
    rand_alpha[idx_tmp_rand]=1-rand_alpha[idx_tmp_rand]
    minus_rand_alpha = 1-rand_alpha

    neg_idx1 =  np.random.choice(cos_topk_idx.cpu(), still_need_neg_num, replace=False)
    neg_idx2 =  0 * np.ones((still_need_neg_num),dtype=np.int64)
    
    neg_emb1 = neg_emb[neg_idx1]
    neg_emb2 = tmp[neg_idx2]
    emb_mixup = torch.mul(rand_alpha, neg_emb1) + torch.mul(minus_rand_alpha, neg_emb2)
    '''
    query_emb = target_emb_tmp[ia,:]
    query_pos_emb = pos_emb
    torch.save(query_emb, "query_emb.pth")
    torch.save(query_pos_emb, "query_pos_emb.pth")
    torch.save(neg_emb, "neg_emb.pth")
    torch.save(chosed_neg_emb, "chosed_neg_emb.pth")
    torch.save(emb_mixup, "emb_mixup.pth")
    print('done')
    time.sleep(10000)
    '''
    return torch.cat((chosed_neg_emb, emb_mixup))
     
def get_mixup_neg_emb(ia, emb, target_emb_tmp, cos, device):
    tmp = emb[ia, :, :]
    query_pos_emb = tmp[0, :]
    # query_emb = target_emb_tmp[ia,:,:]
    query_emb = query_pos_emb
    neg_emb = tmp[1:, :]
    
    neg_emb_len = neg_emb.shape[0]
    emb_len = neg_emb.shape[1]
    
    ratio = 0.55
    ratio_neg_emb_len = int(ratio * neg_emb_len)
    cos_similarity = cos(neg_emb, query_emb)
    cos_topk_idx = torch.topk(cos_similarity, ratio_neg_emb_len).indices

    chosed_neg_emb = neg_emb[cos_topk_idx,:] 
    still_need_neg_num = neg_emb_len - ratio_neg_emb_len
    emb_mixup = torch.zeros([still_need_neg_num, emb_len]).to(device)
    
    rand_alpha = torch.Tensor(np.random.beta(0.5, 0.5, still_need_neg_num)).reshape(still_need_neg_num,1).to(device)
    minus_rand_alpha = 1 - rand_alpha
    neg_idx1 = -1 * np.ones((still_need_neg_num),dtype=np.int64)
    neg_idx2 = -1 * np.ones((still_need_neg_num),dtype=np.int64)
    for i in range(still_need_neg_num):
        idx1, idx2 = np.random.choice(cos_topk_idx.cpu(), 2, replace=False)
        neg_idx1[i] = idx1
        neg_idx2[i] = idx2
    neg_emb1 = neg_emb[neg_idx1]
    neg_emb2 = neg_emb[neg_idx2]
    emb_mixup = torch.mul(rand_alpha, neg_emb1) + torch.mul(minus_rand_alpha, neg_emb2)
    '''
    torch.save(query_emb, "query_emb.pth")
    torch.save(query_pos_emb, "query_pos_emb.pth")
    torch.save(neg_emb, "neg_emb.pth")
    torch.save(chosed_neg_emb, "chosed_neg_emb.pth")
    torch.save(emb_mixup, "emb_mixup.pth")
    '''
    return torch.cat((chosed_neg_emb, emb_mixup))



