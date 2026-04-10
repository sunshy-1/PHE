import sys
import scipy.io as scio 
from PROPOSED_GNN.data import *
from PROPOSED_GNN.model import *
from warnings import filterwarnings
filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser(description='Pre-training HGT on a given graph (heterogeneous / homogeneous)')

'''
   PROPOSED_GNN arguments 
'''
parser.add_argument('--attr_ratio', type=float, default=0.5,
                    help='Ratio of attr-loss against link-loss, range: [0-1]') 
parser.add_argument('--attr_type', type=str, default='text',
                    choices=['text', 'vec'],
                    help='The type of attribute decoder')
parser.add_argument('--neg_samp_num', type=int, default=255,
                    help='Maximum number of negative sample for each target node.')
parser.add_argument('--queue_size', type=int, default=256,
                    help='Max size of adaptive embedding queue.')
parser.add_argument('--w2v_dir', type=str, default='./OAG_dataset/w2v_all',
                    help='The address of word2vec pre-trained model, it is useful if you want caculate the text-loss.')

'''
    Dataset arguments
'''
parser.add_argument('--data_dir', type=str, default = './OAG_dataset/graph_CS.pk',
                    help='The address of preprocessed graph.')
parser.add_argument('--pretrain_model_dir', type=str, default='models/test',
                    help='The address for storing the models and optimization results.')
parser.add_argument('--cuda', type=int, default=2,
                    help='Avaiable GPU ID')      
parser.add_argument('--sample_depth', type=int, default=6,
                    help='How many layers within a mini-batch subgraph')
parser.add_argument('--sample_width', type=int, default=128,
                    help='How many nodes to be sampled per layer per type')
'''
   Model arguments 
'''
parser.add_argument('--conv_name', type=str, default='hgt',
                    choices=['hgt', 'gcn', 'gat', 'rgcn', 'han', 'hetgnn','sage'],
                    help='The name of GNN filter. By default is Heterogeneous Graph Transformer (hgt)')
parser.add_argument('--n_hid', type=int, default=400,
                    help='Number of hidden dimension')
parser.add_argument('--n_heads', type=int, default=8,
                    help='Number of attention head')
parser.add_argument('--n_layers', type=int, default=3,
                    help='Number of GNN layers')
parser.add_argument('--prev_norm', help='Whether to add layer-norm on the previous layers', action='store_true')
parser.add_argument('--last_norm', help='Whether to add layer-norm on the last layers',     action='store_true')
parser.add_argument('--dropout', type=int, default=0.2, help='Dropout ratio')

'''
    Optimization arguments
'''
parser.add_argument('--max_lr', type=float, default=1e-3,
                    help='Maximum learning rate.')
parser.add_argument('--scheduler', type=str, default='cycle',
                    help='Name of learning rate scheduler.' , choices=['cycle', 'cosine'])
parser.add_argument('--n_epoch', type=int, default=20,
                    help='Number of epoch to run')
parser.add_argument('--n_pool', type=int, default=8,
                    help='Number of process to sample subgraph')    
parser.add_argument('--n_batch', type=int, default=32,
                    help='Number of batch (sampled graphs) for each epoch') 
parser.add_argument('--batch_size', type=int, default=1024,
                    help='Number of output nodes for training')    
parser.add_argument('--clip', type=float, default=0.5,
                    help='Gradient Norm Clipping')

'''
    additional arguments
'''
parser.add_argument('--use_text', help='Whether to use text loss', action='store_true')


args = parser.parse_args()
args_print(args)

if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")

print('Start Loading Graph Data...')
graph = renamed_load(open(args.data_dir, 'rb'))
print('Finish Loading Graph Data!')

pre_range   = {t: True for t in graph.times if t != None and t < 2014}
train_range = {t: True for t in graph.times if t != None and t >= 2014  and t <= 2016}
valid_range = {t: True for t in graph.times if t != None and t > 2016  and t <= 2017}
test_range  = {t: True for t in graph.times if t != None and t > 2017}

pre_target_nodes   = []
train_target_nodes = []
target_type = 'paper'
rel_stop_list = ['self', 'rev_PF_in_L0', 'rev_PF_in_L5', 'rev_PV_Repository', 'rev_PV_Patent']


for p_id, _time in enumerate(list(graph.node_feature[target_type]['time'])):
    if _time in pre_range:
        pre_target_nodes += [[p_id, _time]]
    elif _time in train_range:
        train_target_nodes += [[p_id, _time]]
pre_target_nodes = np.array(pre_target_nodes)
train_target_nodes = np.array(train_target_nodes)


def GPT_sample(seed, target_nodes, time_range, batch_size, feature_extractor):
    np.random.seed(seed)
    samp_target_nodes = target_nodes[np.random.choice(len(target_nodes), batch_size)]
    threshold   = 0.5
    feature, times, edge_list, _, attr = sample_subgraph(graph, time_range, \
                inp = {target_type: samp_target_nodes}, feature_extractor = feature_extractor, \
                    sampled_depth = args.sample_depth, sampled_number = args.sample_width)
    rem_edge_list = defaultdict(  #source_type
                        lambda: defaultdict(  #relation_type
                            lambda: [] # [target_id, source_id] 
                                ))
    
    ori_list = {}
    for source_type in edge_list[target_type]:
        ori_list[source_type] = {}
        for relation_type in edge_list[target_type][source_type]:
            ori_list[source_type][relation_type] = np.array(edge_list[target_type][source_type][relation_type])
            el = []
            for target_ser, source_ser in edge_list[target_type][source_type][relation_type]:
                if relation_type not in rel_stop_list and target_ser < batch_size and np.random.random() > threshold:
                    rem_edge_list[source_type][relation_type] += [[target_ser, source_ser]]
                    continue
                el += [[target_ser, source_ser]]
            if not el:
                continue
            el = np.array(el)
            edge_list[target_type][source_type][relation_type] = el

            if relation_type == 'self':
                continue
            else:
                if 'rev_' in relation_type:
                    rev_relation = relation_type[4:]
                else:
                    rev_relation = 'rev_' + relation_type
                edge_list[source_type]['paper'][rev_relation] = list(np.stack((el[:,1], el[:,0])).T)
                
    '''
        Adding feature nodes:
    '''
    n_target_nodes = len(feature[target_type])
    feature[target_type] = np.concatenate((feature[target_type], np.zeros([batch_size, feature[target_type].shape[1]])))
    times[target_type]   = np.concatenate((times[target_type], times[target_type][:batch_size]))

    for source_type in edge_list[target_type]:
        for relation_type in edge_list[target_type][source_type]:
            el = []
            for target_ser, source_ser in edge_list[target_type][source_type][relation_type]:
                if target_ser < batch_size:
                    if relation_type == 'self':
                        el += [[target_ser + n_target_nodes, target_ser + n_target_nodes]]
                    else:
                        el += [[target_ser + n_target_nodes, source_ser]]
            if len(el) > 0:
                edge_list[target_type][source_type][relation_type] = \
                    np.concatenate((edge_list[target_type][source_type][relation_type], el))


    rem_edge_lists = {}
    for source_type in rem_edge_list:
        rem_edge_lists[source_type] = {}
        for relation_type in rem_edge_list[source_type]:
            rem_edge_lists[source_type][relation_type] = np.array(rem_edge_list[source_type][relation_type])
    del rem_edge_list
          
    return to_torch(feature, times, edge_list, graph), rem_edge_lists, ori_list, \
            attr[:batch_size], (n_target_nodes, n_target_nodes + batch_size)


def prepare_data(pool):
    jobs = []
    for _ in np.arange(args.n_batch - 1):
        jobs.append(pool.apply_async(GPT_sample, args=(randint(), pre_target_nodes, pre_range, args.batch_size, feature_OAG)))
    jobs.append(pool.apply_async(GPT_sample, args=(randint(), train_target_nodes, train_range, args.batch_size, feature_OAG)))
    return jobs


pool = mp.Pool(args.n_pool)
st = time.time()
jobs = prepare_data(pool)
repeat_num = int(len(pre_target_nodes) / args.batch_size // args.n_batch)


data, rem_edge_list, ori_edge_list, _, _ = GPT_sample(randint(), pre_target_nodes, pre_range, args.batch_size, feature_OAG)
node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = data
types = graph.get_types()


gnn = GNN(conv_name = args.conv_name, in_dim = len(graph.node_feature[target_type]['emb'].values[0]) + 401, n_hid = args.n_hid, \
          n_heads = args.n_heads, n_layers = args.n_layers, dropout = args.dropout, num_types = len(types), \
          num_relations = len(graph.get_meta_graph()) + 1, prev_norm = args.prev_norm, last_norm = args.last_norm)

gnn_perturb = GNN(conv_name = args.conv_name, in_dim = len(graph.node_feature[target_type]['emb'].values[0]) + 401, n_hid = args.n_hid, \
          n_heads = args.n_heads, n_layers = args.n_layers, dropout = args.dropout, num_types = len(types), \
          num_relations = len(graph.get_meta_graph()) + 1, prev_norm = args.prev_norm, last_norm = args.last_norm).to(device)

if args.attr_type == 'text':  
    from gensim.models import Word2Vec
    w2v_model = Word2Vec.load(args.w2v_dir)
    n_tokens = len(w2v_model.wv.vocab)
    attr_decoder = RNNModel(n_word = n_tokens, ninp = gnn.n_hid, \
               nhid = w2v_model.vector_size, nlayers = 2)
    attr_decoder.from_w2v(torch.FloatTensor(w2v_model.wv.vectors))
else:
    attr_decoder = Matcher(gnn.n_hid, gnn.in_dim)
    
propgnn = PROPOSED_GNN(gnn = gnn, rem_edge_list = rem_edge_list, attr_decoder = attr_decoder, \
                  neg_queue_size = 0, types = types, neg_samp_num = args.neg_samp_num, device = device, \
                  hidden_dim = gnn.n_hid, sample_rate = [2,2,1], nei_num = 4-1, attn_drop = 0.1)
propgnn.init_emb.data = node_feature[node_type == node_dict[target_type][1]].mean(dim=0).detach()
propgnn = propgnn.to(device)

best_val   = 100000
train_step = 0
stats = []
optimizer = torch.optim.AdamW(propgnn.parameters(), weight_decay = 1e-2, eps=1e-06, lr = args.max_lr)

if args.scheduler == 'cycle':
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, pct_start=0.02, anneal_strategy='linear', final_div_factor=100,\
                        max_lr = args.max_lr, total_steps = repeat_num * args.n_batch * args.n_epoch + 1)
elif args.scheduler == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, repeat_num * args.n_batch, eta_min=1e-6)

save_path = 'state_save'
start_epoch_suspend, start_batch_suspend = 0, 0
if os.path.exists(os.path.join(save_path, 'latest'+str(args.cuda)+'.pt')):
    ckpt = torch.load(os.path.join(save_path, 'latest'+str(args.cuda)+'.pt'), map_location='cpu')
    propgnn.load_state_dict(ckpt['propgnn'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    start_epoch_suspend = ckpt['epoch']
    start_batch_suspend = ckpt['batch']
    best_val = ckpt['best_val']


print('Start Pretraining...')
for epoch in np.arange(start_epoch_suspend , args.n_epoch) + 1:
    propgnn.neg_queue_size = args.queue_size * epoch // args.n_epoch
    for batch in np.arange(repeat_num) + 1:
        if batch < start_batch_suspend + 1:
            continue
        train_data = [job.get() for job in jobs[:-1]]
        valid_data = jobs[-1].get()
        pool.close()
        pool.join()
        pool = mp.Pool(args.n_pool)
        jobs = prepare_data(pool)
        et = time.time()
        print('Data Preparation: %.1fs' % (et - st))

        train_link_losses = []
        train_attr_losses = []
        train_part1_losses = []
        train_part2_losses = []
        
        propgnn.train()
        for data, rem_edge_list, ori_edge_list, attr, (start_idx, end_idx) in train_data:
            node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = data
            node_feature = node_feature.detach()
            node_feature[start_idx : end_idx] = propgnn.init_emb
            node_emb = propgnn.gnn(node_feature.to(device), node_type.to(device), edge_time.to(device), \
                                   edge_index.to(device), edge_type.to(device))
            
            num_perturb_gnn = 5
            tmp_sum_perturb = torch.zeros_like(node_emb)
            node_emb_perturb = torch.zeros_like(node_emb)
            
            dict_random_para = {}
            model_dict = propgnn.gnn.state_dict()
            perturb_model_dict = gnn_perturb.state_dict()
            
            for perturb_idx in range(num_perturb_gnn):
                for name, para in propgnn.gnn.named_parameters():
                    perturb_model_dict[name][:] = model_dict[name][:] + (((torch.rand(para.size())-0.5).to(device))*0.15*torch.std(para))
                
                dict_random_para[str(perturb_idx)] = perturb_model_dict
                gnn_perturb.load_state_dict(perturb_model_dict)
                EMB_TMP = gnn_perturb(node_feature.to(device), node_type.to(device), edge_time.to(device), edge_index.to(device), edge_type.to(device))
                tmp_sum_perturb = tmp_sum_perturb.clone().detach() + EMB_TMP.clone().detach()
                
            node_emb_perturb = tmp_sum_perturb / num_perturb_gnn
            
            node_feature = node_feature.to('cpu')
            node_type = node_type.to('cpu')
            edge_time = edge_time.to('cpu')
            edge_index = edge_index.to('cpu')
            edge_type = edge_type.to('cpu')
            torch.cuda.empty_cache() 
                         
            loss_link, _ = propgnn.link_loss(node_emb, node_emb_perturb, rem_edge_list, ori_edge_list, node_dict, target_type, use_queue = True, update_queue=True)
            loss_structure = propgnn.structure_loss(node_emb, node_emb_perturb, rem_edge_list, ori_edge_list, node_dict, target_type, use_queue = True, update_queue=True)
            if args.use_text:
                if args.attr_type == 'text':
                    loss_attr = propgnn.text_loss(node_emb[start_idx : end_idx], attr, w2v_model, device)
                else:
                    loss_attr = propgnn.feat_loss(node_emb[start_idx : end_idx], torch.FloatTensor(attr).to(device))

            if args.use_text:
                loss = loss_link + loss_attr * args.attr_ratio + 0.1 * loss_structure
            else:
                loss = loss_link + 0.1 * loss_structure
           
            optimizer.zero_grad() 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(propgnn.parameters(), args.clip)
            optimizer.step()
            train_part1_losses += [loss_link.item()]
            train_part2_losses += [loss_structure.item()]
            train_link_losses += [loss.item()]
            if args.use_text:
                train_attr_losses += [loss_attr.item()]
            scheduler.step()
            
            node_emb = node_emb.to('cpu')
            node_emb_perturb = node_emb_perturb.to('cpu')
            torch.cuda.empty_cache()
        
        '''
            Valid
        '''
        propgnn.eval()
        with torch.no_grad():
            data, rem_edge_list, ori_edge_list, attr, (start_idx, end_idx) = valid_data
            node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = data
            node_feature = node_feature.detach()
            node_feature[start_idx : end_idx] = propgnn.init_emb
            node_emb = propgnn.gnn(node_feature.to(device), node_type.to(device), edge_time.to(device), \
                                       edge_index.to(device), edge_type.to(device))
            
            node_emb_perturb = torch.zeros_like(node_emb)
            
            for perturb_idx in range(num_perturb_gnn):
                model_dict = dict_random_para[str(perturb_idx)]
                gnn_perturb.load_state_dict(model_dict)
                node_emb_perturb += gnn_perturb(node_feature.to(device), node_type.to(device), edge_time.to(device), edge_index.to(device), edge_type.to(device))
            node_emb_perturb = node_emb_perturb/num_perturb_gnn

            node_feature = node_feature.to('cpu')
            node_type = node_type.to('cpu')
            edge_time = edge_time.to('cpu')
            edge_index = edge_index.to('cpu')
            edge_type = edge_type.to('cpu')
            torch.cuda.empty_cache()
            
            loss_link, ress = propgnn.link_loss(node_emb, node_emb_perturb, rem_edge_list, ori_edge_list, node_dict, target_type, use_queue = True, update_queue=True)
            loss_structure = propgnn.structure_loss(node_emb, node_emb_perturb, rem_edge_list, ori_edge_list, node_dict, target_type, use_queue = True, update_queue=True)
            
            if args.use_text:
                if args.attr_type == 'text':   
                    loss_attr = propgnn.text_loss(node_emb[start_idx : end_idx], attr, w2v_model, device)
                else:
                    loss_attr = propgnn.feat_loss(node_emb[start_idx : end_idx], torch.FloatTensor(attr).to(device))

            if args.use_text:
                valid_loss = loss_link + loss_attr * args.attr_ratio + 0.1*loss_structure
            else:
                valid_loss = loss_link  + 0.1 * loss_structure
            
            node_emb = node_emb.to('cpu')    
            node_emb_perturb = node_emb_perturb.to('cpu')
            torch.cuda.empty_cache()  
            st = time.time()
            if args.use_text:
                print(("Epoch: %d, (%d / %d) %.1fs  LR: %.5f Train Loss: (%.3f, %.3f)  Valid Loss: (%.3f, %.3f)  Norm: %.3f  queue: %d") % \
                    (epoch, batch, repeat_num, (st-et), optimizer.param_groups[0]['lr'], np.average(train_link_losses), np.average(train_attr_losses), \
                    valid_loss, loss_attr, node_emb.norm(dim=1).mean(), propgnn.neg_queue_size))
            else:
                print(("cuda: %d, Epoch: %d, (%d / %d) %.1fs  LR: %.5f Train Loss: %.3f  Valid Loss: %.3f  Norm: %.3f  queue: %d") % \
                  (args.cuda, epoch, batch, repeat_num, (st-et), optimizer.param_groups[0]['lr'], loss, valid_loss, node_emb.norm(dim=1).mean(), propgnn.neg_queue_size))
            
        if valid_loss < best_val:
            best_val = valid_loss
            print('UPDATE!!!')
            torch.save(propgnn.state_dict(), args.pretrain_model_dir)
        torch.cuda.empty_cache()  
    start_batch_suspend = 0
