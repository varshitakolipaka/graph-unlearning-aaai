from models.deletion import GATDelete, GCNDelete, GINDelete
from models.models import GAT, GCN, GIN


def get_model(args, mask_1hop=None, mask_2hop=None, num_nodes=None, num_edge_type=None):

    if 'gnndelete' in args.unlearning_model:
        model_mapping = {'gcn': GCNDelete, 'gat': GATDelete, 'gin': GINDelete}

    else:
        model_mapping = {'gcn': GCN, 'gat': GAT, 'gin': GIN}

    return model_mapping[args.gnn](args, mask_1hop=mask_1hop, mask_2hop=mask_2hop, num_nodes=num_nodes, num_edge_type=num_edge_type)
