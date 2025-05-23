import torch
from torch_geometric.explain.config import ModelMode, ModelTaskLevel
from torch_geometric.utils import k_hop_subgraph, subgraph


def setup_models(models, device):
    res = []
    for name, model in models:
        res.append((name, model.to(device)))
    return res


def custom_iou(gt_nodes, explain_nodes):
    intersection = torch.isin(gt_nodes, explain_nodes).sum().float()
    union = gt_nodes.size(0) + explain_nodes.size(0) - intersection
    return intersection / union


def custom_recall(gt_nodes, explain_nodes):
    intersection = torch.isin(gt_nodes, explain_nodes).sum().float()
    return intersection / gt_nodes.size(0)


def custom_precision(gt_nodes, explain_nodes):
    intersection = torch.isin(gt_nodes, explain_nodes).sum().float()
    return intersection / explain_nodes.size(0)


def custom_fidelity(explainer, explanation, node_mask, max_nodes=None, full_graph=False):
    model_mode = explainer.model_config.mode
    x = explanation.x
    model_config = explainer.model_config
    if full_graph:
        sub_nodes = torch.arange(x.size(0), device=x.device)
        computation_edge_index = explanation.edge_index
    else:
        if explainer.model_config.task_level == ModelTaskLevel.edge:
            index = explanation.edge_label_index.view(-1)
            sub_nodes, computation_edge_index, _, hard_edge_mask = k_hop_subgraph(explanation.edge_label_index.view(-1),
                                                                                  num_hops=explainer.model.model.num_layers,
                                                                                  edge_index=explanation.edge_index)
        else:
            index = explanation.index
            sub_nodes, computation_edge_index, _, hard_edge_mask = k_hop_subgraph(explanation.index.item(),
                                                                                  num_hops=explainer.model.model.num_layers,
                                                                                  edge_index=explanation.edge_index)
    model = explainer.model
    y = explanation.target
    if model_mode == ModelMode.multiclass_classification:
        y_hat = model(x, computation_edge_index).softmax(dim=-1).argmax(dim=-1)
    else:
        if model_config.task_level == ModelTaskLevel.edge:
            y_hat = (model(x, computation_edge_index, edge_label_index=index.view(2, 1)).sigmoid().view(
                -1) > 0.5).long()
        else:
            y_hat = (model(x, computation_edge_index).sigmoid() > 0.5).long()

    top_exp_nodes = node_mask.view(-1).sort(descending=True)[1][:max_nodes] if max_nodes is not None else \
        node_mask.view(-1).sort(descending=True)[1]
    explanation_nodes = top_exp_nodes[torch.isin(top_exp_nodes, sub_nodes)]
    explanation_subgraph_edge_index = subgraph(explanation_nodes, computation_edge_index)[0]

    if model_mode == ModelMode.multiclass_classification:
        explain_y_hat = model(x * node_mask, explanation_subgraph_edge_index).softmax(dim=-1).argmax(dim=-1)
    else:
        if model_config.task_level == ModelTaskLevel.edge:
            explain_y_hat = (model(x * node_mask, explanation_subgraph_edge_index,
                                   edge_label_index=index.view(2, 1)).sigmoid() > 0.5).long()
        else:
            explain_y_hat = (model(x * node_mask, explanation_subgraph_edge_index).sigmoid() > 0.5).long()

    top_nonexp_nodes = (1 - node_mask).view(-1).sort(descending=True)[1]
    non_explanation_nodes = top_nonexp_nodes[torch.isin(top_nonexp_nodes, sub_nodes)]
    non_explanation_subgraph_edge_index = subgraph(non_explanation_nodes, computation_edge_index)[0]

    if model_mode == ModelMode.multiclass_classification:
        complement_y_hat = model(x * (1 - node_mask), non_explanation_subgraph_edge_index).softmax(
            dim=-1).argmax(dim=-1)
    else:
        if model_config.task_level == ModelTaskLevel.edge:
            complement_y_hat = (model(x * (1 - node_mask), non_explanation_subgraph_edge_index,
                                      edge_label_index=index.view(2, 1)).sigmoid() > 0.5).long()
        else:
            complement_y_hat = (model(x * (1 - node_mask), non_explanation_subgraph_edge_index).sigmoid() > 0.5).long()

    if model_config.task_level == ModelTaskLevel.node and not full_graph:
        y = y[index]
        y_hat = y_hat[index]
        explain_y_hat = explain_y_hat[index]
        complement_y_hat = complement_y_hat[index]

    pos_fidelity = ((y_hat == y).float() - (complement_y_hat == y).float()).abs()
    neg_fidelity = ((y_hat == y).float() - (explain_y_hat == y).float()).abs()

    return pos_fidelity.item(), neg_fidelity.item()

def custom_unfaithfulness():
    pass


def get_motif_nodes(start, end, step, node):
    if not (start <= node <= end):
        raise ValueError("Node index must be between start and end (inclusive)")

    start_index = (node - start) // step
    subseq_start = start + start_index * step
    subseq_end = min(subseq_start + step, end + 1)

    return torch.arange(subseq_start, subseq_end)
