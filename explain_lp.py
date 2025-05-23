import time

import numpy as np
import torch
from torch_geometric.explain import unfaithfulness, fidelity, characterization_score, ModelConfig, Explainer, \
    groundtruth_metrics
from torch_geometric.utils import degree, k_hop_subgraph

from explainer_store import get_explainer
from model_store import get_gnn
from utils import setup_models, custom_fidelity, custom_recall, custom_precision, custom_iou, get_motif_nodes


def evaluate_lp_explainer_on_data(explainer, data, edge_label_indices, metric_names, test_nodes_start, test_nodes_end,
                                  num_motif_nodes, use_prob=False):
    eval_metrics = {metric_name: 0 for metric_name in metric_names}
    accs, precisions, recalls, ious, pos_fids, neg_fids, us, inf_times = [], [], [], [], [], [], [], []
    for idx in range(edge_label_indices.size(1)):
        start = time.time()
        explanation = explainer(
            x=data.x,
            edge_index=data.edge_index,
            target=data.edge_label_prob[idx] if use_prob else data.edge_label[idx],
            edge_label_index=edge_label_indices[:, idx].view(-1, 1),
        )
        end = time.time()
        inference_time = end - start
        inf_times.append(inference_time)
        sub_nodes, _, _, hard_edge_mask = k_hop_subgraph(explanation.edge_label_index.view(-1),
                                                         num_hops=explainer.model.model.num_layers,
                                                         edge_index=explanation.edge_index)
        explanation.edge_mask = explanation.edge_mask.float() if explanation.edge_mask.dtype == torch.bool else explanation.edge_mask
        if 'node_mask' in explanation:
            node_mask = explanation.node_mask
        else:
            nodes = explanation.edge_index[:, explanation.edge_mask.bool()].view(-1).unique()
            node_mask = torch.zeros(data.x.size(0), 1, device=data.x.device)
            node_mask[nodes] = 1
        exp_nodes = (node_mask.view(-1).sort(descending=True)[1][:num_motif_nodes]).detach().cpu()
        gt_nodes = (get_motif_nodes(test_nodes_start, test_nodes_end, num_motif_nodes,
                                   explanation.edge_label_index[0].item())).detach().cpu()
        iou = custom_iou(gt_nodes, exp_nodes)
        precision = custom_precision(gt_nodes, exp_nodes)
        recall = custom_recall(gt_nodes, exp_nodes)
        gt, pred = data.node_mask[sub_nodes], node_mask[sub_nodes].view(-1)
        acc = groundtruth_metrics(pred, gt, metrics=['accuracy'], threshold=0.0)

        # iou = binary_jaccard_index(pred, gt, threshold=threshold).cpu()
        pos_fidelity, neg_fidelity = custom_fidelity(explainer, explanation, node_mask)
        u = unfaithfulness(explainer, explanation)
        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        ious.append(iou)
        pos_fids.append(pos_fidelity)
        neg_fids.append(neg_fidelity)
        us.append(u)

    eval_metrics['accuracy'] = np.mean(accs)
    eval_metrics['precision'] = np.mean(precisions)
    eval_metrics['recall'] = np.mean(recalls)
    eval_metrics['iou'] = np.mean(ious)
    eval_metrics['fid+'] = np.mean(pos_fids)
    eval_metrics['fid-'] = np.mean(neg_fids)
    eval_metrics['unfaithfulness'] = np.mean(us)
    eval_metrics['inference_time'] = np.mean(inf_times)
    eval_metrics['characterization_score'] = characterization_score(eval_metrics['fid+'], eval_metrics['fid-'])
    return eval_metrics


def evaluate_lp_explainer(model_path, explainer_name, explainer_config, lp_datasets, metric_names, std=None):
    start_time = time.time()

    print(f'{"-" * 2} Evaluating {explainer_name} explainer on link prediction datasets...')
    exp_eval_metrics = {}
    model_config = ModelConfig(mode='binary_classification', task_level='edge', return_type='raw')
    #for dataset_name, train_data, val_data, test_data in lp_datasets:
    for dataset_name, _, _, test_data in lp_datasets:
        dataset = test_data
        lp_dataset_start_time = time.time()
        motif_nodes = (test_data.y > 0).nonzero().view(-1)
        start = motif_nodes[0].item()
        end = motif_nodes[-1].item()
        step = 5 if 'ba' in dataset_name else 9
        print(f'{"-" * 3} Evaluating {explainer_name} explainer on {dataset_name} dataset...')

        lp_models = get_gnn(model_path, 'lp', 'all', dataset_name, std=std)
        lp_models = setup_models(lp_models, dataset.x.device)

        pos_edges = dataset.edge_label_index[:, (dataset.motif_edge_label.bool() & dataset.edge_label.bool())]
        deg = degree(dataset.edge_index[0], num_nodes=dataset.num_nodes)
        degree_zero_nodes = torch.nonzero(deg == 0, as_tuple=True)[0]
        source_nodes = pos_edges[0]
        target_nodes = pos_edges[1]

        mask = ~torch.isin(source_nodes, degree_zero_nodes) & ~torch.isin(target_nodes, degree_zero_nodes)

        num_examples = 200
        edge_label_indices = pos_edges[:, mask][:, :num_examples]
        for model_name, model in lp_models:
            lp_model_start_time = time.time()
            model_name = f'{model_name}-{std}'
            print(f'{"-" * 5} Evaluating {explainer_name} explainer on {model_name} model...')
            if (explainer_name, model_name) not in exp_eval_metrics:
                exp_eval_metrics[(explainer_name, model_name)] = {}

            use_prob = False
            explainer = get_explainer(explainer_name, explainer_config, model, model_config, dataset=dataset,
                                      dataset_name=dataset_name, edge_label_indices=edge_label_indices)
            if explainer_name == 'ciexplainer':
                use_prob = True
                edge_label_prob = torch.zeros_like(dataset.edge_label[:edge_label_indices.size(1)], dtype=torch.float64,
                                                   device=dataset.edge_label.device)
                for idx in range(edge_label_prob.size(0)):
                    edge_label_prob[idx] = model(dataset.x, dataset.edge_index,
                                                 edge_label_index=edge_label_indices[:, idx].view(-1, 1)).sigmoid()
                dataset.edge_label_prob = edge_label_prob
            res = evaluate_lp_explainer_on_data(explainer, dataset, edge_label_indices, metric_names, start, end, step,
                                                use_prob=use_prob)
            for metric_name, metric_value in res.items():
                exp_eval_metrics[(explainer_name, model_name)][(dataset_name, metric_name)] = metric_value

            lp_model_end_time = time.time()
            lp_mode_elapsed_time = (lp_model_end_time - lp_model_start_time) / 60
            print(f'{"-" * 7} Evaluation on {model_name} model took {lp_mode_elapsed_time:.2f} minutes.')

        lp_dataset_end_time = time.time()
        lp_dataset_elapsed_time = (lp_dataset_end_time - lp_dataset_start_time) / 60
        print(f'{"-" * 6} Evaluation on {dataset_name} dataset took {lp_dataset_elapsed_time:.2f} minutes.')

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f'{"-" * 3} Evaluation on node classification took {elapsed_time:.2f} minutes.')
    return exp_eval_metrics

# def evaluate_lp_explainer_on_data(explainer, data, edge_label_indices, metric_names, use_prob=False):
#     eval_metrics = {metric_name: 0 for metric_name in metric_names}
#     pos_fids, neg_fids, us, inf_times = [], [], [], []
#     for idx in range(edge_label_indices.size(1)):
#         start = time.time()
#         explanation = explainer(
#             x=data.x,
#             edge_index=data.edge_index,
#             target=data.edge_label_prob[idx] if use_prob else data.edge_label[idx],
#             edge_label_index=edge_label_indices[:, idx].view(-1, 1),
#         )
#         end = time.time()
#         inference_time = end - start
#         inf_times.append(inference_time)
#         pos_fidelity, neg_fidelity = fidelity(explainer, explanation)
#         u = unfaithfulness(explainer, explanation)
#         pos_fids.append(pos_fidelity)
#         neg_fids.append(neg_fidelity)
#         us.append(u)
#
#     eval_metrics['fid+'] = np.mean(pos_fids)
#     eval_metrics['fid-'] = np.mean(neg_fids)
#     eval_metrics['unfaithfulness'] = np.mean(us)
#     eval_metrics['inference_time'] = np.mean(inf_times)
#     eval_metrics['characterization_score'] = characterization_score(eval_metrics['fid+'], eval_metrics['fid-'])
#     return eval_metrics
#
# # def evaluate_lp_explainer(edge_label_indices, explainer_name, explainer_algo, explainer_config, lp_datasets, lp_models, metric_names):
# #     start_time = time.time()
# #
# #     print(f'{"-" * 2} Evaluating {explainer_name} explainer on link prediction datasets...')
# #     exp_eval_metrics = {}
# #     model_config = ModelConfig(mode='binary_classification', task_level='edge', return_type='raw')
# #     # for dataset_name, train_data, val_data, test_data in lp_datasets:
# #     for dataset_name, dataset in lp_datasets:
# #         lp_dataset_start_time = time.time()
# #
# #         print(f'{"-" * 3} Evaluating {explainer_name} explainer on {dataset_name} dataset...')
# #
# #         for model_name, model in lp_models:
# #             lp_model_start_time = time.time()
# #
# #             print(f'{"-" * 5} Evaluating {explainer_name} explainer on {model_name} model...')
# #             if (explainer_name, model_name) not in exp_eval_metrics:
# #                 exp_eval_metrics[(explainer_name, model_name)] = {}
# #
# #             use_prob = False
# #             explainer = Explainer(
# #                 model=model,
# #                 algorithm=explainer_algo,
# #                 explanation_type=explainer_config.explanation_type,
# #                 node_mask_type=explainer_config.node_mask_type,
# #                 edge_mask_type=explainer_config.edge_mask_type,
# #                 model_config=model_config,
# #             )
# #
# #             res = evaluate_lp_explainer_on_data(explainer, dataset, edge_label_indices, metric_names, use_prob=use_prob)
# #             for metric_name, metric_value in res.items():
# #                 exp_eval_metrics[(explainer_name, model_name)][(dataset_name, metric_name)] = metric_value
# #
# #             lp_model_end_time = time.time()
# #             lp_mode_elapsed_time = (lp_model_end_time - lp_model_start_time) / 60
# #             print(f'{"-" * 7} Evaluation on {model_name} model took {lp_mode_elapsed_time:.2f} minutes.')
# #
# #         lp_dataset_end_time = time.time()
# #         lp_dataset_elapsed_time = (lp_dataset_end_time - lp_dataset_start_time) / 60
# #         print(f'{"-" * 6} Evaluation on {dataset_name} dataset took {lp_dataset_elapsed_time:.2f} minutes.')
# #
# #     end_time = time.time()
# #     elapsed_time = (end_time - start_time) / 60
# #     print(f'{"-" * 3} Evaluation on node classification took {elapsed_time:.2f} minutes.')
# #     return exp_eval_metrics
#
# def evaluate_lp_explainer(model_path, explainer_name, explainer_config, lp_datasets, metric_names):
#     start_time = time.time()
#
#     print(f'{"-" * 2} Evaluating {explainer_name} explainer on link prediction datasets...')
#     exp_eval_metrics = {}
#     model_config = ModelConfig(mode='binary_classification', task_level='edge', return_type='raw')
#     # for dataset_name, train_data, val_data, test_data in lp_datasets:
#     for dataset_name, test_data in lp_datasets:
#         dataset = test_data
#         lp_dataset_start_time = time.time()
#
#         print(f'{"-" * 3} Evaluating {explainer_name} explainer on {dataset_name} dataset...')
#         lp_models = get_gnn(model_path, 'lp', 'all', dataset_name)
#         lp_models = setup_models(lp_models, dataset.x.device)
#
#         pos_edges = dataset.edge_label_index[:, dataset.edge_label.bool()]
#         deg = degree(dataset.edge_index[0], num_nodes=dataset.num_nodes)
#         degree_zero_nodes = torch.nonzero(deg == 0, as_tuple=True)[0]
#         source_nodes = pos_edges[0]
#         target_nodes = pos_edges[1]
#
#         mask = ~torch.isin(source_nodes, degree_zero_nodes) & ~torch.isin(target_nodes, degree_zero_nodes)
#
#         num_examples = 200
#         edge_label_indices = pos_edges[:, mask][:, :num_examples]
#         for model_name, model in lp_models:
#             lp_model_start_time = time.time()
#
#             print(f'{"-" * 5} Evaluating {explainer_name} explainer on {model_name} model...')
#             if (explainer_name, model_name) not in exp_eval_metrics:
#                 exp_eval_metrics[(explainer_name, model_name)] = {}
#
#             use_prob = False
#             explainer = get_explainer(explainer_name, explainer_config, model, model_config, dataset=dataset,
#                                       dataset_name=dataset_name, edge_label_indices=edge_label_indices)
#             if explainer_name == 'ciexplainer':
#                 use_prob = True
#                 edge_label_prob = torch.zeros_like(dataset.edge_label[:num_examples], dtype=torch.float64,
#                                                    device=dataset.edge_label.device)
#                 for idx in range(edge_label_prob.size(0)):
#                     edge_label_prob[idx] = model(dataset.x, dataset.edge_index,
#                                                  edge_label_index=edge_label_indices[:, idx].view(-1, 1)).sigmoid()
#                 dataset.edge_label_prob = edge_label_prob
#             res = evaluate_lp_explainer_on_data(explainer, dataset, edge_label_indices, metric_names, use_prob=use_prob)
#             for metric_name, metric_value in res.items():
#                 exp_eval_metrics[(explainer_name, model_name)][(dataset_name, metric_name)] = metric_value
#
#             lp_model_end_time = time.time()
#             lp_mode_elapsed_time = (lp_model_end_time - lp_model_start_time) / 60
#             print(f'{"-" * 7} Evaluation on {model_name} model took {lp_mode_elapsed_time:.2f} minutes.')
#
#         lp_dataset_end_time = time.time()
#         lp_dataset_elapsed_time = (lp_dataset_end_time - lp_dataset_start_time) / 60
#         print(f'{"-" * 6} Evaluation on {dataset_name} dataset took {lp_dataset_elapsed_time:.2f} minutes.')
#
#     end_time = time.time()
#     elapsed_time = (end_time - start_time) / 60
#     print(f'{"-" * 3} Evaluation on link prediction took {elapsed_time:.2f} minutes.')
#     return exp_eval_metrics
