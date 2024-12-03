import numpy as np
import torch
import einops
import torch.nn.functional as F
from utils import to_numpy

def top_k_acc(knn_labels, gt_labels, k):
    accuracy_per_sample = torch.any(knn_labels[:, :k] == gt_labels, dim=1).float()
    return torch.mean(accuracy_per_sample)


def getTargetsSingle(targets):
    targets_sg = torch.zeros(len(targets)*2)
    i = 0
    j = 0
    while (i < len(targets)):
        targets_sg[j] = targets[i]
        targets_sg[j+1] = targets[i]
        i += 1
        j += 2
    
    return targets_sg

def calculate_metrics(predictions, gt_labels):
    true_positives = torch.sum((predictions == gt_labels) & (predictions == 1))
    false_positives = torch.sum((predictions != gt_labels) & (predictions == 1))
    false_negatives = torch.sum((predictions != gt_labels) & (predictions == 0))

    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    return precision.item(), recall.item(), f1_score.item()

class Evaluator(object):


    def __init__(self, model, n=2):

        self.model = model

        self.extract_device = 'cuda:2'
        self.eval_device = 'cuda:2'

        self.num_classes = self.model.num_classes
        self.embed_dim = self.model.embed_dim
        self.n = n

    @torch.no_grad()
    def extract(self, dataloader):
  
        self.model.eval()
        self.model.to(self.extract_device)

        num_collections = len(dataloader.dataset)
        num_total_images = num_collections * self.n


        results_dict = {'single': {'logits': np.zeros((num_total_images, self.num_classes)),
                                  'classes': np.zeros(num_total_images),
                                  'paths': []},
                       'mv_collection': {'logits': np.zeros((num_collections, self.num_classes)),
                                    'classes': np.zeros(num_collections),
                                    'paths': []}}

        if self.n == 1:
            del results_dict['mv_collection']

        s = 0
        for i, data in enumerate(dataloader):
            image_pair, targets = data
            targets_sg = getTargetsSingle(targets)
            image_vi, image_ir = image_pair
            images = torch.stack([image_vi, image_ir])
            images = einops.rearrange(images, 'n b c h w -> b n c h w')
            images = images.to(self.extract_device)
            batch_output = self.model(images)
            e = s + len(images)
            # e_sg = s_sg + 2 * len(images)
            for view_type in batch_output:
                if view_type == 'single':
                    multiplier = self.n
                    targets_temp = targets_sg
                else:
                    multiplier = 1
                    targets_temp = targets
                # print(len(batch_output[view_type]['logits']))
                # if view_type == 'single':
                #     results_dict[view_type]['logits'][s_sg:e_sg] = to_numpy(batch_output[view_type]['logits'])
                #     results_dict[view_type]['classes'][s_sg:e_sg] = to_numpy(targets_sg)
                # if view_type == 'mv_collection':
                #     results_dict[view_type]['logits'][s:e] = to_numpy(batch_output[view_type]['logits'])
                #     results_dict[view_type]['classes'][s:e] = to_numpy(targets)
                # results_dict[view_type]['paths'].extend()
                results_dict[view_type]['logits'][s*multiplier: e*multiplier] = to_numpy(batch_output[view_type]['logits'])
                results_dict[view_type]['classes'][s*multiplier: e*multiplier] = to_numpy(targets_temp)
            s = e
            # s_sg = e_sg

        

        for view_type in results_dict:
            results_dict[view_type]['logits'] = torch.from_numpy(results_dict[view_type]['logits']).squeeze()
            results_dict[view_type]['classes'] = results_dict[view_type]['classes'].squeeze()

        return results_dict

    def get_metrics(self, logits, gt_labels):

        try:
            logits = logits
            gt_labels = gt_labels
            predictions = torch.argsort(logits, dim=1, descending=True)
        except torch.cuda.OutOfMemoryError:
            logits = logits.cpu()
            gt_labels = gt_labels.cpu()
            predictions = torch.argsort(logits, dim=1, descending=True)

        class_1 = top_k_acc(predictions, gt_labels, 1)
        class_2 = top_k_acc(predictions, gt_labels, 2)
        class_5 = top_k_acc(predictions, gt_labels, 5)
        class_10 = top_k_acc(predictions, gt_labels, 10)
        class_100 = top_k_acc(predictions, gt_labels, 100)

        binary_predictions = (predictions[:, 0] == gt_labels).float()
        precision, recall, f1_score = calculate_metrics(binary_predictions, gt_labels.float())

        metrics = {
            'top1_acc': class_1.item(),
            'top2_acc': class_2.item(),
            'top5_acc': class_5.item(),
            'top10_acc': class_10.item(),
            'top100_acc': class_100.item(),
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
        }
        return metrics

    def evaluate(self, dataloader):

        results_dict = self.extract(dataloader)
        metrics_dict = {}

        for view_type in results_dict:
            gt_classes = np.expand_dims(results_dict[view_type]['classes'], -1)
            metrics = self.get_metrics(results_dict[view_type]['logits'].to(self.eval_device), torch.from_numpy(gt_classes).to(self.eval_device))
            metrics_dict[view_type] = metrics

        return metrics_dict
