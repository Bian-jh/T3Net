from typing import Dict, List, Tuple

import copy
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops
import math
import scipy
import torchvision
import matplotlib.pyplot as plt

from models.bricks.denoising import GenerateCDNQueries
from models.bricks.losses import sigmoid_focal_loss
from models.detectors.base_detector import DNDETRDetector


def cos_loss(output, target):
    B = output.shape[0]
    output = output.view(B, -1)
    target = target.view(B, -1)
    loss = torch.mean(1-F.cosine_similarity(output, target))
    return loss


class DensityCriterion(nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        beta: float = 0.6,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.crit = torch.nn.MSELoss(reduction='mean')
        self.count_crit = torch.nn.L1Loss()

    def get_crowd_distance(self, points, query_point, k=3):
        leafsize = 2048
        # build kdtree
        tree = scipy.spatial.KDTree(points, leafsize=leafsize)
        # query kdtree, k includes the query point itself
        distances, locations = tree.query(query_point, k=k + 1)

        return distances[1:].mean()

    def asy_gaussian_pdf(self, x, y, mu_x, mu_y, left_sigma, right_sigma, up_sigma, bottom_sigma):
        const = np.log(2.) - np.log(math.pi) - np.log(left_sigma + right_sigma) - np.log(up_sigma + bottom_sigma)
        if x < mu_x:
            sigma_x = left_sigma
        else:
            sigma_x = right_sigma
        if y < mu_y:
            sigma_y = up_sigma
        else:
            sigma_y = bottom_sigma
        log_pdf = const - (x - mu_x) ** 2 / (2 * sigma_x ** 2) - (y - mu_y) ** 2 / (2 * sigma_y ** 2)
        return np.exp(log_pdf)

    def forward(self, foreground_mask, targets, image_sizes):
        gt_boxes_list = []
        center_points = []
        for t, (img_h, img_w) in zip(targets, image_sizes):
            boxes = t["boxes"]
            boxes = box_ops._box_cxcywh_to_xyxy(boxes)
            scale_factor = torch.tensor([img_w, img_h, img_w, img_h], device=boxes.device)
            gt_boxes_list.append((boxes * scale_factor) / 4)  # x1, y1, x2, y2

            center_point = (t["boxes"][:, :2] * torch.tensor([img_w, img_h], device=boxes.device)) // 4
            center_points.append(center_point.int())  # cx, cy

        density_map1 = np.zeros(foreground_mask[0].shape)
        # calculate size-based and crowd-based sigma and get the density map
        for idx, (coord, center_point) in enumerate(zip(gt_boxes_list, center_points)):
            coord = coord.cpu().numpy()
            center_point = center_point.cpu().numpy()
            n = len(coord)
            if n == 0:
                continue

            for i in range(n):
                crowd_sigma = []
                size_sigma = []
                center_x, center_y = center_point[i]

                x1, y1, x2, y2 = coord[i]
                size_sigma.append([(x2 - x1) / 2, (x2 - x1) / 2, (y2 - y1) / 2, (y2 - y1) / 2])
                size_sigma = np.array(size_sigma)
                x1 = int(max(0, x1))
                x2 = int(min(density_map1.shape[3] - 1, x2))
                y1 = int(max(0, y1))
                y2 = int(min(density_map1.shape[2] - 1, y2))
                # left points
                left_points = center_point[center_point[:, 0] <= center_x]
                left_distances = self.get_crowd_distance(left_points, center_point[i])
                # right points
                right_points = center_point[center_point[:, 0] >= center_x]
                right_distances = self.get_crowd_distance(right_points, center_point[i])
                # up points
                up_points = center_point[center_point[:, 1] <= center_y]
                up_distances = self.get_crowd_distance(up_points, center_point[i])
                # bottom points
                bottom_points = center_point[center_point[:, 1] >= center_y]
                bottom_distances = self.get_crowd_distance(bottom_points, center_point[i])

                crowd_sigma.append([left_distances, right_distances, up_distances, bottom_distances])
                crowd_sigma = np.array(crowd_sigma)
                crowd_sigma = size_sigma * (1 + np.log(1.0 + (1 / (crowd_sigma * 0.05 + 1e-5))))
                # if np.any(np.isinf(crowd_sigma)):
                #     crowd_sigma = size_sigma
                sigma = self.beta * size_sigma + (1 - self.beta) * crowd_sigma

                X, Y = np.meshgrid(np.arange(x1, x2 + 1), np.arange(y1, y2 + 1))
                values = np.vectorize(self.asy_gaussian_pdf)(X, Y, center_x, center_y, sigma[0, 0], sigma[0, 1],
                                                             sigma[0, 2], sigma[0, 3])
                # values = np.exp(values)
                values = values / values.sum()
                density_map1[idx, 0, y1:y2 + 1, x1:x2 + 1] += values

        density_map1 = torch.from_numpy(density_map1).to(foreground_mask[0].device).float()

        density_targets = []
        density_targets.append(density_map1)
        density_map2 = F.interpolate(density_map1, size=(foreground_mask[1].shape[2], foreground_mask[1].shape[3]))
        density_targets.append(density_map2)
        density_map3 = F.interpolate(density_map1, size=(foreground_mask[2].shape[2], foreground_mask[2].shape[3]))
        density_targets.append(density_map3)

        spatial_loss = cos_loss(foreground_mask[0].sigmoid(), density_map1) + cos_loss(foreground_mask[1].sigmoid(), density_map2) + cos_loss(foreground_mask[2].sigmoid(), density_map3)
        spatial_loss /= density_map1.shape[0]

        counting_loss = self.count_crit(foreground_mask[0].sigmoid().sum(dim=(2, 3)), density_map1.sum(dim=(2, 3))) + self.count_crit(foreground_mask[1].sigmoid().sum(dim=(2, 3)), density_map2.sum(dim=(2, 3))) + self.count_crit(foreground_mask[2].sigmoid().sum(dim=(2, 3)), density_map3.sum(dim=(2, 3)))

        density_targets = torch.cat([e.flatten(-2) for e in density_targets], -1)
        density_targets = density_targets.squeeze(1)
        foreground_mask = torch.cat([e.flatten(-2) for e in foreground_mask], -1)
        foreground_mask = foreground_mask.squeeze(1)
        # weight = (density_targets > 0).float() * 10 + (density_targets == 0).float() * 0.1
        mask = (density_targets > 0).float()
        num_pos = torch.sum(density_targets > 0.0).clamp_(min=1)
        # salience_loss = (self.crit(foreground_mask.sigmoid(), density_targets) + 10 * spatial_loss) / density_map1.shape[0]

        # salience_loss = (torch.mean(weight * (foreground_mask.sigmoid()-density_targets)**2) + spatial_loss) / density_map1.shape[0]
        global_loss = self.crit(foreground_mask.sigmoid(), density_targets)
        local_loss = torch.sum(mask * (foreground_mask.sigmoid()-density_targets+1.) ** self.gamma * (foreground_mask.sigmoid()-density_targets) ** 2) / num_pos
        return {"loss_spatial": spatial_loss, "loss_global": global_loss, "loss_local": local_loss, "loss_count": counting_loss}
        # return {"loss_global": global_loss, "loss_local": local_loss, "loss_count": counting_loss}


class T3Net(DNDETRDetector):
    def __init__(
        # model structure
        self,
        backbone: nn.Module,
        neck: nn.Module,
        position_embedding: nn.Module,
        transformer: nn.Module,
        criterion: nn.Module,
        postprocessor: nn.Module,
        focus_criterion: nn.Module,
        # model parameters
        num_classes: int = 11,
        num_queries: int = 900,
        denoising_nums: int = 100,
        # model variants
        aux_loss: bool = True,
        min_size: int = None,
        max_size: int = None,
        second_stage: bool = False,
    ):
        super().__init__(min_size, max_size)
        # define model parameters
        self.num_classes = num_classes
        self.aux_loss = aux_loss
        embed_dim = transformer.embed_dim

        # define model structures
        self.backbone = backbone
        self.neck = neck
        self.position_embedding = position_embedding
        self.transformer = transformer
        self.criterion = criterion
        self.postprocessor = postprocessor
        self.label_encoder = nn.Embedding(num_classes, embed_dim)
        self.denoising_nums = denoising_nums
        self.label_noise_prob = 0.5
        self.box_noise_scale = 1.0
        self.num_classes = num_classes
        self.label_embed_dim = embed_dim
        self.focus_criterion = focus_criterion
        self.second_stage = second_stage

    def forward(self, images: List[Tensor], targets: List[Dict] = None):
        # get original image sizes, used for postprocess
        original_image_sizes = self.query_original_sizes(images)
        images, targets, mask = self.preprocess(images, targets)

        # extract features
        multi_level_feats = self.backbone(images.tensors)
        multi_level_feats = self.neck(multi_level_feats)

        multi_level_masks = []
        multi_level_position_embeddings = []
        for feature in multi_level_feats:
            multi_level_masks.append(F.interpolate(mask[None], size=feature.shape[-2:]).to(torch.bool)[0])
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))

        dn_args = [self.denoising_nums, self.label_noise_prob, self.box_noise_scale, self.num_classes,
                   self.label_embed_dim, self.label_encoder]

        # feed into transformer
        outputs_class, outputs_coord, enc_class, enc_coord, foreground_mask, query, noise_query, penalty, sigma, \
            denoising_groups, max_gt_num_per_image, num_select = self.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            dn_args,
            targets,
        )
        # hack implementation for distributed training
        outputs_class[0] += self.label_encoder.weight[0, 0] * 0.0

        # denoising postprocessing
        if denoising_groups is not None and max_gt_num_per_image is not None:
            dn_metas = {
                "denoising_groups": denoising_groups,
                "max_gt_num_per_image": max_gt_num_per_image,
            }
            outputs_class, outputs_coord = self.dn_post_process(outputs_class, outputs_coord, dn_metas)

        # prepare for loss computation
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        # prepare two stage output
        output["enc_outputs"] = {"pred_logits": enc_class, "pred_boxes": enc_coord}

        if self.training:
            # compute loss
            loss_dict = self.criterion(output, targets)
            dn_losses = self.compute_dn_loss(dn_metas, targets)
            loss_dict.update(dn_losses)

            focus_loss = self.focus_criterion(foreground_mask, targets, images.image_sizes)
            loss_dict.update(focus_loss)

            # loss reweighting
            weight_dict = self.criterion.weight_dict
            loss_dict = dict((k, loss_dict[k] * weight_dict[k]) for k in loss_dict.keys() if k in weight_dict)
            weight_dict_sigma = {"loss_restore": 10, "loss_sigma": 0.01}
            loss_dict_sigma = dict()
            loss_dict_sigma["loss_restore"] = F.mse_loss(F.relu(noise_query), F.relu(query).detach())
            loss_dict_sigma["loss_sigma"] = -penalty
            loss_dict_sigma = dict((k, loss_dict_sigma[k] * weight_dict_sigma[k]) for k in loss_dict_sigma.keys() if k in weight_dict_sigma)
            return loss_dict, loss_dict_sigma

        if self.second_stage:
            select_number = num_select
        else:
            select_number = None

        detections = self.postprocessor(output, original_image_sizes, select_number)
        return detections, foreground_mask
