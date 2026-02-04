import copy
import math
from typing import Tuple

import torch
import torchvision
from torch import nn

from models.bricks.base_transformer import TwostageTransformer
from models.bricks.basic import MLP
from models.bricks.ms_deform_attn import MultiScaleDeformableAttention
from models.bricks.position_encoding import PositionEmbeddingLearned, get_sine_pos_embed
from util.misc import inverse_sigmoid
from models.bricks.denoising import prepare_for_cdn


class MaskPredictor(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.h_dim = h_dim
        self.layer1 = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, h_dim),
            nn.GELU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.GELU(),
            nn.Linear(h_dim // 2, h_dim // 4),
            nn.GELU(),
            nn.Linear(h_dim // 4, 1),
        )

        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        z = self.layer1(x)
        z_local, z_global = torch.split(z, self.h_dim // 2, dim=-1)
        z_global = z_global.mean(dim=1, keepdim=True).expand(-1, z_local.shape[1], -1)
        z = torch.cat([z_local, z_global], dim=-1)
        out = self.layer2(z)
        return out


class NoiseLayer(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.h_dim = h_dim
        self.sigma = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.GELU(),
            nn.Linear(h_dim // 2, h_dim // 4),
            nn.GELU(),
            nn.Linear(h_dim // 4, 1),
            # nn.Sigmoid(),
        )

        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, input, unit_noise):
        # add noise to the selected tokens
        sigma = self.sigma(input).sigmoid().clone() * 0.2
        sigma.clamp_(min=0.001)
        # print(sigma.min(), sigma.max())
        noise_input = input + sigma.repeat(1, 1, self.h_dim) * unit_noise
        sigma_vector = torch.log(sigma)
        penalty = torch.mean(sigma_vector) + 0.5 * torch.log(torch.tensor(2 * math.pi)) + torch.tensor(0.5)

        return noise_input, penalty, sigma


class MessageAgg(nn.Module):
    def __init__(self, agg_method="mean"):
        super().__init__()
        self.agg_method = agg_method

    def forward(self, X, path):
        """
            X: [n_node, dim]
            path: col(source) -> row(target)
        """
        X = torch.matmul(path, X)
        if self.agg_method == "mean":
            norm_out = 1 / torch.sum(path, dim=2, keepdim=True)
            norm_out[torch.isinf(norm_out)] = 0
            X = norm_out * X
            return X
        elif self.agg_method == "sum":
            pass
        return X


class HyPConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc = nn.Linear(c1, c2)
        self.v2e = MessageAgg(agg_method="mean")
        self.e2v = MessageAgg(agg_method="mean")

    #     self.init_weights()
    #
    # def init_weights(self):
    #     nn.init.constant_(self.fc.weight, 0)
    #     nn.init.constant_(self.fc.bias, 0)

    def forward(self, indiscriminative_query, discriminative_query, H):
        indiscriminative_query = self.fc(indiscriminative_query)
        # v -> e
        E = self.v2e(indiscriminative_query, H.transpose(1, 2).contiguous()) + discriminative_query
        # E = self.v2e(indiscriminative_query, H.transpose(1, 2).contiguous())
        # e -> v
        v = self.e2v(E, H)

        return v


class Gated_Fusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gating = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(dim)

        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, fea1, fea2):
        gate = self.gating(torch.cat((fea1, fea2), dim=2))
        output = gate * fea1 + (1 - gate) * fea2
        output = self.norm(output)

        return output


class HyperComputeModule(nn.Module):
    def __init__(self, embed_dim=256, threshold=8., d_ffn=1024, dropout=0.1, activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.threshold = threshold
        self.hgconv = HyPConv(embed_dim, embed_dim)

        # self.bn = nn.LayerNorm(embed_dim)
        # self.act = nn.ReLU()

        # ffn
        self.linear1 = nn.Linear(embed_dim, d_ffn)
        self.activation = activation
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, embed_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward_ffn(self, query):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(query))))
        query = query + self.dropout3(src2)
        query = self.norm2(query)
        return query

    def forward(self, discriminative_query, indiscriminative_query):
        feature1 = indiscriminative_query
        feature2 = discriminative_query
        distance = torch.cdist(feature1, feature2)  # [b, nodes, edges]
        hg = distance < self.threshold
        hg = hg.float().to(discriminative_query.device).to(discriminative_query.dtype)
        enhanced_query = self.hgconv(indiscriminative_query, discriminative_query, hg).to(indiscriminative_query.device).to(indiscriminative_query.dtype) + indiscriminative_query
        enhanced_query = self.forward_ffn(enhanced_query)
        # enhanced_query = self.act(self.bn(enhanced_query))

        return enhanced_query


class TodTransformer(TwostageTransformer):
    def __init__(
        self,
        encoder: nn.Module,
        neck: nn.Module,
        decoder: nn.Module,
        num_classes: int,
        num_feature_levels: int = 4,
        two_stage_num_proposals: int = 900,
        level_filter_ratio: Tuple = (0.25, 0.5, 1.0, 1.0),
        layer_filter_ratio: Tuple = (1.0, 0.8, 0.6, 0.6, 0.4, 0.2),
        use_adaptive_query: bool = True,
        dynamic_query_list: Tuple = (300, 500, 900, 1500),
    ):
        super().__init__(num_feature_levels, encoder.embed_dim)
        # model parameters
        self.two_stage_num_proposals = two_stage_num_proposals
        self.num_classes = num_classes

        # salience parameters
        self.register_buffer("level_filter_ratio", torch.Tensor(level_filter_ratio))
        self.register_buffer("layer_filter_ratio", torch.Tensor(layer_filter_ratio))
        self.alpha = nn.Parameter(torch.Tensor(2), requires_grad=True)

        # model structure
        self.encoder = encoder
        self.neck = neck
        self.decoder = decoder
        self.tgt_embed = nn.Embedding(two_stage_num_proposals, self.embed_dim)
        self.encoder_class_head = nn.Linear(self.embed_dim, num_classes)
        self.encoder_bbox_head = MLP(self.embed_dim, self.embed_dim, 4, 3)

        self.enc_mask_predictor = MaskPredictor(self.embed_dim, self.embed_dim)

        self.hyper_module = HyperComputeModule()
        self.gated_fusion = Gated_Fusion(self.embed_dim)

        self.use_adaptive_query = use_adaptive_query
        self.dynamic_query_list = dynamic_query_list

        self.init_weights()

    def init_weights(self):
        # initialize embedding layers
        nn.init.normal_(self.tgt_embed.weight)
        # initialize encoder classification layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.encoder_class_head.bias, bias_value)
        # initiailize encoder regression layers
        nn.init.constant_(self.encoder_bbox_head.layers[-1].weight, 0.0)
        nn.init.constant_(self.encoder_bbox_head.layers[-1].bias, 0.0)
        # initialize alpha
        self.alpha.data.uniform_(-0.3, 0.3)

    def forward(
        self,
        multi_level_feats,
        multi_level_masks,
        multi_level_pos_embeds,
        dn_args,
        targets,
    ):
        # get input for encoder
        feat_flatten = self.flatten_multi_level(multi_level_feats)
        mask_flatten = self.flatten_multi_level(multi_level_masks)
        lvl_pos_embed_flatten = self.get_lvl_pos_embed(multi_level_pos_embeds)
        spatial_shapes, level_start_index, valid_ratios = self.multi_level_misc(multi_level_masks)

        backbone_output_memory = self.gen_encoder_output_proposals(
            feat_flatten + lvl_pos_embed_flatten, mask_flatten, spatial_shapes
        )[0]

        # calculate filtered tokens numbers for each feature map
        reverse_multi_level_masks = [~m for m in multi_level_masks]
        valid_token_nums = torch.stack([m.sum((1, 2)) for m in reverse_multi_level_masks], -1)
        focus_token_nums = (valid_token_nums * self.level_filter_ratio).int()
        level_token_nums = focus_token_nums.max(0)[0]
        focus_token_nums = focus_token_nums.sum(-1)

        # from high level to low level
        batch_size = feat_flatten.shape[0]
        selected_score = []
        selected_inds = []
        salience_score = []
        predict_levels = 3

        for level_idx in range(predict_levels - 1, -1, -1):
            start_index = level_start_index[level_idx]
            end_index = level_start_index[level_idx + 1] if level_idx < spatial_shapes.shape[0] - 1 else None
            level_memory = backbone_output_memory[:, start_index:end_index, :]
            mask = mask_flatten[:, start_index:end_index]
            # update the memory using the higher-level score_prediction
            if level_idx != predict_levels - 1:
                upsample_score = torch.nn.functional.interpolate(
                    score,
                    size=spatial_shapes[level_idx].unbind(),
                    mode="bilinear",
                    align_corners=True,
                )
                upsample_score = upsample_score.view(batch_size, -1, spatial_shapes[level_idx].prod())
                upsample_score = upsample_score.transpose(1, 2)
                level_memory = level_memory + level_memory * upsample_score * self.alpha[level_idx]
            # predict the foreground score of the current layer
            score = self.enc_mask_predictor(level_memory)
            valid_score = score.squeeze(-1).masked_fill(mask, score.min())
            score = score.transpose(1, 2).view(batch_size, -1, *spatial_shapes[level_idx])

            # get the topk salience index of the current feature map level
            level_score, level_inds = valid_score.topk(level_token_nums[level_idx], dim=1)
            level_inds = level_inds + level_start_index[level_idx]
            salience_score.append(score)
            selected_inds.append(level_inds)
            selected_score.append(level_score)

        selected_score = torch.cat(selected_score[::-1], 1)
        index = torch.sort(selected_score, dim=1, descending=True)[1]
        selected_inds = torch.cat(selected_inds[::-1], 1).gather(1, index)

        # further filtering to get foreground tokens
        num_inds = selected_inds.shape[1]
        # change dtype to avoid shape inference error during exporting ONNX
        cast_dtype = num_inds.dtype if torchvision._is_tracing() else torch.int64
        layer_filter_ratio = (num_inds * self.layer_filter_ratio).to(cast_dtype)
        selected_inds = [selected_inds[:, :r] for r in layer_filter_ratio]

        # batch_valid_nums = valid_token_nums.min(0)[0]
        # for level_idx in range(spatial_shapes.shape[0] - 1, predict_levels - 1, -1):
        # deeper layers do not apply token filtering
        add_inds = torch.arange(level_start_index[predict_levels], mask_flatten.shape[1], device=level_start_index.device).repeat(mask_flatten.shape[0], 1)
        selected_inds_all = [torch.cat((i, add_inds), dim=1) for i in selected_inds]

        salience_score = salience_score[::-1]
        foreground_score = self.flatten_multi_level(salience_score).squeeze(-1)
        foreground_score = foreground_score.masked_fill(mask_flatten[:, :level_start_index[predict_levels]], foreground_score.min())

        # transformer encoder
        memory, query, noise_query, penalty, sigma = self.encoder(
            query=feat_flatten,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # salience input
            focus_token_nums=focus_token_nums,
            foreground_inds=selected_inds_all,
            multi_level_masks=multi_level_masks,
        )

        # selected_sigma = int(sigma.shape[1] * 0.4)
        selected_sigma = int(sigma.shape[1] * 0.8)
        sigma_score, sigma_inds = sigma.squeeze(-1).topk(selected_sigma, dim=1, largest=False)

        select_query_inds = sigma_inds.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        select_query = torch.gather(query, 1, select_query_inds)  # [b, selected_sigma, embed_dim]
        select_inds = torch.gather(selected_inds_all[0], 1, sigma_inds)
        discriminative_num = int(select_query.shape[1] * 0.4)
        # print(selected_sigma, discriminative_num)
        discriminative_query = select_query[:, :discriminative_num, :]
        discriminative_inds = select_inds[:, :discriminative_num]
        indiscriminative_query = select_query[:, discriminative_num:, :]
        indiscriminative_inds = select_inds[:, discriminative_num:]

        # perform high-order information exchange between two distinct query groups
        indiscriminative_query_enhance = self.hyper_module(discriminative_query, indiscriminative_query)
        # indiscriminative_query = self.hyper_module(query, indiscriminative_query)
        indiscriminative_query_enhance = self.gated_fusion(indiscriminative_query_enhance, indiscriminative_query)

        memory = memory.scatter(1, indiscriminative_inds.unsqueeze(-1).repeat(1, 1, indiscriminative_query.size(-1)), indiscriminative_query_enhance)

        if self.neck is not None:
            # shapes = spatial_shapes.prod(-1).unbind()
            # shapes = tuple(int(i) for i in shapes)
            # print(spatial_shapes.prod(-1).unbind())
            mask_enhance = torch.zeros((memory.shape[0], memory.shape[1], 1), device=memory.device, dtype=memory.dtype)
            mask_enhance = mask_enhance.scatter(1, indiscriminative_inds.unsqueeze(-1), 1.)
            feat_unflatten = memory.split(spatial_shapes.prod(-1).unbind(), dim=1)
            mask_enhance_unflatten = mask_enhance.split(spatial_shapes.prod(-1).unbind(), dim=1)
            # feat_unflatten = memory.split(shapes, dim=1)
            feat_unflatten = dict((
                i,
                feat.transpose(1, 2).contiguous().reshape(-1, self.embed_dim, *spatial_shape),
            ) for i, (feat, spatial_shape) in enumerate(zip(feat_unflatten, spatial_shapes)))
            mask_enhance_unflatten = dict((
                i,
                feat.transpose(1, 2).contiguous().reshape(-1, 1, *spatial_shape),
            ) for i, (feat, spatial_shape) in enumerate(zip(mask_enhance_unflatten, spatial_shapes)))
            feat_unflatten = list(self.neck(feat_unflatten, mask_enhance_unflatten).values())
            memory = torch.cat([feat.flatten(2).transpose(1, 2) for feat in feat_unflatten], dim=1)

        # get encoder output, classes and coordinates
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes
        )
        enc_outputs_class = self.encoder_class_head(output_memory)
        enc_outputs_coord = self.encoder_bbox_head(output_memory) + output_proposals
        enc_outputs_coord = enc_outputs_coord.sigmoid()

        if not self.use_adaptive_query:
            num_select = self.two_stage_num_proposals
            # get topk output classes and coordinates
            if torchvision._is_tracing():
                topk = torch.min(torch.tensor(self.two_stage_num_proposals * 5), enc_outputs_class.shape[1])
            else:
                topk = min(self.two_stage_num_proposals * 5, enc_outputs_class.shape[1])
            topk_scores, topk_index = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)
            topk_index = self.nms_on_topk_index(
                topk_scores, topk_index, spatial_shapes, level_start_index, self.two_stage_num_proposals, iou_threshold=0.3
            ).unsqueeze(-1)
            enc_outputs_class = enc_outputs_class.gather(1, topk_index.expand(-1, -1, self.num_classes))
            enc_outputs_coord = enc_outputs_coord.gather(1, topk_index.expand(-1, -1, 4))

            # get target and reference points
            reference_points = enc_outputs_coord.detach()
            target = self.tgt_embed.weight.expand(multi_level_feats[0].shape[0], -1, -1)
        else:
            counting = salience_score[0].sigmoid().sum(dim=(2, 3)).max()  # the max counting number within the batch
            num_select = self.find_query_num(self.dynamic_query_list, int(counting))

            # get topk output classes and coordinates
            if torchvision._is_tracing():
                topk = torch.min(torch.tensor(num_select * 5), enc_outputs_class.shape[1])
            else:
                topk = min(num_select * 5, enc_outputs_class.shape[1])
            topk_scores, topk_index = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)
            topk_index = self.nms_on_topk_index(
                topk_scores, topk_index, spatial_shapes, level_start_index, num_select, iou_threshold=0.3
            ).unsqueeze(-1)
            enc_outputs_class = enc_outputs_class.gather(1, topk_index.expand(-1, -1, self.num_classes))
            enc_outputs_coord = enc_outputs_coord.gather(1, topk_index.expand(-1, -1, 4))

            # get target and reference points
            reference_points = enc_outputs_coord.detach()
            target = self.tgt_embed.weight[:num_select, :].expand(multi_level_feats[0].shape[0], -1, -1)

        if self.training:
            # collect ground truth for denoising generation
            gt_labels_list = [t["labels"] for t in targets]
            gt_boxes_list = [t["boxes"] for t in targets]
            noised_results = prepare_for_cdn(gt_labels_list, gt_boxes_list, dn_args, num_select)
            noised_label_query = noised_results[0]
            noised_box_query = noised_results[1]
            attn_mask = noised_results[2]
            denoising_groups = noised_results[3]
            max_gt_num_per_image = noised_results[4]
        else:
            noised_label_query = None
            noised_box_query = None
            attn_mask = None
            denoising_groups = None
            max_gt_num_per_image = None

        # combine with noised_label_query and noised_box_query for denoising training
        if noised_label_query is not None and noised_box_query is not None:
            target = torch.cat([noised_label_query, target], 1)
            reference_points = torch.cat([noised_box_query.sigmoid(), reference_points], 1)

        # decoder
        outputs_classes, outputs_coords = self.decoder(
            query=target,
            value=memory,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            attn_mask=attn_mask,
        )

        return outputs_classes, outputs_coords, enc_outputs_class, enc_outputs_coord, salience_score, query, noise_query,\
            penalty, sigma, denoising_groups, max_gt_num_per_image, num_select

    @staticmethod
    def fast_repeat_interleave(input, repeats):
        """torch.Tensor.repeat_interleave is slow for one-dimension input for unknown reasons. 
        This is a simple faster implementation. Notice the return shares memory with the input.

        :param input: input Tensor
        :param repeats: repeat numbers of each element in the specified dim
        :param dim: the dimension to repeat, defaults to None
        """
        # the following inplementation runs a little faster under one-dimension settings
        return torch.cat([aa.expand(bb) for aa, bb in zip(input, repeats)])

    @torch.no_grad()
    def find_query_num(self, numbers, target):
        # Filter numbers >= target
        candidates = [num for num in numbers if num >= target]

        if candidates:
            return min(candidates)
        else:
            return max(numbers)

    @torch.no_grad()
    def nms_on_topk_index(
        self, topk_scores, topk_index, spatial_shapes, level_start_index, num, iou_threshold=0.3
    ):
        batch_size, num_topk = topk_scores.shape
        if torchvision._is_tracing():
            num_pixels = spatial_shapes.prod(-1).unbind()
        else:
            num_pixels = spatial_shapes.prod(-1).tolist()

        # flatten topk_scores and topk_index for batched_nms
        topk_scores, topk_index = map(lambda x: x.view(-1), (topk_scores, topk_index))

        # get level coordinates for queries and construct boxes for them
        level_index = torch.arange(level_start_index.shape[0], device=level_start_index.device)
        feat_width, start_index, level_idx = map(
            lambda x: self.fast_repeat_interleave(x, num_pixels)[topk_index],
            (spatial_shapes[:, 1], level_start_index, level_index),
        )
        topk_spatial_index = topk_index - start_index
        x = topk_spatial_index % feat_width
        y = torch.div(topk_spatial_index, feat_width, rounding_mode="trunc")
        coordinates = torch.stack([x - 1.0, y - 1.0, x + 1.0, y + 1.0], -1)

        # get unique idx for queries in different images and levels
        image_idx = torch.arange(batch_size).repeat_interleave(num_topk, 0)
        image_idx = image_idx.to(level_idx.device)
        idxs = level_idx + level_start_index.shape[0] * image_idx

        # perform batched_nms
        indices = torchvision.ops.batched_nms(coordinates, topk_scores, idxs, iou_threshold)

        # stack valid index
        results_index = []
        if torchvision._is_tracing():
            min_num = torch.tensor(num)
        else:
            min_num = num
        # get indices in each image
        for i in range(batch_size):
            topk_index_per_image = topk_index[indices[image_idx[indices] == i]]
            if torchvision._is_tracing():
                min_num = torch.min(topk_index_per_image.shape[0], min_num)
            else:
                min_num = min(topk_index_per_image.shape[0], min_num)
            results_index.append(topk_index_per_image)
        return torch.stack([index[:min_num] for index in results_index])


class TodTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        d_ffn=1024,
        dropout=0.1,
        n_heads=8,
        activation=nn.ReLU(inplace=True),
        n_levels=4,
        n_points=4,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # self attention
        self.self_attn = MultiScaleDeformableAttention(embed_dim, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # ffn
        self.linear1 = nn.Linear(embed_dim, d_ffn)
        self.activation = activation
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, embed_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, query):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(query))))
        query = query + self.dropout3(src2)
        query = self.norm2(query)
        return query

    def forward(
        self,
        query,
        query_pos,
        value,  # focus parameter
        reference_points,
        spatial_shapes,
        level_start_index,
        query_key_padding_mask=None,
    ):
        # self attention
        src2 = self.self_attn(
            query=self.with_pos_embed(query, query_pos),
            reference_points=reference_points,
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=query_key_padding_mask,
        )
        query = query + self.dropout1(src2)
        query = self.norm1(query)

        # ffn
        query = self.forward_ffn(query)

        return query


class TodTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.embed_dim = encoder_layer.embed_dim
        self.noise_layer = NoiseLayer(self.embed_dim)

        # learnt background embed for prediction
        # self.background_embedding = PositionEmbeddingLearned(500, num_pos_feats=self.embed_dim // 2)

        self.init_weights()

    def init_weights(self):
        # initialize encoder layers
        for layer in self.layers:
            if hasattr(layer, "init_weights"):
                layer.init_weights()

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, h - 0.5, h, dtype=torch.float32, device=device),
                torch.linspace(0.5, w - 0.5, w, dtype=torch.float32, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * h)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * w)
            ref = torch.stack((ref_x, ref_y), -1)  # [n, h*w, 2]
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)  # [n, s, 2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]  # [n, s, l, 2]
        return reference_points

    def forward(
        self,
        query,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        query_pos=None,
        query_key_padding_mask=None,
        # salience input
        focus_token_nums=None,
        foreground_inds=None,
        multi_level_masks=None,
    ):
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=query.device)
        b, n, s, p = reference_points.shape
        ori_reference_points = reference_points
        ori_pos = query_pos
        value = output = query

        for layer_id, layer in enumerate(self.layers):
            inds_for_query = foreground_inds[layer_id].unsqueeze(-1).expand(-1, -1, self.embed_dim)
            query = torch.gather(output, 1, inds_for_query)
            if layer_id == 0:
                query1 = query.detach()
                unit_noise = torch.randn_like(query1)
                noise_query, penalty, sigma = self.noise_layer(query1, unit_noise)

            query_pos = torch.gather(ori_pos, 1, inds_for_query)
            reference_points = torch.gather(
                ori_reference_points.view(b, n, -1), 1,
                foreground_inds[layer_id].unsqueeze(-1).repeat(1, 1, s * p)
            ).view(b, -1, s, p)
            query = layer(
                query,
                query_pos,
                value,
                reference_points,
                spatial_shapes,
                level_start_index,
                query_key_padding_mask,
            )
            with torch.no_grad():
                noise_query = layer(
                    noise_query,
                    query_pos.detach(),
                    value.detach(),
                    reference_points.detach(),
                    spatial_shapes.detach(),
                    level_start_index.detach(),
                    query_key_padding_mask.detach(),
                )
                # noise_query = layer(
                #     noise_query,
                #     query_pos,
                #     value,
                #     reference_points,
                #     spatial_shapes,
                #     level_start_index,
                #     query_key_padding_mask,
                # )

            outputs = []
            for i in range(foreground_inds[layer_id].shape[0]):
                foreground_inds_no_pad = foreground_inds[layer_id][i][:focus_token_nums[i]]
                query_no_pad = query[i][:focus_token_nums[i]]
                outputs.append(
                    output[i].scatter(
                        0,
                        foreground_inds_no_pad.unsqueeze(-1).repeat(1, query.size(-1)),
                        query_no_pad,
                    )
                )

            output = torch.stack(outputs)

        # add learnt embedding for background
        # if multi_level_masks is not None:
        #     background_embedding = [
        #         self.background_embedding(mask).flatten(2).transpose(1, 2) for mask in multi_level_masks
        #     ]
        #     background_embedding = torch.cat(background_embedding, dim=1)
        #     background_embedding.scatter_(1, inds_for_query, 0)
        #     background_embedding *= (~query_key_padding_mask).unsqueeze(-1)
        #     output = output + background_embedding

        return output, query, noise_query, penalty, sigma


class TodTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        d_ffn=1024,
        n_heads=8,
        dropout=0.1,
        activation=nn.ReLU(inplace=True),
        n_levels=4,
        n_points=4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = n_heads
        # cross attention
        self.cross_attn = MultiScaleDeformableAttention(embed_dim, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # self attention
        self.self_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        # ffn
        self.linear1 = nn.Linear(embed_dim, d_ffn)
        self.activation = activation
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, embed_dim)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.init_weights()

    def init_weights(self):
        # initialize self_attention
        nn.init.xavier_uniform_(self.self_attn.in_proj_weight)
        nn.init.xavier_uniform_(self.self_attn.out_proj.weight)
        # initialize Linear layer
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        query,
        query_pos,
        reference_points,
        value,
        spatial_shapes,
        level_start_index,
        self_attn_mask=None,
        key_padding_mask=None,
    ):
        # self attention
        query_with_pos = key_with_pos = self.with_pos_embed(query, query_pos)
        query2 = self.self_attn(
            query=query_with_pos,
            key=key_with_pos,
            value=query,
            attn_mask=self_attn_mask,
        )[0]
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        # cross attention
        query2 = self.cross_attn(
            query=self.with_pos_embed(query, query_pos),
            reference_points=reference_points,
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask,
        )
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        # ffn
        query = self.forward_ffn(query)

        return query


class TodTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, num_classes):
        super().__init__()
        # parameters
        self.embed_dim = decoder_layer.embed_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # decoder layers and embedding
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.ref_point_head = MLP(2 * self.embed_dim, self.embed_dim, self.embed_dim, 2)

        # iterative bounding box refinement
        self.class_head = nn.ModuleList([nn.Linear(self.embed_dim, num_classes) for _ in range(num_layers)])
        self.bbox_head = nn.ModuleList([MLP(self.embed_dim, self.embed_dim, 4, 3) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(self.embed_dim)

        self.init_weights()

    def init_weights(self):
        # initialize decoder layers
        for layer in self.layers:
            if hasattr(layer, "init_weights"):
                layer.init_weights()
        # initialize decoder classification layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for class_head in self.class_head:
            nn.init.constant_(class_head.bias, bias_value)
        # initiailize decoder regression layers
        for bbox_head in self.bbox_head:
            nn.init.constant_(bbox_head.layers[-1].weight, 0.0)
            nn.init.constant_(bbox_head.layers[-1].bias, 0.0)

    def forward(
        self,
        query,
        reference_points,
        value,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        key_padding_mask=None,
        attn_mask=None,
    ):
        outputs_classes = []
        outputs_coords = []
        valid_ratio_scale = torch.cat([valid_ratios, valid_ratios], -1)[:, None]

        for layer_idx, layer in enumerate(self.layers):
            reference_points_input = reference_points.detach()[:, :, None] * valid_ratio_scale
            query_sine_embed = get_sine_pos_embed(reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            # relation embedding
            query = layer(
                query=query,
                query_pos=query_pos,
                reference_points=reference_points_input,
                value=value,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask,
                self_attn_mask=attn_mask,
            )

            # get output, reference_points are not detached for look_forward_twice
            output_class = self.class_head[layer_idx](self.norm(query))
            output_coord = self.bbox_head[layer_idx](self.norm(query)) + inverse_sigmoid(reference_points)
            output_coord = output_coord.sigmoid()
            outputs_classes.append(output_class)
            outputs_coords.append(output_coord)

            if layer_idx == self.num_layers - 1:
                break

            # iterative bounding box refinement
            reference_points = self.bbox_head[layer_idx](query) + inverse_sigmoid(reference_points.detach())
            reference_points = reference_points.sigmoid()

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        return outputs_classes, outputs_coords
