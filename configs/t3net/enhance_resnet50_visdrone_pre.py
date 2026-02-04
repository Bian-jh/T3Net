from torch import nn
# from torchvision.ops import FrozenBatchNorm2d

from models.backbones.resnet import ResNetBackbone
from models.bricks.misc import FrozenBatchNorm2d
from models.bricks.position_encoding import PositionEmbeddingSine
from models.bricks.post_process import PostProcess
from models.bricks.tod_transformer_enhance import (
    TodTransformer,
    TodTransformerDecoder,
    TodTransformerDecoderLayer,
    TodTransformerEncoder,
    TodTransformerEncoderLayer,
)
from models.bricks.set_criterion import HybridSetCriterion
from models.detectors.tod_detr_enhance import DensityCriterion, T3Net
from models.matcher.hungarian_matcher import HungarianMatcher
from models.necks.channel_mapper import ChannelMapper
from models.necks.enhancenet import EnhanceNetwork

# mostly changed parameters
embed_dim = 256
num_classes = 11
num_queries = 900
num_feature_levels = 5
transformer_enc_layers = 6
transformer_dec_layers = 6
num_heads = 8
dim_feedforward = 2048

# instantiate model components
position_embedding = PositionEmbeddingSine(embed_dim // 2, temperature=10000, normalize=True, offset=-0.5)

backbone = ResNetBackbone(
    "resnet50", norm_layer=FrozenBatchNorm2d, return_indices=(0, 1, 2, 3), freeze_indices=(0,)
)

neck = ChannelMapper(
    in_channels=backbone.num_channels,
    out_channels=embed_dim,
    num_outs=num_feature_levels,
)

transformer = TodTransformer(
    encoder=TodTransformerEncoder(
        encoder_layer=TodTransformerEncoderLayer(
            embed_dim=embed_dim,
            n_heads=num_heads,
            dropout=0.0,
            activation=nn.ReLU(inplace=True),
            n_levels=num_feature_levels,
            n_points=4,
            d_ffn=dim_feedforward,
        ),
        num_layers=transformer_enc_layers,
    ),
    neck=EnhanceNetwork(
        in_channels_list=neck.num_channels,
        out_channels_list=neck.num_channels,
        norm_layer=nn.BatchNorm2d,
        activation=nn.SiLU,
        groups=4,
    ),
    decoder=TodTransformerDecoder(
        decoder_layer=TodTransformerDecoderLayer(
            embed_dim=embed_dim,
            n_heads=num_heads,
            dropout=0.0,
            activation=nn.ReLU(inplace=True),
            n_levels=num_feature_levels,
            n_points=4,
            d_ffn=dim_feedforward,
        ),
        num_layers=transformer_dec_layers,
        num_classes=num_classes,
    ),
    num_classes=num_classes,
    num_feature_levels=num_feature_levels,
    two_stage_num_proposals=num_queries,
    level_filter_ratio=(0.5, 0.5, 0.8, 1.0, 1.0),
    layer_filter_ratio=(0.8, 0.8, 0.8, 0.8, 0.8, 0.8),
    use_adaptive_query=False,
    dynamic_query_list=(300, 500, 900),
)

matcher = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2)

weight_dict = {"loss_class": 1, "loss_bbox": 5, "loss_giou": 2}
weight_dict.update({"loss_class_dn": 1, "loss_bbox_dn": 5, "loss_giou_dn": 2})
weight_dict.update({
    k + f"_{i}": v
    for i in range(transformer_dec_layers - 1)
    for k, v in weight_dict.items()
})
weight_dict.update({"loss_class_enc": 1, "loss_bbox_enc": 5, "loss_giou_enc": 2})
weight_dict.update({"loss_spatial": 1, "loss_global": 1, "loss_local": 5, "loss_count": 0.01})

criterion = HybridSetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict, alpha=0.25, gamma=2.0)
foreground_criterion = DensityCriterion(alpha=0.25, gamma=2.0, beta=0.6)
postprocessor = PostProcess(select_box_nums_for_evaluation=300)

# combine above components to instantiate the model
model = T3Net(
    backbone=backbone,
    neck=neck,
    position_embedding=position_embedding,
    transformer=transformer,
    criterion=criterion,
    focus_criterion=foreground_criterion,
    postprocessor=postprocessor,
    num_classes=num_classes,
    num_queries=num_queries,
    aux_loss=True,
    min_size=800,
    max_size=1333,
)
