import torch
import torch.nn as nn
from mmcv.ops import MultiScaleDeformableAttention


class SingleScaleSemanticDeformableAttention(nn.Module):
    """Use one VGGT feature level to sample one same-view SAM3 feature map."""

    def __init__(self, embed_dims, num_heads=8, num_points=4, dropout=0.1,
                 layer_scale_init=0.0):
        super().__init__()
        self.query_norm = nn.LayerNorm(embed_dims)
        self.value_norm = nn.LayerNorm(embed_dims)
        self.attention = MultiScaleDeformableAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=1,
            num_points=num_points,
            dropout=dropout,
            batch_first=True)
        self.layer_scale = nn.Parameter(
            torch.full((embed_dims,), layer_scale_init))

    @staticmethod
    def _reference_points(height, width, device, dtype):
        y, x = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype) + 0.5,
            torch.arange(width, device=device, dtype=dtype) + 0.5,
            indexing='ij')
        reference_points = torch.stack((x / width, y / height), dim=-1)
        return reference_points.reshape(1, height * width, 1, 2)

    def forward(self, query_map, value_map):
        batch_size, num_views, channels, query_height, query_width = (
            query_map.shape)
        value_batch, value_views, value_channels, value_height, value_width = (
            value_map.shape)
        if (value_batch, value_views, value_channels) != (
                batch_size, num_views, channels):
            raise ValueError(
                'SAM3 and VGGT feature batch, view, or channel dimensions '
                'differ')

        query = query_map.permute(0, 1, 3, 4, 2).reshape(
            batch_size * num_views, query_height * query_width, channels)
        value = value_map.permute(0, 1, 3, 4, 2).reshape(
            batch_size * num_views, value_height * value_width, channels)
        query = self.query_norm(query)
        value = self.value_norm(value)

        reference_points = self._reference_points(
            query_height, query_width, query.device, torch.float32)
        reference_points = reference_points.expand(
            batch_size * num_views, -1, -1, -1)
        spatial_shapes = torch.tensor(
            [[value_height, value_width]], dtype=torch.long,
            device=query.device)
        level_start_index = spatial_shapes.new_zeros(1)

        output_dtype = query.dtype
        with torch.autocast(device_type=query.device.type, enabled=False):
            semantic = self.attention(
                query=query.float(),
                value=value.float(),
                identity=torch.zeros_like(query, dtype=torch.float32),
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index)
        semantic = semantic.to(output_dtype)
        semantic = semantic.reshape(
            batch_size, num_views, query_height, query_width,
            channels).permute(0, 1, 4, 2, 3)
        scale = self.layer_scale.to(semantic).view(1, 1, channels, 1, 1)
        return query_map + scale * semantic


class SemanticDeformableFusion(nn.Module):
    """Fuse four VGGT levels with SAM3 features at 1/7 and 1/14 scales."""

    def __init__(self, embed_dims=512, sam_channels=256, num_heads=8,
                 num_points=4, dropout=0.1, layer_scale_init=0.0):
        super().__init__()
        self.sam_adapters = nn.ModuleList([
            nn.Conv2d(sam_channels, embed_dims, kernel_size=1),
            nn.Conv2d(sam_channels, embed_dims, kernel_size=1),
        ])
        self.attentions = nn.ModuleList([
            SingleScaleSemanticDeformableAttention(
                embed_dims, num_heads, num_points, dropout, layer_scale_init)
            for _ in range(4)
        ])

    @staticmethod
    def _project_views(feature_map, adapter):
        batch_size, num_views, channels, height, width = feature_map.shape
        projected = adapter(feature_map.reshape(
            batch_size * num_views, channels, height, width))
        return projected.reshape(
            batch_size, num_views, projected.shape[1], height, width)

    def forward(self, vggt_feature_maps, sam_feature_maps):
        if len(vggt_feature_maps) != 4:
            raise ValueError(
                'Semantic fusion requires four VGGT feature levels')
        if len(sam_feature_maps) != 2:
            raise ValueError(
                'Semantic fusion requires SAM3 1/7 and 1/14 features')

        sam_fine = self._project_views(
            sam_feature_maps[0], self.sam_adapters[0])
        sam_coarse = self._project_views(
            sam_feature_maps[1], self.sam_adapters[1])
        sam_sources = (sam_fine, sam_fine, sam_coarse, sam_coarse)
        return [
            attention(vggt_feature, sam_feature)
            for attention, vggt_feature, sam_feature in zip(
                self.attentions, vggt_feature_maps, sam_sources)
        ]
