import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def project_queries_to_views(query_xyz, extrinsics, intrinsics,
                             coordinate_scale, image_shape):
    rotation = extrinsics[..., :3, :3]
    translation = extrinsics[..., :3, 3]
    query_xyz = query_xyz / coordinate_scale.to(query_xyz).view(-1, 1, 1)
    camera_points = torch.einsum(
        'bvij,bqj->bvqi', rotation, query_xyz) + translation[:, :, None]
    depth = camera_points[..., 2]
    pixels = torch.einsum('bvij,bvqj->bvqi', intrinsics, camera_points)
    pixels = pixels[..., :2] / depth[..., None].clamp_min(1e-5)
    height, width = image_shape
    reference_points = pixels / pixels.new_tensor([width, height])
    visible = depth > 1e-5
    visible &= torch.isfinite(reference_points).all(dim=-1)
    visible &= (reference_points[..., 0] >= 0) & (reference_points[..., 0] <= 1)
    visible &= (reference_points[..., 1] >= 0) & (reference_points[..., 1] <= 1)
    return reference_points.permute(0, 2, 1, 3), visible.permute(0, 2, 1)


class ProjectedFullCrossAttention(nn.Module):
    """Select views, then map each feature level to one attention head."""

    def __init__(self, embed_dims, num_heads, num_feature_levels,
                 view_topk=8, dropout=0.0):
        super().__init__()
        if num_heads != num_feature_levels:
            raise ValueError(
                'Level-mapped attention requires num_heads == num_feature_levels')
        if embed_dims % num_heads != 0:
            raise ValueError('embed_dims must be divisible by num_heads')
        self.num_feature_levels = num_feature_levels
        self.view_topk = view_topk
        head_dim = embed_dims // num_heads
        self.query_projs = nn.ModuleList(
            [nn.Linear(embed_dims, head_dim) for _ in range(num_heads)])
        self.key_projs = nn.ModuleList(
            [nn.Linear(embed_dims, head_dim) for _ in range(num_heads)])
        self.value_projs = nn.ModuleList(
            [nn.Linear(embed_dims, head_dim) for _ in range(num_heads)])
        self.level_attn = nn.ModuleList([
            nn.MultiheadAttention(
                head_dim, 1, dropout=dropout, batch_first=True)
            for _ in range(num_heads)
        ])
        self.out_proj = nn.Linear(embed_dims, embed_dims)

    def forward(self, query, query_pos, feature_maps, reference_points,
                view_mask):
        batch_size, num_queries, channels = query.shape
        num_views = reference_points.shape[2]
        if len(feature_maps) != self.num_feature_levels:
            raise ValueError('Unexpected number of feature levels')

        value_levels = []
        for feature_map in feature_maps:
            _, views, feature_channels, height, width = feature_map.shape
            if views != num_views or feature_channels != channels:
                raise ValueError('Feature-map views or channels do not match queries')
            value_levels.append(feature_map.permute(0, 1, 3, 4, 2).reshape(
                batch_size, num_views, height * width, channels))

        # View selection uses only projection geometry, before image attention.
        refs = reference_points.nan_to_num(0.5).clamp(0., 1.)
        center_distance = torch.linalg.vector_norm(refs - 0.5, dim=-1)
        geometry_scores = (1.0 - center_distance / math.sqrt(0.5)).clamp_min(0.)
        geometry_scores = geometry_scores * view_mask.float()
        view_quality = geometry_scores.mean(dim=1)
        topk = min(self.view_topk, num_views) if self.view_topk > 0 else num_views
        _, selected_views = view_quality.topk(topk, dim=1)
        selected_valid = view_mask.gather(2, selected_views[:, None].expand(
            -1, num_queries, -1))
        selected_scores = geometry_scores.gather(
            2, selected_views[:, None].expand(-1, num_queries, -1))
        selected_scores = selected_scores.masked_fill(~selected_valid, -torch.inf)

        query_per_view = query[:, None].expand(
            -1, topk, -1, -1).reshape(batch_size * topk, num_queries, channels)
        query_pos_per_view = query_pos[:, None].expand(
            -1, topk, -1, -1).reshape_as(query_per_view)
        query_per_view = query_per_view + query_pos_per_view

        # Head i attends only to feature level i, while covering that level's
        # complete spatial token sequence.
        head_outputs = []
        for level_id in range(self.num_feature_levels):
            selected_tokens = value_levels[level_id].gather(
                1, selected_views[:, :, None, None].expand(
                    -1, -1, value_levels[level_id].shape[2], channels))
            selected_tokens = selected_tokens.reshape(
                batch_size * topk, value_levels[level_id].shape[2], channels)
            with torch.autocast(device_type=query.device.type, enabled=False):
                head_query = self.query_projs[level_id](query_per_view.float())
                head_key = self.key_projs[level_id](selected_tokens.float())
                head_value = self.value_projs[level_id](selected_tokens.float())
                head_output = self.level_attn[level_id](
                    head_query, head_key, head_value, need_weights=False)[0]
            head_outputs.append(head_output)

        attended = self.out_proj(torch.cat(head_outputs, dim=-1))
        attended = attended.to(query.dtype).reshape(
            batch_size, topk, num_queries, channels).permute(0, 2, 1, 3)
        weights = torch.softmax(selected_scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.)
        return (weights[..., None] * attended).sum(dim=2)


class GeometryAwareDecoderLayer(nn.Module):
    def __init__(self, embed_dims, num_heads, feedforward_channels,
                 num_feature_levels, view_topk, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = ProjectedFullCrossAttention(
            embed_dims, num_heads, num_feature_levels, view_topk, dropout)
        self.linear1 = nn.Linear(embed_dims, feedforward_channels)
        self.linear2 = nn.Linear(feedforward_channels, embed_dims)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, query_pos, feature_maps, reference_points,
                view_mask):
        query_norm = self.norm1(query)
        self_attended = self.self_attn(
            query_norm + query_pos, query_norm + query_pos, query_norm,
            need_weights=False)[0]
        query = query + self.dropout(self_attended)
        cross_attended = self.cross_attn(
            self.norm2(query), query_pos, feature_maps, reference_points,
            view_mask)
        query = query + self.dropout(cross_attended)
        ffn = self.linear2(self.dropout(F.gelu(self.linear1(self.norm3(query)))))
        return query + self.dropout(ffn)


class GeometryAwareDeformableDecoder(nn.Module):
    def __init__(self, embed_dims, num_layers, num_heads, feedforward_channels,
                 num_feature_levels, num_points=4, view_topk=8, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            GeometryAwareDecoderLayer(
                embed_dims, num_heads, feedforward_channels,
                num_feature_levels, view_topk, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dims)

    def forward(self, query, feature_maps, reference_points, extrinsics,
                intrinsics, coordinate_scale, image_shape,
                position_embedding, query_projection, reg_branches):
        if len(reg_branches) != len(self.layers):
            raise ValueError('Each decoder layer requires one regression branch')

        intermediate = []
        intermediate_references = []
        for layer_id, layer in enumerate(self.layers):
            query_xyz = reference_points
            query_pos = query_projection(
                position_embedding(query_xyz, input_range=None)).transpose(1, 2)
            image_references, view_mask = project_queries_to_views(
                query_xyz, extrinsics, intrinsics, coordinate_scale, image_shape)
            query = layer(
                query, query_pos, feature_maps, image_references, view_mask)
            output = self.norm(query)
            delta_xyz = reg_branches[layer_id](output.transpose(1, 2)).transpose(1, 2)
            reference_points = reference_points + delta_xyz
            intermediate.append(output.transpose(1, 2))
            intermediate_references.append(reference_points)
            reference_points = reference_points.detach()
        return intermediate, intermediate_references
