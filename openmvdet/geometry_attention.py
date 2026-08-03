import torch
import torch.nn as nn
import torch.nn.functional as F


def project_queries_to_views(query_xyz, extrinsics, intrinsics,
                             coordinate_scale, image_shape):
    rotation = extrinsics[..., :3, :3]
    translation = extrinsics[..., :3, 3]
    query_xyz = query_xyz / coordinate_scale.to(query_xyz).view(-1, 1, 1)
    camera_points = torch.einsum(
        "bvij,bqj->bvqi", rotation, query_xyz) + translation[:, :, None]
    depth = camera_points[..., 2]
    pixels = torch.einsum("bvij,bvqj->bvqi", intrinsics, camera_points)
    pixels = pixels[..., :2] / depth[..., None].clamp_min(1e-5)
    height, width = image_shape
    normalized_pixels = pixels / pixels.new_tensor([width, height])
    visible = depth > 1e-5
    visible &= torch.isfinite(normalized_pixels).all(dim=-1)
    visible &= (normalized_pixels[..., 0] >= 0) & (normalized_pixels[..., 0] <= 1)
    visible &= (normalized_pixels[..., 1] >= 0) & (normalized_pixels[..., 1] <= 1)
    return visible.permute(0, 2, 1)


class ProjectedViewCrossAttention(nn.Module):
    """Attend to all patch tokens from views where each query is visible."""

    def __init__(self, embed_dims, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True)

    def forward(self, query, query_pos, feature_maps, view_mask):
        batch_size, num_queries, channels = query.shape
        num_views = view_mask.shape[-1]
        value_levels = []
        for feature_map in feature_maps:
            _, views, feature_channels, height, width = feature_map.shape
            if views != num_views or feature_channels != channels:
                raise ValueError("Feature-map views or channels do not match queries")
            value_levels.append(feature_map.permute(0, 1, 3, 4, 2).reshape(
                batch_size, num_views, height * width, channels))

        value = torch.cat(value_levels, dim=2).reshape(batch_size, -1, channels)
        tokens_per_view = value.shape[1] // num_views
        fallback = value.new_zeros(batch_size, 1, channels)
        value = torch.cat([value, fallback], dim=1)

        visible_tokens = view_mask.repeat_interleave(tokens_per_view, dim=-1)
        attention_mask = ~visible_tokens
        attention_mask = torch.cat([
            attention_mask,
            torch.zeros(batch_size, num_queries, 1, dtype=torch.bool,
                        device=query.device)
        ], dim=-1)
        attention_mask = attention_mask.repeat_interleave(self.num_heads, dim=0)

        return self.attention(
            query=query + query_pos,
            key=value,
            value=value,
            attn_mask=attention_mask,
            need_weights=False)[0]


class GeometryAwareDecoderLayer(nn.Module):
    def __init__(self, embed_dims, num_heads, feedforward_channels,
                 dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = ProjectedViewCrossAttention(
            embed_dims, num_heads, dropout)
        self.linear1 = nn.Linear(embed_dims, feedforward_channels)
        self.linear2 = nn.Linear(feedforward_channels, embed_dims)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, query_pos, feature_maps, view_mask):
        query_norm = self.norm1(query)
        self_attended = self.self_attn(
            query_norm + query_pos, query_norm + query_pos, query_norm,
            need_weights=False)[0]
        query = query + self.dropout(self_attended)
        cross_attended = self.cross_attn(
            self.norm2(query), query_pos, feature_maps, view_mask)
        query = query + self.dropout(cross_attended)
        ffn = self.linear2(self.dropout(F.gelu(self.linear1(self.norm3(query)))))
        return query + self.dropout(ffn)


class GeometryAwareDecoder(nn.Module):
    def __init__(self, embed_dims, num_layers, num_heads, feedforward_channels,
                 dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            GeometryAwareDecoderLayer(
                embed_dims, num_heads, feedforward_channels, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dims)

    def forward(self, query, feature_maps, reference_points, extrinsics,
                intrinsics, coordinate_scale, image_shape,
                position_embedding, query_projection, reg_branches):
        if len(reg_branches) != len(self.layers):
            raise ValueError("Each decoder layer requires one regression branch")

        intermediate = []
        intermediate_references = []
        for layer_id, layer in enumerate(self.layers):
            query_xyz = reference_points
            query_pos = query_projection(
                position_embedding(query_xyz, input_range=None)).transpose(1, 2)
            view_mask = project_queries_to_views(
                query_xyz, extrinsics, intrinsics, coordinate_scale, image_shape)
            query = layer(query, query_pos, feature_maps, view_mask)
            output = self.norm(query)
            delta_xyz = reg_branches[layer_id](output.transpose(1, 2)).transpose(1, 2)
            reference_points = reference_points + delta_xyz
            intermediate.append(output.transpose(1, 2))
            intermediate_references.append(reference_points)
            reference_points = reference_points.detach()
        return intermediate, intermediate_references
