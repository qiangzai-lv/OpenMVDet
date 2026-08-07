# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence, Union

import mmengine
import numpy as np
import torch
from mmcv import BaseTransform
from mmengine.structures import InstanceData
from numpy import dtype

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import (
    BaseInstance3DBoxes,
    Det3DDataSample,
    PointData,
)
from mmdet3d.structures.points import BasePoints


def to_tensor(
    data: Union[torch.Tensor, np.ndarray, Sequence, int, float]
) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, np.ndarray):
        if data.dtype is dtype('float64'):
            data = data.astype(np.float32)
        return torch.from_numpy(data)
    if isinstance(data, Sequence) and not mmengine.is_str(data):
        return torch.tensor(data)
    if isinstance(data, int):
        return torch.LongTensor([data])
    if isinstance(data, float):
        return torch.FloatTensor([data])
    raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@TRANSFORMS.register_module()
class PackMultiViewDetInputs(BaseTransform):
    INPUT_KEYS = {
        'img', 'depth', 'points', 'pose_matrix', 'axis_align_matrix',
        'avg_distance'
    }
    INSTANCEDATA_3D_KEYS = {
        'gt_bboxes_3d', 'gt_labels_3d', 'attr_labels', 'depths', 'centers_2d'
    }
    INSTANCEDATA_2D_KEYS = {'gt_bboxes', 'gt_bboxes_labels'}
    SEG_KEYS = {
        'gt_seg_map', 'pts_instance_mask', 'pts_semantic_mask',
        'gt_semantic_seg'
    }

    def __init__(
        self,
        keys: tuple,
        meta_keys: tuple = (
            'img_path', 'ori_shape', 'img_shape', 'lidar2img', 'depth2img',
            'cam2img', 'pad_shape', 'scale_factor', 'flip',
            'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
            'box_type_3d', 'img_norm_cfg', 'num_pts_feats', 'pcd_trans',
            'sample_idx', 'pcd_scale_factor', 'pcd_rotation',
            'pcd_rotation_angle', 'lidar_path', 'transformation_3d_flow',
            'trans_mat', 'affine_aug', 'sweep_img_metas', 'ori_cam2img',
            'cam2global', 'crop_offset', 'img_crop_offset',
            'resize_img_shape', 'lidar2cam', 'ori_lidar2img',
            'num_ref_frames', 'num_views', 'ego2global', 'axis_align_matrix'
        )
    ) -> None:
        self.keys = keys
        self.meta_keys = meta_keys

    @staticmethod
    def _remove_prefix(key: str) -> str:
        return key[3:] if key.startswith('gt_') else key

    def transform(self, results: Union[dict, List[dict]]) -> Union[dict, List[dict]]:
        if isinstance(results, list):
            if len(results) == 1:
                return self.pack_single_results(results[0])
            return [self.pack_single_results(single_result) for single_result in results]
        if isinstance(results, dict):
            return self.pack_single_results(results)
        raise NotImplementedError

    def _format_images(self, results: dict) -> None:
        if 'img' not in results:
            return
        if isinstance(results['img'], list):
            imgs = np.stack(results['img'], axis=0)
            if imgs.flags.c_contiguous:
                results['img'] = to_tensor(imgs).permute(0, 3, 1, 2).contiguous()
            else:
                results['img'] = to_tensor(
                    np.ascontiguousarray(imgs.transpose(0, 3, 1, 2)))
            return

        image = results['img']
        if len(image.shape) < 3:
            image = np.expand_dims(image, -1)
        if image.flags.c_contiguous:
            results['img'] = to_tensor(image).permute(2, 0, 1).contiguous()
        else:
            results['img'] = to_tensor(
                np.ascontiguousarray(image.transpose(2, 0, 1)))

    def _format_depth(self, results: dict) -> None:
        if 'depth' not in results:
            return
        if isinstance(results['depth'], list):
            results['depth'] = to_tensor(np.ascontiguousarray(
                np.stack(results['depth'], axis=0)))
            return
        depth = results['depth']
        if len(depth.shape) < 3:
            depth = np.expand_dims(depth, -1)
        results['depth'] = to_tensor(np.ascontiguousarray(depth))

    def _collect_metainfo(self, results: dict) -> dict:
        data_metas = {}
        for key in self.meta_keys:
            if key in results:
                data_metas[key] = results[key]
            elif 'images' in results:
                image_metas = [
                    image[key] for image in results['images'].values()
                    if key in image
                ]
                if image_metas:
                    data_metas[key] = image_metas[0] if len(image_metas) == 1 else image_metas
            elif 'lidar_points' in results and key in results['lidar_points']:
                data_metas[key] = results['lidar_points'][key]
        return data_metas

    def pack_single_results(self, results: dict) -> dict:
        if isinstance(results.get('points'), BasePoints):
            results['points'] = results['points'].tensor
        self._format_images(results)
        self._format_depth(results)

        tensor_keys = {
            'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
            'gt_bboxes_labels', 'attr_labels', 'pts_instance_mask',
            'pts_semantic_mask', 'centers_2d', 'depths', 'gt_labels_3d',
            'pose_matrix', 'axis_align_matrix', 'avg_distance'
        }
        for key in tensor_keys:
            if key in results:
                results[key] = [to_tensor(value) for value in results[key]] \
                    if isinstance(results[key], list) else to_tensor(results[key])
        if ('gt_bboxes_3d' in results and
                not isinstance(results['gt_bboxes_3d'], BaseInstance3DBoxes)):
            results['gt_bboxes_3d'] = to_tensor(results['gt_bboxes_3d'])
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = to_tensor(results['gt_semantic_seg'][None])
        if 'gt_seg_map' in results:
            results['gt_seg_map'] = results['gt_seg_map'][None, ...]

        data_sample = Det3DDataSample()
        data_sample.set_metainfo(self._collect_metainfo(results))
        gt_instances_3d = InstanceData()
        gt_instances = InstanceData()
        gt_pts_seg = PointData()
        inputs = {}

        for key in self.keys:
            if key not in results:
                continue
            if key in self.INPUT_KEYS:
                inputs[key] = results[key]
            elif key in self.INSTANCEDATA_3D_KEYS:
                gt_instances_3d[self._remove_prefix(key)] = results[key]
            elif key in self.INSTANCEDATA_2D_KEYS:
                target_key = 'labels' if key == 'gt_bboxes_labels' else self._remove_prefix(key)
                gt_instances[target_key] = results[key]
            elif key in self.SEG_KEYS:
                gt_pts_seg[self._remove_prefix(key)] = results[key]
            else:
                raise NotImplementedError(
                    f'Add {key} to the appropriate PackMultiViewDetInputs field.')

        data_sample.gt_instances_3d = gt_instances_3d
        data_sample.gt_instances = gt_instances
        data_sample.gt_pts_seg = gt_pts_seg
        data_sample.eval_ann_info = results.get('eval_ann_info')
        return {'data_samples': data_sample, 'inputs': inputs}

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(keys={self.keys})(meta_keys={self.meta_keys})'
