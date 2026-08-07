# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
from mmcv.transforms import BaseTransform, Compose
from PIL import Image

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.points import get_points_type


def _sample_multiview_ids(num_views, n_images, loading, sample_freq):
    if loading == 'random':
        return np.random.choice(
            num_views, n_images, replace=n_images > num_views).tolist()
    if loading == 'gap':
        return np.linspace(0, num_views - 1, n_images, dtype=np.int64).tolist()

    ids = np.arange(0, n_images * sample_freq, sample_freq)
    if len(ids) == 0 or ids[-1] >= num_views:
        raise ValueError('Requested views exceed the available image sequence')
    return ids.tolist()


def _load_depth(depth_path, image_shape):
    if depth_path.endswith('.npy'):
        return np.load(depth_path)
    depth = np.asarray(Image.open(depth_path), dtype=np.float32) / 1000
    return mmcv.imresize(depth, (image_shape[1], image_shape[0]))


def read_pose_matrix(file_path):
    try:
        with open(file_path, 'r') as file:
            matrix = [list(map(float, line.strip().split())) for line in file]
        pose_matrix = np.array(matrix)
        if pose_matrix.shape != (4, 4):
            raise ValueError('The input file does not contain a valid 4x4 pose matrix.')
        return pose_matrix
    except Exception as error:
        print(f'Error reading pose matrix: {error}')
        return None


@TRANSFORMS.register_module()
class ProjectPCtoFirstFrameAndNorm(BaseTransform):
    def __init__(self, coord_type):
        self.coord_type = coord_type

    def transform(self, results: dict) -> dict:
        pose_file_path = results['img_path'][0][:-3] + 'txt'
        pose_matrix = read_pose_matrix(pose_file_path)
        extrinsic_matrix = np.linalg.inv(pose_matrix)
        points = results['points']
        points_pos = points[:, :3]
        points_pos_homo = np.hstack(
            (points_pos, np.ones((points_pos.shape[0], 1))))
        points_in_first_axis = np.dot(
            extrinsic_matrix, points_pos_homo.T).T[:, :3]
        avg_distance = np.mean(np.linalg.norm(points_in_first_axis, axis=1))
        pc_cam_with_rgb = np.hstack((points_in_first_axis, points[:, 3:]))

        points_class = get_points_type(self.coord_type)
        results['points'] = points_class(
            pc_cam_with_rgb,
            points_dim=points.points_dim,
            attribute_dims=points.attribute_dims)
        results['pose_matrix'] = pose_matrix
        results['avg_distance'] = avg_distance
        return results


@TRANSFORMS.register_module()
class ProjectPCtoFirstFrameAndNormArkit(BaseTransform):
    def __init__(self, coord_type):
        self.coord_type = coord_type

    def transform(self, results: dict) -> dict:
        pose_file_path = results['img_path'][0].replace('_color.png', '_pose.npy')
        pose_matrix = np.load(pose_file_path)
        extrinsic_matrix = np.linalg.inv(pose_matrix)
        points = results['points']
        points_pos = points[:, :3]
        points_pos_homo = np.hstack(
            (points_pos, np.ones((points_pos.shape[0], 1))))
        points_in_first_axis = np.dot(
            extrinsic_matrix, points_pos_homo.T).T[:, :3]
        avg_distance = np.mean(np.linalg.norm(points_in_first_axis, axis=1))
        pc_cam_with_rgb = np.hstack((points_in_first_axis, points[:, 3:]))

        points_class = get_points_type(self.coord_type)
        results['points'] = points_class(
            pc_cam_with_rgb,
            points_dim=points.points_dim,
            attribute_dims=points.attribute_dims)
        results['pose_matrix'] = pose_matrix
        results['avg_distance'] = avg_distance
        return results


@TRANSFORMS.register_module()
class NormBoxes(BaseTransform):
    def transform(self, results: dict) -> dict:
        gt_bboxes_3d = results['gt_bboxes_3d']
        norm_scale = results['avg_distance']
        gt_bboxes_concatenated = np.concatenate([
            gt_bboxes_3d.gravity_center / norm_scale,
            gt_bboxes_3d.tensor[:, 3:6] / norm_scale,
        ], axis=1)
        results['gt_bboxes_3d'] = results['box_type_3d'](
            gt_bboxes_concatenated,
            box_dim=6,
            with_yaw=False,
            origin=(.5, .5, .5))
        return results


class _BaseMultiViewPipeline(BaseTransform):
    def __init__(self,
                 transforms: dict,
                 n_images: int,
                 mean: tuple = (123.675, 116.28, 103.53),
                 std: tuple = (58.395, 57.12, 57.375),
                 loading: str = 'random',
                 sample_freq: int = 3,
                 normalize: bool = True):
        self.transforms = Compose(transforms)
        self.n_images = n_images
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.loading = loading
        self.sample_freq = sample_freq
        self.normalize = normalize

    def _load_source_views(self, results, collect_intrinsics=False):
        ids = _sample_multiview_ids(
            len(results['img_info']), self.n_images, self.loading,
            self.sample_freq)
        imgs, depths, extrinsics, intrinsics, image_paths = [], [], [], [], []
        for index in ids:
            image_path = results['img_info'][index]['filename']
            view_results = self.transforms(dict(img_path=image_path))
            if self.normalize:
                for key in view_results.get('img_fields', ['img']):
                    view_results[key] = mmcv.imnormalize(
                        view_results[key], self.mean, self.std, True)
                view_results['img_norm_cfg'] = dict(
                    mean=self.mean, std=self.std, to_rgb=True)
            imgs.append(view_results['img'])
            image_paths.append(image_path)
            extrinsics.append(results['lidar2img']['extrinsic'][index])
            if collect_intrinsics:
                intrinsics.append(results['lidar2img']['intrinsic'][index])
            if 'depth_info' in results:
                depths.append(_load_depth(
                    results['depth_info'][index]['filename'],
                    view_results['img_shape']))
        return imgs, depths, extrinsics, intrinsics, image_paths


@TRANSFORMS.register_module()
class MultiViewPipeline_Tgt(_BaseMultiViewPipeline):
    """Load ScanNet source views for multi-view detection."""

    def transform(self, results: dict) -> dict:
        imgs, depths, extrinsics, _, image_paths = self._load_source_views(results)
        results['img'] = imgs
        results['img_path'] = image_paths
        results['lidar2img']['extrinsic'] = extrinsics
        if depths:
            results['depth'] = depths
        return results


@TRANSFORMS.register_module()
class MultiViewPipeline_ARKit(_BaseMultiViewPipeline):
    """Load ARKit source views for multi-view detection."""

    def transform(self, results: dict) -> dict:
        imgs, depths, extrinsics, intrinsics, image_paths = self._load_source_views(
            results, collect_intrinsics=True)
        results['img'] = imgs
        results['img_path'] = image_paths
        results['lidar2img']['extrinsic'] = extrinsics
        results['lidar2img']['intrinsic'] = intrinsics
        if depths:
            results['depth'] = depths
        return results


@TRANSFORMS.register_module()
class RandomShiftOrigin(BaseTransform):
    def __init__(self, std):
        self.std = std

    def transform(self, results):
        results['lidar2img']['origin'] += np.random.normal(.0, self.std, 3)
        return results
