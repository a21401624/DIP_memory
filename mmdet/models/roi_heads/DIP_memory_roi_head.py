# Author: Hu Yuxuan
import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor, build_loss
from mmcv.runner import BaseModule

import sys
if sys.version_info >= (3, 7):
    from mmdet.utils.contextmanagers import completed


@HEADS.register_module()
class DIPMemoryRoIHead(BaseModule):
    """
        No mask head and shared head.
        Don't use seperate test_mixins.
    """
    def __init__(self,
                 baseline,
                 memory_size,
                 temperature,
                 gamma,
                 sim_thresh,
                 assign_thresh,
                 c,
                 k,
                 fusion_type='cat-conv',
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 work_dir=None,
                 comment=None):
        super(DIPMemoryRoIHead, self).__init__(init_cfg)
        self.baseline = baseline
        self.memory_size = memory_size
        self.temperature = temperature
        self.gamma = gamma
        self.sim_thresh = sim_thresh
        self.assign_thresh = assign_thresh
        self.k = k
        self.c = c
        assert fusion_type in ['cat-conv', 'add']
        self.fusion_type = fusion_type
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.work_dir = work_dir

        feature_channel = bbox_head.in_channels
        roi_feat_size = bbox_head.roi_feat_size

        if bbox_head is not None:
            self.init_bbox_head(bbox_roi_extractor, bbox_head)

        if self.fusion_type == 'cat-conv':
            self.f_conv = nn.Conv2d(2 * feature_channel, feature_channel, 1)
        if not self.baseline:
            # Memory
            key = torch.zeros(self.memory_size, 
                                        feature_channel * roi_feat_size * roi_feat_size)
            self.register_buffer('key', key)
            value = torch.zeros(self.memory_size, 
                                        feature_channel * roi_feat_size * roi_feat_size)
            self.register_buffer('value', value)
            self.register_buffer("memory_init", torch.tensor(False))

        self.init_assigner_sampler()
        self.identity1 = nn.Identity()
        self.identity2 = nn.Identity()
        self.f_t = nn.Identity()
        self.f_v = nn.Identity()
        self.f_a = nn.Identity()
        self.f_c1 = nn.Identity()
        self.f_c2 = nn.Identity()
        self.rois = nn.Identity()
        self.pos_locs = nn.Identity()

    @property
    def with_bbox(self):
        """bool: whether the RoI head contains a `bbox_head`"""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        f_t = self.bbox_roi_extractor(
                x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        f_a, _ = self._memory_read(f_t)
        f_c1 = self.f_conv(torch.cat((f_t, f_a), dim=1))
        if self.with_bbox:
            bbox_results = self._bbox_forward(f_c1)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        return outs

    def forward_train(self,
                      x1,
                      x2,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x1 (list[Tensor]): list of multi-level img features.
            x2 (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposal_list (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x1, x2, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            if not self.baseline:
                losses.update(bbox_results['loss_bbox1'])
                losses.update(bbox_results['loss_bbox2'])
                losses.update(bbox_results['loss_aux'])
            else:
                losses.update(bbox_results['loss_bbox'])

        return losses

    def _memory_init(self, f_t_flatten, f_v_flatten, pos_inds):
        """Init each memory slot by randomly select two roi feats and do random combination.
        """
        for i in range(self.memory_size):
            roi_feat_inds = torch.randint(0, f_t_flatten.shape[0], (2,))
            roi_feat_weights = torch.rand(2,)
            roi_feat_weights[1] = 1 - roi_feat_weights[0]
            self.key.data[i] = roi_feat_weights[0] * f_t_flatten[roi_feat_inds[0]] + \
                roi_feat_weights[1] * f_t_flatten[roi_feat_inds[1]]
            self.value.data[i] = roi_feat_weights[0] * f_v_flatten[roi_feat_inds[0]] + \
                roi_feat_weights[1] * f_v_flatten[roi_feat_inds[1]]
        self.memory_init = torch.tensor(True).to(self.key.device)
        if self.work_dir is not None:
            np.save(os.path.join(self.work_dir, 'init_key.npy'), self.key.cpu().numpy())
            np.save(os.path.join(self.work_dir, 'init_value.npy'), self.value.cpu().numpy())

    def _memory_read(self, f_t: torch.Tensor):
        """
            input: f_t (total_number_of_rois, feature_channel, roi_feat_size, roi_feat_size)
            output: f_a, has the same shape as f_t.
                    W_K, W_V has shape (total_number_of_rois, memory_size)
        """
        ori_shape = f_t.shape
        f_t = f_t.flatten(1)
        f_t_norm = F.normalize(f_t, dim=1)
        key_norm = F.normalize(self.key, dim=1)
        S_K = self.identity2(torch.matmul(f_t_norm, key_norm.T)) # (num_of_query, num_of_key)
        W_H = self.identity2(torch.softmax(S_K/self.temperature, dim=1))
        f_a = torch.matmul(W_H, self.value)
        f_a = self.f_a(torch.reshape(f_a, ori_shape))
        reliable = torch.max(S_K, dim=1)[0]
        return f_a, reliable

    def _memory_update(self,
                       f_t_flatten: torch.Tensor, 
                       f_v_flatten: torch.Tensor, 
                       loss_aux):
        """
            input: f_t_flatten, f_v (total_number_of_rois, feature_channel * roi_feat_size * roi_feat_size)
            output: f_c1, has the same shape as f_t.
        """
        f_t_norm = F.normalize(f_t_flatten, dim=1)
        norm_key = F.normalize(self.key, dim=1)
        S_K = self.identity1(torch.matmul(f_t_norm, norm_key.T))
        W_H = self.identity1(torch.softmax(S_K/self.temperature, dim=1))
        W_V = torch.softmax(S_K/self.temperature, dim=0)

        sim_key = torch.matmul(norm_key, norm_key.T)
        sim_key = torch.triu(sim_key, diagonal=1)
        y_idcs, x_idcs = torch.where(sim_key > self.sim_thresh)
        delete_index = []
        for i, j in zip(y_idcs, x_idcs):
            if i not in delete_index and j not in delete_index:
                delete_index.append(j)
        loss_aux['delete_memory_num'] = torch.tensor(len(delete_index)).float()

        no_assign_inds = torch.where(torch.max(S_K, dim=1)[0] <= self.assign_thresh)[0]
        no_assign_inds = no_assign_inds[torch.argsort(torch.mean(S_K[no_assign_inds], dim=1))] # ascending order
        loss_aux['no_assign_query_num'] = torch.tensor(no_assign_inds.shape[0]).float()
        # If len(delete_index)==0, no delete.
        # If no_assign_inds.shape[0]==0, no delete.
        # If no_assign_inds.shape[0] > len(delete_index), 
        #       select new key-vaule pairs from no_assign_inds.
        # If no_assign_inds.shape[0] == len(delete_index), use all key-vaule pairs.
        # If no_assign_inds.shape[0] < len(delete_index), partly delete memory.
        if len(delete_index) > 0 and no_assign_inds.shape[0] > 0:
            if no_assign_inds.shape[0] < len(delete_index):
                delete_index = random.sample(delete_index, no_assign_inds.shape[0])
            W_H[:, delete_index] = 0
            nearest_slot = torch.argmax(W_H, dim=1) #(num_of_rois,)
            # nearest_slot[no_assign_inds] = -1
            for idx, i in enumerate(delete_index):
                self.key.data[i] = f_t_flatten[no_assign_inds[idx]]
                self.value.data[i] = f_v_flatten[no_assign_inds[idx]]
        else:
            nearest_slot = torch.argmax(W_H, dim=1) #(num_of_rois,)

        key_update = torch.zeros((self.key).shape).to(self.key.device)
        value_update = torch.zeros((self.value).shape).to(self.value.device)
        gamma = torch.ones((self.key.shape[0], 1)).to(self.key.device)
        for i in range(self.memory_size):
            # key_name2 = str(i) + 'assign_num'
            # if i in delete_index:
            #     loss_aux[key_name2] = torch.tensor(-1).float()
            # else:
            #     loss_aux[key_name2] =  torch.sum(nearest_slot==i).float()
            if torch.sum(nearest_slot==i) >= 1 and i not in delete_index:
                assigned_roi_feats = torch.where(nearest_slot==i)[0]
                roi_feat_w = W_V[assigned_roi_feats, i]
                roi_feat_w = roi_feat_w / torch.sum(roi_feat_w)
                key_update[i] = torch.sum(torch.unsqueeze(roi_feat_w, dim=1) * \
                                    f_t_flatten[assigned_roi_feats], dim=0)
                value_update[i] = torch.sum(torch.unsqueeze(roi_feat_w, dim=1) * \
                                    f_v_flatten[assigned_roi_feats], dim=0)
                gamma[i, 0] = self.gamma * gamma[i, 0]

        if self.training:
            self.key.data = gamma * self.key.data + (1 - gamma) * key_update
            self.value.data = gamma * self.value.data + (1 - gamma) * value_update
    
    def _bbox_forward(self, bbox_feats: torch.Tensor):
        """Box head forward function used in both training and testing."""
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _feature_fusion(self, f_1, f_2, weight=None):
        """f_1 should be f_t and f_2 can be f_a or f_v."""
        if self.fusion_type == 'cat-conv':
            f = self.f_conv(torch.cat([f_1, f_2], dim=1))
        elif self.fusion_type == 'add':
            f = (f_1 + f_2) / 2
        return f

    def _bbox_forward_train(self, x1, x2, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = self.rois(bbox2roi([res.bboxes for res in sampling_results]))

        pos_inds = torch.ones((rois.shape[0],), dtype=torch.bool) #pos:1 neg:0
        start = 0
        for res in sampling_results:
            start += res.pos_inds.shape[0]
            pos_inds[start: start + res.neg_inds.shape[0]] = 0
            start += res.neg_inds.shape[0]
        pos_inds = self.pos_locs(pos_inds)

        bbox_head_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                    gt_labels, self.train_cfg)

        f_t = self.f_t(self.bbox_roi_extractor(
            x1[:self.bbox_roi_extractor.num_inputs], rois))
        f_v = self.f_v(self.bbox_roi_extractor(
            x2[:self.bbox_roi_extractor.num_inputs], rois))

        if not self.baseline:
            f_t_flat = f_t.flatten(1).detach()
            f_v_flat = f_v.flatten(1).detach()
            # init value at the first iter of training.
            loss_aux = dict()
            if not self.memory_init:
                self._memory_init(f_t_flat, f_v_flat, pos_inds)
            else:
                self._memory_update(f_t_flat, f_v_flat, loss_aux)

            f_a, reliable = self._memory_read(f_t)
            # loss_aux['not_reliable_num'] = (reliable.shape[0] - torch.sum(reliable))
            loss_aux['reliable_max'] = torch.max(reliable)
            loss_aux['reliable_min'] = torch.min(reliable)
            reliable = 1/(1+torch.exp(-self.k*(reliable-self.c)))
            f_a = f_a * reliable.view(f_a.shape[0], 1, 1, 1)

            f_c1 = self.f_c1(self._feature_fusion(f_t, f_a))
            bbox_results1 = self._bbox_forward(f_c1)

            f_c2 = self.f_c2(self._feature_fusion(f_t, f_v))
            bbox_results2 = self._bbox_forward(f_c2)

            # labels, label_weights, bbox_targets, bbox_weights = bbox_head_targets
            # label_weights_1 = label_weights * reliable
            # bbox_weights_1 = bbox_weights * torch.unsqueeze(reliable, dim=1)
            # bbox_head_targets1 = (labels, label_weights_1, bbox_targets, bbox_weights_1)

            loss_bbox1 = self.bbox_head.loss(bbox_results1['cls_score'],
                                            bbox_results1['bbox_pred'], rois,
                                            *bbox_head_targets)
            for key in loss_bbox1.keys():
                loss_bbox1[key + "_1"] = loss_bbox1.pop(key)

            loss_bbox2 = self.bbox_head.loss(bbox_results2['cls_score'],
                                            bbox_results2['bbox_pred'], rois,
                                            *bbox_head_targets)
            for key in loss_bbox2.keys():
                loss_bbox2[key + "_2"] = loss_bbox2.pop(key)

            bbox_results = dict()
            bbox_results.update(loss_bbox1=loss_bbox1)
            bbox_results.update(loss_bbox2=loss_bbox2)
            bbox_results.update(loss_aux=loss_aux)
        else:
            f_c = self._feature_fusion(f_t, f_v)
            bbox_results = self._bbox_forward(f_c)

            loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                            bbox_results['bbox_pred'], rois,
                                            *bbox_head_targets)
            bbox_results = dict()
            bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    async def async_test_bboxes(self,
                                    x1,
                                    x2,
                                    img_metas,
                                    proposals,
                                    rcnn_test_cfg,
                                    rescale=False,
                                    **kwargs):
            """Asynchronized test for box head without augmentation."""
            rois = bbox2roi(proposals)
            f_t = self.bbox_roi_extractor(
                x1[:len(self.bbox_roi_extractor.featmap_strides)], rois)
            if not self.baseline:
                f_a, _ = self._memory_read(f_t)
                f_c1 = self._feature_fusion(f_t, f_a)
            else:
                f_v = self.bbox_roi_extractor(
                    x2[:len(self.bbox_roi_extractor.featmap_strides)], rois)
                f_c1 = self._feature_fusion(f_t, f_v)
            sleep_interval = rcnn_test_cfg.get('async_sleep_interval', 0.017)

            async with completed(
                    __name__, 'bbox_head_forward',
                    sleep_interval=sleep_interval):
                cls_score, bbox_pred = self.bbox_head(f_c1)

            img_shape = img_metas[0]['img_shape']
            scale_factor = img_metas[0]['scale_factor']
            det_bboxes, det_labels = self.bbox_head.get_bboxes(
                rois,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=rescale,
                cfg=rcnn_test_cfg)
            return det_bboxes, det_labels

    async def async_simple_test(self,
                                x1,
                                x2,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x1, x2, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        return bbox_results

    def simple_test_bboxes(self,
                           x1,
                           x2,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x1 (tuple[Tensor]): Feature maps of all scale level.
            x2 can be None or the same as x1 (baseline setting).
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """
        rois = self.rois(bbox2roi(proposals))

        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0, ), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros(
                    (0, self.bbox_head.fc_cls.out_features))
            # There is no proposal in the whole batch
            return [det_bbox] * batch_size, [det_label] * batch_size

        if not self.baseline:
            f_t = self.f_t(self.bbox_roi_extractor(
                x1[:self.bbox_roi_extractor.num_inputs], rois))
            f_a, reliable = self._memory_read(f_t)
            f_a = f_a * reliable.view(f_a.shape[0], 1, 1, 1)
            f_c1 = self._feature_fusion(f_t, f_a)
            bbox_results = self._bbox_forward(f_c1)
        else:
            f_t = self.bbox_roi_extractor(
                x1[:self.bbox_roi_extractor.num_inputs], rois)
            f_v = self.bbox_roi_extractor(
                x2[:self.bbox_roi_extractor.num_inputs], rois)
            f_c = self._feature_fusion(f_t, f_v)

            bbox_results = self._bbox_forward(f_c)

        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0, ), dtype=torch.long)
                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros(
                        (0, self.bbox_head.fc_cls.out_features))

            else:
                det_bbox, det_label = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels

    def simple_test(self,
                    x1,
                    x2,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation.

        Args:
            x1 (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            x2 can be None or the same with x1 (baseline setting).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x1, x2, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        return bbox_results

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        raise NotImplementedError
