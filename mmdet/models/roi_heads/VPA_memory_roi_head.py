# Author: Yuxuan Hu
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
class VPAMemoryRoIHead(BaseModule):
    """
        No mask head and shared head.
        Don't use seperate test_mixins.
        'baseline' means no VPA and both images are used in inference.
    """
    def __init__(self,
                 baseline,
                 vpa_slot_size,
                 dropout,
                 temperature,
                 loss_m1,
                 loss_m2,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(VPAMemoryRoIHead, self).__init__(init_cfg)
        self.baseline = baseline
        self.vpa_slot_size = vpa_slot_size
        self.temperature = temperature
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        feature_channel = bbox_head.in_channels
        roi_feat_size = bbox_head.roi_feat_size

        if bbox_head is not None:
            self.init_bbox_head(bbox_roi_extractor, bbox_head)

        self.dropout = nn.Dropout(dropout)
        self.f_conv = nn.Conv2d(2 * feature_channel, feature_channel, 1)
        if not self.baseline:
            # VPA Memory
            self.key = nn.Parameter(torch.Tensor(self.vpa_slot_size, 
                                        feature_channel * roi_feat_size * roi_feat_size))
            nn.init.normal_(self.key, 0, 0.5)
            self.value = nn.Parameter(torch.Tensor(self.vpa_slot_size, 
                                        feature_channel * roi_feat_size * roi_feat_size))
            nn.init.normal_(self.value, 0, 0.5)

            self.loss_m1 = build_loss(loss_m1)
            self.loss_m2 = build_loss(loss_m2)

        self.identity = nn.Identity()
        self.rois = nn.Identity()
        self.f_a = nn.Identity()
        self.f_v = nn.Identity()
        self.f_t = nn.Identity()
        
        self.init_assigner_sampler()

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
        f_a, _ = self._vpa_forward(f_t)
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
            proposals (list[Tensors]): list of region proposals.
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
                losses.update(bbox_results['loss_mal'])
            else:
                losses.update(bbox_results['loss_bbox'])

        return losses

    def _vpa_forward(self, f_t: torch.Tensor):
        """
            input: f_t (total_number_of_rois, feature_channel, roi_feat_size, roi_feat_size)
            output: f_a, has the same shape as f_t.
                    S_T, has shape (total_number_of_rois, vpa_slot_size)
        """
        ori_shape = f_t.shape
        f_t_flat = f_t.flatten(1)
        f_t_norm = F.normalize(f_t_flat, dim=1)
        norm_key = F.normalize(self.key, dim=1)
        S_T = torch.matmul(f_t_norm, norm_key.T)
        A = torch.softmax(S_T/self.temperature, dim=1)
        f_a = torch.matmul(A, self.value.detach())
        f_a = self.f_a(torch.reshape(f_a, ori_shape))
        return f_a, S_T

    def _vpa_forward_train(self, f_t: torch.Tensor, f_v: torch.Tensor):
        """
            input: f_t, f_v (total_number_of_rois, feature_channel, roi_feat_size, roi_feat_size)
            output: f_a, has the same shape as f_t.
        """
        f_a, S_T = self._vpa_forward(f_t)
        # S_T.shape: (total_number_of_rois, vpa_slot_size)
        f_v = f_v.flatten(1).detach()
        f_v_norm = F.normalize(f_v, dim=1)
        norm_value = F.normalize(self.value, dim=1)
        S_V = torch.matmul(f_v_norm, norm_value.T)
        loss_m2 = self.loss_m2(S_V, S_T.detach(), avg_factor=S_V.shape[0])
        return f_a, loss_m2
    
    def _bbox_forward(self, bbox_feats: torch.Tensor):
        """Box head forward function used in both training and testing."""
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x1, x2, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                    gt_labels, self.train_cfg)

        f_t = self.f_t(self.bbox_roi_extractor(
            x1[:self.bbox_roi_extractor.num_inputs], rois))
        f_v = self.f_v(self.bbox_roi_extractor(
            x2[:self.bbox_roi_extractor.num_inputs], rois))

        if not self.baseline:
            f_a, loss_m2 = self._vpa_forward_train(f_t, f_v)
            loss_m1 = self.loss_m1(f_a, f_v.detach())
            loss_mal = dict(loss_m1=loss_m1, loss_m2=loss_m2)
            f_c1 = self.f_conv(self.dropout(torch.cat((f_t, f_a), dim=1)))
            f_c2 = self.f_conv(self.dropout(torch.cat((f_t, f_v), dim=1)))
            bbox_results1 = self._bbox_forward(f_c1)
            bbox_results2 = self._bbox_forward(f_c2)

            loss_bbox1 = self.bbox_head.loss(bbox_results1['cls_score'],
                                            bbox_results1['bbox_pred'], rois,
                                            *bbox_targets)
            for key in loss_bbox1.keys():
                loss_bbox1[key + "_1"] = loss_bbox1.pop(key)

            loss_bbox2 = self.bbox_head.loss(bbox_results2['cls_score'],
                                            bbox_results2['bbox_pred'], rois,
                                            *bbox_targets)
            for key in loss_bbox2.keys():
                loss_bbox2[key + "_2"] = loss_bbox2.pop(key)

            bbox_results = dict()
            bbox_results.update(loss_bbox1=loss_bbox1)
            bbox_results.update(loss_bbox2=loss_bbox2)
            bbox_results.update(loss_mal=loss_mal)
        else:
            f_c = self.f_conv(self.dropout(torch.cat((f_t, f_v), dim=1)))
            bbox_results = self._bbox_forward(f_c)

            loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                            bbox_results['bbox_pred'], rois,
                                            *bbox_targets)
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
                f_a, _ = self._vpa_forward(f_t)
                f_c1 = self.f_conv(torch.cat((f_t, f_a), dim=1))
            else:
                f_v = self.bbox_roi_extractor(
                    x2[:len(self.bbox_roi_extractor.featmap_strides)], rois)
                f_c1 = self.f_conv(torch.cat((f_t, f_v), dim=1))
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
            f_t = self.bbox_roi_extractor(
                x1[:self.bbox_roi_extractor.num_inputs], rois)
            f_a, _ = self._vpa_forward(f_t)

            f_c1 = self.f_conv(torch.cat((f_t, f_a), dim=1))
            bbox_results = self._bbox_forward(f_c1)
        else:
            f_t = self.bbox_roi_extractor(
                x1[:self.bbox_roi_extractor.num_inputs], rois)
            f_v = self.bbox_roi_extractor(
                x2[:self.bbox_roi_extractor.num_inputs], rois)
            f_c = self.f_conv(torch.cat((f_t, f_v), dim=1))

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
