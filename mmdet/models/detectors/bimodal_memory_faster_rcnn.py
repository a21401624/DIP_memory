# Author: Yuxuan Hu
import warnings
import torch

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from mmcv.runner import auto_fp16


@DETECTORS.register_module()
class BimodalMemoryFasterRCNN(BaseDetector):
    """
        We construct DIPMemoryRoIHead/ based on general roi_head.
        'baseline' means no VPA module and both images are used in inference.
    """
    def __init__(self,
                 backbone1,
                 backbone2,
                 baseline=False,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(BimodalMemoryFasterRCNN, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone1.pretrained = pretrained
            backbone2.pretrained = pretrained
        self.backbone1 = build_backbone(backbone1)
        self.backbone2 = build_backbone(backbone2)

        self.baseline = baseline

        if neck is not None:
            self.neck = build_neck(neck)
            self.neck2 = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.baseline = baseline
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, img1, img2):
        """
            This method is abstract method so we must implement it. 
            So we use it in the training phase.
        """
        x1 = self.backbone1(img1)
        x2 = self.backbone2(img2)
        if self.with_neck:
            x1 = self.neck(x1)
            x2 = self.neck2(x2)
        return x1, x2

    def forward_dummy(self, img1, img2):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x1 = self.backbone1(img1)
        if self.with_neck:
            x1 = self.neck(x1)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x1)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img1.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x1, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img1,
                      img2,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img1, img2 (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()

        x1, x2 = self.extract_feat(img1, img2)

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x1,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x1, x2, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks)
        losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img1,
                                img2,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x1 = self.backbone1(img1)
        if self.with_neck:
            x1 = self.neck(x1)
        if self.baseline:
            x2 = self.backbone2(img2)
            if self.with_neck:
                x2 = self.neck2(x2)
        else:
            x2 = None

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x1, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x1, x2, proposal_list, img_meta, rescale=rescale)

    async def aforward_test(self, *, img1, img2, img_metas, **kwargs):
        for var, name in [(img1, 'img1'), (img2, 'img2'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(img1)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(img1)}) '
                             f'!= num of image metas ({len(img_metas)})')
        # TODO: remove the restriction of samples_per_gpu == 1 when prepared
        samples_per_gpu = img1[0].size(0)
        assert samples_per_gpu == 1

        if num_augs == 1:
            return await self.async_simple_test(img1[0], img2[0], img_metas[0], **kwargs)
        else:
            raise NotImplementedError

    def simple_test(self, img1, img2, img_metas, proposals=None, rescale=False, **kwargs):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x1 = self.backbone1(img1)
        if self.with_neck:
            x1 = self.neck(x1)
        if self.baseline:
            x2 = self.backbone2(img2)
            if self.with_neck:
                x2 = self.neck2(x2)
        else:
            x2 = None

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x1, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x1, x2, proposal_list, img_metas, rescale=rescale)

    def forward_test(self, imgs1, imgs2, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs1, 'imgs1'), (imgs2, 'imgs2'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs1)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs1)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img1, img_meta in zip(imgs1, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img1.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs1[0], imgs2[0], img_metas[0], **kwargs)
        else:
            assert imgs1[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs1[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs1, imgs2, img_metas, **kwargs)

    def aug_test(self, imgs1, imgs2, img_metas, rescale=False, **kwargs):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        raise NotImplementedError

    @auto_fp16(apply_to=('img1', 'img2', ))
    def forward(self, img1, img2, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img1[0], img2[0], img_metas[0])

        if return_loss:
            return self.forward_train(img1, img2, img_metas, **kwargs)
        else:
            return self.forward_test(img1, img2, img_metas, **kwargs)

