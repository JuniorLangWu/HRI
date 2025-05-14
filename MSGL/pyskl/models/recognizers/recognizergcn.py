from ..builder import RECOGNIZERS
from .base import BaseRecognizer

import numpy as np
import torch
import random


@RECOGNIZERS.register_module()
class RecognizerGCN(BaseRecognizer):
    """GCN-based recognizer for skeleton-based action recognition. """
    def stackmix(self, x, y, alpha, prob, nframes=32):
        if prob < 0:
            raise ValueError('prob must be a positive value')
        
        batch_size = x.size()[0]
        k = random.random()
        lam = 1
        batch_idx = torch.arange(batch_size)
        if k > 1 - prob:
            batch_idx = torch.randperm(batch_size)
            lam = np.random.beta(alpha, alpha)
            cut_idx = int(lam * nframes)
            shuffled_x = torch.cat((x[:, :, :cut_idx, :, :], x[batch_idx][:, :, cut_idx:, :, :]), dim=2)    #[N, M, T, V, C]
            mixed_x = lam * x + (1 - lam) * x[batch_idx]
            # shuffled_y = torch.cat((y[:, :cut_idx], y[batch_idx][:, cut_idx:]), dim=1)    #[B, T]
            # cls_y = torch.cat((y[:, :cut_idx] * (cut_idx / nframes), y[batch_idx][:, cut_idx:] * (1 - cut_idx / nframes)), dim=1)
            return shuffled_x, mixed_x, lam, batch_idx
        else:
            return x, x, lam, batch_idx

    def forward_train(self, keypoint, label, **kwargs):
        """Defines the computation performed at every call when training."""
        assert self.with_cls_head
        assert keypoint.shape[1] == 1
        
        #数据增强方式加载        
        data_enhance = self.train_cfg.get('data_enhance', 'None')
        # print(data_enhance)
        # exit(0)
        # print(keypoint.shape)
        # print(label[0])
        # exit(0)
        keypoint = keypoint[:, 0]
        N, M, T, V, C = keypoint.shape
        # label = label.unsqueeze(1) 
        # label = label.expand(N, T)
        losses = dict()
        if data_enhance == 'stackmix':
            shuffled_x, mixed_x, lam, batch_idx =  self.stackmix(keypoint, label, alpha=0.4, prob=0.5, nframes=T)
            keypoint = shuffled_x
            x = self.extract_feat(keypoint)
            cls_score = self.cls_head(x)
            gt_label = label.squeeze(-1)
            # print(data_enhance)
            #借鉴mixup
            loss1 = self.cls_head.loss(cls_score, gt_label)
            loss2 = self.cls_head.loss(cls_score, gt_label[batch_idx])
            loss ={}
            for (key1, value1), (key2, value2) in zip(loss1.items(), loss2.items()):
                assert key1 == key2
                loss[key1] = lam * value1 + (1 - lam) * value2

        elif data_enhance == 'stackmixup':
            shuffled_x, mixed_x, lam, batch_idx =  self.stackmix(keypoint, label, alpha=0.4, prob=0.5, nframes=T)
            # keypoint = shuffled_x
            keypoint = (shuffled_x + mixed_x)/2
            x = self.extract_feat(keypoint)
            cls_score = self.cls_head(x)
            gt_label = label.squeeze(-1)
            # print(data_enhance)
            #借鉴mixup
            loss1 = self.cls_head.loss(cls_score, gt_label)
            loss2 = self.cls_head.loss(cls_score, gt_label[batch_idx])
            loss ={}
            for (key1, value1), (key2, value2) in zip(loss1.items(), loss2.items()):
                assert key1 == key2
                loss[key1] = ((1+2*lam) * value1 + (3 - 2*lam) * value2)/4

        elif data_enhance == 'mixup':
            shuffled_x, mixed_x, lam, batch_idx =  self.stackmix(keypoint, label, alpha=0.4, prob=0.5, nframes=T)
            # keypoint = shuffled_x
            keypoint = mixed_x
            x = self.extract_feat(keypoint)
            cls_score = self.cls_head(x)
            gt_label = label.squeeze(-1)
            # print(data_enhance)
            #借鉴mixup
            loss1 = self.cls_head.loss(cls_score, gt_label)
            loss2 = self.cls_head.loss(cls_score, gt_label[batch_idx])
            loss ={}
            for (key1, value1), (key2, value2) in zip(loss1.items(), loss2.items()):
                assert key1 == key2
                loss[key1] = lam * value1 + (1 - lam) * value2

        else:
            x = self.extract_feat(keypoint)
            cls_score = self.cls_head(x)
            gt_label = label.squeeze(-1)
            # print('None')
            loss = self.cls_head.loss(cls_score, gt_label)

        losses.update(loss)
        # exit(0)
        return losses

    def forward_test(self, keypoint, **kwargs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        assert self.with_cls_head or self.feat_ext
        bs, nc = keypoint.shape[:2]
        keypoint = keypoint.reshape((bs * nc, ) + keypoint.shape[2:])

        x = self.extract_feat(keypoint)
        feat_ext = self.test_cfg.get('feat_ext', False)
        pool_opt = self.test_cfg.get('pool_opt', 'all')
        score_ext = self.test_cfg.get('score_ext', False)
        if feat_ext or score_ext:
            assert bs == 1
            assert isinstance(pool_opt, str)
            dim_idx = dict(n=0, m=1, t=3, v=4)

            if pool_opt == 'all':
                pool_opt == 'nmtv'
            if pool_opt != 'none':
                for digit in pool_opt:
                    assert digit in dim_idx

            if isinstance(x, tuple) or isinstance(x, list):
                x = torch.cat(x, dim=2)
            assert len(x.shape) == 5, 'The shape is N, M, C, T, V'
            if pool_opt != 'none':
                for d in pool_opt:
                    x = x.mean(dim_idx[d], keepdim=True)

            if score_ext:
                w = self.cls_head.fc_cls.weight
                b = self.cls_head.fc_cls.bias
                x = torch.einsum('nmctv,oc->nmotv', x, w)
                if b is not None:
                    x = x + b[..., None, None]
                x = x[None]
            return x.data.cpu().numpy().astype(np.float16)

        cls_score = self.cls_head(x)
        cls_score = cls_score.reshape(bs, nc, cls_score.shape[-1])
        # harmless patch
        if 'average_clips' not in self.test_cfg:
            self.test_cfg['average_clips'] = 'prob'
        # print(cls_score.shape)
        cls_score = self.average_clip(cls_score)
        # print(cls_score.shape)
        # exit(0)
        if isinstance(cls_score, tuple) or isinstance(cls_score, list):
            cls_score = [x.data.cpu().numpy() for x in cls_score]
            return [[x[i] for x in cls_score] for i in range(bs)]

        return cls_score.data.cpu().numpy()

    def forward(self, keypoint, label=None, return_loss=True, **kwargs):
        # print(label)
        # exit(0)
        """Define the computation performed at every call."""
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            return self.forward_train(keypoint, label, **kwargs)

        return self.forward_test(keypoint, **kwargs)

    def extract_feat(self, keypoint):
        """Extract features through a backbone.

        Args:
            keypoint (torch.Tensor): The input keypoints.

        Returns:
            torch.tensor: The extracted features.
        """
        return self.backbone(keypoint)
