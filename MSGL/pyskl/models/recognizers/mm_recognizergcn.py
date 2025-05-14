from ..builder import RECOGNIZERS
from .base import BaseRecognizer

import numpy as np
import torch
import random

@RECOGNIZERS.register_module()
class MBRecognizerGCN(BaseRecognizer):
    """MultiModality GCN_j or b recognizer model framework."""
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
        labels = label
        assert self.with_cls_head
        assert keypoint.shape[1] == 1
        #数据增强方式加载        
        data_enhance = self.train_cfg.get('data_enhance', 'None')         
        keypoint = keypoint[:, 0]
        N, M, T, V, C = keypoint.shape

        losses = dict()
        if data_enhance == 'stackmix':
            shuffled_x, mixed_x, lam, batch_idx =  self.stackmix(keypoint, label, alpha=0.4, prob=0.5, nframes=T)
            keypoint = shuffled_x

            x_j, x_b  = self.backbone(keypoint)
            cls_scores = self.cls_head((x_j, x_b))

            gt_labels = labels.squeeze()
            loss_components = self.cls_head.loss_components
            loss_weights = self.cls_head.loss_weights
            # print(data_enhance)
            #借鉴mixup    
            for loss_name, weight in zip(loss_components, loss_weights):
                cls_score = cls_scores[loss_name]
                loss1 = self.cls_head.loss(cls_score, gt_labels)
                loss2 = self.cls_head.loss(cls_score, gt_labels[batch_idx])

                loss_cls ={}
                for (key1, value1), (key2, value2) in zip(loss1.items(), loss2.items()):
                    assert key1 == key2
                    loss_cls[key1] = lam * value1 + (1 - lam) * value2
                loss_cls = {loss_name + '_' + k: v for k, v in loss_cls.items()}
                loss_cls[f'{loss_name}_loss_cls'] *= weight
                losses.update(loss_cls)
        elif data_enhance == 'stackmixup':
            shuffled_x, mixed_x, lam, batch_idx =  self.stackmix(keypoint, label, alpha=0.4, prob=0.5, nframes=T)
            keypoint = (shuffled_x + mixed_x)/2

            x_j, x_b  = self.backbone(keypoint)
            cls_scores = self.cls_head((x_j, x_b))

            gt_labels = labels.squeeze()
            loss_components = self.cls_head.loss_components
            loss_weights = self.cls_head.loss_weights
            # print(data_enhance)
            #借鉴mixup    
            for loss_name, weight in zip(loss_components, loss_weights):
                cls_score = cls_scores[loss_name]
                loss1 = self.cls_head.loss(cls_score, gt_labels)
                loss2 = self.cls_head.loss(cls_score, gt_labels[batch_idx])
                loss_cls ={}
                for (key1, value1), (key2, value2) in zip(loss1.items(), loss2.items()):
                    assert key1 == key2
                    loss_cls[key1] = ((1+2*lam) * value1 + (3 - 2*lam) * value2)/4
                loss_cls = {loss_name + '_' + k: v for k, v in loss_cls.items()}
                loss_cls[f'{loss_name}_loss_cls'] *= weight
                losses.update(loss_cls)
        else:
            x_j, x_b = self.backbone(keypoint)
            # Which will return 2 cls_scores: ['j', 'b']
            cls_scores = self.cls_head((x_j, x_b))
            # print('None')
            gt_labels = labels.squeeze()
            loss_components = self.cls_head.loss_components
            loss_weights = self.cls_head.loss_weights
            for loss_name, weight in zip(loss_components, loss_weights):
                cls_score = cls_scores[loss_name]
                loss_cls = self.cls_head.loss(cls_score, gt_labels)
                loss_cls = {loss_name + '_' + k: v for k, v in loss_cls.items()}
                loss_cls[f'{loss_name}_loss_cls'] *= weight
                losses.update(loss_cls)
        # exit(0)
        return losses

    def forward_test(self, keypoint, **kwargs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        assert self.with_cls_head
        assert keypoint.shape[1] == 1         

        bs, nc = keypoint.shape[:2]
        keypoint = keypoint[:, 0]
        x_j, x_b = self.backbone(keypoint)
        cls_scores = self.cls_head((x_j, x_b))
        # print(cls_scores)
        # exit(0)
        for k in cls_scores:
            cls_score = self.average_clip(cls_scores[k][None])
            cls_scores[k] = cls_score.data.cpu().numpy()[0]

        # cuz we use extend for accumulation
        return [cls_scores]


    def forward(self, keypoint, label=None, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            return self.forward_train(keypoint, label, **kwargs)

        return self.forward_test(keypoint, **kwargs)
    
@RECOGNIZERS.register_module()
class MBmeanRecognizerGCN(BaseRecognizer):
    """MultiModality GCN_j or b recognizer model framework."""
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
        labels = label
        assert self.with_cls_head
        assert keypoint.shape[1] == 1
        #数据增强方式加载        
        data_enhance = self.train_cfg.get('data_enhance', 'None')         
        keypoint = keypoint[:, 0]
        N, M, T, V, C = keypoint.shape

        losses = dict()
        if data_enhance == 'stackmix':
            shuffled_x, mixed_x, lam, batch_idx =  self.stackmix(keypoint, label, alpha=0.4, prob=0.5, nframes=T)
            keypoint = shuffled_x

            x_j, x_b  = self.backbone(keypoint)
            cls_scores = self.cls_head((x_j, x_b))

            gt_labels = labels.squeeze()
            loss_components = self.cls_head.loss_components
            loss_weights = self.cls_head.loss_weights
            # print(data_enhance)
            #借鉴mixup    
            for loss_name, weight in zip(loss_components, loss_weights):
                cls_score = cls_scores[loss_name]
                loss1 = self.cls_head.loss(cls_score, gt_labels)
                loss2 = self.cls_head.loss(cls_score, gt_labels[batch_idx])

                loss_cls ={}
                for (key1, value1), (key2, value2) in zip(loss1.items(), loss2.items()):
                    assert key1 == key2
                    loss_cls[key1] = lam * value1 + (1 - lam) * value2
                loss_cls = {loss_name + '_' + k: v for k, v in loss_cls.items()}
                loss_cls[f'{loss_name}_loss_cls'] *= weight
                losses.update(loss_cls)
        elif data_enhance == 'stackmixup':
            shuffled_x, mixed_x, lam, batch_idx =  self.stackmix(keypoint, label, alpha=0.4, prob=0.5, nframes=T)
            keypoint = (shuffled_x + mixed_x)/2

            x_j, x_b  = self.backbone(keypoint)
            cls_scores = self.cls_head((x_j, x_b))

            gt_labels = labels.squeeze()
            loss_components = self.cls_head.loss_components
            loss_weights = self.cls_head.loss_weights
            # print(data_enhance)
            #借鉴mixup    
            for loss_name, weight in zip(loss_components, loss_weights):
                cls_score = cls_scores[loss_name]
                loss1 = self.cls_head.loss(cls_score, gt_labels)
                loss2 = self.cls_head.loss(cls_score, gt_labels[batch_idx])
                loss_cls ={}
                for (key1, value1), (key2, value2) in zip(loss1.items(), loss2.items()):
                    assert key1 == key2
                    loss_cls[key1] = ((1+2*lam) * value1 + (3 - 2*lam) * value2)/4
                loss_cls = {loss_name + '_' + k: v for k, v in loss_cls.items()}
                loss_cls[f'{loss_name}_loss_cls'] *= weight
                losses.update(loss_cls)
        else:
            x_j, x_b = self.backbone(keypoint)
            # Which will return 2 cls_scores: ['j', 'b']
            cls_scores = self.cls_head((x_j, x_b))
            # print('None')
            gt_labels = labels.squeeze()
            loss_components = self.cls_head.loss_components
            loss_weights = self.cls_head.loss_weights
            for loss_name, weight in zip(loss_components, loss_weights):
                cls_score = cls_scores[loss_name]
                loss_cls = self.cls_head.loss(cls_score, gt_labels)
                loss_cls = {loss_name + '_' + k: v for k, v in loss_cls.items()}
                loss_cls[f'{loss_name}_loss_cls'] *= weight
                losses.update(loss_cls)
        # exit(0)
        return losses

    def forward_test(self, keypoint, **kwargs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        assert self.with_cls_head
        assert keypoint.shape[1] == 1         

        bs, nc = keypoint.shape[:2]
        keypoint = keypoint[:, 0]
        x_j, x_b = self.backbone(keypoint)
        cls_scores = self.cls_head((x_j, x_b))
        # print(cls_scores)
        # exit(0)
        cls_score_list = []
        for k in cls_scores:
            cls_score = cls_scores[k].reshape(bs, nc, cls_scores[k].shape[-1])
            # print(cls_score.shape)
            # exit(0)
            cls_score_list.append(self.average_clip(cls_score))
        #     cls_scores[k] = cls_score.data.cpu().numpy()[0]
        #     cls_score_list.append(cls_scores[k])
        # cls_score_average = np.mean(cls_score_list, axis=0)


        cls_score_np = [x.data.cpu().numpy() for x in cls_score_list]
        # print(cls_score_np[0].shape)
        cls_score_avg = np.mean(cls_score_np, axis=0)
        # cls_score_max = np.max(cls_score_np, axis=0)
        # print(cls_score_out.shape)
        # print([[cls_score_out[i]] for i in range(bs)])
        # exit(0)
        return [cls_score_avg[i]for i in range(bs)]

    def forward(self, keypoint, label=None, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            return self.forward_train(keypoint, label, **kwargs)

        return self.forward_test(keypoint, **kwargs)
