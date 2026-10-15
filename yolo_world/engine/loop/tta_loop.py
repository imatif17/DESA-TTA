# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from mmengine's TestLoop for DESA-TTA (Belal et al., 2026).
import copy
import logging
import math
from typing import Dict, Sequence, Tuple, Union

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from mmengine.evaluator import Evaluator
from mmengine.logging import HistoryBuffer, print_log
from mmengine.optim import OptimWrapper
from mmengine.registry import LOOPS
from mmengine.runner.amp import autocast
from mmengine.runner.base_loop import BaseLoop
from mmengine.structures import BaseDataElement
from mmengine.utils import is_list_of


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


@torch.no_grad()
def init_teacher_from_student(runner_model):
    teacher = copy.deepcopy(unwrap_model(runner_model))
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


@torch.no_grad()
def ema_update_teacher(teacher_model, student_model, momentum=0.9996):
    teacher = unwrap_model(teacher_model)
    student = unwrap_model(student_model)

    for teacher_param, student_param in zip(
            teacher.parameters(), student.parameters()):
        teacher_param.data.mul_(momentum).add_(
            student_param.data, alpha=1.0 - momentum)

    for teacher_buffer, student_buffer in zip(
            teacher.buffers(), student.buffers()):
        teacher_buffer.copy_(student_buffer)


class DriftRangeBoxScoreDynamicEMA:
    """Dynamic EMA coefficient for the mean-teacher update (D-EMA).

    For each test image the teacher momentum ``m`` is chosen inside a range
    ``[m_lo(d), m_hi(d)]`` that depends on the teacher drift ``d``:

    * ``s(d, lam) = d^p / (d^p + lam^p)``  (drift progress in [0, 1])
    * ``m_lo(d) = ema_min_early + (ema_min_late - ema_min_early) * s(d, ema_min_drift_scale)``
    * ``m_hi(d) = ema_max_early + (ema_max_late - ema_max_early) * s(d, ema_max_drift_scale)``

    Inside that range, ``m`` moves towards ``m_lo`` when the teacher is
    uncertain, measured from its pseudo-labels:

    * ``u = clip((N / M_max) * -log(mean_score), 0, 1)``
    * ``m = m_hi - (m_hi - m_lo) * u``

    where ``N`` is the number of teacher boxes above the pseudo-label
    threshold and ``mean_score`` their mean confidence. With
    ``stat_momentum=0`` and ``ema_smooth=0`` (the defaults used in the
    paper) no extra temporal smoothing is applied.
    """

    def __init__(self,
                 ema_min_early=0.9930,
                 ema_min_late=0.9990,
                 ema_max_early=0.9980,
                 ema_max_late=0.9996,
                 ema_min_drift_scale=0.017,
                 ema_max_drift_scale=0.014,
                 ema_drift_power=6.0,
                 stat_momentum=0.95,
                 ema_smooth=0.90,
                 max_per_img=300,
                 eps=1e-8):
        self.ema = None

        self.ema_min_early = float(ema_min_early)
        self.ema_min_late = float(ema_min_late)
        self.ema_max_early = float(ema_max_early)
        self.ema_max_late = float(ema_max_late)
        self.ema_min_drift_scale = float(ema_min_drift_scale)
        self.ema_max_drift_scale = float(ema_max_drift_scale)
        self.ema_drift_power = float(ema_drift_power)
        self.stat_momentum = float(stat_momentum)
        self.ema_smooth = float(ema_smooth)
        self.max_per_img = float(max_per_img)
        self.eps = float(eps)

        if not 0.0 <= self.ema_min_early <= self.ema_min_late <= 1.0:
            raise ValueError(
                'Expected 0 <= ema_min_early <= ema_min_late <= 1.')
        if not 0.0 <= self.ema_max_early <= self.ema_max_late <= 1.0:
            raise ValueError(
                'Expected 0 <= ema_max_early <= ema_max_late <= 1.')
        if self.ema_min_early > self.ema_max_early:
            raise ValueError('ema_min_early must not exceed ema_max_early.')
        if self.ema_min_late > self.ema_max_late:
            raise ValueError('ema_min_late must not exceed ema_max_late.')
        if self.ema_min_drift_scale <= 0.0:
            raise ValueError('ema_min_drift_scale must be positive.')
        if self.ema_max_drift_scale <= 0.0:
            raise ValueError('ema_max_drift_scale must be positive.')
        if self.ema_drift_power <= 0.0:
            raise ValueError('ema_drift_power must be positive.')
        if not 0.0 <= self.stat_momentum < 1.0:
            raise ValueError('stat_momentum must be in [0, 1).')
        if not 0.0 <= self.ema_smooth < 1.0:
            raise ValueError('ema_smooth must be in [0, 1).')
        if self.max_per_img <= 0.0:
            raise ValueError('max_per_img must be positive.')
        if self.eps <= 0.0:
            raise ValueError('eps must be positive.')

        self.box_ma = None
        self.score_ma = None
        self.last_stats = {}

    def _update_ma(self, old, new):
        new = float(new)
        if old is None:
            return new
        return self.stat_momentum * old + (1.0 - self.stat_momentum) * new

    @staticmethod
    def _clamp01(value):
        return min(max(float(value), 0.0), 1.0)

    def _drift_progress(self, drift, scale):
        drift_power = max(float(drift), 0.0) ** self.ema_drift_power
        scale_power = float(scale) ** self.ema_drift_power
        return self._clamp01(drift_power / (drift_power + scale_power))

    def _ema_range_from_drift(self, teacher_source_drift):
        min_progress = self._drift_progress(
            teacher_source_drift, self.ema_min_drift_scale)
        max_progress = self._drift_progress(
            teacher_source_drift, self.ema_max_drift_scale)

        ema_min = (
            self.ema_min_early
            + (self.ema_min_late - self.ema_min_early) * min_progress)
        ema_max = (
            self.ema_max_early
            + (self.ema_max_late - self.ema_max_early) * max_progress)

        ema_min = min(max(float(ema_min), 0.0), 1.0)
        ema_max = min(max(float(ema_max), ema_min), 1.0)
        return ema_min, ema_max

    def update(self,
               num_teacher_boxes,
               teacher_score_mean,
               teacher_source_drift,
               has_teacher_update=True):
        self.box_ma = self._update_ma(self.box_ma, num_teacher_boxes)
        self.score_ma = self._update_ma(
            self.score_ma,
            max(float(teacher_score_mean), self.eps))

        ema_min, ema_max = self._ema_range_from_drift(
            teacher_source_drift)
        drift_progress_min = self._drift_progress(
            teacher_source_drift, self.ema_min_drift_scale)
        drift_progress_max = self._drift_progress(
            teacher_source_drift, self.ema_max_drift_scale)

        density = self._clamp01(
            self.box_ma / max(self.max_per_img, 1.0))
        score_risk = -math.log(max(float(self.score_ma), self.eps))
        uncertainty = self._clamp01(density * score_risk)
        ema_target = ema_max - (ema_max - ema_min) * uncertainty

        if has_teacher_update:
            if self.ema is None:
                self.ema = ema_target
            else:
                self.ema = (
                    self.ema_smooth * self.ema
                    + (1.0 - self.ema_smooth) * ema_target)
            self.ema = min(max(float(self.ema), ema_min), ema_max)

        ema_value = float(self.ema if self.ema is not None else ema_target)
        self.last_stats = {
            'teacher_source_drift': float(teacher_source_drift),
            'drift_progress_min': float(drift_progress_min),
            'drift_progress_max': float(drift_progress_max),
            'ema_min': float(ema_min),
            'ema_max': float(ema_max),
            'uncertainty': float(uncertainty),
            'ema_target': float(ema_target),
            'ema_value': float(ema_value),
            'ema_range': float(ema_max - ema_min),
            'effective_teacher_update': float(
                (1.0 - ema_value) if has_teacher_update else 0.0),
            'num_teacher_boxes': int(num_teacher_boxes),
            'teacher_score_mean': float(teacher_score_mean),
            'box_ma': float(self.box_ma),
            'score_ma': float(self.score_ma),
        }

        return ema_value


def teacher_prediction_stats(outputs, score_thr=0.0):
    num_teacher_boxes = 0
    score_sum = 0.0
    score_count = 0

    for output in outputs:
        pred_instances = getattr(output, 'pred_instances', None)
        if pred_instances is None or not hasattr(pred_instances, 'scores'):
            continue

        scores = pred_instances.scores.detach().float()
        if score_thr is not None:
            scores = scores[scores > float(score_thr)]
        num_teacher_boxes += int(scores.numel())
        if scores.numel() > 0:
            score_sum += float(scores.sum().item())
            score_count += int(scores.numel())

    teacher_score_mean = (
        score_sum / score_count if score_count > 0 else 0.0)
    return num_teacher_boxes, teacher_score_mean


def split_weak_strong_to_two_batches(data_batch):
    weak_batch = {
        'inputs': data_batch['inputs']['weak'],
        'data_samples': data_batch['data_samples']['weak'],
    }
    strong_batch = {
        'inputs': data_batch['inputs']['strong'],
        'data_samples': data_batch['data_samples']['strong'],
    }
    return weak_batch, strong_batch


@LOOPS.register_module()
class TTALoop(BaseLoop):
    """DESA-TTA test loop: mean teacher + dynamic EMA + source anchoring.

    For every incoming batch (batch size 1 in the paper):

    1. The teacher predicts on the weak view. These predictions are the ones
       that are evaluated, and those above ``score_thr`` become pseudo-labels.
    2. The student (``runner.model``) takes one AdamW step on the strong view
       with the pseudo-labels. Only BatchNorm affine parameters are updated.
    3. Source anchoring: the student BN parameters are pulled back towards
       their pretrained values, ``theta <- (1 - a) * theta + a * theta_0`` with
       ``a = alpha_max * d^p / (d^p + lambda_alpha^p)``, where ``d`` is the
       relative L1 drift of the student from the source, averaged over BN
       layers.
    4. The teacher is updated by EMA with the dynamic momentum from
       :class:`DriftRangeBoxScoreDynamicEMA` (or ``ema`` if
       ``dynamic_ema=False``).

    Hyper-parameters are read from ``cfg.tta_lr``; see
    ``configs/voc_c/desa_tta_yolo_world_s_voc_c.py`` for the mapping to the
    paper's notation.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, dict],
                 evaluator: Union[Evaluator, dict, list],
                 fp16: bool = False):
        super().__init__(runner, dataloader)

        if isinstance(evaluator, (dict, list)):
            self.evaluator = runner.build_evaluator(evaluator)
        else:
            self.evaluator = evaluator

        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. `dataset_meta` in evaluator, metric and '
                'visualizer will be None.',
                logger='current',
                level=logging.WARNING)

        self.fp16 = fp16
        self.test_loss: Dict[str, HistoryBuffer] = {}

        for param in runner.model.parameters():
            param.requires_grad_(False)

        self.norm_params = []
        for module_name, module in runner.model.named_modules():
            if not isinstance(module, torch.nn.BatchNorm2d):
                continue
            for param_name, param in module.named_parameters():
                param.requires_grad_(True)
                self.norm_params.append(param)

        if not self.norm_params:
            raise RuntimeError('No BatchNorm2d affine parameters were found.')

        tta_lr = self.runner.cfg.tta_lr
        self.threshold = float(tta_lr.get('score_thr', 0.4))
        self.fixed_ema = float(tta_lr.get('ema', 0.9996))
        self.dynamic_ema = bool(tta_lr.get('dynamic_ema', True))

        self.bn_zs_reset_max = float(
            tta_lr.get('alpha_max', tta_lr.get('bn_zs_reset_max', 0.0)))
        self.bn_zs_drift_mode = str(
            tta_lr.get('bn_zs_drift_mode', 'avg')).lower()
        self.bn_zs_drift_eps = float(
            tta_lr.get('eps', tta_lr.get('bn_zs_drift_eps', 1e-6)))
        self.bn_zs_drift_scale = float(
            tta_lr.get('lambda_alpha', tta_lr.get('bn_zs_drift_scale', 0.085)))
        self.bn_zs_drift_power = float(tta_lr.get('p', 2.0))

        if not 0.0 <= self.bn_zs_reset_max < 1.0:
            raise ValueError('bn_zs_reset_max must be in [0, 1).')
        if self.bn_zs_drift_mode != 'avg':
            raise ValueError(
                'Only bn_zs_drift_mode="avg" (a single scalar anchoring '
                'strength) is supported.')
        if self.bn_zs_drift_eps <= 0.0:
            raise ValueError('bn_zs_drift_eps must be positive.')
        if self.bn_zs_drift_scale <= 0.0:
            raise ValueError('bn_zs_drift_scale must be positive.')
        if self.bn_zs_drift_power <= 0.0:
            raise ValueError('bn_zs_drift_power must be positive.')

        ema_min_late = float(tta_lr.get(
            'm_min', tta_lr.get('ema_min_late', 0.9990)))
        ema_max_late = float(tta_lr.get(
            'm_max', tta_lr.get('ema_max_late', 0.9996)))
        ema_min_early = float(tta_lr.get(
            'ema_min_early',
            ema_min_late - float(tta_lr.get(
                'delta_min', tta_lr.get('ema_min_gap', 0.0060)))))
        ema_max_early = float(tta_lr.get(
            'ema_max_early',
            ema_max_late - float(tta_lr.get(
                'delta_max', tta_lr.get('ema_max_gap', 0.0036)))))

        self.ema_controller = DriftRangeBoxScoreDynamicEMA(
            ema_min_early=ema_min_early,
            ema_min_late=ema_min_late,
            ema_max_early=ema_max_early,
            ema_max_late=ema_max_late,
            ema_min_drift_scale=tta_lr.get(
                'lambda_min', tta_lr.get('ema_min_drift_scale', 0.024)),
            ema_max_drift_scale=tta_lr.get(
                'lambda_max', tta_lr.get('ema_max_drift_scale', 0.014)),
            ema_drift_power=self.bn_zs_drift_power,
            stat_momentum=tta_lr.get('ema_stat_momentum', 0.0),
            ema_smooth=tta_lr.get('ema_smooth', 0.0),
            max_per_img=tta_lr.get(
                'M_max', tta_lr.get('ema_max_per_img', 300)),
            eps=tta_lr.get('eps', 1e-8))

        self.adapt_opt = optim.AdamW(
            [{'params': self.norm_params,
              'lr': float(tta_lr.get('norm', 6e-3))}],
            weight_decay=0)
        self.optim_wrapper = OptimWrapper(optimizer=self.adapt_opt)

        param_to_index = {
            id(param): index
            for index, param in enumerate(self.norm_params)
        }
        self._bn_zs_groups = []

        for module in unwrap_model(self.runner.model).modules():
            if not isinstance(module, torch.nn.BatchNorm2d):
                continue

            weight_index = (
                param_to_index.get(id(module.weight))
                if module.weight is not None else None)
            bias_index = (
                param_to_index.get(id(module.bias))
                if module.bias is not None else None)

            if weight_index is not None or bias_index is not None:
                self._bn_zs_groups.append((weight_index, bias_index))

        grouped_indices = {
            index
            for group in self._bn_zs_groups
            for index in group
            if index is not None
        }
        if len(grouped_indices) != len(self.norm_params):
            raise RuntimeError(
                'Could not associate every adapted BatchNorm2d parameter '
                'with its channel group.')

        self._teacher_norm_params_cache = None

    def run(self) -> dict:
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')

        self.teacher_model = init_teacher_from_student(self.runner.model)
        self._orig_norm_params = [
            param.detach().clone() for param in self.norm_params]
        self._teacher_norm_params_cache = None

        self.runner.model.eval()
        self.test_loss.clear()

        for idx, data_batch in enumerate(self.dataloader):
            self._adapt_and_test(idx, data_batch)

        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        if self.test_loss:
            metrics.update(_parse_losses(self.test_loss, 'test'))

        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')
        return metrics

    @torch.no_grad()
    def _bn_group_channel_drift(self, weight_index, bias_index):
        delta_abs = None
        source_abs = None

        for param_index in (weight_index, bias_index):
            if param_index is None:
                continue

            current = self.norm_params[param_index].detach().float()
            source = self._orig_norm_params[param_index].detach().to(
                device=current.device,
                dtype=torch.float32)

            current_delta_abs = torch.abs(current - source)
            current_source_abs = torch.abs(source)

            if delta_abs is None:
                delta_abs = current_delta_abs
                source_abs = current_source_abs
            else:
                delta_abs += current_delta_abs
                source_abs += current_source_abs

        if delta_abs is None:
            return None

        return delta_abs / torch.clamp(
            source_abs,
            min=self.bn_zs_drift_eps)

    def _drift_to_bn_zs_reset_alpha(self, drift):
        drift = torch.clamp(drift, min=0.0)
        drift_power = torch.pow(drift, self.bn_zs_drift_power)
        scale_power = self.bn_zs_drift_scale ** self.bn_zs_drift_power
        reset_alpha = self.bn_zs_reset_max * (
            drift_power / (drift_power + scale_power))
        return torch.clamp(
            reset_alpha,
            min=0.0,
            max=self.bn_zs_reset_max)

    @torch.no_grad()
    def _apply_scalar_bn_zs_reset(self, reset_alpha):
        reset_alpha = float(reset_alpha)
        for param, source in zip(self.norm_params, self._orig_norm_params):
            param.data.mul_(1.0 - reset_alpha).add_(
                source.to(device=param.device, dtype=param.dtype),
                alpha=reset_alpha)

    @torch.no_grad()
    def _apply_individual_bn_zs_reset(self, group_alphas):
        for (weight_index, bias_index), reset_alpha in zip(
                self._bn_zs_groups, group_alphas):
            for param_index in (weight_index, bias_index):
                if param_index is None:
                    continue

                param = self.norm_params[param_index]
                source = self._orig_norm_params[param_index].to(
                    device=param.device,
                    dtype=param.dtype)
                alpha = reset_alpha.to(
                    device=param.device,
                    dtype=param.dtype)
                param.data.copy_(param.data * (1.0 - alpha) + source * alpha)

    @torch.no_grad()
    def _soft_reset_bn_to_zs(self):
        default_stats = {
            'anchor_drift_mean': 0.0,
            'anchor_reset_alpha_mean': 0.0,
            'anchor_reset_alpha_max': 0.0,
        }

        group_drifts = [
            self._bn_group_channel_drift(weight_index, bias_index)
            for weight_index, bias_index in self._bn_zs_groups
        ]
        group_drifts = [drift for drift in group_drifts if drift is not None]
        if not group_drifts:
            return default_stats

        if self.bn_zs_drift_mode == 'avg':
            layer_drifts = torch.stack([
                drift.float().mean()
                for drift in group_drifts
            ])
            anchor_drift = layer_drifts.mean()
            anchor_drift_mean = float(anchor_drift.item())
            reset_alpha = float(
                self._drift_to_bn_zs_reset_alpha(
                    anchor_drift).item())
            if reset_alpha > 0.0:
                self._apply_scalar_bn_zs_reset(reset_alpha)
            return {
                'anchor_drift_mean': anchor_drift_mean,
                'anchor_reset_alpha_mean': reset_alpha,
                'anchor_reset_alpha_max': reset_alpha,
            }

        group_alphas = [
            self._drift_to_bn_zs_reset_alpha(drift)
            for drift in group_drifts
        ]
        if self.bn_zs_reset_max > 0.0:
            self._apply_individual_bn_zs_reset(group_alphas)

        all_drifts = torch.cat(
            [drift.reshape(-1) for drift in group_drifts])
        all_alphas = torch.cat(
            [alpha.reshape(-1) for alpha in group_alphas])
        return {
            'anchor_drift_mean': float(all_drifts.mean().item()),
            'anchor_reset_alpha_mean': float(all_alphas.mean().item()),
            'anchor_reset_alpha_max': float(all_alphas.max().item()),
        }

    @torch.no_grad()
    def _teacher_norm_params(self):
        if self._teacher_norm_params_cache is None:
            params = []
            for module in unwrap_model(self.teacher_model).modules():
                if not isinstance(module, torch.nn.BatchNorm2d):
                    continue
                if module.weight is not None:
                    params.append(module.weight)
                if module.bias is not None:
                    params.append(module.bias)

            if len(params) != len(self.norm_params):
                raise RuntimeError(
                    'Student and teacher have different numbers of adapted '
                    'BatchNorm2d parameters.')
            self._teacher_norm_params_cache = params

        return self._teacher_norm_params_cache

    @torch.no_grad()
    def _teacher_source_l1_drift(self):
        teacher_params = self._teacher_norm_params()
        layer_drifts = []

        for weight_index, bias_index in self._bn_zs_groups:
            delta_abs = None
            source_abs = None

            for param_index in (weight_index, bias_index):
                if param_index is None:
                    continue

                teacher_param = teacher_params[param_index].detach().float()
                source = self._orig_norm_params[param_index].detach().to(
                    device=teacher_param.device,
                    dtype=torch.float32)

                current_delta_abs = torch.abs(teacher_param - source)
                current_source_abs = torch.abs(source)

                if delta_abs is None:
                    delta_abs = current_delta_abs
                    source_abs = current_source_abs
                else:
                    delta_abs += current_delta_abs
                    source_abs += current_source_abs

            if delta_abs is None:
                continue

            channel_drift = delta_abs / torch.clamp(
                source_abs,
                min=self.bn_zs_drift_eps)

            layer_drifts.append(channel_drift.float().mean())

        if not layer_drifts:
            return 0.0

        return float(torch.stack(layer_drifts).mean().item())

    def _adapt_and_test(self,
                        idx: int,
                        data_batch: Sequence[dict]) -> None:
        weak_data_batch, strong_data_batch = \
            split_weak_strong_to_two_batches(data_batch)

        self.teacher_model.eval()
        self.runner.model.eval()

        with torch.no_grad():
            with autocast(enabled=self.fp16):
                outputs = self.teacher_model.test_step(weak_data_batch)

        outputs, self.test_loss = _update_losses(outputs, self.test_loss)

        train_batch = attach_pseudo_gt_weak_to_strong(
            weak_data_batch,
            strong_data_batch,
            outputs,
            score_thr=self.threshold)
        has_teacher_update = train_batch is not None

        if self.dynamic_ema:
            num_teacher_boxes, teacher_score_mean = \
                teacher_prediction_stats(outputs, score_thr=self.threshold)
            teacher_source_drift = self._teacher_source_l1_drift()
            ema_value = self.ema_controller.update(
                num_teacher_boxes=num_teacher_boxes,
                teacher_score_mean=teacher_score_mean,
                teacher_source_drift=teacher_source_drift,
                has_teacher_update=has_teacher_update)
        else:
            ema_value = self.fixed_ema

        self.evaluator.process(
            data_samples=outputs,
            data_batch=data_batch)
        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)

        if train_batch is None:
            return

        self.runner.model.train()
        for param in self.norm_params:
            param.requires_grad_(True)

        self.adapt_opt.zero_grad()
        self.runner.model.train_step(
            train_batch,
            optim_wrapper=self.optim_wrapper)

        self._soft_reset_bn_to_zs()

        ema_update_teacher(
            self.teacher_model,
            self.runner.model,
            momentum=float(ema_value))


def _to_3x3_homography(value, device):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        homography = value.to(device=device, dtype=torch.float32)
    else:
        homography = torch.tensor(
            value, device=device, dtype=torch.float32)

    if homography.numel() == 9:
        homography = homography.view(3, 3)
    if homography.shape != (3, 3):
        raise ValueError(
            f'homography_matrix must be (3, 3), got '
            f'{tuple(homography.shape)}')
    return homography


def _letterbox_homography_from_meta(meta, device):
    scale_factor = meta.get('scale_factor')
    pad_param = meta.get('pad_param')
    if scale_factor is None or pad_param is None:
        return None

    if isinstance(scale_factor, torch.Tensor):
        scale_w, scale_h = (
            float(scale_factor[0].item()),
            float(scale_factor[1].item()))
    else:
        scale_w, scale_h = map(float, scale_factor[:2])

    if isinstance(pad_param, torch.Tensor):
        top, _, left, _ = [float(value.item()) for value in pad_param]
    else:
        top, _, left, _ = map(float, pad_param)

    return torch.tensor(
        [[scale_w, 0.0, left],
         [0.0, scale_h, top],
         [0.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float32)


def _pad_only_homography_from_meta(meta, device):
    pad_param = meta.get('pad_param')
    if pad_param is None:
        return None

    if isinstance(pad_param, torch.Tensor):
        top, _, left, _ = [float(value.item()) for value in pad_param]
    else:
        top, _, left, _ = map(float, pad_param)

    return torch.tensor(
        [[1.0, 0.0, left],
         [0.0, 1.0, top],
         [0.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float32)


def _get_homography_orig_to_final(meta, device):
    homography = _to_3x3_homography(
        meta.get('homography_matrix'), device)
    if homography is None:
        return _letterbox_homography_from_meta(meta, device)

    pad_homography = _pad_only_homography_from_meta(meta, device)
    if pad_homography is None:
        return homography
    return pad_homography @ homography


def _apply_homography_to_boxes_xyxy(boxes, homography):
    if boxes.numel() == 0:
        return boxes

    x1, y1, x2, y2 = boxes.unbind(dim=1)
    corners = torch.stack([
        torch.stack([x1, y1], dim=1),
        torch.stack([x2, y1], dim=1),
        torch.stack([x2, y2], dim=1),
        torch.stack([x1, y2], dim=1),
    ], dim=1)

    ones = torch.ones(
        (*corners.shape[:2], 1),
        device=boxes.device,
        dtype=boxes.dtype)
    points = torch.cat([corners, ones], dim=2)
    transformed = points @ homography.transpose(0, 1)
    transformed_xy = (
        transformed[..., :2]
        / transformed[..., 2:3].clamp(min=1e-8))

    x_min = transformed_xy[..., 0].min(dim=1).values
    y_min = transformed_xy[..., 1].min(dim=1).values
    x_max = transformed_xy[..., 0].max(dim=1).values
    y_max = transformed_xy[..., 1].max(dim=1).values
    return torch.stack([x_min, y_min, x_max, y_max], dim=1)


def attach_pseudo_gt_weak_to_strong(weak_data_batch,
                                    strong_data_batch,
                                    weak_outputs,
                                    score_thr=0.3):
    weak_inputs = weak_data_batch['inputs']
    strong_inputs = strong_data_batch['inputs']
    weak_samples = weak_data_batch['data_samples']
    strong_samples = strong_data_batch['data_samples']

    if not isinstance(weak_inputs, list):
        raise RuntimeError(
            f"Expected weak_data_batch['inputs'] to be a list, "
            f'got {type(weak_inputs)}')
    if not isinstance(strong_inputs, list):
        raise RuntimeError(
            f"Expected strong_data_batch['inputs'] to be a list, "
            f'got {type(strong_inputs)}')

    batch_size = len(strong_inputs)
    if len(weak_inputs) != batch_size:
        raise RuntimeError(
            f'len(weak_inputs)={len(weak_inputs)} != '
            f'len(strong_inputs)={batch_size}')
    if len(weak_samples) != batch_size or len(strong_samples) != batch_size:
        raise RuntimeError(
            f'Mismatched sample counts: len(weak_samples)={len(weak_samples)}, '
            f'len(strong_samples)={len(strong_samples)}, B={batch_size}')
    if len(weak_outputs) != batch_size:
        raise RuntimeError(
            f'len(weak_outputs)={len(weak_outputs)} != batch size {batch_size}')

    device = strong_inputs[0].device
    inputs_tensor = torch.stack([
        image.to(device=device, dtype=torch.float32)
        for image in strong_inputs
    ], dim=0)

    all_bboxes_labels = []
    all_texts = []
    imgs_whwh_list = []
    n_max_boxes_list = []

    for image_index in range(batch_size):
        image = inputs_tensor[image_index]
        image_h, image_w = image.shape[-2:]
        imgs_whwh_list.append([
            float(image_w), float(image_h),
            float(image_w), float(image_h)])

        weak_output = weak_outputs[image_index]
        predictions = weak_output.pred_instances
        keep = predictions.scores > score_thr

        strong_sample = strong_samples[image_index]
        texts = getattr(strong_sample, 'texts', None)
        if texts is None and hasattr(strong_sample, 'metainfo'):
            texts = strong_sample.metainfo.get('texts')
        if texts is None:
            texts = getattr(weak_output, 'texts', None)
        all_texts.append(texts)

        if keep.sum() == 0:
            n_max_boxes_list.append(0.0)
            continue

        boxes = predictions.bboxes[keep].detach().to(
            device=device,
            dtype=torch.float32)
        labels = predictions.labels[keep].detach().to(
            device=device,
            dtype=torch.float32)

        homography = _get_homography_orig_to_final(
            strong_sample.metainfo, device)
        if homography is None:
            raise RuntimeError(
                f'Missing required metainfo for sample {image_index} '
                '(scale_factor/pad_param or homography_matrix).')

        boxes = _apply_homography_to_boxes_xyxy(boxes, homography)
        boxes[:, 0::2].clamp_(0.0, float(image_w - 1))
        boxes[:, 1::2].clamp_(0.0, float(image_h - 1))

        box_wh = (boxes[:, 2:4] - boxes[:, 0:2]).clamp(min=0.0)
        valid = (box_wh[:, 0] > 1.0) & (box_wh[:, 1] > 1.0)
        if valid.sum() == 0:
            n_max_boxes_list.append(0.0)
            continue

        boxes = boxes[valid]
        labels = labels[valid]
        num_boxes = int(boxes.shape[0])
        n_max_boxes_list.append(float(num_boxes))

        batch_indices = torch.full(
            (num_boxes, 1),
            float(image_index),
            device=device,
            dtype=torch.float32)
        all_bboxes_labels.append(torch.cat([
            batch_indices,
            labels.view(-1, 1),
            boxes,
        ], dim=1))

    if not all_bboxes_labels:
        return None

    return {
        'data_samples': {
            'bboxes_labels': torch.cat(all_bboxes_labels, dim=0),
            'texts': all_texts,
            'imgs_whwh': torch.tensor(
                imgs_whwh_list,
                device=device,
                dtype=torch.float32),
            'n_max_boxes': torch.tensor(
                n_max_boxes_list,
                device=device,
                dtype=torch.float32),
        },
        'inputs': inputs_tensor,
    }


def _parse_losses(losses: Dict[str, HistoryBuffer],
                  stage: str) -> Dict[str, float]:
    all_loss = 0
    loss_dict = {}

    for loss_name, loss_value in losses.items():
        avg_loss = loss_value.mean()
        loss_dict[loss_name] = avg_loss
        if 'loss' in loss_name:
            all_loss += avg_loss

    loss_dict[f'{stage}_loss'] = all_loss
    return loss_dict


def _update_losses(outputs: list,
                   losses: dict) -> Tuple[list, dict]:
    if (outputs
            and isinstance(outputs[-1], BaseDataElement)
            and outputs[-1].keys() == ['loss']):
        loss = outputs[-1].loss
        outputs = outputs[:-1]
    else:
        loss = {}

    for loss_name, loss_value in loss.items():
        if loss_name not in losses:
            losses[loss_name] = HistoryBuffer()
        if isinstance(loss_value, torch.Tensor):
            losses[loss_name].update(loss_value.item())
        elif is_list_of(loss_value, torch.Tensor):
            for loss_value_item in loss_value:
                losses[loss_name].update(loss_value_item.item())

    return outputs, losses
