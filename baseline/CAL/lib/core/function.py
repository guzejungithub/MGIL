# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Hanbin Dai (daihanbin.ac@gmail.com)
# M by Chen Wang (wangchen199179@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os

import numpy as np
import torch

from core.evaluate import accuracy2, get_final_preds_offset
from utils.transforms import flip_back
from utils.vis import save_debug_images
#from testing import pickle_save as ps
#import sys
logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter2()

    in_size  = np.array(config.MODEL.IMAGE_SIZE)
    out_size = np.array(config.MODEL.HEATMAP_SIZE)
    stride = in_size / out_size

    # switch to train mode
    model.train()
    if config.TRAIN.FINETUNE_HM or config.TRAIN.FINETUNE_OM:
        model.module.freeze_weights(config.TRAIN.FINETUNE_HM,
                                    config.TRAIN.FINETUNE_OM)

    end = time.time()
    for i, (input, target, target_offset, mask_01, mask_g, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs, hm_hps = model(input)
        # mask = binary_mask(outputs.detach().cpu().numpy())
        # target_mask = target_mask * mask
        target = target.cuda(non_blocking=True)
        target_offset = target_offset.cuda(non_blocking=True)
        mask_01 = mask_01.cuda(non_blocking=True)
        mask_g = mask_g.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        # build GMM mask
        mask_g = criterion.gmm_mask(outputs, target, target_weight)

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
        else:
            output = outputs
            loss, offset_loss = criterion(output, hm_hps, target, target_offset,
                             mask_01, mask_g, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy2(output.detach().cpu().numpy(),
                                          hm_hps.detach().cpu().numpy(),
                                          target.detach().cpu().numpy(),
                                          target_offset.detach().cpu().numpy(),
                                          stride, config.DATASET.LOCREF_STDEV)

        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t' \
                  'Offset Loss {omloss:.5f}'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc, omloss= offset_loss.item())
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             epoch, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter2()

    in_size = np.array(config.MODEL.IMAGE_SIZE)
    out_size = np.array(config.MODEL.HEATMAP_SIZE)
    stride = in_size / out_size

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_offset, mask_01, mask_g, target_weight, meta) in enumerate(val_loader):
            # compute output
            outputs, output_offsets = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
                output_offset = output_offsets[-1]
            else:
                output = outputs
                output_offset = output_offsets

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                # input_flipped = torch.from_numpy(input_flipped).cuda()
                input_flipped = input.flip(3)
                outputs_flipped, output_offsets_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped
                    offset_flipped = output_offsets_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                offset_flipped = offset_flipped.cpu().numpy()
                offset_flipped[:, 0::2, :, :] = flip_back(
                    -offset_flipped[:, 0::2, :, :],
                    val_dataset.flip_pairs)
                offset_flipped[:, 1::2, :, :] = flip_back(
                    offset_flipped[:, 1::2, :, :],
                    val_dataset.flip_pairs)
                offset_flipped = torch.from_numpy(offset_flipped.copy()).cuda()

                output = (output + output_flipped) * 0.5
                output_offset = (output_offset + offset_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_offset = target_offset.cuda(non_blocking=True)
            mask_01 = mask_01.cuda(non_blocking=True)
            mask_g = mask_g.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss, offset_loss = criterion(output, output_offset, target, target_offset,
                             mask_01, mask_g, target_weight)

            num_images = input.size(0)

            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy2(output.detach().cpu().numpy(),
                                              output_offset.detach().cpu().numpy(),
                                              target.detach().cpu().numpy(),
                                              target_offset.detach().cpu().numpy(),
                                              stride, config.DATASET.LOCREF_STDEV)

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds_offset(output.clone().cpu().numpy(),
                output_offset.clone().cpu().numpy(), config.DATASET.LOCREF_STDEV,
                c, s, stride, in_size, out_size)
            #d = {'target': target.detach().cpu().numpy(),
            #     'target_off': target_offset.detach().cpu().numpy(),
            #     'target_mask': target_mask.detach().cpu().numpy(),
            #     'target_ce': target_mask.detach().cpu().numpy(),
            #     'target_weight': target_weight.detach().cpu().numpy(),
            #     'meta': meta}
            #d.update({'output': output.detach().cpu().numpy(),
            #          'output_offset': output_offset.detach().cpu().numpy(),
            #          'output_flipped': output_flipped.detach().cpu().numpy(),
            #          'offset_flipped': offset_flipped.detach().cpu().numpy()})
            #d.update({'input': input.detach().cpu().numpy(),
            #          'input_flipped': input_flipped.detach().cpu().numpy()})
            #ps(d); print('save!'); sys.exit(0)
            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


class AverageMeter2(AverageMeter):
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = np.zeros(3)
        self.avgs = np.zeros(3)

    def update(self, val, n=1):
        self.vals = val
        self.sum += val * n
        self.count += n
        for i in range(3):
            if self.count[i] != 0:
                self.avgs[i] = self.sum[i] / self.count[i]
        self.val = self.vals[0]
        self.avg = self.avgs[0]


def binary_mask(batch_heatmaps):
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    mask_reshaped = np.zeros_like(heatmaps_reshaped)
    idx = np.argmax(heatmaps_reshaped, 2)

    for b in range(batch_size):
        for k in range(num_joints):
            mask_reshaped[b, k, int(idx[b,k])] = 1
    mask = mask_reshaped.reshape((batch_size, num_joints, -1, width))

    return mask
