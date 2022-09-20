import argparse
import os
import datetime
import logging
import time
import math
import numpy as np
from collections import OrderedDict

import torch
import torch.nn.functional as F

from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_model, build_adversarial_discriminator, build_feature_extractor, build_classifier,build_dual
from core.solver import adjust_learning_rate
from core.utils.misc import mkdir, AverageMeter, intersectionAndUnionGPU
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger

def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict

def cos_distance(vector1,vector2):
    dot_product = torch.dot(vector1,vector2)
    normA = torch.dot(vector1,vector1)
    normB = torch.dot(vector2,vector2)
    return dot_product / ((normA*normB)**0.5)

def soft_label_cross_entropy(pred, soft_label, pixel_weights=None):
    loss = -soft_label.float()*F.log_softmax(pred, dim=1)
    if pixel_weights is None:
        return torch.mean(torch.sum(loss, dim=1))
    return torch.mean(pixel_weights*torch.sum(loss, dim=1))

def train(cfg, local_rank, distributed):
    logger = logging.getLogger("FADA.trainer")
    logger.info("Start training")

    feature_extractor = build_feature_extractor(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    feature_extractor.to(device)


    dual1 = build_dual(cfg)
    dual1.to(device)
    dual2 = build_dual(cfg)
    dual2.to(device)

    classifier = build_classifier(cfg)
    classifier.to(device)
    classifierall = build_classifier(cfg)
    classifierall.to(device)
    classifier1 = build_classifier(cfg)
    classifier1.to(device)
    classifier2 = build_classifier(cfg)
    classifier2.to(device)

    model_D = build_adversarial_discriminator(cfg)
    model_D.to(device)
    model_D1 = build_adversarial_discriminator(cfg)
    model_D1.to(device)

    if local_rank==0:
        print(feature_extractor)
        print(model_D)

    batch_size = cfg.SOLVER.BATCH_SIZE//3
    print(batch_size)
    
    optimizer_fea = torch.optim.SGD(feature_extractor.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_fea.zero_grad()


    optimizer_dual1 = torch.optim.SGD(dual1.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_dual1.zero_grad()
    optimizer_dual2 = torch.optim.SGD(dual2.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_dual2.zero_grad()

    optimizer_cls = torch.optim.SGD(classifier.parameters(), lr=cfg.SOLVER.BASE_LR*10, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_cls.zero_grad()
    optimizer_clsall = torch.optim.SGD(classifierall.parameters(), lr=cfg.SOLVER.BASE_LR * 10, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_clsall.zero_grad()

    optimizer_cls1 = torch.optim.SGD(classifier1.parameters(), lr=cfg.SOLVER.BASE_LR * 10, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_cls1.zero_grad()
    optimizer_cls2 = torch.optim.SGD(classifier2.parameters(), lr=cfg.SOLVER.BASE_LR * 10, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_cls2.zero_grad()

    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=cfg.SOLVER.BASE_LR_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()
    optimizer_D1 = torch.optim.Adam(model_D1.parameters(), lr=cfg.SOLVER.BASE_LR_D, betas=(0.9, 0.99))
    optimizer_D1.zero_grad()

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = local_rank == 0

    start_epoch = 0
    iteration = 0
    
    if cfg.resume:
        logger.info("Loading checkpoint from {}".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
        model_weights = checkpoint['feature_extractor'] if distributed else strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
        feature_extractor.load_state_dict(model_weights)
        classifier_weights = checkpoint['classifier'] if distributed else strip_prefix_if_present(checkpoint['classifier'], 'module.')
        classifier.load_state_dict(classifier_weights)
        classifier_weights1 = checkpoint['classifier'] if distributed else strip_prefix_if_present(checkpoint['classifier'], 'module.')
        classifier1.load_state_dict(classifier_weights1)
        classifier_weights2 = checkpoint['classifier'] if distributed else strip_prefix_if_present(checkpoint['classifier'], 'module.')
        classifier2.load_state_dict(classifier_weights2)
        classifier_weightsa = checkpoint['classifier'] if distributed else strip_prefix_if_present(checkpoint['classifier'], 'module.')
        classifierall.load_state_dict(classifier_weightsa)
        if "dual1" in checkpoint:
            logger.info("Loading dual from {}".format(cfg.resume))
            dual_weights1 = checkpoint['dual1'] if distributed else strip_prefix_if_present(checkpoint['dual1'], 'module.')
            dual1.load_state_dict(dual_weights1)
        if "dual2" in checkpoint:
            logger.info("Loading dual from {}".format(cfg.resume))
            dual_weights2 = checkpoint['dual2'] if distributed else strip_prefix_if_present(checkpoint['dual2'], 'module.')
            dual2.load_state_dict(dual_weights2)
        if "model_D" in checkpoint:
            logger.info("Loading model_D from {}".format(cfg.resume))
            model_D_weights = checkpoint['model_D'] if distributed else strip_prefix_if_present(checkpoint['model_D'], 'module.')
            model_D.load_state_dict(model_D_weights)
        if "model_D1" in checkpoint:
            logger.info("Loading model_D1 from {}".format(cfg.resume))
            model_D1_weights = checkpoint['model_D1'] if distributed else strip_prefix_if_present(checkpoint['model_D1'], 'module.')
            model_D1.load_state_dict(model_D1_weights)


    src_train_data = build_dataset(cfg, mode='train', is_source=True, index=0)
    tgt_train_data = build_dataset(cfg, mode='train', is_source=False, index=0)
    tgt_train_data1 = build_dataset(cfg, mode='train', is_source=False, index=1)

    if distributed:
        src_train_sampler = torch.utils.data.distributed.DistributedSampler(src_train_data)
        tgt_train_sampler = torch.utils.data.distributed.DistributedSampler(tgt_train_data)
    else:
        src_train_sampler = None
        tgt_train_sampler = None
    
    src_train_loader = torch.utils.data.DataLoader(src_train_data, batch_size=batch_size, shuffle=(src_train_sampler is None), num_workers=4, pin_memory=True, sampler=src_train_sampler, drop_last=True)
    tgt_train_loader = torch.utils.data.DataLoader(tgt_train_data, batch_size=batch_size, shuffle=(tgt_train_sampler is None), num_workers=4, pin_memory=True, sampler=tgt_train_sampler, drop_last=True)
    tgt_train_loader1 = torch.utils.data.DataLoader(tgt_train_data1, batch_size=batch_size, shuffle=(tgt_train_sampler is None), num_workers=4, pin_memory=True, sampler=tgt_train_sampler, drop_last=True)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    KL_loss = torch.nn.KLDivLoss()

    alpha = cfg.SOLVER.Alpha
    max_iters = cfg.SOLVER.MAX_ITER
    source_label = 0
    target_label = 1
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    feature_extractor.train()
    classifier.train()
    classifier1.train()
    classifier2.train()
    classifierall.train()
    dual1.train()
    dual2.train()
    model_D.train()
    model_D1.train()
    start_training_time = time.time()
    end = time.time()
    for i, ((src_input, src_label, src_name), (tgt_input, _, _), (tgt_input1, _, _)) in enumerate(zip(src_train_loader, tgt_train_loader, tgt_train_loader1)):
        data_time = time.time() - end

        current_lr = adjust_learning_rate(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR, iteration, max_iters, power=cfg.SOLVER.LR_POWER)
        current_lr_D = adjust_learning_rate(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR_D, iteration, max_iters, power=cfg.SOLVER.LR_POWER)
        for index in range(len(optimizer_fea.param_groups)):
            optimizer_fea.param_groups[index]['lr'] = current_lr
        for index in range(len(optimizer_dual1.param_groups)):
            optimizer_dual1.param_groups[index]['lr'] = current_lr
        for index in range(len(optimizer_dual2.param_groups)):
            optimizer_dual2.param_groups[index]['lr'] = current_lr
        for index in range(len(optimizer_cls.param_groups)):
            optimizer_cls.param_groups[index]['lr'] = current_lr * 10
        for index in range(len(optimizer_clsall.param_groups)):
            optimizer_clsall.param_groups[index]['lr'] = current_lr * 10
        for index in range(len(optimizer_cls1.param_groups)):
            optimizer_cls1.param_groups[index]['lr'] = current_lr * 10
        for index in range(len(optimizer_cls2.param_groups)):
            optimizer_cls2.param_groups[index]['lr'] = current_lr * 10
        for index in range(len(optimizer_D1.param_groups)):
            optimizer_D1.param_groups[index]['lr'] = current_lr_D
        for index in range(len(optimizer_D.param_groups)):
            optimizer_D.param_groups[index]['lr'] = current_lr_D

        feature_extractor.train()
        dual1.train()
        dual2.train()
        classifier.train()
        classifierall.train()
        classifier1.train()
        classifier2.train()

#       torch.distributed.barrier()

        optimizer_fea.zero_grad()
        optimizer_cls.zero_grad()
        optimizer_clsall.zero_grad()
        optimizer_cls1.zero_grad()
        optimizer_cls2.zero_grad()
        optimizer_dual1.zero_grad()
        optimizer_dual2.zero_grad()


        optimizer_D.zero_grad()
        optimizer_D1.zero_grad()

        src_input = src_input.cuda(non_blocking=True)
        src_label = src_label.cuda(non_blocking=True).long()
        tgt_input = tgt_input.cuda(non_blocking=True)
        tgt_input1 = tgt_input1.cuda(non_blocking=True)
        src_size = src_input.shape[-2:]
        tgt_size = tgt_input.shape[-2:]
        tgt_size1 = tgt_input1.shape[-2:]


        src_fea = feature_extractor(src_input)
        src_br1 = dual1(src_fea)
        src_br2 = dual2(src_fea)
        src_pred1 = classifier(src_br1, src_size)
        src_pred2 = classifier(src_br2, src_size)
        temperature = 1.8
        src_pred1 = src_pred1.div(temperature)
        src_pred2 = src_pred2.div(temperature)
        loss_seg1 = criterion(src_pred1, src_label)
        loss_seg2 = criterion(src_pred2, src_label)
        loss_seg = 0.5*loss_seg1 + 0.5*loss_seg2
        loss_seg.backward()

        # generate soft labels
        src_soft_label1 = F.softmax(src_pred1, dim=1).detach()
        src_soft_label1[src_soft_label1>0.9] = 0.9
        src_soft_label2 = F.softmax(src_pred2, dim=1).detach()
        src_soft_label2[src_soft_label2 > 0.9] = 0.9
        
        tgt_fea_f = feature_extractor(tgt_input)
        tgt_fea = dual1(tgt_fea_f)
        tgt_pred = classifier(tgt_fea, tgt_size)
        tgt_pred = tgt_pred.div(temperature)
        tgt_soft_label = F.softmax(tgt_pred, dim=1)
        tgt_soft_label = tgt_soft_label.detach()
        tgt_soft_label[tgt_soft_label>0.9] = 0.9

        tgt_fea_f1 = feature_extractor(tgt_input1)
        tgt_fea1 = dual2(tgt_fea_f1)
        tgt_pred1 = classifier(tgt_fea1, tgt_size1)
        tgt_pred1 = tgt_pred1.div(temperature)
        tgt_soft_label1 = F.softmax(tgt_pred1, dim=1)
        tgt_soft_label1 = tgt_soft_label1.detach()
        tgt_soft_label1[tgt_soft_label1 > 0.9] = 0.9

        tgt_D_pred = model_D(tgt_fea, tgt_size)
        loss_adv_tgt = 0.001*soft_label_cross_entropy(tgt_D_pred, torch.cat((tgt_soft_label, torch.zeros_like(tgt_soft_label)), dim=1))
        loss_adv_tgt.backward(retain_graph=True)

        tgt_D_pred1 = model_D1(tgt_fea1, tgt_size1)
        loss_adv_tgt1 = 0.001*soft_label_cross_entropy(tgt_D_pred1, torch.cat((tgt_soft_label1, torch.zeros_like(tgt_soft_label1)), dim=1))
        loss_adv_tgt1.backward(retain_graph=True)

        tgt_cls = classifier1(tgt_fea_f, tgt_size)
        tgt_cls1 = classifier2(tgt_fea_f1, tgt_size1)

        loss_cls = soft_label_cross_entropy(tgt_cls, tgt_soft_label)
        loss_cls1 = soft_label_cross_entropy(tgt_cls1, tgt_soft_label1)
        loss_clsall = alpha*loss_cls + alpha*loss_cls1
        loss_clsall.backward()

        tgt_clsall = classifierall(tgt_fea_f.detach(), tgt_size)
        tgt_clsall1 = classifierall(tgt_fea_f1.detach(), tgt_size1)
        loss_kl = KL_loss(tgt_cls.softmax(dim=1).log().detach(),tgt_clsall.softmax(dim=1))
        loss_kl1 = KL_loss(tgt_cls1.softmax(dim=1).log().detach(), tgt_clsall1.softmax(dim=1))

        loss_kl_all = loss_kl+loss_kl1
        loss_kl_all.backward()

        optimizer_fea.step()
        optimizer_dual1.step()
        optimizer_dual2.step()
        optimizer_cls.step()
        optimizer_cls2.step()
        optimizer_cls1.step()
        optimizer_clsall.step()

        optimizer_D.zero_grad()
        optimizer_D1.zero_grad()

        src_D_pred = model_D(src_br1.detach(), src_size)
        loss_D_src = 0.5*soft_label_cross_entropy(src_D_pred, torch.cat((src_soft_label1, torch.zeros_like(src_soft_label1)), dim=1))
        loss_D_src.backward()
        tgt_D_pred = model_D(tgt_fea.detach(), tgt_size)
        loss_D_tgt = 0.5*soft_label_cross_entropy(tgt_D_pred, torch.cat((torch.zeros_like(tgt_soft_label), tgt_soft_label), dim=1))
        loss_D_tgt.backward()

        src_D_pred1 = model_D1(src_br2.detach(), src_size)
        loss_D_src1 = 0.5 * soft_label_cross_entropy(src_D_pred1, torch.cat((src_soft_label2, torch.zeros_like(src_soft_label2)), dim=1))
        loss_D_src1.backward()
        tgt_D_pred1 = model_D1(tgt_fea1.detach(), tgt_size1)
        loss_D_tgt1 = 0.5 * soft_label_cross_entropy(tgt_D_pred1, torch.cat((torch.zeros_like(tgt_soft_label1), tgt_soft_label1), dim=1))
        loss_D_tgt1.backward()

        # torch.distributed.barrier()
        optimizer_D.step()
        optimizer_D1.step()


        meters.update(loss_seg=loss_seg.item())
        meters.update(loss_adv_tgt=loss_adv_tgt.item())
        meters.update(loss_clsall=loss_clsall.item())
        meters.update(loss_cls=loss_cls.item())
        meters.update(loss_cls1=loss_cls1.item())
        meters.update(loss_kl_all=loss_kl_all.item())
        meters.update(loss_D=(loss_D_src.item()+loss_D_tgt.item()))
        meters.update(loss_D_src=loss_D_src.item())
        meters.update(loss_D_tgt=loss_D_tgt.item())
        meters.update(loss_D1=(loss_D_src1.item() + loss_D_tgt1.item()))
        meters.update(loss_D_src1=loss_D_src1.item())
        meters.update(loss_D_tgt1=loss_D_tgt1.item())

        iteration = iteration + 1

        n = src_input.size(0)

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (cfg.SOLVER.STOP_ITER - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iters:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr=optimizer_fea.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )



        if (iteration == cfg.SOLVER.MAX_ITER or iteration % cfg.SOLVER.CHECKPOINT_PERIOD==0) and save_to_disk:
            filename = os.path.join(output_dir, "model_iter{:06d}.pth".format(iteration))
            torch.save({'iteration': iteration, 'feature_extractor': feature_extractor.state_dict(), 'classifier':classifier.state_dict(),
                        'model_D': model_D.state_dict(), 'optimizer_fea': optimizer_fea.state_dict(), 'optimizer_cls': optimizer_cls.state_dict(),
                        'optimizer_dual1': optimizer_dual1.state_dict(),'dual1':dual1.state_dict(),
                        'optimizer_dual2': optimizer_dual2.state_dict(), 'dual2': dual2.state_dict(),'classifier1':classifier1.state_dict(),'optimizer_cls1': optimizer_cls1.state_dict(),
                        'classifier2': classifier2.state_dict(), 'optimizer_cls2': optimizer_cls2.state_dict(),'classifierall': classifierall.state_dict(), 'optimizer_clsall': optimizer_clsall.state_dict(),
                       'model_D1': model_D1.state_dict(),'optimizer_D1': optimizer_D1.state_dict(),'optimizer_D': optimizer_D.state_dict()}, filename)

        if iteration % 2000 == 0 or iteration == max_iters:
            intersection_meter = AverageMeter()
            union_meter = AverageMeter()
            target_meter = AverageMeter()
            test_data = build_dataset(cfg, mode='test', is_source=False,index = 0)
            test_sampler = None
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,num_workers=4, pin_memory=True, sampler=test_sampler)
            feature_extractor.eval()
            dual1.eval()
            classifier.eval()
            with torch.no_grad():
                for i, (x, y, _) in enumerate(test_loader):
                    x = x.cuda(non_blocking=True)
                    y = y.cuda(non_blocking=True).long()

                    size = y.shape[-2:]
                    fea = feature_extractor(x)
                    fea_br1 = dual1(fea)
                    output = classifier(fea_br1, size)
                    output = output.max(1)[1]
                    intersection, union, target = intersectionAndUnionGPU(output, y, cfg.MODEL.NUM_CLASSES,cfg.INPUT.IGNORE_LABEL)
                    intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
                    intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)


            iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
            accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

            logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
            for i in range(cfg.MODEL.NUM_CLASSES):
                logger.info('Class_{} {} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, test_data.trainid2name[i],iou_class[i],accuracy_class[i]))

        if iteration % 2000 == 0 or iteration == max_iters:
            intersection_meter = AverageMeter()
            union_meter = AverageMeter()
            target_meter = AverageMeter()
            test_data = build_dataset(cfg, mode='test', is_source=False,index = 1)
            test_sampler = None
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,num_workers=4, pin_memory=True, sampler=test_sampler)
            feature_extractor.eval()
            dual2.eval()
            classifier2.eval()
            with torch.no_grad():
                for i, (x, y, _) in enumerate(test_loader):
                    x = x.cuda(non_blocking=True)
                    y = y.cuda(non_blocking=True).long()

                    size = y.shape[-2:]
                    fea = feature_extractor(x)
                    fea_br2 = dual2(fea)
                    output = classifier(fea_br2, size)
                    output = output.max(1)[1]
                    intersection, union, target = intersectionAndUnionGPU(output, y, cfg.MODEL.NUM_CLASSES,cfg.INPUT.IGNORE_LABEL)
                    intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
                    intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)


            iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
            accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

            logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
            for i in range(cfg.MODEL.NUM_CLASSES):
                logger.info('Class_{} {} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, test_data.trainid2name[i],iou_class[i],accuracy_class[i]))

        if iteration % 2000 == 0 or iteration == max_iters:
            intersection_meter = AverageMeter()
            union_meter = AverageMeter()
            target_meter = AverageMeter()
            test_data = build_dataset(cfg, mode='test', is_source=False,index = 0)
            test_sampler = None
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,num_workers=4, pin_memory=True, sampler=test_sampler)
            feature_extractor.eval()
            classifier1.eval()
            with torch.no_grad():
                for i, (x, y, _) in enumerate(test_loader):
                    x = x.cuda(non_blocking=True)
                    y = y.cuda(non_blocking=True).long()

                    size = y.shape[-2:]
                    fea = feature_extractor(x)
                    output = classifier1(fea, size)
                    output = output.max(1)[1]
                    intersection, union, target = intersectionAndUnionGPU(output, y, cfg.MODEL.NUM_CLASSES,cfg.INPUT.IGNORE_LABEL)
                    intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
                    intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)


            iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
            accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

            logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
            for i in range(cfg.MODEL.NUM_CLASSES):
                logger.info('Class_{} {} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, test_data.trainid2name[i],iou_class[i],accuracy_class[i]))


        if iteration % 2000 == 0 or iteration == max_iters:
            intersection_meter = AverageMeter()
            union_meter = AverageMeter()
            target_meter = AverageMeter()
            test_data = build_dataset(cfg, mode='test', is_source=False,index = 1)
            test_sampler = None
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,num_workers=4, pin_memory=True, sampler=test_sampler)
            feature_extractor.eval()
            classifier2.eval()
            with torch.no_grad():
                for i, (x, y, _) in enumerate(test_loader):
                    x = x.cuda(non_blocking=True)
                    y = y.cuda(non_blocking=True).long()

                    size = y.shape[-2:]
                    fea = feature_extractor(x)
                    output = classifier2(fea, size)
                    output = output.max(1)[1]
                    intersection, union, target = intersectionAndUnionGPU(output, y, cfg.MODEL.NUM_CLASSES,cfg.INPUT.IGNORE_LABEL)
                    intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
                    intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)


            iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
            accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

            logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
            for i in range(cfg.MODEL.NUM_CLASSES):
                logger.info('Class_{} {} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, test_data.trainid2name[i],iou_class[i],accuracy_class[i]))

        if iteration % 2000 == 0 or iteration == max_iters:
            intersection_meter = AverageMeter()
            union_meter = AverageMeter()
            target_meter = AverageMeter()
            test_data = build_dataset(cfg, mode='test', is_source=False,index = 0)
            test_sampler = None
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,num_workers=4, pin_memory=True, sampler=test_sampler)
            feature_extractor.eval()
            classifierall.eval()
            with torch.no_grad():
                for i, (x, y, _) in enumerate(test_loader):
                    x = x.cuda(non_blocking=True)
                    y = y.cuda(non_blocking=True).long()

                    size = y.shape[-2:]
                    fea = feature_extractor(x)
                    output = classifierall(fea, size)
                    output = output.max(1)[1]
                    intersection, union, target = intersectionAndUnionGPU(output, y, cfg.MODEL.NUM_CLASSES,cfg.INPUT.IGNORE_LABEL)
                    intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
                    intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)


            iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
            accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

            logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
            for i in range(cfg.MODEL.NUM_CLASSES):
                logger.info('Class_{} {} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, test_data.trainid2name[i],iou_class[i],accuracy_class[i]))


        if iteration % 2000 == 0 or iteration == max_iters:
            intersection_meter = AverageMeter()
            union_meter = AverageMeter()
            target_meter = AverageMeter()
            test_data = build_dataset(cfg, mode='test', is_source=False,index = 1)
            test_sampler = None
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,num_workers=4, pin_memory=True, sampler=test_sampler)
            feature_extractor.eval()
            classifierall.eval()
            with torch.no_grad():
                for i, (x, y, _) in enumerate(test_loader):
                    x = x.cuda(non_blocking=True)
                    y = y.cuda(non_blocking=True).long()

                    size = y.shape[-2:]
                    fea = feature_extractor(x)
                    output = classifierall(fea, size)
                    output = output.max(1)[1]
                    intersection, union, target = intersectionAndUnionGPU(output, y, cfg.MODEL.NUM_CLASSES,cfg.INPUT.IGNORE_LABEL)
                    intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
                    intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)


            iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
            accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

            logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
            for i in range(cfg.MODEL.NUM_CLASSES):
                logger.info('Class_{} {} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, test_data.trainid2name[i],iou_class[i],accuracy_class[i]))

        if iteration == cfg.SOLVER.MAX_ITER:
            break
        if iteration == cfg.SOLVER.STOP_ITER:
            break
            
    
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (cfg.SOLVER.MAX_ITER)
        )
    )

    return feature_extractor, classifier          

def run_test(cfg, feature_extractor, classifier, local_rank, distributed):
    logger = logging.getLogger("FADA.tester")
    if local_rank==0:
        logger.info('>>>>>>>>>>>>>>>> Start Testing >>>>>>>>>>>>>>>>')
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    
    if distributed:
        feature_extractor, classifier = feature_extractor.module, classifier.module
    torch.cuda.empty_cache()  # TODO check if it helps
    dataset_name = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        mkdir(output_folder)

    test_data = build_dataset(cfg, mode='test', is_source=False)
    if distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    else:
        test_sampler = None
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, sampler=test_sampler)
    feature_extractor.eval()
    classifier.eval()
    end = time.time()
    with torch.no_grad():
        for i, (x, y, _) in enumerate(test_loader):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True).long()
            
            size = y.shape[-2:]

            output = classifier(feature_extractor(x), size)
            output = output.max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output, y, cfg.MODEL.NUM_CLASSES, cfg.INPUT.IGNORE_LABEL)
            if distributed:
                torch.distributed.all_reduce(intersection), torch.distributed.all_reduce(union), torch.distributed.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            batch_time.update(time.time() - end)
            end = time.time()
    
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if local_rank==0:
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(cfg.MODEL.NUM_CLASSES):
            logger.info('Class_{} {} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, test_data.trainid2name[i], iou_class[i], accuracy_class[i]))


def main():
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Training")
    parser.add_argument("-cfg",
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("FADA", output_dir, args.local_rank)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    fea, cls = train(cfg, args.local_rank, args.distributed)

    if not args.skip_test:
        run_test(cfg, fea, cls, args.local_rank, args.distributed)


if __name__ == "__main__":
    main()
