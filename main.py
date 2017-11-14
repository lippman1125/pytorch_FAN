import os
import time
import matplotlib
matplotlib.use('Agg')
from progress.bar import Bar
import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets

import models
from datasets import W300LP
from utils.logger import Logger, savefig
from utils.imutils import batch_with_heatmap
from utils.evaluation import accuracy, AverageMeter, final_preds, calc_metrics
from utils.misc import adjust_learning_rate, save_checkpoint, save_pred
import opts

args = opts.argparser()
model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))
# torch.setdefaulttensortype('torch.FloatTensor')

best_acc = 0.
idx = range(1, 69, 1)


def main(args):
    global best_acc

    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

    print("==> Creating model '{}', stacks={}, blocks={}".format(args.netType, args.nStacks,
                                                                 args.nModules))

    print("=> Models will be saved at: {}".format(args.checkpoint))

    model = models.__dict__[args.netType](
        num_stacks=args.nStacks, num_blocks=args.nModules, num_classes=68)

    model = torch.nn.DataParallel(model).cuda()

    criterion = torch.nn.MSELoss(size_average=True).cuda()

    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    title = "300-W-LP" + args.netType

    val_loader = torch.utils.data.DataLoader(
        W300LP(args, 'val'),
        batch_size=args.val_batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> Loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Valid Loss', 'Train Acc', 'Val Acc'])

    train_loader = torch.utils.data.DataLoader(
        W300LP(args, 'train'),
        batch_size=args.train_batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)
    cudnn.benchmark = True
    print('=> Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / (1024. * 1024)))

    if args.evaluation:
        print('=> Evaluation only')
        loss, acc, predictions = validate(val_loader, model, criterion, args.netType, args.debug,
                                          args.flip)
        save_pred(predictions, checkpoint=args.checkpoint)
        return

    lr = args.lr
    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)
        print('=> Epoch: %d | LR %.8f' % (epoch + 1, lr))

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, args.netType,
                                      args.debug, args.flip)
        # do not save predictions in model file
        valid_loss, valid_acc, predictions = validate(val_loader, model, criterion, args.netType, args.debug,
                                         args.flip)

        logger.append([int(epoch + 1), lr, train_loss, valid_loss, train_acc, valid_acc])

        is_best = valid_acc >= best_acc
        best_acc = max(valid_acc, best_acc)
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'netType': args.netType,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            },
            is_best,
            predictions,
            checkpoint=args.checkpoint)

    logger.close()
    logger.plot(['Train Acc', 'Valid Acc'])
    savefig(os.path.join(args.checkpoint, 'log.eps'))


def train(loader, model, criterion, optimizer, netType, debug=False, flip=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    model.train()
    end = time.time()

    gt_win, pred_win = None, None
    bar = Bar('Processing', max=len(loader))
    for i, (inputs, target) in enumerate(loader):
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(inputs.cuda())
        target_var = torch.autograd.Variable(target.cuda(async=True))

        if debug:
            gt_batch_img = batch_with_heatmap(inputs, target)
            # pred_batch_img = batch_with_heatmap(inputs, score_map)
            if not gt_win or not pred_win:
                plt.subplot(121)
                gt_win = plt.imshow(gt_batch_img)
                # plt.subplot(122)
                # pred_win = plt.imshow(pred_batch_img)
            else:
                gt_win.set_data(gt_batch_img)
                # pred_win.set_data(pred_batch_img)
            plt.pause(.05)
            plt.draw()

        output = model(input_var)
        score_map = output[-1].data.cpu()

        if flip:
            flip_input_var = torch.autograd.Variable(
                torch.from_numpy(shufflelr(inputs.clone().numpy())).float().cuda())
            flip_output_var = model(flip_input_var)
            flip_output = flip_back(flip_output_var[-1].data.cpu())
            score_map += flip_output

        # intermediate supervision
        loss = 0
        for o in output:
            loss += criterion(o, target_var)
        acc, _ = accuracy(score_map, target.cpu(), idx, thr=0.07)

        losses.update(loss.data[0], inputs.size(0))
        acces.update(acc[0], inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
            batch=i + 1,
            size=len(loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            acc=acces.avg)
        bar.next()

    bar.finish()

    return losses.avg, acces.avg


def validate(loader, model, criterion, netType, debug, flip):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()
    end = time.time()

    # predictions
    predictions = torch.Tensor(loader.dataset.__len__(), 68, 2)

    model.eval()
    gt_win, pred_win = None, None
    bar = Bar('Processing', max=len(loader))
    all_dists = torch.zeros((68, loader.dataset.__len__()))
    for i, (inputs, target, meta) in enumerate(loader):
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(inputs.cuda())
        target_var = torch.autograd.Variable(target.cuda(async=True))

        output = model(input_var)
        score_map = output[-1].data.cpu()

        if flip:
            flip_input_var = torch.autograd.Variable(
                torch.from_numpy(shufflelr(inputs.clone().numpy())).float().cuda())
            flip_output_var = model(flip_input_var)
            flip_output = flip_back(flip_output_var[-1].data.cpu())
            score_map += flip_output

        # intermediate supervision
        loss = 0
        for o in output:
            loss += criterion(o, target_var)
        acc, batch_dists = accuracy(score_map, target.cpu(), idx, thr=0.07)
        all_dists[:, i * args.val_batch:(i + 1) * args.val_batch] = batch_dists

        preds = final_preds(score_map, meta['center'], meta['scale'], [64, 64])
        for n in range(score_map.size(0)):
            predictions[meta['index'][n], :, :] = preds[n, :, :]

        if debug:
            gt_batch_img = batch_with_heatmap(inputs, target)
            pred_batch_img = batch_with_heatmap(inputs, score_map)
            if not gt_win or not pred_win:
                plt.subplot(121)
                gt_win = plt.imshow(gt_batch_img)
                plt.subplot(122)
                pred_win = plt.imshow(pred_batch_img)
            else:
                gt_win.set_data(gt_batch_img)
                pred_win.set_data(pred_batch_img)
            plt.pause(.05)
            plt.draw()

        losses.update(loss.data[0], inputs.size(0))
        acces.update(acc[0], inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
            batch=i + 1,
            size=len(loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            acc=acces.avg)
        bar.next()

    bar.finish()
    mean_error = torch.mean(all_dists)
    auc = calc_metrics(all_dists, show_curve=False)
    print("=> Mean Error: {}. AUC@0.07: {}".format(mean_error, auc))
    return losses.avg, acces.avg, predictions


if __name__ == '__main__':
    main(args)