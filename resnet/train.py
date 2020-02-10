from resnet.resnet_plus_lstm import resnet18rnn, resnet50rnn, resnet34rnn
from resnet.convlstm import convlstmresnet9
from datasets import kitti
from datasets import hlw
from utilities.tee import Tee
import torch
from torch import nn
from torchvision import transforms
from tensorboardX import SummaryWriter
import datetime
import os
import numpy as np
import math
import time
import platform
import shutil
import sklearn.metrics
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F
import argparse
from utilities.losses import *
import random

#torch.backends.cudnn.benchmark = True

class Config:
    net_type = 'res18'
    dataset = 'kitti'
    # dataset = 'hlw'
    finetune = True
    num_epochs = 128 #200
    base_learning_rate = 0.1 / 128. #32.
    sequence_length = 16
    batch_size = 8
    batch_size_updates = None#[8, 16, 24, 32]
    batch_size_multi = 2
    learning_rate_updates = None#[8, 16, 24, 32]
    learning_rate_multi = 0.5
    optimizer = 'sgd'
    loss = 'huber'
    random_seed = 1
    hostname = ''
    downscale = 2
    cutout = True
    cutout_central_bias = False
    finetune_from = None #"/data/kluger/checkpoints/horizon_sequences/res18_fine/d1/1/b32_181011-092706"
    use_dropblock = False
    dropblock_drop_prob = 0.1
    regional_pool = None #(3,3)
    fc_layer = True
    conv_lstm = True
    gradient_clip = 0#1.0
    temporal_loss = True
    workers = 2
    random_subsampling = 1.5


class CosineAnnealingCustom:

    def __init__(self, begin, end, T_max):
        self.T_max = T_max
        self.begin = begin
        self.end = end
        self.inv = begin < end

    def get(self, epoch):
        if not self.inv:
            return self.end + (self.begin - self.end) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        else:
            return self.begin + (self.end - self.begin) * (1 - math.cos(math.pi * epoch / self.T_max)) / 2


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# def horizon_error(width, height):
#
#     def f(estm_ang, estm_off, true_ang, true_off):
#         errors = []
#
#         for b in range(estm_ang.shape[0]):
#             for s in range(estm_ang.shape[1]):
#
#                 offset = true_off[b,s].squeeze()
#                 offset_estm = estm_off[b,s].squeeze()
#                 angle = true_ang[b,s].squeeze()
#                 angle_estm = estm_ang[b,s].squeeze()
#
#                 offset += 0.5
#                 offset *= height
#
#                 offset_estm += 0.5
#                 offset_estm *= height
#
#                 true_mp = np.array([width / 2., offset])
#                 true_nv = np.array([np.sin(angle), np.cos(angle)])
#                 true_hl = np.array([true_nv[0], true_nv[1], -np.dot(true_nv, true_mp)])
#                 true_h1 = np.cross(true_hl, np.array([1, 0, 0]))
#                 true_h2 = np.cross(true_hl, np.array([1, 0, -width]))
#                 true_h1 /= true_h1[2]
#                 true_h2 /= true_h2[2]
#
#                 estm_mp = np.array([width / 2., offset_estm])
#                 estm_nv = np.array([np.sin(angle_estm), np.cos(angle_estm)])
#                 estm_hl = np.array([estm_nv[0], estm_nv[1], -np.dot(estm_nv, estm_mp)])
#                 estm_h1 = np.cross(estm_hl, np.array([1, 0, 0]))
#                 estm_h2 = np.cross(estm_hl, np.array([1, 0, -width]))
#                 estm_h1 /= estm_h1[2]
#                 estm_h2 /= estm_h2[2]
#
#                 err1 = np.minimum(np.linalg.norm(estm_h1 - true_h1), np.linalg.norm(estm_h1 - true_h2))
#                 err2 = np.minimum(np.linalg.norm(estm_h2 - true_h1), np.linalg.norm(estm_h2 - true_h2))
#
#                 err = np.maximum(err1, err2) / height
#                 errors += [err]
#
#         return errors
#
#     return f

def horizon_error(width, height):

    calc_hlr = calc_horizon_leftright(width, height)

    def f(estm_ang, estm_off, true_ang, true_off):
        errors = []

        for b in range(estm_ang.shape[0]):
            for s in range(estm_ang.shape[1]):

                offset = true_off[b,s].squeeze()
                offset_estm = estm_off[b,s].squeeze()
                angle = true_ang[b,s].squeeze()
                angle_estm = estm_ang[b,s].squeeze()

                ylt, yrt = calc_hlr(offset, angle)
                yle, yre = calc_hlr(offset_estm, angle_estm)

                err1 = np.abs((ylt-yle).cpu().detach().numpy())
                err2 = np.abs((yrt-yre).cpu().detach().numpy())

                err = np.maximum(err1, err2)
                errors += [err]

        return errors

    return f


def calc_horizon_leftright(width, height):
    wh = 0.5 * width*1./height

    def f(offset, angle):
        term2 = wh * torch.tan(torch.clamp(angle, -math.pi/3., math.pi/3.))
        return offset + 0.5 + term2, offset + 0.5 - term2

    return f


class TemporalMSELoss(_Loss):
    def __init__(self, size_average=True, reduce=True, reduction='elementwise_mean', from_start=False):
        super(TemporalMSELoss, self).__init__(size_average, reduce, reduction)
        self.from_start = from_start

    def forward(self, input, target):

        S = input.shape[1]

        input_diffs = []
        target_diffs = []

        if self.from_start:
            for s in range(1,S):
                input_diffs += [input[:,s,:]-input[:,0,:]]
                target_diffs += [target[:,s,:]-target[:,0,:]]
        else:
            for s in range(1,S):
                input_diffs += [input[:,s,:]-input[:,s-1,:]]
                target_diffs += [target[:,s,:]-target[:,s-1,:]]

        target_diffs = torch.stack(target_diffs, dim=1)
        input_diffs = torch.stack(input_diffs, dim=1)

        return F.mse_loss(input_diffs, target_diffs, reduction=self.reduction)


class MaxErrorLoss(_Loss):
    def __init__(self, size_average=True, reduce=True, reduction='elementwise_mean', from_start=False):
        super(MaxErrorLoss, self).__init__(size_average, reduce, reduction)
        self.from_start = from_start

    def forward(self, input, target):

        S = input.shape[1]

        input_diffs = []
        target_diffs = []

        if self.from_start:
            for s in range(1,S):
                input_diffs += [input[:,s,:]-input[:,0,:]]
                target_diffs += [target[:,s,:]-target[:,0,:]]
        else:
            for s in range(1,S):
                input_diffs += [input[:,s,:]-input[:,s-1,:]]
                target_diffs += [target[:,s,:]-target[:,s-1,:]]

        target_diffs = torch.stack(target_diffs, dim=1)
        input_diffs = torch.stack(input_diffs, dim=1)

        return F.mse_loss(input_diffs, target_diffs, reduction=self.reduction)


class CalcConfidenceTarget(torch.nn.Module):
    def __init__(self, max_error, device):
        super(CalcConfidenceTarget, self).__init__()
        self.max_error = max_error
        self.device = device

    def forward(self, input, target):

        input = torch.squeeze(input)
        target = torch.squeeze(target)

        # print("input shape: ", input.shape)
        # exit(0)

        ones = torch.ones(input.shape, dtype=torch.long).to(self.device)
        zero = torch.zeros(input.shape, dtype=torch.long).to(self.device)

        diffs = (input-target)
        diffs = diffs*diffs

        target = torch.where(diffs < self.max_error, zero, ones)

        return target.detach()

# class SqrtL1Loss(_Loss):
#
#     __constants__ = ['reduction']
#
#     def __init__(self, size_average=None, reduce=None, reduction='mean', delta=0.25):
#         self.reduce = reduce
#         self.delta = 0.25
#         self.a = 2*np.sqrt(delta)
#         self.b = -delta
#         super(SqrtL1Loss, self).__init__(size_average, reduce, reduction)
#
#     def forward(self, input, target):
#         absdiff = torch.clamp(torch.abs(input - target), 0, 1000.)
#         sqrt = self.a*torch.sqrt(absdiff)+self.b
#
#         losses = torch.where(absdiff <= self.delta, absdiff, sqrt)
#
#         if not (self.reduce == False):
#             return torch.mean(losses)
#         return losses
#         # return F.l1_loss(input, target, reduction=self.reduction)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--net', default='res18', type=str, metavar='NET', help='network type')
    parser.add_argument('--set', default='kitti', type=str, metavar='DS', help='dataset')
    parser.add_argument('--gpu', default='0', type=str, metavar='DS', help='dataset')
    parser.add_argument('--finetune', dest='finetune', action='store_true', help='finetune the CNN')
    parser.add_argument('--epochs', default=128, type=int, metavar='N', help='num epochs')
    parser.add_argument('--baselr', default=0.1 / 128, type=float, metavar='lr', help='base learning rate')
    parser.add_argument('--lr_reduction', default=1e-2, type=float, metavar='lr', help='base learning rate')
    parser.add_argument('--seqlength', default=1, type=int, metavar='N', help='sequence length')
    parser.add_argument('--seqlength_val', default=512, type=int, metavar='N', help='sequence length')
    parser.add_argument('--batch', default=8 * 16, type=int, metavar='B', help='batch size')
    parser.add_argument('--batch_val', default=1, type=int, metavar='B', help='batch size')
    parser.add_argument('--optimizer', default='sgd', type=str, metavar='optm', help='optimizer')
    parser.add_argument('--loss', default='huber', type=str, metavar='LF', help='loss function')
    parser.add_argument('--lossmax', default='l1', type=str, metavar='LF', help='loss function')
    parser.add_argument('--seed', default=1, type=int, metavar='S', help='random seed')
    parser.add_argument('--downscale', default=2, type=float, metavar='D', help='downscale factor')
    parser.add_argument('--cutout', default=512, type=int, help='use cutout', nargs='?', dest='cutout', const=512)
    parser.add_argument('--fc', dest='fc_layer', action='store_true', help='use FC layer')
    parser.add_argument('--convlstm', dest='conv_lstm', action='store_true', help='use Conv LSTM layer')
    parser.add_argument('--temploss', dest='temporal_loss', action='store_true', help='use temporal loss')
    parser.add_argument('--temploss2', dest='temporal_loss_2', action='store_true', help='use temporal loss')
    parser.add_argument('--templossonly', dest='temporal_loss_only', action='store_true', help='use temporal loss')
    parser.add_argument('--workers', default=3, type=int, metavar='W', help='number of workers')
    parser.add_argument('--random_subsampling', default=1., type=float, metavar='S', help='random subsampling factor')
    parser.add_argument('--conv_lstm_skip', dest='conv_lstm_skip', action='store_true', help='skip connection')
    parser.add_argument('--trainable_lstm_init', dest='trainable_lstm_init', action='store_true', help='')
    parser.add_argument('--confidence', dest='confidence', action='store_true', help='')
    parser.add_argument('--confidence_max_err', default=1e-4, type=float, metavar='S', help='random subsampling factor')
    parser.add_argument('--ema', default=0., type=float, metavar='S', help='')
    parser.add_argument('--angle_loss_weight', default=1., type=float, metavar='S', help='random subsampling factor')
    parser.add_argument('--lstm_state_reduction', default=1., type=float, metavar='S', help='random subsampling factor')
    parser.add_argument('--lstm_depth', default=2, type=int, metavar='S', help='random subsampling factor')
    parser.add_argument('--lstm_mem', default=0, type=int, metavar='S', help='random subsampling factor')
    parser.add_argument('--load', default=None, type=str, metavar='DS', help='dataset')
    parser.add_argument('--eval', dest='eval', action='store_true', help='')
    parser.add_argument('--eval_train', dest='eval_train', action='store_true', help='')
    parser.add_argument('--relulstm', dest='relulstm', action='store_true', help='')
    parser.add_argument('--overlap', default=0, type=int, metavar='S', help='random subsampling factor')
    parser.add_argument('--split', default=5, type=int, metavar='S', help='random subsampling factor')
    parser.add_argument('--fchead2', dest='fchead2', action='store_true', help='')
    parser.add_argument('--bn', dest='bn', action='store_true', help='')
    parser.add_argument('--skip', dest='skip', action='store_true', help='')
    parser.add_argument('--skip2', dest='skip2', action='store_true', help='')
    parser.add_argument('--bias', dest='bias', action='store_true', help='')
    parser.add_argument('--peephole', dest='peephole', action='store_true', help='')
    parser.add_argument('--max_error_loss', dest='max_error_loss', action='store_true', help='')
    parser.add_argument('--max_error_loss_only', dest='max_error_loss_only', action='store_true', help='')
    parser.add_argument('--no_fill_up', dest='no_fill_up', action='store_true', help='')
    parser.add_argument('--ar', dest='ar', action='store_true', help='')
    parser.add_argument('--nobn', dest='nobn', action='store_true', help='')
    parser.add_argument('--fp16', dest='fp16', action='store_true', help='')
    parser.add_argument('--kalman', dest='kalman', action='store_true', help='')
    parser.add_argument('--no_modelzoo_load', dest='nomzload', action='store_true', help='')
    parser.add_argument('--h_skip', dest='h_skip', action='store_true', help='')
    parser.add_argument('--simple_skip', dest='simple_skip', action='store_true', help='')
    parser.add_argument('--mixup', dest='mixup', action='store_true', help='')
    parser.add_argument('--layernorm', dest='layernorm', action='store_true', help='')
    parser.add_argument('--lstm_leakyrelu', dest='lstm_leakyrelu', action='store_true', help='')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    hostname = platform.node()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', 0)
    # device = torch.device('cpu')

    torch.backends.cudnn.deterministic = True

    # config = Config()
    # config.hostname = hostname


    if args.set == 'kitti':
        DS = kitti
        pixel_mean = [0.362365, 0.377767, 0.366744]
    elif args.set == 'hlw':
        DS = hlw
        pixel_mean = [0.469719773, 0.462005855, 0.454649294]
    else:
        assert False

    WIDTH = int(DS.WIDTH // args.downscale)
    HEIGHT = int(DS.HEIGHT // args.downscale)

    #
    # config.net_type = 'res18'
    #
    # config.finetune = True
    # config.num_epochs = 64
    # config.base_learning_rate = 0.001 / 32.
    #
    # config.sequence_length = 1
    # config.batch_size = 2

    learning_rate = args.baselr * args.batch * args.seqlength

    images_per_batch = args.batch * args.seqlength

    # config.batch_size_updates = [8, 16, 24, 32]

    workers = args.workers

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if 'daidalos' in hostname:
        target_base = "/tnt/data/kluger/checkpoints/horizon_sequences"
        root_dir = "/tnt/data/kluger/datasets/kitti/horizons" if args.set == 'kitti' else "/tnt/data/scene_understanding/HLW"
        csv_base = "/tnt/home/kluger/tmp/kitti_split_%d" % args.split
        pdf_file = "/tnt/home/kluger/tmp/kitti_split/data_pdfs.pkl"
    elif 'athene' in hostname:
        target_base = "/data/kluger/checkpoints/horizon_sequences"
        root_dir = "/phys/intern/kluger/tmp/kitti/horizons" if args.set == 'kitti' else "/phys/intern/kluger/tmp/HLW"
        csv_base = "/home/kluger/tmp/kitti_split_%d" % args.split
        pdf_file = "/home/kluger/tmp/kitti_split/data_pdfs.pkl"
    elif 'hekate' in hostname:
        target_base = "/data/kluger/checkpoints/horizon_sequences"
        root_dir = "/phys/ssd/kitti/horizons" if args.set == 'kitti' else "/data/scene_understanding/HLW"
        csv_base = "/home/kluger/tmp/kitti_split_%d" % args.split
        pdf_file = "/home/kluger/tmp/kitti_split/data_pdfs.pkl"
    elif 'persephone' in hostname or 'hades' in hostname:
        target_base = "/data/kluger/checkpoints/horizon_sequences"
        root_dir = "/phys/ssd/kluger/tmp/kitti/horizons" if args.set == 'kitti' else "/phys/ssd/kluger/tmp/HLW"
        csv_base = "/home/kluger/tmp/kitti_split_%d" % args.split
        pdf_file = "/home/kluger/tmp/kitti_split/data_pdfs.pkl"
    else:
        target_base = "/data/kluger/checkpoints/horizon_sequences"
        root_dir = "/data/kluger/datasets/kitti/horizons" if args.set == 'kitti' else "/phys/ssd/kluger/tmp/HLW"
        csv_base = "/home/kluger/tmp/kitti_split_%d" % args.split
        pdf_file = "/home/kluger/tmp/kitti_split/data_pdfs.pkl"


    if args.downscale > 1 and args.set == 'kitti':
        root_dir += "_s%.3f" % (1./args.downscale)

    if args.ema > 0:
        root_dir += "_ema%.3f" % args.ema

    pdf_file = None

    if args.finetune:
        target_directory = target_base + "/%s/%s_fine/d%d/%d/" % (args.set, args.net, args.downscale, args.seqlength)
    else:
        target_directory = target_base + "/%s/%s/d%d/%d/" % (args.set, args.net, args.downscale, args.seqlength)

    date_and_time = datetime.datetime.now().strftime("%y%m%d-%H%M%S")

    checkpoint_directory = target_directory + "b%d_" % args.batch + \
                           ("nocutout_" if (not args.cutout) else ("biasedcutout_" if (False) else "")) + date_and_time

    if not os.path.exists(checkpoint_directory):
        os.makedirs(checkpoint_directory)

    log_file = os.path.join(checkpoint_directory, "output.log")
    log = Tee(os.path.join(checkpoint_directory, log_file), "w", file_only=False)

    print("hostname: ", hostname)

    print(args)

    if args.net == 'res18':
        modelfun = resnet18rnn
    elif args.net == 'res34':
        modelfun = resnet34rnn
    elif args.net == 'res50':
        modelfun = resnet50rnn
    elif args.net == 'convlstm9':
        model = convlstmresnet9(device=device, batch_norm=args.bn, hidden_plane_reduction=args.lstm_state_reduction
                                ).to(device)
    else:
        assert False

    model = modelfun(args.finetune, regional_pool=None, use_fc=args.fc_layer, use_convlstm=args.conv_lstm,
                     width=WIDTH, height=HEIGHT, trainable_lstm_init=args.trainable_lstm_init,
                     conv_lstm_skip=args.conv_lstm_skip, confidence=args.confidence, second_head=(args.ema > 0),
                     relu_lstm=args.relulstm, second_head_fc=args.fchead2, lstm_bn=args.bn, lstm_skip=args.skip,
                     lstm_bias=args.bias, lstm_peephole=args.peephole, ar=args.ar, kalman=args.kalman,
                     lstm_state_reduction=args.lstm_state_reduction, bn=(not args.nobn), load=not(args.nomzload),
                     h_skip=args.h_skip, lstm_skip2=args.skip2, lstm_depth=args.lstm_depth,
                     lstm_simple_skip=args.simple_skip, lstm_mem=args.lstm_mem, layernorm=args.layernorm,
                     lstm_leakyrelu=args.lstm_leakyrelu
                     ).to(device)

    # model = nn.DataParallel(model)

    if args.fp16:
        model.half()  # convert to half precision
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    if args.load is not None:
        load_from_path = args.load
        print("load weights from ", load_from_path)
        checkpoint = torch.load(load_from_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'], strict=True)

    model = nn.DataParallel(model)

    if args.loss == 'mse':
        criterion = nn.MSELoss()
    elif args.loss == 'huber':
        criterion = nn.SmoothL1Loss()
    elif args.loss == 'l1':
        criterion = nn.L1Loss()
    else:
        assert False

    if args.lossmax == 'mse':
        criterionmax = nn.MSELoss(size_average=False, reduce=False)
        scalemax = 0.1
    elif args.lossmax == 'huber':
        criterionmax = nn.SmoothL1Loss(size_average=False, reduce=False)
        scalemax = 0.1
    elif args.lossmax == 'l1':
        criterionmax = nn.L1Loss(size_average=False, reduce=False)
        scalemax = 0.1
    elif args.lossmax == 'sqrt':
        criterionmax = SqrtL1Loss(size_average=False, reduce=False)
        scalemax = 0.1#0.02
    else:
        assert False

    temp_criterion = TemporalMSELoss(from_start=args.temporal_loss_2)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-4)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-4, momentum=0.9)
    else:
        assert False

    horizon_error_function = horizon_error(WIDTH, HEIGHT)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, int(args.epochs), eta_min=learning_rate*args.lr_reduction)

    # max_err_scheduler = CosineAnnealingCustom(0, 0.1, args.epochs)
    max_err_scheduler = CosineAnnealingCustom(0, 1., args.epochs)

    tfs = transforms.Compose([
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25),
                transforms.RandomGrayscale(p=0.1),
                # transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=pixel_mean, std=[1., 1., 1.]),
            ])
    if args.cutout > 0:
        tfs.transforms.append(DS.Cutout(length=args.cutout, bias=False))
    # tfs = transforms.Compose([
    #             transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
    #             transforms.RandomGrayscale(p=0.2),
    #             # transforms.RandomHorizontalFlip(p=0.5),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=pixel_mean, std=[1., 1., 1.]),
    #             Cutout(length=625)
    #         ])
    tfs_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=pixel_mean, std=[1., 1., 1.]),
            ])

    if args.set == 'kitti':

        train_dataset = DS.KittiRawDatasetPP(root_dir=root_dir, pdf_file=pdf_file, random_subsampling=args.random_subsampling,
                                             csv_file=csv_base + "/train.csv", seq_length=args.seqlength,
                                             im_height=HEIGHT, im_width=WIDTH, fill_up=(not args.no_fill_up),
                                             scale=1./args.downscale, transform=tfs, get_split_data=(args.ema > 0),
                                             overlap=args.overlap, zero_start=args.temporal_loss_2)
        val_dataset = DS.KittiRawDatasetPP(root_dir=root_dir, pdf_file=pdf_file, augmentation=False,
                                           csv_file=csv_base + "/val.csv", seq_length=args.seqlength_val,
                                           im_height=HEIGHT, im_width=WIDTH, fill_up=False,
                                           scale=1./args.downscale, transform=tfs_val, get_split_data=(args.ema > 0),
                                           zero_start=args.temporal_loss_2)
    elif args.set == 'hlw':

        train_dataset = DS.HLWDataset(root_dir=root_dir, transform=tfs, augmentation=True, set='train', scale=1./args.downscale)
        val_dataset = DS.HLWDataset(root_dir=root_dir, augmentation=False, transform=tfs_val, set='val', scale=1./args.downscale)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch,
                                               shuffle=True, num_workers=workers)
    if args.mixup:
        train_loader2 = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch,
                                               shuffle=True, num_workers=workers)


    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=args.batch_val,
                                              shuffle=False, num_workers=workers)

    # For updating learning rate
    def update_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def update_batchsize(loader, batch_size):
        loader = torch.utils.data.DataLoader(dataset=loader.dataset, batch_size=batch_size, shuffle=True, num_workers=loader.num_workers)
        return loader

    def save_checkpoint(state, is_best, folder, epoch, loss):
        filename = folder + "/" + "%03d_%.6f.ckpt" % (epoch, loss)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, folder + '/model_best.ckpt')



    print(checkpoint_directory)

    if args.confidence:
        ConfidenceTarget = CalcConfidenceTarget(args.confidence_max_err, device)
        assert False

    if args.temporal_loss_only:
        assert False

    # Train the model
    total_step = len(train_loader)
    curr_lr = learning_rate
    curr_batch_size = args.batch
    best_val_loss = 10000.

    calc_hlr = calc_horizon_leftright(width=WIDTH, height=HEIGHT)

    best_auc = {'epoch': 0, 'max_err': np.inf, 'auc':0}
    best_err = {'epoch': 0, 'max_err': np.inf, 'auc':0}

    tensorboard_directory = checkpoint_directory + "/tensorboard/"
    if not os.path.exists(tensorboard_directory):
        os.makedirs(tensorboard_directory)
    tensorboard_writer = SummaryWriter(tensorboard_directory)

    for epoch in range(args.epochs):

        # if config.use_dropblock:
        #     model.set_dropblock_prob(1-1.*epoch/config.num_epochs*config.dropblock_drop_prob)

        if not args.eval:
            scheduler.step()
            adjust_learning_rate(optimizer, scheduler.get_lr()[0])

            losses = []
            offset_losses = []
            angle_losses = []
            offset_ema_losses = []
            angle_ema_losses = []
            offset_dif_losses = []
            angle_dif_losses = []
            temp_offset_losses = []
            temp_angle_losses = []
            confidence_losses = []
            max_err_losses = []

            tt0 = time.time()

            model.train()
            for i, sample in enumerate(train_loader):
            # for i in range(len(train_loader)):

                # if i > 400: break

                # sample = next(iter(train_loader))
                # images = sample['images'].to(device, non_blocking=True)
                # offsets = sample['offsets'].to(device, non_blocking=True)
                # angles = sample['angles'].to(device, non_blocking=True)

                if args.mixup:
                    sample2 = next(iter(train_loader2))
                    lam = np.random.beta(0.2, 0.2)
                    images = sample['images']
                    offsets = sample['offsets']
                    angles = sample['angles']

                    images2 = sample2['images']
                    offsets2 = sample2['offsets']
                    angles2 = sample2['angles']

                    B1 = images.shape[0]
                    B2 = images2.shape[0]
                    B = np.minimum(B1, B2)

                    images = images[:B]
                    images2 = images2[:B]
                    offsets = offsets[:B]
                    offsets2 = offsets2[:B]
                    angles = angles[:B]
                    angles2 = angles2[:B]

                    images = lam * images + (1. - lam) * images2
                    offsets = lam * offsets + (1. - lam) * offsets2
                    angles = lam * angles + (1. - lam) * angles2


                    images = images.to(device, non_blocking=True)
                    offsets = offsets.to(device, non_blocking=True)
                    angles = angles.to(device, non_blocking=True)

                else:
                    images = sample['images'].to(device, non_blocking=True)
                    offsets = sample['offsets'].to(device, non_blocking=True)
                    angles = sample['angles'].to(device, non_blocking=True)



                if args.ema > 0:
                    offsets_ema = sample['offsets_ema'].to(device, non_blocking=True)
                    angles_ema = sample['angles_ema'].to(device, non_blocking=True)
                    offsets_dif = offsets - offsets_ema
                    angles_dif = angles - angles_ema

                # Forward pass
                if args.confidence:
                    assert False, "confidence: not implemented"
                    output_offsets, output_angles, output_confidence = model(images)
                    confidence_target = ConfidenceTarget(output_offsets.view(-1, 1), offsets.view(-1, 1))
                    confidence_loss = torch.nn.CrossEntropyLoss()(output_confidence.view(-1, 2), confidence_target)
                    confidence_losses += [confidence_loss]
                else:
                    if args.ema > 0:
                        output_offsets_dif, output_angles_dif, output_offsets_ema, output_angles_ema = \
                            model(images)
                    else:
                        output_offsets, output_angles = model(images)

                if args.ema > 0:
                    output_offsets = output_offsets_ema + output_offsets_dif
                    output_angles = output_angles_ema + output_angles_dif
                    offset_ema_loss = criterion(output_offsets_ema, offsets_ema)
                    angle_ema_loss = criterion(output_angles_ema, angles_ema)
                    offset_dif_loss = criterion(output_offsets_dif, offsets_dif)
                    angle_dif_loss = criterion(output_angles_dif, angles_dif)

                offset_loss = criterion(output_offsets, offsets)
                angle_loss = criterion(output_angles, angles)

                loss = 0

                # if args.max_error_loss:
                hl_true, hr_true = calc_hlr(offsets, angles)
                hl_estm, hr_estm = calc_hlr(output_offsets, output_angles)
                hl_err = criterionmax(hl_estm, hl_true)
                hr_err = criterionmax(hr_estm, hr_true)
                h_errs = torch.clamp(torch.max(hl_err, hr_err), 0, 1.)
                max_err_loss = torch.mean(h_errs)
                max_err_losses += [max_err_loss]

                if args.ema > 0:
                    loss += offset_ema_loss + angle_ema_loss * args.angle_loss_weight + \
                            offset_dif_loss + angle_dif_loss * args.angle_loss_weight
                else:
                    loss += offset_loss + angle_loss * args.angle_loss_weight

                if args.temporal_loss:
                    temp_offset_loss = temp_criterion(output_offsets, offsets)
                    temp_angle_loss = temp_criterion(output_angles, angles)
                    loss += temp_offset_loss + temp_angle_loss * args.angle_loss_weight
                    temp_offset_losses.append(temp_offset_loss)
                    temp_angle_losses.append(temp_angle_loss)

                if args.max_error_loss:
                    if args.max_error_loss_only:
                        loss = max_err_loss
                    else:
                        # loss = max_err_scheduler.get(epoch) * max_err_loss + (1-max_err_scheduler.get(epoch)) * loss
                        loss = max_err_scheduler.get(epoch) * max_err_loss * scalemax + (1-max_err_scheduler.get(epoch)) * loss

                tt3 = time.time()

                # if args.confidence:
                #     loss += confidence_loss

                # images = sample['images'].to(device, non_blocking=True)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                tt4 = time.time()
                # if config.gradient_clip > 0:
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                tt5 = time.time()

                # print('data loading : %f' % (tt01-tt0) )
                # print('data to dev. : %f' % (tt1-tt01) )
                # print('forward pass : %f' % (tt2-tt1) )
                # print('loss calculat: %f' % (tt3-tt2) )
                # print('backward pass: %f' % (tt4-tt3) )
                # print('optimization : %f' % (tt5-tt4) )
                # print("...")

                # losses.append(loss.item())
                losses.append(loss)
                offset_losses.append(offset_loss)
                angle_losses.append(angle_loss)
                if args.ema > 0:
                    offset_ema_losses.append(offset_ema_loss)
                    angle_ema_losses.append(angle_ema_loss)
                    offset_dif_losses.append(offset_dif_loss)
                    angle_dif_losses.append(angle_dif_loss)

                if (i+1) % 100 == 0:
                    # average_loss = np.mean(losses)
                    losses_tensor = torch.stack(losses, dim=0).view(-1)
                    average_loss = losses_tensor.mean().item()

                    offset_losses_tensor = torch.stack(offset_losses, dim=0).view(-1)
                    average_offset_loss = offset_losses_tensor.mean().item()
                    angle_losses_tensor = torch.stack(angle_losses, dim=0).view(-1)
                    average_angle_loss = angle_losses_tensor.mean().item()

                    # average_offset_loss = np.mean(offset_losses)
                    # average_angle_loss = np.mean(angle_losses)

                    num_iteration = int((epoch*total_step + i) * images_per_batch / 128.)

                    if args.temporal_loss:
                        # temp_average_offset_loss = np.mean(temp_offset_losses)
                        # temp_average_angle_loss = np.mean(temp_angle_losses)
                        temp_offset_losses_tensor = torch.stack(temp_offset_losses, dim=0).view(-1)
                        temp_average_offset_loss = temp_offset_losses_tensor.mean().item()
                        temp_angle_losses_tensor = torch.stack(temp_angle_losses, dim=0).view(-1)
                        temp_average_angle_loss = temp_angle_losses_tensor.mean().item()

                    if args.confidence:
                        confidence_losses_tensor = torch.stack(confidence_losses, dim=0).view(-1)
                        average_confidence_loss = confidence_losses_tensor.mean().item()
                        tensorboard_writer.add_scalar('train/confidence_loss', confidence_loss.item(), num_iteration)
                        tensorboard_writer.add_scalar('train/confidence_loss_avg', average_confidence_loss, num_iteration)

                    if args.ema > 0:
                        offset_ema_losses_tensor = torch.stack(offset_ema_losses, dim=0).view(-1)
                        average_offset_ema_loss = offset_ema_losses_tensor.mean().item()
                        angle_ema_losses_tensor = torch.stack(angle_ema_losses, dim=0).view(-1)
                        average_angle_ema_loss = angle_ema_losses_tensor.mean().item()
                        tensorboard_writer.add_scalar('train/offset_ema_loss', offset_ema_loss.item(), num_iteration)
                        tensorboard_writer.add_scalar('train/angle_ema_loss', angle_ema_loss.item(), num_iteration)
                        tensorboard_writer.add_scalar('train/offset_ema_loss_avg', average_offset_ema_loss, num_iteration)
                        tensorboard_writer.add_scalar('train/angle_ema_loss_avg', average_angle_ema_loss, num_iteration)
                        offset_dif_losses_tensor = torch.stack(offset_dif_losses, dim=0).view(-1)
                        average_offset_dif_loss = offset_dif_losses_tensor.mean().item()
                        angle_dif_losses_tensor = torch.stack(angle_dif_losses, dim=0).view(-1)
                        average_angle_dif_loss = angle_dif_losses_tensor.mean().item()
                        tensorboard_writer.add_scalar('train/offset_dif_loss', offset_dif_loss.item(), num_iteration)
                        tensorboard_writer.add_scalar('train/angle_dif_loss', angle_dif_loss.item(), num_iteration)
                        tensorboard_writer.add_scalar('train/offset_dif_loss_avg', average_offset_dif_loss, num_iteration)
                        tensorboard_writer.add_scalar('train/angle_dif_loss_avg', average_angle_dif_loss, num_iteration)

                    # if args.max_error_loss:
                    max_err_losses_tensor = torch.stack(max_err_losses, dim=0).view(-1)
                    average_max_err_loss = max_err_losses_tensor.mean().item()
                    tensorboard_writer.add_scalar('train/max_err_loss', max_err_loss.item(), num_iteration)
                    tensorboard_writer.add_scalar('train/max_err_loss_avg', average_max_err_loss, num_iteration)


                    print ("Epoch [{}/{}], Step [{}/{}] Losses: {:.6f} {:.6f} {:.6f}, Avg.: {:.6f} {:.6f} {:.6f}"
                           .format(epoch+1, args.epochs, i+1, total_step, offset_loss.item(), angle_loss.item(), loss.item(),
                                   average_offset_loss, average_angle_loss, average_loss), end="\r")

                    tensorboard_writer.add_scalar('train/loss', loss.item(), num_iteration)
                    tensorboard_writer.add_scalar('train/offset_loss', offset_loss.item(), num_iteration)
                    tensorboard_writer.add_scalar('train/angle_loss', angle_loss.item(), num_iteration)
                    tensorboard_writer.add_scalar('train/loss_avg', average_loss, num_iteration)
                    tensorboard_writer.add_scalar('train/offset_loss_avg', average_offset_loss, num_iteration)
                    tensorboard_writer.add_scalar('train/angle_loss_avg', average_angle_loss, num_iteration)
                    if args.temporal_loss:
                        tensorboard_writer.add_scalar('train/temp_offset_loss', temp_offset_loss.item(), num_iteration)
                        tensorboard_writer.add_scalar('train/temp_angle_loss', temp_angle_loss.item(), num_iteration)
                        tensorboard_writer.add_scalar('train/temp_offset_loss_avg', temp_average_offset_loss, num_iteration)
                        tensorboard_writer.add_scalar('train/temp_angle_loss_avg', temp_average_angle_loss, num_iteration)

                    tensorboard_writer.add_scalar('learning_rate', scheduler.get_lr()[0], num_iteration)
                    # tensorboard_writer.add_histogram('gradients', np.concatenate([x.grad.cpu().detach().numpy().flatten() for x in model.parameters() if x is not None]), num_iteration)

                tt0 = time.time()


        # test on validation set:
        model.eval()
        with torch.no_grad():
            losses = []
            offset_losses = []
            angle_losses = []
            offset_ema_losses = []
            angle_ema_losses = []
            offset_dif_losses = []
            angle_dif_losses = []
            temp_offset_losses = []
            temp_angle_losses = []
            confidence_losses = []
            max_err_losses = []

            all_horizon_errors = []

            image_count = 0

            for idx, sample in enumerate(val_loader):
                images = sample['images'].to(device)
                offsets = sample['offsets'].to(device)
                angles = sample['angles'].to(device)
                if args.ema > 0:
                    offsets_ema = sample['offsets_ema'].to(device)
                    angles_ema = sample['angles_ema'].to(device)
                    offsets_dif = offsets - offsets_ema
                    angles_dif = angles - angles_ema
                    
                # print(images.shape)
                image_count += images.shape[0]*images.shape[1]

                if args.confidence:
                    output_offsets, output_angles, output_confidence = model(images)
                    confidence_target = ConfidenceTarget(output_offsets.view(-1, 1), offsets.view(-1, 1))
                    confidence_loss = torch.nn.CrossEntropyLoss()(output_confidence.view(-1, 2), confidence_target)
                    confidence_losses += [confidence_loss]
                else:
                    if args.ema > 0:
                        output_offsets_dif, output_angles_dif, output_offsets_ema, output_angles_ema = \
                            model(images)
                    else:
                        output_offsets, output_angles = \
                            model(images)

                if args.ema > 0:
                    output_offsets = output_offsets_ema + output_offsets_dif
                    output_angles = output_angles_ema + output_angles_dif
                    offset_ema_loss = criterion(output_offsets_ema, offsets_ema)
                    angle_ema_loss = criterion(output_angles_ema, angles_ema)
                    offset_dif_loss = criterion(output_offsets_dif, offsets_dif)
                    angle_dif_loss = criterion(output_angles_dif, angles_dif)

                offset_loss = criterion(output_offsets, offsets)
                angle_loss = criterion(output_angles, angles)

                loss = offset_loss + angle_loss * args.angle_loss_weight
                if args.ema > 0:
                    loss = offset_ema_loss + angle_ema_loss * args.angle_loss_weight + \
                           offset_dif_loss + angle_dif_loss * args.angle_loss_weight
                else:
                    loss = offset_loss + angle_loss * args.angle_loss_weight
                    
                if args.temporal_loss:
                    temp_offset_loss = temp_criterion(output_offsets, offsets)
                    temp_angle_loss = temp_criterion(output_angles, angles)
                    loss += temp_offset_loss + temp_angle_loss * args.angle_loss_weight
                    temp_offset_losses.append(temp_offset_loss.item())
                    temp_angle_losses.append(temp_angle_loss.item())

                if args.confidence:
                    loss += confidence_loss

                # if args.max_error_loss:
                hl_true, hr_true = calc_hlr(offsets, angles)
                hl_estm, hr_estm = calc_hlr(output_offsets, output_angles)
                hl_err = criterionmax(hl_estm, hl_true)
                hr_err = criterionmax(hr_estm, hr_true)
                h_errs = torch.max(hl_err, hr_err)
                max_err_loss = torch.mean(h_errs)
                # loss += max_err_loss
                max_err_losses += [max_err_loss]

                if args.max_error_loss:
                    if args.max_error_loss_only:
                        loss = max_err_loss * scalemax
                    else:
                        loss = max_err_scheduler.get(epoch) * scalemax * max_err_loss + (1 - max_err_scheduler.get(epoch)) * loss

                # all_horizon_errors += horizon_error_function(output_angles.cpu().detach().numpy(),
                #                                              output_offsets.cpu().detach().numpy(),
                #                                              angles.cpu().detach().numpy(),
                #                                              offsets.cpu().detach().numpy(),)
                all_horizon_errors += horizon_error_function(output_angles,
                                                             output_offsets,
                                                             angles,
                                                             offsets)
                # print(all_horizon_errors[:8])
                # exit(0)
                #print(output_offsets[0, :8])
                #print(images.shape)
                #print(images[0,0,0,20,100:108])
                #exit(0)

                losses.append(loss.item())
                offset_losses.append(offset_loss.item())
                angle_losses.append(angle_loss.item())
                if args.ema > 0:
                    offset_ema_losses.append(offset_ema_loss.item())
                    angle_ema_losses.append(angle_ema_loss.item())
                    offset_dif_losses.append(offset_dif_loss.item())
                    angle_dif_losses.append(angle_dif_loss.item())

                # if (idx+1) % 10 == 0:
                #     print(idx+1)

            average_loss = np.mean(losses)
            average_offset_loss = np.mean(offset_losses)
            average_angle_loss = np.mean(angle_losses)
            if args.ema > 0:
                average_offset_ema_loss = np.mean(offset_ema_losses)
                average_angle_ema_loss = np.mean(angle_ema_losses)
                average_offset_dif_loss = np.mean(offset_dif_losses)
                average_angle_dif_loss = np.mean(angle_dif_losses)
            if args.temporal_loss:
                temp_average_offset_loss = np.mean(temp_offset_losses)
                temp_average_angle_loss = np.mean(temp_angle_losses)

            # num_iteration = epoch * total_step

            num_iteration = int((epoch * total_step + idx) * images_per_batch / 128.)

            if args.confidence:
                confidence_losses_tensor = torch.stack(confidence_losses, dim=0).view(-1)
                average_confidence_loss = confidence_losses_tensor.mean().item()
                tensorboard_writer.add_scalar('val/confidence_loss', confidence_loss.item(), num_iteration)
                tensorboard_writer.add_scalar('val/confidence_loss_avg', average_confidence_loss, num_iteration)

            # if args.max_error_loss:
            max_err_losses_tensor = torch.stack(max_err_losses, dim=0).view(-1)
            average_max_err_loss = max_err_losses_tensor.mean().item()
            # tensorboard_writer.add_scalar('val/max_err_loss', max_err_loss.item(), num_iteration)
            tensorboard_writer.add_scalar('val/max_err_loss_avg', average_max_err_loss, num_iteration)

            error_arr = np.array(all_horizon_errors)
            error_arr_idx = np.argsort(error_arr)
            error_arr = np.sort(error_arr)
            num_values = len(all_horizon_errors)

            plot_points = np.zeros((num_values, 2))

            err_cutoff = 0.25

            midfraction = 1.

            try:
                for i in range(num_values):
                    fraction = (i + 1) * 1.0 / num_values
                    value = error_arr[i]
                    plot_points[i, 1] = fraction
                    plot_points[i, 0] = value
                    if i > 0:
                        lastvalue = error_arr[i - 1]
                        if lastvalue < err_cutoff and value > err_cutoff:
                            midfraction = (lastvalue * plot_points[i - 1, 1] + value * fraction) / (value + lastvalue)

                if plot_points[-1, 0] < err_cutoff:
                    plot_points = np.vstack([plot_points, np.array([err_cutoff, 1])])
                else:
                    # print("midfraction: ", midfraction)
                    plot_points = np.vstack([plot_points, np.array([err_cutoff, midfraction])])

                sorting = np.argsort(plot_points[:, 0])
                plot_points = plot_points[sorting, :]

                auc = sklearn.metrics.auc(plot_points[plot_points[:, 0] <= err_cutoff, 0],
                                          plot_points[plot_points[:, 0] <= err_cutoff, 1])
                auc = auc / err_cutoff
            except:
                auc = 0

            print("\nValidation [{}/{}],  Avg.: {:.4f} {:.4f} {:.4f} {:.4f}"
                  .format(epoch + 1, args.epochs, average_offset_loss, average_angle_loss, average_loss, auc))

            if best_err['max_err'] > average_max_err_loss:
                best_err['epoch'] = epoch
                best_err['max_err'] = average_max_err_loss
                best_err['auc'] = auc

            if best_auc['auc'] < auc:
                best_auc['epoch'] = epoch
                best_auc['max_err'] = average_max_err_loss
                best_auc['auc'] = auc

            print("Best Err: %.4f -- AUC: %.4f -- epoch %d" % (best_err['max_err'], best_err['auc'], best_err['epoch']))
            print("Best AUC: %.4f -- Err: %.4f -- epoch %d" % (best_auc['auc'], best_auc['max_err'], best_auc['epoch']))

            #print("%d images" % image_count)
            #print(error_arr.shape)
            #print(np.mean(error_arr))
            #print("errors: \n", error_arr)
            #print(error_arr_idx)

            if args.eval:
                exit(0)

            tensorboard_writer.add_scalar('val/loss_avg', average_loss, num_iteration)
            tensorboard_writer.add_scalar('val/offset_loss_avg', average_offset_loss, num_iteration)
            tensorboard_writer.add_scalar('val/angle_loss_avg', average_angle_loss, num_iteration)
            if args.ema > 0:
                tensorboard_writer.add_scalar('val/offset_ema_loss_avg', average_offset_ema_loss, num_iteration)
                tensorboard_writer.add_scalar('val/angle_ema_loss_avg', average_angle_ema_loss, num_iteration)
                tensorboard_writer.add_scalar('val/offset_dif_loss_avg', average_offset_dif_loss, num_iteration)
                tensorboard_writer.add_scalar('val/angle_dif_loss_avg', average_angle_dif_loss, num_iteration)
            if args.temporal_loss:
                tensorboard_writer.add_scalar('val/temp_offset_loss_avg', temp_average_offset_loss, num_iteration)
                tensorboard_writer.add_scalar('val/temp_angle_loss_avg', temp_average_angle_loss, num_iteration)
            tensorboard_writer.add_scalar('val/auc', auc, num_iteration)

        # torch.save(model.state_dict(), checkpoint_directory + "/" + "%03d_%.6f.ckpt" % (epoch, average_loss))

        is_best = (average_loss < best_val_loss)
        best_val_loss = average_loss if is_best else best_val_loss

        save_checkpoint({
                'epoch': epoch,
                'args': args,
                'state_dict': model.module.state_dict(),
                'val_loss': average_loss,
                'optimizer' : optimizer.state_dict(),
                # 'all_gradients' : np.concatenate([x.grad.cpu().detach().numpy().flatten() for x in model.parameters() if x is not None])
            }, is_best, checkpoint_directory, epoch, average_loss)

        # if config.batch_size_updates is not None:
        #     # Increase batch size
        #     if (epoch+1) in config.batch_size_updates:
        #         curr_batch_size *= config.batch_size_multi
        #         train_loader = update_batchsize(train_loader, curr_batch_size)
        #         print("new batch size: ", train_loader.batch_size)
        # if config.learning_rate_updates is not None:
        #     # Decay learning rate
        #     if (epoch+1) in config.learning_rate_updates == 0:
        #         curr_lr *= config.learning_rate_multi
        #         update_lr(optimizer, curr_lr)
        #         print("new learning rate: ", train_loader.batch_size)

    tensorboard_writer.close()
    log.__del__()

